//! Scripted fake upstream provider.
//!
//! `FakeUpstream` binds an ephemeral loopback port and answers every request with a scripted
//! [`Reply`]. Replies are sequences of [`Step`]s, so tests control exactly how bytes are chunked,
//! when the server pauses, and whether the connection drops or stalls. Every request is captured
//! (method, URI, headers, body) for assertions about what adapters actually send.
//!
//! [`FakeUpstream::proxy`] turns the same server into a recording passthrough: each request is
//! forwarded to a real base URL, the response is relayed chunk by chunk, and the request/response
//! pair is captured. Credential headers are marked sensitive so they never show up in `Debug` output.

use axum::{
	Router,
	body::{Body, Bytes, to_bytes},
	extract::{Request, State},
	http::{HeaderMap, HeaderValue, Method, StatusCode, header::CONTENT_TYPE},
	response::Response,
	routing::any,
};
use futures_util::StreamExt;
use serde_json::Value;
use std::{
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	time::Duration,
};
use tokio::{sync::mpsc, task::JoinHandle};

/// Pause inserted between chunks by [`Reply::split_at`] so the client observes each chunk separately.
pub const FLUSH_GAP: Duration = Duration::from_millis(2);

pub const REDACTED: &str = "[REDACTED]";

/// Request headers whose values are credentials.
pub const SECRET_HEADERS: [&str; 6] = ["authorization", "proxy-authorization", "x-api-key", "x-goog-api-key", "api-key", "cookie"];

const HOP_BY_HOP_HEADERS: [&str; 10] = [
	"host",
	"connection",
	"keep-alive",
	"proxy-connection",
	"te",
	"trailer",
	"transfer-encoding",
	"upgrade",
	"content-length",
	"accept-encoding",
];

#[derive(Clone, Debug)]
pub struct CapturedRequest {
	/// Arrival order across the whole server, starting at 0.
	pub sequence: usize,
	pub method: Method,
	/// Path plus query string, e.g. `/v1/models?pageToken=2`.
	pub uri: String,
	pub path: String,
	pub headers: HeaderMap,
	pub body: Bytes,
	/// Body parsed as JSON, or `Value::Null` when it is not JSON.
	pub json: Value,
}

impl CapturedRequest {
	pub fn header(&self, name: &str) -> Option<&str> {
		self.headers.get(name).and_then(|value| value.to_str().ok())
	}

	/// Lowercase header names with credential values replaced by [`REDACTED`], sorted by name.
	pub fn redacted_headers(&self) -> Vec<(String, String)> {
		let mut headers: Vec<(String, String)> = self
			.headers
			.iter()
			.map(|(name, value)| {
				let name = name.as_str().to_ascii_lowercase();
				let value = if SECRET_HEADERS.contains(&name.as_str()) {
					REDACTED.to_string()
				} else {
					value.to_str().unwrap_or("[non-utf8]").to_string()
				};
				(name, value)
			})
			.collect();
		headers.sort();
		headers
	}
}

/// The real provider's answer to a proxied request, as relayed to the caller.
#[derive(Clone, Debug)]
pub struct CapturedResponse {
	/// `sequence` of the request this answers.
	pub sequence: usize,
	pub status: StatusCode,
	pub headers: HeaderMap,
	pub body: Bytes,
}

impl CapturedResponse {
	pub fn content_type(&self) -> &str {
		self.headers.get(CONTENT_TYPE).and_then(|value| value.to_str().ok()).unwrap_or_default()
	}

	pub fn is_sse(&self) -> bool {
		self.content_type().starts_with("text/event-stream")
	}
}

/// A proxied request with the response that came back, once its body finished streaming.
#[derive(Clone, Debug)]
pub struct Exchange {
	pub request: CapturedRequest,
	pub response: Option<CapturedResponse>,
}

#[derive(Clone, Debug)]
pub enum Step {
	/// Write these bytes as one body frame.
	Bytes(Bytes),
	/// Sleep before the next step.
	Delay(Duration),
	/// Abort the connection mid-body.
	Disconnect,
	/// Never produce another byte.
	Stall,
}

#[derive(Clone, Debug)]
pub struct Reply {
	pub status: StatusCode,
	pub headers: Vec<(String, String)>,
	pub steps: Vec<Step>,
}

impl Reply {
	pub fn new(status: StatusCode, content_type: &str, body: impl Into<Bytes>) -> Self {
		Self {
			status,
			headers: vec![(CONTENT_TYPE.to_string(), content_type.to_string())],
			steps: vec![Step::Bytes(body.into())],
		}
	}

	pub fn json(value: Value) -> Self {
		Self::new(StatusCode::OK, "application/json", value.to_string())
	}

	pub fn sse(body: impl Into<Bytes>) -> Self {
		Self::new(StatusCode::OK, "text/event-stream", body.into())
	}

	/// SSE reply whose chunks are each written and flushed separately.
	pub fn sse_chunks<I, B>(chunks: I) -> Self
	where
		I: IntoIterator<Item = B>,
		B: Into<Bytes>,
	{
		Self::sse(Bytes::new()).with_chunks(chunks.into_iter().map(Into::into).collect())
	}

	pub fn status(mut self, status: StatusCode) -> Self {
		self.status = status;
		self
	}

	pub fn header(mut self, name: &str, value: &str) -> Self {
		self.headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
		self.headers.push((name.to_string(), value.to_string()));
		self
	}

	pub fn then(mut self, step: Step) -> Self {
		self.steps.push(step);
		self
	}

	/// Concatenation of every `Step::Bytes` in the script.
	pub fn body_bytes(&self) -> Vec<u8> {
		self.steps
			.iter()
			.filter_map(|step| match step {
				Step::Bytes(bytes) => Some(bytes.as_ref()),
				_ => None,
			})
			.flatten()
			.copied()
			.collect()
	}

	/// Re-chunk the whole body at the given byte offsets (unsorted and out-of-range offsets are
	/// tolerated), flushing each piece separately. Non-byte steps after the body are kept.
	pub fn split_at(self, offsets: &[usize]) -> Self {
		let body = self.body_bytes();
		let mut cuts: Vec<usize> = offsets.iter().copied().filter(|offset| *offset > 0 && *offset < body.len()).collect();
		cuts.sort_unstable();
		cuts.dedup();
		let mut chunks = Vec::with_capacity(cuts.len() + 1);
		let mut start = 0;
		for cut in cuts.into_iter().chain(std::iter::once(body.len())) {
			chunks.push(Bytes::copy_from_slice(&body[start..cut]));
			start = cut;
		}
		self.with_chunks(chunks)
	}

	fn with_chunks(mut self, chunks: Vec<Bytes>) -> Self {
		let trailing: Vec<Step> = self.steps.into_iter().filter(|step| !matches!(step, Step::Bytes(_))).collect();
		let mut steps = Vec::with_capacity(chunks.len() * 2 + trailing.len());
		for (index, chunk) in chunks.into_iter().enumerate() {
			if index > 0 {
				steps.push(Step::Delay(FLUSH_GAP));
			}
			steps.push(Step::Bytes(chunk));
		}
		steps.extend(trailing);
		self.steps = steps;
		self
	}

	fn into_response(self) -> Response<Body> {
		let mut response = Response::new(stream_body(self.steps));
		*response.status_mut() = self.status;
		for (name, value) in self.headers {
			response.headers_mut().append(
				axum::http::HeaderName::try_from(name).expect("valid header name"),
				HeaderValue::try_from(value).expect("valid header value"),
			);
		}
		response
	}
}

fn stream_body(steps: Vec<Step>) -> Body {
	Body::from_stream(async_stream::stream! {
		for step in steps {
			match step {
				Step::Bytes(bytes) => yield Ok::<Bytes, std::io::Error>(bytes),
				Step::Delay(duration) => tokio::time::sleep(duration).await,
				Step::Disconnect => {
					yield Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "fake upstream disconnect"));
					return;
				}
				Step::Stall => std::future::pending::<()>().await,
			}
		}
	})
}

type Responder = Arc<dyn Fn(&CapturedRequest) -> Reply + Send + Sync>;

#[derive(Clone)]
enum Mode {
	Scripted(Responder),
	Proxy { client: reqwest::Client, target: Arc<str> },
}

#[derive(Clone)]
struct ServerState {
	mode: Mode,
	sequence: Arc<AtomicUsize>,
	captured: mpsc::UnboundedSender<CapturedRequest>,
	responses: mpsc::UnboundedSender<CapturedResponse>,
}

pub struct FakeUpstream {
	base_url: String,
	incoming: mpsc::UnboundedReceiver<CapturedRequest>,
	incoming_responses: mpsc::UnboundedReceiver<CapturedResponse>,
	captured: Vec<CapturedRequest>,
	captured_responses: Vec<CapturedResponse>,
	server: JoinHandle<()>,
}

impl FakeUpstream {
	/// Start a server that answers each request with `responder(request)`.
	pub async fn start(responder: impl Fn(&CapturedRequest) -> Reply + Send + Sync + 'static) -> Self {
		Self::launch(Mode::Scripted(Arc::new(responder))).await
	}

	/// Recording passthrough: forward every request to `target_base_url` (scheme, authority and any
	/// path prefix), relay the response chunk by chunk without buffering, and capture both sides.
	/// Read them back with [`FakeUpstream::exchanges`].
	pub async fn proxy(target_base_url: &str) -> Self {
		let client = reqwest::Client::builder()
			.redirect(reqwest::redirect::Policy::none())
			.connect_timeout(Duration::from_secs(15))
			.build()
			.expect("build proxy client");
		Self::launch(Mode::Proxy {
			client,
			target: target_base_url.trim_end_matches('/').into(),
		})
		.await
	}

	async fn launch(mode: Mode) -> Self {
		let (sender, incoming) = mpsc::unbounded_channel();
		let (response_sender, incoming_responses) = mpsc::unbounded_channel();
		let state = ServerState {
			mode,
			sequence: Arc::new(AtomicUsize::new(0)),
			captured: sender,
			responses: response_sender,
		};
		let app = Router::new().fallback(any(handle)).with_state(state);
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind fake upstream");
		let address = listener.local_addr().expect("fake upstream address");
		let server = tokio::spawn(async move {
			axum::serve(listener, app).await.expect("fake upstream server");
		});
		Self {
			base_url: format!("http://{address}"),
			incoming,
			incoming_responses,
			captured: Vec::new(),
			captured_responses: Vec::new(),
			server,
		}
	}

	/// Answer every request with the same reply.
	pub async fn always(reply: Reply) -> Self {
		Self::start(move |_| reply.clone()).await
	}

	/// Answer the n-th request with the n-th reply; requests beyond the script get HTTP 500.
	pub async fn sequence(replies: Vec<Reply>) -> Self {
		let next = AtomicUsize::new(0);
		Self::start(move |_| {
			replies
				.get(next.fetch_add(1, Ordering::SeqCst))
				.cloned()
				.unwrap_or_else(|| Reply::new(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "fake upstream script exhausted"))
		})
		.await
	}

	pub fn base_url(&self) -> String {
		self.base_url.clone()
	}

	/// Every request received so far, in arrival order.
	pub fn requests(&mut self) -> &[CapturedRequest] {
		while let Ok(request) = self.incoming.try_recv() {
			self.captured.push(request);
		}
		&self.captured
	}

	/// Requests received so far whose path equals `path`.
	pub fn requests_to(&mut self, path: &str) -> Vec<CapturedRequest> {
		self.requests().iter().filter(|request| request.path == path).cloned().collect()
	}

	/// Proxy mode: every request received so far with the response relayed for it, ordered by arrival.
	/// A response is present once its body finished streaming to the caller.
	pub fn exchanges(&mut self) -> Vec<Exchange> {
		self.requests();
		while let Ok(response) = self.incoming_responses.try_recv() {
			self.captured_responses.push(response);
		}
		let mut exchanges: Vec<Exchange> = self
			.captured
			.iter()
			.map(|request| Exchange {
				request: request.clone(),
				response: self.captured_responses.iter().find(|response| response.sequence == request.sequence).cloned(),
			})
			.collect();
		exchanges.sort_by_key(|exchange| exchange.request.sequence);
		exchanges
	}
}

impl Drop for FakeUpstream {
	fn drop(&mut self) {
		self.server.abort();
	}
}

fn mark_sensitive(headers: &mut HeaderMap) {
	for name in SECRET_HEADERS {
		if let Some(value) = headers.get_mut(name) {
			value.set_sensitive(true);
		}
	}
}

async fn capture(state: &ServerState, request: Request) -> CapturedRequest {
	let (parts, body) = request.into_parts();
	let body = to_bytes(body, usize::MAX).await.unwrap_or_default();
	let mut headers = parts.headers;
	mark_sensitive(&mut headers);
	CapturedRequest {
		sequence: state.sequence.fetch_add(1, Ordering::SeqCst),
		method: parts.method,
		uri: parts.uri.path_and_query().map(|value| value.to_string()).unwrap_or_default(),
		path: parts.uri.path().to_string(),
		headers,
		json: serde_json::from_slice(&body).unwrap_or(Value::Null),
		body,
	}
}

async fn handle(State(state): State<ServerState>, request: Request) -> Response<Body> {
	let captured = capture(&state, request).await;
	match state.mode.clone() {
		Mode::Scripted(responder) => {
			let reply = responder(&captured);
			let _ = state.captured.send(captured);
			reply.into_response()
		}
		Mode::Proxy { client, target } => {
			let _ = state.captured.send(captured.clone());
			forward(client, &target, captured, state.responses.clone()).await
		}
	}
}

/// Reports the relayed reply once, at EOF or when the client drops the body early (adapters stop
/// reading after the terminal event, before the upstream closes the stream).
struct ResponseCapture {
	sequence: usize,
	status: StatusCode,
	headers: HeaderMap,
	seen: Vec<u8>,
	responses: Option<mpsc::UnboundedSender<CapturedResponse>>,
}

impl Drop for ResponseCapture {
	fn drop(&mut self) {
		if let Some(responses) = self.responses.take() {
			let _ = responses.send(CapturedResponse {
				sequence: self.sequence,
				status: self.status,
				headers: std::mem::take(&mut self.headers),
				body: Bytes::from(std::mem::take(&mut self.seen)),
			});
		}
	}
}

async fn forward(client: reqwest::Client, target: &str, request: CapturedRequest, responses: mpsc::UnboundedSender<CapturedResponse>) -> Response<Body> {
	let mut call = client.request(request.method.clone(), format!("{target}{}", request.uri)).body(request.body.clone());
	for (name, value) in request.headers.iter() {
		if !HOP_BY_HOP_HEADERS.contains(&name.as_str()) {
			call = call.header(name, value);
		}
	}
	let upstream = match call.send().await {
		Ok(upstream) => upstream,
		Err(error) => return Reply::new(StatusCode::BAD_GATEWAY, "text/plain", format!("proxy could not reach upstream: {error}")).into_response(),
	};
	let status = upstream.status();
	let mut headers = upstream.headers().clone();
	for name in HOP_BY_HOP_HEADERS {
		headers.remove(name);
	}
	let relayed = headers.clone();
	headers.remove(axum::http::header::CONTENT_LENGTH);
	let mut capture = ResponseCapture {
		sequence: request.sequence,
		status,
		headers: relayed,
		seen: Vec::new(),
		responses: Some(responses),
	};
	let mut chunks = upstream.bytes_stream();
	let body = Body::from_stream(async_stream::stream! {
		while let Some(chunk) = chunks.next().await {
			match chunk {
				Ok(bytes) => {
					capture.seen.extend_from_slice(&bytes);
					yield Ok::<Bytes, std::io::Error>(bytes);
				}
				Err(error) => {
					capture.responses = None;
					yield Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, error.to_string()));
					return;
				}
			}
		}
	});
	let mut response = Response::new(body);
	*response.status_mut() = status;
	*response.headers_mut() = headers;
	response
}

//! Scripted fake upstream provider.
//!
//! `FakeUpstream` binds an ephemeral loopback port and answers every request with a scripted
//! [`Reply`]. Replies are sequences of [`Step`]s, so tests control exactly how bytes are chunked,
//! when the server pauses, and whether the connection drops or stalls. Every request is captured
//! (method, URI, headers, body) for assertions about what adapters actually send.

use axum::{
	Router,
	body::{Body, Bytes, to_bytes},
	extract::{Request, State},
	http::{HeaderMap, HeaderValue, Method, StatusCode, header::CONTENT_TYPE},
	response::Response,
	routing::any,
};
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

#[derive(Clone, Debug)]
pub struct CapturedRequest {
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
struct ServerState {
	responder: Responder,
	captured: mpsc::UnboundedSender<CapturedRequest>,
}

pub struct FakeUpstream {
	base_url: String,
	incoming: mpsc::UnboundedReceiver<CapturedRequest>,
	captured: Vec<CapturedRequest>,
	server: JoinHandle<()>,
}

impl FakeUpstream {
	/// Start a server that answers each request with `responder(request)`.
	pub async fn start(responder: impl Fn(&CapturedRequest) -> Reply + Send + Sync + 'static) -> Self {
		let (sender, incoming) = mpsc::unbounded_channel();
		let state = ServerState {
			responder: Arc::new(responder),
			captured: sender,
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
			captured: Vec::new(),
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
}

impl Drop for FakeUpstream {
	fn drop(&mut self) {
		self.server.abort();
	}
}

async fn handle(State(state): State<ServerState>, request: Request) -> Response<Body> {
	let (parts, body) = request.into_parts();
	let body = to_bytes(body, usize::MAX).await.unwrap_or_default();
	let captured = CapturedRequest {
		method: parts.method,
		uri: parts.uri.path_and_query().map(|value| value.to_string()).unwrap_or_default(),
		path: parts.uri.path().to_string(),
		headers: parts.headers,
		json: serde_json::from_slice(&body).unwrap_or(Value::Null),
		body,
	};
	let reply = (state.responder)(&captured);
	let _ = state.captured.send(captured);
	reply.into_response()
}

use crate::adapter::{AdapterError, endpoint_timeout, next_chunk, provider_headers, provider_request, provider_url, send_with_timeout, shared_http_client};
use crate::catalog::UsageBreakdown;
use crate::types::{
	ByteStreamEvent, ByteStreamResponse, Embedding, EmbeddingRequestIR, EmbeddingResponse, EmbeddingVector, ImageInput, ImageOperation, ImageRequestIR, ImageResponse,
	ModelRef, SpeechRequestIR, TranscriptionOperation, TranscriptionRequestIR,
};
use futures::stream::BoxStream;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Shared by the OpenAI and openai_compat adapters: the OpenAI REST endpoints beyond chat.
/// The client's raw fields are forwarded with the set typed IR fields written over them, so hosts
/// can change the IR after conversion.
pub(crate) async fn embeddings(request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
	let mut body = request.raw_request.unwrap_or_default();
	body.insert("input".into(), json!(request.input));
	set_if_some(&mut body, "dimensions", request.dimensions.map(Value::from));
	set_if_some(&mut body, "encoding_format", request.encoding_format.map(Value::from));
	body.insert("model".into(), json!(request.model.model_id));
	let (value, provider_cost) = send(&request.model, "v1/embeddings", |call| call.json(&body)).await?;
	let data = value.get("data").ok_or_else(|| AdapterError::http("embeddings response did not contain data"))?;
	let data = Vec::<WireEmbedding>::deserialize(data).map_err(|error| AdapterError::http(format!("invalid embeddings response: {error}")))?;
	let data = data
		.into_iter()
		.enumerate()
		.map(|(position, item)| Embedding {
			index: item.index.unwrap_or(position as u32),
			embedding: item.embedding,
		})
		.collect();
	let usage = value.get("usage");
	let input_tokens = usage
		.and_then(|usage| usage.get("prompt_tokens").or_else(|| usage.get("total_tokens")))
		.and_then(Value::as_u64)
		.unwrap_or(0);
	Ok(EmbeddingResponse {
		data,
		usage: UsageBreakdown {
			input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
			requests: 1,
			..UsageBreakdown::default()
		},
		provider_cost,
		raw: value,
	})
}

/// Some OpenAI-compatible servers omit `index`; it then defaults to the item's position.
#[derive(Deserialize)]
struct WireEmbedding {
	#[serde(default)]
	index: Option<u32>,
	embedding: EmbeddingVector,
}

pub(crate) async fn image(request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
	let ImageRequestIR {
		model,
		operation,
		prompt,
		input_images,
		options,
		mask,
		raw_request,
		..
	} = request;
	let input_count = input_images.len() as u32;
	let default_n = raw_request.is_none().then_some(1);
	let mut fields = raw_request.unwrap_or_default();
	fields.insert("prompt".into(), json!(prompt));
	set_if_some(&mut fields, "n", options.n.or(default_n).map(Value::from));
	for (name, value) in [
		("size", options.size),
		("quality", options.quality),
		("output_format", options.output_format),
		("background", options.background),
	] {
		set_if_some(&mut fields, name, value.map(Value::from));
	}
	fields.insert("model".into(), json!(model.model_id));
	let (value, provider_cost) = if operation == ImageOperation::Edit {
		if input_images.is_empty() {
			return Err(AdapterError::invalid("editing requires an input image"));
		}
		let mut form = text_form(fields);
		let image_field = if input_count > 1 { "image[]" } else { "image" };
		for (index, input) in input_images.into_iter().enumerate() {
			form = form.part(image_field, file_part(input, format!("image-{index}"))?);
		}
		if let Some(mask) = mask {
			form = form.part("mask", file_part(mask, "mask".to_string())?);
		}
		send(&model, "v1/images/edits", |call| call.multipart(form)).await?
	} else {
		send(&model, "v1/images/generations", |call| call.json(&fields)).await?
	};
	let mut response = crate::image::response_from_openai(&value, input_count)?;
	response.usage.provider_cost = provider_cost;
	response.raw = Some(value);
	Ok(response)
}

fn set_if_some(fields: &mut Map<String, Value>, name: &str, value: Option<Value>) {
	if let Some(value) = value {
		fields.insert(name.into(), value);
	}
}

fn file_part(input: ImageInput, file_name: String) -> Result<reqwest::multipart::Part, AdapterError> {
	reqwest::multipart::Part::bytes(input.bytes)
		.file_name(file_name)
		.mime_str(&input.media_type)
		.map_err(|error| AdapterError::invalid(error.to_string()))
}

async fn send(model: &ModelRef, path: &str, body: impl FnOnce(reqwest::RequestBuilder) -> reqwest::RequestBuilder) -> Result<(Value, Option<f64>), AdapterError> {
	let (response, header_cost) = open(model, path, body, false).await?;
	let value: Value = response.json().await.map_err(|error| AdapterError::http(error.to_string()))?;
	let cost = header_cost.or_else(|| body_cost(model, &value));
	Ok((value, cost))
}

/// Sends the request and checks its status. A `streaming` call has no total timeout: the endpoint
/// timeout bounds the wait for headers here and each idle gap in [`relay`].
async fn open(
	model: &ModelRef,
	path: &str,
	body: impl FnOnce(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
	streaming: bool,
) -> Result<(reqwest::Response, Option<f64>), AdapterError> {
	let endpoint = &model.provider.endpoint;
	let builder = shared_http_client().post(provider_url(&endpoint.base_url, path));
	let mut call = if streaming {
		provider_headers(builder, endpoint)?
	} else {
		provider_request(builder, endpoint)?
	};
	if let Some(api_key) = endpoint.api_key.as_deref() {
		call = call.bearer_auth(api_key);
	}
	let timeout = if streaming { endpoint_timeout(endpoint) } else { None };
	let response = send_with_timeout(body(call), timeout, streaming).await?;
	if !response.status().is_success() {
		// The error body of a streamed call is read under the same idle bound as its headers.
		let error = crate::upstream::http_error(response, endpoint, &model.provider.name);
		return Err(match timeout {
			Some(duration) => tokio::time::timeout(duration, error).await.unwrap_or(AdapterError::Timeout),
			None => error.await,
		});
	}
	let header_cost = endpoint.reported_cost.as_ref().and_then(|config| config.read_headers(response.headers()));
	Ok((response, header_cost.map(|cost| cost.total)))
}

fn body_cost(model: &ModelRef, value: &Value) -> Option<f64> {
	model
		.provider
		.endpoint
		.reported_cost
		.as_ref()
		.and_then(|config| config.read_body(value))
		.map(|cost| cost.total)
}

/// Multipart text fields; an array becomes one part per item under the same name, as OpenAI
/// expects for `timestamp_granularities[]` and `include[]`.
fn text_form(fields: Map<String, Value>) -> reqwest::multipart::Form {
	let text = |value: Value| match value {
		Value::String(text) => text,
		other => other.to_string(),
	};
	let mut form = reqwest::multipart::Form::new();
	for (name, value) in fields {
		match value {
			Value::Array(items) => {
				for item in items {
					form = form.text(name.clone(), text(item));
				}
			}
			value => form = form.text(name, text(value)),
		}
	}
	form
}

pub(crate) async fn transcription(request: TranscriptionRequestIR) -> Result<ByteStreamResponse, AdapterError> {
	let TranscriptionRequestIR {
		model,
		operation,
		audio,
		language,
		prompt,
		response_format,
		timestamp_granularities,
		stream,
		raw_request,
		..
	} = request;
	let mut fields = raw_request.unwrap_or_default();
	for (name, value) in [("language", language), ("prompt", prompt), ("response_format", response_format)] {
		set_if_some(&mut fields, name, value.map(Value::from));
	}
	if !timestamp_granularities.is_empty() {
		fields.remove("timestamp_granularities");
		fields.insert("timestamp_granularities[]".into(), json!(timestamp_granularities));
	}
	if stream || fields.contains_key("stream") {
		fields.insert("stream".into(), json!(stream));
	}
	fields.insert("model".into(), json!(model.model_id));
	let file = reqwest::multipart::Part::bytes(audio.bytes)
		.file_name(audio.filename)
		.mime_str(&audio.media_type)
		.map_err(|error| AdapterError::invalid(error.to_string()))?;
	let form = text_form(fields).part("file", file);
	let path = match operation {
		TranscriptionOperation::Transcribe => "v1/audio/transcriptions",
		TranscriptionOperation::Translate => "v1/audio/translations",
	};
	let (response, header_cost) = open(&model, path, |call| call.multipart(form), stream).await?;
	let content_type = content_type(&response);
	if stream {
		let timeout = endpoint_timeout(&model.provider.endpoint);
		return Ok(ByteStreamResponse {
			content_type,
			provider_cost: header_cost,
			body: relay(response, timeout, transcription_usage),
		});
	}
	let bytes = response.bytes().await.map_err(|error| AdapterError::http(error.to_string()))?;
	let value = serde_json::from_slice::<Value>(&bytes).ok();
	let usage = value.as_ref().and_then(transcription_usage);
	let provider_cost = header_cost.or_else(|| value.as_ref().and_then(|value| body_cost(&model, value)));
	let events = usage.map(ByteStreamEvent::Usage).into_iter().chain([ByteStreamEvent::Data(bytes)]);
	Ok(ByteStreamResponse {
		content_type,
		provider_cost,
		body: Box::pin(futures::stream::iter(events)),
	})
}

pub(crate) async fn speech(request: SpeechRequestIR) -> Result<ByteStreamResponse, AdapterError> {
	let SpeechRequestIR {
		model,
		input,
		voice,
		instructions,
		response_format,
		speed,
		stream_format,
		raw_request,
		..
	} = request;
	let mut body = raw_request.unwrap_or_default();
	body.insert("input".into(), json!(input));
	set_if_some(&mut body, "voice", voice);
	set_if_some(&mut body, "speed", speed.map(Value::from));
	for (name, value) in [("instructions", instructions), ("response_format", response_format), ("stream_format", stream_format)] {
		set_if_some(&mut body, name, value.map(Value::from));
	}
	body.insert("model".into(), json!(model.model_id));
	let (response, provider_cost) = open(&model, "v1/audio/speech", |call| call.json(&body), true).await?;
	Ok(ByteStreamResponse {
		content_type: content_type(&response),
		provider_cost,
		body: relay(response, endpoint_timeout(&model.provider.endpoint), speech_usage),
	})
}

fn content_type(response: &reqwest::Response) -> String {
	response
		.headers()
		.get(reqwest::header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.unwrap_or("application/octet-stream")
		.to_string()
}

/// Relays the body under the idle-gap timeout. Non-SSE bodies pass through chunk by chunk. SSE
/// bodies are relayed one complete event at a time, so a failure never lands inside a forwarded
/// event; usage is emitted before the event that carries it, and an in-band provider error event
/// ends the stream as an error instead of being relayed raw.
fn relay(mut response: reqwest::Response, timeout: Option<std::time::Duration>, usage_from: fn(&Value) -> Option<UsageBreakdown>) -> BoxStream<'static, ByteStreamEvent> {
	let sse = content_type(&response).starts_with("text/event-stream");
	Box::pin(async_stream::stream! {
		let mut parser = crate::sse::SseParser::new();
		loop {
			let chunk = match next_chunk(&mut response, timeout).await {
				Ok(Some(chunk)) => chunk,
				Ok(None) => break,
				Err(error) => {
					yield ByteStreamEvent::Error(error.into());
					break;
				}
			};
			if !sse {
				yield ByteStreamEvent::Data(chunk);
				continue;
			}
			for event in parser.feed(&chunk) {
				let data = serde_json::from_str::<Value>(&event.data).ok();
				if data.as_ref().is_some_and(is_error_event) {
					yield ByteStreamEvent::Error(AdapterError::http("provider reported an error in the event stream").into());
					return;
				}
				if let Some(usage) = data.as_ref().and_then(usage_from) {
					yield ByteStreamEvent::Usage(usage);
				}
				yield ByteStreamEvent::Data(sse_frame(&event));
			}
		}
	})
}

fn is_error_event(data: &Value) -> bool {
	data.get("error").is_some_and(|error| !error.is_null()) || data.get("type").and_then(Value::as_str) == Some("error")
}

fn sse_frame(event: &crate::sse::SseEvent) -> bytes::Bytes {
	let mut frame = String::with_capacity(event.data.len() + 16);
	if let Some(name) = &event.event_type {
		frame.push_str("event: ");
		frame.push_str(name);
		frame.push('\n');
	}
	for line in event.data.split('\n') {
		frame.push_str("data: ");
		frame.push_str(line);
		frame.push('\n');
	}
	frame.push('\n');
	frame.into()
}

fn count(value: &Value, key: &str) -> Option<u32> {
	value.get(key).and_then(Value::as_u64).map(|count| u32::try_from(count).unwrap_or(u32::MAX))
}

/// Transcription usage, from a JSON response or a `transcript.text.done` event: token usage
/// (gpt-4o-transcribe, audio input tokens split out when reported), else billed audio seconds
/// (`usage.type: "duration"`, or whisper's `verbose_json` `duration`).
fn transcription_usage(value: &Value) -> Option<UsageBreakdown> {
	let usage = value.get("usage").filter(|usage| usage.is_object());
	if let Some(usage) = usage.filter(|usage| usage.get("input_tokens").is_some() || usage.get("output_tokens").is_some()) {
		let input = count(usage, "input_tokens").unwrap_or(0);
		let details = usage.get("input_token_details").unwrap_or(&Value::Null);
		let audio = count(details, "audio_tokens").unwrap_or(0).min(input);
		return Some(UsageBreakdown {
			input_tokens: count(details, "text_tokens").unwrap_or(input - audio),
			input_audio_tokens: audio,
			output_tokens: count(usage, "output_tokens").unwrap_or(0),
			requests: 1,
			..UsageBreakdown::default()
		});
	}
	let seconds = usage
		.and_then(|usage| usage.get("seconds"))
		.or_else(|| value.get("duration"))
		.and_then(Value::as_f64)?;
	Some(UsageBreakdown {
		input_audio_seconds: seconds,
		requests: 1,
		..UsageBreakdown::default()
	})
}

/// Speech usage from a `speech.audio.done` event; its output tokens are audio tokens.
fn speech_usage(value: &Value) -> Option<UsageBreakdown> {
	let usage = value.get("usage").filter(|usage| usage.is_object())?;
	Some(UsageBreakdown {
		input_tokens: count(usage, "input_tokens").unwrap_or(0),
		output_audio_tokens: count(usage, "output_tokens").unwrap_or(0),
		requests: 1,
		..UsageBreakdown::default()
	})
}

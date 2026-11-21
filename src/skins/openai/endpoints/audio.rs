use crate::skins::common::context::{Metadata, SkinContext, request_metadata};
use crate::skins::openai::errors::invalid_body;
use crate::types::{AudioInput, ByteStreamEvent, ByteStreamResponse, ModelOperation, ModelRef, SpeechRequestIR, TranscriptionOperation, TranscriptionRequestIR};
use axum::extract::{State, multipart::Multipart};
use axum::http::{HeaderValue, StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// `POST /v1/audio/transcriptions` and `/v1/audio/translations` (multipart), and
/// `POST /v1/audio/speech` (JSON). Responses relay the provider's bytes and content type.
pub struct OpenAIAudioSkin;

#[derive(Debug, thiserror::Error)]
pub enum AudioSkinError {
	#[error(transparent)]
	Json(serde_json::Error),
	#[error("{0}")]
	Invalid(String),
}

#[derive(Deserialize)]
struct TranscriptionFields {
	#[serde(default)]
	language: Option<String>,
	#[serde(default)]
	prompt: Option<String>,
	#[serde(default)]
	response_format: Option<String>,
	#[serde(default, deserialize_with = "bool_or_string")]
	stream: Option<bool>,
}

#[derive(Deserialize)]
struct SpeechFields {
	input: String,
	#[serde(default)]
	voice: Option<Value>,
	#[serde(default)]
	instructions: Option<String>,
	#[serde(default)]
	response_format: Option<String>,
	#[serde(default)]
	speed: Option<f64>,
	#[serde(default)]
	stream_format: Option<String>,
}

impl OpenAIAudioSkin {
	pub async fn handle_transcriptions(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
	) -> Response {
		Self::transcribe(ctx, metadata, multipart, TranscriptionOperation::Transcribe).await
	}

	pub async fn handle_translations(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
	) -> Response {
		Self::transcribe(ctx, metadata, multipart, TranscriptionOperation::Translate).await
	}

	pub async fn handle_speech(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		crate::server::SkinAwareJson(body): crate::server::SkinAwareJson<Map<String, Value>>,
	) -> Response {
		let model = match resolve(&ctx, &body, ModelOperation::TextToSpeech).await {
			Ok(model) => model,
			Err(response) => return response,
		};
		let mut request = match Self::speech_to_ir(body, model) {
			Ok(request) => request,
			Err(error) => return ctx.handle_json_error(error),
		};
		request.metadata = request_metadata(metadata);
		match ctx.speech(request).await {
			Ok(response) => relay(response),
			Err(error) => ctx.handle_inference_error(&error),
		}
	}

	/// Converts the multipart text fields (a repeated field as an array) and the uploaded `file`
	/// of a transcription or translation into the IR for an already resolved `model`.
	pub fn transcription_to_ir(
		fields: Map<String, Value>,
		audio: Option<AudioInput>,
		operation: TranscriptionOperation,
		model: ModelRef,
	) -> Result<TranscriptionRequestIR, AudioSkinError> {
		let audio = audio.ok_or_else(|| AudioSkinError::Invalid("Missing required parameter: 'file'.".into()))?;
		let typed: TranscriptionFields = serde_json::from_value(Value::Object(fields.clone())).map_err(AudioSkinError::Json)?;
		let timestamp_granularities = match fields.get("timestamp_granularities[]").or_else(|| fields.get("timestamp_granularities")) {
			Some(Value::String(granularity)) => vec![granularity.clone()],
			Some(Value::Array(items)) => items.iter().filter_map(|item| item.as_str().map(str::to_string)).collect(),
			_ => Vec::new(),
		};
		Ok(TranscriptionRequestIR {
			model,
			operation,
			audio,
			language: typed.language,
			prompt: typed.prompt,
			response_format: typed.response_format,
			timestamp_granularities,
			stream: typed.stream.unwrap_or(false),
			raw_request: Some(fields),
			metadata: BTreeMap::new(),
		})
	}

	/// Converts an OpenAI speech body into the IR for an already resolved `model`.
	pub fn speech_to_ir(body: Map<String, Value>, model: ModelRef) -> Result<SpeechRequestIR, serde_json::Error> {
		let fields: SpeechFields = serde_json::from_value(Value::Object(body.clone()))?;
		Ok(SpeechRequestIR {
			model,
			input: fields.input,
			voice: fields.voice,
			instructions: fields.instructions,
			response_format: fields.response_format,
			speed: fields.speed,
			stream_format: fields.stream_format,
			raw_request: Some(body),
			metadata: BTreeMap::new(),
		})
	}

	async fn transcribe(
		ctx: SkinContext,
		metadata: Metadata,
		multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
		operation: TranscriptionOperation,
	) -> Response {
		let (fields, audio) = match read_multipart(multipart).await {
			Ok(parts) => parts,
			Err(response) => return response,
		};
		let model = match resolve(&ctx, &fields, ModelOperation::SpeechToText).await {
			Ok(model) => model,
			Err(response) => return response,
		};
		let mut request = match Self::transcription_to_ir(fields, audio, operation, model) {
			Ok(request) => request,
			Err(AudioSkinError::Json(error)) => return ctx.handle_json_error(error),
			Err(AudioSkinError::Invalid(message)) => return invalid_body(StatusCode::BAD_REQUEST, message),
		};
		request.metadata = request_metadata(metadata);
		match ctx.transcription(request).await {
			Ok(response) => relay(response),
			Err(error) => ctx.handle_inference_error(&error),
		}
	}
}

#[allow(clippy::result_large_err, reason = "the error is the HTTP response handlers return as is")]
async fn resolve(ctx: &SkinContext, body: &Map<String, Value>, operation: ModelOperation) -> Result<ModelRef, Response> {
	match body.get("model") {
		Some(Value::String(model)) => ctx.resolve_model_for(model, operation).await,
		_ => Err(ctx.handle_json_error(serde::de::Error::missing_field("model"))),
	}
}

#[allow(clippy::result_large_err, reason = "the error is the HTTP response handlers return as is")]
async fn read_multipart(multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>) -> Result<(Map<String, Value>, Option<AudioInput>), Response> {
	let mut multipart = multipart.map_err(|rejection| invalid_body(rejection.status(), rejection.body_text()))?;
	let mut fields = Map::new();
	let mut audio = None;
	while let Some(field) = multipart.next_field().await.map_err(|error| invalid_body(error.status(), error.body_text()))? {
		let name = field.name().unwrap_or_default().to_string();
		let filename = field.file_name().unwrap_or("audio").to_string();
		let media_type = field.content_type().unwrap_or("application/octet-stream").to_string();
		let bytes = field.bytes().await.map_err(|error| invalid_body(error.status(), error.body_text()))?;
		if name == "file" {
			audio = Some(AudioInput {
				bytes: bytes.into(),
				filename,
				media_type,
			});
			continue;
		}
		let text = Value::String(String::from_utf8_lossy(&bytes).into_owned());
		match fields.get_mut(&name) {
			Some(Value::Array(items)) => items.push(text),
			Some(existing) => *existing = Value::Array(vec![existing.take(), text]),
			None => {
				fields.insert(name, text);
			}
		}
	}
	Ok((fields, audio))
}

/// Streams the provider body with its content type. A mid-stream failure becomes an OpenAI error
/// event on SSE bodies; other bodies are cut off, since bytes were already sent.
fn relay(response: ByteStreamResponse) -> Response {
	let sse = response.content_type.starts_with("text/event-stream");
	let content_type = HeaderValue::try_from(response.content_type).unwrap_or(HeaderValue::from_static("application/octet-stream"));
	let body = response.body.filter_map(move |event| async move {
		match event {
			ByteStreamEvent::Data(bytes) => Some(Ok(bytes)),
			ByteStreamEvent::Usage(_) => None,
			ByteStreamEvent::Error(error) if sse => Some(Ok(crate::skins::openai::errors::openai_stream_error_frame(&error))),
			ByteStreamEvent::Error(error) => Some(Err(std::io::Error::other(error.to_string()))),
		}
	});
	([(CONTENT_TYPE, content_type)], axum::body::Body::from_stream(body)).into_response()
}

/// Multipart carries `stream` as text, JSON as a boolean.
fn bool_or_string<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<bool>, D::Error> {
	match Option::<Value>::deserialize(deserializer)? {
		None | Some(Value::Null) => Ok(None),
		Some(Value::Bool(value)) => Ok(Some(value)),
		Some(Value::String(text)) => match text.trim() {
			"true" => Ok(Some(true)),
			"false" => Ok(Some(false)),
			_ => Err(serde::de::Error::custom("stream must be a boolean")),
		},
		Some(_) => Err(serde::de::Error::custom("stream must be a boolean")),
	}
}

use super::context::{SkinContext, SkinRequestMetadata};
use crate::types::{EmbeddingInput, EmbeddingRequestIR, ImageInput, ImageOperation, ImageOptions, ImageRequestIR, ImageResponse, ModelOperation, ModelRef, ProviderKind};
use axum::extract::{Extension, State, multipart::Multipart};
use axum::response::{IntoResponse, Response};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

type Metadata = Option<Extension<SkinRequestMetadata>>;

/// `POST /v1/embeddings`.
pub struct OpenAIEmbeddingsSkin;

#[derive(Deserialize)]
struct EmbeddingFields {
	model: String,
	input: EmbeddingInput,
	#[serde(default)]
	dimensions: Option<u32>,
	#[serde(default)]
	encoding_format: Option<String>,
}

impl OpenAIEmbeddingsSkin {
	/// Converts an OpenAI embeddings body into the IR for an already resolved `model`.
	pub fn external_to_ir(body: Map<String, Value>, model: ModelRef) -> Result<EmbeddingRequestIR, serde_json::Error> {
		let fields: EmbeddingFields = serde_json::from_value(Value::Object(body.clone()))?;
		Ok(EmbeddingRequestIR {
			model,
			input: fields.input,
			dimensions: fields.dimensions,
			encoding_format: fields.encoding_format,
			raw_request: Some(body),
			metadata: BTreeMap::new(),
		})
	}

	pub async fn handle_embeddings(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		crate::server::SkinAwareJson(body): crate::server::SkinAwareJson<Map<String, Value>>,
	) -> Response {
		let fields: EmbeddingFields = match serde_json::from_value(Value::Object(body.clone())) {
			Ok(fields) => fields,
			Err(error) => return ctx.handle_json_error(error),
		};
		let model = match ctx.resolve_model_for(&fields.model, ModelOperation::Embeddings).await {
			Ok(model) => model,
			Err(response) => return response,
		};
		let mut request = match Self::external_to_ir(body, model) {
			Ok(request) => request,
			Err(error) => return ctx.handle_json_error(error),
		};
		request.metadata = request_metadata(metadata);
		match ctx.embeddings(request).await {
			Ok(response) => axum::Json(response.raw).into_response(),
			Err(error) => ctx.handle_inference_error(&error),
		}
	}
}

/// `POST /v1/images/generations` (JSON) and `POST /v1/images/edits` (multipart).
pub struct OpenAIImagesSkin;

#[derive(Deserialize)]
struct ImageFields {
	#[allow(dead_code, reason = "required in the body; resolved before conversion")]
	model: String,
	prompt: String,
	#[serde(default, deserialize_with = "number_or_string")]
	n: Option<u32>,
	#[serde(default)]
	size: Option<String>,
	#[serde(default)]
	quality: Option<String>,
	#[serde(default)]
	output_format: Option<String>,
	#[serde(default)]
	background: Option<String>,
	#[serde(default)]
	response_format: Option<String>,
}

impl OpenAIImagesSkin {
	pub async fn handle_generations(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		crate::server::SkinAwareJson(body): crate::server::SkinAwareJson<Map<String, Value>>,
	) -> Response {
		Self::execute(ctx, metadata, body, Vec::new(), None, ImageOperation::Generate).await
	}

	pub async fn handle_edits(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
	) -> Response {
		let mut multipart = match multipart {
			Ok(multipart) => multipart,
			Err(rejection) => return invalid_body(rejection.status(), rejection.body_text()),
		};
		let mut fields = Map::new();
		let mut images = Vec::new();
		let mut mask = None;
		loop {
			let field = match multipart.next_field().await {
				Ok(Some(field)) => field,
				Ok(None) => break,
				Err(error) => return invalid_body(error.status(), error.body_text()),
			};
			let name = field.name().unwrap_or_default().to_string();
			let media_type = field.content_type().unwrap_or("image/png").to_string();
			let bytes = match field.bytes().await {
				Ok(bytes) => bytes,
				Err(error) => return invalid_body(error.status(), error.body_text()),
			};
			match name.as_str() {
				"image" | "image[]" => images.push(ImageInput {
					bytes: bytes.to_vec(),
					media_type,
				}),
				"mask" => {
					mask = Some(ImageInput {
						bytes: bytes.to_vec(),
						media_type,
					})
				}
				_ => {
					fields.insert(name, Value::String(String::from_utf8_lossy(&bytes).into_owned()));
				}
			}
		}
		Self::execute(ctx, metadata, fields, images, mask, ImageOperation::Edit).await
	}

	/// Converts an OpenAI Images body (the JSON body, or the multipart text fields of an edit) into
	/// the IR for an already resolved `model`.
	pub fn external_to_ir(
		body: Map<String, Value>,
		input_images: Vec<ImageInput>,
		mask: Option<ImageInput>,
		operation: ImageOperation,
		model: ModelRef,
	) -> Result<ImageRequestIR, ImageSkinError> {
		let fields: ImageFields = serde_json::from_value(Value::Object(body.clone())).map_err(ImageSkinError::Json)?;
		if operation == ImageOperation::Edit && input_images.is_empty() {
			return Err(ImageSkinError::Invalid("Missing required parameter: 'image'.".into()));
		}
		if fields.response_format.as_deref() == Some("url") && !relays_openai_images(&model) {
			return Err(ImageSkinError::Invalid(
				"Invalid value for 'response_format': 'url' is only supported by OpenAI-compatible providers.".into(),
			));
		}
		Ok(ImageRequestIR {
			model,
			operation,
			prompt: fields.prompt,
			request_id: None,
			input_images,
			options: ImageOptions {
				size: fields.size,
				aspect_ratio: None,
				quality: fields.quality,
				output_format: fields.output_format,
				background: fields.background,
				n: fields.n,
			},
			mask,
			raw_request: Some(body),
			metadata: BTreeMap::new(),
		})
	}

	/// Renders an image response as the OpenAI Images JSON: the provider's own JSON when it speaks
	/// that protocol, otherwise `{created, data: [{b64_json}], usage}`.
	pub fn render(response: ImageResponse) -> Value {
		render_images(response)
	}

	async fn execute(
		ctx: SkinContext,
		metadata: Metadata,
		body: Map<String, Value>,
		input_images: Vec<ImageInput>,
		mask: Option<ImageInput>,
		operation: ImageOperation,
	) -> Response {
		let model_name = match body.get("model") {
			Some(Value::String(model)) => model.clone(),
			_ => return ctx.handle_json_error(serde::de::Error::missing_field("model")),
		};
		let required = if operation == ImageOperation::Edit {
			ModelOperation::ImageEditing
		} else {
			ModelOperation::ImageGeneration
		};
		let model = match ctx.resolve_model_for(&model_name, required).await {
			Ok(model) => model,
			Err(response) => return response,
		};
		let mut request = match Self::external_to_ir(body, input_images, mask, operation, model) {
			Ok(request) => request,
			Err(ImageSkinError::Json(error)) => return ctx.handle_json_error(error),
			Err(ImageSkinError::Invalid(message)) => return invalid_body(axum::http::StatusCode::BAD_REQUEST, message),
		};
		let metadata = request_metadata(metadata);
		request.request_id = metadata.get("request_id").cloned();
		request.metadata = metadata;
		match ctx.image(request).await {
			Ok(response) => axum::Json(render_images(response)).into_response(),
			Err(error) => ctx.handle_inference_error(&error),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum ImageSkinError {
	#[error(transparent)]
	Json(serde_json::Error),
	#[error("{0}")]
	Invalid(String),
}

fn relays_openai_images(model: &ModelRef) -> bool {
	matches!(model.provider.endpoint.kind, ProviderKind::OpenAI | ProviderKind::OpenAICompat)
}

fn render_images(response: ImageResponse) -> Value {
	if let Some(raw) = response.raw {
		return raw;
	}
	let created = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |elapsed| elapsed.as_secs());
	let usage = &response.usage;
	json!({
		"created": created,
		"data": response.images.iter().map(|image| json!({"b64_json": BASE64.encode(&image.bytes)})).collect::<Vec<_>>(),
		"usage": {
			"input_tokens": usage.input_tokens,
			"output_tokens": usage.output_tokens,
			"total_tokens": usage.input_tokens + usage.output_tokens,
		},
	})
}

fn request_metadata(metadata: Metadata) -> BTreeMap<String, String> {
	let mut values = BTreeMap::from([("request_id".to_string(), uuid::Uuid::new_v4().to_string())]);
	if let Some(Extension(metadata)) = metadata {
		values.extend(metadata.0);
	}
	values
}

fn invalid_body(status: axum::http::StatusCode, message: String) -> Response {
	let error = super::openai_error_response(message, "invalid_request_error", "invalid_request_body");
	(status, axum::Json(error)).into_response()
}

/// Multipart edits carry `n` as text, JSON generations as a number.
fn number_or_string<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<u32>, D::Error> {
	match Option::<Value>::deserialize(deserializer)? {
		None | Some(Value::Null) => Ok(None),
		Some(Value::Number(number)) => number
			.as_u64()
			.and_then(|n| u32::try_from(n).ok())
			.map(Some)
			.ok_or_else(|| serde::de::Error::custom("n must be a positive integer")),
		Some(Value::String(text)) => text.trim().parse().map(Some).map_err(|_| serde::de::Error::custom("n must be a positive integer")),
		Some(_) => Err(serde::de::Error::custom("n must be a positive integer")),
	}
}

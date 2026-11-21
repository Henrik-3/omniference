//! What the decision skins share: their error, the image check and the handler around a protocol's conversion.

use super::context::{Metadata, SkinContext, request_metadata};
use crate::skins::openai::errors::invalid_body;
use crate::types::{DecisionInput, DecisionPart, DecisionQuestion, DecisionRequestIR, DecisionResponse, Modality, ModelOperation, ModelRef};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value};

#[derive(Debug, thiserror::Error)]
pub enum DecisionSkinError {
	#[error(transparent)]
	Json(serde_json::Error),
	#[error("{0}")]
	Invalid(String),
}

/// Resolves the model, converts the body with `to_ir`, runs the decision and renders it with `render`.
pub(crate) async fn handle(
	ctx: SkinContext,
	metadata: Metadata,
	body: Map<String, Value>,
	to_ir: fn(Map<String, Value>, ModelRef) -> Result<DecisionRequestIR, DecisionSkinError>,
	render: fn(&Map<String, Value>, &[DecisionQuestion], DecisionResponse) -> Value,
) -> Response {
	let model_name = match body.get("model") {
		Some(Value::String(model)) => model.clone(),
		_ => return ctx.handle_json_error(serde::de::Error::missing_field("model")),
	};
	let model = match ctx.resolve_model_for(&model_name, ModelOperation::Decision).await {
		Ok(model) => model,
		Err(response) => return response,
	};
	let mut request = match to_ir(body.clone(), model) {
		Ok(request) => request,
		Err(DecisionSkinError::Json(error)) => return ctx.handle_json_error(error),
		Err(DecisionSkinError::Invalid(message)) => return invalid_body(axum::http::StatusCode::BAD_REQUEST, message),
	};
	request.metadata = request_metadata(metadata);
	let questions = request.questions.clone();
	match ctx.decision(request).await {
		Ok(response) => axum::Json(render(&body, &questions, response)).into_response(),
		Err(error) => ctx.handle_inference_error(&error),
	}
}

/// Refuses images for a model that does not take them.
pub(crate) fn check_images(request: &DecisionRequestIR) -> Result<(), DecisionSkinError> {
	let image = |parts: &[DecisionPart]| parts.iter().any(|part| matches!(part, DecisionPart::Image { .. }));
	let has_images = match &request.input {
		DecisionInput::Content(parts) => image(parts),
		DecisionInput::Messages(items) => serde_json::to_string(items).is_ok_and(|text| text.contains("\"input_image\"")),
	} || request.questions.iter().any(|question| image(&question.instructions));
	if has_images && !request.model.input_modalities.contains(&Modality::Image) {
		return Err(DecisionSkinError::Invalid(format!("Model '{}' does not accept images.", request.model.alias)));
	}
	Ok(())
}

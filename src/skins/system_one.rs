//! `POST /v1/systemone`: the client side of the System One decision protocol ([`crate::adapters::SystemOneAdapter`]).

use crate::adapters::system_one::option_key;
use crate::skins::common::context::{Metadata, SkinContext};
use crate::skins::common::decisions::{DecisionSkinError, check_images, handle};
use crate::types::{
	DecisionAnswerKind, DecisionInput, DecisionLevel, DecisionOption, DecisionPart, DecisionProtocol, DecisionQuestion, DecisionQuestionKind, DecisionRequestIR,
	DecisionResponse, ModelRef, ProviderKind,
};
use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// `POST /v1/systemone` (TypeSafe System One).
pub struct SystemOneSkin;

#[derive(Deserialize)]
struct SystemOneFields {
	state: Value,
	questions: BTreeMap<String, SystemOneQuestion>,
	#[serde(default)]
	assets: BTreeMap<String, SystemOneAsset>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SystemOneQuestion {
	Noul {
		#[serde(default)]
		instructions: Value,
		#[serde(default)]
		criteria: Option<NoulCriteria>,
	},
	Choice {
		#[serde(default)]
		instructions: Value,
		criteria: BTreeMap<String, Value>,
	},
	Score {
		#[serde(default)]
		instructions: Value,
		criteria: Vec<Value>,
	},
	Box {
		#[serde(default)]
		instructions: Value,
	},
}

#[derive(Deserialize)]
struct NoulCriteria {
	#[serde(default, rename = "true")]
	if_true: Option<Value>,
	#[serde(default, rename = "false")]
	if_false: Option<Value>,
}

#[derive(Deserialize)]
struct SystemOneAsset {
	#[serde(default)]
	format: Option<String>,
	#[serde(default)]
	data_base64: Option<String>,
}

impl SystemOneSkin {
	/// Converts a System One body into the IR for an already resolved `model`. Asset references
	/// used as a whole state, state item or instructions value become images; for providers other
	/// than System One, references anywhere else are rejected because they cannot be translated.
	pub fn external_to_ir(body: Map<String, Value>, model: ModelRef) -> Result<DecisionRequestIR, DecisionSkinError> {
		let fields: SystemOneFields = serde_json::from_value(Value::Object(body.clone())).map_err(DecisionSkinError::Json)?;
		let images = fields
			.assets
			.into_iter()
			.map(|(reference, asset)| {
				let media_type = match asset.format.as_deref() {
					Some("png") => "image/png",
					Some("jpg" | "jpeg") => "image/jpeg",
					_ => return Err(DecisionSkinError::Invalid("Invalid asset 'format': expected png, jpg or jpeg.".into())),
				};
				let data = asset
					.data_base64
					.ok_or_else(|| DecisionSkinError::Invalid("Assets need 'data_base64'; multipart parts are not supported.".into()))?;
				Ok((reference, format!("data:{media_type};base64,{data}")))
			})
			.collect::<Result<BTreeMap<_, _>, DecisionSkinError>>()?;
		let parts = |value: &Value| state_parts(value, &images);
		let questions = fields
			.questions
			.into_iter()
			.map(|(id, question)| {
				let (instructions, kind) = match question {
					SystemOneQuestion::Noul { instructions, criteria } => {
						let criteria = criteria.unwrap_or(NoulCriteria { if_true: None, if_false: None });
						(
							instructions,
							DecisionQuestionKind::Predicate {
								if_true: criteria.if_true,
								if_false: criteria.if_false,
							},
						)
					}
					SystemOneQuestion::Choice { instructions, criteria } => {
						let options = criteria
							.into_iter()
							.map(|(value, description)| DecisionOption {
								value: json!(value),
								description: (!description.is_null()).then_some(description),
							})
							.collect();
						(instructions, DecisionQuestionKind::Choice { options })
					}
					SystemOneQuestion::Score { instructions, criteria } => {
						let levels = criteria
							.into_iter()
							.map(|criterion| match criterion.get("label").and_then(Value::as_str) {
								Some(label) => DecisionLevel {
									label: label.to_string(),
									description: criterion.get("description").cloned(),
								},
								None => DecisionLevel {
									label: criterion.as_str().map_or_else(|| criterion.to_string(), str::to_string),
									description: None,
								},
							})
							.collect();
						(instructions, DecisionQuestionKind::Score { levels })
					}
					SystemOneQuestion::Box { instructions } => (instructions, DecisionQuestionKind::Box),
				};
				DecisionQuestion {
					id: Some(id),
					instructions: parts(&instructions),
					kind,
				}
			})
			.collect();
		let request = DecisionRequestIR {
			input: DecisionInput::Content(parts(&fields.state)),
			questions,
			protocol: DecisionProtocol::SystemOne,
			raw_request: Some(body),
			metadata: BTreeMap::new(),
			model,
		};
		if request.model.provider.endpoint.kind != ProviderKind::SystemOne && nests_references(&request, &images) {
			return Err(DecisionSkinError::Invalid(
				"Assets used inside longer strings, objects or criteria can only be sent to System One providers.".into(),
			));
		}
		check_images(&request)?;
		Ok(request)
	}

	pub async fn handle_systemone(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		crate::server::SkinAwareJson(body): crate::server::SkinAwareJson<Map<String, Value>>,
	) -> Response {
		handle(ctx, metadata, body, Self::external_to_ir, systemone_response).await
	}
}

/// System One state or instructions: an array is one part per item; a string that names an asset is that image.
fn state_parts(value: &Value, images: &BTreeMap<String, String>) -> Vec<DecisionPart> {
	let part = |value: &Value| match value {
		Value::String(text) => match images.get(text) {
			Some(url) => DecisionPart::Image { url: url.clone(), detail: None },
			None => DecisionPart::Text { text: text.clone() },
		},
		other => DecisionPart::Json { data: other.clone() },
	};
	match value {
		Value::Null => Vec::new(),
		Value::Array(items) => items.iter().map(part).collect(),
		other => vec![part(other)],
	}
}

/// System One also replaces references inside longer strings, objects and criteria; those have no
/// image equivalent in other protocols.
/// Only content is searched (texts, JSON string values, criteria), never question names or field names.
fn nests_references(request: &DecisionRequestIR, images: &BTreeMap<String, String>) -> bool {
	fn in_text(text: &str, images: &BTreeMap<String, String>) -> bool {
		images.keys().any(|reference| text.contains(reference.as_str()))
	}
	fn in_value(value: &Value, images: &BTreeMap<String, String>) -> bool {
		match value {
			Value::String(text) => in_text(text, images),
			Value::Array(items) => items.iter().any(|item| in_value(item, images)),
			Value::Object(fields) => fields.values().any(|item| in_value(item, images)),
			_ => false,
		}
	}
	let in_parts = |parts: &[DecisionPart]| {
		parts.iter().any(|part| match part {
			DecisionPart::Text { text } => in_text(text, images),
			DecisionPart::Json { data } => in_value(data, images),
			DecisionPart::Image { .. } => false,
		})
	};
	let in_kind = |kind: &DecisionQuestionKind| match kind {
		DecisionQuestionKind::Predicate { if_true, if_false } => [if_true, if_false].into_iter().flatten().any(|value| in_value(value, images)),
		DecisionQuestionKind::Choice { options } => options.iter().filter_map(|option| option.description.as_ref()).any(|value| in_value(value, images)),
		DecisionQuestionKind::Score { levels } => levels
			.iter()
			.any(|level| in_text(&level.label, images) || level.description.as_ref().is_some_and(|value| in_value(value, images))),
		DecisionQuestionKind::Box => false,
	};
	matches!(&request.input, DecisionInput::Content(parts) if in_parts(parts))
		|| request.questions.iter().any(|question| in_parts(&question.instructions) || in_kind(&question.kind))
}

fn systemone_response(body: &Map<String, Value>, questions: &[DecisionQuestion], response: DecisionResponse) -> Value {
	if response.protocol == DecisionProtocol::SystemOne {
		return response.raw;
	}
	let answers: Map<String, Value> = response
		.answers
		.iter()
		.zip(questions)
		.map(|(answer, question)| {
			let key = question.id.clone().unwrap_or_default();
			let value = match &answer.kind {
				DecisionAnswerKind::Predicate { probability } => json!({"type": "noul", "noul": probability}),
				DecisionAnswerKind::Choice {
					choice,
					confidence,
					probabilities,
				} => {
					let probabilities: Map<String, Value> = probabilities.iter().map(|item| (option_key(&item.value), json!(item.probability))).collect();
					json!({"type": "choice", "choice": option_key(choice), "confidence": confidence, "probabilities": probabilities})
				}
				DecisionAnswerKind::Score {
					score,
					confidence,
					probabilities,
				} => {
					// The legend repeats the score criteria as the client sent them.
					let criteria = body
						.get("questions")
						.and_then(|questions| questions.get(&key))
						.and_then(|question| question.get("criteria"));
					let legend: Map<String, Value> = criteria
						.and_then(Value::as_array)
						.into_iter()
						.flatten()
						.enumerate()
						.map(|(index, criterion)| (index.to_string(), criterion.clone()))
						.collect();
					let probabilities: Map<String, Value> = probabilities
						.iter()
						.enumerate()
						.map(|(index, probability)| (index.to_string(), json!(probability)))
						.collect();
					json!({"type": "score", "score": score, "confidence": confidence, "legend": legend, "probabilities": probabilities})
				}
				DecisionAnswerKind::Box {
					top,
					bottom,
					left,
					right,
					confidence,
				} => json!({"type": "box", "box": {"top": top, "bottom": bottom, "left": left, "right": right}, "confidence": confidence}),
				DecisionAnswerKind::Refusal => json!({"type": "refusal"}),
			};
			(key, value)
		})
		.collect();
	json!({
		"model": response.model,
		"answers": answers,
		"usage": {"input_tokens": response.usage.input_tokens, "output_tokens": 0},
	})
}

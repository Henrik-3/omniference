use crate::skins::common::context::{Metadata, SkinContext};
use crate::skins::common::decisions::{DecisionSkinError, check_images, handle};
use crate::types::{
	DecisionAnswerKind, DecisionInput, DecisionLevel, DecisionOption, DecisionPart, DecisionProtocol, DecisionQuestion, DecisionQuestionKind, DecisionRequestIR,
	DecisionResponse, ModelRef,
};
use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// `POST /v1/decisions` (OpenAI).
pub struct OpenAIDecisionsSkin;

#[derive(Deserialize)]
struct OpenAIFields {
	input: Value,
	questions: Vec<OpenAIQuestion>,
}

#[derive(Deserialize)]
struct OpenAIQuestion {
	#[serde(default)]
	name: Option<String>,
	#[serde(default)]
	instructions: Value,
	#[serde(flatten)]
	kind: OpenAIQuestionKind,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIQuestionKind {
	Predicate,
	Choice { choices: Vec<DecisionOption> },
	Score { levels: Vec<DecisionLevel> },
}

impl OpenAIDecisionsSkin {
	/// Converts an OpenAI decisions body into the IR for an already resolved `model`. A single
	/// user message becomes content; any other input list is kept as messages (OpenAI only).
	pub fn external_to_ir(body: Map<String, Value>, model: ModelRef) -> Result<DecisionRequestIR, DecisionSkinError> {
		let fields: OpenAIFields = serde_json::from_value(Value::Object(body.clone())).map_err(DecisionSkinError::Json)?;
		let input = match fields.input {
			Value::String(text) => DecisionInput::Content(vec![DecisionPart::Text { text }]),
			Value::Array(items) => match items.as_slice() {
				[item] if item.get("role") == Some(&json!("user")) && item.get("type").is_none_or(|kind| kind == "message") => item
					.get("content")
					.and_then(openai_parts)
					.map_or(DecisionInput::Messages(items.clone()), DecisionInput::Content),
				_ => DecisionInput::Messages(items),
			},
			_ => return Err(DecisionSkinError::Invalid("Invalid 'input': expected a string or an array of input items.".into())),
		};
		let questions = fields
			.questions
			.into_iter()
			.map(|question| {
				let instructions = match &question.instructions {
					Value::Null => Vec::new(),
					instructions => openai_parts(instructions)
						.ok_or_else(|| DecisionSkinError::Invalid("Invalid 'instructions': only input_text, input_image and input_json parts are supported.".into()))?,
				};
				let kind = match question.kind {
					OpenAIQuestionKind::Predicate => DecisionQuestionKind::Predicate { if_true: None, if_false: None },
					OpenAIQuestionKind::Choice { choices } => DecisionQuestionKind::Choice { options: choices },
					OpenAIQuestionKind::Score { levels } => DecisionQuestionKind::Score { levels },
				};
				Ok(DecisionQuestion {
					id: question.name,
					instructions,
					kind,
				})
			})
			.collect::<Result<_, DecisionSkinError>>()?;
		let request = DecisionRequestIR {
			model,
			input,
			questions,
			protocol: DecisionProtocol::OpenAI,
			raw_request: Some(body),
			metadata: BTreeMap::new(),
		};
		check_images(&request)?;
		Ok(request)
	}

	pub async fn handle_decisions(
		State(ctx): State<SkinContext>,
		metadata: Metadata,
		crate::server::SkinAwareJson(body): crate::server::SkinAwareJson<Map<String, Value>>,
	) -> Response {
		handle(ctx, metadata, body, Self::external_to_ir, |_, questions, response| {
			openai_response(questions, response)
		})
		.await
	}
}

/// OpenAI content: a string, or `input_text`/`input_image`/`input_json` parts.
fn openai_parts(content: &Value) -> Option<Vec<DecisionPart>> {
	let part = |part: &Value| match part.get("type")?.as_str()? {
		"input_text" => Some(DecisionPart::Text {
			text: part.get("text")?.as_str()?.to_string(),
		}),
		"input_json" => Some(DecisionPart::Json { data: part.get("data")?.clone() }),
		"input_image" => Some(DecisionPart::Image {
			url: part.get("image_url")?.as_str()?.to_string(),
			detail: part.get("detail").and_then(Value::as_str).map(str::to_string),
		}),
		_ => None,
	};
	match content {
		Value::String(text) => Some(vec![DecisionPart::Text { text: text.clone() }]),
		Value::Array(parts) => parts.iter().map(part).collect(),
		_ => None,
	}
}

fn openai_response(questions: &[DecisionQuestion], response: DecisionResponse) -> Value {
	if response.protocol == DecisionProtocol::OpenAI {
		return response.raw;
	}
	let answers: Vec<Value> = response
		.answers
		.iter()
		.zip(questions)
		.map(|(answer, question)| {
			let mut value = serde_json::to_value(&answer.kind).unwrap_or_default();
			if let (DecisionAnswerKind::Score { probabilities, .. }, DecisionQuestionKind::Score { levels }) = (&answer.kind, &question.kind) {
				let levels = probabilities.iter().zip(levels).enumerate();
				value["probabilities"] = levels
					.map(|(index, (probability, level))| json!({"value": index, "label": level.label, "probability": probability}))
					.collect();
			}
			value["name"] = json!(answer.id);
			value
		})
		.collect();
	let usage = &response.usage;
	json!({
		"model": response.model,
		"answers": answers,
		"usage": {
			"input_tokens": usage.input_tokens,
			"input_tokens_details": {"cached_tokens": usage.cached_input_tokens, "cache_write_tokens": usage.cache_write_tokens},
			"output_tokens": 0,
			"output_tokens_details": {"reasoning_tokens": 0},
			"total_tokens": usage.input_tokens,
		},
	})
}

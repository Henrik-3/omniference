use crate::{
	adapters::{{AdapterError, ChatAdapter}, common::http::{provider_request, provider_url, send_with_timeout, shared_http_client}},
	catalog::UsageBreakdown,
	stream::StreamEvent,
	types::*,
};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use tokio_util::sync::CancellationToken;

/// TypeSafe-style System One servers: `POST {base}/v1/systemone` with bearer auth. The base URL is
/// the provider's, so TypeSafe, DigitalOcean and self-hosted servers all work.
pub struct SystemOneAdapter;

#[async_trait]
impl ChatAdapter for SystemOneAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::SystemOne
	}

	async fn execute_chat(
		&self,
		_ir: ChatRequestIR,
		_cancel: CancellationToken,
	) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		Err(AdapterError::invalid("System One providers only answer decisions"))
	}

	/// `GET /v1/models` lists `{name, description, release_date}` entries, as a bare array or under
	/// `models` (some servers use the OpenAI `data`/`id` shape instead).
	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let mut call = provider_request(shared_http_client().get(provider_url(&endpoint.base_url, "v1/models")), endpoint)?;
		if let Some(api_key) = endpoint.api_key.as_deref() {
			call = call.bearer_auth(api_key);
		}
		let response = send_with_timeout(call, None, false).await?;
		if !response.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(response, endpoint, provider_name).await);
		}
		let value: Value = response
			.json()
			.await
			.map_err(|error| AdapterError::http(format!("Failed to parse models response: {error}")))?;
		let entries = value
			.get("models")
			.or_else(|| value.get("data"))
			.unwrap_or(&value)
			.as_array()
			.ok_or_else(|| AdapterError::http("models response did not contain a model list"))?;
		Ok(entries
			.iter()
			.filter_map(|entry| entry.get("name").or_else(|| entry.get("id")).and_then(Value::as_str))
			.map(|model| DiscoveredModel {
				id: format!("{}/{model}", provider_name.to_lowercase()),
				name: model.to_string(),
				provider_name: provider_name.to_string(),
				provider_kind: ProviderKind::SystemOne,
				input_modalities: vec![Modality::Text, Modality::Image],
				output_modalities: vec![Modality::Text],
				capabilities: vec![ModelCapabilities::SystemOne],
				context_length: None,
				max_tokens: None,
				pricing: None,
				reasoning_budget: None,
			})
			.collect())
	}

	/// A System One request is relayed with only `model` rewritten; an OpenAI decisions request is
	/// rendered from the IR. Images and box questions need the `TypeSafe-Danger-Zone-Images` header.
	async fn execute_decision(&self, request: DecisionRequestIR) -> Result<DecisionResponse, AdapterError> {
		let keys = question_keys(&request.questions)?;
		let mut body = match request.raw_request.clone().filter(|_| request.protocol == DecisionProtocol::SystemOne) {
			Some(raw) => raw,
			None => request_body(&request, &keys)?,
		};
		body.insert("model".into(), json!(request.model.model_id));
		let images = body.contains_key("assets") || request.questions.iter().any(|question| question.kind == DecisionQuestionKind::Box);
		let (value, provider_cost) = crate::adapters::openai::endpoints::send(&request.model, "v1/systemone", |call| {
			let call = call.json(&body);
			if images { call.header("TypeSafe-Danger-Zone-Images", "true") } else { call }
		})
		.await?;
		let answers = value
			.get("answers")
			.and_then(Value::as_object)
			.ok_or_else(|| AdapterError::http("System One response did not contain answers"))?;
		let answers = request
			.questions
			.iter()
			.zip(&keys)
			.map(|(question, key)| {
				let answer = answers
					.get(key)
					.ok_or_else(|| AdapterError::http(format!("System One response has no answer for question `{key}`")))?;
				Ok(DecisionAnswer {
					id: question.id.clone(),
					kind: answer_kind(question, answer)?,
				})
			})
			.collect::<Result<Vec<_>, AdapterError>>()?;
		let input_tokens = value.pointer("/usage/input_tokens").and_then(Value::as_u64).unwrap_or(0);
		Ok(DecisionResponse {
			model: value.get("model").and_then(Value::as_str).unwrap_or(&request.model.model_id).to_string(),
			answers,
			usage: UsageBreakdown {
				input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
				requests: 1,
				..UsageBreakdown::default()
			},
			provider_cost,
			protocol: DecisionProtocol::SystemOne,
			raw: value,
		})
	}
}

/// The System One spelling of a choice value: strings as they are, booleans as `true`/`false`.
pub(crate) fn option_key(value: &Value) -> String {
	match value {
		Value::String(text) => text.clone(),
		other => other.to_string(),
	}
}

/// System One keys questions by name; unnamed OpenAI questions take their position.
fn question_keys(questions: &[DecisionQuestion]) -> Result<Vec<String>, AdapterError> {
	let keys: Vec<String> = questions
		.iter()
		.enumerate()
		.map(|(position, question)| question.id.clone().unwrap_or_else(|| position.to_string()))
		.collect();
	if keys.iter().collect::<BTreeSet<_>>().len() != keys.len() {
		return Err(AdapterError::invalid("System One needs a unique name for every question"));
	}
	Ok(keys)
}

fn request_body(request: &DecisionRequestIR, keys: &[String]) -> Result<Map<String, Value>, AdapterError> {
	let DecisionInput::Content(parts) = &request.input else {
		return Err(AdapterError::invalid(
			"System One needs a single user input to use as state; conversations, assistant turns and tool calls are not supported",
		));
	};
	let mut assets = Map::new();
	let state = state_value(parts, &mut assets)?;
	let mut questions = Map::new();
	for (question, key) in request.questions.iter().zip(keys) {
		let mut value = Map::new();
		match &question.kind {
			DecisionQuestionKind::Predicate { if_true, if_false } => {
				value.insert("type".into(), json!("noul"));
				let mut criteria = Map::new();
				if let Some(if_true) = if_true {
					criteria.insert("true".into(), if_true.clone());
				}
				if let Some(if_false) = if_false {
					criteria.insert("false".into(), if_false.clone());
				}
				if !criteria.is_empty() {
					value.insert("criteria".into(), Value::Object(criteria));
				}
			}
			DecisionQuestionKind::Choice { options } => {
				value.insert("type".into(), json!("choice"));
				let criteria = options
					.iter()
					.map(|option| (option_key(&option.value), option.description.clone().unwrap_or(Value::Null)));
				value.insert("criteria".into(), Value::Object(criteria.collect()));
			}
			DecisionQuestionKind::Score { levels } => {
				value.insert("type".into(), json!("score"));
				let criteria = levels.iter().map(|level| match &level.description {
					None => json!(level.label),
					Some(description) => json!({"label": level.label, "description": description}),
				});
				value.insert("criteria".into(), Value::Array(criteria.collect()));
			}
			DecisionQuestionKind::Box => {
				value.insert("type".into(), json!("box"));
			}
		}
		if !question.instructions.is_empty() {
			value.insert("instructions".into(), state_value(&question.instructions, &mut assets)?);
		}
		questions.insert(key.clone(), Value::Object(value));
	}
	let mut body = Map::from_iter([("state".to_string(), state), ("questions".to_string(), Value::Object(questions))]);
	if !assets.is_empty() {
		body.insert("assets".into(), Value::Object(assets));
	}
	Ok(body)
}

/// One part is sent as itself, several as an array; images become asset references.
fn state_value(parts: &[DecisionPart], assets: &mut Map<String, Value>) -> Result<Value, AdapterError> {
	let mut values = parts
		.iter()
		.map(|part| match part {
			DecisionPart::Text { text } => Ok(json!(text)),
			DecisionPart::Json { data } => Ok(data.clone()),
			DecisionPart::Image { url, .. } => {
				let (format, data) = inline_image(url)?;
				let reference = format!("image-{}", uuid::Uuid::new_v4());
				assets.insert(reference.clone(), json!({"type": "image", "format": format, "data_base64": data}));
				Ok(json!(reference))
			}
		})
		.collect::<Result<Vec<_>, AdapterError>>()?;
	Ok(if values.len() == 1 { values.remove(0) } else { Value::Array(values) })
}

/// System One takes inline PNG or JPEG bytes only.
fn inline_image(url: &str) -> Result<(&'static str, &str), AdapterError> {
	let (media_type, data) = url
		.strip_prefix("data:")
		.and_then(|rest| rest.split_once(";base64,"))
		.ok_or_else(|| AdapterError::invalid("System One needs images as base64 data URLs"))?;
	match media_type {
		"image/png" => Ok(("png", data)),
		"image/jpeg" | "image/jpg" => Ok(("jpeg", data)),
		_ => Err(AdapterError::invalid("System One supports PNG and JPEG images only")),
	}
}

fn answer_kind(question: &DecisionQuestion, answer: &Value) -> Result<DecisionAnswerKind, AdapterError> {
	let invalid = || AdapterError::http("invalid System One answer");
	let number = |key: &str| answer.get(key).and_then(Value::as_f64).ok_or_else(invalid);
	let probabilities = || answer.get("probabilities").and_then(Value::as_object).ok_or_else(invalid);
	Ok(match (answer.get("type").and_then(Value::as_str), &question.kind) {
		(Some("noul"), _) => DecisionAnswerKind::Predicate { probability: number("noul")? },
		(Some("choice"), DecisionQuestionKind::Choice { options }) => {
			let chosen = answer.get("choice").and_then(Value::as_str).ok_or_else(invalid)?;
			let probabilities = probabilities()?;
			DecisionAnswerKind::Choice {
				choice: options
					.iter()
					.find(|option| option_key(&option.value) == chosen)
					.map_or_else(|| json!(chosen), |option| option.value.clone()),
				confidence: number("confidence")?,
				probabilities: options
					.iter()
					.filter_map(|option| {
						let probability = probabilities.get(&option_key(&option.value))?.as_f64()?;
						Some(DecisionProbability {
							value: option.value.clone(),
							probability,
						})
					})
					.collect(),
			}
		}
		(Some("score"), DecisionQuestionKind::Score { levels }) => {
			let probabilities = probabilities()?;
			DecisionAnswerKind::Score {
				score: number("score")?,
				confidence: number("confidence")?,
				probabilities: (0..levels.len())
					.map(|level| probabilities.get(&level.to_string()).and_then(Value::as_f64).unwrap_or(0.0))
					.collect(),
			}
		}
		(Some("box"), _) => {
			let edge = |key: &str| answer.pointer(&format!("/box/{key}")).and_then(Value::as_f64).ok_or_else(invalid);
			DecisionAnswerKind::Box {
				top: edge("top")?,
				bottom: edge("bottom")?,
				left: edge("left")?,
				right: edge("right")?,
				confidence: answer.get("confidence").and_then(Value::as_f64),
			}
		}
		(Some("refusal"), _) => DecisionAnswerKind::Refusal,
		_ => return Err(invalid()),
	})
}

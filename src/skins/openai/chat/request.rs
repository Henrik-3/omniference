//! Chat Completions request → IR.

use crate::skins::{Skin, SkinErrorHandler};
use crate::types::*;


use std::collections::BTreeMap;
use uuid::Uuid;

use super::*;
use crate::skins::openai::{OPENAI_ERROR_HANDLER, anthropic_cache_from_extra};

fn dedup_reasoning_blocks(blocks: Vec<ReasoningBlock>) -> Vec<ReasoningBlock> {
	let mut seen = std::collections::HashSet::new();
	blocks
		.into_iter()
		.filter(|block| {
			if block.signature.is_none() && block.data.is_none() {
				return true;
			}
			seen.insert((block.format.clone(), block.signature.clone(), block.data.clone()))
		})
		.collect()
}

fn chat_reasoning_parts(extra: &BTreeMap<String, serde_json::Value>) -> Vec<ContentPart> {
	let structured = |key: &str, parse: fn(&serde_json::Value) -> Option<ReasoningBlock>| -> Vec<ReasoningBlock> {
		extra
			.get(key)
			.and_then(|value| value.as_array())
			.map(|items| items.iter().filter_map(parse).collect())
			.unwrap_or_default()
	};
	let mut blocks = structured("reasoning_details", ReasoningBlock::from_reasoning_detail);
	if blocks.is_empty() {
		blocks = structured("thinking_blocks", ReasoningBlock::from_thinking_block);
	}
	if blocks.is_empty() {
		let content = extra.get("reasoning_content").and_then(|value| value.as_str()).unwrap_or("");
		let reasoning = extra.get("reasoning").and_then(|value| value.as_str()).unwrap_or("");
		let mut unsigned = String::new();
		if !content.is_empty() {
			unsigned.push_str(content);
		}
		if !reasoning.is_empty() && reasoning != content {
			unsigned.push_str(reasoning);
		}
		if !unsigned.is_empty() {
			blocks.push(ReasoningBlock::unsigned(unsigned));
		}
	}
	dedup_reasoning_blocks(blocks).into_iter().map(ContentPart::Reasoning).collect()
}

/// A Chat function without `parameters` takes no arguments.
fn function_parameters(parameters: serde_json::Value) -> serde_json::Value {
	if parameters.is_null() { serde_json::json!({"type": "object", "properties": {}}) } else { parameters }
}

impl Skin for OpenAIChatSkin {
	type Request = OpenAIChatRequest;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		let original_request = req.clone();
		// `OpenAIFunctionCall` tolerates a missing upstream `function.name` (response-only leniency),
		// so the inbound contract is enforced here: a nameless tool call must stay a 400.
		for msg in &req.messages {
			if msg.tool_calls.as_ref().is_some_and(|calls| calls.iter().any(|call| call.function.name.is_empty())) {
				anyhow::bail!("assistant message `tool_calls` entries need a `function.name`");
			}
		}
		// Only openai_compat (and custom adapters) get the original body, where the provider judges unknown parts.
		if !matches!(model.provider.endpoint.kind, ProviderKind::OpenAICompat | ProviderKind::Custom(_)) {
			if let Some(tool) = req.tools.iter().flatten().find(|tool| tool.r#type != "function") {
				anyhow::bail!("tool type `{}` is not supported for this provider", tool.r#type);
			}
			let unknown = req.messages.iter().find_map(|msg| match &msg.content {
				// A known part without its payload object would be dropped, so it is refused too.
				Some(OpenAIMessageContent::Parts(items)) => items.iter().find(|item| match item.kind.as_str() {
					"text" => item.text.is_none(),
					"image_url" => item.image_url.is_none(),
					"audio" | "input_audio" => item.audio.is_none() && item.input_audio.is_none(),
					"file" => item.file.is_none(),
					"refusal" => false,
					_ => true,
				}),
				_ => None,
			});
			if let Some(item) = unknown {
				anyhow::bail!("content part type `{}` is not supported for this provider", item.kind);
			}
		}
		let messages: Vec<Message> = req
			.messages
			.into_iter()
			.map(|msg| {
				let role = match msg.role.as_str() {
					"developer" => Role::Developer,
					"system" => Role::System,
					// Legacy `function` results were always read as user messages.
					"user" | "function" => Role::User,
					"assistant" => Role::Assistant,
					"tool" => Role::Tool,
					other => anyhow::bail!("message role `{other}` is not supported"),
				};

				let mut parts: Vec<ContentPart> = Vec::new();
				match msg.content {
					Some(OpenAIMessageContent::Text(s)) => {
						parts.push(ContentPart::Text(s));
					}
					Some(OpenAIMessageContent::Parts(items)) => {
						for item in items {
							match item.kind.as_str() {
								"text" => {
									if let Some(t) = item.text {
										parts.push(ContentPart::Text(t));
									}
								}
								"image_url" => {
									if let Some(img) = item.image_url {
										let url = match img {
											OpenAIImageUrl::Url(u) => u,
											OpenAIImageUrl::Obj { url, .. } => url,
										};
										parts.push(ContentPart::ImageUrl { url, mime: None });
									}
								}
								"audio" | "input_audio" => {
									if let Some(audio) = item.input_audio.or(item.audio) {
										parts.push(ContentPart::Audio {
											data: audio.data,
											format: format!("{:?}", audio.format).to_lowercase(),
										});
									}
								}
								"refusal" => {
									if let Some(refusal) = item.extra.get("refusal").and_then(serde_json::Value::as_str) {
										parts.push(ContentPart::Text(refusal.to_string()));
									}
								}
								"file" => {
									if let Some(file) = item.file {
										parts.push(ContentPart::File {
											file_id: file.file_id,
											filename: file.filename,
											file_data: file.file_data,
										});
									}
								}
								_ => {}
							}
						}
					}
					None => {}
				}

				// Handle tool_calls from assistant messages
				if let Some(tool_calls) = msg.tool_calls {
					for tool_call in tool_calls {
						parts.push(ContentPart::ToolCall {
							id: tool_call.id,
							name: tool_call.function.name,
							arguments: tool_call.function.arguments,
						});
					}
				}

				parts.extend(chat_reasoning_parts(&msg.extra));

				Ok(Message {
					role,
					parts,
					name: msg.name.or(msg.tool_call_id),
				})
			})
			.collect::<anyhow::Result<_>>()?;

		let mut metadata: BTreeMap<String, String> = BTreeMap::new();
		metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
		if let Some(user) = req.user {
			metadata.insert("user".to_string(), user);
		}
		if let Some(seed) = req.seed {
			metadata.insert("seed".to_string(), seed.to_string());
		}
		if let Some(ref rf) = req.response_format {
			metadata.insert("response_format".to_string(), serde_json::to_string(&rf).unwrap_or_default());
		}
		if let Some(ref lb) = req.logit_bias {
			metadata.insert("logit_bias".to_string(), serde_json::to_string(lb).unwrap_or_default());
		}
		if let Some(lp) = req.logprobs {
			metadata.insert("logprobs".to_string(), lp.to_string());
		}
		if let Some(tlp) = req.top_logprobs {
			metadata.insert("top_logprobs".to_string(), tlp.to_string());
		}
		if let Some(n) = req.n {
			metadata.insert("n".to_string(), n.to_string());
		}
		if let Some(so) = req.stream_options {
			metadata.insert("stream_options".to_string(), serde_json::to_string(&so).unwrap_or_default());
		}
		if let Some(mods) = req.modalities {
			metadata.insert("modalities".to_string(), serde_json::to_string(&mods).unwrap_or_default());
		}
		if let Some(ref audio) = req.audio {
			metadata.insert("audio".to_string(), serde_json::to_string(audio).unwrap_or_default());
		}
		if let Some(ptc) = req.parallel_tool_calls {
			metadata.insert("parallel_tool_calls".to_string(), ptc.to_string());
		}
		if let Some(store) = req.store {
			metadata.insert("store".to_string(), store.to_string());
		}
		if let Some(req_metadata) = req.metadata {
			metadata.insert("request_metadata".to_string(), format!("{:?}", req_metadata));
		}
		if let Some(ref prediction) = req.prediction {
			metadata.insert("prediction".to_string(), format!("{:?}", prediction));
		}
		if let Some(service_tier) = req.service_tier {
			metadata.insert("service_tier".to_string(), serde_json::to_string(&service_tier).unwrap_or_default());
		}
		if let Some(verbosity) = req.verbosity {
			metadata.insert("verbosity".to_string(), verbosity);
		}
		if let Some(ref web_search_options) = req.web_search_options {
			metadata.insert("web_search_options".to_string(), serde_json::to_string(web_search_options).unwrap_or_default());
		}
		if let Some(ref prompt_cache_options) = req.prompt_cache_options {
			metadata.insert("prompt_cache_options".to_string(), prompt_cache_options.to_string());
		}
		if let Some(ref prompt_cache_retention) = req.prompt_cache_retention {
			metadata.insert("prompt_cache_retention".to_string(), prompt_cache_retention.clone());
		}
		if let Some(ref moderation) = req.moderation {
			metadata.insert("moderation".to_string(), moderation.to_string());
		}

		let anthropic_cache = anthropic_cache_from_extra(&req.extra)?;

		// Tools mapping
		let mut tools: Vec<ToolSpec> = req
			.tools
			.unwrap_or_default()
			.into_iter()
			.filter_map(|t| {
				if t.r#type == "function" {
					let function = t.function?;
					Some(ToolSpec::JsonSchema {
						name: function.name,
						description: function.description,
						schema: function_parameters(function.parameters),
						strict: function.strict,
						defer_loading: false,
					})
				} else {
					None
				}
			})
			.collect();
		// Map legacy functions
		if let Some(funcs) = req.functions {
			for f in funcs {
				// simple de-dup by name
				let name = f.name.clone();
				if tools.iter().any(|t| matches!(t, ToolSpec::JsonSchema { name: n, .. } if *n == name)) {
					continue;
				}
				tools.push(ToolSpec::JsonSchema {
					name,
					description: f.description,
					schema: function_parameters(f.parameters),
					strict: f.strict,
					defer_loading: false,
				});
			}
		}

		// Tool choice mapping
		let tool_choice = if let Some(fc) = req.function_call {
			if fc == serde_json::json!("none") {
				ToolChoice::None
			} else if fc == serde_json::json!("auto") {
				ToolChoice::Auto
			} else if let Some(name) = fc.get("name").and_then(|n| n.as_str()) {
				ToolChoice::Named(name.to_string())
			} else {
				ToolChoice::Auto
			}
		} else {
			match req.tool_choice {
				None => ToolChoice::Auto,
				Some(choice) => match choice {
					OpenAIToolChoice::String(s) if s == "none" => ToolChoice::None,
					OpenAIToolChoice::String(s) if s == "auto" => ToolChoice::Auto,
					OpenAIToolChoice::String(s) if s == "required" => ToolChoice::Required,
					OpenAIToolChoice::Named { function, .. } => ToolChoice::Named(function.name),
					OpenAIToolChoice::Allowed { allowed_tools, .. } => ToolChoice::Allowed {
						mode: allowed_tools.mode.clone(),
						tools: allowed_tools.tools.iter().map(|tool| tool.function.name.clone()).collect(),
					},
					_ => ToolChoice::Auto,
				},
			}
		};

		Ok(crate::ChatRequestIR {
			model: model.clone(),
			messages,
			tools,
			tool_choice,
			sampling: Sampling {
				temperature: req.temperature,
				top_p: req.top_p,
				top_k: None,
				max_tokens: req.max_completion_tokens.or(req.max_tokens),
				stop: match req.stop {
					Some(OpenAIStop::Single(s)) => vec![s],
					Some(OpenAIStop::Many(v)) => v,
					None => Vec::new(),
				},
				presence_penalty: if req.presence_penalty.is_some() && req.presence_penalty != Some(0.0) {
					req.presence_penalty
				} else {
					None
				},
				frequency_penalty: if req.frequency_penalty.is_some() && req.frequency_penalty != Some(0.0) {
					req.frequency_penalty
				} else {
					None
				},
				parallel_tool_calls: req.parallel_tool_calls,
				seed: req.seed,
				logit_bias: req.logit_bias,
				logprobs: req.logprobs,
				top_logprobs: req.top_logprobs,
			},
			stream: req.stream.unwrap_or(false),
			response_format: req.response_format.map(|format| match format {
				OpenAIResponseFormat::Simple { r#type } if r#type == "json_object" => ResponseFormat::JsonObject,
				OpenAIResponseFormat::JsonSchema { json_schema, .. } => ResponseFormat::JsonSchema {
					name: json_schema.name,
					description: json_schema.description,
					schema: json_schema.schema,
					strict: json_schema.strict,
				},
				_ => ResponseFormat::Text,
			}),
			audio_output: req.audio.map(|audio| AudioOutput {
				voice: audio.voice.map(|v| format!("{:?}", v).to_lowercase()),
				format: audio.format.map(|f| format!("{:?}", f).to_lowercase()),
			}),
			web_search_options: req.web_search_options.map(|wso| WebSearchOptions {
				user_location: wso.user_location.and_then(|ul| ul.approximate).map(|loc| UserLocation {
					country: loc.country,
					region: loc.region,
					city: loc.city,
					timezone: loc.timezone,
				}),
				search_context_size: wso.search_context_size,
			}),
			prediction: req.prediction.and_then(|p| {
				if p.r#type == Some("content".to_string()) {
					p.content.map(|content| PredictionConfig {
						content: Some(match content {
							serde_json::Value::String(text) => PredictionContent::Text(text),
							value => PredictionContent::Text(value.to_string()),
						}),
					})
				} else {
					None
				}
			}),
			reasoning: req.reasoning_effort.map(|effort| ReasoningConfig {
				effort: Some(format!("{:?}", effort).to_lowercase()),
				budget_tokens: None,
				summary: None,
			}),
			metadata,
			request_timeout: None,
			cache_key: req.prompt_cache_key,
			anthropic_cache,
			safety_identifier: req.safety_identifier,
			openai_chat_request: Some(Box::new(original_request)),
			openai_responses_request: None,
			anthropic_messages_request: None,
			provider_routing: None,
			extra_body: serde_json::Map::new(),
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-chat"
	}
}

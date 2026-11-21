//! IR stream events → Responses stream events.

use super::tools::ClientToolNames;
use super::*;


/// What the Responses stream renderer needs from the request besides the events.
pub(super) struct StreamContext {
	pub(super) request_id: String,
	pub(super) model_alias: String,
	pub(super) message_id: String,
	pub(super) created_at: i64,
	pub(super) reasoning_summary_requested: bool,
	pub(super) reasoning_encrypted_requested: bool,
	pub(super) client_tools: ClientToolNames,
}

/// Renders the IR stream as Responses events; a relayed Responses stream passes through as its raw events.
pub(super) fn sse_stream(
	stream: Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>,
	context: StreamContext,
) -> impl futures_util::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> {
	let StreamContext {
		request_id,
		model_alias,
		message_id,
		created_at,
		reasoning_summary_requested,
		reasoning_encrypted_requested,
		client_tools,
	} = context;
	async_stream::stream! {
			let mut stream = stream;
			let mut raw_events_seen = false;
			let mut sequence_number = 0i64;
			let mut output_index = 0i64;
			let mut terminated = false;

			let mut message_index: Option<i64> = None;
			let mut content_part_added = false;
			let mut text = String::new();
			let mut text_logprobs: Vec<Logprob> = Vec::new();
			let mut annotations: Vec<serde_json::Value> = Vec::new();

			let mut refusal_part_added = false;
			let mut refusal_content_index = 0;
			let mut refusal = String::new();

			let mut reasoning_output: Vec<(i64, serde_json::Value)> = Vec::new();
			let mut reasoning_item_id: Option<String> = None;
			let mut reasoning_index = 0i64;
			let mut reasoning_text = String::new();
			let mut reasoning_part_added = false;

			let mut tool_order: Vec<String> = Vec::new();
			let mut tools: BTreeMap<String, ResponsesToolBuild> = BTreeMap::new();

			let mut input_tokens = 0u32;
			let mut output_tokens = 0u32;
			let mut cached_tokens = 0i64;
			let mut reasoning_tokens = 0i64;
			let mut finish_reason = None::<crate::stream::FinishReason>;

			macro_rules! ensure_created {
				() => {
					if sequence_number == 0 {
						yield responses_sse_event(
							"response.created",
							serde_json::json!({
								"type": "response.created",
								"sequence_number": sequence_number,
								"response": {
									"id": request_id,
									"object": "response",
									"created_at": created_at,
									"status": "in_progress",
									"model": model_alias,
									"output": [],
								},
							}),
							&mut sequence_number,
						);
					}
				};
			}

			macro_rules! open_message_item {
				() => {
					if message_index.is_none() {
						message_index = Some(output_index);
						output_index += 1;
						yield responses_sse_event(
							"response.output_item.added",
							serde_json::json!({
								"type": "response.output_item.added",
								"sequence_number": sequence_number,
								"output_index": message_index.unwrap(),
								"item": {"id": message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
							}),
							&mut sequence_number,
						);
					}
				};
			}

			macro_rules! open_function_call {
				($id:ident, $name:expr) => {
					let item_id = format!("fc_{}", Uuid::new_v4().to_string().replace('-', ""));
					tool_order.push($id.clone());
					tools.insert(
						$id.clone(),
						ResponsesToolBuild {
							output_index,
							item_id,
							name: $name,
							arguments: String::new(),
							done: false,
						},
					);
					output_index += 1;
					let tool = tools.get(&$id).unwrap();
					yield responses_sse_event(
						"response.output_item.added",
						serde_json::json!({
							"type": "response.output_item.added",
							"sequence_number": sequence_number,
							"output_index": tool.output_index,
							"item": client_tools.open_item(&tool.item_id, &$id, &tool.name),
						}),
						&mut sequence_number,
					);
				};
			}

			macro_rules! close_reasoning_summary {
				($item_id:ident, $index:ident, $text:ident) => {
					yield responses_sse_event(
						"response.reasoning_summary_text.done",
						serde_json::json!({
							"type": "response.reasoning_summary_text.done",
							"sequence_number": sequence_number,
							"item_id": $item_id,
							"output_index": $index,
							"summary_index": 0,
							"text": $text,
						}),
						&mut sequence_number,
					);
					yield responses_sse_event(
						"response.reasoning_summary_part.done",
						serde_json::json!({
							"type": "response.reasoning_summary_part.done",
							"sequence_number": sequence_number,
							"item_id": $item_id,
							"output_index": $index,
							"summary_index": 0,
							"part": {"type": "summary_text", "text": $text},
						}),
						&mut sequence_number,
					);
				};
			}

			while let Some(ev) = stream.next().await {
				if let StreamEvent::OpenAIResponsesEvent { event, data } = ev {
					if let Some(error) = crate::skins::openai::errors::upstream_payload_error(&data) {
						let number = data.get("sequence_number").and_then(serde_json::Value::as_i64).unwrap_or(sequence_number);
						yield Ok::<_, std::convert::Infallible>(crate::skins::openai::errors::responses_stream_error_event(&error, number));
						break;
					}
					raw_events_seen = true;
					if let Some(number) = data.get("sequence_number").and_then(serde_json::Value::as_i64) {
						sequence_number = number + 1;
					}
					let mut response_event = axum::response::sse::Event::default().data(serde_json::to_string(&data).unwrap());
					if let Some(event) = event.or_else(|| data.get("type").and_then(serde_json::Value::as_str).map(str::to_string)) {
						response_event = response_event.event(event);
					}
					yield Ok::<_, std::convert::Infallible>(response_event);
					continue;
				}

				if raw_events_seen {
					match ev {
						StreamEvent::Error { code, message } => {
							let error = crate::adapters::InferenceError::from_stream_error(code, message);
							yield Ok::<_, std::convert::Infallible>(crate::skins::openai::errors::responses_stream_error_event(&error, sequence_number));
							terminated = true;
							break;
						}
						StreamEvent::Done => {
							terminated = true;
							break;
						}
						_ => continue,
					}
				}

				match ev {
					StreamEvent::TextDelta { content } | StreamEvent::FinalMessage { content, .. } => {
						if content.is_empty() {
							continue;
						}
						ensure_created!();
						open_message_item!();
						if !content_part_added {
							content_part_added = true;
							yield responses_sse_event(
								"response.content_part.added",
								serde_json::json!({
									"type": "response.content_part.added",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": message_index.unwrap(),
									"content_index": 0,
									"part": {"type": "output_text", "text": "", "annotations": []},
								}),
								&mut sequence_number,
							);
						}
						text.push_str(&content);
						yield responses_sse_event(
							"response.output_text.delta",
							serde_json::json!({
								"type": "response.output_text.delta",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": message_index.unwrap(),
								"content_index": 0,
								"delta": content,
							}),
							&mut sequence_number,
						);
					}
					StreamEvent::RefusalDelta { content } => {
						if content.is_empty() {
							continue;
						}
						ensure_created!();
						open_message_item!();
						if !refusal_part_added {
							refusal_part_added = true;
							refusal_content_index = if content_part_added { 1 } else { 0 };
							yield responses_sse_event(
								"response.content_part.added",
								serde_json::json!({
									"type": "response.content_part.added",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": message_index.unwrap(),
									"content_index": refusal_content_index,
									"part": {"type": "refusal", "refusal": ""},
								}),
								&mut sequence_number,
							);
						}
						refusal.push_str(&content);
						yield responses_sse_event(
							"response.refusal.delta",
							serde_json::json!({
								"type": "response.refusal.delta",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": message_index.unwrap(),
								"content_index": refusal_content_index,
								"delta": content,
							}),
							&mut sequence_number,
						);
					}
					StreamEvent::Logprobs { logprobs } => {
						text_logprobs.extend(responses_text_logprobs(&logprobs));
					}
					StreamEvent::Annotation { annotation } => annotations.push(annotation),
					StreamEvent::ReasoningDelta { content } if reasoning_summary_requested => {
						ensure_created!();
						if reasoning_item_id.is_none() {
							let item_id = format!("rs_{}", Uuid::new_v4().to_string().replace('-', ""));
							reasoning_item_id = Some(item_id.clone());
							reasoning_index = output_index;
							output_index += 1;
							reasoning_text.clear();
							yield responses_sse_event(
								"response.output_item.added",
								serde_json::json!({
									"type": "response.output_item.added",
									"sequence_number": sequence_number,
									"output_index": reasoning_index,
									"item": {"id": item_id, "type": "reasoning", "summary": []},
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.reasoning_summary_part.added",
								serde_json::json!({
									"type": "response.reasoning_summary_part.added",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": reasoning_index,
									"summary_index": 0,
									"part": {"type": "summary_text", "text": ""},
								}),
								&mut sequence_number,
							);
							reasoning_part_added = true;
						}
						if !content.is_empty() {
							reasoning_text.push_str(&content);
							let item_id = reasoning_item_id.clone().unwrap();
							yield responses_sse_event(
								"response.reasoning_summary_text.delta",
								serde_json::json!({
									"type": "response.reasoning_summary_text.delta",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": reasoning_index,
									"summary_index": 0,
									"delta": content,
								}),
								&mut sequence_number,
							);
						}
					}
					StreamEvent::ReasoningDelta { .. } => {}
					StreamEvent::ReasoningBlock { block } => {
						if let Some(item_id) = reasoning_item_id.take() {
							let index = reasoning_index;
							let effective_text = if block.text.is_empty() { reasoning_text.clone() } else { block.text.clone() };
							let mut effective = block.clone();
							effective.text = effective_text.clone();
							let item = responses_reasoning_output_item(&item_id, &effective, reasoning_summary_requested, reasoning_encrypted_requested).unwrap_or_else(|| serde_json::json!({"id": item_id, "type": "reasoning", "summary": []}));
							if reasoning_part_added {
								close_reasoning_summary!(item_id, index, effective_text);
							}
							yield responses_sse_event(
								"response.output_item.done",
								serde_json::json!({
									"type": "response.output_item.done",
									"sequence_number": sequence_number,
									"output_index": index,
									"item": item,
								}),
								&mut sequence_number,
							);
							reasoning_output.push((index, item));
							reasoning_text.clear();
							reasoning_part_added = false;
							continue;
						}
						let item_id = block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().to_string().replace('-', "")));
						let Some(item) = responses_reasoning_output_item(
							&item_id,
							&block,
							reasoning_summary_requested,
							reasoning_encrypted_requested,
						) else {
							continue;
						};
						ensure_created!();
						let index = output_index;
						output_index += 1;
						yield responses_sse_event(
							"response.output_item.added",
							serde_json::json!({
								"type": "response.output_item.added",
								"sequence_number": sequence_number,
								"output_index": index,
								"item": item,
							}),
							&mut sequence_number,
						);
						if reasoning_summary_requested && !block.text.is_empty() {
							yield responses_sse_event(
								"response.reasoning_summary_part.added",
								serde_json::json!({
									"type": "response.reasoning_summary_part.added",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": index,
									"summary_index": 0,
									"part": {"type": "summary_text", "text": ""},
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.reasoning_summary_text.delta",
								serde_json::json!({
									"type": "response.reasoning_summary_text.delta",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": index,
									"summary_index": 0,
									"delta": block.text,
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.reasoning_summary_text.done",
								serde_json::json!({
									"type": "response.reasoning_summary_text.done",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": index,
									"summary_index": 0,
									"text": block.text,
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.reasoning_summary_part.done",
								serde_json::json!({
									"type": "response.reasoning_summary_part.done",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": index,
									"summary_index": 0,
									"part": {"type": "summary_text", "text": block.text},
								}),
								&mut sequence_number,
							);
						}
						yield responses_sse_event(
							"response.output_item.done",
							serde_json::json!({
								"type": "response.output_item.done",
								"sequence_number": sequence_number,
								"output_index": index,
								"item": item,
							}),
							&mut sequence_number,
						);
						reasoning_output.push((index, item));
					}
					StreamEvent::ToolCallStart { id, name, .. } if !tools.contains_key(&id) => {
						ensure_created!();
						open_function_call!(id, name.clone());
					}
					StreamEvent::ToolCallDelta { id, args_delta_json } => {
						let delta = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
						if delta.is_empty() {
							continue;
						}
						ensure_created!();
						if !tools.contains_key(&id) {
							open_function_call!(id, String::new());
						}
						let tool = tools.get_mut(&id).unwrap();
						tool.arguments.push_str(&delta);
						if client_tools.is_custom(&tool.name) {
							continue;
						}
						yield responses_sse_event(
							"response.function_call_arguments.delta",
							serde_json::json!({
								"type": "response.function_call_arguments.delta",
								"sequence_number": sequence_number,
								"item_id": tool.item_id,
								"output_index": tool.output_index,
								"delta": delta,
							}),
							&mut sequence_number,
						);
					}
					StreamEvent::ToolCallEnd { id, args_json } => {
						if let Some(tool) = tools.get_mut(&id) {
							tool.arguments = args_json.as_str().map(str::to_string).unwrap_or_else(|| args_json.to_string());
							tool.done = true;
							for (event, mut data) in client_tools.arguments_done_events(&tool.item_id, tool.output_index, &tool.name, &tool.arguments) {
								data["sequence_number"] = serde_json::json!(sequence_number);
								yield responses_sse_event(event, data, &mut sequence_number);
							}
							yield responses_sse_event(
								"response.output_item.done",
								serde_json::json!({
									"type": "response.output_item.done",
									"sequence_number": sequence_number,
									"output_index": tool.output_index,
									"item": client_tools.done_item(&tool.item_id, &id, &tool.name, &tool.arguments),
								}),
								&mut sequence_number,
							);
						}
					}
					StreamEvent::Tokens { input, output } => {
						input_tokens = input;
						output_tokens = output;
					}
					StreamEvent::OpenAIMetadata {
						prompt_tokens_details,
						completion_tokens_details,
						..
					} => {
						cached_tokens = prompt_tokens_details.map(|details| details.cached_tokens).unwrap_or_default() as i64;
						reasoning_tokens = completion_tokens_details.map(|details| details.reasoning_tokens).unwrap_or_default() as i64;
					}
					StreamEvent::Finish { reason } => {
						finish_reason = Some(reason);
					}
					StreamEvent::Done => {
						break;
					}
					StreamEvent::Error { code, message } => {
						tracing::error!(%code, %message, "Stream error");
						let error = crate::adapters::InferenceError::from_stream_error(code, message);
						yield Ok::<_, std::convert::Infallible>(crate::skins::openai::errors::responses_stream_error_event(&error, sequence_number));
						terminated = true;
						break;
					}
					_ => {}
				}
			}

			if !terminated {
				ensure_created!();
				let truncated = matches!(finish_reason, Some(crate::stream::FinishReason::Length));
				let filtered = matches!(finish_reason, Some(crate::stream::FinishReason::ContentFilter));
				let incomplete = truncated || filtered;
				let incomplete_reason = if filtered { "content_filter" } else { "max_output_tokens" };
				let message_status = if incomplete { "incomplete" } else { "completed" };


				if let Some(item_id) = reasoning_item_id.take() {
					let item = serde_json::json!({"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning_text}]});
					if reasoning_part_added {
						close_reasoning_summary!(item_id, reasoning_index, reasoning_text);
					}
					yield responses_sse_event(
						"response.output_item.done",
						serde_json::json!({
							"type": "response.output_item.done",
							"sequence_number": sequence_number,
							"output_index": reasoning_index,
							"item": item,
						}),
						&mut sequence_number,
					);
					reasoning_output.push((reasoning_index, item));
				}
				let mut text_part = None::<serde_json::Value>;
				if message_index.is_some() && content_part_added {
					let mut part = serde_json::json!({"type": "output_text", "text": text, "annotations": annotations});
					if !text_logprobs.is_empty() {
						part["logprobs"] = serde_json::to_value(&text_logprobs).unwrap_or(serde_json::Value::Null);
					}
					text_part = Some(part);
				}
				let refusal_part: Option<serde_json::Value> = (message_index.is_some() && refusal_part_added)
					.then(|| serde_json::json!({"type": "refusal", "refusal": refusal}));
				let message_content: Vec<serde_json::Value> = text_part.clone().into_iter().chain(refusal_part.clone()).collect();

				if let Some(index) = message_index {
					if let Some(part) = text_part {
						for (annotation_index, annotation) in annotations.iter().enumerate() {
							yield responses_sse_event(
								"response.output_text.annotation.added",
								serde_json::json!({
									"type": "response.output_text.annotation.added",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": index,
									"content_index": 0,
									"annotation_index": annotation_index,
									"annotation": annotation,
								}),
								&mut sequence_number,
							);
						}
						yield responses_sse_event(
							"response.output_text.done",
							serde_json::json!({
								"type": "response.output_text.done",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": index,
								"content_index": 0,
								"text": text,
							}),
							&mut sequence_number,
						);
						yield responses_sse_event(
							"response.content_part.done",
							serde_json::json!({
								"type": "response.content_part.done",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": index,
								"content_index": 0,
								"part": part,
							}),
							&mut sequence_number,
						);
					}
					if let Some(part) = refusal_part {
						yield responses_sse_event(
							"response.refusal.done",
							serde_json::json!({
								"type": "response.refusal.done",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": index,
								"content_index": refusal_content_index,
								"refusal": refusal,
							}),
							&mut sequence_number,
						);
						yield responses_sse_event(
							"response.content_part.done",
							serde_json::json!({
								"type": "response.content_part.done",
								"sequence_number": sequence_number,
								"item_id": message_id,
								"output_index": index,
								"content_index": refusal_content_index,
								"part": part,
							}),
							&mut sequence_number,
						);
					}
					yield responses_sse_event(
						"response.output_item.done",
						serde_json::json!({
							"type": "response.output_item.done",
							"sequence_number": sequence_number,
							"output_index": index,
							"item": {"id": message_id, "type": "message", "status": message_status, "role": "assistant", "content": message_content},
						}),
						&mut sequence_number,
					);
				}

				for id in &tool_order {
					let tool = tools.get(id).unwrap();
					if tool.done {
						continue;
					}
					for (event, mut data) in client_tools.arguments_done_events(&tool.item_id, tool.output_index, &tool.name, &tool.arguments) {
						data["sequence_number"] = serde_json::json!(sequence_number);
						yield responses_sse_event(event, data, &mut sequence_number);
					}
					yield responses_sse_event(
						"response.output_item.done",
						serde_json::json!({
							"type": "response.output_item.done",
							"sequence_number": sequence_number,
							"output_index": tool.output_index,
							"item": client_tools.done_item(&tool.item_id, id, &tool.name, &tool.arguments),
						}),
						&mut sequence_number,
					);
				}

				let mut output: Vec<(i64, serde_json::Value)> = reasoning_output;
				if let Some(index) = message_index {
					output.push((
						index,
						serde_json::json!({"id": message_id, "type": "message", "status": message_status, "role": "assistant", "content": message_content}),
					));
				}
				for id in &tool_order {
					let tool = tools.get(id).unwrap();
					output.push((
						tool.output_index,
						client_tools.done_item(&tool.item_id, id, &tool.name, &tool.arguments),
					));
				}
				output.sort_by_key(|(index, _)| *index);
				let output: Vec<serde_json::Value> = output.into_iter().map(|(_, item)| item).collect();

				let mut terminal = responses_base(&request_id, created_at, &model_alias);
				if incomplete {
					terminal["type"] = serde_json::json!("response.incomplete");
					terminal["status"] = serde_json::json!("incomplete");
					terminal["incomplete_details"] = serde_json::json!({"reason": incomplete_reason});
				} else {
					terminal["type"] = serde_json::json!("response.completed");
					terminal["status"] = serde_json::json!("completed");
				}
				terminal["sequence_number"] = serde_json::json!(sequence_number);
				terminal["response"] = serde_json::json!({
					"id": request_id,
					"object": "response",
					"created_at": created_at,
					"status": if incomplete { "incomplete" } else { "completed" },
					"model": model_alias,
					"output": output,
					"usage": {
						"input_tokens": input_tokens,
						"input_tokens_details": {"cached_tokens": cached_tokens},
						"output_tokens": output_tokens,
						"output_tokens_details": {"reasoning_tokens": reasoning_tokens},
						"total_tokens": input_tokens + output_tokens,
					},
				});
				if incomplete {
					terminal["response"]["incomplete_details"] = serde_json::json!({"reason": incomplete_reason});
				}
				let name = if incomplete { "response.incomplete" } else { "response.completed" };
				yield responses_sse_event(name, terminal, &mut sequence_number);
			}
		}
}

//! IR stream events → Anthropic Messages events, and the non-streamed message folded from them.

use super::*;

pub(super) fn stop_reason(reason: Option<FinishReason>, tool_used: bool) -> &'static str {
	match reason {
		Some(FinishReason::Length) => "max_tokens",
		Some(FinishReason::ContentFilter) => "refusal",
		Some(FinishReason::ToolCalls) => "tool_use",
		Some(FinishReason::Stop) => "end_turn",
		None if tool_used => "tool_use",
		None => "end_turn",
	}
}

pub(super) enum OpenBlock {
	Text,
	Thinking,
}

/// Renders neutral stream events as the Anthropic Messages event protocol.
pub(super) struct MessageRenderer {
	id: String,
	model: String,
	started: bool,
	index: usize,
	open: Option<OpenBlock>,
	/// Tool calls (id, name, arguments) started but not yet ended; each block is rendered at its end so interleaved calls never mix.
	pending_tools: Vec<(String, String, String)>,
	/// A thinking block streamed from deltas has not been matched with its `ReasoningBlock` yet.
	thinking_unmatched: bool,
	tool_used: bool,
	finish: Option<FinishReason>,
	input_tokens: u32,
	output_tokens: u32,
	cached_tokens: u32,
	cache_write_tokens: u32,
}

impl MessageRenderer {
	pub(super) fn new(request_id: &str, model: String) -> Self {
		Self {
			id: format!("msg_{}", request_id.replace('-', "")),
			model,
			started: false,
			index: 0,
			open: None,
			pending_tools: Vec::new(),
			thinking_unmatched: false,
			tool_used: false,
			finish: None,
			input_tokens: 0,
			output_tokens: 0,
			cached_tokens: 0,
			cache_write_tokens: 0,
		}
	}

	fn start(&mut self, out: &mut Vec<(&'static str, Value)>) {
		if !self.started {
			self.started = true;
			out.push((
				"message_start",
				json!({"type": "message_start", "message": {
					"id": self.id,
					"type": "message",
					"role": "assistant",
					"model": self.model,
					"content": [],
					"stop_reason": null,
					"stop_sequence": null,
					"usage": {"input_tokens": 0, "output_tokens": 0}
				}}),
			));
		}
	}

	fn open_block(&mut self, out: &mut Vec<(&'static str, Value)>, block: OpenBlock, content_block: Value) {
		self.close_block(out);
		out.push((
			"content_block_start",
			json!({"type": "content_block_start", "index": self.index, "content_block": content_block}),
		));
		self.open = Some(block);
	}

	fn delta(&self, out: &mut Vec<(&'static str, Value)>, delta: Value) {
		out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": delta})));
	}

	fn close_block(&mut self, out: &mut Vec<(&'static str, Value)>) {
		if self.open.take().is_none() {
			return;
		}
		out.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
		self.index += 1;
	}

	fn text(&mut self, out: &mut Vec<(&'static str, Value)>, text: String) {
		if text.is_empty() {
			return;
		}
		if !matches!(self.open, Some(OpenBlock::Text)) {
			self.open_block(out, OpenBlock::Text, json!({"type": "text", "text": ""}));
		}
		self.delta(out, json!({"type": "text_delta", "text": text}));
	}

	fn thinking(&mut self, out: &mut Vec<(&'static str, Value)>, text: String) {
		if text.is_empty() {
			return;
		}
		if !matches!(self.open, Some(OpenBlock::Thinking)) {
			self.open_block(out, OpenBlock::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}));
			self.thinking_unmatched = true;
		}
		self.delta(out, json!({"type": "thinking_delta", "thinking": text}));
	}

	fn reasoning_block(&mut self, out: &mut Vec<(&'static str, Value)>, block: ReasoningBlock) {
		if block.format == ReasoningBlock::FORMAT_ANTHROPIC && block.kind == ReasoningKind::Redacted {
			self.open_block(out, OpenBlock::Thinking, json!({"type": "redacted_thinking", "data": block.data.unwrap_or_default()}));
			self.close_block(out);
			return;
		}
		let signature = if block.is_unsigned() {
			String::new()
		} else if block.format == ReasoningBlock::FORMAT_ANTHROPIC {
			block.signature.clone().unwrap_or_default()
		} else {
			block.to_envelope().unwrap_or_default()
		};
		let streamed = std::mem::take(&mut self.thinking_unmatched);
		if streamed && matches!(self.open, Some(OpenBlock::Thinking)) {
			if !signature.is_empty() {
				self.delta(out, json!({"type": "signature_delta", "signature": signature}));
			}
			self.close_block(out);
			return;
		}
		// A block whose streamed text was already closed by later output repeats it only when the
		// signature cannot carry it (an Anthropic signature); an `omn1.` envelope holds its own text.
		let text = if streamed && block.format != ReasoningBlock::FORMAT_ANTHROPIC {
			String::new()
		} else {
			block.text
		};
		if text.is_empty() && signature.is_empty() {
			return;
		}
		self.open_block(out, OpenBlock::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}));
		if !text.is_empty() {
			self.delta(out, json!({"type": "thinking_delta", "thinking": text}));
		}
		if !signature.is_empty() {
			self.delta(out, json!({"type": "signature_delta", "signature": signature}));
		}
		self.close_block(out);
	}

	fn tool_block(&mut self, out: &mut Vec<(&'static str, Value)>, id: String, name: String, args: String) {
		self.open_block(out, OpenBlock::Text, json!({"type": "tool_use", "id": id, "name": name, "input": {}}));
		if !args.trim().is_empty() {
			let partial_json = if serde_json::from_str::<Value>(&args).is_ok_and(|input| input.is_object()) {
				args
			} else {
				crate::adapters::common::tools::tool_input_object(&args).to_string()
			};
			self.delta(out, json!({"type": "input_json_delta", "partial_json": partial_json}));
		}
		self.close_block(out);
	}

	fn finish(&mut self, out: &mut Vec<(&'static str, Value)>) {
		self.start(out);
		for (id, name, args) in std::mem::take(&mut self.pending_tools) {
			self.tool_block(out, id, name, args);
		}
		self.close_block(out);
		out.push((
			"message_delta",
			json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(self.finish, self.tool_used), "stop_sequence": null}, "usage": {
				"input_tokens": self.input_tokens.saturating_sub(self.cached_tokens + self.cache_write_tokens),
				"output_tokens": self.output_tokens,
				"cache_read_input_tokens": self.cached_tokens,
				"cache_creation_input_tokens": self.cache_write_tokens
			}}),
		));
		out.push(("message_stop", json!({"type": "message_stop"})));
	}

	/// Returns the protocol events for `event`, and whether the message ended.
	pub(super) fn push(&mut self, event: StreamEvent) -> (Vec<(&'static str, Value)>, bool) {
		let mut out = Vec::new();
		if let StreamEvent::Error { code, message } = event {
			let (_, body) = inference_error_body(&InferenceError::from_stream_error(code, message));
			out.push(("error", body));
			return (out, true);
		}
		self.start(&mut out);
		match event {
			StreamEvent::TextDelta { content } => self.text(&mut out, content),
			StreamEvent::RefusalDelta { content } => {
				self.finish.get_or_insert(FinishReason::ContentFilter);
				self.text(&mut out, content)
			}
			StreamEvent::FinalMessage { content, .. } => self.text(&mut out, content),
			StreamEvent::ReasoningDelta { content } => self.thinking(&mut out, content),
			StreamEvent::ReasoningBlock { block } => self.reasoning_block(&mut out, block),
			StreamEvent::ToolCallStart { id, name, .. } => {
				self.tool_used = true;
				self.pending_tools.push((id, name, String::new()));
			}
			StreamEvent::ToolCallDelta { id, args_delta_json } => {
				if let Some((_, _, args)) = self.pending_tools.iter_mut().find(|tool| tool.0 == id) {
					args.push_str(args_delta_json.as_str().map_or_else(|| args_delta_json.to_string(), str::to_string).as_str());
				}
			}
			StreamEvent::ToolCallEnd { id, args_json } => {
				if let Some(position) = self.pending_tools.iter().position(|tool| tool.0 == id) {
					let (id, name, _) = self.pending_tools.remove(position);
					self.tool_block(&mut out, id, name, args_json.as_str().map_or_else(|| args_json.to_string(), str::to_string));
				}
			}
			StreamEvent::Tokens { input, output } => {
				self.input_tokens = input;
				self.output_tokens = output;
			}
			StreamEvent::OpenAIMetadata {
				prompt_tokens_details: Some(details),
				..
			} => {
				self.cached_tokens = details.cached_tokens;
				self.cache_write_tokens = details.cache_write_tokens;
			}
			StreamEvent::Finish { reason } => self.finish = Some(reason),
			StreamEvent::Done => {
				self.finish(&mut out);
				return (out, true);
			}
			_ => {}
		}
		(out, false)
	}
}

/// Folds rendered protocol events into a non-streaming Messages response.
pub(super) fn assemble_message(events: Vec<(&'static str, Value)>) -> Value {
	let mut message = json!({});
	let mut partial_json: BTreeMap<usize, String> = BTreeMap::new();
	for (name, data) in events {
		match name {
			"message_start" => message = data["message"].clone(),
			"content_block_start" => {
				if let Some(content) = message["content"].as_array_mut() {
					content.push(data["content_block"].clone());
				}
			}
			"content_block_delta" => {
				let index = data["index"].as_u64().unwrap_or_default() as usize;
				let block = &mut message["content"][index];
				let delta = &data["delta"];
				match block_type(delta) {
					"text_delta" => {
						block["text"] = json!(format!(
							"{}{}",
							block["text"].as_str().unwrap_or_default(),
							delta["text"].as_str().unwrap_or_default()
						))
					}
					"thinking_delta" => {
						block["thinking"] = json!(format!(
							"{}{}",
							block["thinking"].as_str().unwrap_or_default(),
							delta["thinking"].as_str().unwrap_or_default()
						))
					}
					"signature_delta" => block["signature"] = delta["signature"].clone(),
					"input_json_delta" => partial_json.entry(index).or_default().push_str(delta["partial_json"].as_str().unwrap_or_default()),
					_ => {}
				}
			}
			"content_block_stop" => {
				let index = data["index"].as_u64().unwrap_or_default() as usize;
				if let Some(args) = partial_json.remove(&index) {
					message["content"][index]["input"] = crate::adapters::common::tools::tool_input_object(&args);
				}
			}
			"message_delta" => {
				message["stop_reason"] = data["delta"]["stop_reason"].clone();
				message["stop_sequence"] = data["delta"]["stop_sequence"].clone();
				message["usage"] = data["usage"].clone();
			}
			_ => {}
		}
	}
	message
}

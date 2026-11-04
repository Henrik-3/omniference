use std::borrow::Cow;

/// Represents a parsed SSE event
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
	/// The event type (from `event:` line), if present
	pub event_type: Option<String>,
	/// The data payload (from `data:` lines, joined with `\n`)
	pub data: String,
}

const BOM: &[u8] = b"\xEF\xBB\xBF";

/// Incremental Server-Sent Events parser following the WHATWG event-stream grammar.
///
/// Input is raw bytes so that transport chunk boundaries may fall anywhere, including inside a
/// multi-byte UTF-8 sequence. Lines end with `\r\n`, `\n`, or `\r`; a blank line dispatches the
/// pending event. Events without `data:` lines are not emitted.
pub struct SseParser {
	buffer: Vec<u8>,
	bom_checked: bool,
	event_type: Option<String>,
	data: Vec<String>,
}

impl SseParser {
	pub fn new() -> Self {
		Self {
			buffer: Vec::new(),
			bom_checked: false,
			event_type: None,
			data: Vec::new(),
		}
	}

	/// Feed a chunk of bytes and return every event completed by it.
	pub fn feed(&mut self, chunk: impl AsRef<[u8]>) -> Vec<SseEvent> {
		self.buffer.extend_from_slice(chunk.as_ref());

		if !self.bom_checked {
			if self.buffer.len() < BOM.len() && BOM.starts_with(&self.buffer) {
				return Vec::new();
			}
			if self.buffer.starts_with(BOM) {
				self.buffer.drain(..BOM.len());
			}
			self.bom_checked = true;
		}

		let mut events = Vec::new();
		let mut start = 0;
		while let Some(offset) = self.buffer[start..].iter().position(|byte| matches!(byte, b'\n' | b'\r')) {
			let end = start + offset;
			let terminator_len = match (self.buffer[end], self.buffer.get(end + 1)) {
				(b'\r', Some(b'\n')) => 2,
				(b'\r', None) => break,
				_ => 1,
			};
			let line = String::from_utf8_lossy(&self.buffer[start..end]).into_owned();
			start = end + terminator_len;
			if let Some(event) = self.process_line(&line) {
				events.push(event);
			}
		}
		self.buffer.drain(..start);

		events
	}

	fn process_line(&mut self, line: &str) -> Option<SseEvent> {
		if line.is_empty() {
			let event_type = self.event_type.take();
			if self.data.is_empty() {
				return None;
			}
			let data = std::mem::take(&mut self.data).join("\n");
			return Some(SseEvent { event_type, data });
		}
		if line.starts_with(':') {
			return None;
		}

		let (field, value) = match line.split_once(':') {
			Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
			None => (line, ""),
		};
		match field {
			"event" => self.event_type = Some(value.to_string()),
			"data" => self.data.push(value.to_string()),
			_ => {}
		}
		None
	}

	/// Whether a partial line or an undispatched event is pending.
	pub fn has_remaining(&self) -> bool {
		!self.buffer.is_empty() || !self.data.is_empty()
	}

	/// Pending partial-line bytes, decoded lossily (useful for debugging).
	pub fn remaining(&self) -> Cow<'_, str> {
		String::from_utf8_lossy(&self.buffer)
	}
}

impl Default for SseParser {
	fn default() -> Self {
		Self::new()
	}
}

/// SSE (Server-Sent Events) parser that properly buffers incomplete chunks.
///
/// HTTP chunks don't guarantee alignment with SSE event boundaries. This parser
/// buffers incoming data and only yields complete events (those ending with `\n\n`).
/// Represents a parsed SSE event
#[derive(Debug, Clone)]
pub struct SseEvent {
	/// The event type (from `event:` line), if present
	pub event_type: Option<String>,
	/// The data payload (from `data:` line)
	pub data: String,
}

/// An SSE parser that buffers incomplete chunks and yields complete events.
pub struct SseParser {
	buffer: String,
}

impl SseParser {
	pub fn new() -> Self {
		Self { buffer: String::new() }
	}

	/// Feed a chunk of data into the parser and return any complete events.
	///
	/// This properly handles:
	/// - Events split across multiple chunks
	/// - Multiple events in a single chunk
	/// - Partial events that need to wait for more data
	pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
		self.buffer.push_str(chunk);

		let mut events = Vec::new();

		// SSE events are separated by double newlines
		// We need to handle both \n\n and \r\n\r\n
		loop {
			// Find the next complete event (ends with \n\n or \r\n\r\n)
			let unix = self.buffer.find("\n\n").map(|pos| (pos, 2));
			let windows = self.buffer.find("\r\n\r\n").map(|pos| (pos, 4));
			let split_pos = match (unix, windows) {
				(Some(unix), Some(windows)) => Some(if unix.0 < windows.0 { unix } else { windows }),
				(Some(delimiter), None) | (None, Some(delimiter)) => Some(delimiter),
				(None, None) => None,
			};

			match split_pos {
				Some((pos, delim_len)) => {
					// Extract the complete event
					let event_str: String = self.buffer.drain(..pos + delim_len).collect();

					// Parse the event
					if let Some(event) = Self::parse_event(&event_str) {
						events.push(event);
					}
				}
				None => {
					// No complete event yet, wait for more data
					break;
				}
			}
		}

		events
	}

	/// Parse a single SSE event string into an SseEvent
	fn parse_event(event_str: &str) -> Option<SseEvent> {
		let mut event_type = None;
		let mut data_parts = Vec::new();

		for line in event_str.lines() {
			let line = line.trim();
			if line.is_empty() {
				continue;
			}

			if let Some(value) = line.strip_prefix("event:") {
				event_type = Some(value.trim().to_string());
			} else if let Some(value) = line.strip_prefix("data:") {
				data_parts.push(value.trim().to_string());
			}
			// Ignore other fields like id:, retry:, comments (:)
		}

		if data_parts.is_empty() {
			return None;
		}

		// Join multiple data lines with newlines (per SSE spec)
		let data = data_parts.join("\n");

		Some(SseEvent { event_type, data })
	}

	/// Check if there's any remaining buffered data
	pub fn has_remaining(&self) -> bool {
		!self.buffer.is_empty()
	}

	/// Get remaining buffer content (useful for debugging)
	#[allow(dead_code)]
	pub fn remaining(&self) -> &str {
		&self.buffer
	}
}

impl Default for SseParser {
	fn default() -> Self {
		Self::new()
	}
}

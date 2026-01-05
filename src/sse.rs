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
        Self {
            buffer: String::new(),
        }
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
            let split_pos = if let Some(pos) = self.buffer.find("\n\n") {
                Some((pos, 2)) // Unix-style
            } else if let Some(pos) = self.buffer.find("\r\n\r\n") {
                Some((pos, 4)) // Windows-style
            } else {
                None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: message\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type.as_deref(), Some("message"));
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn test_split_event() {
        let mut parser = SseParser::new();

        // First chunk - incomplete
        let events = parser.feed("event: message\ndata: hel");
        assert_eq!(events.len(), 0);

        // Second chunk - completes the event
        let events = parser.feed("lo world\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello world");
    }

    #[test]
    fn test_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn test_json_split_midway() {
        // This simulates the exact issue: JSON cut off mid-value
        let mut parser = SseParser::new();

        let events = parser.feed(r#"event: response.created
data: {"type":"response.created","instructions":nul"#);
        assert_eq!(events.len(), 0);

        // Complete the JSON
        let events = parser.feed(r#"l,"status":"ok"}

"#);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].data,
            r#"{"type":"response.created","instructions":null,"status":"ok"}"#
        );
    }

    #[test]
    fn test_done_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
    }
}

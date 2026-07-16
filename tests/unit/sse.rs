use omniference::sse::SseParser;

#[test]
fn parses_complete_event() {
	let mut parser = SseParser::new();
	let events = parser.feed("event: message\ndata: hello\n\n");
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].event_type.as_deref(), Some("message"));
	assert_eq!(events[0].data, "hello");
}

#[test]
fn buffers_split_events() {
	let mut parser = SseParser::new();
	assert!(parser.feed("event: message\ndata: hel").is_empty());
	let events = parser.feed("lo world\n\n");
	assert_eq!(events[0].data, "hello world");
}

#[test]
fn parses_multiple_events_in_one_chunk() {
	let mut parser = SseParser::new();
	let events = parser.feed("data: first\n\ndata: second\n\n");
	assert_eq!(events.len(), 2);
	assert_eq!(events[0].data, "first");
	assert_eq!(events[1].data, "second");
}

#[test]
fn buffers_json_split_mid_value() {
	let mut parser = SseParser::new();
	assert!(
		parser
			.feed("event: response.created\ndata: {\"type\":\"response.created\",\"instructions\":nul")
			.is_empty()
	);
	let events = parser.feed("l,\"status\":\"ok\"}\n\n");
	assert_eq!(events[0].data, "{\"type\":\"response.created\",\"instructions\":null,\"status\":\"ok\"}");
}

#[test]
fn parses_done_event() {
	let mut parser = SseParser::new();
	let events = parser.feed("data: [DONE]\n\n");
	assert_eq!(events[0].data, "[DONE]");
}

#[test]
fn parses_crlf_event() {
	let mut parser = SseParser::new();
	let events = parser.feed("event: message\r\ndata: hello\r\n\r\n");
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].event_type.as_deref(), Some("message"));
	assert_eq!(events[0].data, "hello");
}

#[test]
fn parses_mixed_event_delimiters_in_order() {
	let mut parser = SseParser::new();
	let events = parser.feed("data: first\r\n\r\ndata: second\n\n");
	assert_eq!(events.len(), 2);
	assert_eq!(events[0].data, "first");
	assert_eq!(events[1].data, "second");
}

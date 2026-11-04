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

#[test]
fn parses_lone_carriage_return_line_endings() {
	let mut parser = SseParser::new();
	let events = parser.feed("event: message\rdata: hello\r\rdata: next\r\r\n");
	assert_eq!(events.len(), 2);
	assert_eq!(events[0].event_type.as_deref(), Some("message"));
	assert_eq!(events[0].data, "hello");
	assert_eq!(events[1].event_type, None);
	assert_eq!(events[1].data, "next");
}

#[test]
fn waits_for_line_feed_after_trailing_carriage_return() {
	let mut parser = SseParser::new();
	assert!(parser.feed("data: hello\r\n\r").is_empty());
	let events = parser.feed("\ndata: next\n\n");
	assert_eq!(events.iter().map(|event| event.data.as_str()).collect::<Vec<_>>(), ["hello", "next"]);
}

#[test]
fn joins_multi_line_data_and_ignores_comments_and_unknown_fields() {
	let mut parser = SseParser::new();
	let events = parser.feed(": keep-alive\nid: 7\nretry: 100\ndata: first\ndata:second\ndata\n\n");
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].data, "first\nsecond\n");
}

#[test]
fn strips_only_one_leading_space_from_values() {
	let mut parser = SseParser::new();
	let events = parser.feed("data:  padded \n\n");
	assert_eq!(events[0].data, " padded ");
}

#[test]
fn event_without_data_resets_event_type() {
	let mut parser = SseParser::new();
	let events = parser.feed("event: ping\n\ndata: payload\n\n");
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].event_type, None);
}

#[test]
fn strips_leading_byte_order_mark_split_across_chunks() {
	let mut parser = SseParser::new();
	assert!(parser.feed(b"\xEF".as_slice()).is_empty());
	assert!(parser.feed(b"\xBB".as_slice()).is_empty());
	let events = parser.feed(b"\xBFevent: message\ndata: hello\n\n".as_slice());
	assert_eq!(events[0].event_type.as_deref(), Some("message"));
}

#[test]
fn reassembles_multi_byte_characters_split_across_chunks() {
	let payload = "data: Grüße 👋 世界\n\n".as_bytes();
	for split in 1..payload.len() {
		let mut parser = SseParser::new();
		let mut events = parser.feed(&payload[..split]);
		events.extend(parser.feed(&payload[split..]));
		assert_eq!(events.len(), 1, "split at {split}");
		assert_eq!(events[0].data, "Grüße 👋 世界", "split at {split}");
	}
}

#[test]
fn reports_incomplete_trailing_event() {
	let mut parser = SseParser::new();
	parser.feed("data: complete\n\ndata: partial\n");
	assert!(parser.has_remaining());
	parser.feed("\n");
	assert!(!parser.has_remaining());
}

mod properties {
	use omniference::sse::{SseEvent, SseParser};
	use proptest::prelude::*;

	const LINE_ENDINGS: [&str; 3] = ["\n", "\r\n", "\r"];

	#[derive(Debug, Clone)]
	struct ScriptedEvent {
		event_type: Option<String>,
		data: Vec<String>,
		comment: bool,
		endings: Vec<usize>,
	}

	fn field_value() -> impl Strategy<Value = String> {
		"[^\r\n]{0,24}"
	}

	fn scripted_event() -> impl Strategy<Value = ScriptedEvent> {
		(
			proptest::option::of("[a-z.]{1,16}"),
			proptest::collection::vec(field_value(), 1..4),
			any::<bool>(),
			proptest::collection::vec(0..LINE_ENDINGS.len(), 8),
		)
			.prop_map(|(event_type, data, comment, endings)| ScriptedEvent {
				event_type,
				data,
				comment,
				endings,
			})
	}

	fn encode(events: &[ScriptedEvent], bom: bool) -> Vec<u8> {
		let mut out = String::new();
		if bom {
			out.push('\u{FEFF}');
		}
		for event in events {
			let mut lines = Vec::new();
			if event.comment {
				lines.push(": comment".to_string());
			}
			if let Some(event_type) = &event.event_type {
				lines.push(format!("event: {event_type}"));
			}
			lines.extend(event.data.iter().map(|value| format!("data: {value}")));
			lines.push(String::new());
			for (index, line) in lines.iter().enumerate() {
				let mut ending = LINE_ENDINGS[event.endings[index % event.endings.len()]];
				if line.is_empty() && ending == "\n" && out.ends_with('\r') {
					ending = "\r\n";
				}
				out.push_str(line);
				out.push_str(ending);
			}
		}
		if out.ends_with('\r') {
			out.push('\n');
		}
		out.into_bytes()
	}

	fn expected(events: &[ScriptedEvent]) -> Vec<SseEvent> {
		events
			.iter()
			.map(|event| SseEvent {
				event_type: event.event_type.clone(),
				data: event.data.join("\n"),
			})
			.collect()
	}

	fn feed_split(bytes: &[u8], cuts: &[usize]) -> Vec<SseEvent> {
		let mut cuts: Vec<usize> = cuts.iter().map(|cut| cut % (bytes.len() + 1)).collect();
		cuts.sort_unstable();
		let mut parser = SseParser::new();
		let mut events = Vec::new();
		let mut start = 0;
		for cut in cuts.into_iter().chain(std::iter::once(bytes.len())) {
			events.extend(parser.feed(&bytes[start..cut]));
			start = cut;
		}
		assert!(!parser.has_remaining(), "parser kept {:?}", parser.remaining());
		events
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(512))]

		#[test]
		fn any_chunking_yields_the_encoded_events(
			events in proptest::collection::vec(scripted_event(), 0..6),
			bom in any::<bool>(),
			cuts in proptest::collection::vec(any::<usize>(), 0..24),
		) {
			let bytes = encode(&events, bom);
			prop_assert_eq!(feed_split(&bytes, &cuts), expected(&events));
		}

		#[test]
		fn byte_at_a_time_matches_single_chunk(
			events in proptest::collection::vec(scripted_event(), 0..4),
		) {
			let bytes = encode(&events, false);
			let all_cuts: Vec<usize> = (0..bytes.len()).collect();
			prop_assert_eq!(feed_split(&bytes, &all_cuts), feed_split(&bytes, &[]));
		}
	}
}

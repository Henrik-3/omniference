//! Streaming adapters must end promptly with a `cancelled` error when their token is cancelled,
//! even while the upstream is stalled mid-stream.

use super::stream_rechunking::{Case, cases, request};
use crate::common::upstream::{FakeUpstream, Reply, Step};
use futures_util::StreamExt;
use omniference::stream::StreamEvent;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn first_text_chunk(body: &str) -> String {
	let start = body.find(super::stream_rechunking::PIECES[0]).expect("first piece in body");
	let end = start + body[start..].find("\n\n").expect("event terminator") + 2;
	body[..end].to_string()
}

async fn assert_cancel_ends_stalled_stream(name: &str) {
	let case: Case = cases().into_iter().find(|case| case.name == name).expect("known case");
	let reply = Reply::sse(first_text_chunk(&case.body)).then(Step::Stall);
	let upstream = FakeUpstream::always(reply).await;
	let cancel = CancellationToken::new();
	let mut stream = case
		.adapter
		.execute_chat(request(&case, upstream.base_url()), cancel.clone())
		.await
		.unwrap_or_else(|error| panic!("{name}: execute_chat failed: {error}"));

	let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
		.await
		.unwrap_or_else(|_| panic!("{name}: no first event"));
	assert!(first.is_some(), "{name}: stream ended before first event");
	cancel.cancel();

	let rest: Vec<StreamEvent> = tokio::time::timeout(Duration::from_secs(3), stream.collect())
		.await
		.unwrap_or_else(|_| panic!("{name}: stream did not end after cancellation"));
	match rest.last() {
		Some(StreamEvent::Error { code, .. }) => assert_eq!(code, "cancelled", "{name}: {rest:?}"),
		other => panic!("{name}: expected cancelled error, got {other:?} in {rest:?}"),
	}
}

#[tokio::test]
async fn gemini_cancel_ends_stalled_stream() {
	assert_cancel_ends_stalled_stream("gemini").await;
}

#[tokio::test]
async fn anthropic_cancel_ends_stalled_stream() {
	assert_cancel_ends_stalled_stream("anthropic").await;
}

#[tokio::test]
async fn openai_chat_cancel_ends_stalled_stream() {
	assert_cancel_ends_stalled_stream("openai-chat").await;
}

#[tokio::test]
async fn openai_responses_cancel_ends_stalled_stream() {
	assert_cancel_ends_stalled_stream("openai-responses").await;
}

#[tokio::test]
async fn openrouter_cancel_ends_stalled_stream() {
	assert_cancel_ends_stalled_stream("openrouter").await;
}

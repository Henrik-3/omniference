use crate::common::upstream::{FakeUpstream, Reply, Step};
use crate::unit::stream_rechunking::{cases, request};
use futures_util::StreamExt;
use omniference::{adapter::AdapterError, stream::StreamEvent};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn steady_stream_longer_than_timeout_succeeds() {
	for case in cases() {
		let mut reply = Reply::sse(case.body.clone()).split_at(&[
			case.body.len() / 6,
			case.body.len() / 3,
			case.body.len() / 2,
			(case.body.len() * 2) / 3,
			(case.body.len() * 5) / 6,
		]);
		for step in &mut reply.steps {
			if let Step::Delay(duration) = step {
				*duration = Duration::from_millis(80);
			}
		}
		let upstream = FakeUpstream::always(reply).await;
		let mut ir = request(&case, upstream.base_url());
		ir.model.provider.endpoint.timeout = Some(300);
		let stream = case
			.adapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.unwrap_or_else(|error| panic!("{}: {error}", case.name));
		let events = tokio::time::timeout(Duration::from_secs(10), stream.collect::<Vec<_>>())
			.await
			.unwrap_or_else(|_| panic!("{}: stream hung", case.name));
		assert!(
			!events.iter().any(|event| matches!(event, StreamEvent::Error { .. })),
			"{}: unexpected error in {events:?}",
			case.name
		);
		assert!(matches!(events.last(), Some(StreamEvent::Done)), "{}: did not complete: {events:?}", case.name);
	}
}

#[tokio::test]
async fn stalled_stream_times_out() {
	for case in cases() {
		let prefix = &case.body[..case.body.len().min(24)];
		let reply = Reply::sse(prefix.to_string()).then(Step::Stall);
		let upstream = FakeUpstream::always(reply).await;
		let mut ir = request(&case, upstream.base_url());
		ir.model.provider.endpoint.timeout = Some(300);
		let stream = case
			.adapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.unwrap_or_else(|error| panic!("{}: {error}", case.name));
		let events = tokio::time::timeout(Duration::from_secs(10), stream.collect::<Vec<_>>())
			.await
			.unwrap_or_else(|_| panic!("{}: stalled stream did not time out", case.name));
		assert!(
			events.iter().any(|event| matches!(event, StreamEvent::Error { code, .. } if code == "timeout")),
			"{}: expected timeout error in {events:?}",
			case.name
		);
	}
}

#[tokio::test]
async fn slow_headers_time_out() {
	let case = &cases()[0];
	let body = case.body.clone();
	let upstream = FakeUpstream::start(move |_| {
		std::thread::sleep(Duration::from_millis(800));
		Reply::sse(body.clone())
	})
	.await;
	let mut ir = request(case, upstream.base_url());
	ir.model.provider.endpoint.timeout = Some(300);
	let result = case.adapter.execute_chat(ir, CancellationToken::new()).await;
	assert!(matches!(result, Err(AdapterError::Timeout)), "expected header timeout, got stream start");
}

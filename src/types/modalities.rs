use super::ModelRef;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
	Text(String),
	Texts(Vec<String>),
	Tokens(Vec<u32>),
	TokenBatches(Vec<Vec<u32>>),
}

/// An embeddings request. `raw_request` holds the client's original OpenAI-style body; adapters
/// forward it with only `model` rewritten so provider-specific fields survive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingRequestIR {
	pub model: ModelRef,
	pub input: EmbeddingInput,
	#[serde(default)]
	pub dimensions: Option<u32>,
	#[serde(default)]
	pub encoding_format: Option<String>,
	#[serde(default)]
	pub raw_request: Option<serde_json::Map<String, serde_json::Value>>,
	#[serde(default)]
	pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingVector {
	Float(Vec<f32>),
	Base64(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
	pub index: u32,
	pub embedding: EmbeddingVector,
}

/// `raw` is the provider's JSON response, relayed unchanged to OpenAI-protocol clients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
	pub data: Vec<Embedding>,
	pub usage: crate::catalog::UsageBreakdown,
	#[serde(default)]
	pub provider_cost: Option<f64>,
	pub raw: serde_json::Value,
}

/// A rerank request (Jina/Cohere v2/vLLM shape). `documents` stay JSON because providers accept
/// strings or objects; `raw_request` holds the client's body, forwarded like embeddings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankRequestIR {
	pub model: ModelRef,
	pub query: String,
	pub documents: Vec<serde_json::Value>,
	#[serde(default)]
	pub top_n: Option<u32>,
	#[serde(default)]
	pub return_documents: Option<bool>,
	#[serde(default)]
	pub raw_request: Option<serde_json::Map<String, serde_json::Value>>,
	#[serde(default)]
	pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RerankResult {
	pub index: u32,
	pub relevance_score: f64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub document: Option<serde_json::Value>,
}

/// `raw` is the provider's JSON response, relayed unchanged to OpenAI-protocol clients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankResponse {
	pub results: Vec<RerankResult>,
	pub usage: crate::catalog::UsageBreakdown,
	#[serde(default)]
	pub provider_cost: Option<f64>,
	pub raw: serde_json::Value,
}

/// An uploaded audio file, sent upstream as the multipart `file` part.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AudioInput {
	pub bytes: Vec<u8>,
	/// Providers read the audio format from its extension, so the client's name is kept.
	pub filename: String,
	pub media_type: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionOperation {
	/// `/audio/transcriptions`: text in the spoken language.
	Transcribe,
	/// `/audio/translations`: English text.
	Translate,
}

/// A speech-to-text request. `raw_request` holds the client's multipart text fields (a field sent
/// more than once, such as `timestamp_granularities[]`, as an array); adapters forward them with
/// `model` rewritten and the set typed fields written over them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionRequestIR {
	pub model: ModelRef,
	pub operation: TranscriptionOperation,
	pub audio: AudioInput,
	#[serde(default)]
	pub language: Option<String>,
	#[serde(default)]
	pub prompt: Option<String>,
	#[serde(default)]
	pub response_format: Option<String>,
	#[serde(default)]
	pub timestamp_granularities: Vec<String>,
	/// Relays the provider's SSE transcript events instead of one response.
	#[serde(default)]
	pub stream: bool,
	#[serde(default)]
	pub raw_request: Option<serde_json::Map<String, serde_json::Value>>,
	#[serde(default)]
	pub metadata: BTreeMap<String, String>,
}

/// A text-to-speech request. `raw_request` holds the client's original JSON body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeechRequestIR {
	pub model: ModelRef,
	pub input: String,
	/// A voice name, or an object such as `{"id": "voice_123"}` for custom voices.
	#[serde(default)]
	pub voice: Option<serde_json::Value>,
	#[serde(default)]
	pub instructions: Option<String>,
	#[serde(default)]
	pub response_format: Option<String>,
	#[serde(default)]
	pub speed: Option<f64>,
	/// `audio` (raw bytes) or `sse` (`speech.audio.delta` events).
	#[serde(default)]
	pub stream_format: Option<String>,
	#[serde(default)]
	pub raw_request: Option<serde_json::Map<String, serde_json::Value>>,
	#[serde(default)]
	pub metadata: BTreeMap<String, String>,
}

/// One item of a [`ByteStreamResponse`] body.
#[derive(Debug)]
pub enum ByteStreamEvent {
	/// Body bytes, relayed to the client unchanged.
	Data(bytes::Bytes),
	/// Usage the provider reported so far; a later report replaces an earlier one.
	Usage(crate::catalog::UsageBreakdown),
	/// The upstream failed mid-body; no more events follow.
	Error(crate::adapter::InferenceError),
}

/// A provider response relayed as bytes (audio, a transcript, or SSE), used by speech-to-text and
/// text-to-speech. The body is read lazily, so dropping it closes the upstream connection.
pub struct ByteStreamResponse {
	pub content_type: String,
	/// Cost the provider reported in its response headers.
	pub provider_cost: Option<f64>,
	pub body: futures::stream::BoxStream<'static, ByteStreamEvent>,
}

impl std::fmt::Debug for ByteStreamResponse {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("ByteStreamResponse")
			.field("content_type", &self.content_type)
			.field("provider_cost", &self.provider_cost)
			.finish_non_exhaustive()
	}
}

impl ByteStreamResponse {
	/// Reads the whole body, for callers that do not stream it on.
	pub async fn into_bytes(self) -> Result<bytes::Bytes, crate::adapter::InferenceError> {
		use futures::StreamExt;
		let mut body = self.body;
		let mut bytes = bytes::BytesMut::new();
		while let Some(event) = body.next().await {
			match event {
				ByteStreamEvent::Data(chunk) => bytes.extend_from_slice(&chunk),
				ByteStreamEvent::Usage(_) => {}
				ByteStreamEvent::Error(error) => return Err(error),
			}
		}
		Ok(bytes.freeze())
	}
}

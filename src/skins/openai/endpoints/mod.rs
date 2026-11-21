//! The OpenAI endpoints beyond chat: embeddings, rerank, images, audio and decisions.

pub mod audio;
pub mod decisions;
pub mod media;

pub use audio::{AudioSkinError, OpenAIAudioSkin};
pub use decisions::OpenAIDecisionsSkin;
pub use media::{ImageSkinError, OpenAIEmbeddingsSkin, OpenAIImagesSkin, OpenAIRerankSkin};

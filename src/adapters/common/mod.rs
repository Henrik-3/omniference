//! Helpers shared by several adapters.

pub mod extra_body;
pub mod http;
pub(crate) mod image;
pub mod sse;
pub(crate) mod tools;
pub mod upstream;

/// Whether the request asks for more than one candidate (`n > 1`), which adapters of single-candidate providers refuse.
pub(crate) fn requests_multiple_candidates(metadata: &std::collections::BTreeMap<String, String>) -> bool {
	metadata.get("n").and_then(|n| n.parse::<u32>().ok()).is_some_and(|n| n > 1)
}

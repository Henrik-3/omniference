//! Integration tests for end-to-end functionality
//!
//! These tests verify that components work together correctly.
//! Some tests require live API connections and can be skipped
//! by setting SKIP_LIVE_TESTS=true.

pub mod service;
pub mod server;
pub mod endpoints;

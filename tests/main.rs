//! Main test entry point
//!
//! This file registers all test modules and provides a unified test suite.

// Common utilities shared across all tests
mod common;

// Unit tests for individual components
mod unit;

// Integration tests for component interaction
mod integration;

// Re-export common utilities for use in test modules
pub use common::*;

// Main test module for cargo test
#[cfg(test)]
mod tests {
	#[test]
	fn test_suite_loads() {
		// This test verifies that the test suite compiles and loads correctly
		println!("✅ Omniference test suite loaded successfully");
	}
}

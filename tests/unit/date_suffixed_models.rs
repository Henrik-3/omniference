#[cfg(test)]
mod date_suffixed_models_tests {
	use omniference::catalog::normalize_model_id;

	#[test]
	fn test_normalize_model_id_with_hyphenated_date() {
		assert_eq!(normalize_model_id("o3-2025-04-16"), "o3");
	}

	#[test]
	fn test_normalize_model_id_with_compact_date() {
		assert_eq!(normalize_model_id("gpt-4-20250416"), "gpt-4");
	}

	#[test]
	fn test_normalize_model_id_without_date() {
		assert_eq!(normalize_model_id("gpt-4o"), "gpt-4o");
	}

	#[test]
	fn test_normalize_model_id_o1_without_date() {
		assert_eq!(normalize_model_id("o1-preview"), "o1-preview");
	}

	#[test]
	fn test_normalize_model_id_case_insensitive() {
		assert_eq!(normalize_model_id("O3-2025-04-16"), "o3");
	}

	#[test]
	fn test_normalize_model_id_multiple_dates() {
		assert_eq!(normalize_model_id("gpt-4.1-2025-04-16"), "gpt-4.1");
	}

	#[test]
	fn test_normalize_model_id_compact_then_hyphenated() {
		assert_eq!(normalize_model_id("gpt-4-2025-04-16-20250416"), "gpt-4");
	}

	#[test]
	fn test_normalize_model_id_hyphenated_then_compact_strips_once() {
		assert_eq!(normalize_model_id("gpt-4-20250416-2025-04-16"), "gpt-4-20250416");
	}

	#[test]
	fn test_normalize_model_id_non_digit_suffix_kept() {
		assert_eq!(normalize_model_id("gpt-4-1234567a"), "gpt-4-1234567a");
		assert_eq!(normalize_model_id("gpt-4-1234567"), "gpt-4-1234567");
		assert_eq!(normalize_model_id("gpt-4-2025-4-16"), "gpt-4-2025-4-16");
	}

	#[test]
	fn test_normalize_model_id_uppercase_compact() {
		assert_eq!(normalize_model_id("GPT-4-20250416"), "gpt-4");
	}
}

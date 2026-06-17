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
}

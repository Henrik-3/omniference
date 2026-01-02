#[cfg(test)]
mod date_suffixed_models_tests {
    use omniference::adapters::openai_responses::OpenAIResponsesAdapter;

    #[test]
    fn test_normalize_model_id_with_hyphenated_date() {
        let model_id = "o3-2025-04-16";
        let normalized = OpenAIResponsesAdapter::normalize_model_id(model_id);
        assert_eq!(normalized, "o3");
    }

    #[test]
    fn test_normalize_model_id_with_compact_date() {
        let model_id = "gpt-4-20250416";
        let normalized = OpenAIResponsesAdapter::normalize_model_id(model_id);
        assert_eq!(normalized, "gpt-4");
    }

    #[test]
    fn test_normalize_model_id_without_date() {
        let model_id = "gpt-4o";
        let normalized = OpenAIResponsesAdapter::normalize_model_id(model_id);
        assert_eq!(normalized, "gpt-4o");
    }

    #[test]
    fn test_normalize_model_id_o3_with_date() {
        let model_id = "o3-2025-04-16";
        let capabilities = OpenAIResponsesAdapter::parse_model_capabilities(model_id);

        assert!(capabilities.input_modalities.len() > 0);
        assert!(capabilities.output_modalities.len() > 0);
        assert!(capabilities
            .capabilities
            .contains(&omniference::types::ModelCapabilities::Tools));
    }

    #[test]
    fn test_normalize_model_id_gpt_5_with_date() {
        let model_id = "gpt-5-pro-2025-12-01";
        let capabilities = OpenAIResponsesAdapter::parse_model_capabilities(model_id);

        assert!(capabilities
            .input_modalities
            .contains(&omniference::types::Modality::Image));
        assert!(capabilities
            .output_modalities
            .contains(&omniference::types::Modality::Text));
        assert!(capabilities.context_length.is_some());
        assert!(capabilities.max_tokens.is_some());
    }

    #[test]
    fn test_normalize_model_id_o1_without_date() {
        let model_id = "o1-preview";
        let normalized = OpenAIResponsesAdapter::normalize_model_id(model_id);
        assert_eq!(normalized, "o1-preview");
    }

    #[test]
    fn test_normalize_model_id_case_insensitive() {
        let model_id = "O3-2025-04-16";
        let normalized = OpenAIResponsesAdapter::normalize_model_id(model_id);
        assert_eq!(normalized, "o3");
    }

    #[test]
    fn test_normalize_model_id_multiple_dates() {
        let model_id = "gpt-4.1-2025-04-16";
        let capabilities = OpenAIResponsesAdapter::parse_model_capabilities(model_id);

        // Should still match the gpt-4.1 family capabilities
        assert!(capabilities
            .input_modalities
            .contains(&omniference::types::Modality::Text));
        assert!(capabilities.context_length.is_some());
    }
}

#[cfg(test)]
mod tests {
    // Simple test to verify configuration can be loaded
    #[test]
    fn test_example_config_exists() {
        // Test that the example configuration file exists
        let example_path = "tests/config/test_config.example.json";
        assert!(std::path::Path::new(example_path).exists(), 
                "Example config file should exist at {}", example_path);
    }

    #[test]
    fn test_gitignore_excludes_config() {
        // Test that sensitive config files are properly ignored
        let gitignore_path = ".gitignore";
        let gitignore_content = std::fs::read_to_string(gitignore_path)
            .expect("Should be able to read .gitignore");
        
        // Check that test config files are excluded
        assert!(gitignore_content.contains("tests/config/test_config.json"));
        assert!(gitignore_content.contains("tests/config/*.json"));
        
        // Check that example file is not excluded
        assert!(gitignore_content.contains("!tests/config/test_config.example.json"));
    }

    #[test]
    fn test_environment_variables() {
        // Test that environment variables can be read (even if not set)
        let skip_tests = std::env::var("SKIP_LIVE_TESTS");
        match skip_tests {
            Ok(val) => println!("SKIP_LIVE_TESTS is set to: {}", val),
            Err(_) => println!("SKIP_LIVE_TESTS is not set"),
        }
        
        let log_responses = std::env::var("LOG_RESPONSES");
        match log_responses {
            Ok(val) => println!("LOG_RESPONSES is set to: {}", val),
            Err(_) => println!("LOG_RESPONSES is not set"),
        }
    }
}
# Test Configuration

This directory contains configuration files for running tests with various AI providers.

## Setup

1. **Environment Variables (Recommended)**
   Set these environment variables to configure providers without committing secrets:
   
   ```bash
   # OpenAI Configuration
   export OPENAI_API_KEY="sk-your-openai-api-key-here"
   
   # Anthropic Configuration  
   export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key-here"
   
   # Test Configuration Options
   export SKIP_LIVE_TESTS="false"  # Set to "true" to skip live API calls
   export LOG_RESPONSES="true"     # Set to "true" to log API responses
   ```

2. **Configuration File (Alternative)**
   Copy the example configuration and customize it:
   
   ```bash
   cp tests/config/test_config.example.json tests/config/test_config.json
   ```
   
   Then edit `tests/config/test_config.json` to add your API keys and configure providers.

## Configuration Structure

The configuration supports multiple AI providers:

- **Ollama** (Local - no API key required)
- **OpenAI** (OpenAI API)
- **OpenAI Compat** (OpenAI-compatible endpoints)
- **Anthropic** (Claude API)
- **Google** (Gemini API)

## Usage in Tests

```rust
use crate::config::TestConfig;
use crate::config::helpers;

#[tokio::test]
async fn test_something() {
    let config = TestConfig::load().unwrap();
    
    if helpers::should_run_live_tests() && helpers::is_provider_enabled("openai") {
        // Run test with live API calls
        let provider = config.get_provider("openai").unwrap();
        let endpoint = helpers::create_endpoint_from_config(provider);
        // ... test code ...
    } else {
        // Skip live test
        println!("Skipping live test - provider not configured or tests disabled");
    }
}
```

## Security

- **NEVER** commit `tests/config/test_config.json` to version control
- The `.gitignore` file excludes all JSON files in this directory except the example
- Use environment variables for sensitive data whenever possible
- The example configuration contains placeholder values only
# OpenAI Responses API Test Configuration

This document describes how to configure and run the comprehensive OpenAI Responses API tests.

## Environment Variables

To run the live tests, you need to set the following environment variables:

### Required for Live Tests
- `OPENAI_API_KEY`: Your OpenAI API key (or OpenRouter API key) to test with gpt-3.5-turbo model
- `OPENAI_BASE_URL`: Base URL for the API (defaults to https://openrouter.ai/api for OpenRouter)
- `SKIP_LIVE_TESTS`: Set to "true" to skip tests that make actual API calls

### Example Configuration
```bash
# For OpenRouter (recommended for testing)
export OPENAI_API_KEY="your_openrouter_api_key_here"
export OPENAI_BASE_URL="https://openrouter.ai/api"

# For OpenAI directly
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_BASE_URL="https://api.openai.com"

# Or skip live tests entirely
export SKIP_LIVE_TESTS="true"
```

## Running the Tests

### Run All Integration Tests
```bash
cargo test --test integration
```

### Run Only OpenAI Responses Tests
```bash
cargo test --test integration test_openai_responses
```

### Run with Live API Calls
```bash
OPENAI_API_KEY="your_key" cargo test --test integration test_openai_responses
```

### Run Without Live API Calls
```bash
SKIP_LIVE_TESTS="true" cargo test --test integration test_openai_responses
```

## Test Coverage

The test suite covers all major parameters of the OpenAI Responses API:

### Core Parameters
- `model`: Model selection (tested with openai/gpt-3.5-turbo)
- `input`: Various input formats (string, message objects, content parts)
- `instructions`: System instructions
- `max_output_tokens`: Output length control

### Sampling Parameters
- `temperature`: Randomness control (0.0-2.0)
- `top_p`: Nucleus sampling (0.0-1.0)

### Advanced Features
- `parallel_tool_calls`: Enable/disable parallel tool execution
- `stream`: Streaming vs non-streaming responses
- `store`: Response storage configuration
- `background`: Background processing

### Safety & Caching
- `safety_identifier`: Safety policy enforcement
- `prompt_cache_key`: Response caching
- `service_tier`: Latency tier selection
- `truncation`: Truncation strategy

### Tools & Functions
- `tools`: Function definitions for tool calling
- `tool_choice`: Tool selection strategy

### Metadata & Tracking
- `metadata`: Custom key-value pairs
- `user`: User identification
- `conversation`: Conversation context

### Text Configuration
- `text.verbosity`: Response detail level
- `text.format`: JSON schema formatting

### Reasoning
- `reasoning.enabled`: Enable reasoning models

### Streaming Options
- `stream_options.include_obfuscation`: Obfuscation in streams

## Test Scenarios

1. **Minimal Request**: Tests basic functionality with minimal parameters
2. **Comprehensive Request**: Tests all supported parameters together
3. **Streaming Request**: Tests streaming response functionality
4. **Invalid Model**: Tests error handling for non-existent models
5. **Malformed Request**: Tests JSON parsing error handling

## Expected Behavior

- Tests should pass with valid API keys and proper configuration
- Tests should gracefully skip when `SKIP_LIVE_TESTS=true`
- Tests should handle various HTTP status codes appropriately:
  - `200 OK`: Successful response
  - `400 Bad Request`: Invalid request format
  - `401 Unauthorized`: Invalid API key
  - `422 Unprocessable Entity`: Valid JSON but invalid parameters

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set `OPENROUTER_API_KEY` environment variable
2. **Rate Limiting**: OpenRouter may rate limit requests; add delays if needed
3. **Model Availability**: Ensure gpt-5-nano is available on OpenRouter
4. **Network Issues**: Tests may fail due to network connectivity

### Debug Mode
Run tests with debug output:
```bash
RUST_LOG=debug cargo test --test integration test_openai_responses
```

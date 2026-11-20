use crate::{
	adapter::{AdapterError, ChatAdapter},
	adapter::{provider_request, provider_url},
	adapters::dialects::OpenRouterDialect,
	image::{input_reference, response_from_openai},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;

/// OpenRouter app attribution, sent on every OpenRouter call. A header of the same name in the endpoint
/// `extra_headers` or, for chat, the vendor profile `[headers]` replaces it.
const ATTRIBUTION: [(&str, &str); 2] = [("HTTP-Referer", env!("CARGO_PKG_REPOSITORY")), ("X-Title", "Omniference")];

fn add_attribution(headers: &mut BTreeMap<String, String>, profile_headers: Option<&BTreeMap<String, String>>) {
	for (name, value) in ATTRIBUTION {
		let taken = headers
			.keys()
			.chain(profile_headers.into_iter().flat_map(BTreeMap::keys))
			.any(|key| key.eq_ignore_ascii_case(name));
		if !taken {
			headers.insert(name.to_string(), value.to_string());
		}
	}
}

fn attributed(endpoint: &ProviderEndpoint) -> ProviderEndpoint {
	let mut endpoint = endpoint.clone();
	add_attribution(&mut endpoint.extra_headers, None);
	endpoint
}

pub struct OpenRouterAdapter;

#[async_trait]
impl ChatAdapter for OpenRouterAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenRouter
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let endpoint = &attributed(endpoint);
		let client = crate::adapter::shared_http_client().clone();
		// Use /api/v1/models/user to respect user settings
		let url = format!("{}/v1/models/user", endpoint.base_url);

		let mut request = client.get(&url);

		if let Some(api_key) = &endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}

		let resp = crate::adapter::send_with_timeout(request, crate::adapter::endpoint_timeout(endpoint), false).await?;

		if !resp.status().is_success() {
			return Err(crate::upstream::http_error(resp, endpoint, provider_name).await);
		}

		let models_response: OpenRouterModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

		let discovered_models: Vec<DiscoveredModel> = models_response
			.data
			.into_iter()
			.map(|model| {
				let capabilities = self.live_model_facts(&model);
				DiscoveredModel {
					id: format!("{}/{}", provider_name.to_lowercase(), model.id),
					name: model.name,
					provider_name: provider_name.to_string(),
					provider_kind: ProviderKind::OpenRouter,
					input_modalities: capabilities.input_modalities,
					output_modalities: capabilities.output_modalities,
					capabilities: capabilities.capabilities,
					context_length: capabilities.context_length,
					max_tokens: capabilities.max_tokens,
					pricing: self.live_pricing(&model.pricing),
					reasoning_budget: None,
				}
			})
			.collect();

		Ok(discovered_models)
	}

	async fn execute_image(&self, mut request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		crate::image::require_single_unmasked(&request)?;
		add_attribution(&mut request.model.provider.endpoint.extra_headers, None);
		let endpoint_config = &request.model.provider.endpoint;
		let api_key = endpoint_config
			.api_key
			.as_deref()
			.ok_or_else(|| AdapterError::invalid("provider API key is missing"))?;
		// shortcut: multipart edit fields arrive as strings, so numeric extras (seed, output_compression) stay strings; parse them if OpenRouter refuses that
		let mut body = Value::Object(request.raw_request.clone().unwrap_or_default());
		let options = &request.options;
		for (name, value) in [
			("size", &options.size),
			("aspect_ratio", &options.aspect_ratio),
			("quality", &options.quality),
			("background", &options.background),
			("output_format", &options.output_format),
		] {
			if let Some(value) = value {
				body[name] = json!(value);
			}
		}
		if let Some(n) = options.n {
			body["n"] = json!(n);
		}
		if body.get("output_format").is_none() {
			body["output_format"] = json!("png");
		}
		body["model"] = json!(request.model.model_id);
		body["prompt"] = json!(request.prompt);
		if request.operation == ImageOperation::Edit {
			if request.input_images.is_empty() {
				return Err(AdapterError::invalid("editing requires an input image"));
			}
			body["input_references"] = json!(request.input_images.iter().map(input_reference).collect::<Vec<_>>());
		}
		let response = provider_request(
			crate::adapter::shared_http_client().post(provider_url(&endpoint_config.base_url, "v1/images")),
			endpoint_config,
		)?
		.bearer_auth(api_key)
		.json(&body)
		.send()
		.await
		.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(crate::upstream::http_error(response, endpoint_config, &request.model.provider.name).await);
		}
		let value: Value = response.json().await.map_err(|error| AdapterError::http(error.to_string()))?;
		let provider_cost = value
			.get("cost")
			.and_then(Value::as_f64)
			.or_else(|| value.pointer("/usage/cost").and_then(Value::as_f64));
		let mut result = response_from_openai(&value, request.input_images.len() as u32)?;
		result.usage.provider_cost = provider_cost;
		Ok(result)
	}

	async fn discover_image_models(&self, provider_name: &str, endpoint_config: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let endpoint_config = &attributed(endpoint_config);
		let mut call = provider_request(
			crate::adapter::shared_http_client().get(provider_url(&endpoint_config.base_url, "v1/images/models")),
			endpoint_config,
		)?;
		if let Some(api_key) = &endpoint_config.api_key {
			call = call.bearer_auth(api_key);
		}
		let response = call.send().await.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(crate::upstream::http_error(response, endpoint_config, provider_name).await);
		}
		let value: Value = response.json().await.map_err(|error| AdapterError::http(error.to_string()))?;
		let models = value
			.get("data")
			.or_else(|| value.get("models"))
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		Ok(models
			.into_iter()
			.filter_map(|model| {
				let id = model.get("id").or_else(|| model.get("model")).and_then(Value::as_str)?;
				let name = model.get("name").or_else(|| model.get("display_name")).and_then(Value::as_str).unwrap_or(id);
				let input = model
					.pointer("/architecture/input_modalities")
					.or_else(|| model.get("input_modalities"))
					.and_then(Value::as_array);
				let can_edit = input.is_some_and(|modalities| {
					modalities
						.iter()
						.any(|modality| modality.as_str().is_some_and(|value| value.eq_ignore_ascii_case("image")))
				});
				Some(DiscoveredModel {
					id: format!("{}/{}", provider_name.to_ascii_lowercase(), id),
					name: name.to_string(),
					provider_name: provider_name.to_string(),
					provider_kind: ProviderKind::OpenRouter,
					input_modalities: if can_edit { vec![Modality::Text, Modality::Image] } else { vec![Modality::Text] },
					output_modalities: vec![Modality::Image],
					capabilities: if can_edit {
						vec![ModelCapabilities::ImageGeneration, ModelCapabilities::ImageEditing]
					} else {
						vec![ModelCapabilities::ImageGeneration]
					},
					context_length: None,
					max_tokens: None,
					pricing: None,
					reasoning_budget: None,
				})
			})
			.collect())
	}

	/// Chat runs through OpenRouter's Responses API with the OpenRouter dialect, unless the provider's
	/// vendor profile names a dialect with a Responses part.
	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let mut ir = ir;
		let provider = &mut ir.model.provider;
		add_attribution(
			&mut provider.endpoint.extra_headers,
			provider.resolved_profile.as_deref().map(|vendor| &vendor.profile.headers),
		);
		let dialect = ir
			.model
			.provider
			.resolved_profile
			.as_deref()
			.and_then(|vendor| vendor.responses_part())
			.unwrap_or(&OpenRouterDialect);
		super::OpenAIResponsesAdapter.execute_with(ir, cancel, dialect).await
	}
}

impl OpenRouterAdapter {
	/// Parse model capabilities from OpenRouter's detailed model information
	fn live_model_facts(&self, model: &OpenRouterModel) -> ModelCapabilitiesWithModalities {
		let mut capabilities = ModelCapabilitiesWithModalities {
			context_length: model.context_length,
			max_tokens: model.top_provider.as_ref().and_then(|tp| tp.max_completion_tokens),
			capabilities: vec![],
			input_modalities: vec![],
			output_modalities: vec![],
		};
		let arch = &model.architecture;

		for input_modality in &arch.input_modalities {
			match input_modality.as_str() {
				"text" => capabilities.input_modalities.push(Modality::Text),
				"image" => capabilities.input_modalities.push(Modality::Image),
				"file" | "pdf" => capabilities.input_modalities.push(Modality::File),
				"audio" => capabilities.input_modalities.push(Modality::Audio),
				"video" => capabilities.input_modalities.push(Modality::Video),
				_ => {}
			}
		}

		for output_modality in &arch.output_modalities {
			match output_modality.as_str() {
				"text" => capabilities.output_modalities.push(Modality::Text),
				"image" => capabilities.output_modalities.push(Modality::Image),
				"audio" => capabilities.output_modalities.push(Modality::Audio),
				"embeddings" => capabilities.output_modalities.push(Modality::Embeddings),
				_ => {}
			}
		}

		if model.supported_parameters.iter().any(|p| p == "tools" || p == "tool_choice") {
			capabilities.capabilities.push(ModelCapabilities::Tools);
		}

		capabilities.capabilities.extend(Self::reasoning_capabilities(model));

		capabilities
	}

	/// Reasoning support from `supported_parameters` and the `reasoning` object: `Reasoning`
	/// plus one capability per supported effort level that Omniference knows.
	fn reasoning_capabilities(model: &OpenRouterModel) -> Vec<ModelCapabilities> {
		let advertised = model.supported_parameters.iter().any(|p| p == "reasoning" || p == "include_reasoning");
		if !advertised && model.reasoning.is_none() {
			return Vec::new();
		}
		let mut capabilities = vec![ModelCapabilities::Reasoning];
		let efforts = model.reasoning.iter().flat_map(|reasoning| &reasoning.supported_efforts);
		for capability in efforts.filter_map(|effort| crate::catalog::convert::reasoning_effort_capability(effort)) {
			if !capabilities.contains(&capability) {
				capabilities.push(capability);
			}
		}
		capabilities
	}

	/// Build a [`crate::catalog::ModelPricing`] from OpenRouter's discovery pricing.
	///
	/// Delegates to [`crate::catalog::pricing_from_catalog`] so discovery and the
	/// on-demand catalog metadata path share one USD-per-token → per-million conversion.
	fn live_pricing(&self, pricing: &OpenRouterPricing) -> Option<crate::catalog::ModelPricing> {
		crate::catalog::pricing_from_catalog(&crate::catalog::OpenRouterCatalogPricing {
			prompt: pricing.prompt.clone(),
			completion: pricing.completion.clone(),
			..Default::default()
		})
	}
}

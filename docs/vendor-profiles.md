# Vendor profiles and dialects

Many vendors speak "OpenAI Chat Completions", "OpenAI Responses" or "Anthropic Messages" but differ in details: a field they reject, a field with
another name, reasoning sent in their own format. Omniference keeps one implementation of each protocol
(the adapter) and describes each vendor's differences separately:

- a **vendor profile** (`catalog/providers/<slug>.toml`): data only, simple JSON changes to the request
- a **dialect** (`src/adapters/dialects/`): Rust code, for anything a profile cannot express

The vendor is separate from the protocol. `ProviderKind` picks the adapter; the profile only changes
what that adapter sends and how it reads fields the vendor interprets differently. Never pick an
adapter by host.

Profiles and dialects apply to three adapters, each with its own protocol:

| Adapter | Protocol | Rules table | Dialect trait |
| ------- | -------- | ----------- | ------------- |
| `OpenAIAdapter` (`openai_compat.rs`) | `/v1/chat/completions` | `[request.chat]` | `ChatDialect` |
| `OpenAIResponsesAdapter` (`openai_responses.rs`) | `/v1/responses` | `[request.responses]` | `ResponsesDialect` |
| `AnthropicAdapter` (`anthropic.rs`) | `/v1/messages` | `[request.messages]` | `MessagesDialect` |

`OpenRouterAdapter` runs chat through OpenRouter's Responses API: it hands the request to the Responses
adapter with the `openrouter` dialect (or the Responses part of the dialect its profile names), so
`[request.responses]` rules and `[headers]` apply to it too. Its model discovery and image generation stay
its own. Every OpenRouter adapter call (chat, discovery, images) sends the fixed app attribution headers
`HTTP-Referer` (the repository URL) and `X-Title: Omniference`; a header of the same name in the endpoint
`extra_headers` or, for chat, the profile `[headers]` replaces them. The Gemini adapter ignores profiles.

## How a request passes through

For a provider with a resolved profile, each of the three adapters builds the body in this order:

1. The base adapter builds the request from the `ChatRequestIR` (typed for Chat and Messages, a JSON
   value for Responses). A request the Anthropic Messages skin forwards raw skips steps 1 and 2: the
   client's body is the request, and only the rules and headers apply.
2. The dialect's `patch_request` for that protocol changes it (Rust, conditional logic).
3. `extra_body` from the client is merged into the JSON.
4. The profile's rules **for that protocol** change the JSON: `reject`, then `rename`, `drop` and `default`.
5. The profile's `[headers]` are added; endpoint `extra_headers` win over them.

When a Chat reply comes back, the dialect's read hooks (`delta_reasoning`, `message_reasoning`) decide
where reasoning text is taken from. On Responses, `replays_reasoning` decides which reasoning blocks of
earlier turns are sent back and `usage_cost` reads a provider-reported cost. Messages has no read hooks yet.

## How a provider finds its vendor

`OmniferenceService::register_provider` resolves the profile once, through
`catalog::providers::resolve_provider`, and stores it in `ProviderConfig::resolved_profile`:

1. `ProviderConfig::catalog_provider_slug`, when it is set (this is also the slug used for models.dev
   prices), else
2. the one profile whose `hosts` contain the host of the base URL, else
3. no vendor: the provider keeps the generic behaviour.

Two profiles that claim the same host, an unknown dialect name or an invalid rule fail registration with
`ProviderRegistrationError::InvalidProfile`. Library users who build a `ModelRef` by hand call
`resolve_provider` themselves.

## Writing a profile

```toml
# catalog/providers/example.toml
slug = "example"                 # required, must equal the file name
hosts = ["api.example.com"]      # optional, matched when no slug is configured
dialect = "example"              # optional, a name registered in DIALECTS

[request.chat]
rename = { "max_completion_tokens" = "max_tokens" }
reject = ["logprobs"]
drop = ["stream_options"]
default = { "parallel_tool_calls" = false }

[request.responses]
reject = ["background"]

[request.messages]
drop = ["cache_control"]

[headers]
"x-example-unit" = "USD"
```

| Rule | Effect |
| ---- | ------ |
| `rename` | Moves a field to another key **at the same level** (only the last key may change). If the client set both keys, the request is a 400 `invalid_request`. |
| `reject` | Fields the vendor cannot honour. A client that sets one gets a 400 `invalid_request` instead of a silently ignored setting. |
| `drop` | Fields removed before sending. Only for fields whose loss changes nothing for the client. |
| `default` | Values sent when the request leaves the field unset (missing or `null`). |

Each protocol has its own table, because one vendor can serve several protocols on one host (DeepSeek
serves `/anthropic` on `api.deepseek.com`): a `[request.chat]` rule never touches a Responses or Messages
body. A flat `[request]` with `rename`, `reject`, ... directly inside is refused when the profile loads.

Paths are dotted key names, with `[]` for every item of an array: `messages[].name`,
`tools[].function.strict`. There are no wildcards or indexes, and a path that does not exist does nothing.

`[headers]` are shared by all protocols: they describe the vendor, not the protocol. Credentials never go
into `[headers]`; they belong in the endpoint's `extra_headers` or API key.

### Layers

A profile can be layered, and each layer overrides only what it sets (`rename`, `default` and
`headers` key by key, the other fields as a whole). Each protocol table is layered on its own:

1. `catalog/providers/<slug>.toml` (checked in, compiled into the crate)
2. `$OMNIFERENCE_CATALOG_OVERRIDE_DIR/providers/<slug>.toml` (deployment overrides)
3. `ProviderConfig::profile` (inline, per provider; may not set `slug` or `hosts`)

## Writing a dialect

Use a dialect when a profile is not enough: the change depends on the request or on a value, moves a
value to another place, or reads a reply field in the vendor's own format.

A dialect implements one trait per protocol it differs in:

| Trait | `patch_request` gets | Read hooks |
| ----- | -------------------- | ---------- |
| `ChatDialect` | `&mut OpenAIChatRequest` | `delta_reasoning`, `message_reasoning` |
| `ResponsesDialect` | `&mut serde_json::Value` (the Responses adapter builds JSON, not a typed struct) | `replays_reasoning`, `usage_cost` |
| `MessagesDialect` | `&mut AnthropicMessagesRequest` | none yet |

```rust
// src/adapters/dialects/example.rs
use super::ChatDialect;
use crate::{adapter::AdapterError, types::{ChatRequestIR, OpenAIChatRequest}};

/// Example AI (`api.example.com`).
pub struct ExampleDialect;

impl ChatDialect for ExampleDialect {
	/// Example takes the tool choice `any` instead of `required`.
	fn patch_request(&self, request: &mut OpenAIChatRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		// change only what Example needs; everything else stays the base request
		Ok(())
	}
}
```

Then register it in `DIALECTS` (`src/adapters/dialects/mod.rs`) under the name the profile uses, with a
part for each protocol it implements:

```rust
("example", Dialect { chat: Some(&ExampleDialect), responses: None, messages: None }),
```

A vendor has one `dialect` name. A protocol whose part is `None` keeps the base behaviour, and an unknown
name fails registration.

Hooks that exist today (`MessagesDialect` has only `patch_request`):

| Hook | Trait | Called | Default |
| ---- | ----- | ------ | ------- |
| `patch_request` | all three | after the base adapter built the request | no change |
| `delta_reasoning` | `ChatDialect` | for each streamed delta | `reasoning_content`, else `reasoning` |
| `message_reasoning` | `ChatDialect` | for a non-streamed message | `reasoning_content`, else `reasoning` |
| `replays_reasoning` | `ResponsesDialect` | for each reasoning block of an earlier turn | only `openai-responses-v1` blocks |
| `usage_cost` | `ResponsesDialect` | on the `usage` of a finished response, when no header or `reported_cost` gave a cost | none |

Rules for hooks:

- A default method is the base behaviour, so a dialect overrides only where its vendor differs.
- Add a hook only when a real vendor needs it, one named difference per hook (for example "the vendor's
  routing fields"), not a general "change anything" hook.
- The Chat and Messages `patch_request` work on the typed request, which has no slot for unknown fields.
  A vendor that needs fields of its own there needs a new hook for that difference.
- Response parsing stays tolerant and generic. Profiles and dialects cover what is sent and the fields a
  vendor fills differently.
- Reasoning efforts a model does not support are a catalog fact (models.dev `reasoning_options`), not a
  vendor quirk: no dialect rewrites efforts.

`MistralDialect` (`src/adapters/dialects/mistral.rs`) is the reference. On Chat it maps `allowed_tools`
in `patch_request` and reads Mistral's `thinking` content parts in the two reasoning hooks. On Responses
it maps `allowed_tools` the same way and refuses fields Mistral accepts only with one value (`store`,
`background`, `truncation`, `top_logprobs`, and `n`, which Mistral would silently ignore), while `mistral.toml` rejects the always-unsupported ones
under `[request.responses]` (`previous_response_id`, `conversation`, `moderation`, `prompt`).

`OpenRouterDialect` (`src/adapters/dialects/openrouter.rs`) has only a Responses part. `patch_request` adds
what the base Responses request lacks (reasoning `max_tokens` from `budget_tokens`, `provider` routing,
`stop`, presence and frequency penalties, never over the client's own) and maps `allowed_tools` like
Mistral. `replays_reasoning` sends back every signed reasoning format, because OpenRouter routes each
block to the upstream that produced it; such a block goes out as a `reasoning` item with `format`,
`signature` and its text as `reasoning_text` content (`summary_text` for summaries). `usage_cost` reads
`usage.cost`, else `cost_details.upstream_inference_cost`, else the sum of its input and output parts.

`CortecsDialect` (`src/adapters/dialects/cortecs.rs`) has only a Responses part. `patch_request` maps
`ChatRequestIR.provider_routing` onto Cortecs' request fields: `only` becomes `allowed_providers`, and
`order` becomes `allowed_providers` in that order (slugs outside `only` dropped, the rest of `only` appended)
with `provider_order: "custom"`. Cortecs never falls back beyond `allowed_providers`, so a caller that wants
fallbacks lists every provider in `order`; an `only` that leaves no provider is refused. The client can't
set provider routing through a skin, so this is tested in `tests/unit/vendor_profiles.rs`, not as a case.

## Testing

Every dialect behaviour and vendor rule gets its own case in the case matrix (`tests/README.md`, "Case
matrix"), in a folder named `<vendor>_<scenario>` under the adapter directory of its protocol
(`openai_compat`, `openai_responses` or `anthropic`):

```
tests/cases/openai_chat/openai_compat/mistral_allowed_tools/
├── case.toml               # vendor = "mistral"
├── client_request.json
├── upstream_response.sse
└── *.snap                  # accepted with INSTA_UPDATE=always, reviewed by eye
```

`case.toml` sets `vendor = "<slug>"`, which the runner passes as `catalog_provider_slug`, so the case
resolves the profile. The fake upstream runs on localhost, so without it a case never picks up a vendor.
Check the `upstream_request.snap` for the changed body.

To check against the real vendor, add a `[[target]]` with `vendor = "<slug>"` to `tests/live_targets.toml`.
`CASE_LIVE=1` replays the vendor's cases on its targets only.

## Checklist for a new vendor

1. `catalog/providers/<slug>.toml` with `slug` and `hosts`.
2. Request rules under `[request.<protocol>]` for simple differences; a dialect part plus a `DIALECTS`
   entry only where rules cannot do it.
3. A `<vendor>_<scenario>` case per behaviour, with `vendor` in `case.toml`.
4. A live target with `vendor`, and a live run of its cases.
5. A short note in `AGENTS.md` under "Vendor profiles" for behaviour a reader would not guess.

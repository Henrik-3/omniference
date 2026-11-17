# Vendor profiles and dialects

Many vendors speak "OpenAI Chat Completions" but differ in details: a field they reject, a field with
another name, reasoning sent in their own format. Omniference keeps one implementation of each protocol
(the adapter) and describes each vendor's differences separately:

- a **vendor profile** (`catalog/providers/<slug>.toml`): data only, simple JSON changes to the request
- a **dialect** (`src/adapters/dialects/`): Rust code, for anything a profile cannot express

The vendor is separate from the protocol. `ProviderKind` picks the adapter; the profile only changes
what that adapter sends and how it reads fields the vendor interprets differently. Never pick an
adapter by host.

> Profiles and dialects apply only to the OpenAI Chat adapter (`src/adapters/openai_compat.rs`,
> `ProviderKind::OpenAICompat`). The Responses, Anthropic, Gemini and OpenRouter adapters ignore them.

## How a request passes through

For a provider with a resolved profile, the Chat adapter builds the body in this order:

1. The base adapter builds the typed `OpenAIChatRequest` from the `ChatRequestIR`.
2. `ChatDialect::patch_request` changes the typed request (Rust, conditional logic).
3. `extra_body` from the client is merged into the JSON.
4. The profile's `[request]` rules change the JSON: `reject`, then `rename`, `drop` and `default`.
5. The profile's `[headers]` are added; endpoint `extra_headers` win over them.

When the reply comes back, the dialect's read hooks (`delta_reasoning`, `message_reasoning`) decide
where reasoning text is taken from.

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

[request]
rename = { "max_completion_tokens" = "max_tokens" }
reject = ["logprobs"]
drop = ["stream_options"]
default = { "parallel_tool_calls" = false }

[headers]
"x-example-unit" = "USD"
```

| Rule | Effect |
| ---- | ------ |
| `rename` | Moves a field to another key **at the same level** (only the last key may change). If the client set both keys, the request is a 400 `invalid_request`. |
| `reject` | Fields the vendor cannot honour. A client that sets one gets a 400 `invalid_request` instead of a silently ignored setting. |
| `drop` | Fields removed before sending. Only for fields whose loss changes nothing for the client. |
| `default` | Values sent when the request leaves the field unset (missing or `null`). |

Paths are dotted key names, with `[]` for every item of an array: `messages[].name`,
`tools[].function.strict`. There are no wildcards or indexes, and a path that does not exist does nothing.

Credentials never go into `[headers]`; they belong in the endpoint's `extra_headers` or API key.

### Layers

A profile can be layered, and each layer overrides only what it sets (`rename`, `default` and
`headers` key by key, the other fields as a whole):

1. `catalog/providers/<slug>.toml` (checked in, compiled into the crate)
2. `$OMNIFERENCE_CATALOG_OVERRIDE_DIR/providers/<slug>.toml` (deployment overrides)
3. `ProviderConfig::profile` (inline, per provider; may not set `slug` or `hosts`)

## Writing a dialect

Use a dialect when a profile is not enough: the change depends on the request, moves a value to another
place, or reads a reply field in the vendor's own format.

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

Then register it in `DIALECTS` (`src/adapters/dialects/mod.rs`) under the name the profile uses.

Hooks that exist today (`ChatDialect`):

| Hook | Called | Default |
| ---- | ------ | ------- |
| `patch_request` | after the base adapter built the typed request | no change |
| `delta_reasoning` | for each streamed delta | `reasoning_content`, else `reasoning` |
| `message_reasoning` | for a non-streamed message | `reasoning_content`, else `reasoning` |

Rules for hooks:

- A default method is the base behaviour, so a dialect overrides only where its vendor differs.
- Add a hook only when a real vendor needs it, one named difference per hook (for example "the vendor's
  routing fields"), not a general "change anything" hook.
- `patch_request` works on the typed request, which has no slot for unknown fields. A vendor that needs
  fields of its own needs a new hook for that difference.
- Response parsing stays tolerant and generic. Profiles and dialects cover what is sent and the fields a
  vendor fills differently.
- Reasoning efforts a model does not support are a catalog fact (models.dev `reasoning_options`), not a
  vendor quirk: no dialect rewrites efforts.

`MistralDialect` (`src/adapters/dialects/mistral.rs`) is the reference: it maps `allowed_tools` in
`patch_request` and reads Mistral's `thinking` content parts in the two reasoning hooks.

## Testing

Every dialect behaviour and vendor rule gets its own case in the case matrix (`tests/README.md`, "Case
matrix"), in a folder named `<vendor>_<scenario>`:

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
2. Request rules for simple differences; a dialect plus a `DIALECTS` entry only where rules cannot do it.
3. A `<vendor>_<scenario>` case per behaviour, with `vendor` in `case.toml`.
4. A live target with `vendor`, and a live run of its cases.
5. A short note in `AGENTS.md` under "Vendor profiles" for behaviour a reader would not guess.

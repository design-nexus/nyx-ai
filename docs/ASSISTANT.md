# Nyx Assistant and Nyx Notes on AI Core

What the Assistant's in-process runner does today, and the AI Core interface that
replaces it. The migration itself is Handoff 05; this is what it builds on.

## Boundary

| AI Core | Nyx Assistant |
|---|---|
| Model files, llama.cpp, loading, unloading, scheduling | Personas, planner prompts, conversation state |
| Streaming generation, tool-call transport, grammars | Which tools a turn offers, and running them |
| The shared action registry (discovery, argument checks, routing) | Policy: capabilities, grants, budgets, approvals, the audit journal |
| Battery saver (models on the CPU on battery) | Shorter replies on battery (`max_tokens`), voice behaviour |
| Embeddings | Memory and retrieval |

Voice stays outside AI Core: audio capture, voice activity detection, speech
recognition, speech synthesis, conversation mode and audio sessions belong to
Nyx Voice and the Assistant's services.

## The runner's features, mapped

| Assistant today (`assistant-inference`) | AI Core |
|---|---|
| Hand-written Gemma 4 and Qwen 3 prompt formats | The model's own chat template (`--jinja`); `apply_template` shows the prompt |
| GBNF grammars (`generate_with_grammar`) | `ChatRequest.grammar` (not with `tools` or `response_format`) |
| JSON output | `response_format` (`json_object` or `json_schema`) |
| Tool calls parsed from text | `tools` and `tool_choice`; calls arrive as `tool_calls` deltas |
| Vision projector loaded on first use | Image parts in `messages`; the projector loads with the model |
| Battery mode: 0 GPU layers | `battery_saver` (Settings or `nyx-ai service battery_saver=on`): on battery every model loads with no GPU layers; idle models reload the new way |
| Battery mode: short replies | The Assistant's own `sampling.max_tokens`, using `ServiceSettings.on_battery` |
| Prefix KV reuse | llama.cpp's prompt cache per slot (automatic for a repeated prefix) |
| Cancel | `cancel`, or closing the connection |
| Its own model folder and downloads | `~/.local/share/nyx-assistant/models` is one of AI Core's `model_dirs`; downloads go through `download_model` |

## Interactive turns

- Send turns with `priority: interactive`. They go ahead of `normal` and
  `background` work (Glimpse indexing, Notes summaries) and age so nothing
  starves.
- Name the model, or use the `assistant` default role from `service_settings`
  (set in Nyx Settings, Local AI).
- A turn that needs a model that isn't loaded gets `queued` and `model_loading`
  events; show them instead of a silent wait. With `max_loaded = 1`, AI Core
  unloads the other model once its requests finish (it never cuts one off).
- `load_model` with `keep: true` pins the Assistant's model while voice
  conversation mode is on; `unload_model` releases the pin.

## Grammars

```json
{"type": "chat", "model": "gemma-4-E2B-it-Q4_K_M", "priority": "interactive",
 "messages": [{"role": "user", "content": "Is Paris the capital of France?"}],
 "grammar": "root ::= \"yes\" | \"no\"", "thinking": false}
```

Verified with Gemma 4 E2B on `prism-b12128-d7b2896`: answers are only `yes` or
`no`, on the GPU and on the CPU. Grammars are up to 64 KB; a grammar llama.cpp
can't parse fails the request with the worker's message.

## Actions

The Assistant's tool registry can take app actions from AI Core instead of
discovering apps itself:

- `list_actions` returns every running Nyx app's control-interface tools and the
  actions programs registered, with JSON Schemas, side effects and whether they
  ask first.
- `invoke_action` checks the arguments and routes the call. Actions that ask
  first need `confirmed: true`; send it only after the Assistant's own approval
  step (its policy and grants decide, AI Core only enforces that the step
  happened). The owning app may still refuse (`action_failed`).
- The Assistant can offer its own actions to other programs with
  `serve_actions`.

## Notes

Notes calls the Assistant's model API (`model_generate`, `model_embed`,
`model_capabilities` on `assistant.sock`). On AI Core:

| Notes today | AI Core |
|---|---|
| `model_capabilities` | `service_settings` (defaults) and `list_capabilities` |
| `model_generate {system, text, max_tokens: 2048, temperature: 0.25}` | `chat` with a system and a user message, `sampling: {max_tokens: 2048, temperature: 0.25}`, `priority: normal`, the `chat` (or `assistant`) default model |
| `model_embed {input}` | `embed` with the `embedding` default model; store `identity` with the index |

Notes should depend on `nyx-ai-client` directly rather than go through the
Assistant.

## Consumer isolation

- Every connection is identified by the program it runs; untrusted programs
  can't remove models, change settings or serve actions, and can run only
  read-only actions.
- Requests are per connection: one app can't cancel or read another's.
- Prompts, replies and action arguments are never logged; diagnostics are
  counters and timings.

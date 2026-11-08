# Nyx AI Core protocol, API version 1

Newline-delimited JSON over the Unix socket `$XDG_RUNTIME_DIR/nyx-ai/ai.sock`
(folder `0700`, socket `0600`; `NYX_AI_SOCKET` overrides the path). The Rust types
are in `crates/nyx-ai-proto`; applications should use `nyx-ai-client` rather
than speaking it by hand.

## Connection

1. The server accepts only peers of the same user (`SO_PEERCRED`). It names the
   peer by the program it runs (`nyx-peer`: the daemon's own folder or
   `/etc/nyx/trusted-bin`), or `untrusted:<declared name>` otherwise.
2. The client sends `{"hello":{"api_version":1,"client":"nyx-chat"}}`.
3. The server answers `{"hello":{"api_version":1,"server":"0.1.0","client":"nyx-chat"}}`,
   or `{"error":{"error":{"kind":"incompatible_version","message":"…"}}}` and closes.
4. The client sends requests: `{"request":{"id":1,"op":{"type":"list_models"}}}`.
   IDs are the client's own and must be unique among its open requests.
5. For each request the server sends zero or more
   `{"event":{"id":1,"event":{…}}}` and then exactly one
   `{"result":{"id":1,"result":{…}}}` or `{"error":{"id":1,"error":{"kind":…,"message":…}}}`.

Closing the connection cancels its requests, except downloads, which belong to the
daemon. Lines are at most 64 MiB. A connection may have 32 requests open.

## Rules

- Requests are strict. An unknown field is `invalid_request`; an unknown
  operation is `unsupported_operation`. Nothing is silently ignored.
- Replies may gain optional fields within version 1; clients ignore fields they
  don't know. A change an older client can't ignore bumps `api_version`.
- Untrusted peers may read and run inference; `remove_model`,
  `configure_model` and `configure_service` need a trusted program (the daemon
  allows untrusted peers for these only when started with
  `NYX_AI_ALLOW_UNTRUSTED=1`, for test copies).

## Operations

| `type` | Fields | Events | Result |
|---|---|---|---|
| `list_models` | | | `models` |
| `model_info` | `model` | | `model_info` |
| `load_model` | `model`, `keep` (pin for this app) | `model_loading` | `loaded` |
| `unload_model` | `model` | | `unloaded` (`unloaded: false` while something else uses it) |
| `remove_model` | `model` | | `ok` |
| `model_settings` | `model` (`*` for defaults) | | `settings` |
| `configure_model` | `model`, `settings` (`{key: value}`, `""` removes) | | `settings` |
| `download_model` | `model` (`org/repo:QUANT`) | `progress` | `ok` |
| `cancel_download` | `model` | | `ok` |
| `downloads` | | | `downloads` |
| `chat` | see below | `accepted`, `queued`, `model_loading`, `started`, `delta` | `finished` |
| `embed` | `model`, `input` (strings), `priority` | `accepted`, `queued`, `model_loading`, `started` | `embeddings` |
| `apply_template` | as `chat` | | `template` |
| `cancel` | `request` (an id on this connection) | | `ok` |
| `resource_status` | | | `resources` |
| `worker_status` | | | `worker` |
| `list_capabilities` | | | `capabilities` |
| `service_settings` | | | `service` |
| `configure_service` | `max_loaded`, `idle_unload_seconds` (both optional) | | `service` |

`remove_model` deletes cache models only (folder and preset models are refused
with `permission_denied`). For a model still downloading it stops the download and
drops its partial files.

### `chat`

```json
{"type": "chat", "model": "org/repo:Q4_K_M",
 "messages": [
   {"role": "system", "content": "Be brief."},
   {"role": "user", "content": [{"type": "text", "text": "What is this?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
     "function": {"name": "web_search", "arguments": "{\"query\":\"x\"}"}}]},
   {"role": "tool", "tool_call_id": "c1", "content": "…"}],
 "tools": [{"type": "function", "function": {"name": "web_search", "description": "…",
   "parameters": {"type": "object"}}}],
 "tool_choice": "auto",
 "response_format": {"type": "json_schema", "schema": {"type": "object"}},
 "sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
   "repeat_penalty": 1.1, "seed": 1, "max_tokens": 1024, "stop": ["\n\n"]},
 "reasoning_effort": "high", "thinking": true, "priority": "interactive"}
```

- Images must be `data:image/…;base64,` URLs and need a model with `vision`;
  `input_audio` needs `audio`. Otherwise `unsupported_capability`.
- `priority` is `interactive`, `normal` (default) or `background`.
- `delta` events carry `content`, `reasoning` and `tool_calls`
  (`{index, id?, name?, arguments?}`; parts with the same index concatenate).
- `finished` has `finish_reason`, `usage` (`prompt_tokens`,
  `completion_tokens`) and `timings` (`queued_ms`, `load_ms`, `first_token_ms`,
  `total_ms`, `tokens_per_second`).

### `embed`

`embeddings` has `vectors`, `dimensions` and `identity`
(`id`, `file`, `size`, `revision`, `architecture`). Store the identity with an
index and compare it before mixing vectors. The model needs `embedding`.

### Models

A `model` has `id`, `name`, `source` (`cache`, `folder`, `preset`), `path`,
`size`, `architecture`, `quantization`, `parameters`, `context_length`,
`capabilities` (`chat`, `tools`, `thinking`, `vision`, `audio`, `embedding`,
`draft`), `state` (`downloading`, `validating`, `installed`, `loading`, `ready`,
`busy`, `unloading`, `failed`, `missing`), `revision`, `pinned`, `removable`,
`pending_settings` and `error`.

Capabilities come from GGUF metadata and the worker, never from the file name.
`tools` and `thinking` mean the chat template handles them.

### Settings keys

`configure_model` accepts only these keys, checked before they're written:

| Key | Values |
|---|---|
| `ctx-size` | 0 (model default) to 1048576 |
| `n-gpu-layers` | `auto`, `all`, or 0 to 999 |
| `flash-attn` | `on`, `off`, `auto` |
| `cache-type-k`, `cache-type-v` | `f32`, `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl` |
| `temp` | 0 to 5 |
| `top-p`, `min-p` | 0 to 1 |
| `top-k` | 0 to 1000 |
| `repeat-penalty` | 0 to 5 |
| `parallel` | 1 to 16 |
| `embeddings` | `true`, `false` |

Other keys already in `~/.config/nyx-ai/models.ini` are kept as they are.
Changes to a loaded model apply when it's next idle (it unloads and the next
request loads it with the new settings); `pending_settings` shows that.

## OpenAI-compatible endpoint

Separate from this protocol: `http://127.0.0.1:8080/v1` (see `ARCHITECTURE.md`).
`GET /v1/models` lists usable models; `POST /v1/chat/completions`,
`/v1/completions`, `/v1/embeddings` and other `/v1/*` routes the worker serves
take the usual OpenAI bodies with a `model` from that list. Errors use OpenAI's
shape (`{"error": {"code", "message", "type"}}`).

## Errors

`incompatible_version`, `invalid_request`, `unsupported_operation`,
`unknown_model`, `unknown_request`, `unsupported_capability`,
`resource_exhausted`, `busy`, `permission_denied`, `worker_unavailable`,
`worker_failed`, `model_load_failed`, `download_failed`, `cancelled`, `timeout`,
`internal`. The client adds `unavailable` when it can't reach the daemon.

## Changes

- Version 1 (November 2026): first release.

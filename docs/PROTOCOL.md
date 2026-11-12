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
| `embed` | `model`, `input` (strings, `{"image": …}`, `{"audio": …}`), `priority` | `accepted`, `queued`, `model_loading`, `started` | `embeddings` |
| `apply_template` | as `chat` | | `template` |
| `cancel` | `request` (an id on this connection) | | `ok` |
| `resource_status` | | | `resources` |
| `worker_status` | | | `worker` |
| `list_capabilities` | | | `capabilities` |
| `service_settings` | | | `service` |
| `configure_service` | `max_loaded`, `idle_unload_seconds`, `model_updates`, `defaults` (`{role: model}`, `""` clears), `battery_saver`, `add_model_dirs` (folders to list models from as well; absolute or under `~/`), `openai` (the endpoint for coding tools on or off) (all optional) | | `service` |
| `check_updates` | | | `models` (those with `update_available`) |
| `list_actions` | | | `actions` |
| `invoke_action` | `action`, `arguments` (JSON), `confirmed` | | `action_done` (`result`) |
| `serve_actions` | `actions` (specs, below) | `invoke` | `ok` when the request ends |
| `action_result` | `invocation`, `ok` or `error` | | `ok` |

### Updates

`download_model` always fetches the repository's newest revision: the model's
files plus the companions in `sidecars` (`settings.toml`; `mmproj` and `mtp` by
default). The revision goes into use when none of the repository's models is busy
or was used in the last two minutes; until then the old files keep serving.
`check_updates` compares each cached model with the newest revision and sets
`update_available` and `update_size` (bytes to download) on the models whose files
or companions differ. With `model_updates = "auto"` AI Core checks every
`update_check_hours` and downloads updates itself while nothing runs and the
computer is on mains power, pausing otherwise.

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

- `grammar` is a GBNF grammar (llama.cpp's format, up to 64 KB) the reply must
  follow. It can't go with `tools` or `response_format`.
- Images must be `data:image/…;base64,` URLs and need a model with `vision`;
  `input_audio` needs `audio`. Otherwise `unsupported_capability`.
- `priority` is `interactive`, `normal` (default) or `background`.
- `delta` events carry `content`, `reasoning` and `tool_calls`
  (`{index, id?, name?, arguments?}`; parts with the same index concatenate).
- `finished` has `finish_reason`, `usage` (`prompt_tokens`,
  `completion_tokens`) and `timings` (`queued_ms`, `load_ms`, `first_token_ms`,
  `total_ms`, `tokens_per_second`).

### `embed`

`input` items are strings, or `{"image": "<base64>"}` and `{"audio": "<base64>"}`
(the file's bytes, or a `data:` URL; up to 36 MB), which need a model with
`vision` or `audio` (an embedding model with its projector, like EmbeddingGemma 2).
Vectors are unit-length.

`embeddings` has `vectors`, `dimensions` and `identity` (`id`, `file`, `size`,
`revision`, `architecture`, `projector`, and `settings`: the load settings that
change vectors). Store the identity with an index and compare it before mixing
vectors. The model needs `embedding`. See `GLIMPSE.md`.

### Models

A `model` has `id`, `name`, `source` (`cache`, `folder`, `preset`), `path`,
`size`, `architecture`, `quantization`, `parameters`, `context_length`,
`capabilities` (`chat`, `tools`, `thinking`, `vision`, `audio`, `embedding`,
`draft`), `state` (`downloading`, `validating`, `installed`, `loading`, `ready`,
`busy`, `unloading`, `failed`, `missing`), `revision`, `pinned`, `removable`,
`pending_settings`, `error`, `update_available` and `update_size`.

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
| `pooling` | `none`, `mean`, `cls`, `last`, `rank` |
| `batch-size`, `ubatch-size` | 1 to 65536 |
| `image-min-tokens`, `image-max-tokens` | 1 to 16384 |

Other keys already in `~/.config/nyx-ai/models.ini` are kept as they are.
Changes to a loaded model apply when it's next idle (it unloads and the next
request loads it with the new settings); `pending_settings` shows that.

### Actions

The shared action registry: what apps can do for an assistant (or any caller).
Actions come from two places:

- Nyx apps, through nyx-actions (nyx-core `crates/nyx-actions`, contract in its
  `docs/CONTRACT.md`), the registry the `nyx` command and Glimpse link too:
  installed apps' manifests (`nyx/actions/<app-id>.json` in the XDG data folders)
  and running apps' D-Bus control interfaces (`io.github.design_nexus.nyx.Control`
  at the GApplication object path; `Describe`, API 1's `Call`, API 2's `Invoke`),
  asked at most every 10 s. An installed app that isn't running is started by
  D-Bus for the call (`activation: dbus` in its manifest) or listed as unavailable
  until it is opened. An API 1 tool without annotations asks first.
- Providers connected to AI Core: `serve_actions` offers specs for as long as
  the request stays open. Each call arrives as an `invoke` event (`invocation`,
  `action`, `arguments`, `caller`), answered on the same connection with
  `action_result`. Closing the request or the connection withdraws the actions
  and fails calls in flight with `unavailable`. Only trusted programs may serve.

A spec (and an `actions` entry, which adds `owner` (the app's ID, or the serving
program), `source` (`dbus` for a running app, `manifest` for an installed one,
`connection`) and `available`):

```json
{"id": "notes.create", "description": "Create a note.",
 "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
   "required": ["text"]},
 "output_schema": {"type": "object"}, "permissions": ["notes.write"],
 "confirmation": "ask", "side_effects": "writes", "timeout_ms": 30000}
```

- `id` is `app.verb` in lowercase (dots, digits, `-` and `_` allowed after the app).
- `confirmation` is `ask` (default) or `never`; `side_effects` is `read_only`,
  `writes`, `external`, `destructive` or `unknown` (default, treated as `writes`).
- `timeout_ms` defaults to 30 s, at most 300 s.
- An app's `privileged` actions read as `destructive` here. `list_actions` with
  `"detail": true` adds `apps`: nyx-actions catalog entries (the app's name,
  namespaces, aliases, the API it runs, and each action's `risk`, `title`,
  `aliases`, `positional` arguments, `state` and `reason`). An older AI Core refuses
  the field; `nyx-ai-client`'s `list_actions_detail` falls back.

`invoke_action` checks, in order: the action exists (`unknown_action`) and is
available (`unavailable`); an untrusted caller may run only `read_only` actions
(`permission_denied`); an action that asks first needs `confirmed: true`
(`confirmation_required`), which the caller sends only after the person agreed in
its own UI; the arguments match `input_schema` (`invalid_request`, naming the
place). Then it routes the call, naming the caller to the app (`on_behalf_of`).
The owning app checks again (its control helper applies the same rules to AI
Core's call) and its own permissions, and may refuse: its message comes back as
`action_failed`, or `permission_denied` when it says so. AI Core logs the action,
caller, owner, outcome and duration, never the arguments or result.

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
`unknown_action`, `confirmation_required`, `action_failed`, `internal`. The client adds `unavailable` when it can't reach the daemon.

## Changes

- Version 1 (November 2026): first release.
- Version 1, additions: `check_updates`; `Model.update_available` and
  `update_size`; `ServiceSettings.openai_endpoint`, `model_updates` and
  `defaults`; `configure_service`'s `model_updates` and `defaults`.
- Version 1, additions: actions (`list_actions`, `invoke_action`,
  `serve_actions`, `action_result`, the `invoke` event, and the errors
  `unknown_action`, `confirmation_required` and `action_failed`).
- Version 1, additions: `list_actions`' `detail` and the reply's `apps`; installed
  apps' actions (`source: manifest`), started for the call.
- Version 1, additions: image and audio `embed` inputs; `ModelIdentity.projector`
  and `settings`; the settings keys `pooling`, `batch-size`, `ubatch-size`,
  `image-min-tokens` and `image-max-tokens`.
- Version 1, additions: `ChatRequest.grammar`; `configure_service`'s
  `battery_saver`; `ServiceSettings.battery_saver`, `on_battery` and `cpu_only`.
- Version 1, additions: `configure_service`'s `add_model_dirs` (an app's own model
  folder, such as Glimpse's; the models are listed at once, never changed). An
  older daemon rejects the field, so clients treat `invalid_request` as "not
  available".
- Version 1, additions: `configure_service`'s `openai` and
  `ServiceSettings.openai_key_file`. The endpoint is off until `openai` turns it
  on; an older daemon rejects the field.

## Battery saver

With `battery_saver` on, AI Core reads `/sys/class/power_supply` every 15 s. On
battery every model loads with no GPU layers (the saved settings in
`models.ini` stay as they are); models loaded the other way unload once idle and
load the new way on their next request, and the same happens back on mains
power. `service` reports `battery_saver`, `on_battery` and `cpu_only` (models load
on the CPU now). Apps choose shorter replies themselves.

## Default models

`defaults` names a model for each role: `chat` (new chats in apps without a pick of
their own), `assistant`, `embedding` (search and knowledge bases) and `coding` (the
OpenAI-compatible API uses it when a request names no model; `embedding` serves
`/v1/embeddings` the same way). A default must be installed and able to do its role.

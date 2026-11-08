# Nyx AI Core

The shared local model runtime for NyxOS. One daemon, `nyx-aid`, runs llama.cpp
(PrismML's build) for every Nyx app: it keeps the model store, downloads models,
holds load-time settings, budgets memory, schedules requests and loads and
unloads models. Apps keep their own conversations, prompts and indexes and talk
to it through `nyx-ai-client`.

```sh
./install.sh            # build, install, enable the socket (starts on first use)
nyx-ai status           # worker, memory, loaded models
nyx-ai models           # installed models and what they can do
nyx-ai pull org/repo:Q4_K_M
nyx-ai chat org/repo:Q4_K_M "Hello"
```

## Model updates

AI Core checks Hugging Face daily for newer revisions of cached models
(`nyx-ai updates` checks now). With `model_updates = "auto"` (the default) it
downloads them while nothing is running and the computer is on mains power, and
switches a model over once it hasn't been used for two minutes. Set `notify` to
only report updates, or `off`. llama.cpp itself never downloads.

## Coding tools

AI Core serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (loopback
only; systemd starts AI Core on the first request). Point a coding tool there, pick
a model ID from `GET /v1/models`, and give any placeholder key if it wants one.
Requests share the scheduler with the Nyx apps: models load on demand and each
request waits its turn. Set `openai_endpoint = ""` in `settings.toml` to turn it
off (and drop the `127.0.0.1:8080` line from `nyx-aid.socket`).

- [Architecture](docs/ARCHITECTURE.md)
- [Protocol](docs/PROTOCOL.md)
- [PrismML compatibility](docs/PRISM.md)
- [Audit of the runners it replaces](docs/AUDIT.md)

## Files

| Path | What |
|---|---|
| `~/.config/nyx-ai/settings.toml` | `max_loaded`, `idle_unload_seconds`, `model_dirs` |
| `~/.config/nyx-ai/models.ini` | Load-time settings per model (llama.cpp preset format) |
| `~/.local/state/nyx-ai/registry.json` | What AI Core knows about each model |
| `~/.local/share/nyx-ai/models/` | A folder for GGUF files you put there yourself |
| `~/.cache/huggingface/hub/` | The shared model cache downloads go to |
| `~/.local/lib/nyx-ai/llama.cpp/` | The llama.cpp build |
| `$XDG_RUNTIME_DIR/nyx-ai/ai.sock` | The socket apps connect to |

## Using it from an app

```rust
let client = nyx_ai_client::Client::new("nyx-myapp");
let models = client.list_models().await?;
let reply = client.chat(request).await?.collect_chat().await?;
```

Depend on `nyx-ai-client` by Git revision.

## Actions

AI Core keeps one list of what apps can do, for Nyx Assistant and anything else
that acts for the person: `nyx-ai actions` lists it and `nyx-ai act ID JSON`
runs one. Running Nyx apps' D-Bus control interfaces are found on their own;
other programs offer actions with `Client::serve_actions`. AI Core checks the
arguments against each action's schema and asks callers to confirm actions that
change things; the app that owns an action decides whether to do it. See
`docs/PROTOCOL.md`.

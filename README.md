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

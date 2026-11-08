# Inference infrastructure audit

Stage A of Handoff 04, taken on 7 November 2026 from the checkouts in `~/Projects`.
It records what ran local models before Nyx AI Core, what each piece did well, and
what had to move.

## Summary

Before AI Core there were three independent llama.cpp runners, plus a fourth
consumer riding on one of them. None shared models, downloads, memory accounting
or code.

| Consumer | How it ran llama.cpp | Build | Models |
|---|---|---|---|
| Nyx Chat | HTTP to `llama-server` in router mode, a user unit on `127.0.0.1:8080` | PrismML `prism-b10826-c9c34bc`, CUDA 13.3 prebuilt | Hugging Face hub cache |
| Nyx Glimpse | `glimpse-embed`, a C++ shim linked to llama.cpp and mtmd, JSON lines over stdin/stdout | upstream `b11461`, built locally (CPU, CUDA or Vulkan) | `~/.local/share/nyx-glimpse/Models` |
| Nyx Assistant | in-process through `llama-cpp-2` 0.1.158 (`cuda`, `mtmd`) | upstream, built by cargo | `~/.local/share/nyx-assistant/models`, plus scans of LM Studio and the HF cache |
| Nyx Notes | Assistant's `model_generate` / `model_embed` on `assistant.sock` | (Assistant's) | (Assistant's) |

## Nyx Chat (`nyx-chat/apps/chat`)

- **Runner.** `data/nyx-llama-server.service` runs `scripts/llama-server-launch`,
  which reads `llama_models_max` (1, 2, 4) and `llama_idle_seconds` (-1, 300, 900,
  3600) from Chat's `settings.toml` and execs `llama-server --models-max N
  --sleep-idle-seconds S --host 127.0.0.1 --port 8080 --jinja --models-preset
  ~/.config/nyx-chat/llama-models.ini`. The unit is installed by
  `scripts/install-llama.sh`, which downloads and checksums Prism's release into
  `~/.local/lib/nyx-chat/llama.cpp`. The user enables it from Settings.
- **API use** (`src/llama/api.rs`, `src/llama/chat.rs`): `/health`, `/models`
  (`?reload=1`), `/props?model=`, `/models/load`, `/models/unload`,
  `DELETE /models`, `/apply-template` (reasoning-effort probe), `/v1/embeddings`,
  streaming `/v1/chat/completions` with `reasoning_content` and tool-call deltas.
- **Presets** (`src/llama/presets.rs`): Chat owns `ctx-size`, `n-gpu-layers`,
  `flash-attn`, `cache-type-k/v`, `temp`, `top-p`, `top-k`, `min-p`,
  `repeat-penalty` per model and keeps other keys. Changes apply through
  `GET /models?reload=1` once the model is idle.
- **Downloads** (`src/hf/download.rs`, `src/pulls.rs`): into the HF hub cache
  layout (`models--org--repo/{blobs,snapshots,refs}`), revision pinned to the
  repo `sha`, `.incomplete` resume with `Range`, LFS SHA-256 check, split files and
  the matching `mmproj` (Q8_0 preferred). One download at a time, with speed and
  ETA.
- **Resources** (`src/hw.rs`): AMD VRAM from sysfs, NVIDIA from `nvidia-smi` only
  while the card is awake, else a cached value or the BAR1 size; RAM from
  `/proc/meminfo`; a fit estimate including the KV cache.
- **Ollama.** A second backend (`src/ollama/`), also used for remote hosts.
  Ollama is the serde default for old host entries; the default host is llama.cpp.
- **Stays in Chat:** `chat.db` (chats, branching messages, prompts, memories,
  knowledge bases and their vectors, arena), attachments, the HF marketplace
  browsing UI, tools and MCP, voice.

What to preserve: streaming with thinking, cancel, tool calls, JSON output,
vision with `mmproj`, the reasoning-effort probe, embeddings for knowledge bases,
per-model runtime settings, the download queue, Bonsai `PQ*`/`PTQ*` quants and
companion files (`mmproj`, `dflash`, `drafter`).

## Nyx Glimpse (`nyx-glimpse/apps/glimpse`)

- **Runner.** `inference/` builds `glimpse-embed` against llama.cpp `b11461`
  (`scripts/setup-runtime.sh`, or `pkgs/nyx-glimpse.nix` on NixOS). It embeds
  text, images and audio with EmbeddingGemma 2 (`embeddinggemma-2-Q8_0.gguf` plus
  its mmproj), mean pooling, 768 dimensions, L2-normalised.
- **Index identity** is the constant `glimpse-embeddinggemma2-q8-v1`; the models
  are pinned by SHA-256 and HF revision. Changing the model without bumping the
  constant would mix vectors.
- **Lifecycle.** The launcher starts a text-only helper on the first semantic
  query and kills it after 30 s idle. The indexer daemon starts a multimodal helper
  per pass (300 s cap) and pauses on battery or heat.
- **Stays in Glimpse:** the SQLite index, ranking, extraction, plugins and UI.

To move later (Handoff 05): text and multimodal embeddings. The first Prism
pin (`prism-b10826`) couldn't load EmbeddingGemma 2; `prism-b12128` can. Multimodal
embeddings through the router are not verified; see `PRISM.md`.

## Nyx Assistant (`nyx-assistant/apps/assistant`)

- **Runner.** `crates/assistant-inference` (`LlamaCppBackend`) runs a dedicated
  thread over `llama-cpp-2`: GBNF grammars, prefix KV reuse, battery mode (0 GPU
  layers, short replies), a vision projector loaded on first use, hand-written
  Gemma 4 and Qwen 3 prompt formats.
- **Model API** on `$XDG_RUNTIME_DIR/nyx-assistant/assistant.sock`
  (`daemon/nyx-assistantd/src/model_api.rs`, `api_version: 1`), used by Notes.
- **Actions and policy:** MCP-style tool registry with JSON Schema validation,
  `assistant-policy` (capabilities, grants bound to argument digests, budgets,
  audit journal), bubblewrap sandboxes, a pkexec helper with a fixed verb set.
- **Stays in Assistant:** voice (Silero VAD, Nemotron or Parakeet STT, Kokoro
  TTS), personas, planner prompts, approvals, Aura and the HUD.

PrismML: benchmarks only. `docs/BENCHMARKS.md` records that upstream llama.cpp
loads Bonsai 9B but produces nonsense without `prism.hadamard`.

## Shared infrastructure

- `nyx-core`: `paths` works without GTK (`default-features = false`);
  `control` is the `io.github.design_nexus.nyx.Control` D-Bus interface apps expose
  for the Assistant.
- `nyx-peer`: which trusted Nyx program a local caller runs (own folder, or
  `/etc/nyx/trusted-bin` on NixOS; Nix `.NAME-wrapped` binaries).
- `nyx-plugin-protocol`: newline-delimited JSON with `API_VERSION`, the cleanest
  versioned-protocol pattern in the suite.
- `assistant-ipc` and `notes-ai/src/provider.rs`: the NDJSON socket server and a
  hardened client (socket type and mode checks, timeouts, size limits).
- Settings drives services through their CLIs with `--json`
  (`sections/assistant/client.rs`), never by editing their files.

## Packaging

- Arch/Omarchy: each repo's `install.sh`; Chat's fetched Prism's CUDA build.
- NyxOS: `nyx-packages` builds every app with Nix. There is no Prism package and
  every llama.cpp build is CPU-only (`glimpse-llama-cpp`, Assistant's
  `llama-cpp-2`). Chat on NyxOS leaves the server out. `nyxos/docs/handoff.md`
  still has "Chat defaults to llama.cpp, which NyxOS doesn't ship" open.

## Hardware on the development laptop

NVIDIA GeForce RTX 5080 Laptop GPU (16 GB), 30 GB RAM, x86-64 (Alder Lake CPU
variants in the Prism build). The CUDA build is the only GPU backend installed.

## What moved in the first pass

- AI Core (`nyx-aid`) owns the single llama.cpp build (Prism) as a supervised
  router worker on a private Unix socket, the shared model store (the HF hub
  cache plus read-only model folders), downloads, load-time presets, memory
  budgeting, scheduling, leases and idle unloading.
- Chat's local llama.cpp host goes through `nyx-ai-client`; its own unit, launcher
  and Prism installer are gone. Remote llama.cpp hosts and Ollama are untouched.
- Glimpse, Assistant and Notes keep their runners until Handoff 05.

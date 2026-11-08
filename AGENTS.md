# Nyx AI Core

The shared local model runtime: `nyx-aid` (daemon), `nyx-ai` (CLI),
`nyx-ai-proto` (wire types) and `nyx-ai-client` (the client applications use).
See `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md`, `docs/PRISM.md`,
`docs/BENCHMARKS.md` and `docs/RECOVERY.md`; consumers: `docs/GLIMPSE.md` and
`docs/ASSISTANT.md`.

Commands: `cargo build --release --locked`; `cargo test --locked`; `./install.sh`

- One llama.cpp build (Prism, `scripts/install-llama.sh`). Never add a second
  inference engine.
- AI Core owns models, downloads, load-time settings, memory and scheduling.
  Applications own conversations, prompts, attachments, indexes and UI; never move
  those in here.
- The protocol is versioned. Requests are strict (`deny_unknown_fields`). A change
  that an older client can't ignore bumps `API_VERSION`; document it in
  `docs/PROTOCOL.md`.
- Never log prompts, replies, embedding input or attachments.
- Never delete model files outside the shared Hugging Face cache.
- Tests use the stub worker (`nyx-ai-stub-worker`). Do not download models or run
  real inference for routine verification.
- A test copy of the daemon: `NYX_AI_SOCKET` (a private folder; the worker's
  socket and key go next to it), scratch `XDG_CONFIG_HOME`, `XDG_STATE_HOME` and
  `XDG_DATA_HOME`, and a `settings.toml` with `model_updates = "off"` and
  `openai_endpoint` off port 8080. `NYX_AI_POWER_SUPPLY` points it at a fake
  `/sys/class/power_supply`. Clients reach it with the same `NYX_AI_SOCKET`.

Shared dependencies use immutable GitHub revisions.

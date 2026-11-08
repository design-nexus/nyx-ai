# Nyx AI Core

The shared local model runtime: `nyx-aid` (daemon), `nyx-ai` (CLI),
`nyx-ai-proto` (wire types) and `nyx-ai-client` (the client applications use).
See `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md` and `docs/PRISM.md`.

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

Shared dependencies use immutable GitHub revisions.

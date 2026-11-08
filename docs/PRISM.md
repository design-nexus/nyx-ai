# PrismML compatibility

AI Core runs one llama.cpp build: PrismML's fork
([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)), pinned in
`scripts/install-llama.sh`. It is the router worker for every model, not a
second runtime.

## Pinned release

| | |
|---|---|
| Tag | `prism-b12128-d7b2896` (31 October 2026; upstream merged at `d854d98`) |
| Asset | `llama-prism-b12128-d7b2896-bin-linux-cuda-13.3-x64.tar.gz` (204 MB) |
| SHA-256 | `9f3a8d2de2b69936b8ec7c994579ad419cb2918785483d80091b6b2776b880bd` |
| Installed in | `~/.local/lib/nyx-ai/llama.cpp` (`.release` holds the tag) |

## What Nyx relies on that upstream lacks

| Feature | Used by | Upstream status (Nov 2026) |
|---|---|---|
| `PQ2_0` (ggml type 142) and `PTQ1_0` (type 143) ternary packings | Ternary Bonsai 2 (9B, 27B) | Not in upstream b12199 (7 Nov); it rejects those files as unknown types. Upstream runs Bonsai's plain `Q2_0` file from b11556 on |
| `prism.hadamard` activation transform and its fused CUDA kernels | Ternary Bonsai 2 | Not in upstream for these types |
| MTP heads and DFlash2/DSpark drafters for Bonsai | speculative decoding | Prism only |

The `prism` branch carries about 250 commits over upstream (Nov 2026), so AI Core
pins Prism's releases rather than keeping its own merge.

Everything else AI Core uses is upstream router-mode `llama-server`:
`--models-preset`, `--no-models-autoload`, `--models-max`, `--host *.sock`,
`--api-key-file`, `/models`, `/models/load`, `/models/unload`, `DELETE /models`,
`/props?model=`, `/apply-template`, `/v1/chat/completions` (SSE, tools,
`reasoning_content`, `response_format`, image parts with `mmproj`) and
`/v1/embeddings`.

## Checking upstream parity before dropping the fork

Do not switch because a feature has a similar name upstream. Switch only when
an upstream build passes all of these with the same models:

1. `nyx-ai pull prism-ml/Ternary-Bonsai-2-9B-gguf:PQ2_0` (or the model already
   in the cache) loads, and `nyx-ai chat` gives coherent answers to a fixed set of
   prompts, matching the Prism build's answers at temperature 0.
2. The same for `PTQ1_0` and the 27B model.
3. Tool-call formatting: a request with one tool produces a well-formed
   `tool_calls` delta.
4. Context: a 16k-token prompt is answered without errors.
5. Memory: `nyx-ai status` shows no more than 10% more memory than the Prism
   build for the same model and context.
6. Speed: tokens per second within 10% of the Prism build.
7. Gemma 4 E4B (vision with `mmproj`) and EmbeddingGemma behave the same.

Record the results here with the upstream tag tested.

## Updating the Prism pin

1. `scripts/check-llama.sh` lists newer releases with each asset's SHA-256.
2. Update `tag`, `asset` and `digest` in `scripts/install-llama.sh`.
3. Run checks 1 to 7 above against the new build.
4. Reinstall (`./install.sh`); the script replaces the build only when the tag
   changes (into a fresh folder, swapped in). The installer restarts
   `nyx-aid.socket`, so the next request starts the new worker.

## Verified through AI Core (RTX 5080 Laptop, CUDA)

`prism-b12128-d7b2896`, 7 November 2026:

| Model | Check | Result |
|---|---|---|
| Gemma 4 E2B Q4_K_M + BF16 mmproj (folder) | chat, tool call, image | all correct; about 190-210 tokens/s |
| Ternary Bonsai 2 9B `PQ2_0` (cache) | chat at temperature 0 | correct ("Paris"); about 100 tokens/s |
| EmbeddingGemma 2 Q8_0 (Glimpse's folder) | embed three texts | 768 dimensions; cat/kitten 0.94, cat/tax 0.53 |

`prism-b10826-c9c34bc`, the earlier pin:

| Model | Check | Result |
|---|---|---|
| Gemma 4 E2B Q4_K_M + BF16 mmproj (folder) | chat, tool call, image, JSON output | all correct; load 7.2 s, about 160-190 tokens/s |
| Ternary Bonsai 2 9B `PQ2_0` (cache) | chat at temperature 0 | correct ("Paris"); load 3.4 s, about 70 tokens/s |
| EmbeddingGemma 2 Q8_0 (Glimpse's folder) | embed | **fails**: `unknown model architecture: 'gemma-embedding2'` |

## Offline worker

The worker runs with `LLAMA_ARG_OFFLINE=1`, which every model instance
inherits. Without it, loading a cache model through the router's `--hf-repo`
path contacts Hugging Face: it moves `refs/main` to the repository's newest
revision and downloads missing sidecars (projectors, drafters) as part of the
load. AI Core pins revisions and downloads only through `download_model`, so a
load must use only what's on disk.

## Open questions for Handoff 05

- **Multimodal embeddings.** `prism-b12128` embeds text with EmbeddingGemma 2.
  Glimpse also embeds images and audio through mtmd; whether the router's
  `/v1/embeddings` takes image and audio parts for it is not verified. If not, AI Core gets a second worker
  *kind* built from the same llama.cpp tree (the generalised `glimpse-embed`),
  supervised the same way.
- **GBNF grammars** (Assistant's planner): the router accepts `grammar` in chat
  requests; AI Core's protocol needs a validated `grammar` field.
- **Battery mode** (Assistant): a preset variant with `n-gpu-layers = 0` and a
  short output cap, chosen per request class.
- **NyxOS.** There is no Prism package and Nix builds are CPU-only. The NyxOS
  package needs the Prism tree built with CUDA (unfree, opt-in) or Vulkan, chosen
  from `nyxos/modules/nixos/hardware.nix`.

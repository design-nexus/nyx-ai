# Benchmarks

Measured through AI Core's socket (the timings in each `finished` result), so
they include the scheduler and router, not just llama.cpp.

## RTX 5080 Laptop (16 GB), CUDA, `prism-b12128-d7b2896`, 8 November 2026

Defaults: `n-gpu-layers = auto`, the model's own context, flash attention
`auto`. Chat requests: one fixed 20-word prompt, temperature 0, 256 tokens,
thinking off. GPU memory is `nvidia-smi` with the model loaded minus with
nothing loaded (1.6 GB of desktop use).

| Model | Size | Cold load | First token, new prompt | First token, repeated prompt | Tokens/s | GPU memory |
|---|---|---|---|---|---|---|
| Gemma 4 E2B Q4_K_M + BF16 projector (folder) | 3.4 GB | 3.1 to 5.7 s | 165 ms | 19 ms | 158 | 3.0 GB |
| Gemma 4 E4B QAT Q4_0 (cache) | 5.2 GB | 6.7 to 9.3 s | 186 ms | 22 ms | 86 | 4.4 GB |
| Ternary Bonsai 2 9B PQ2_0 (cache) | 2.4 GB | 2.8 to 4.2 s | 201 ms | 39 ms | 86 | 3.0 GB |

- Cold load is from the request to the model being ready. The higher figure is
  with the file not yet in the page cache, the lower with it cached.
- "Repeated prompt" reuses llama.cpp's prompt cache for the slot; the first
  figure is a fresh prompt.

### Switching

Alternating requests between Gemma 4 E2B and Bonsai 9B (64 tokens each), after
one warm-up round:

| Models kept loaded | Load per request (median) | Generation (median) |
|---|---|---|
| 1 | 1.8 s | 0.6 s |
| 2 | 0 s | 0.6 s |

With `max_loaded = 1` every switch unloads one model and loads the other once
the first has no requests left. Two fit together in 16 GB, so on this GPU
`max_loaded = 2` removes the switch cost for a chat model and the Assistant's.

### Embeddings

EmbeddingGemma 2 Q8_0 with its Q8_0 projector, `pooling = mean`,
`ctx-size`/`batch-size`/`ubatch-size = 2048`, images 140 to 280 tokens:

| Input | Throughput |
|---|---|
| Short texts (256 in one request, about 20 words each) | 305 per second |
| Images (4K and 8K JPEG wallpapers, 0.7 to 2.8 MB) | 2.6 per second |
| GPU memory | 1.4 GB |

Image throughput is mostly decoding and resizing the large JPEGs; smaller
photos go faster.

### CPU (battery saver)

Gemma 4 E2B with `n-gpu-layers = 0` (battery saver on battery): load 6.6 s,
first token about 0.3 s for a short prompt, 16 to 19 tokens per second.

## Reproducing

Run a test copy of the daemon with its own socket and settings (see
`AGENTS.md`), then time requests with `nyx-ai chat` and `nyx-ai embed`, or read
`timings` from the `finished` results over the socket. Check `nvidia-smi` before
and after loading. Record the GPU, driver, llama.cpp tag and date with the
results here.

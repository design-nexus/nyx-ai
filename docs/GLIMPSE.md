# Semantic search on AI Core (Glimpse)

Glimpse embeds text, images and audio into one vector space with EmbeddingGemma 2
and its projector. AI Core's worker produces the same vectors as Glimpse's own
runtime, so Glimpse can drop its llama.cpp build and call AI Core without
rebuilding its index.

## Verified

`prism-b12128-d7b2896` through AI Core's router, against Glimpse's
`glimpse-embed` (CPU), with the settings below, 8 November 2026:

| Input | Cosine, AI Core vs Glimpse |
|---|---|
| Text ("a glowing face in the dark") | 1.0000 |
| Image (JPEG wallpaper) | 0.9999 |
| Image (another JPEG wallpaper) | 0.9998 |
| Audio (WAV, 1.5 s) | 1.0000 |

Vectors come back unit-length. The two wallpapers score 0.97 with each other and
about 0.58 with the text and the audio.

## Model

Either keep Glimpse's folder and add it to AI Core's `model_dirs`
(`~/.config/nyx-ai/settings.toml`; folder models are read-only and never
deleted), or download into the shared cache:

```sh
nyx-ai pull ggml-org/embeddinggemma-2-GGUF:Q8_0
```

The projector (`mmproj-embeddinggemma-2-Q8_0.gguf`) comes with it. AI Core lists
the model with `embedding`, `vision` and `audio`.

## Settings

These load settings make the vectors match Glimpse's. Set them once with
`configure_model` (Glimpse is a trusted program); they apply on the next load:

| Key | Value | Why |
|---|---|---|
| `pooling` | `mean` | Glimpse mean-pools |
| `ctx-size`, `batch-size`, `ubatch-size` | `2048` | One input up to 2048 tokens in a single non-causal batch |
| `image-min-tokens`, `image-max-tokens` | `140`, `280` | The image resolution the projector sees; other values give other vectors |

```sh
nyx-ai settings embeddinggemma-2-Q8_0 pooling=mean ctx-size=2048 batch-size=2048 \
  ubatch-size=2048 image-min-tokens=140 image-max-tokens=280
```

## Calls

```rust
use nyx_ai_client::proto::{EmbedInput, Priority};
let client = nyx_ai_client::Client::new("nyx-glimpse");
let inputs = vec![
    EmbedInput::from("receipt from the hardware store"),
    EmbedInput::Image { image: base64_of_file },
    EmbedInput::Audio { audio: base64_of_file },
];
let e = client.embed_inputs(model, inputs, Priority::Background).await?;
```

- Index with `Priority::Background`; queries with `Priority::Interactive`.
  With one model loaded at a time, indexing waits for chats and the scheduler
  swaps models when nothing else needs them.
- Images and audio are the file's bytes in base64 (or a `data:` URL), up to
  36 MB each. Video isn't embedded; take frames and send them as images.
- Text over 2048 tokens fails with the worker's error; split it first.

## Identity

Store `e.identity` with the index and compare it before adding vectors or
searching. Rebuild when any of these change: `file`, `size`, `revision`,
`architecture`, `projector` or `settings` (`pooling`, `image-min-tokens`,
`image-max-tokens`). Glimpse's current index (`glimpse-embeddinggemma2-q8-v1`)
corresponds to:

```json
{"file": "embeddinggemma-2-Q8_0.gguf", "size": 309855456,
 "architecture": "gemma-embedding2", "projector": "mmproj-embeddinggemma-2-Q8_0.gguf",
 "settings": {"image-max-tokens": "280", "image-min-tokens": "140", "pooling": "mean"}}
```

Glimpse should name its model itself rather than follow the `embedding` default
role: a default someone changes in Settings would otherwise make new vectors
that don't mix with the index.

## What stays in Glimpse

The index, ranking, file reading and decoding, thumbnails, video frames, and
the choice of model. AI Core owns the model files, llama.cpp, loading and
scheduling.

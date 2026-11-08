//! Nyx AI Core's daemon: see `docs/ARCHITECTURE.md`.

pub mod config;
pub mod diagnostics;
pub mod downloads;
pub mod gguf;
pub mod hf;
pub mod openai;
pub mod paths;
pub mod presets;
pub mod registry;
pub mod resources;
pub mod scheduler;
pub mod server;
pub mod service;
pub mod stream;
pub mod worker;

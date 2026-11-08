//! Nyx AI Core's daemon: see `docs/ARCHITECTURE.md`.

pub mod actions;
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
#[cfg(test)]
pub(crate) mod testhub;
pub mod updates;
pub mod worker;

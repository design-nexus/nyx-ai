//! `~/.config/nyx-ai/settings.toml`: how many models stay loaded, how long an
//! idle one stays, and which folders hold models.

use crate::paths::{self, Paths};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MAX_LOADED_LIMIT: u32 = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Models loaded at once (1 to 8).
    pub max_loaded: u32,
    /// Seconds before an idle, unpinned model unloads; -1 never.
    pub idle_unload_seconds: i64,
    /// Folders AI Core lists models from and never changes.
    pub model_dirs: Vec<String>,
    /// A llama-server to use instead of the installed build.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    /// Where coding tools reach the OpenAI-compatible API (loopback only); empty
    /// turns it off. `nyx-aid.socket` listens on the same address.
    pub openai_endpoint: String,
    /// `auto` downloads newer revisions of cached models while idle on AC power,
    /// `notify` only reports them, `off` doesn't check.
    pub model_updates: String,
    pub update_check_hours: u64,
    /// Companion files downloads include: `mmproj`, `mtp`, `dflash`, `eagle3`, `dspark`.
    pub sidecars: Vec<String>,
    /// Default models by role (`chat`, `assistant`, `embedding`, `coding`).
    pub defaults: std::collections::BTreeMap<String, String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_loaded: 1,
            idle_unload_seconds: 900,
            model_dirs: vec!["~/.local/share/nyx-ai/models".into(), "~/.local/share/nyx-assistant/models".into()],
            worker: None,
            openai_endpoint: "127.0.0.1:8080".into(),
            model_updates: "auto".into(),
            update_check_hours: 24,
            sidecars: vec!["mmproj".into(), "mtp".into()],
            defaults: Default::default(),
        }
    }
}

impl Settings {
    /// Read the settings; the first time, start from Chat's llama.cpp choices.
    pub fn load(paths: &Paths) -> Settings {
        let file = paths.settings();
        if let Ok(text) = std::fs::read_to_string(&file) {
            match toml::from_str::<Settings>(&text) {
                Ok(settings) => return settings.clamped(),
                Err(e) => {
                    tracing::warn!(error = %e, "settings.toml is unreadable; using defaults");
                    let _ = std::fs::rename(&file, file.with_extension("toml.bad"));
                }
            }
        }
        let settings = Settings::imported(paths).clamped();
        if let Err(e) = settings.save(paths) {
            tracing::warn!(error = %e, "couldn't write settings.toml");
        }
        settings
    }

    fn imported(paths: &Paths) -> Settings {
        let mut settings = Settings::default();
        let chat = paths.config_home.join("nyx-chat/settings.toml");
        if let Ok(text) = std::fs::read_to_string(chat)
            && let Ok(value) = toml::from_str::<toml::Value>(&text)
        {
            if let Some(n) = value.get("llama_models_max").and_then(|v| v.as_integer()) {
                settings.max_loaded = n.clamp(1, MAX_LOADED_LIMIT as i64) as u32;
            }
            if let Some(n) = value.get("llama_idle_seconds").and_then(|v| v.as_integer()) {
                settings.idle_unload_seconds = n;
            }
        }
        settings
    }

    pub fn clamped(mut self) -> Settings {
        self.max_loaded = self.max_loaded.clamp(1, MAX_LOADED_LIMIT);
        if self.idle_unload_seconds < -1 {
            self.idle_unload_seconds = -1;
        }
        if !matches!(self.model_updates.as_str(), "auto" | "notify" | "off") {
            self.model_updates = "auto".into();
        }
        self.update_check_hours = self.update_check_hours.clamp(1, 24 * 30);
        self.sidecars.retain(|k| crate::hf::SIDECARS.contains(&k.as_str()));
        self.sidecars.dedup();
        self.defaults.retain(|role, model| nyx_ai_proto::ROLES.contains(&role.as_str()) && nyx_ai_proto::valid_model_id(model).is_ok());
        self
    }

    pub fn save(&self, paths: &Paths) -> anyhow::Result<()> {
        let text = format!(
            "# Nyx AI Core. max_loaded: models loaded at once. idle_unload_seconds: -1 keeps them.\n{}",
            toml::to_string(self)?
        );
        paths::write_atomic(&paths.settings(), text.as_bytes())?;
        Ok(())
    }

    /// The OpenAI-compatible endpoint's address, if it's on and on loopback.
    pub fn openai_addr(&self) -> Option<std::net::SocketAddr> {
        let addr: std::net::SocketAddr = self.openai_endpoint.trim().parse().ok()?;
        addr.ip().is_loopback().then_some(addr)
    }

    pub fn model_dirs(&self) -> Vec<PathBuf> {
        self.model_dirs.iter().map(|d| paths::expand(d)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_chat_choices_once() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        std::fs::create_dir_all(paths.config_home.join("nyx-chat")).unwrap();
        std::fs::write(paths.config_home.join("nyx-chat/settings.toml"), "llama_models_max = 2\nllama_idle_seconds = -1\n").unwrap();
        let s = Settings::load(&paths);
        assert_eq!((s.max_loaded, s.idle_unload_seconds), (2, -1));
        std::fs::write(paths.config_home.join("nyx-chat/settings.toml"), "llama_models_max = 4\n").unwrap();
        assert_eq!(Settings::load(&paths).max_loaded, 2);
    }

    #[test]
    fn openai_endpoint_stays_on_loopback() {
        let mut s = Settings::default();
        assert_eq!(s.openai_addr().unwrap().to_string(), "127.0.0.1:8080");
        s.openai_endpoint = "0.0.0.0:8080".into();
        assert!(s.openai_addr().is_none());
        s.openai_endpoint = String::new();
        assert!(s.openai_addr().is_none());
    }

    #[test]
    fn clamps() {
        let s = Settings { max_loaded: 99, idle_unload_seconds: -7, ..Default::default() }.clamped();
        assert_eq!((s.max_loaded, s.idle_unload_seconds), (MAX_LOADED_LIMIT, -1));
        let s = Settings { model_updates: "sometimes".into(), update_check_hours: 0, sidecars: vec!["mtp".into(), "x".into()], ..Default::default() }
            .clamped();
        assert_eq!((s.model_updates.as_str(), s.update_check_hours, s.sidecars.as_slice()), ("auto", 1, &["mtp".to_string()][..]));
        let old: Settings = toml::from_str("max_loaded = 2\n").unwrap();
        assert_eq!((old.model_updates.as_str(), old.sidecars.len()), ("auto", 2));
    }
}

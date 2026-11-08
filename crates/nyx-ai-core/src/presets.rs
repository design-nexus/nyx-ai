//! Load-time settings per model: `~/.config/nyx-ai/models.ini`, in the router's
//! preset format. AI Core changes only the keys it validates and keeps every
//! other line as it is.

use crate::paths::{self, Paths};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const DEFAULTS: &str = "version = 1\n\n[*]\nctx-size = 8192\nn-gpu-layers = auto\nflash-attn = auto\n";

/// The settings `configure_model` accepts, by their canonical name.
pub const KEYS: &[&str] = &[
    "ctx-size",
    "n-gpu-layers",
    "flash-attn",
    "cache-type-k",
    "cache-type-v",
    "temp",
    "top-p",
    "top-k",
    "min-p",
    "repeat-penalty",
    "parallel",
    "embeddings",
    "pooling",
    "batch-size",
    "ubatch-size",
    "image-min-tokens",
    "image-max-tokens",
];

/// Settings that change what a load allocates (the rest are sampling defaults).
pub const RUNTIME_KEYS: &[&str] = &[
    "ctx-size",
    "n-gpu-layers",
    "flash-attn",
    "cache-type-k",
    "cache-type-v",
    "parallel",
    "embeddings",
    "pooling",
    "batch-size",
    "ubatch-size",
    "image-min-tokens",
    "image-max-tokens",
];

/// Settings that change what an embedding model's vectors are.
pub const VECTOR_KEYS: &[&str] = &["pooling", "image-min-tokens", "image-max-tokens"];

pub fn canonical(key: &str) -> &str {
    match key.trim() {
        "c" | "ctx-size" | "LLAMA_ARG_CTX_SIZE" => "ctx-size",
        "ngl" | "gpu-layers" | "n-gpu-layers" | "LLAMA_ARG_N_GPU_LAYERS" => "n-gpu-layers",
        "fa" | "flash-attn" | "LLAMA_ARG_FLASH_ATTN" => "flash-attn",
        "ctk" | "cache-type-k" => "cache-type-k",
        "ctv" | "cache-type-v" => "cache-type-v",
        "temp" | "temperature" => "temp",
        "np" | "parallel" | "LLAMA_ARG_N_PARALLEL" => "parallel",
        "embedding" | "embeddings" | "LLAMA_ARG_EMBEDDINGS" => "embeddings",
        "b" | "batch-size" | "LLAMA_ARG_BATCH" => "batch-size",
        "ub" | "ubatch-size" | "LLAMA_ARG_UBATCH" => "ubatch-size",
        "pooling" | "LLAMA_ARG_POOLING" => "pooling",
        other => other,
    }
}

const KV_TYPES: &[&str] = &["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];

/// Check one setting; returns the canonical key and normalised value.
pub fn validate(key: &str, value: &str) -> Result<(String, String), String> {
    let key = canonical(key);
    let value = value.trim();
    let number = |lo: f64, hi: f64| -> Result<String, String> {
        match value.parse::<f64>() {
            Ok(v) if (lo..=hi).contains(&v) => Ok(value.to_string()),
            _ => Err(format!("{key} must be a number from {lo} to {hi}")),
        }
    };
    let integer = |lo: i64, hi: i64| -> Result<String, String> {
        match value.parse::<i64>() {
            Ok(v) if (lo..=hi).contains(&v) => Ok(v.to_string()),
            _ => Err(format!("{key} must be a whole number from {lo} to {hi}")),
        }
    };
    let value = match key {
        "ctx-size" => integer(0, 1 << 20)?,
        "n-gpu-layers" if matches!(value, "auto" | "all") => value.to_string(),
        "n-gpu-layers" => integer(0, 999)?,
        "flash-attn" if matches!(value, "on" | "off" | "auto") => value.to_string(),
        "flash-attn" => return Err("flash-attn must be on, off or auto".into()),
        "cache-type-k" | "cache-type-v" if KV_TYPES.contains(&value) => value.to_string(),
        "cache-type-k" | "cache-type-v" => return Err(format!("{key} must be one of {}", KV_TYPES.join(", "))),
        "temp" | "repeat-penalty" => number(0.0, 5.0)?,
        "top-p" | "min-p" => number(0.0, 1.0)?,
        "top-k" => integer(0, 1000)?,
        "parallel" => integer(1, 16)?,
        "embeddings" if matches!(value, "true" | "false") => value.to_string(),
        "embeddings" => return Err("embeddings must be true or false".into()),
        "pooling" if matches!(value, "none" | "mean" | "cls" | "last" | "rank") => value.to_string(),
        "pooling" => return Err("pooling must be none, mean, cls, last or rank".into()),
        "batch-size" | "ubatch-size" => integer(1, 65536)?,
        "image-min-tokens" | "image-max-tokens" => integer(1, 16384)?,
        other => return Err(format!("{other} isn't a setting AI Core manages")),
    };
    Ok((key.to_string(), value))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ini {
    /// `("", header lines)`, then `(name, lines including "[name]")`.
    sections: Vec<(String, Vec<String>)>,
}

fn key_value(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with([';', '#', '[']) {
        return None;
    }
    let (key, value) = line.split_once('=')?;
    Some((key.trim(), value.trim()))
}

impl Ini {
    pub fn parse(text: &str) -> Ini {
        let mut sections: Vec<(String, Vec<String>)> = vec![(String::new(), Vec::new())];
        for line in text.lines() {
            if let Some(name) = line.trim().strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                sections.push((name.trim().to_string(), vec![line.to_string()]));
            } else {
                sections.last_mut().expect("header section").1.push(line.to_string());
            }
        }
        Ini { sections }
    }

    pub fn render(&self) -> String {
        let mut lines: Vec<&str> = self.sections.iter().flat_map(|(_, lines)| lines.iter().map(String::as_str)).collect();
        while lines.last().is_some_and(|l| l.trim().is_empty()) {
            lines.pop();
        }
        let mut text = lines.join("\n");
        text.push('\n');
        text
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.sections.iter().skip(1).map(|(name, _)| name.as_str())
    }

    pub fn section(&self, name: &str) -> Option<&[String]> {
        self.sections.iter().skip(1).find(|(n, _)| n == name).map(|(_, lines)| lines.as_slice())
    }

    /// The keys a section sets, by canonical name.
    pub fn values(&self, name: &str) -> BTreeMap<String, String> {
        self.section(name)
            .into_iter()
            .flatten()
            .filter_map(|l| key_value(l))
            .map(|(k, v)| (canonical(k).to_string(), v.to_string()))
            .collect()
    }

    /// What a load of `name` uses: `[*]` overlaid with the model's own keys.
    pub fn effective(&self, name: &str) -> BTreeMap<String, String> {
        let mut all = self.values("*");
        if name != "*" {
            all.extend(self.values(name));
        }
        all
    }

    /// Apply validated changes; an empty value removes the key.
    pub fn set(&mut self, name: &str, changes: &BTreeMap<String, String>) -> Result<(), String> {
        if name != "*" {
            nyx_ai_proto::valid_model_id(name)?;
        }
        let mut checked = Vec::new();
        for (key, value) in changes {
            if value.trim().is_empty() {
                let key = canonical(key);
                if !KEYS.contains(&key) {
                    return Err(format!("{key} isn't a setting AI Core manages"));
                }
                checked.push((key.to_string(), None));
            } else {
                let (key, value) = validate(key, value)?;
                checked.push((key, Some(value)));
            }
        }
        if !self.sections[0].1.iter().any(|l| key_value(l).is_some_and(|(k, _)| k == "version")) {
            self.sections[0].1.insert(0, "version = 1".into());
        }
        let index = match self.sections.iter().skip(1).position(|(n, _)| n == name) {
            Some(i) => i + 1,
            None => {
                if let Some(last) = self.sections.last_mut()
                    && last.1.last().is_some_and(|l| !l.trim().is_empty())
                {
                    last.1.push(String::new());
                }
                self.sections.push((name.to_string(), vec![format!("[{name}]")]));
                self.sections.len() - 1
            }
        };
        let lines = &mut self.sections[index].1;
        for (key, value) in checked {
            lines.retain(|l| key_value(l).is_none_or(|(k, _)| canonical(k) != key));
            if let Some(value) = value {
                let at = lines.iter().rposition(|l| !l.trim().is_empty()).map_or(lines.len(), |i| i + 1);
                lines.insert(at, format!("{key} = {value}"));
            }
        }
        Ok(())
    }

    fn lines_mut(&mut self, name: &str) -> &mut Vec<String> {
        if let Some(i) = self.sections.iter().skip(1).position(|(n, _)| n == name) {
            return &mut self.sections[i + 1].1;
        }
        self.sections.push((name.to_string(), vec![String::new(), format!("[{name}]")]));
        &mut self.sections.last_mut().expect("just pushed").1
    }

    fn replace_section(&mut self, name: &str, lines: Option<&Vec<String>>) {
        match (self.sections.iter().skip(1).position(|(n, _)| n == name), lines) {
            (Some(i), Some(lines)) => self.sections[i + 1].1 = lines.clone(),
            (Some(i), None) => {
                self.sections.remove(i + 1);
            }
            (None, Some(lines)) => self.sections.push((name.to_string(), lines.clone())),
            (None, None) => {}
        }
    }
}

/// A model found in a folder, which the router learns about from its preset.
#[derive(Clone, Debug, PartialEq)]
pub struct FolderEntry {
    pub id: String,
    pub path: PathBuf,
    pub mmproj: Option<PathBuf>,
}

/// The file the router reads: `models.ini`, with sections whose changes wait for
/// a busy model put back as they were (`held`), folder models added, and
/// embedding models switched to embeddings.
pub fn router_ini(
    ini: &Ini,
    held: &BTreeMap<String, Option<Vec<String>>>,
    folders: &[FolderEntry],
    embedding: &BTreeSet<String>,
) -> String {
    let mut out = ini.clone();
    for (name, old) in held {
        out.replace_section(name, old.as_ref());
    }
    for entry in folders {
        let has_model = out.values(&entry.id).contains_key("model");
        let lines = out.lines_mut(&entry.id);
        if !has_model {
            lines.push(format!("model = {}", entry.path.display()));
            if let Some(mmproj) = &entry.mmproj {
                lines.push(format!("mmproj = {}", mmproj.display()));
            }
        }
    }
    for id in embedding {
        if !out.effective(id).contains_key("embeddings") {
            out.lines_mut(id).push("embeddings = true".into());
        }
    }
    out.render()
}

/// Read `models.ini`; the first time, start from Chat's llama.cpp presets.
pub fn load_or_import(paths: &Paths) -> Ini {
    let file = paths.models_ini();
    if let Ok(text) = std::fs::read_to_string(&file) {
        return Ini::parse(&text);
    }
    let chat = paths.config_home.join("nyx-chat/llama-models.ini");
    let text = std::fs::read_to_string(chat).unwrap_or_else(|_| DEFAULTS.to_string());
    let ini = Ini::parse(&text);
    if let Err(e) = save(paths, &ini) {
        tracing::warn!(error = %e, "couldn't write models.ini");
    }
    ini
}

pub fn save(paths: &Paths, ini: &Ini) -> std::io::Result<()> {
    paths::write_atomic(&paths.models_ini(), ini.render().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changes(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn keeps_other_keys_and_replaces_managed_ones() {
        let mut ini = Ini::parse("version = 1\n; mine\n[*]\nc = 4096\n\n[org/m:Q4_K_M]\nchat-template = chatml\nctx-size = 8192\n");
        ini.set("org/m:Q4_K_M", &changes(&[("ctx-size", "65536"), ("cache-type-k", "q8_0")])).unwrap();
        ini.set("*", &changes(&[("ctx-size", "16384")])).unwrap();
        let text = ini.render();
        assert!(text.contains("chat-template = chatml") && text.contains("; mine"));
        assert!(text.contains("ctx-size = 65536") && !text.contains("ctx-size = 8192"));
        assert!(!text.contains("c = 4096") && text.contains("ctx-size = 16384"));
        assert_eq!(ini.effective("org/m:Q4_K_M")["ctx-size"], "65536");
        assert_eq!(ini.effective("other")["ctx-size"], "16384");
        assert_eq!(Ini::parse(&text).render(), text);
        ini.set("org/m:Q4_K_M", &changes(&[("cache-type-k", "")])).unwrap();
        assert!(!ini.render().contains("cache-type-k"));
    }

    #[test]
    fn validates() {
        assert!(validate("ctx-size", "-5").is_err());
        assert!(validate("n-gpu-layers", "auto").is_ok());
        assert!(validate("cache-type-k", "q3_9").is_err());
        assert!(validate("model", "/etc/passwd").is_err());
        assert!(validate("top-p", "1.5").is_err());
        assert_eq!(validate("c", "4096").unwrap(), ("ctx-size".into(), "4096".into()));
        let mut ini = Ini::parse(DEFAULTS);
        assert!(ini.set("[evil]", &changes(&[("ctx-size", "1")])).is_err());
        assert!(ini.set("m", &changes(&[("model", "/x.gguf")])).is_err());
    }

    #[test]
    fn router_file_holds_folders_and_embeddings() {
        let mut ini = Ini::parse(DEFAULTS);
        ini.set("busy", &changes(&[("ctx-size", "999")])).unwrap();
        let held = [("busy".to_string(), None)].into();
        let folders = [FolderEntry { id: "local".into(), path: "/m/a.gguf".into(), mmproj: Some("/m/mmproj-a.gguf".into()) }];
        let text = router_ini(&ini, &held, &folders, &["embed".to_string()].into());
        assert!(!text.contains("[busy]"));
        assert!(text.contains("[local]\nmodel = /m/a.gguf\nmmproj = /m/mmproj-a.gguf"));
        assert!(text.contains("[embed]\nembeddings = true"));
    }
}

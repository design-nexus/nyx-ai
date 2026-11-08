//! The authoritative model list: the router's `/models` joined with each file's
//! GGUF metadata, models found in folders, and what was known before (so a
//! model whose file went away shows as missing rather than vanishing).

use crate::gguf::{self, Meta};
use crate::hf;
use crate::paths::{self, Paths};
use crate::presets::FolderEntry;
use nyx_ai_proto::{Capability, Source};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One model as the router lists it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterItem {
    pub id: String,
    pub path: Option<PathBuf>,
    pub source: Option<String>,
    /// `unloaded`, `loading`, `loaded`, `sleeping`, `downloading`, `incomplete`.
    pub status: String,
    pub failed: Option<String>,
    pub inputs: Vec<String>,
}

impl RouterItem {
    pub fn parse(item: &Value) -> Option<RouterItem> {
        let id = item["id"].as_str()?.to_string();
        let status = &item["status"];
        let failed = (status["failed"].as_bool() == Some(true)).then(|| match status["exit_code"].as_i64() {
            Some(code) => format!("the model's llama.cpp instance exited with code {code}"),
            None => "the model failed to load".to_string(),
        });
        Some(RouterItem {
            id,
            path: item["path"].as_str().filter(|p| !p.is_empty()).map(PathBuf::from),
            source: item["source"].as_str().map(String::from),
            status: status["value"].as_str().unwrap_or("unloaded").to_string(),
            failed,
            inputs: item["architecture"]["input_modalities"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub source: Source,
    pub path: Option<PathBuf>,
    pub size: u64,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub audio: bool,
    #[serde(default)]
    pub draft: bool,
    /// The projector the worker pairs with it (images and audio).
    #[serde(default)]
    pub projector: Option<PathBuf>,
    #[serde(default)]
    pub error: Option<String>,
    /// Not in the latest listing.
    #[serde(default)]
    pub missing: bool,
}

impl Record {
    pub fn capabilities(&self) -> Vec<Capability> {
        if self.draft {
            return vec![Capability::Draft];
        }
        let mut caps = Vec::new();
        if let Some(meta) = &self.meta {
            if meta.chat() {
                caps.push(Capability::Chat);
                if meta.tools() {
                    caps.push(Capability::Tools);
                }
                if meta.thinking() {
                    caps.push(Capability::Thinking);
                }
            }
            if meta.embedding() {
                caps.push(Capability::Embedding);
            }
        }
        if self.vision {
            caps.push(Capability::Vision);
        }
        if self.audio {
            caps.push(Capability::Audio);
        }
        caps
    }

    pub fn quantization(&self) -> Option<String> {
        self.meta.as_ref().and_then(|m| m.quantization()).map(String::from).or_else(|| {
            let name = self.path.as_ref()?.file_name()?.to_string_lossy().into_owned();
            let tag = hf::quant_tag(&name);
            (tag != name.trim_end_matches(".gguf")).then_some(tag)
        })
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    version: u32,
    models: BTreeMap<String, Record>,
}

type MetaKey = (PathBuf, u64, i64);

pub struct Registry {
    paths: Paths,
    records: Mutex<BTreeMap<String, Record>>,
    metas: Mutex<HashMap<MetaKey, Result<Meta, String>>>,
    folders: Mutex<Vec<FolderEntry>>,
}

impl Registry {
    pub fn load(paths: &Paths) -> Registry {
        let file = paths.registry();
        let records = match std::fs::read(&file) {
            Ok(bytes) => match serde_json::from_slice::<Saved>(&bytes) {
                Ok(saved) => saved.models,
                Err(e) => {
                    tracing::warn!(error = %e, "registry.json is unreadable; rebuilding it");
                    let _ = std::fs::rename(&file, file.with_extension("json.bad"));
                    BTreeMap::new()
                }
            },
            Err(_) => BTreeMap::new(),
        };
        Registry { paths: paths.clone(), records: Mutex::new(records), metas: Mutex::new(HashMap::new()), folders: Mutex::new(Vec::new()) }
    }

    fn meta(&self, path: &Path) -> Result<Meta, String> {
        let stat = std::fs::metadata(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
        let key = (path.to_path_buf(), stat.len(), stat.mtime());
        if let Some(found) = self.metas.lock().expect("meta cache").get(&key) {
            return found.clone();
        }
        let result = gguf::read(path).map_err(|e| format!("unsupported or damaged GGUF file: {e}"));
        self.metas.lock().expect("meta cache").insert(key, result.clone());
        result
    }

    /// Models in the folders, with their projectors paired. Split files list
    /// once, by their first part.
    pub fn scan_folders(&self, dirs: &[PathBuf], cache: &Path) -> Vec<FolderEntry> {
        let cache = std::fs::canonicalize(cache).unwrap_or_else(|_| cache.to_path_buf());
        let mut seen = BTreeSet::new();
        let mut entries: Vec<FolderEntry> = Vec::new();
        let mut walk: Vec<(PathBuf, u32)> = dirs.iter().map(|d| (d.clone(), 0)).collect();
        while let Some((dir, depth)) = walk.pop() {
            let Ok(list) = std::fs::read_dir(&dir) else { continue };
            let mut models = Vec::new();
            let mut projectors = Vec::new();
            let mut files: Vec<PathBuf> = list.flatten().map(|e| e.path()).collect();
            files.sort();
            for path in files {
                if path.is_dir() {
                    if depth < 2 {
                        walk.push((path, depth + 1));
                    }
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if !name.to_ascii_lowercase().ends_with(".gguf") {
                    continue;
                }
                if let Some(pos) = name.find("-of-")
                    && !name[..pos].ends_with("-00001")
                {
                    continue;
                }
                let real = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if real.starts_with(&cache) || !seen.insert(real) {
                    continue;
                }
                match self.meta(&path) {
                    Ok(meta) if meta.projector() => projectors.push(path),
                    Ok(_) => models.push(path),
                    Err(e) => tracing::debug!(path = %path.display(), error = %e, "skipping file"),
                }
            }
            for path in models {
                let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let stem = match stem.find("-00001-of-") {
                    Some(pos) => stem[..pos].to_string(),
                    None => stem,
                };
                let mmproj = pair_projector(&stem, &projectors);
                let mut id: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.+".contains(c) { c } else { '-' }).collect();
                let base = id.clone();
                let mut n = 2;
                while entries.iter().any(|e| e.id == id) {
                    id = format!("{base}-{n}");
                    n += 1;
                }
                entries.push(FolderEntry { id, path, mmproj });
            }
        }
        *self.folders.lock().expect("folders") = entries.clone();
        entries
    }

    pub fn folder_ids(&self) -> BTreeSet<String> {
        self.folders.lock().expect("folders").iter().map(|e| e.id.clone()).collect()
    }

    /// Fold the router's listing in; returns every known model.
    pub fn update(&self, items: &[RouterItem]) -> BTreeMap<String, Record> {
        let folders: HashMap<String, FolderEntry> = self.folders.lock().expect("folders").iter().map(|e| (e.id.clone(), e.clone())).collect();
        let mut fresh = BTreeMap::new();
        for item in items {
            let folder = folders.get(&item.id);
            // A cached model with its own section in models.ini lists as a preset.
            let cached = if folder.is_none() && item.path.is_none() { hf::cached_file(&self.paths.cache, &item.id) } else { None };
            let source = if folder.is_some() {
                Source::Folder
            } else if item.source.as_deref() == Some("preset") && cached.is_none() {
                Source::Preset
            } else {
                Source::Cache
            };
            let path = item.path.clone().or_else(|| folder.map(|f| f.path.clone())).or(cached);
            let mut record = Record {
                source,
                size: path.as_deref().map(total_size).unwrap_or(0),
                revision: path.as_deref().and_then(hf::revision_of),
                path: path.clone(),
                meta: None,
                vision: item.inputs.iter().any(|i| i == "image"),
                audio: item.inputs.iter().any(|i| i == "audio"),
                draft: source == Source::Cache && hf::cached_auxiliary_model(&self.paths.cache, &item.id),
                projector: folder.and_then(|f| f.mmproj.clone()).or_else(|| path.as_deref().and_then(hf::snapshot_projector)),
                error: item.failed.clone(),
                missing: false,
            };
            match path.as_deref().map(|p| self.meta(p)) {
                Some(Ok(meta)) => record.meta = Some(meta),
                Some(Err(e)) if item.status != "downloading" && item.status != "incomplete" => record.error = Some(e),
                _ => {}
            }
            if let Some(mmproj) = folder.and_then(|f| f.mmproj.as_deref())
                && let Ok(meta) = self.meta(mmproj)
            {
                record.vision |= meta.vision_encoder;
                record.audio |= meta.audio_encoder;
            }
            if item.status == "incomplete" {
                record.error = Some("the download didn't finish; download it again to resume".into());
            }
            fresh.insert(item.id.clone(), record);
        }
        let mut records = self.records.lock().expect("records");
        for (id, old) in records.iter() {
            if !fresh.contains_key(id) && old.path.as_deref().is_some_and(|p| !p.exists()) {
                fresh.insert(id.clone(), Record { missing: true, ..old.clone() });
            }
        }
        if *records != fresh {
            *records = fresh.clone();
            let saved = Saved { version: 1, models: fresh.clone() };
            if let Err(e) = paths::write_atomic(&self.paths.registry(), &serde_json::to_vec_pretty(&saved).unwrap_or_default()) {
                tracing::warn!(error = %e, "couldn't save registry.json");
            }
        }
        fresh
    }

    /// The last known list, for when the worker can't be asked.
    pub fn known(&self) -> BTreeMap<String, Record> {
        self.records.lock().expect("records").clone()
    }

    pub fn get(&self, id: &str) -> Option<Record> {
        self.records.lock().expect("records").get(id).cloned()
    }

    pub fn forget(&self, id: &str) {
        let mut records = self.records.lock().expect("records");
        if records.remove(id).is_some() {
            let saved = Saved { version: 1, models: records.clone() };
            let _ = paths::write_atomic(&self.paths.registry(), &serde_json::to_vec_pretty(&saved).unwrap_or_default());
        }
    }
}

/// The projector for a model in the same folder: the only one, or the one whose
/// name shares the most with the model's.
fn pair_projector(stem: &str, projectors: &[PathBuf]) -> Option<PathBuf> {
    if projectors.len() <= 1 {
        return projectors.first().cloned();
    }
    let model = stem.to_ascii_lowercase();
    projectors
        .iter()
        .max_by_key(|p| {
            let name = p.file_stem().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            let name = name.replace("mmproj-", "").replace("-mmproj", "");
            model.chars().zip(name.chars()).take_while(|(a, b)| a == b).count()
        })
        .cloned()
}

/// A file's size, or all parts of a split model.
fn total_size(path: &Path) -> u64 {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    if let (Some(pos), Some(dir)) = (name.find("-00001-of-"), path.parent()) {
        let prefix = &name[..pos];
        return std::fs::read_dir(dir)
            .map(|list| {
                list.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(prefix) && e.file_name().to_string_lossy().contains("-of-"))
                    .filter_map(|e| std::fs::metadata(e.path()).ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0);
    }
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::{TestValue, build, chat_model};

    #[test]
    fn folders_pair_projectors_and_skip_shards() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("gemma-4-E2B-it-Q4_K_M.gguf"), chat_model()).unwrap();
        std::fs::write(
            models.join("mmproj-gemma-4-E2B-it-BF16.gguf"),
            build(&[("general.architecture", TestValue::Str("clip")), ("clip.has_vision_encoder", TestValue::Bool(true))]),
        )
        .unwrap();
        std::fs::write(models.join("big-Q8_0-00001-of-00002.gguf"), chat_model()).unwrap();
        std::fs::write(models.join("big-Q8_0-00002-of-00002.gguf"), b"rest").unwrap();
        std::fs::write(models.join("notes.txt"), b"x").unwrap();
        std::fs::write(models.join("broken.gguf"), b"nope").unwrap();
        let registry = Registry::load(&paths);
        let found = registry.scan_folders(std::slice::from_ref(&models), &paths.cache);
        let ids: Vec<_> = found.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["big-Q8_0", "gemma-4-E2B-it-Q4_K_M"]);
        assert!(found[1].mmproj.as_ref().unwrap().ends_with("mmproj-gemma-4-E2B-it-BF16.gguf"));

        let items = vec![RouterItem { id: "gemma-4-E2B-it-Q4_K_M".into(), source: Some("preset".into()), status: "unloaded".into(), ..Default::default() }];
        let records = registry.update(&items);
        let r = &records["gemma-4-E2B-it-Q4_K_M"];
        assert_eq!(r.source, Source::Folder);
        assert_eq!(r.capabilities(), [Capability::Chat, Capability::Tools, Capability::Thinking, Capability::Vision]);
        assert_eq!(r.quantization().as_deref(), Some("Q4_K_M"));
        assert!(paths.registry().exists());
    }

    #[test]
    fn remembers_missing_models_and_survives_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        let file = dir.path().join("m.gguf");
        std::fs::write(&file, chat_model()).unwrap();
        let registry = Registry::load(&paths);
        registry.update(&[RouterItem { id: "m".into(), path: Some(file.clone()), status: "unloaded".into(), ..Default::default() }]);
        std::fs::remove_file(&file).unwrap();
        let records = Registry::load(&paths).update(&[]);
        assert!(records["m"].missing);
        std::fs::write(paths.registry(), b"{oops").unwrap();
        assert!(Registry::load(&paths).known().is_empty());
        assert!(paths.registry().with_extension("json.bad").exists());
    }

    #[test]
    fn parses_router_items() {
        let item = serde_json::json!({"id": "o/r:Q4_K_M", "path": "/c/x.gguf", "source": "cache",
            "status": {"value": "unloaded", "failed": true, "exit_code": 1},
            "architecture": {"input_modalities": ["text", "image"]}});
        let parsed = RouterItem::parse(&item).unwrap();
        assert_eq!(parsed.status, "unloaded");
        assert!(parsed.failed.unwrap().contains("code 1"));
        assert_eq!(parsed.inputs, ["text", "image"]);
    }
}

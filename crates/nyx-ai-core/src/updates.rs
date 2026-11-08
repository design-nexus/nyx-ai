//! Newer revisions of cached models. A check compares what a model would
//! download today (its files and companions at the repository's newest
//! revision) with what its snapshot links to now.

use crate::downloads::{self, File, current_revision, parse_model, selected_files, snapshot_links};
use crate::hf::repo_dir;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct Update {
    pub revision: String,
    /// Bytes to download (0 when the blobs are already here).
    pub size: u64,
    /// Files changed or companions were added; false when only other files
    /// (a README) changed, which needs no download.
    pub content: bool,
}

/// What `quant` needs to match `tree`: `None` if its files and companions are
/// already linked in the current snapshot.
pub fn needed(cache: &Path, repo: &str, current: Option<&str>, tree: &[File], quant: &str, sidecars: &[String]) -> Result<Option<u64>, String> {
    let files = selected_files(tree, quant, sidecars)?;
    let linked = current.map(|r| snapshot_links(cache, repo, r)).unwrap_or_default();
    let blobs = repo_dir(cache, repo).join("blobs");
    let mut missing = false;
    let mut size = 0;
    for f in &files {
        let oid = &f.lfs.as_ref().expect("selected files have checksums").oid;
        if linked.get(&f.path) != Some(oid) {
            missing = true;
            if std::fs::metadata(blobs.join(oid)).map(|m| m.len()).ok() != Some(f.size) {
                size += f.size;
            }
        }
    }
    Ok(missing.then_some(size))
}

/// Whether the computer runs on mains power. One without a battery always does.
pub fn on_ac(power_supply: &Path) -> bool {
    let read = |dir: &Path, name: &str| std::fs::read_to_string(dir.join(name)).map(|s| s.trim().to_string()).unwrap_or_default();
    let mut battery = false;
    for entry in std::fs::read_dir(power_supply).into_iter().flatten().flatten() {
        let dir = entry.path();
        match read(&dir, "type").as_str() {
            "Battery" => battery = true,
            "Mains" | "USB" if read(&dir, "online") == "1" => return true,
            _ => {}
        }
    }
    !battery
}

#[derive(Default)]
pub struct Updates {
    found: Mutex<BTreeMap<String, Update>>,
    /// Downloaded revisions waiting for their repository's models to be free.
    staged: Mutex<BTreeMap<String, String>>,
    pub last_check: Mutex<Option<Instant>>,
    pub checking: std::sync::atomic::AtomicBool,
}

impl Updates {
    /// Check `models` (cached `org/repo:QUANT` IDs) against Hugging Face.
    pub async fn check(&self, cache: &Path, endpoint: &str, token: Option<&str>, sidecars: &[String], models: &[String]) -> BTreeMap<String, Update> {
        let mut by_repo: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        for model in models {
            if let Ok((repo, quant)) = parse_model(model) {
                by_repo.entry(repo).or_default().push((model.clone(), quant));
            }
        }
        let Ok(client) = downloads::client() else { return BTreeMap::new() };
        let mut found = BTreeMap::new();
        for (repo, models) in by_repo {
            let (sha, tree) = match downloads::latest(&client, endpoint, token, &repo).await {
                Ok(latest) => latest,
                Err(e) => {
                    tracing::debug!(%repo, error = %e, "couldn't check for updates");
                    continue;
                }
            };
            let current = current_revision(cache, &repo);
            if current.as_deref() == Some(sha.as_str()) {
                continue;
            }
            for (model, quant) in models {
                match needed(cache, &repo, current.as_deref(), &tree, &quant, sidecars) {
                    Ok(Some(size)) => {
                        found.insert(model, Update { revision: sha.clone(), size, content: true });
                    }
                    Ok(None) => {
                        found.insert(model, Update { revision: sha.clone(), size: 0, content: false });
                    }
                    Err(e) => tracing::debug!(%model, error = %e, "can't compare with the newest revision"),
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        *self.found.lock().expect("updates") = found.clone();
        *self.last_check.lock().expect("updates") = Some(Instant::now());
        found
    }

    pub fn get(&self, model: &str) -> Option<Update> {
        let update = self.found.lock().expect("updates").get(model).cloned()?;
        let repo = parse_model(model).ok()?.0;
        (!self.staged.lock().expect("updates").contains_key(&repo)).then_some(update)
    }

    pub fn found(&self) -> BTreeMap<String, Update> {
        self.found.lock().expect("updates").clone()
    }

    pub fn stage(&self, repo: &str, revision: &str) {
        self.staged.lock().expect("updates").insert(repo.into(), revision.into());
    }

    pub fn staged(&self) -> BTreeMap<String, String> {
        self.staged.lock().expect("updates").clone()
    }

    pub fn is_staged(&self, repo: &str) -> bool {
        self.staged.lock().expect("updates").contains_key(repo)
    }

    /// The repository's revision is in use: nothing is pending for its models.
    pub fn switched(&self, repo: &str) {
        self.staged.lock().expect("updates").remove(repo);
        self.found.lock().expect("updates").retain(|model, _| parse_model(model).is_ok_and(|(r, _)| r != repo));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::downloads::tests::{fake_hub, gguf_bytes, progress};
    use crate::downloads::{download, switch_revision};

    fn kinds(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn finds_content_and_docs_only_updates() {
        let (endpoint, hub) = fake_hub().await;
        let cache = tempfile::tempdir().unwrap();
        let (q4, q8, mtp) = (gguf_bytes(1, 500), gguf_bytes(2, 500), gguf_bytes(3, 700));
        hub.set("a", &[("m-Q4_0.gguf", &q4), ("m-Q8_0.gguf", &q8)]);
        for model in ["o/r:Q4_0", "o/r:Q8_0"] {
            let rev = download(cache.path(), &endpoint, None, model, &kinds(&["mtp"]), &progress()).await.unwrap();
            switch_revision(cache.path(), "o/r", &rev).unwrap();
        }
        let updates = Updates::default();
        let models = ["o/r:Q4_0".to_string(), "o/r:Q8_0".to_string()];
        let sidecars = kinds(&["mtp"]);
        assert!(updates.check(cache.path(), &endpoint, None, &sidecars, &models).await.is_empty());

        // A README commit: a new revision, the same files.
        hub.set("b", &[("m-Q4_0.gguf", &q4), ("m-Q8_0.gguf", &q8)]);
        let found = updates.check(cache.path(), &endpoint, None, &sidecars, &models).await;
        assert_eq!(found["o/r:Q4_0"], Update { revision: "b".repeat(40), size: 0, content: false });

        // An MTP head for Q4_0 appears; Q8_0 is unchanged.
        hub.set("c", &[("m-Q4_0.gguf", &q4), ("m-Q8_0.gguf", &q8), ("m-Q4_0-MTP-Q8_0.gguf", &mtp)]);
        let found = updates.check(cache.path(), &endpoint, None, &sidecars, &models).await;
        assert_eq!(found["o/r:Q4_0"], Update { revision: "c".repeat(40), size: 708, content: true });
        assert!(!found["o/r:Q8_0"].content);
        assert!(updates.check(cache.path(), &endpoint, None, &[], &models).await.values().all(|u| !u.content));

        updates.stage("o/r", &"c".repeat(40));
        assert!(updates.get("o/r:Q4_0").is_none(), "staged updates aren't offered again");
        updates.switched("o/r");
        assert!(updates.found().is_empty() && updates.staged().is_empty());
    }

    #[test]
    fn power_supply() {
        let dir = tempfile::tempdir().unwrap();
        assert!(on_ac(dir.path()), "no battery: a desktop");
        let supply = |name: &str, kind: &str, online: Option<&str>| {
            let d = dir.path().join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("type"), format!("{kind}\n")).unwrap();
            if let Some(o) = online {
                std::fs::write(d.join("online"), format!("{o}\n")).unwrap();
            }
        };
        supply("BAT0", "Battery", None);
        supply("AC", "Mains", Some("0"));
        assert!(!on_ac(dir.path()));
        supply("ucsi-source-psy-1", "USB", Some("1"));
        assert!(on_ac(dir.path()));
    }
}

//! Downloads from Hugging Face into the shared hub cache, one at a time. A
//! download belongs to the daemon: it goes on when the client that asked
//! disconnects, and resumes from its `.incomplete` file after an interruption.

use crate::hf::{self, repo_dir, valid_file, valid_oid, valid_repo};
use futures_util::StreamExt;
use nyx_ai_proto::{Download, DownloadState, Error, ErrorKind};
use reqwest::header::RANGE;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use futures_util::future::BoxFuture;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::watch;

#[derive(Deserialize)]
struct RepoInfo {
    sha: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lfs {
    pub oid: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct File {
    pub path: String,
    pub size: u64,
    pub lfs: Option<Lfs>,
}

/// The files one quant needs: its parts, and the companions in `sidecars` that
/// the repository has for it: the folder's projector, the quant's MTP head,
/// and drafters (Q8_0 preferred, then the smallest).
pub fn selected_files(tree: &[File], quant: &str, sidecars: &[String]) -> Result<Vec<File>, String> {
    let gguf = |f: &&File| f.path.to_ascii_lowercase().ends_with(".gguf");
    let first = tree
        .iter()
        .filter(gguf)
        .filter(|f| !hf::auxiliary_gguf(&f.path) && hf::quant_tag(&f.path) == quant)
        .min_by_key(|f| f.path.len())
        // The router names some files by their last part (`gemma-4-E4B_q4_0-it.gguf` is `IT`).
        .or_else(|| {
            tree.iter()
                .filter(gguf)
                .filter(|f| !hf::auxiliary_gguf(&f.path) && hf::router_tag(f.path.rsplit('/').next().unwrap_or(&f.path)) == quant)
                .min_by_key(|f| f.path.len())
        })
        .ok_or_else(|| {
            if tree.iter().any(|f| hf::auxiliary_gguf(&f.path) && hf::quant_tag(&f.path) == quant) {
                format!("{quant} is only a companion drafter or projector in this repository; choose a main model quantization")
            } else {
                format!("{quant} is not in the Hugging Face repository")
            }
        })?;
    let mut files = if let Some((prefix, _)) = first.path.rsplit_once("-00001-of-") {
        tree.iter()
            .filter(|f| f.path.starts_with(prefix) && f.path.contains("-of-") && hf::quant_tag(&f.path) == quant)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        vec![first.clone()]
    };
    let folder = Path::new(&first.path).parent();
    let stem = first.path.rsplit('/').next().unwrap_or(&first.path).trim_end_matches(".gguf").to_ascii_lowercase();
    let stem = stem.split("-00001-of-").next().unwrap_or(&stem).to_string();
    for kind in sidecars {
        let pick = tree
            .iter()
            .filter(gguf)
            .filter(|f| Path::new(&f.path).parent() == folder && hf::sidecar_kind(&f.path) == Some(kind.as_str()))
            // An MTP head belongs to one quant: its name starts with the model's.
            .filter(|f| kind != "mtp" || f.path.rsplit('/').next().unwrap_or(&f.path).to_ascii_lowercase().starts_with(&stem))
            .min_by_key(|f| (hf::quant_tag(&f.path) != "Q8_0", f.size));
        if let Some(file) = pick {
            files.push(file.clone());
        }
    }
    if files.iter().any(|f| !valid_file(&f.path) || f.lfs.as_ref().is_none_or(|l| !valid_oid(&l.oid))) {
        return Err("the repository has an invalid GGUF file path or checksum".into());
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// `org/repo:QUANT`, with the quant spelled the way file names carry it.
pub fn parse_model(model: &str) -> Result<(String, String), Error> {
    let (repo, tag) = model
        .rsplit_once(':')
        .ok_or_else(|| Error::new(ErrorKind::InvalidRequest, "name a Hugging Face model as org/repo:QUANT"))?;
    if !valid_repo(repo) {
        return Err(Error::new(ErrorKind::InvalidRequest, "invalid Hugging Face repository name"));
    }
    Ok((repo.to_string(), hf::quant_tag(tag)))
}

/// Bytes free where the cache lives.
fn free_space(path: &Path) -> Option<u64> {
    let mut dir = path.to_path_buf();
    while !dir.exists() {
        dir = dir.parent()?.to_path_buf();
    }
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: statvfs fills the struct for a valid NUL-terminated path.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

const SPACE_MARGIN: u64 = 1 << 30;

struct Progress<'a> {
    tx: &'a watch::Sender<Download>,
    last: Instant,
}

impl Progress<'_> {
    fn set(&mut self, status: &str, completed: u64, total: u64, force: bool) {
        if force || self.last.elapsed() >= Duration::from_millis(300) {
            self.last = Instant::now();
            self.tx.send_modify(|d| {
                d.status = status.into();
                d.completed = completed;
                d.total = total;
            });
        }
    }
}

fn failed(what: &'static str) -> impl Fn(reqwest::Error) -> Error {
    move |e| Error::new(ErrorKind::DownloadFailed, format!("{what}: {e}"))
}

fn fetch(client: &reqwest::Client, token: Option<&str>, url: String) -> reqwest::RequestBuilder {
    let request = client.get(url);
    match token {
        Some(t) => request.bearer_auth(t),
        None => request,
    }
}

/// What every file of one download shares.
struct Job<'a> {
    endpoint: &'a str,
    client: &'a reqwest::Client,
    token: Option<&'a str>,
    repo: &'a str,
    revision: &'a str,
    base: &'a Path,
    total: u64,
}

async fn download_file(job: &Job<'_>, file: &File, completed: &mut u64, progress: &mut Progress<'_>) -> Result<(), Error> {
    let Job { endpoint, client, token, repo, revision, base, total } = *job;
    let io = |e: std::io::Error| Error::new(ErrorKind::DownloadFailed, format!("writing the download: {e}"));
    let oid = &file.lfs.as_ref().expect("checked by selected_files").oid;
    let blob = base.join("blobs").join(oid);
    let partial = base.join("blobs").join(format!("{oid}.incomplete"));
    tokio::fs::create_dir_all(base.join("blobs")).await.map_err(io)?;
    if tokio::fs::metadata(&blob).await.is_ok_and(|m| m.len() == file.size) {
        *completed += file.size;
    } else {
        let mut offset = tokio::fs::metadata(&partial).await.map(|m| m.len()).unwrap_or(0);
        if offset > file.size {
            tokio::fs::remove_file(&partial).await.map_err(io)?;
            offset = 0;
        }
        *completed += offset;
        progress.set("Downloading", *completed, total, true);
        if offset < file.size {
            let url = format!("{endpoint}/{repo}/resolve/{revision}/{}", hf::encode_path(&file.path));
            let mut request = fetch(client, token, url);
            if offset > 0 {
                request = request.header(RANGE, format!("bytes={offset}-"));
            }
            let response = request
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| Error::new(ErrorKind::DownloadFailed, format!("Hugging Face file download: {e}")))?;
            if offset > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                *completed -= offset;
                offset = 0;
            }
            let mut out = tokio::fs::OpenOptions::new().create(true).write(true).truncate(false).open(&partial).await.map_err(io)?;
            if offset == 0 {
                out.set_len(0).await.map_err(io)?;
            } else {
                out.seek(std::io::SeekFrom::Start(offset)).await.map_err(io)?;
            }
            let mut stream = response.bytes_stream();
            loop {
                let next = tokio::time::timeout(Duration::from_secs(60), stream.next())
                    .await
                    .map_err(|_| Error::new(ErrorKind::DownloadFailed, "download stalled; download again to resume"))?;
                let Some(chunk) = next else { break };
                let chunk = chunk.map_err(|e| Error::new(ErrorKind::DownloadFailed, format!("download interrupted ({e}); download again to resume")))?;
                out.write_all(&chunk).await.map_err(io)?;
                *completed += chunk.len() as u64;
                progress.set("Downloading", *completed, total, false);
            }
            out.flush().await.map_err(io)?;
        }
        let actual = tokio::fs::metadata(&partial).await.map_err(io)?.len();
        if actual != file.size {
            return Err(Error::new(ErrorKind::DownloadFailed, format!("incomplete download of {}: {actual} of {} bytes", file.path, file.size)));
        }
        progress.set("Checking", *completed, total, true);
        let check_path = partial.clone();
        let digest = tokio::task::spawn_blocking(move || -> std::io::Result<String> {
            use std::io::Read;
            let mut input = std::fs::File::open(check_path)?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let count = input.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            Ok(format!("{:x}", hash.finalize()))
        })
        .await
        .map_err(|e| Error::new(ErrorKind::Internal, e.to_string()))?
        .map_err(io)?;
        if digest != *oid {
            tokio::fs::remove_file(&partial).await.map_err(io)?;
            return Err(Error::new(ErrorKind::DownloadFailed, format!("checksum mismatch for {}", file.path)));
        }
        if file.path.to_ascii_lowercase().ends_with(".gguf") && !crate::gguf::looks_like_gguf(&partial) {
            tokio::fs::remove_file(&partial).await.map_err(io)?;
            return Err(Error::new(ErrorKind::DownloadFailed, format!("{} isn't a GGUF file", file.path)));
        }
        tokio::fs::rename(&partial, &blob).await.map_err(io)?;
    }
    let snapshot = base.join("snapshots").join(revision).join(&file.path);
    tokio::fs::create_dir_all(snapshot.parent().expect("snapshot has a folder")).await.map_err(io)?;
    if tokio::fs::symlink_metadata(&snapshot).await.is_err() {
        let depth = Path::new(&file.path).components().count();
        let link = PathBuf::from("../".repeat(depth + 1)).join("blobs").join(oid);
        std::os::unix::fs::symlink(link, snapshot).map_err(io)?;
    }
    progress.set("Downloading", *completed, total, true);
    Ok(())
}

pub fn client() -> Result<reqwest::Client, Error> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .user_agent(concat!("nyx-ai/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(failed("HTTP client"))
}

/// A repository's newest revision and its files.
pub async fn latest(client: &reqwest::Client, endpoint: &str, token: Option<&str>, repo: &str) -> Result<(String, Vec<File>), Error> {
    let encoded = hf::encode_path(repo);
    let info: RepoInfo = fetch(client, token, format!("{endpoint}/api/models/{encoded}"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(failed("Hugging Face model"))?
        .json()
        .await
        .map_err(failed("Hugging Face model"))?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::new(ErrorKind::DownloadFailed, "Hugging Face returned an invalid revision"));
    }
    let tree: Vec<File> = fetch(client, token, format!("{endpoint}/api/models/{encoded}/tree/{}?recursive=true", info.sha))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(failed("Hugging Face file list"))?
        .json()
        .await
        .map_err(failed("Hugging Face file list"))?;
    Ok((info.sha, tree))
}

/// Download one model's files at the repository's newest revision into `cache`,
/// with its snapshot links. Returns the revision; `switch_revision` makes it
/// the one in use.
pub async fn download(
    cache: &Path,
    endpoint: &str,
    token: Option<&str>,
    model: &str,
    sidecars: &[String],
    tx: &watch::Sender<Download>,
) -> Result<String, Error> {
    let (repo, quant) = parse_model(model)?;
    let client = client()?;
    let mut progress = Progress { tx, last: Instant::now() };
    progress.set("Looking up the model", 0, 0, true);
    let (sha, tree) = latest(&client, endpoint, token, &repo).await?;
    let files = selected_files(&tree, &quant, sidecars).map_err(|e| Error::new(ErrorKind::DownloadFailed, e))?;
    let total: u64 = files.iter().map(|f| f.size).sum();
    let base = repo_dir(cache, &repo);
    let present: u64 = files
        .iter()
        .map(|f| {
            let oid = &f.lfs.as_ref().expect("checked").oid;
            std::fs::metadata(base.join("blobs").join(oid))
                .or_else(|_| std::fs::metadata(base.join("blobs").join(format!("{oid}.incomplete"))))
                .map(|m| m.len().min(f.size))
                .unwrap_or(0)
        })
        .sum();
    if let Some(free) = free_space(cache)
        && free < total.saturating_sub(present) + SPACE_MARGIN
    {
        return Err(Error::new(
            ErrorKind::ResourceExhausted,
            format!("not enough disk space: the model needs {} GB more and {} GB are free", (total - present) / 1_000_000_000 + 1, free / 1_000_000_000),
        ));
    }
    let mut completed = 0;
    let job = Job { endpoint, client: &client, token, repo: &repo, revision: &sha, base: &base, total };
    for file in &files {
        download_file(&job, file, &mut completed, &mut progress).await?;
    }
    progress.set("Downloaded", total, total, true);
    Ok(sha)
}

/// The revision `refs/main` names.
pub fn current_revision(cache: &Path, repo: &str) -> Option<String> {
    let text = std::fs::read_to_string(repo_dir(cache, repo).join("refs/main")).ok()?;
    Some(text.trim().to_string()).filter(|r| !r.is_empty())
}

/// Every file in a snapshot and the blob (oid) it links to.
pub fn snapshot_links(cache: &Path, repo: &str, revision: &str) -> BTreeMap<String, String> {
    let root = repo_dir(cache, repo).join("snapshots").join(revision);
    let mut out = BTreeMap::new();
    let mut dirs = vec![root.clone()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if let Ok(target) = std::fs::read_link(&path) {
                let oid = target.file_name().unwrap_or_default().to_string_lossy().into_owned();
                out.insert(path.strip_prefix(&root).unwrap_or(&path).display().to_string(), oid);
            } else if path.is_dir() {
                dirs.push(path);
            }
        }
    }
    out
}

/// Make `revision` the one in use: link everything else the old snapshot had
/// (other quants stay), point `refs/main` at it, drop the old snapshot, and
/// delete blobs no snapshot links to any more.
pub fn switch_revision(cache: &Path, repo: &str, revision: &str) -> std::io::Result<()> {
    let base = repo_dir(cache, repo);
    let new = base.join("snapshots").join(revision);
    if !new.is_dir() {
        return Err(std::io::Error::other(format!("{repo} has no snapshot {revision}")));
    }
    let old = current_revision(cache, repo);
    if old.as_deref() == Some(revision) {
        return Ok(());
    }
    if let Some(old) = &old {
        let old_root = base.join("snapshots").join(old);
        for (rel, _) in snapshot_links(cache, repo, old) {
            let target = std::fs::read_link(old_root.join(&rel))?;
            let link = new.join(&rel);
            if std::fs::symlink_metadata(&link).is_err() {
                if let Some(parent) = link.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::os::unix::fs::symlink(target, link)?;
            }
        }
    }
    crate::paths::write_atomic(&base.join("refs/main"), revision.as_bytes())?;
    if let Some(old) = old {
        let _ = std::fs::remove_dir_all(base.join("snapshots").join(old));
    }
    prune(cache, repo);
    Ok(())
}

/// Delete whole blobs no snapshot of the repository links to.
pub fn prune(cache: &Path, repo: &str) -> u64 {
    let base = repo_dir(cache, repo);
    let mut used = std::collections::BTreeSet::new();
    for snapshot in std::fs::read_dir(base.join("snapshots")).into_iter().flatten().flatten() {
        let name = snapshot.file_name().to_string_lossy().into_owned();
        used.extend(snapshot_links(cache, repo, &name).into_values());
    }
    let mut freed = 0;
    for blob in std::fs::read_dir(base.join("blobs")).into_iter().flatten().flatten() {
        let name = blob.file_name().to_string_lossy().into_owned();
        if name.ends_with(".incomplete") || used.contains(&name) {
            continue;
        }
        let size = blob.metadata().map(|m| m.len()).unwrap_or(0);
        if std::fs::remove_file(blob.path()).is_ok() {
            freed += size;
        }
    }
    freed
}

struct Entry {
    tx: watch::Sender<Download>,
    abort: Option<tokio::task::AbortHandle>,
    /// Started by the updater, which may pause it.
    auto: bool,
}

/// Runs after each download with `(model, revision)`, to put the revision in use.
pub type OnDone = Box<dyn Fn(String, String) -> BoxFuture<'static, ()> + Send + Sync>;

/// The queue.
pub struct Downloads {
    cache: PathBuf,
    endpoint: String,
    config_home: PathBuf,
    sidecars: Vec<String>,
    entries: Mutex<Vec<Entry>>,
    on_done: OnDone,
}

impl Downloads {
    /// `endpoint`: Hugging Face's address (`hf::endpoint()`).
    pub fn new(cache: PathBuf, endpoint: String, config_home: PathBuf, sidecars: Vec<String>, on_done: OnDone) -> Arc<Downloads> {
        Arc::new(Downloads { cache, endpoint, config_home, sidecars, entries: Mutex::new(Vec::new()), on_done })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn sidecars(&self) -> &[String] {
        &self.sidecars
    }

    /// Queue a download (or join the one already going) and watch it.
    pub fn start(self: &Arc<Self>, model: &str) -> Result<watch::Receiver<Download>, Error> {
        self.queue(model, false)
    }

    /// Queue an update the updater may pause.
    pub fn start_auto(self: &Arc<Self>, model: &str) -> Result<watch::Receiver<Download>, Error> {
        self.queue(model, true)
    }

    fn queue(self: &Arc<Self>, model: &str, auto: bool) -> Result<watch::Receiver<Download>, Error> {
        parse_model(model)?;
        let mut entries = self.entries.lock().expect("downloads");
        if let Some(e) = entries.iter_mut().find(|e| e.tx.borrow().model == model && active(e.tx.borrow().state)) {
            e.auto &= auto;
            return Ok(e.tx.subscribe());
        }
        entries.retain(|e| e.tx.borrow().model != model);
        let (tx, rx) = watch::channel(Download {
            model: model.into(),
            state: DownloadState::Queued,
            status: "Waiting".into(),
            completed: 0,
            total: 0,
            error: None,
        });
        entries.push(Entry { tx, abort: None, auto });
        drop(entries);
        self.pump();
        Ok(rx)
    }

    /// Stop the updater's downloads; their partial files stay for next time.
    pub fn pause_auto(self: &Arc<Self>) -> Vec<String> {
        let models: Vec<String> = self
            .entries
            .lock()
            .expect("downloads")
            .iter()
            .filter(|e| e.auto && active(e.tx.borrow().state))
            .map(|e| e.tx.borrow().model.clone())
            .collect();
        for model in &models {
            self.cancel(model);
        }
        models
    }

    pub fn cancel(self: &Arc<Self>, model: &str) -> bool {
        let mut entries = self.entries.lock().expect("downloads");
        let Some(entry) = entries.iter_mut().find(|e| e.tx.borrow().model == model && active(e.tx.borrow().state)) else {
            return false;
        };
        if let Some(abort) = entry.abort.take() {
            abort.abort();
        }
        entry.tx.send_modify(|d| {
            d.state = DownloadState::Cancelled;
            d.status = "Cancelled".into();
        });
        drop(entries);
        self.pump();
        true
    }

    pub fn list(&self) -> Vec<Download> {
        self.entries.lock().expect("downloads").iter().map(|e| e.tx.borrow().clone()).collect()
    }

    pub fn active(&self, model: &str) -> bool {
        self.entries.lock().expect("downloads").iter().any(|e| e.tx.borrow().model == model && active(e.tx.borrow().state))
    }

    fn pump(self: &Arc<Self>) {
        let mut entries = self.entries.lock().expect("downloads");
        if entries.iter().any(|e| e.tx.borrow().state == DownloadState::Running) {
            return;
        }
        let Some(entry) = entries.iter_mut().find(|e| e.tx.borrow().state == DownloadState::Queued) else { return };
        entry.tx.send_modify(|d| {
            d.state = DownloadState::Running;
            d.status = "Starting".into();
        });
        let tx = entry.tx.clone();
        let me = self.clone();
        let task = tokio::spawn(async move {
            let model = tx.borrow().model.clone();
            let token = hf::token(&me.config_home);
            tracing::info!(%model, "download started");
            let result = download(&me.cache, &me.endpoint, token.as_deref(), &model, &me.sidecars, &tx).await;
            if let Ok(revision) = &result {
                tx.send_modify(|d| d.status = "Putting it in use".into());
                (me.on_done)(model.clone(), revision.clone()).await;
            }
            tx.send_modify(|d| match &result {
                Ok(_) => {
                    d.state = DownloadState::Done;
                    d.status = "Done".into();
                    d.completed = d.total;
                }
                Err(e) => {
                    d.state = DownloadState::Failed;
                    d.status = "Failed".into();
                    d.error = Some(e.message.clone());
                }
            });
            match &result {
                Ok(revision) => tracing::info!(%model, %revision, "download finished"),
                Err(e) => tracing::warn!(%model, error = %e, "download failed"),
            }
            me.entries.lock().expect("downloads").iter_mut().for_each(|e| {
                if e.tx.same_channel(&tx) {
                    e.abort = None;
                }
            });
            me.pump();
        });
        entry.abort = Some(task.abort_handle());
    }
}

/// Drop the partial files of a download that never finished, and the
/// repository's folder if nothing finished in it either. Returns whether there
/// was anything to drop.
pub fn discard_partial(cache: &Path, model: &str) -> bool {
    let Ok((repo, _)) = parse_model(model) else { return false };
    let base = repo_dir(cache, &repo);
    let mut found = false;
    for entry in std::fs::read_dir(base.join("blobs")).into_iter().flatten().flatten() {
        if entry.file_name().to_string_lossy().ends_with(".incomplete") {
            found |= std::fs::remove_file(entry.path()).is_ok();
        }
    }
    let finished = std::fs::read_dir(base.join("snapshots")).is_ok_and(|mut d| d.next().is_some());
    if found && !finished {
        let _ = std::fs::remove_dir_all(&base);
    }
    found
}

fn active(state: DownloadState) -> bool {
    matches!(state, DownloadState::Queued | DownloadState::Running)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn file(path: &str, size: u64) -> File {
        File { path: path.into(), size, lfs: Some(Lfs { oid: "a".repeat(64) }) }
    }

    fn names(files: &[File]) -> Vec<&str> {
        files.iter().map(|f| f.path.as_str()).collect()
    }

    fn kinds(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn selects_one_quant_and_its_companions() {
        let tree = [
            file("m-PQ2_0.gguf", 10),
            file("m-PTQ1_0.gguf", 8),
            file("m-mmproj-BF16.gguf", 4),
            file("m-mmproj-Q8_0.gguf", 2),
            file("m-PQ2_0-MTP-Q8_0.gguf", 5),
            file("m-PTQ1_0-MTP-Q8_0.gguf", 5),
            file("m-DFlash2-Q8_0.gguf", 3),
        ];
        let files = selected_files(&tree, "PTQ1_0", &kinds(&["mmproj", "mtp"])).unwrap();
        assert_eq!(names(&files), ["m-PTQ1_0-MTP-Q8_0.gguf", "m-PTQ1_0.gguf", "m-mmproj-Q8_0.gguf"]);
        let files = selected_files(&tree, "PQ2_0", &kinds(&["mmproj", "mtp", "dflash"])).unwrap();
        assert_eq!(names(&files), ["m-DFlash2-Q8_0.gguf", "m-PQ2_0-MTP-Q8_0.gguf", "m-PQ2_0.gguf", "m-mmproj-Q8_0.gguf"]);
        assert_eq!(names(&selected_files(&tree, "PQ2_0", &[]).unwrap()), ["m-PQ2_0.gguf"]);
    }

    #[test]
    fn selects_split_parts() {
        let tree = [file("Q8_0/a-Q8_0-00001-of-00002.gguf", 5), file("Q8_0/a-Q8_0-00002-of-00002.gguf", 5), file("a-Q4_K_M.gguf", 3)];
        assert_eq!(selected_files(&tree, "Q8_0", &[]).unwrap().len(), 2);
    }

    #[test]
    fn rejects_drafters_and_bad_paths() {
        let tree = [file("model-DFlash2-Q8_0.gguf", 8)];
        assert!(selected_files(&tree, "Q8_0", &[]).unwrap_err().contains("companion drafter"));
        let tree = [file("../escape-Q4_0.gguf", 8)];
        assert!(selected_files(&tree, "Q4_0", &[]).unwrap_err().contains("invalid"));
        let tree = [File { path: "m-Q4_0.gguf".into(), size: 1, lfs: None }];
        assert!(selected_files(&tree, "Q4_0", &[]).is_err());
        assert!(parse_model("../x:Q4").is_err());
        assert_eq!(parse_model("o/r:q4_k_m").unwrap(), ("o/r".into(), "Q4_K_M".into()));
    }

    #[test]
    fn measures_free_space() {
        assert!(free_space(Path::new("/tmp/does/not/exist")).is_some_and(|f| f > 0));
    }

    pub(crate) use crate::testhub::{fake_hub, gguf_bytes};

    pub(crate) fn progress() -> watch::Sender<Download> {
        watch::channel(Download { model: "o/r:Q4_0".into(), state: DownloadState::Running, status: String::new(), completed: 0, total: 0, error: None }).0
    }

    #[tokio::test]
    async fn downloads_resumes_and_checks() {
        let body = gguf_bytes(1, 200_000);
        let oid = format!("{:x}", Sha256::digest(&body));
        let (endpoint, hub) = fake_hub().await;
        hub.set("a", &[("m-Q4_0.gguf", &body)]);
        let cache = tempfile::tempdir().unwrap();
        let base = repo_dir(cache.path(), "o/r");
        std::fs::create_dir_all(base.join("blobs")).unwrap();
        std::fs::write(base.join("blobs").join(format!("{oid}.incomplete")), &body[..50_000]).unwrap();
        let tx = progress();
        let revision = download(cache.path(), &endpoint, None, "o/r:q4_0", &[], &tx).await.unwrap();
        assert_eq!(revision, "a".repeat(40));
        assert_eq!(hub.ranges.load(Ordering::SeqCst), 1, "it resumed from the partial file");
        assert_eq!(std::fs::read(base.join("blobs").join(&oid)).unwrap(), body);
        let snapshot = base.join("snapshots").join(&revision).join("m-Q4_0.gguf");
        assert_eq!(std::fs::read(&snapshot).unwrap(), body);
        assert!(current_revision(cache.path(), "o/r").is_none(), "a download doesn't put itself in use");
        assert_eq!((tx.borrow().completed, tx.borrow().total), (body.len() as u64, body.len() as u64));
        switch_revision(cache.path(), "o/r", &revision).unwrap();
        assert_eq!(current_revision(cache.path(), "o/r").as_deref(), Some(revision.as_str()));
        assert_eq!(hf::cached_file(cache.path(), "o/r:Q4_0").unwrap(), snapshot);

        hub.lie_about("m-Q4_0.gguf");
        let other = tempfile::tempdir().unwrap();
        let err = download(other.path(), &endpoint, None, "o/r:Q4_0", &[], &progress()).await.unwrap_err();
        assert!(err.message.contains("checksum mismatch"), "{err:?}");
        assert!(!repo_dir(other.path(), "o/r").join("blobs").join(format!("{}.incomplete", "b".repeat(64))).exists());
        let err = download(other.path(), &endpoint, None, "o/r:Q8_0", &[], &progress()).await.unwrap_err();
        assert!(err.message.contains("not in the Hugging Face repository"));
    }

    #[tokio::test]
    async fn switching_keeps_other_quants_and_prunes_replaced_blobs() {
        let (endpoint, hub) = fake_hub().await;
        let cache = tempfile::tempdir().unwrap();
        let (q4, q8, q4_new) = (gguf_bytes(1, 1000), gguf_bytes(2, 1000), gguf_bytes(3, 1000));
        hub.set("a", &[("m-Q4_0.gguf", &q4), ("m-Q8_0.gguf", &q8)]);
        for model in ["o/r:Q4_0", "o/r:Q8_0"] {
            let rev = download(cache.path(), &endpoint, None, model, &[], &progress()).await.unwrap();
            switch_revision(cache.path(), "o/r", &rev).unwrap();
        }
        hub.set("b", &[("m-Q4_0.gguf", &q4_new), ("m-Q8_0.gguf", &q8)]);
        let rev = download(cache.path(), &endpoint, None, "o/r:Q4_0", &[], &progress()).await.unwrap();
        switch_revision(cache.path(), "o/r", &rev).unwrap();
        let links = snapshot_links(cache.path(), "o/r", &rev);
        assert_eq!(links.len(), 2, "the Q8_0 file came along: {links:?}");
        assert_eq!(std::fs::read(hf::cached_file(cache.path(), "o/r:Q4_0").unwrap()).unwrap(), q4_new);
        assert_eq!(std::fs::read(hf::cached_file(cache.path(), "o/r:Q8_0").unwrap()).unwrap(), q8);
        let blobs = std::fs::read_dir(repo_dir(cache.path(), "o/r").join("blobs")).unwrap().count();
        assert_eq!(blobs, 2, "the old Q4_0 blob is gone");
        assert!(!repo_dir(cache.path(), "o/r").join("snapshots").join("a".repeat(40)).exists());
    }

    #[test]
    fn discards_unfinished_downloads_only() {
        let cache = tempfile::tempdir().unwrap();
        let base = repo_dir(cache.path(), "o/r");
        std::fs::create_dir_all(base.join("blobs")).unwrap();
        std::fs::write(base.join("blobs/x.incomplete"), b"part").unwrap();
        assert!(discard_partial(cache.path(), "o/r:Q4_0"));
        assert!(!base.exists());
        std::fs::create_dir_all(base.join("snapshots/rev")).unwrap();
        std::fs::create_dir_all(base.join("blobs")).unwrap();
        std::fs::write(base.join("blobs/y.incomplete"), b"part").unwrap();
        std::fs::write(base.join("blobs/done"), b"whole").unwrap();
        assert!(discard_partial(cache.path(), "o/r:Q8_0"));
        assert!(base.join("blobs/done").exists() && !base.join("blobs/y.incomplete").exists());
        assert!(!discard_partial(cache.path(), "o/r:Q8_0"));
    }
}

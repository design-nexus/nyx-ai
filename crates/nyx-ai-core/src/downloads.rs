//! Downloads from Hugging Face into the shared hub cache, one at a time. A
//! download belongs to the daemon: it goes on when the client that asked
//! disconnects, and resumes from its `.incomplete` file after an interruption.

use crate::hf::{self, repo_dir, valid_file, valid_oid, valid_repo};
use futures_util::StreamExt;
use nyx_ai_proto::{Download, DownloadState, Error, ErrorKind};
use reqwest::header::RANGE;
use serde::Deserialize;
use sha2::{Digest, Sha256};
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

/// The files one quant needs: its parts and the folder's projector (Q8_0
/// preferred).
pub fn selected_files(tree: &[File], quant: &str) -> Result<Vec<File>, String> {
    let gguf = |f: &&File| f.path.to_ascii_lowercase().ends_with(".gguf");
    let first = tree
        .iter()
        .filter(gguf)
        .filter(|f| !hf::auxiliary_gguf(&f.path) && hf::quant_tag(&f.path) == quant)
        .min_by_key(|f| f.path.len())
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
    if let Some(projector) = tree
        .iter()
        .filter(gguf)
        .filter(|f| f.path.to_ascii_lowercase().contains("mmproj") && Path::new(&f.path).parent() == Path::new(&first.path).parent())
        .min_by_key(|f| (hf::quant_tag(&f.path) != "Q8_0", f.size))
    {
        files.push(projector.clone());
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
    client: &'a reqwest::Client,
    token: Option<&'a str>,
    repo: &'a str,
    revision: &'a str,
    base: &'a Path,
    total: u64,
}

async fn download_file(job: &Job<'_>, file: &File, completed: &mut u64, progress: &mut Progress<'_>) -> Result<(), Error> {
    let Job { client, token, repo, revision, base, total } = *job;
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
            let url = format!("{}/{repo}/resolve/{revision}/{}", hf::endpoint(), hf::encode_path(&file.path));
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

/// Download one model into `cache`.
pub async fn download(cache: &Path, token: Option<&str>, model: &str, tx: &watch::Sender<Download>) -> Result<(), Error> {
    let (repo, quant) = parse_model(model)?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .user_agent(concat!("nyx-ai/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(failed("HTTP client"))?;
    let mut progress = Progress { tx, last: Instant::now() };
    progress.set("Looking up the model", 0, 0, true);
    let encoded = hf::encode_path(&repo);
    let info: RepoInfo = fetch(&client, token, format!("{}/api/models/{encoded}", hf::endpoint()))
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
    let tree: Vec<File> = fetch(&client, token, format!("{}/api/models/{encoded}/tree/{}?recursive=true", hf::endpoint(), info.sha))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(failed("Hugging Face file list"))?
        .json()
        .await
        .map_err(failed("Hugging Face file list"))?;
    let files = selected_files(&tree, &quant).map_err(|e| Error::new(ErrorKind::DownloadFailed, e))?;
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
    let job = Job { client: &client, token, repo: &repo, revision: &info.sha, base: &base, total };
    for file in &files {
        download_file(&job, file, &mut completed, &mut progress).await?;
    }
    tokio::fs::create_dir_all(base.join("refs")).await.map_err(|e| Error::new(ErrorKind::DownloadFailed, e.to_string()))?;
    tokio::fs::write(base.join("refs/main"), &info.sha).await.map_err(|e| Error::new(ErrorKind::DownloadFailed, e.to_string()))?;
    progress.set("Done", total, total, true);
    Ok(())
}

struct Entry {
    tx: watch::Sender<Download>,
    abort: Option<tokio::task::AbortHandle>,
}

/// The queue. `on_done` runs after each successful download (to refresh the
/// model list).
pub struct Downloads {
    cache: PathBuf,
    config_home: PathBuf,
    entries: Mutex<Vec<Entry>>,
    on_done: Box<dyn Fn() + Send + Sync>,
}

impl Downloads {
    pub fn new(cache: PathBuf, config_home: PathBuf, on_done: impl Fn() + Send + Sync + 'static) -> Arc<Downloads> {
        Arc::new(Downloads { cache, config_home, entries: Mutex::new(Vec::new()), on_done: Box::new(on_done) })
    }

    /// Queue a download (or join the one already going) and watch it.
    pub fn start(self: &Arc<Self>, model: &str) -> Result<watch::Receiver<Download>, Error> {
        parse_model(model)?;
        let mut entries = self.entries.lock().expect("downloads");
        if let Some(e) = entries.iter().find(|e| e.tx.borrow().model == model && active(e.tx.borrow().state)) {
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
        entries.push(Entry { tx, abort: None });
        drop(entries);
        self.pump();
        Ok(rx)
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
            let result = download(&me.cache, token.as_deref(), &model, &tx).await;
            tx.send_modify(|d| match &result {
                Ok(()) => {
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
                Ok(()) => {
                    tracing::info!(%model, "download finished");
                    (me.on_done)();
                }
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

fn active(state: DownloadState) -> bool {
    matches!(state, DownloadState::Queued | DownloadState::Running)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: u64) -> File {
        File { path: path.into(), size, lfs: Some(Lfs { oid: "a".repeat(64) }) }
    }

    #[test]
    fn selects_one_quant_and_its_projector() {
        let tree = [file("model-PQ2_0.gguf", 10), file("model-PTQ1_0.gguf", 8), file("mmproj-BF16.gguf", 4), file("mmproj-Q8_0.gguf", 2)];
        let files = selected_files(&tree, "PTQ1_0").unwrap();
        assert_eq!(files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), ["mmproj-Q8_0.gguf", "model-PTQ1_0.gguf"]);
    }

    #[test]
    fn selects_split_parts() {
        let tree = [file("Q8_0/a-Q8_0-00001-of-00002.gguf", 5), file("Q8_0/a-Q8_0-00002-of-00002.gguf", 5), file("a-Q4_K_M.gguf", 3)];
        assert_eq!(selected_files(&tree, "Q8_0").unwrap().len(), 2);
    }

    #[test]
    fn rejects_drafters_and_bad_paths() {
        let tree = [file("model-DFlash2-Q8_0.gguf", 8)];
        assert!(selected_files(&tree, "Q8_0").unwrap_err().contains("companion drafter"));
        let tree = [file("../escape-Q4_0.gguf", 8)];
        assert!(selected_files(&tree, "Q4_0").unwrap_err().contains("invalid"));
        let tree = [File { path: "m-Q4_0.gguf".into(), size: 1, lfs: None }];
        assert!(selected_files(&tree, "Q4_0").is_err());
        assert!(parse_model("../x:Q4").is_err());
        assert_eq!(parse_model("o/r:q4_k_m").unwrap(), ("o/r".into(), "Q4_K_M".into()));
    }

    #[test]
    fn measures_free_space() {
        assert!(free_space(Path::new("/tmp/does/not/exist")).is_some_and(|f| f > 0));
    }
}

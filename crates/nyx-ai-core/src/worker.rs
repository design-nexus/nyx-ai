//! The llama.cpp worker: `llama-server` in router mode, started and watched by
//! AI Core, reachable only through a socket in AI Core's private folder.

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::Method;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use nyx_ai_proto::{Error, ErrorKind, WorkerState};
use serde_json::Value;
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::watch;

const MAX_BODY: usize = 256 << 20;
const CRASH_WINDOW: Duration = Duration::from_secs(120);
const CRASH_LIMIT: usize = 5;
const FAILED_PAUSE: Duration = Duration::from_secs(60);
const START_TIMEOUT: Duration = Duration::from_secs(60);

// ---------- HTTP over the worker socket ----------

#[derive(Clone)]
pub struct Http {
    socket: PathBuf,
    key: Arc<String>,
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A response whose body is read as it arrives. Dropping it closes the
/// connection, which stops the worker's work on it.
pub struct Response {
    pub status: u16,
    body: Incoming,
    _conn: AbortOnDrop,
}

impl Response {
    pub async fn chunk(&mut self) -> Option<Result<Bytes, Error>> {
        loop {
            match self.body.frame().await? {
                Ok(frame) => {
                    if let Ok(data) = frame.into_data() {
                        return Some(Ok(data));
                    }
                }
                Err(e) => return Some(Err(Error::new(ErrorKind::WorkerFailed, format!("llama.cpp stopped answering: {e}")))),
            }
        }
    }

    pub async fn json(mut self) -> Result<Value, Error> {
        let mut body = Vec::new();
        while let Some(chunk) = self.chunk().await {
            body.extend_from_slice(&chunk?);
            if body.len() > MAX_BODY {
                return Err(Error::new(ErrorKind::WorkerFailed, "llama.cpp reply too large"));
            }
        }
        let value: Value = serde_json::from_slice(&body).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).into_owned()));
        if (200..300).contains(&self.status) {
            Ok(value)
        } else {
            Err(http_error(self.status, &value))
        }
    }
}

fn http_error(status: u16, body: &Value) -> Error {
    let message = body["error"]["message"]
        .as_str()
        .or(body["error"].as_str())
        .or(body.as_str())
        .unwrap_or("request failed")
        .chars()
        .take(300)
        .collect::<String>();
    let kind = match status {
        400 => ErrorKind::InvalidRequest,
        404 => ErrorKind::UnknownModel,
        503 => ErrorKind::Busy,
        _ => ErrorKind::WorkerFailed,
    };
    Error::new(kind, format!("llama.cpp: {}", message.trim()))
}

impl Http {
    pub fn new(socket: PathBuf, key: String) -> Http {
        Http { socket, key: Arc::new(key) }
    }

    pub async fn send(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Response, Error> {
        let stream = UnixStream::connect(&self.socket)
            .await
            .map_err(|e| Error::new(ErrorKind::WorkerUnavailable, format!("llama.cpp isn't reachable: {e}")))?;
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|e| Error::new(ErrorKind::WorkerUnavailable, format!("llama.cpp handshake: {e}")))?;
        let task = AbortOnDrop(tokio::spawn(async move {
            let _ = conn.await;
        }));
        let mut request = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header("host", "localhost")
            .header("authorization", format!("Bearer {}", self.key));
        let bytes = match body {
            Some(value) => {
                request = request.header("content-type", "application/json");
                Bytes::from(serde_json::to_vec(value).expect("JSON values serialize"))
            }
            None => Bytes::new(),
        };
        let request = request.body(Full::new(bytes)).map_err(|e| Error::new(ErrorKind::Internal, e.to_string()))?;
        let response = sender
            .send_request(request)
            .await
            .map_err(|e| Error::new(ErrorKind::WorkerFailed, format!("llama.cpp dropped the request: {e}")))?;
        Ok(Response { status: response.status().as_u16(), body: response.into_body(), _conn: task })
    }

    pub async fn json(&self, method: Method, path: &str, body: Option<&Value>, timeout: Duration) -> Result<Value, Error> {
        tokio::time::timeout(timeout, async { self.send(method, path, body).await?.json().await })
            .await
            .map_err(|_| Error::new(ErrorKind::Timeout, format!("llama.cpp didn't answer {path} in time")))?
    }

    pub async fn get(&self, path: &str) -> Result<Value, Error> {
        self.json(Method::GET, path, None, Duration::from_secs(30)).await
    }

    pub async fn post(&self, path: &str, body: &Value) -> Result<Value, Error> {
        self.json(Method::POST, path, Some(body), Duration::from_secs(60)).await
    }
}

pub fn encode_query(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

// ---------- The supervisor ----------

#[derive(Clone, Debug)]
pub struct Config {
    /// `llama-server`; `None` means it isn't installed.
    pub program: Option<PathBuf>,
    pub preset: PathBuf,
    pub socket: PathBuf,
    pub key_file: PathBuf,
    pub cache: PathBuf,
    /// The router's own cap; AI Core's scheduler keeps below it.
    pub max_models: u32,
}

/// The installed build, else one on `PATH`.
pub fn find_program(configured: Option<&str>, lib: &Path) -> Option<PathBuf> {
    if let Some(p) = configured.filter(|p| !p.is_empty()) {
        let p = crate::paths::expand(p);
        return p.is_file().then_some(p);
    }
    let installed = lib.join("llama-server");
    if installed.is_file() {
        return Some(installed);
    }
    std::env::var_os("PATH")?.to_str()?.split(':').map(|d| Path::new(d).join("llama-server")).find(|p| p.is_file())
}

/// The release tag `install-llama.sh` wrote next to the build.
pub fn release(program: &Path) -> Option<String> {
    let text = std::fs::read_to_string(program.parent()?.join(".release")).ok()?;
    Some(text.trim().to_string()).filter(|t| !t.is_empty())
}

#[derive(Default)]
struct Inner {
    state: Option<WorkerState>,
    pid: Option<u32>,
    started: Option<Instant>,
    restarts: u32,
    crashes: VecDeque<Instant>,
    last_error: Option<String>,
    /// Lines the worker printed about errors, newest last.
    errors: VecDeque<String>,
    /// Which spawn is current; exits of older ones are ignored.
    spawn: u64,
}

pub struct Worker {
    config: Config,
    http: Http,
    inner: Mutex<Inner>,
    /// Bumped each time a worker becomes ready, so users of the worker notice a
    /// restart (every loaded model is gone).
    generation: AtomicU64,
    changed: watch::Sender<u64>,
    start_lock: tokio::sync::Mutex<()>,
    shutting_down: AtomicBool,
}

pub struct Snapshot {
    pub state: WorkerState,
    pub program: Option<PathBuf>,
    pub pid: Option<u32>,
    pub restarts: u32,
    pub last_error: Option<String>,
    pub uptime: Duration,
}

impl Worker {
    pub fn new(config: Config) -> Arc<Worker> {
        let key = random_key();
        let http = Http::new(config.socket.clone(), key);
        let state = if config.program.is_some() { WorkerState::Stopped } else { WorkerState::Missing };
        Arc::new(Worker {
            config,
            http,
            inner: Mutex::new(Inner { state: Some(state), ..Default::default() }),
            generation: AtomicU64::new(0),
            changed: watch::channel(0).0,
            start_lock: tokio::sync::Mutex::new(()),
            shutting_down: AtomicBool::new(false),
        })
    }

    pub fn program(&self) -> Option<&Path> {
        self.config.program.as_deref()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    pub fn snapshot(&self) -> Snapshot {
        let inner = self.inner.lock().expect("worker state");
        Snapshot {
            state: inner.state.unwrap_or(WorkerState::Stopped),
            program: self.config.program.clone(),
            pid: inner.pid,
            restarts: inner.restarts,
            last_error: inner.last_error.clone(),
            uptime: inner.started.map(|s| s.elapsed()).unwrap_or_default(),
        }
    }

    /// The newest error line the worker printed about loading a model.
    pub fn load_error(&self) -> Option<String> {
        let inner = self.inner.lock().expect("worker state");
        let newest = |needle: &str| inner.errors.iter().rev().find(|l| l.contains(needle));
        newest("error loading model").or_else(|| newest("failed to load")).map(|l| {
            let text = l.split_once(" E ").map_or(l.as_str(), |(_, rest)| rest);
            text.trim().to_string()
        })
    }

    fn state(&self) -> WorkerState {
        self.inner.lock().expect("worker state").state.unwrap_or(WorkerState::Stopped)
    }

    fn set_state(&self, state: WorkerState) {
        self.inner.lock().expect("worker state").state = Some(state);
        self.changed.send_modify(|n| *n += 1);
    }

    /// The worker's HTTP client once it's ready, starting it if needed.
    pub async fn ready(self: &Arc<Self>) -> Result<Http, Error> {
        let deadline = Instant::now() + START_TIMEOUT * 2;
        let mut changes = self.subscribe();
        loop {
            if self.shutting_down.load(Ordering::SeqCst) {
                return Err(Error::new(ErrorKind::WorkerUnavailable, "Nyx AI Core is shutting down"));
            }
            match self.state() {
                WorkerState::Ready => return Ok(self.http.clone()),
                WorkerState::Missing => {
                    return Err(Error::new(
                        ErrorKind::WorkerUnavailable,
                        "llama.cpp isn't installed; reinstall Nyx AI Core (scripts/install-llama.sh)",
                    ));
                }
                WorkerState::Failed => {
                    let (recent, error) = {
                        let inner = self.inner.lock().expect("worker state");
                        (inner.crashes.back().is_some_and(|t| t.elapsed() < FAILED_PAUSE), inner.last_error.clone())
                    };
                    if recent {
                        return Err(Error::new(
                            ErrorKind::WorkerFailed,
                            format!("llama.cpp keeps crashing: {}", error.unwrap_or_else(|| "no error message".into())),
                        ));
                    }
                    self.inner.lock().expect("worker state").crashes.clear();
                    self.start().await?;
                }
                WorkerState::Stopped => self.start().await?,
                WorkerState::Starting | WorkerState::Restarting => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(Error::new(ErrorKind::Timeout, "llama.cpp took too long to start"));
                    }
                    let _ = tokio::time::timeout(left.min(Duration::from_millis(500)), changes.changed()).await;
                }
            }
        }
    }

    async fn start(self: &Arc<Self>) -> Result<(), Error> {
        let _guard = self.start_lock.lock().await;
        if matches!(self.state(), WorkerState::Ready | WorkerState::Starting) {
            return Ok(());
        }
        let program = self.config.program.clone().ok_or_else(|| Error::new(ErrorKind::WorkerUnavailable, "llama.cpp isn't installed"))?;
        self.set_state(WorkerState::Starting);
        let spawn = {
            let mut inner = self.inner.lock().expect("worker state");
            inner.spawn += 1;
            inner.spawn
        };
        let child = match self.spawn(&program) {
            Ok(child) => child,
            Err(e) => {
                let message = format!("couldn't start llama.cpp: {e}");
                let mut inner = self.inner.lock().expect("worker state");
                inner.last_error = Some(message.clone());
                inner.state = Some(WorkerState::Stopped);
                drop(inner);
                self.changed.send_modify(|n| *n += 1);
                return Err(Error::new(ErrorKind::WorkerUnavailable, message));
            }
        };
        let pid = child.id();
        self.inner.lock().expect("worker state").pid = pid;
        tokio::spawn(self.clone().watch(child, spawn));

        let started = Instant::now();
        loop {
            if self.state() != WorkerState::Starting {
                let error = self.inner.lock().expect("worker state").last_error.clone();
                return Err(Error::new(ErrorKind::WorkerFailed, error.unwrap_or_else(|| "llama.cpp exited while starting".into())));
            }
            if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_secs(2), self.http.get("/health")).await {
                break;
            }
            if started.elapsed() > START_TIMEOUT {
                self.kill_group();
                let message = "llama.cpp didn't become ready in a minute".to_string();
                self.inner.lock().expect("worker state").last_error = Some(message.clone());
                return Err(Error::new(ErrorKind::Timeout, message));
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        {
            let mut inner = self.inner.lock().expect("worker state");
            inner.started = Some(Instant::now());
            inner.state = Some(WorkerState::Ready);
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.changed.send_modify(|n| *n += 1);
        tracing::info!(pid, "llama.cpp worker ready");
        Ok(())
    }

    fn spawn(&self, program: &Path) -> std::io::Result<tokio::process::Child> {
        std::fs::create_dir_all(self.config.socket.parent().unwrap_or(Path::new("/tmp")))?;
        let _ = std::fs::remove_file(&self.config.socket);
        crate::paths::write_atomic(&self.config.key_file, self.http.key.as_bytes())?;
        std::fs::set_permissions(&self.config.key_file, std::fs::Permissions::from_mode(0o600))?;
        let mut cmd = tokio::process::Command::new(program);
        cmd.arg("--host")
            .arg(&self.config.socket)
            .arg("--api-key-file")
            .arg(&self.config.key_file)
            .arg("--models-preset")
            .arg(&self.config.preset)
            .args(["--models-max", &self.config.max_models.to_string(), "--no-models-autoload", "--jinja", "--no-webui"])
            .env("LLAMA_CACHE", &self.config.cache)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0)
            .kill_on_drop(true);
        for (name, _) in std::env::vars_os() {
            if name.to_string_lossy().starts_with("LLAMA_ARG_") {
                cmd.env_remove(name);
            }
        }
        // Model instances inherit this: a load uses only files on disk and never
        // fetches a missing projector or drafter.
        cmd.env("LLAMA_ARG_OFFLINE", "1").env_remove("HF_TOKEN");
        if let Some(dir) = program.parent() {
            let mut libs = dir.as_os_str().to_owned();
            if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH").filter(|v| !v.is_empty()) {
                libs.push(":");
                libs.push(existing);
            }
            cmd.env("LD_LIBRARY_PATH", libs);
        }
        // SAFETY: prctl is async-signal-safe; this runs in the child before exec.
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }
        cmd.spawn()
    }

    fn watch(self: Arc<Self>, child: tokio::process::Child, spawn: u64) -> futures_util::future::BoxFuture<'static, ()> {
        Box::pin(self.watch_inner(child, spawn))
    }

    async fn watch_inner(self: Arc<Self>, mut child: tokio::process::Child, spawn: u64) {
        for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Send + Unpin>),
            child.stderr.take().map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Send + Unpin>)]
        .into_iter()
        .flatten()
        {
            let me = self.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(pipe).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama", "{line}");
                    let lower = line.to_ascii_lowercase();
                    if lower.contains("error") || lower.contains("failed") || lower.contains("abort") {
                        let mut inner = me.inner.lock().expect("worker state");
                        inner.errors.push_back(line.chars().take(240).collect());
                        if inner.errors.len() > 20 {
                            inner.errors.pop_front();
                        }
                    }
                }
            });
        }
        let status = child.wait().await;
        let current = self.inner.lock().expect("worker state").spawn == spawn;
        if !current {
            return;
        }
        if self.shutting_down.load(Ordering::SeqCst) {
            self.set_state(WorkerState::Stopped);
            return;
        }
        let code = status.map(|s| s.to_string()).unwrap_or_else(|e| e.to_string());
        let (state, delay) = {
            let mut inner = self.inner.lock().expect("worker state");
            inner.pid = None;
            inner.started = None;
            let now = Instant::now();
            inner.crashes.push_back(now);
            while inner.crashes.front().is_some_and(|t| now.duration_since(*t) > CRASH_WINDOW) {
                inner.crashes.pop_front();
            }
            let detail = inner.errors.back().cloned().unwrap_or_default();
            inner.last_error = Some(if detail.is_empty() { format!("llama.cpp exited ({code})") } else { format!("llama.cpp exited ({code}): {detail}") });
            let crashes = inner.crashes.len();
            if crashes >= CRASH_LIMIT {
                inner.state = Some(WorkerState::Failed);
                (WorkerState::Failed, Duration::ZERO)
            } else {
                inner.restarts += 1;
                inner.state = Some(WorkerState::Restarting);
                (WorkerState::Restarting, Duration::from_secs(1 << (crashes - 1).min(4)))
            }
        };
        self.changed.send_modify(|n| *n += 1);
        tracing::warn!(%code, ?state, "llama.cpp worker exited");
        if state == WorkerState::Restarting {
            tokio::time::sleep(delay).await;
            if self.state() == WorkerState::Restarting && !self.shutting_down.load(Ordering::SeqCst) {
                self.set_state(WorkerState::Stopped);
                if let Err(e) = self.start().await {
                    tracing::warn!(error = %e, "llama.cpp worker didn't restart");
                }
            }
        }
    }

    fn kill_group(&self) {
        if let Some(pid) = self.inner.lock().expect("worker state").pid {
            // SAFETY: signalling our own child's process group.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }

    /// Stop the worker and everything it started.
    pub async fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        let pid = self.inner.lock().expect("worker state").pid;
        if let Some(pid) = pid {
            // SAFETY: signalling our own child's process group.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGTERM);
            }
            for _ in 0..50 {
                if self.inner.lock().expect("worker state").pid.is_none() || self.state() == WorkerState::Stopped {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            self.kill_group();
        }
        let _ = std::fs::remove_file(&self.config.socket);
        let _ = std::fs::remove_file(&self.config.key_file);
    }
}

fn random_key() -> String {
    let mut bytes = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = f.read_exact(&mut bytes);
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_from_llama_replies() {
        let e = http_error(404, &serde_json::json!({"error": {"code": 404, "message": "model not found"}}));
        assert_eq!(e.kind, ErrorKind::UnknownModel);
        assert_eq!(e.message, "llama.cpp: model not found");
        assert_eq!(http_error(500, &Value::String("boom".into())).kind, ErrorKind::WorkerFailed);
        assert_eq!(encode_query("org/m:Q4_K_M"), "org%2Fm%3AQ4_K_M");
        assert_eq!(random_key().len(), 64);
    }
}

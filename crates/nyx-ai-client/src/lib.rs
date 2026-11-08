//! The client applications use to reach Nyx AI Core.
//!
//! Each call opens its own connection, so a dropped [`Stream`] cancels its request
//! and a restarted daemon needs no reconnect logic beyond the retry in
//! [`Client::connect`].

pub use nyx_ai_proto as proto;

use proto::{
    API_VERSION, Capabilities, ChatRequest, ClientLine, Download, Embeddings, Error, ErrorKind, Event, Hello, MAX_LINE, Model,
    ModelDetail, ModelSettings, Op, Priority, Reply, Request, ResourceStatus, ServerLine, ServiceSettings, ToolCallDelta,
    WorkerStatus, encode,
};
use std::collections::BTreeMap;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

pub type Result<T> = std::result::Result<T, Error>;

/// Where the daemon listens: `NYX_AI_SOCKET`, else `$XDG_RUNTIME_DIR/nyx-ai/ai.sock`.
pub fn socket_path() -> PathBuf {
    if let Some(path) = std::env::var_os(proto::SOCKET_ENV).filter(|p| !p.is_empty()) {
        return path.into();
    }
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        // SAFETY: getuid has no preconditions.
        .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", unsafe { libc::getuid() })));
    runtime.join("nyx-ai").join(proto::SOCKET_NAME)
}

/// The socket must be a socket, ours, and closed to everyone else.
pub fn check_socket(path: &Path) -> Result<()> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| Error::new(ErrorKind::Unavailable, format!("Nyx AI Core isn't running ({}: {e})", path.display())))?;
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    if !meta.file_type().is_socket() || meta.uid() != uid || meta.mode() & 0o077 != 0 {
        return Err(Error::new(ErrorKind::PermissionDenied, format!("{} isn't a private socket of this user", path.display())));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Client {
    app: String,
    socket: PathBuf,
}

impl Client {
    /// `app` names the application, e.g. `nyx-chat`.
    pub fn new(app: &str) -> Client {
        Client { app: app.into(), socket: socket_path() }
    }

    pub fn with_socket(app: &str, socket: impl Into<PathBuf>) -> Client {
        Client { app: app.into(), socket: socket.into() }
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// The socket exists and passes [`check_socket`]. With socket activation the
    /// daemon may still be starting.
    pub fn available(&self) -> bool {
        check_socket(&self.socket).is_ok()
    }

    async fn connect(&self) -> Result<Connection> {
        let mut delay = Duration::from_millis(100);
        let mut attempt = 0;
        let stream = loop {
            check_socket(&self.socket)?;
            match UnixStream::connect(&self.socket).await {
                Ok(stream) => break stream,
                Err(e) if attempt < 3 && matches!(e.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound) => {
                    attempt += 1;
                    tokio::time::sleep(delay).await;
                    delay *= 3;
                }
                Err(e) => return Err(Error::new(ErrorKind::Unavailable, format!("Nyx AI Core isn't reachable: {e}"))),
            }
        };
        let (read, write) = stream.into_split();
        let mut conn = Connection { reader: BufReader::new(read), writer: write, line: Vec::new() };
        conn.send(&ClientLine::Hello(Hello { api_version: API_VERSION, client: self.app.clone() })).await?;
        let reply = tokio::time::timeout(Duration::from_secs(10), conn.read())
            .await
            .map_err(|_| Error::new(ErrorKind::Timeout, "Nyx AI Core didn't answer"))??;
        match reply {
            ServerLine::Hello(welcome) if welcome.api_version == API_VERSION => Ok(conn),
            ServerLine::Hello(welcome) => Err(Error::new(
                ErrorKind::IncompatibleVersion,
                format!("Nyx AI Core speaks version {}, this app version {API_VERSION}", welcome.api_version),
            )),
            ServerLine::Error { error, .. } => Err(error),
            other => Err(unexpected(&other)),
        }
    }

    /// Send one operation and wait for its result, skipping events.
    pub async fn call(&self, op: Op) -> Result<Reply> {
        let mut stream = self.stream(op).await?;
        while let Some(item) = stream.next().await {
            if let Item::Done(reply) = item? {
                return Ok(reply);
            }
        }
        Err(Error::new(ErrorKind::Unavailable, "Nyx AI Core closed the connection"))
    }

    /// Send one operation and read its events and result as they come.
    pub async fn stream(&self, op: Op) -> Result<Stream> {
        let mut conn = self.connect().await?;
        conn.send(&ClientLine::Request(Box::new(Request { id: 1, op }))).await?;
        Ok(Stream { conn, id: 1, done: false })
    }

    pub async fn list_models(&self) -> Result<Vec<Model>> {
        match self.call(Op::ListModels {}).await? {
            Reply::Models { models } => Ok(models),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn model_info(&self, model: &str) -> Result<ModelDetail> {
        match self.call(Op::ModelInfo { model: model.into() }).await? {
            Reply::ModelInfo(detail) => Ok(*detail),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn load_model(&self, model: &str, keep: bool) -> Result<()> {
        match self.call(Op::LoadModel { model: model.into(), keep }).await? {
            Reply::Loaded { .. } => Ok(()),
            other => Err(unexpected(&other)),
        }
    }

    /// Whether it actually unloaded (false while something else uses it).
    pub async fn unload_model(&self, model: &str) -> Result<bool> {
        match self.call(Op::UnloadModel { model: model.into() }).await? {
            Reply::Unloaded { unloaded, .. } => Ok(unloaded),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn remove_model(&self, model: &str) -> Result<()> {
        expect_ok(self.call(Op::RemoveModel { model: model.into() }).await?)
    }

    pub async fn model_settings(&self, model: &str) -> Result<ModelSettings> {
        match self.call(Op::ModelSettings { model: model.into() }).await? {
            Reply::Settings(s) => Ok(s),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn configure_model(&self, model: &str, settings: BTreeMap<String, String>) -> Result<ModelSettings> {
        match self.call(Op::ConfigureModel { model: model.into(), settings }).await? {
            Reply::Settings(s) => Ok(s),
            other => Err(unexpected(&other)),
        }
    }

    /// Streams `progress` events and ends with `ok`.
    pub async fn download_model(&self, model: &str) -> Result<Stream> {
        self.stream(Op::DownloadModel { model: model.into() }).await
    }

    pub async fn cancel_download(&self, model: &str) -> Result<()> {
        expect_ok(self.call(Op::CancelDownload { model: model.into() }).await?)
    }

    /// Check Hugging Face now; returns the models with updates.
    pub async fn check_updates(&self) -> Result<Vec<Model>> {
        match self.call(Op::CheckUpdates {}).await? {
            Reply::Models { models } => Ok(models),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn downloads(&self) -> Result<Vec<Download>> {
        match self.call(Op::Downloads {}).await? {
            Reply::Downloads { downloads } => Ok(downloads),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<Stream> {
        self.stream(Op::Chat(request)).await
    }

    pub async fn embed(&self, model: &str, input: Vec<String>, priority: Priority) -> Result<Embeddings> {
        match self.call(Op::Embed { model: model.into(), input, priority }).await? {
            Reply::Embeddings(e) => Ok(e),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn apply_template(&self, request: ChatRequest) -> Result<String> {
        match self.call(Op::ApplyTemplate(request)).await? {
            Reply::Template { prompt } => Ok(prompt),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn resource_status(&self) -> Result<ResourceStatus> {
        match self.call(Op::ResourceStatus {}).await? {
            Reply::Resources(r) => Ok(r),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn worker_status(&self) -> Result<WorkerStatus> {
        match self.call(Op::WorkerStatus {}).await? {
            Reply::Worker(w) => Ok(w),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn capabilities(&self) -> Result<Capabilities> {
        match self.call(Op::ListCapabilities {}).await? {
            Reply::Capabilities(c) => Ok(c),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn service_settings(&self) -> Result<ServiceSettings> {
        match self.call(Op::ServiceSettings {}).await? {
            Reply::Service(s) => Ok(s),
            other => Err(unexpected(&other)),
        }
    }

    pub async fn configure_service(&self, max_loaded: Option<u32>, idle_unload_seconds: Option<i64>) -> Result<ServiceSettings> {
        match self.call(Op::ConfigureService { max_loaded, idle_unload_seconds }).await? {
            Reply::Service(s) => Ok(s),
            other => Err(unexpected(&other)),
        }
    }
}

fn expect_ok(reply: Reply) -> Result<()> {
    match reply {
        Reply::Ok => Ok(()),
        other => Err(unexpected(&other)),
    }
}

fn unexpected(what: &impl std::fmt::Debug) -> Error {
    let text = format!("{what:?}");
    let short: String = text.chars().take(120).collect();
    Error::new(ErrorKind::Internal, format!("unexpected reply from Nyx AI Core: {short}"))
}

struct Connection {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
    line: Vec<u8>,
}

impl Connection {
    async fn send(&mut self, line: &ClientLine) -> Result<()> {
        let text = encode(line);
        self.writer.write_all(text.as_bytes()).await.map_err(lost)?;
        self.writer.flush().await.map_err(lost)
    }

    /// Cancel-safe: a partial line stays buffered for the next call.
    async fn read(&mut self) -> Result<ServerLine> {
        loop {
            let buf = self.reader.fill_buf().await.map_err(lost)?;
            if buf.is_empty() {
                return Err(Error::new(ErrorKind::Unavailable, "Nyx AI Core closed the connection"));
            }
            let (take, end) = match buf.iter().position(|b| *b == b'\n') {
                Some(i) => (i + 1, true),
                None => (buf.len(), false),
            };
            self.line.extend_from_slice(&buf[..take]);
            self.reader.consume(take);
            if self.line.len() > MAX_LINE {
                return Err(Error::new(ErrorKind::Internal, "reply line too long"));
            }
            if end {
                let line = std::mem::take(&mut self.line);
                return serde_json::from_slice(&line)
                    .map_err(|e| Error::new(ErrorKind::Internal, format!("unreadable reply from Nyx AI Core: {e}")));
            }
        }
    }
}

fn lost(e: std::io::Error) -> Error {
    Error::new(ErrorKind::Unavailable, format!("lost the connection to Nyx AI Core: {e}"))
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Event(Event),
    Done(Reply),
}

/// One request's events and result. Dropping it closes the connection, which
/// cancels the request (downloads go on in the daemon).
pub struct Stream {
    conn: Connection,
    id: u64,
    done: bool,
}

impl Stream {
    /// The next event or the result; `None` after the result or an error.
    pub async fn next(&mut self) -> Option<Result<Item>> {
        if self.done {
            return None;
        }
        loop {
            let line = match self.conn.read().await {
                Ok(line) => line,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            match line {
                ServerLine::Event { id, event } if id == self.id => return Some(Ok(Item::Event(event))),
                ServerLine::Result { id, result } if id == self.id => {
                    self.done = true;
                    return Some(Ok(Item::Done(result)));
                }
                ServerLine::Error { id, error } if id == Some(self.id) || id.is_none() => {
                    self.done = true;
                    return Some(Err(error));
                }
                // The reply to our own `cancel`.
                _ => continue,
            }
        }
    }

    /// Ask the daemon to stop; the stream then ends with a `cancelled` error.
    pub async fn cancel(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.conn.send(&ClientLine::Request(Box::new(Request { id: self.id + 1, op: Op::Cancel { request: self.id } }))).await
    }

    /// Read a chat to its end.
    pub async fn collect_chat(mut self) -> Result<ChatOutcome> {
        let mut out = ChatOutcome::default();
        while let Some(item) = self.next().await {
            match item? {
                Item::Event(Event::Delta { content, reasoning, tool_calls }) => {
                    out.content.push_str(&content);
                    out.reasoning.push_str(&reasoning);
                    out.add_tool_calls(&tool_calls);
                }
                Item::Event(_) => {}
                Item::Done(Reply::Finished(finished)) => {
                    out.finished = finished;
                    return Ok(out);
                }
                Item::Done(other) => return Err(unexpected(&other)),
            }
        }
        Err(Error::new(ErrorKind::Unavailable, "the chat ended without a result"))
    }
}

/// A whole chat reply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatOutcome {
    pub content: String,
    pub reasoning: String,
    /// `(id, name, arguments)` in call order.
    pub tool_calls: Vec<(String, String, String)>,
    pub finished: proto::Finished,
}

impl ChatOutcome {
    /// Fold tool-call parts into whole calls.
    pub fn add_tool_calls(&mut self, parts: &[ToolCallDelta]) {
        for part in parts {
            let index = part.index as usize;
            while self.tool_calls.len() <= index {
                self.tool_calls.push(Default::default());
            }
            let call = &mut self.tool_calls[index];
            if let Some(id) = &part.id {
                call.0.push_str(id);
            }
            if let Some(name) = &part.name {
                call.1.push_str(name);
            }
            if let Some(args) = &part.arguments {
                call.2.push_str(args);
            }
        }
    }
}

/// Blocking calls, for programs without a tokio runtime (CLIs, settings pages).
pub mod blocking {
    use super::*;

    fn runtime() -> Result<tokio::runtime::Runtime> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::new(ErrorKind::Internal, e.to_string()))
    }

    pub fn call(client: &Client, op: Op) -> Result<Reply> {
        runtime()?.block_on(client.call(op))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tokio::net::UnixListener;

    async fn fake_server(path: PathBuf, reply: impl Fn(Request) -> Vec<ServerLine> + Send + 'static) {
        let listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let (read, mut write) = stream.into_split();
                let mut lines = BufReader::new(read).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let out = match proto::decode_client_line(&line).unwrap() {
                        ClientLine::Hello(h) => vec![ServerLine::Hello(proto::Welcome {
                            api_version: h.api_version,
                            server: "test".into(),
                            client: h.client,
                        })],
                        ClientLine::Request(r) => reply(*r),
                    };
                    for l in out {
                        write.write_all(encode(&l).as_bytes()).await.unwrap();
                    }
                }
            }
        });
    }

    #[tokio::test]
    async fn streams_and_collects_a_chat() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ai.sock");
        fake_server(path.clone(), |r| {
            let delta = |content: &str, tool_calls| ServerLine::Event {
                id: r.id,
                event: Event::Delta { content: content.into(), reasoning: String::new(), tool_calls },
            };
            vec![
                ServerLine::Event { id: r.id, event: Event::Accepted },
                delta("Hel", vec![]),
                delta(
                    "lo",
                    vec![ToolCallDelta { index: 0, id: Some("c".into()), name: Some("calc".into()), arguments: Some("{".into()) }],
                ),
                delta("", vec![ToolCallDelta { index: 0, arguments: Some("}".into()), ..Default::default() }]),
                ServerLine::Result { id: r.id, result: Reply::Finished(proto::Finished { finish_reason: "stop".into(), ..Default::default() }) },
            ]
        })
        .await;
        let client = Client::with_socket("test", &path);
        assert!(client.available());
        let out = client.chat(ChatRequest::default()).await.unwrap().collect_chat().await.unwrap();
        assert_eq!(out.content, "Hello");
        assert_eq!(out.tool_calls, vec![("c".into(), "calc".into(), "{}".into())]);
        assert_eq!(out.finished.finish_reason, "stop");
    }

    #[tokio::test]
    async fn errors_and_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ai.sock");
        fake_server(path.clone(), |r| {
            vec![ServerLine::Error { id: Some(r.id), error: Error::new(ErrorKind::UnknownModel, "no such model") }]
        })
        .await;
        let client = Client::with_socket("test", &path);
        assert_eq!(client.model_info("x").await.unwrap_err().kind, ErrorKind::UnknownModel);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert_eq!(client.list_models().await.unwrap_err().kind, ErrorKind::PermissionDenied);
        let missing = Client::with_socket("test", dir.path().join("none.sock"));
        assert_eq!(missing.list_models().await.unwrap_err().kind, ErrorKind::Unavailable);
    }
}

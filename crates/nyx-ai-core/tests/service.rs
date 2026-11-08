//! The daemon end to end against the stub worker: socket, handshake, model
//! list, chat, embeddings, eviction, cancel, crash recovery and settings.

use nyx_ai_client::proto::{
    ChatRequest, ClientLine, ErrorKind, Event, Hello, Message, ModelState, Op, Priority, Role, ServerLine, encode,
};
use nyx_ai_client::{Client, Item};
use nyx_ai_core::paths::Paths;
use nyx_ai_core::service::{Caller, Options, Service};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;

fn gguf(pairs: &[(&str, Value)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((pairs.len() as u64).to_le_bytes());
    let string = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (key, value) in pairs {
        string(&mut out, key);
        match value {
            Value::U32(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            Value::Str(s) => {
                out.extend(8u32.to_le_bytes());
                string(&mut out, s);
            }
        }
    }
    out
}

enum Value {
    U32(u32),
    Str(&'static str),
}

fn chat_file() -> Vec<u8> {
    gguf(&[
        ("general.architecture", Value::Str("gemma4")),
        ("general.file_type", Value::U32(15)),
        ("tokenizer.chat_template", Value::Str("{% if tools %}{% endif %}{{ enable_thinking }}")),
        ("gemma4.context_length", Value::U32(8192)),
    ])
}

fn embed_file() -> Vec<u8> {
    gguf(&[("general.architecture", Value::Str("gemma-embedding")), ("gemma-embedding.pooling_type", Value::U32(1))])
}

fn cached(paths: &Paths, repo: &str, file: &str, bytes: &[u8]) {
    let dir = paths.cache.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots/0123456789abcdef0123456789abcdef01234567");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(file), bytes).unwrap();
}

struct Harness {
    _dir: tempfile::TempDir,
    paths: Paths,
    service: Arc<Service>,
    client: Client,
    stop: CancellationToken,
}

impl Harness {
    async fn start() -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        let folder = dir.path().join("models");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("local-chat-Q8_0.gguf"), chat_file()).unwrap();
        std::fs::create_dir_all(&paths.config).unwrap();
        std::fs::write(paths.settings(), format!("max_loaded = 1\nidle_unload_seconds = -1\nmodel_dirs = [{:?}]\n", folder.display().to_string())).unwrap();
        cached(&paths, "org/chat", "chat-Q4_K_M.gguf", &chat_file());
        cached(&paths, "org/embed", "embed-Q8_0.gguf", &embed_file());
        cached(&paths, "org/other", "other-Q4_K_M.gguf", &chat_file());
        cached(&paths, "org/broken", "broken-Q4_K_M.gguf", &chat_file());
        let options = Options { allow_untrusted: false, worker: Some(env!("CARGO_BIN_EXE_nyx-ai-stub-worker").into()) };
        let listener = nyx_ai_core::server::listen(&paths.socket).unwrap();
        let service = Service::new(paths.clone(), options);
        let stop = CancellationToken::new();
        service.run_background(stop.clone());
        tokio::spawn(nyx_ai_core::server::serve(service.clone(), listener, stop.clone()));
        let client = Client::with_socket("nyx-test", &paths.socket);
        Harness { _dir: dir, paths, service, client, stop }
    }

    async fn stop(self) {
        self.stop.cancel();
        self.service.shutdown().await;
    }
}

fn ask(model: &str, text: &str) -> ChatRequest {
    ChatRequest { model: model.into(), messages: vec![Message::text(Role::User, text)], priority: Priority::Interactive, ..Default::default() }
}

#[tokio::test(flavor = "multi_thread")]
async fn lists_models_and_chats() {
    let h = Harness::start().await;
    let models = h.client.list_models().await.unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["local-chat-Q8_0", "org/broken:Q4_K_M", "org/chat:Q4_K_M", "org/embed:Q8_0", "org/other:Q4_K_M"]);
    let chat = models.iter().find(|m| m.id == "org/chat:Q4_K_M").unwrap();
    assert_eq!(chat.state, ModelState::Installed);
    assert!(chat.removable && chat.quantization.as_deref() == Some("Q4_K_M"));
    assert!(!models[0].removable);

    let mut stream = h.client.chat(ask("org/chat:Q4_K_M", "hello")).await.unwrap();
    let mut seen = Vec::new();
    let mut text = String::new();
    let finished = loop {
        match stream.next().await.unwrap().unwrap() {
            Item::Event(Event::Delta { content, .. }) => text.push_str(&content),
            Item::Event(e) => seen.push(e),
            Item::Done(nyx_ai_client::proto::Reply::Finished(f)) => break f,
            Item::Done(other) => panic!("{other:?}"),
        }
    };
    assert_eq!(text, "tok0 tok1 tok2 tok3 tok4 ");
    assert_eq!(seen[0], Event::Accepted);
    assert!(seen.contains(&Event::ModelLoading { model: "org/chat:Q4_K_M".into() }));
    assert_eq!(*seen.last().unwrap(), Event::Started);
    assert_eq!((finished.finish_reason.as_str(), finished.usage.completion_tokens), ("stop", 5));
    assert!(finished.timings.load_ms > 0);

    let status = h.client.resource_status().await.unwrap();
    assert_eq!(status.loaded.len(), 1);
    assert_eq!(status.loaded[0].slots, 2);
    let listed = h.client.list_models().await.unwrap();
    assert_eq!(listed.iter().find(|m| m.id == "org/chat:Q4_K_M").unwrap().state, ModelState::Ready);
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn embeddings_evict_the_chat_model_and_check_capabilities() {
    let h = Harness::start().await;
    let out = h.client.chat(ask("org/chat:Q4_K_M", "hi")).await.unwrap().collect_chat().await.unwrap();
    assert!(!out.content.is_empty());
    let e = h.client.embed("org/embed:Q8_0", vec!["a".into(), "abcd".into()], Priority::Background).await.unwrap();
    assert_eq!((e.vectors.len(), e.dimensions), (2, 8));
    assert_eq!((e.vectors[0][0], e.vectors[1][0]), (1.0, 4.0));
    assert_eq!(e.identity.file, "embed-Q8_0.gguf");
    assert_eq!(e.identity.revision.as_deref(), Some("0123456789abcdef0123456789abcdef01234567"));
    let status = h.client.resource_status().await.unwrap();
    assert_eq!(status.loaded.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["org/embed:Q8_0"]);
    assert_eq!(h.client.worker_status().await.unwrap().metrics.evictions, 1);

    let err = h.client.chat(ask("org/embed:Q8_0", "hi")).await.unwrap().collect_chat().await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedCapability);
    let err = h.client.embed("org/chat:Q4_K_M", vec!["x".into()], Priority::Normal).await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedCapability);
    let err = h.client.chat(ask("org/missing:Q4_0", "hi")).await.unwrap().collect_chat().await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnknownModel);
    let err = h.client.chat(ask("org/broken:Q4_K_M", "hi")).await.unwrap().collect_chat().await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::ModelLoadFailed);
    let router = std::fs::read_to_string(h.paths.router_ini()).unwrap();
    assert!(router.contains("[org/embed:Q8_0]\nembeddings = true"), "{router}");
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_and_disconnect_stop_generation() {
    let h = Harness::start().await;
    let mut stream = h.client.chat(ask("org/chat:Q4_K_M", "count:500")).await.unwrap();
    loop {
        if let Item::Event(Event::Delta { .. }) = stream.next().await.unwrap().unwrap() {
            break;
        }
    }
    stream.cancel().await.unwrap();
    let err = loop {
        match stream.next().await.unwrap() {
            Ok(_) => continue,
            Err(e) => break e,
        }
    };
    assert_eq!(err.kind, ErrorKind::Cancelled);

    let mut dropped = h.client.chat(ask("org/chat:Q4_K_M", "count:500")).await.unwrap();
    while !matches!(dropped.next().await.unwrap().unwrap(), Item::Event(Event::Delta { .. })) {}
    drop(dropped);
    let mut idle = false;
    for _ in 0..50 {
        if h.client.resource_status().await.unwrap().loaded[0].active == 0 {
            idle = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(idle, "the dropped request kept its slot");
    let m = h.client.worker_status().await.unwrap().metrics;
    assert_eq!(m.cancelled, 2);
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_worker_crash_fails_the_request_and_recovers() {
    let h = Harness::start().await;
    let err = h.client.chat(ask("org/chat:Q4_K_M", "crash")).await.unwrap().collect_chat().await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::WorkerFailed, "{err:?}");
    let mut recovered = None;
    for _ in 0..60 {
        if let Ok(out) = h.client.chat(ask("org/chat:Q4_K_M", "again")).await.unwrap().collect_chat().await {
            recovered = Some(out);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(recovered.unwrap().content, "tok0 tok1 tok2 tok3 tok4 ");
    let worker = h.client.worker_status().await.unwrap();
    assert!(worker.restarts >= 1 && worker.metrics.worker_restarts >= 1);
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn settings_pins_and_removal() {
    let h = Harness::start().await;
    let s = h.client.configure_model("org/chat:Q4_K_M", [("ctx-size".to_string(), "4096".to_string())].into()).await.unwrap();
    assert_eq!(s.values["ctx-size"], "4096");
    assert_eq!(s.effective["n-gpu-layers"], "auto");
    let bad = h.client.configure_model("org/chat:Q4_K_M", [("model".to_string(), "/etc/passwd".to_string())].into()).await.unwrap_err();
    assert_eq!(bad.kind, ErrorKind::InvalidRequest);
    assert!(std::fs::read_to_string(h.paths.models_ini()).unwrap().contains("ctx-size = 4096"));
    let router = std::fs::read_to_string(h.paths.router_ini()).unwrap();
    assert!(router.contains("[local-chat-Q8_0]\nmodel = "), "{router}");

    h.client.load_model("org/chat:Q4_K_M", true).await.unwrap();
    assert!(h.client.list_models().await.unwrap().iter().any(|m| m.id == "org/chat:Q4_K_M" && m.pinned));
    assert!(h.client.unload_model("org/chat:Q4_K_M").await.unwrap());

    h.client.remove_model("org/other:Q4_K_M").await.unwrap();
    assert!(!h.client.list_models().await.unwrap().iter().any(|m| m.id == "org/other:Q4_K_M"));
    let err = h.client.remove_model("local-chat-Q8_0").await.unwrap_err();
    assert_eq!(err.kind, ErrorKind::PermissionDenied);
    assert!(h._dir.path().join("models/local-chat-Q8_0.gguf").exists());

    let service = h.client.configure_service(Some(2), Some(300)).await.unwrap();
    assert_eq!((service.max_loaded, service.idle_unload_seconds), (2, 300));
    assert!(h.client.configure_service(Some(99), None).await.is_err());

    let untrusted = Caller { name: "untrusted:x".into(), trusted: false };
    let op = Op::RemoveModel { model: "org/chat:Q4_K_M".into() };
    assert_eq!(h.service.authorize(&untrusted, &op).unwrap_err().kind, ErrorKind::PermissionDenied);
    assert!(h.service.authorize(&untrusted, &Op::ListModels {}).is_ok());
    h.stop().await;
}

async fn raw(socket: &Path, lines: &[String]) -> Vec<ServerLine> {
    let stream = tokio::net::UnixStream::connect(socket).await.unwrap();
    let (read, mut write) = stream.into_split();
    for line in lines {
        write.write_all(line.as_bytes()).await.unwrap();
    }
    let mut reader = BufReader::new(read).lines();
    let mut out = Vec::new();
    while let Ok(Ok(Some(line))) = tokio::time::timeout(Duration::from_millis(500), reader.next_line()).await {
        out.push(serde_json::from_str(&line).unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn protocol_errors_are_explicit() {
    let h = Harness::start().await;
    let old = raw(&h.paths.socket, &[encode(&ClientLine::Hello(Hello { api_version: 99, client: "old".into() }))]).await;
    assert!(matches!(&old[0], ServerLine::Error { error, .. } if error.kind == ErrorKind::IncompatibleVersion));

    let hello = encode(&ClientLine::Hello(Hello { api_version: 1, client: "t".into() }));
    let lines = raw(
        &h.paths.socket,
        &[
            hello,
            "{\"request\":{\"id\":1,\"op\":{\"type\":\"teleport\"}}}\n".into(),
            "{\"request\":{\"id\":2,\"op\":{\"type\":\"list_models\",\"x\":1}}}\n".into(),
            "{\"request\":{\"id\":3,\"op\":{\"type\":\"cancel\",\"request\":77}}}\n".into(),
            "not json\n".into(),
        ],
    )
    .await;
    let kinds: Vec<(Option<u64>, ErrorKind)> = lines
        .iter()
        .filter_map(|l| match l {
            ServerLine::Error { id, error } => Some((*id, error.kind)),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        [
            (Some(1), ErrorKind::UnsupportedOperation),
            (Some(2), ErrorKind::InvalidRequest),
            (Some(3), ErrorKind::UnknownRequest),
            (None, ErrorKind::InvalidRequest)
        ]
    );
    let mode = std::fs::metadata(&h.paths.socket).map(|m| std::os::unix::fs::PermissionsExt::mode(&m.permissions())).unwrap();
    assert_eq!(mode & 0o777, 0o600);
    h.stop().await;
}

//! The daemon end to end against the stub worker: socket, handshake, model
//! list, chat, embeddings, eviction, cancel, crash recovery and settings.

use nyx_ai_client::proto::{
    ActionSpec, ChatRequest, Confirmation, EmbedInput, SideEffects, ClientLine, ErrorKind, Event, Hello, Message, ModelState, Op, Priority, Role, ServerLine, encode,
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
        Harness::start_with("", None, None).await
    }

    /// `settings`: extra lines for settings.toml; `hub`: a fake Hugging Face;
    /// `power`: a fake /sys/class/power_supply.
    async fn start_with(settings: &str, hub: Option<String>, power: Option<std::path::PathBuf>) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        let folder = dir.path().join("models");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("local-chat-Q8_0.gguf"), chat_file()).unwrap();
        std::fs::create_dir_all(&paths.config).unwrap();
        std::fs::write(
            paths.settings(),
            format!("max_loaded = 1\nidle_unload_seconds = -1\nmodel_dirs = [{:?}]\n{settings}", folder.display().to_string()),
        )
        .unwrap();
        cached(&paths, "org/chat", "chat-Q4_K_M.gguf", &chat_file());
        cached(&paths, "org/embed", "embed-Q8_0.gguf", &embed_file());
        cached(&paths, "org/embed", "mmproj-embed-Q8_0.gguf", &embed_file());
        cached(&paths, "org/other", "other-Q4_K_M.gguf", &chat_file());
        cached(&paths, "org/broken", "broken-Q4_K_M.gguf", &chat_file());
        let options = Options {
            allow_untrusted: false,
            worker: Some(env!("CARGO_BIN_EXE_nyx-ai-stub-worker").into()),
            power_supply: power,
            hf_endpoint: hub,
            update_idle_secs: Some(0),
        };
        let listener = nyx_ai_core::server::listen(&paths.socket).unwrap().unix;
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
    assert_eq!(e.identity.projector.as_deref(), Some("mmproj-embed-Q8_0.gguf"));

    let media = vec![EmbedInput::from("abc"), EmbedInput::Image { image: "aGVsbG8=".into() }, EmbedInput::Audio { audio: "data:audio/wav;base64,aGk=".into() }];
    let e = h.client.embed_inputs("org/embed:Q8_0", media, Priority::Normal).await.unwrap();
    assert_eq!(e.vectors.iter().map(|v| v[0]).collect::<Vec<_>>(), [3.0, 1000.0, 1000.0]);
    let bad = h.client.embed_inputs("org/embed:Q8_0", vec![EmbedInput::Image { image: "not base64!".into() }], Priority::Normal).await;
    assert_eq!(bad.unwrap_err().kind, ErrorKind::InvalidRequest);
    let pictures = h.client.embed_inputs("org/chat:Q4_K_M", vec![EmbedInput::Image { image: "aGk=".into() }], Priority::Normal).await;
    assert_eq!(pictures.unwrap_err().kind, ErrorKind::UnsupportedCapability);
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

/// The apps of Handoff 05 at once: Chat and the Assistant chatting on one model, and
/// Glimpse indexing in the background on another. With one model loaded at a time,
/// both chats finish, the indexing waits for them rather than taking the model away
/// mid-reply, and each app's connection sees only its own requests.
#[tokio::test(flavor = "multi_thread")]
async fn chat_assistant_and_glimpse_share_ai_core() {
    let h = Harness::start().await;
    let chat = Client::with_socket("nyx-chat", &h.paths.socket);
    let assistant = Client::with_socket("nyx-assistant", &h.paths.socket);
    let glimpse = Client::with_socket("nyx-glimpse", &h.paths.socket);
    let ask_long = |text: &str| ChatRequest { model: "org/chat:Q4_K_M".into(), messages: vec![Message::text(Role::User, text)], priority: Priority::Interactive, ..Default::default() };
    // Chat's reply is under way before the others arrive.
    let mut chat_stream = chat.chat(ask_long("count:40")).await.unwrap();
    while !matches!(chat_stream.next().await.unwrap().unwrap(), Item::Event(Event::Delta { .. })) {}
    let indexing = tokio::spawn({
        let glimpse = glimpse.clone();
        async move { glimpse.embed("org/embed:Q8_0", vec!["notes".into(), "photos".into()], Priority::Background).await }
    });
    let spoken = tokio::spawn({
        let assistant = assistant.clone();
        async move { assistant.chat(ask("org/chat:Q4_K_M", "what's next")).await.unwrap().collect_chat().await }
    });
    let mut chat_text = String::new();
    let chat_done = loop {
        match chat_stream.next().await.unwrap().unwrap() {
            Item::Event(Event::Delta { content, .. }) => chat_text.push_str(&content),
            Item::Event(_) => {}
            Item::Done(reply) => break reply,
        }
    };
    assert!(matches!(chat_done, nyx_ai_client::proto::Reply::Finished(_)), "Chat's reply finished whole");
    let answer = spoken.await.unwrap().unwrap();
    assert!(!answer.content.is_empty(), "the Assistant's turn finished");
    let vectors = indexing.await.unwrap().unwrap();
    assert_eq!(vectors.vectors.len(), 2, "Glimpse's indexing ran after the chats");
    let m = h.client.worker_status().await.unwrap().metrics;
    assert!(m.evictions >= 1, "one model at a time: the embedding model replaced the chat model once it was free");
    // Cancelling is per connection: the Assistant can't cancel Chat's request.
    let mut chat_again = chat.chat(ask_long("count:200")).await.unwrap();
    while !matches!(chat_again.next().await.unwrap().unwrap(), Item::Event(Event::Delta { .. })) {}
    let stranger = raw(&h.paths.socket, &[
        encode(&ClientLine::Hello(Hello { api_version: nyx_ai_client::proto::API_VERSION, client: "nyx-assistant".into() })),
        encode(&ClientLine::Request(Box::new(nyx_ai_client::proto::Request { id: 9, op: Op::Cancel { request: 1 } }))),
    ])
    .await;
    assert!(!stranger.iter().any(|l| matches!(l, ServerLine::Error { error, .. } if error.kind == ErrorKind::Cancelled)));
    assert!(matches!(chat_again.next().await, Some(Ok(Item::Event(Event::Delta { .. })))), "Chat's reply goes on");
    chat_again.cancel().await.unwrap();
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

    // An app's own model folder: listed at once, added once, left as it is.
    let own = h._dir.path().join("app-models");
    std::fs::create_dir_all(&own).unwrap();
    std::fs::write(own.join("app-embed-Q8_0.gguf"), embed_file()).unwrap();
    let dir = own.display().to_string();
    let s = h.client.add_model_dirs(vec![dir.clone()]).await.unwrap();
    assert!(s.model_dirs.contains(&dir));
    assert!(h.client.list_models().await.unwrap().iter().any(|m| m.id == "app-embed-Q8_0" && !m.removable));
    let again = h.client.add_model_dirs(vec![dir.clone()]).await.unwrap();
    assert_eq!(again.model_dirs.iter().filter(|d| **d == dir).count(), 1);
    for bad in ["relative/models", "/no/such/folder", "~/../etc"] {
        assert_eq!(h.client.add_model_dirs(vec![bad.into()]).await.unwrap_err().kind, ErrorKind::InvalidRequest, "{bad}");
    }

    let untrusted = Caller { name: "untrusted:x".into(), trusted: false, connection: 0 };
    let op = Op::RemoveModel { model: "org/chat:Q4_K_M".into() };
    assert_eq!(h.service.authorize(&untrusted, &op).unwrap_err().kind, ErrorKind::PermissionDenied);
    assert!(h.service.authorize(&untrusted, &Op::ListModels {}).is_ok());
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_endpoint_for_coding_tools() {
    let h = Harness::start().await;
    let tcp = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    tcp.set_nonblocking(true).unwrap();
    let base = format!("http://{}", tcp.local_addr().unwrap());
    tokio::spawn(nyx_ai_core::openai::serve(h.service.clone(), tcp, h.stop.clone()));
    let http = reqwest::Client::new();

    let models: serde_json::Value = http.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().filter_map(|m| m["id"].as_str()).collect();
    assert!(ids.contains(&"org/chat:Q4_K_M") && ids.contains(&"org/embed:Q8_0"), "{ids:?}");

    let chat = serde_json::json!({"model": "org/chat:Q4_K_M", "stream": true, "messages": [{"role": "user", "content": "hi"}]});
    let reply = http.post(format!("{base}/v1/chat/completions")).json(&chat).send().await.unwrap();
    assert_eq!(reply.status(), 200);
    assert_eq!(reply.headers()["content-type"], "text/event-stream");
    let text = reply.text().await.unwrap();
    assert!(text.contains("tok4") && text.contains("[DONE]"), "{text}");

    let status = |body: serde_json::Value, path: &'static str| {
        let (http, base) = (http.clone(), base.clone());
        async move { http.post(format!("{base}{path}")).json(&body).send().await.unwrap().status().as_u16() }
    };
    assert_eq!(status(serde_json::json!({"messages": []}), "/v1/chat/completions").await, 400);
    assert_eq!(status(serde_json::json!({"model": "org/none:Q4_0", "messages": []}), "/v1/chat/completions").await, 404);
    assert_eq!(status(serde_json::json!({"model": "org/chat:Q4_K_M", "input": ["x"]}), "/v1/embeddings").await, 400);
    assert_eq!(status(serde_json::json!({"model": "org/embed:Q8_0", "input": ["x"]}), "/v1/embeddings").await, 200);
    assert_eq!(http.get(format!("{base}/v1/secret")).send().await.unwrap().status(), 400);

    // Without a model, the coding default answers; defaults must fit their role.
    let unnamed = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
    assert_eq!(status(unnamed.clone(), "/v1/chat/completions").await, 400);
    let set = |role: &str, model: &str| nyx_ai_client::proto::Op::ConfigureService {
        max_loaded: None,
        idle_unload_seconds: None,
        model_updates: None,
        defaults: Some([(role.to_string(), model.to_string())].into()),
        battery_saver: None,
        add_model_dirs: None,
    };
    let s = h.client.configure_service_with(set("coding", "org/chat:Q4_K_M")).await.unwrap();
    assert_eq!(s.defaults["coding"], "org/chat:Q4_K_M");
    assert_eq!(status(unnamed, "/v1/chat/completions").await, 200);
    assert_eq!(h.client.configure_service_with(set("embedding", "org/chat:Q4_K_M")).await.unwrap_err().kind, ErrorKind::UnsupportedCapability);
    assert_eq!(h.client.configure_service_with(set("chat", "org/none:Q4_0")).await.unwrap_err().kind, ErrorKind::UnknownModel);
    assert_eq!(h.client.configure_service_with(set("lunch", "org/chat:Q4_K_M")).await.unwrap_err().kind, ErrorKind::InvalidRequest);
    assert!(h.client.configure_service_with(set("coding", "")).await.unwrap().defaults.is_empty());

    let slow = serde_json::json!({"model": "org/chat:Q4_K_M", "stream": true, "messages": [{"role": "user", "content": "count:500"}]});
    let mut reply = http.post(format!("{base}/v1/chat/completions")).json(&slow).send().await.unwrap();
    assert!(reply.chunk().await.unwrap().is_some());
    drop(reply);
    let mut idle = false;
    for _ in 0..50 {
        if h.client.resource_status().await.unwrap().loaded.iter().all(|m| m.active == 0) {
            idle = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(idle, "a dropped request kept its slot");
    h.stop().await;
}

#[path = "../src/testhub.rs"]
mod testhub;

async fn finish(stream: nyx_ai_client::Result<nyx_ai_client::Stream>) {
    let mut stream = stream.unwrap();
    while let Some(item) = stream.next().await {
        if let Item::Done(_) = item.unwrap() {
            return;
        }
    }
    panic!("no result");
}

fn revision(paths: &Paths) -> String {
    std::fs::read_to_string(paths.cache.join("models--o--r/refs/main")).unwrap_or_default()
}

fn power(dir: &Path, mains: bool) {
    for (name, kind, online) in [("BAT0", "Battery", "0"), ("AC", "Mains", if mains { "1" } else { "0" })] {
        std::fs::create_dir_all(dir.join(name)).unwrap();
        std::fs::write(dir.join(name).join("type"), kind).unwrap();
        std::fs::write(dir.join(name).join("online"), online).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn updates_wait_for_a_free_model_and_for_mains_power() {
    let (endpoint, hub) = testhub::fake_hub().await;
    let (v1, mut v2, mut v3) = (chat_file(), chat_file(), chat_file());
    v2.extend(b"two");
    v3.extend(b"three!");
    hub.set("a", &[("m-Q4_0.gguf", &v1)]);
    let supply = tempfile::tempdir().unwrap();
    power(supply.path(), false);
    let h = Harness::start_with("model_updates = \"auto\"\n", Some(endpoint), Some(supply.path().into())).await;

    finish(h.client.download_model("o/r:Q4_0").await).await;
    assert_eq!(revision(&h.paths), "a".repeat(40));
    assert!(h.client.list_models().await.unwrap().iter().any(|m| m.id == "o/r:Q4_0" && m.has(nyx_ai_client::proto::Capability::Chat)));
    assert!(h.client.check_updates().await.unwrap().is_empty());

    hub.set("b", &[("m-Q4_0.gguf", &v2)]);
    let found = h.client.check_updates().await.unwrap();
    assert_eq!(found.iter().map(|m| (m.id.as_str(), m.update_size)).collect::<Vec<_>>(), [("o/r:Q4_0", Some(v2.len() as u64))]);

    // On battery the updater waits.
    h.service.auto_update();
    assert!(h.client.downloads().await.unwrap().iter().all(|d| d.model != "o/r:Q4_0" || d.state != nyx_ai_client::proto::DownloadState::Running));

    // A chat holds the model; an update fetched meanwhile waits for it.
    let mut chat = h.client.chat(ask("o/r:Q4_0", "count:300")).await.unwrap();
    while !matches!(chat.next().await.unwrap().unwrap(), Item::Event(Event::Delta { .. })) {}
    finish(h.client.download_model("o/r:Q4_0").await).await;
    assert_eq!(revision(&h.paths), "a".repeat(40), "switched under a running chat");
    drop(chat);
    for _ in 0..50 {
        if h.client.resource_status().await.unwrap().loaded.iter().all(|m| m.active == 0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    h.service.apply_staged().await;
    assert_eq!(revision(&h.paths), "b".repeat(40));
    let file = h.paths.cache.join("models--o--r/snapshots").join("b".repeat(40)).join("m-Q4_0.gguf");
    assert_eq!(std::fs::read(file).unwrap(), v2);
    assert!(!h.client.list_models().await.unwrap().iter().any(|m| m.update_available));

    // On mains power and idle, the updater fetches and switches by itself.
    power(supply.path(), true);
    hub.set("c", &[("m-Q4_0.gguf", &v3)]);
    assert_eq!(h.client.check_updates().await.unwrap().len(), 1);
    h.service.auto_update();
    let mut switched = false;
    for _ in 0..100 {
        h.service.apply_staged().await;
        if revision(&h.paths) == "c".repeat(40) {
            switched = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(switched, "the automatic update didn't land");
    let blobs = std::fs::read_dir(h.paths.cache.join("models--o--r/blobs")).unwrap().count();
    assert_eq!(blobs, 1, "replaced revisions are pruned");
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

fn action(id: &str, side_effects: SideEffects, confirmation: Confirmation) -> ActionSpec {
    ActionSpec {
        id: id.into(),
        description: format!("test {id}"),
        input_schema: serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        output_schema: None,
        permissions: vec!["notes.read".into()],
        confirmation,
        side_effects,
        timeout_ms: Some(5000),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn apps_offer_actions_and_calls_are_checked() {
    use serde_json::json;
    let h = Harness::start().await;
    let specs = vec![action("notes.find", SideEffects::ReadOnly, Confirmation::Never), action("notes.create", SideEffects::Writes, Confirmation::Ask)];
    let mut server = h.client.serve_actions(specs.clone()).await.unwrap();
    let provider = tokio::spawn(async move {
        while let Some(Ok(call)) = server.next().await {
            let outcome = match call.action.as_str() {
                "notes.find" => Ok(json!({"found": call.arguments["text"], "caller": call.caller})),
                _ => Err("the notebook is locked".to_string()),
            };
            server.answer(call.id, outcome).await.unwrap();
        }
    });
    let mut listed = Vec::new();
    for _ in 0..100 {
        listed = h.client.list_actions().await.unwrap();
        if listed.iter().any(|a| a.spec.id == "notes.find") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let find = listed.iter().find(|a| a.spec.id == "notes.find").expect("listed");
    assert_eq!(find.source, "connection");
    assert_eq!(find.spec.permissions, vec!["notes.read".to_string()]);

    let found = h.client.invoke_action("notes.find", json!({"text": "milk"}), false).await.unwrap();
    assert_eq!(found["found"], "milk");
    let kind = |r: nyx_ai_client::Result<serde_json::Value>| r.unwrap_err().kind;
    assert_eq!(kind(h.client.invoke_action("notes.find", json!({"text": 3}), false).await), ErrorKind::InvalidRequest);
    assert_eq!(kind(h.client.invoke_action("notes.create", json!({"text": "milk"}), false).await), ErrorKind::ConfirmationRequired);
    let refused = h.client.invoke_action("notes.create", json!({"text": "milk"}), true).await.unwrap_err();
    assert_eq!((refused.kind, refused.message.as_str()), (ErrorKind::ActionFailed, "the notebook is locked"));
    assert_eq!(kind(h.client.invoke_action("notes.nothing", json!({}), true).await), ErrorKind::UnknownAction);

    let taken = h.client.serve_actions(specs).await.unwrap().next().await.unwrap().unwrap_err();
    assert_eq!(taken.kind, ErrorKind::InvalidRequest);

    provider.abort();
    let _ = provider.await;
    let mut gone = false;
    for _ in 0..100 {
        if !h.client.list_actions().await.unwrap().iter().any(|a| a.spec.id == "notes.find") {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(gone, "a closed connection withdraws its actions");
    h.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn battery_saver_moves_models_to_the_cpu_on_battery() {
    let power_dir = tempfile::tempdir().unwrap();
    power(power_dir.path(), true);
    let h = Harness::start_with("", None, Some(power_dir.path().to_path_buf())).await;
    let saver = |on: bool| Op::ConfigureService { max_loaded: None, idle_unload_seconds: None, model_updates: None, defaults: None, battery_saver: Some(on), add_model_dirs: None };
    h.client.chat(ask("org/chat:Q4_K_M", "hi")).await.unwrap().collect_chat().await.unwrap();

    let s = h.client.configure_service_with(saver(true)).await.unwrap();
    assert!(s.battery_saver && !s.on_battery && !s.cpu_only);
    assert_eq!(h.client.resource_status().await.unwrap().loaded.len(), 1, "on mains nothing changes");

    power(power_dir.path(), false);
    h.service.apply_power().await;
    let s = h.client.service_settings().await.unwrap();
    assert!(s.on_battery && s.cpu_only);
    assert!(std::fs::read_to_string(h.paths.router_ini()).unwrap().contains("n-gpu-layers = 0"));
    assert!(h.client.resource_status().await.unwrap().loaded.is_empty(), "the idle model unloads to come back on the CPU");

    power(power_dir.path(), true);
    h.service.apply_power().await;
    assert!(!h.client.service_settings().await.unwrap().cpu_only);
    assert!(!std::fs::read_to_string(h.paths.router_ini()).unwrap().contains("n-gpu-layers = 0"));
    h.stop().await;
}

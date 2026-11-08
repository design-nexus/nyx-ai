//! A stand-in for llama-server's router, for tests: the same flags and
//! endpoints, no models. Cache models are the `.gguf` files under `LLAMA_CACHE`
//! (`models--org--repo/snapshots/REV/name-QUANT.gguf` lists as `org/repo:QUANT`);
//! preset models come from the `--models-preset` file's `model =` lines. A model
//! whose ID contains `broken` fails to load.
//!
//! A chat whose last message contains `count:N` streams N tokens; `crash` makes
//! the process exit mid-reply. `STUB_DELAY_MS` sets the time between tokens and
//! `STUB_SLOTS` the slots per model.

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone)]
struct Model {
    path: String,
    source: &'static str,
    status: &'static str,
    failed: bool,
}

struct Stub {
    key: String,
    preset: Option<PathBuf>,
    models: Mutex<BTreeMap<String, Model>>,
}

type Body = UnsyncBoxBody<Bytes, Infallible>;

fn reply(status: StatusCode, value: Value) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(value.to_string())).boxed_unsync())
        .expect("valid response")
}

fn error(status: StatusCode, message: &str) -> Response<Body> {
    reply(status, json!({"error": {"code": status.as_u16(), "message": message}}))
}

impl Stub {
    fn read_models(&self) {
        let mut models = self.models.lock().unwrap();
        let old = models.clone();
        models.clear();
        let cache = PathBuf::from(std::env::var_os("LLAMA_CACHE").unwrap_or_default());
        for repo in std::fs::read_dir(&cache).into_iter().flatten().flatten() {
            let name = repo.file_name().to_string_lossy().into_owned();
            let Some(rest) = name.strip_prefix("models--") else { continue };
            let repo_id = rest.replacen("--", "/", 1);
            for rev in std::fs::read_dir(repo.path().join("snapshots")).into_iter().flatten().flatten() {
                for file in std::fs::read_dir(rev.path()).into_iter().flatten().flatten() {
                    let file_name = file.file_name().to_string_lossy().into_owned();
                    let Some(stem) = file_name.strip_suffix(".gguf") else { continue };
                    if stem.contains("mmproj") {
                        continue;
                    }
                    let quant = stem.rsplit('-').next().unwrap_or(stem).to_ascii_uppercase();
                    let path = file.path().display().to_string();
                    models.insert(format!("{repo_id}:{quant}"), Model { path, source: "cache", status: "unloaded", failed: false });
                }
            }
        }
        if let Some(preset) = &self.preset {
            let text = std::fs::read_to_string(preset).unwrap_or_default();
            let mut section = String::new();
            for line in text.lines() {
                let line = line.trim();
                if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                    section = name.into();
                } else if let Some(path) = line.strip_prefix("model =") {
                    models.insert(section.clone(), Model { path: path.trim().into(), source: "preset", status: "unloaded", failed: false });
                }
            }
        }
        for (id, m) in models.iter_mut() {
            if let Some(o) = old.get(id) {
                m.status = o.status;
                m.failed = o.failed;
            }
        }
    }

    fn list(&self) -> Value {
        let models = self.models.lock().unwrap();
        let data: Vec<Value> = models
            .iter()
            .map(|(id, m)| {
                let mut status = json!({"value": m.status});
                if m.failed {
                    status["failed"] = json!(true);
                    status["exit_code"] = json!(1);
                }
                json!({"id": id, "source": m.source, "status": status,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}})
            })
            .collect();
        json!({"data": data})
    }

    fn set(&self, id: &str, status: &'static str, failed: bool) {
        if let Some(m) = self.models.lock().unwrap().get_mut(id) {
            m.status = status;
            m.failed = failed;
        }
    }

    fn status(&self, id: &str) -> Option<&'static str> {
        self.models.lock().unwrap().get(id).map(|m| m.status)
    }
}

async fn handle(stub: Arc<Stub>, req: Request<Incoming>) -> Result<Response<Body>, Infallible> {
    let auth = req.headers().get("authorization").and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
    if auth != format!("Bearer {}", stub.key) {
        return Ok(error(StatusCode::UNAUTHORIZED, "Invalid API Key"));
    }
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();
    let query = uri.query().unwrap_or_default().to_string();
    let param = |name: &str| {
        query.split('&').find_map(|kv| kv.strip_prefix(&format!("{name}="))).map(|v| {
            let mut out = Vec::new();
            let b = v.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'%' && i + 2 < b.len() {
                    out.push(u8::from_str_radix(&v[i + 1..i + 3], 16).unwrap_or(b'?'));
                    i += 3;
                } else {
                    out.push(b[i]);
                    i += 1;
                }
            }
            String::from_utf8_lossy(&out).into_owned()
        })
    };
    let body: Value = req.into_body().collect().await.map(|b| serde_json::from_slice(&b.to_bytes()).unwrap_or(Value::Null)).unwrap_or(Value::Null);
    let model = body["model"].as_str().map(String::from).or_else(|| param("model")).unwrap_or_default();
    Ok(match (method.as_str(), path.as_str()) {
        ("GET", "/health") => reply(StatusCode::OK, json!({"status": "ok"})),
        ("GET", "/models") => {
            if param("reload").is_some() {
                stub.read_models();
            }
            reply(StatusCode::OK, stub.list())
        }
        ("POST", "/models/load") => {
            if stub.status(&model).is_none() {
                return Ok(error(StatusCode::NOT_FOUND, "model not found"));
            }
            stub.set(&model, "loading", false);
            let stub = stub.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(80)).await;
                if model.contains("broken") {
                    stub.set(&model, "unloaded", true);
                } else {
                    stub.set(&model, "loaded", false);
                }
            });
            reply(StatusCode::OK, json!({"success": true}))
        }
        ("POST", "/models/unload") => {
            stub.set(&model, "unloaded", false);
            reply(StatusCode::OK, json!({"success": true}))
        }
        ("DELETE", "/models") => {
            let found = stub.models.lock().unwrap().get(&model).cloned();
            match found {
                Some(m) if m.source == "cache" => {
                    let _ = std::fs::remove_file(&m.path);
                    stub.models.lock().unwrap().remove(&model);
                    reply(StatusCode::OK, json!({"success": true}))
                }
                _ => error(StatusCode::BAD_REQUEST, "only cached models can be deleted"),
            }
        }
        ("GET", "/props") => {
            let slots: u64 = std::env::var("STUB_SLOTS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            reply(StatusCode::OK, json!({"total_slots": slots, "default_generation_settings": {"n_ctx": 4096},
                "chat_template_caps": {"supports_reasoning_effort": false}}))
        }
        ("POST", "/v1/chat/completions") => {
            if stub.status(&model) != Some("loaded") {
                return Ok(error(StatusCode::BAD_REQUEST, "model is not loaded"));
            }
            let last = body["messages"].as_array().and_then(|m| m.last()).map(|m| m["content"].to_string()).unwrap_or_default();
            let count: usize = last.split("count:").nth(1).and_then(|s| s.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()).unwrap_or(5);
            let crash = last.contains("crash");
            let delay = std::env::var("STUB_DELAY_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
            let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(4);
            tokio::spawn(async move {
                for i in 0..count {
                    let chunk = json!({"choices": [{"delta": {"content": format!("tok{i} ")}}]});
                    if tx.send(Bytes::from(format!("data: {chunk}\n\n"))).await.is_err() {
                        return;
                    }
                    if crash {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        std::process::exit(3);
                    }
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                let end = json!({"choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": count}, "timings": {"prompt_ms": 3.0, "predicted_per_second": 100.0}});
                let _ = tx.send(Bytes::from(format!("data: {end}\n\ndata: [DONE]\n\n"))).await;
            });
            let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<_, Infallible>(Frame::data(b)), rx)) });
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(StreamBody::new(stream.boxed()).boxed_unsync())
                .expect("valid response")
        }
        ("POST", "/v1/embeddings") => {
            if stub.status(&model) != Some("loaded") {
                return Ok(error(StatusCode::BAD_REQUEST, "model is not loaded"));
            }
            let mut data: Vec<Value> = body["input"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(i, text)| {
                    let n = text.as_str().unwrap_or_default().len() as f32;
                    json!({"index": i, "embedding": [n, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]})
                })
                .collect();
            data.reverse();
            reply(StatusCode::OK, json!({"data": data}))
        }
        ("POST", "/apply-template") => reply(StatusCode::OK, json!({"prompt": "<stub template>"})),
        _ => error(StatusCode::NOT_FOUND, "not found"),
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let host = arg("--host").expect("--host");
    let key = arg("--api-key-file").and_then(|f| std::fs::read_to_string(f).ok()).unwrap_or_default().trim().to_string();
    let stub = Arc::new(Stub { key, preset: arg("--models-preset").map(PathBuf::from), models: Mutex::new(BTreeMap::new()) });
    stub.read_models();
    let _ = std::fs::remove_file(&host);
    let listener = tokio::net::UnixListener::bind(&host).expect("bind");
    loop {
        let Ok((stream, _)) = listener.accept().await else { continue };
        let stub = stub.clone();
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |req| handle(stub.clone(), req));
            let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), service).await;
        });
    }
}

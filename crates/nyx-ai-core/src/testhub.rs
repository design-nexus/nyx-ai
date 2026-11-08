//! A Hugging Face stand-in for tests (the unit tests and `tests/service.rs`).
#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// (revision, [(path, bytes, oid the tree reports)])
type Repo = (String, Vec<(String, Vec<u8>, String)>);

/// A Hugging Face stand-in serving one repository, `o/r`, whose revision
/// and files a test can change.
#[derive(Clone, Default)]
pub struct Hub {
    state: Arc<Mutex<Repo>>,
    pub ranges: Arc<AtomicUsize>,
    pub file_gets: Arc<AtomicUsize>,
}

impl Hub {
    pub fn set(&self, revision: &str, files: &[(&str, &[u8])]) {
        let files = files.iter().map(|(p, b)| (p.to_string(), b.to_vec(), format!("{:x}", Sha256::digest(b)))).collect();
        *self.state.lock().unwrap() = (revision.repeat(40 / revision.len()), files);
    }

    pub fn lie_about(&self, path: &str) {
        for f in self.state.lock().unwrap().1.iter_mut().filter(|f| f.0 == path) {
            f.2 = "b".repeat(64);
        }
    }
}

pub async fn fake_hub() -> (String, Hub) {
    use http_body_util::Full;
    use hyper::{Response, body::Bytes};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hub = Hub::default();
    let shared = hub.clone();
    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            let hub = shared.clone();
            tokio::spawn(async move {
                let service = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    let hub = hub.clone();
                    async move {
                        let path = req.uri().path().to_string();
                        let (sha, files) = hub.state.lock().unwrap().clone();
                        let not_found = || {
                            let mut r = Response::new(Full::new(Bytes::new()));
                            *r.status_mut() = hyper::StatusCode::NOT_FOUND;
                            r
                        };
                        let reply = if path == "/api/models/o/r" {
                            Response::new(Full::new(Bytes::from(format!("{{\"sha\":\"{sha}\"}}"))))
                        } else if path == format!("/api/models/o/r/tree/{sha}") {
                            let mut tree: Vec<serde_json::Value> = files
                                .iter()
                                .map(|(p, b, oid)| serde_json::json!({"path": p, "size": b.len(), "lfs": {"oid": oid}}))
                                .collect();
                            tree.push(serde_json::json!({"path": "README.md", "size": 3}));
                            Response::new(Full::new(Bytes::from(serde_json::to_vec(&tree).unwrap())))
                        } else if let Some(name) = path.strip_prefix(&format!("/o/r/resolve/{sha}/")) {
                            match files.iter().find(|f| f.0 == name) {
                                Some((_, body, _)) => {
                                    hub.file_gets.fetch_add(1, Ordering::SeqCst);
                                    let start = req
                                        .headers()
                                        .get("range")
                                        .and_then(|r| r.to_str().ok()?.strip_prefix("bytes=")?.strip_suffix('-')?.parse::<usize>().ok());
                                    match start {
                                        Some(start) => {
                                            hub.ranges.fetch_add(1, Ordering::SeqCst);
                                            let mut r = Response::new(Full::new(Bytes::from(body[start..].to_vec())));
                                            *r.status_mut() = hyper::StatusCode::PARTIAL_CONTENT;
                                            r
                                        }
                                        None => Response::new(Full::new(Bytes::from(body.clone()))),
                                    }
                                }
                                None => not_found(),
                            }
                        } else {
                            not_found()
                        };
                        Ok::<_, std::convert::Infallible>(reply)
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new().serve_connection(hyper_util::rt::TokioIo::new(stream), service).await;
            });
        }
    });
    (format!("http://{addr}"), hub)
}

pub fn gguf_bytes(seed: u8, len: usize) -> Vec<u8> {
    let mut body = b"GGUF\x03\0\0\0".to_vec();
    body.extend((0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)));
    body
}


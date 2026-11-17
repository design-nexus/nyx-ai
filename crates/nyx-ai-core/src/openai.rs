//! An OpenAI-compatible endpoint for coding tools, on loopback. Requests go
//! through the scheduler like any app's (loading the model, waiting their turn)
//! and on to the worker unchanged; the worker itself stays private.
//!
//! Requests need the API key (`Authorization: Bearer KEY`) and a loopback `Host`,
//! and POSTs need a JSON body, so web pages can't reach it through the browser.

use crate::service::Service;
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, Full, Limited, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use nyx_ai_proto::{Error, ErrorKind, MAX_LINE};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

type Body = UnsyncBoxBody<Bytes, Infallible>;

fn json_reply(status: StatusCode, value: &Value) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(value.to_string())).boxed_unsync())
        .expect("valid response")
}

fn error_reply(e: &Error) -> Response<Body> {
    let status = match e.kind {
        ErrorKind::InvalidRequest | ErrorKind::UnsupportedCapability => StatusCode::BAD_REQUEST,
        ErrorKind::UnknownModel => StatusCode::NOT_FOUND,
        ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
        ErrorKind::ResourceExhausted | ErrorKind::Busy | ErrorKind::WorkerUnavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        ErrorKind::Timeout => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json_reply(
        status,
        &json!({"error": {"code": status.as_u16(), "message": e.message, "type": kind}}),
    )
}

fn refusal(status: StatusCode, message: &str) -> Response<Body> {
    let kind = if status == StatusCode::UNAUTHORIZED {
        "authentication_error"
    } else {
        "invalid_request_error"
    };
    json_reply(
        status,
        &json!({"error": {"code": status.as_u16(), "message": message, "type": kind}}),
    )
}

/// Why `req` is refused before it's read: a `Host` that isn't loopback (DNS
/// rebinding), a missing or wrong key, or a POST that isn't JSON.
fn refuse(service: &Service, req: &Request<Incoming>) -> Option<Response<Body>> {
    let header = |name: hyper::header::HeaderName| {
        req.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
    };
    let host = header(hyper::header::HOST);
    let name = match host.rsplit_once(':') {
        Some((name, port)) if !name.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => host,
    };
    if !matches!(name, "127.0.0.1" | "localhost" | "[::1]") {
        return Some(refusal(
            StatusCode::FORBIDDEN,
            "only local coding tools can use this endpoint",
        ));
    }
    if req.uri().path() == "/health" {
        return None;
    }
    let sent = header(hyper::header::AUTHORIZATION)
        .strip_prefix("Bearer ")
        .unwrap_or("")
        .trim();
    let Some(key) = service.openai_key() else {
        return Some(refusal(
            StatusCode::UNAUTHORIZED,
            "the endpoint has no API key yet",
        ));
    };
    let same = sent.len() == key.len()
        && sent
            .bytes()
            .zip(key.bytes())
            .fold(0u8, |d, (a, b)| d | (a ^ b))
            == 0;
    if !same {
        return Some(refusal(
            StatusCode::UNAUTHORIZED,
            "send the API key from ~/.config/nyx-ai/api-key as a Bearer token",
        ));
    }
    if req.method() == hyper::Method::POST
        && !header(hyper::header::CONTENT_TYPE).starts_with("application/json")
    {
        return Some(refusal(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "the request body must be JSON (Content-Type: application/json)",
        ));
    }
    None
}

async fn handle(
    service: Arc<Service>,
    req: Request<Incoming>,
    stop: CancellationToken,
) -> Result<Response<Body>, Infallible> {
    if let Some(refused) = refuse(&service, &req) {
        return Ok(refused);
    }
    let method = req.method().clone();
    let path = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    let bare = req.uri().path().to_string();
    match (method.as_str(), bare.as_str()) {
        ("GET", "/health") => return Ok(json_reply(StatusCode::OK, &json!({"status": "ok"}))),
        ("GET", "/v1/models" | "/models") => {
            return Ok(json_reply(StatusCode::OK, &service.openai_models().await));
        }
        ("POST", p) if p.starts_with("/v1/") && !p.contains("..") => {}
        _ => {
            return Ok(error_reply(&Error::new(
                ErrorKind::InvalidRequest,
                format!("{method} {bare} isn't served here"),
            )));
        }
    }
    let body = match Limited::new(req.into_body(), MAX_LINE).collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return Ok(error_reply(&Error::new(
                ErrorKind::InvalidRequest,
                format!("couldn't read the request: {e}"),
            )));
        }
    };
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => {
            return Ok(error_reply(&Error::new(
                ErrorKind::InvalidRequest,
                "the request body must be a JSON object",
            )));
        }
    };
    let cancel = stop.child_token();
    let (mut upstream, lease) = match service.forward(&path, &body, &cancel).await {
        Ok(pair) => pair,
        Err(e) => return Ok(error_reply(&e)),
    };
    let status = StatusCode::from_u16(upstream.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .content_type
        .clone()
        .unwrap_or_else(|| "application/json".into());
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(16);
    let mut timings = Timings::new(&content_type);
    let recorder = service.clone();
    tokio::spawn(async move {
        let _lease = lease;
        loop {
            let chunk = tokio::select! {
                c = upstream.chunk() => c,
                _ = tx.closed() => break,
                _ = cancel.cancelled() => break,
            };
            match chunk {
                Some(Ok(bytes)) => {
                    timings.push(&bytes);
                    if tx.send(bytes).await.is_err() {
                        break;
                    }
                }
                _ => break,
            }
        }
        // The worker's own timings, as for requests made through AI Core's socket, so
        // `nyx-ai status` counts these too.
        if let Some((prompt_ms, per_second)) = timings.finish() {
            recorder.forwarded(prompt_ms, per_second);
        }
    });
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|b| (Ok::<_, Infallible>(Frame::data(b)), rx))
    });
    Ok(Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(StreamBody::new(stream.boxed()).boxed_unsync())
        .expect("valid response"))
}

/// Serve until `stop` is cancelled. Only loopback peers are answered.
/// llama.cpp's `timings` in a forwarded reply: in the stream's events, or in the one
/// JSON reply (kept up to `MAX_LINE`; a longer one isn't timed).
enum Timings {
    Stream(crate::stream::Sse),
    Json(Vec<u8>),
    Skip,
}

impl Timings {
    fn new(content_type: &str) -> Self {
        if content_type.starts_with("text/event-stream") {
            Self::Stream(crate::stream::Sse::default())
        } else {
            Self::Json(Vec::new())
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        match self {
            Self::Stream(sse) => {
                sse.push(chunk);
            }
            Self::Json(body) if body.len() + chunk.len() <= MAX_LINE => {
                body.extend_from_slice(chunk)
            }
            Self::Json(_) => *self = Self::Skip,
            Self::Skip => {}
        }
    }

    fn finish(self) -> Option<(f64, f64)> {
        match self {
            Self::Stream(sse) => sse.timings,
            Self::Json(body) => {
                let value: Value = serde_json::from_slice(&body).ok()?;
                let t = value["timings"].as_object()?;
                Some((
                    t.get("prompt_ms").and_then(Value::as_f64).unwrap_or(0.0),
                    t.get("predicted_per_second")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                ))
            }
            Self::Skip => None,
        }
    }
}

pub async fn serve(
    service: Arc<Service>,
    listener: std::net::TcpListener,
    stop: CancellationToken,
) {
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "can't use the OpenAI endpoint's socket");
            return;
        }
    };
    if let Ok(addr) = listener.local_addr() {
        tracing::info!(%addr, "OpenAI-compatible endpoint ready");
    }
    loop {
        let (stream, peer) = tokio::select! {
            a = listener.accept() => match a {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(error = %e, "accept failed");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
            },
            _ = stop.cancelled() => break,
        };
        if !peer.ip().is_loopback() {
            continue;
        }
        let (service, stop) = (service.clone(), stop.clone());
        tokio::spawn(async move {
            let conn_stop = stop.child_token();
            let svc = hyper::service::service_fn(move |req| {
                handle(service.clone(), req, conn_stop.clone())
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), svc)
                .await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwarded_stream_timings_are_read() {
        let mut t = Timings::new("text/event-stream; charset=utf-8");
        t.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_");
        t.push(b"reason\":\"stop\"}],\"timings\":{\"prompt_ms\":21.5,\"predicted_per_second\":33.0}}\n\ndata: [DONE]\n\n");
        assert_eq!(t.finish(), Some((21.5, 33.0)));
    }

    #[test]
    fn forwarded_json_timings_are_read() {
        let mut t = Timings::new("application/json");
        t.push(b"{\"choices\":[],\"timings\":{\"prompt_ms\":");
        t.push(b"8.0,\"predicted_per_second\":50.5}}");
        assert_eq!(t.finish(), Some((8.0, 50.5)));
    }

    #[test]
    fn replies_without_timings_or_too_long_record_nothing() {
        let mut t = Timings::new("application/json");
        t.push(b"{\"data\":[]}");
        assert_eq!(t.finish(), None);
        let mut t = Timings::new("application/json");
        t.push(&vec![b' '; MAX_LINE]);
        t.push(b"{\"timings\":{\"prompt_ms\":1.0}}");
        assert_eq!(t.finish(), None);
    }
}

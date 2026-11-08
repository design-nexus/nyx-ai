//! `nyx-aid`: Nyx AI Core's daemon. Started by `nyx-aid.socket` the first time
//! a client connects, or directly.
//!
//! It also serves an OpenAI-compatible API for coding tools at
//! `openai_endpoint` in settings.toml (127.0.0.1:8080 unless turned off).
//!
//! `NYX_AI_SOCKET` moves the socket (test copies), `NYX_AI_WORKER` uses another
//! llama-server, and `NYX_AI_ALLOW_UNTRUSTED=1` lets programs outside the trusted
//! folders change settings and remove models.

use nyx_ai_core::paths::Paths;
use nyx_ai_core::server;
use nyx_ai_core::service::{Options, Service};
use std::process::ExitCode;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .without_time()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_target(false)
        .init();
    if std::env::args().any(|a| a == "--version") {
        println!("nyx-aid {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let paths = Paths::from_env();
    let options = Options {
        allow_untrusted: std::env::var("NYX_AI_ALLOW_UNTRUSTED").is_ok_and(|v| v == "1"),
        worker: std::env::var_os("NYX_AI_WORKER").filter(|v| !v.is_empty()).map(Into::into),
    };
    let listeners = match server::listen(&paths.socket) {
        Ok(listeners) => listeners,
        Err(e) => {
            tracing::error!(error = %e, socket = %paths.socket.display(), "can't listen");
            return ExitCode::FAILURE;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            tracing::error!(error = %e, "no runtime");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async move {
        let service = Service::new(paths, options);
        let stop = CancellationToken::new();
        service.run_background(stop.clone());
        let server = tokio::spawn(server::serve(service.clone(), listeners.unix, stop.clone()));
        let openai = match (service.openai_addr(), listeners.tcp) {
            (Some(_), Some(tcp)) => Some(tcp),
            (Some(addr), None) => std::net::TcpListener::bind(addr)
                .and_then(|l| l.set_nonblocking(true).map(|_| l))
                .map_err(|e| tracing::warn!(%addr, error = %e, "the OpenAI-compatible endpoint can't listen"))
                .ok(),
            (None, _) => None,
        };
        if let Some(tcp) = openai {
            tokio::spawn(nyx_ai_core::openai::serve(service.clone(), tcp, stop.clone()));
        }
        tracing::info!(socket = %service.paths().socket.display(), "Nyx AI Core ready");
        let (Ok(mut term), Ok(mut int)) = (signal(SignalKind::terminate()), signal(SignalKind::interrupt())) else {
            tracing::error!("can't watch signals");
            return ExitCode::FAILURE;
        };
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
        tracing::info!("stopping");
        stop.cancel();
        service.shutdown().await;
        let _ = server.await;
        ExitCode::SUCCESS
    })
}

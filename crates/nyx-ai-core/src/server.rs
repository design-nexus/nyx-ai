//! The socket: same-user peers only, a versioned handshake, then requests that
//! run side by side and end with exactly one result or error each.

use crate::service::{Caller, Service};
use nyx_ai_proto::{
    API_VERSION, ClientLine, Error, ErrorKind, Event, MAX_LINE, Op, Reply, Request, ServerLine, Welcome, decode_client_line, encode,
};
use std::collections::HashMap;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::OwnedReadHalf;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

const MAX_CONNECTIONS: usize = 64;
const MAX_REQUESTS: usize = 32;

/// What the daemon listens on: its Unix socket, and the TCP socket for the
/// OpenAI-compatible endpoint when systemd passed one.
pub struct Listeners {
    pub unix: std::os::unix::net::UnixListener,
    pub tcp: Option<std::net::TcpListener>,
}

/// The sockets systemd passed (socket activation), or a new Unix socket at
/// `path`. Call before starting threads: it clears systemd's variables.
pub fn listen(path: &Path) -> std::io::Result<Listeners> {
    let ours = std::env::var("LISTEN_PID").ok().and_then(|p| p.parse::<u32>().ok()) == Some(std::process::id());
    let count = std::env::var("LISTEN_FDS").ok().and_then(|n| n.parse::<i32>().ok()).unwrap_or(0);
    // SAFETY: these variables are only read here, before any other thread looks at them.
    unsafe {
        std::env::remove_var("LISTEN_PID");
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_FDNAMES");
    }
    if ours && count >= 1 {
        use std::os::fd::FromRawFd;
        let (mut unix, mut tcp) = (None, None);
        for fd in 3..3 + count {
            // systemd passes them inheritable; the worker must not keep the port open.
            // SAFETY: fcntl on an fd number only changes that descriptor's flags.
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
            // SAFETY: a zeroed sockaddr_storage is valid, and getsockname fills it for the fd systemd passed.
            let family = unsafe {
                let mut addr: libc::sockaddr_storage = std::mem::zeroed();
                let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                if libc::getsockname(fd, &mut addr as *mut _ as *mut libc::sockaddr, &mut len) != 0 {
                    continue;
                }
                addr.ss_family as i32
            };
            // SAFETY: systemd hands these fds to us as listening sockets.
            match family {
                libc::AF_UNIX if unix.is_none() => unix = Some(unsafe { std::os::unix::net::UnixListener::from_raw_fd(fd) }),
                libc::AF_INET | libc::AF_INET6 if tcp.is_none() => tcp = Some(unsafe { std::net::TcpListener::from_raw_fd(fd) }),
                _ => {}
            }
        }
        if let Some(unix) = unix {
            unix.set_nonblocking(true)?;
            if let Some(tcp) = &tcp {
                tcp.set_nonblocking(true)?;
            }
            return Ok(Listeners { unix, tcp });
        }
    }
    let dir = path.parent().unwrap_or(Path::new("/tmp"));
    std::fs::create_dir_all(dir)?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if !meta.file_type().is_socket() {
            return Err(std::io::Error::other(format!("{} exists and isn't a socket", path.display())));
        }
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            return Err(std::io::Error::new(std::io::ErrorKind::AddrInUse, "Nyx AI Core is already running"));
        }
        std::fs::remove_file(path)?;
    }
    let unix = std::os::unix::net::UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    unix.set_nonblocking(true)?;
    Ok(Listeners { unix, tcp: None })
}

/// Serve until `stop` is cancelled.
pub async fn serve(service: Arc<Service>, listener: std::os::unix::net::UnixListener, stop: CancellationToken) {
    let listener = match UnixListener::from_std(listener) {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!(error = %e, "can't use the socket");
            return;
        }
    };
    let slots = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    loop {
        let accepted = tokio::select! {
            a = listener.accept() => a,
            _ = stop.cancelled() => break,
        };
        let stream = match accepted {
            Ok((stream, _)) => stream,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            tracing::warn!("too many connections; refusing one");
            continue;
        };
        let service = service.clone();
        let stop = stop.child_token();
        tokio::spawn(async move {
            connection(service, stream, stop).await;
            drop(permit);
        });
    }
}

/// Name the peer by the program it runs.
fn identify(pid: Option<i32>, declared: &str) -> Caller {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let connection = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let declared: String = declared.chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).take(64).collect();
    let own = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
    let program = pid
        .and_then(|pid| std::fs::read_link(format!("/proc/{pid}/exe")).ok())
        .zip(own)
        .and_then(|(exe, own)| nyx_peer::program(&exe, &own));
    match program {
        Some(name) => Caller { name, trusted: true, connection },
        None => Caller { name: format!("untrusted:{}", if declared.is_empty() { "unknown" } else { &declared }), trusted: false, connection },
    }
}

struct Lines {
    reader: BufReader<OwnedReadHalf>,
    line: Vec<u8>,
}

impl Lines {
    /// The next line, `None` at the end; too long is an error. Cancel-safe.
    async fn next(&mut self) -> std::io::Result<Option<String>> {
        loop {
            let buf = self.reader.fill_buf().await?;
            if buf.is_empty() {
                return Ok(None);
            }
            let (take, end) = match buf.iter().position(|b| *b == b'\n') {
                Some(i) => (i + 1, true),
                None => (buf.len(), false),
            };
            self.line.extend_from_slice(&buf[..take]);
            self.reader.consume(take);
            if self.line.len() > MAX_LINE {
                return Err(std::io::Error::other("line too long"));
            }
            if end {
                let line = std::mem::take(&mut self.line);
                return Ok(Some(String::from_utf8_lossy(&line).trim().to_string()));
            }
        }
    }
}

async fn connection(service: Arc<Service>, stream: UnixStream, stop: CancellationToken) {
    let cred = match stream.peer_cred() {
        Ok(cred) => cred,
        Err(e) => {
            tracing::warn!(error = %e, "no peer credentials; closing");
            return;
        }
    };
    // SAFETY: getuid has no preconditions.
    if cred.uid() != unsafe { libc::getuid() } {
        tracing::warn!(uid = cred.uid(), "refusing another user's connection");
        return;
    }
    let (read, mut write) = stream.into_split();
    let mut lines = Lines { reader: BufReader::new(read), line: Vec::new() };
    let (out, mut outbox) = mpsc::channel::<ServerLine>(64);
    let writer_stop = stop.clone();
    let writer = tokio::spawn(async move {
        while let Some(line) = outbox.recv().await {
            if write.write_all(encode(&line).as_bytes()).await.is_err() {
                writer_stop.cancel();
                break;
            }
        }
        let _ = write.shutdown().await;
    });

    let hello = tokio::time::timeout(Duration::from_secs(10), lines.next()).await;
    let caller = match hello.ok().and_then(|r| r.ok()).flatten().map(|l| decode_client_line(&l)) {
        Some(Ok(ClientLine::Hello(hello))) if hello.api_version == API_VERSION => {
            let caller = identify(cred.pid(), &hello.client);
            let _ = out
                .send(ServerLine::Hello(Welcome { api_version: API_VERSION, server: env!("CARGO_PKG_VERSION").into(), client: caller.name.clone() }))
                .await;
            caller
        }
        Some(Ok(ClientLine::Hello(hello))) => {
            let message = format!("this Nyx AI Core speaks version {API_VERSION}; the client speaks {}", hello.api_version);
            let _ = out.send(ServerLine::Error { id: None, error: Error::new(ErrorKind::IncompatibleVersion, message) }).await;
            drop(out);
            let _ = writer.await;
            return;
        }
        _ => {
            let error = Error::new(ErrorKind::InvalidRequest, "start with hello");
            let _ = out.send(ServerLine::Error { id: None, error }).await;
            drop(out);
            let _ = writer.await;
            return;
        }
    };
    tracing::debug!(client = %caller.name, "connected");

    let open: Arc<Mutex<HashMap<u64, CancellationToken>>> = Arc::default();
    loop {
        let line = tokio::select! {
            l = lines.next() => l,
            _ = stop.cancelled() => break,
        };
        let line = match line {
            Ok(Some(line)) if line.is_empty() => continue,
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(e) => {
                let _ = out.send(ServerLine::Error { id: None, error: Error::new(ErrorKind::InvalidRequest, e.to_string()) }).await;
                break;
            }
        };
        let request = match decode_client_line(&line) {
            Ok(ClientLine::Request(request)) => request,
            Ok(ClientLine::Hello(_)) => {
                let _ = out.send(ServerLine::Error { id: None, error: Error::new(ErrorKind::InvalidRequest, "already said hello") }).await;
                continue;
            }
            Err((id, error)) => {
                let _ = out.send(ServerLine::Error { id, error }).await;
                continue;
            }
        };
        let Request { id, op } = *request;
        if let Op::Cancel { request } = op {
            let token = open.lock().expect("requests").get(&request).cloned();
            let reply = match token {
                Some(token) => {
                    token.cancel();
                    ServerLine::Result { id, result: Reply::Ok }
                }
                None => ServerLine::Error { id: Some(id), error: Error::new(ErrorKind::UnknownRequest, format!("no open request {request}")) },
            };
            let _ = out.send(reply).await;
            continue;
        }
        let refused = {
            let open = open.lock().expect("requests");
            if open.contains_key(&id) {
                Some(Error::new(ErrorKind::InvalidRequest, format!("request {id} is already open")))
            } else if open.len() >= MAX_REQUESTS {
                Some(Error::new(ErrorKind::Busy, "too many open requests on this connection"))
            } else {
                service.authorize(&caller, &op).err()
            }
        };
        if let Some(error) = refused {
            let _ = out.send(ServerLine::Error { id: Some(id), error }).await;
            continue;
        }
        let token = stop.child_token();
        open.lock().expect("requests").insert(id, token.clone());
        let (service, caller, out, open) = (service.clone(), caller.clone(), out.clone(), open.clone());
        tokio::spawn(async move {
            let (events, mut inbox) = mpsc::channel::<Event>(32);
            let forward_out = out.clone();
            let forward = tokio::spawn(async move {
                while let Some(event) = inbox.recv().await {
                    if forward_out.send(ServerLine::Event { id, event }).await.is_err() {
                        break;
                    }
                }
            });
            let name = op.name();
            let result = service.handle(&caller, op, events, token).await;
            let _ = forward.await;
            open.lock().expect("requests").remove(&id);
            let line = match result {
                Ok(result) => ServerLine::Result { id, result },
                Err(error) => {
                    if error.kind != ErrorKind::Cancelled {
                        tracing::debug!(client = %caller.name, op = name, kind = ?error.kind, "request failed");
                    }
                    ServerLine::Error { id: Some(id), error }
                }
            };
            let _ = out.send(line).await;
        });
    }
    for token in open.lock().expect("requests").values() {
        token.cancel();
    }
    drop(out);
    let _ = writer.await;
    tracing::debug!(client = %caller.name, "disconnected");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_names_are_labelled() {
        let caller = identify(None, "nyx chat\n");
        assert!(!caller.trusted);
        assert_eq!(caller.name, "untrusted:nyxchat");
    }
}

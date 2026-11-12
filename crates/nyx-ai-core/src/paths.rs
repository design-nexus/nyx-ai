//! Where AI Core keeps things. Built once from the environment (or by a test).

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Paths {
    /// `~/.config/nyx-ai`: `settings.toml`, `models.ini`, `huggingface.token`.
    pub config: PathBuf,
    /// `~/.local/state/nyx-ai`: `registry.json`, the generated `router.ini`.
    pub state: PathBuf,
    /// `~/.local/share/nyx-ai`: `models/`, a folder for GGUF files put by hand.
    pub data: PathBuf,
    /// `$XDG_RUNTIME_DIR/nyx-ai` (`0700`): the sockets and the worker's key.
    pub runtime: PathBuf,
    pub socket: PathBuf,
    /// The shared Hugging Face hub cache the router reads.
    pub cache: PathBuf,
    /// Where `install-llama.sh` puts the llama.cpp build.
    pub lib: PathBuf,
    /// Other applications' config, for importing settings once.
    pub config_home: PathBuf,
    pub data_home: PathBuf,
}

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

pub fn home() -> PathBuf {
    env_dir("HOME").unwrap_or_else(|| PathBuf::from("/"))
}

/// `LLAMA_CACHE`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`, then
/// `~/.cache/huggingface/hub`: the order llama.cpp and Chat use.
pub fn hub_cache() -> PathBuf {
    for name in ["LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(path) = env_dir(name) {
            return path;
        }
    }
    if let Some(path) = env_dir("HF_HOME") {
        return path.join("hub");
    }
    env_dir("XDG_CACHE_HOME").unwrap_or_else(|| home().join(".cache")).join("huggingface/hub")
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = home();
        let config_home = env_dir("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"));
        let data_home = env_dir("XDG_DATA_HOME").unwrap_or_else(|| home.join(".local/share"));
        let state_home = env_dir("XDG_STATE_HOME").unwrap_or_else(|| home.join(".local/state"));
        let socket = env_dir(nyx_ai_proto::SOCKET_ENV).unwrap_or_else(|| {
            env_dir("XDG_RUNTIME_DIR")
                // SAFETY: getuid has no preconditions.
                .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", unsafe { libc::getuid() })))
                .join("nyx-ai")
                .join(nyx_ai_proto::SOCKET_NAME)
        });
        let runtime = socket.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"));
        Paths {
            config: config_home.join("nyx-ai"),
            state: state_home.join("nyx-ai"),
            data: data_home.join("nyx-ai"),
            runtime,
            socket,
            cache: hub_cache(),
            lib: env_dir("NYX_AI_LIB").unwrap_or_else(|| home.join(".local/lib/nyx-ai/llama.cpp")),
            config_home,
            data_home,
        }
    }

    /// Everything under one folder, for tests.
    pub fn under(root: &std::path::Path) -> Paths {
        Paths {
            config: root.join("config/nyx-ai"),
            state: root.join("state/nyx-ai"),
            data: root.join("data/nyx-ai"),
            runtime: root.join("run"),
            socket: root.join("run/ai.sock"),
            cache: root.join("cache/huggingface/hub"),
            lib: root.join("lib"),
            config_home: root.join("config"),
            data_home: root.join("data"),
        }
    }

    pub fn settings(&self) -> PathBuf {
        self.config.join("settings.toml")
    }

    pub fn models_ini(&self) -> PathBuf {
        self.config.join("models.ini")
    }

    pub fn router_ini(&self) -> PathBuf {
        self.state.join("router.ini")
    }

    pub fn registry(&self) -> PathBuf {
        self.state.join("registry.json")
    }

    pub fn worker_socket(&self) -> PathBuf {
        self.runtime.join("worker.sock")
    }

    /// The OpenAI-compatible endpoint's API key (0600), for coding tools to send.
    pub fn api_key(&self) -> PathBuf {
        self.config.join("api-key")
    }

    pub fn worker_key(&self) -> PathBuf {
        self.runtime.join("worker.key")
    }

    pub fn own_models(&self) -> PathBuf {
        self.data.join("models")
    }
}

/// `~/x` and `$HOME/x` to absolute paths.
pub fn expand(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        home().join(rest)
    } else if let Some(rest) = path.strip_prefix("$HOME/") {
        home().join(rest)
    } else {
        PathBuf::from(path)
    }
}

/// Write `content` to `path` through a temporary file, so a crash never leaves
/// half a file.
pub fn write_atomic(path: &std::path::Path, content: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&temp, content)?;
    std::fs::rename(temp, path)
}

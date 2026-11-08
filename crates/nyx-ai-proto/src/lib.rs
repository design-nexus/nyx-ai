//! The Nyx AI Core protocol: one JSON object per line over a Unix socket.
//!
//! A client opens with [`ClientLine::Hello`] and the server answers with
//! [`ServerLine::Hello`], or an `incompatible_version` error and a closed
//! connection. Each [`Request`] carries a client-chosen `id`; the server answers it
//! with any number of [`ServerLine::Event`]s and then exactly one
//! [`ServerLine::Result`] or [`ServerLine::Error`].
//!
//! Requests are strict: unknown fields and unknown operations are errors, never
//! ignored. Replies may gain optional fields within an `API_VERSION`; a change an
//! older client can't ignore bumps it. See `docs/PROTOCOL.md`.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const API_VERSION: u32 = 1;

/// The longest line either side accepts (a chat with a few images fits).
pub const MAX_LINE: usize = 64 << 20;

/// The socket's name inside `$XDG_RUNTIME_DIR/nyx-ai/`.
pub const SOCKET_NAME: &str = "ai.sock";

/// Overrides the socket path, for test copies of the daemon and its clients.
pub const SOCKET_ENV: &str = "NYX_AI_SOCKET";

// ---------- Lines ----------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ClientLine {
    Hello(Hello),
    Request(Box<Request>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    pub api_version: u32,
    /// What the client calls itself, e.g. `nyx-chat`. The server checks it
    /// against the program the peer actually runs.
    pub client: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: u64,
    pub op: Op,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ServerLine {
    Hello(Welcome),
    Event { id: u64, event: Event },
    Result { id: u64, result: Reply },
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<u64>,
        error: Error,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Welcome {
    pub api_version: u32,
    pub server: String,
    /// The identity the server gave this connection: the trusted program's name,
    /// or `untrusted:<declared name>`.
    pub client: String,
}

// ---------- Operations ----------

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Background,
    #[default]
    Normal,
    Interactive,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    ListModels {},
    ModelInfo {
        model: String,
    },
    /// Load a model now. `keep` pins it for this application until `unload_model`.
    LoadModel {
        model: String,
        #[serde(default)]
        keep: bool,
    },
    /// Drop this application's pin and unload the model unless something else
    /// still uses it.
    UnloadModel {
        model: String,
    },
    /// Delete a model from the shared Hugging Face cache. Models in folders are
    /// never deleted.
    RemoveModel {
        model: String,
    },
    /// Load-time settings for a model, or `*` for the defaults.
    ModelSettings {
        model: String,
    },
    /// Change load-time settings. An empty value removes the key.
    ConfigureModel {
        model: String,
        settings: BTreeMap<String, String>,
    },
    /// Download `org/repo:QUANT` from Hugging Face into the shared cache. Streams
    /// `progress`; the download goes on if the client disconnects.
    DownloadModel {
        model: String,
    },
    CancelDownload {
        model: String,
    },
    Downloads {},
    Chat(ChatRequest),
    Embed {
        model: String,
        input: Vec<String>,
        #[serde(default)]
        priority: Priority,
    },
    /// The prompt the model's template makes of a chat, without generating.
    ApplyTemplate(ChatRequest),
    /// Cancel one of this connection's requests.
    Cancel {
        request: u64,
    },
    ResourceStatus {},
    WorkerStatus {},
    ListCapabilities {},
    ServiceSettings {},
    ConfigureService {
        #[serde(default)]
        max_loaded: Option<u32>,
        #[serde(default)]
        idle_unload_seconds: Option<i64>,
    },
}

/// Every operation's `type`, for capability discovery.
pub const OPERATIONS: &[&str] = &[
    "list_models",
    "model_info",
    "load_model",
    "unload_model",
    "remove_model",
    "model_settings",
    "configure_model",
    "download_model",
    "cancel_download",
    "downloads",
    "chat",
    "embed",
    "apply_template",
    "cancel",
    "resource_status",
    "worker_status",
    "list_capabilities",
    "service_settings",
    "configure_service",
];

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::ListModels {} => "list_models",
            Op::ModelInfo { .. } => "model_info",
            Op::LoadModel { .. } => "load_model",
            Op::UnloadModel { .. } => "unload_model",
            Op::RemoveModel { .. } => "remove_model",
            Op::ModelSettings { .. } => "model_settings",
            Op::ConfigureModel { .. } => "configure_model",
            Op::DownloadModel { .. } => "download_model",
            Op::CancelDownload { .. } => "cancel_download",
            Op::Downloads {} => "downloads",
            Op::Chat(_) => "chat",
            Op::Embed { .. } => "embed",
            Op::ApplyTemplate(_) => "apply_template",
            Op::Cancel { .. } => "cancel",
            Op::ResourceStatus {} => "resource_status",
            Op::WorkerStatus {} => "worker_status",
            Op::ListCapabilities {} => "list_capabilities",
            Op::ServiceSettings {} => "service_settings",
            Op::ConfigureService { .. } => "configure_service",
        }
    }
}

// ---------- Chat ----------

/// An OpenAI-style chat, limited to what AI Core checks and passes on.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Sampling::is_empty")]
    pub sampling: Sampling,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// For templates that switch thinking on and off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub priority: Priority,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl Sampling {
    pub fn is_empty(&self) -> bool {
        *self == Sampling::default()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Role,
    #[serde(default, deserialize_with = "null_as_default")]
    pub content: Content,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn text(role: Role, text: impl Into<String>) -> Message {
        Message { role, content: Content::Text(text.into()), tool_calls: Vec::new(), tool_call_id: None }
    }
}

fn null_as_default<'de, D: Deserializer<'de>, T: Default + Deserialize<'de>>(d: D) -> Result<T, D::Error> {
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Part {
    Text { text: String },
    /// A `data:image/…;base64,…` URL. Remote URLs are refused so the worker
    /// never fetches anything.
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InputAudio {
    /// Base64.
    pub data: String,
    /// `wav` or `mp3`.
    pub format: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_kind() -> String {
    "function".into()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FunctionCall {
    pub name: String,
    /// JSON text, as the model wrote it.
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: Value },
}

pub const REASONING_EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

impl ChatRequest {
    pub fn has_images(&self) -> bool {
        self.parts().any(|p| matches!(p, Part::ImageUrl { .. }))
    }

    pub fn has_audio(&self) -> bool {
        self.parts().any(|p| matches!(p, Part::InputAudio { .. }))
    }

    fn parts(&self) -> impl Iterator<Item = &Part> {
        self.messages.iter().flat_map(|m| match &m.content {
            Content::Parts(parts) => parts.as_slice(),
            Content::Text(_) => &[],
        })
    }

    /// Structural checks that don't need the model.
    pub fn validate(&self) -> Result<(), String> {
        valid_model_id(&self.model)?;
        if self.messages.is_empty() {
            return Err("a chat needs at least one message".into());
        }
        for part in self.parts() {
            match part {
                Part::ImageUrl { image_url } => {
                    let url = &image_url.url;
                    if !url.starts_with("data:image/") || !url.contains(";base64,") {
                        return Err("images must be base64 data: URLs".into());
                    }
                }
                Part::InputAudio { input_audio } => {
                    if !matches!(input_audio.format.as_str(), "wav" | "mp3") {
                        return Err("audio must be wav or mp3".into());
                    }
                }
                Part::Text { .. } => {}
            }
        }
        for message in &self.messages {
            for call in &message.tool_calls {
                valid_tool_name(&call.function.name)?;
            }
            if message.role == Role::Tool && message.tool_call_id.as_deref().is_none_or(str::is_empty) {
                return Err("a tool message needs tool_call_id".into());
            }
        }
        for tool in &self.tools {
            if tool.kind != "function" {
                return Err(format!("unsupported tool type {}", tool.kind));
            }
            valid_tool_name(&tool.function.name)?;
            if tool.function.parameters.as_ref().is_some_and(|p| !p.is_object()) {
                return Err(format!("tool {} parameters must be a JSON schema object", tool.function.name));
            }
        }
        if let Some(choice) = &self.tool_choice
            && !(choice.is_object() || matches!(choice.as_str(), Some("auto" | "none" | "required")))
        {
            return Err("tool_choice must be auto, none, required or a function".into());
        }
        if let Some(ResponseFormat::JsonSchema { schema }) = &self.response_format
            && !schema.is_object()
        {
            return Err("response_format schema must be an object".into());
        }
        if let Some(effort) = &self.reasoning_effort
            && !REASONING_EFFORTS.contains(&effort.as_str())
        {
            return Err(format!("unknown reasoning effort {effort}"));
        }
        let s = &self.sampling;
        let range = |name: &str, v: Option<f64>, lo: f64, hi: f64| match v {
            Some(v) if !(lo..=hi).contains(&v) || v.is_nan() => Err(format!("{name} must be between {lo} and {hi}")),
            _ => Ok(()),
        };
        range("temperature", s.temperature, 0.0, 5.0)?;
        range("top_p", s.top_p, 0.0, 1.0)?;
        range("min_p", s.min_p, 0.0, 1.0)?;
        range("repeat_penalty", s.repeat_penalty, 0.0, 5.0)?;
        range("presence_penalty", s.presence_penalty, -2.0, 2.0)?;
        range("frequency_penalty", s.frequency_penalty, -2.0, 2.0)?;
        if s.top_k.is_some_and(|k| !(0..=1000).contains(&k)) {
            return Err("top_k must be between 0 and 1000".into());
        }
        if s.stop.len() > 16 {
            return Err("at most 16 stop sequences".into());
        }
        Ok(())
    }
}

fn valid_tool_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'));
    if ok { Ok(()) } else { Err(format!("invalid tool name {name:?}")) }
}

/// A model ID: `org/repo:QUANT` for the cache, or a plain name for a folder or
/// preset model. No control characters, brackets or path tricks.
pub fn valid_model_id(id: &str) -> Result<(), String> {
    let ok = !id.is_empty()
        && id.len() <= 256
        && !id.starts_with(['/', '.', '-'])
        && !id.contains("..")
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | ':' | '_' | '-' | '.' | '+'));
    if ok { Ok(()) } else { Err(format!("invalid model id {id:?}")) }
}

// ---------- Events ----------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Accepted,
    /// Waiting behind `position` other requests for the model.
    Queued { position: usize },
    ModelLoading { model: String },
    Started,
    Delta {
        #[serde(default, skip_serializing_if = "String::is_empty")]
        content: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        reasoning: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCallDelta>,
    },
    Progress { status: String, completed: u64, total: u64 },
}

/// Part of a tool call; parts with the same `index` concatenate.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------- Replies ----------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reply {
    Ok,
    Models { models: Vec<Model> },
    ModelInfo(Box<ModelDetail>),
    Loaded { model: String },
    Unloaded { model: String, unloaded: bool },
    Settings(ModelSettings),
    Downloads { downloads: Vec<Download> },
    Finished(Finished),
    Embeddings(Embeddings),
    Template { prompt: String },
    Resources(ResourceStatus),
    Worker(WorkerStatus),
    Capabilities(Capabilities),
    Service(ServiceSettings),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Tools,
    Thinking,
    Vision,
    Audio,
    Embedding,
    /// Only a speculative drafter or projector; not a model to chat with.
    Draft,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Downloading,
    Validating,
    Installed,
    Loading,
    Ready,
    Busy,
    Unloading,
    Failed,
    /// Recorded before, but its file is gone.
    Missing,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// The shared Hugging Face cache, which AI Core manages.
    Cache,
    /// A model folder AI Core reads but never changes.
    Folder,
    /// A `model =` entry in `models.ini`.
    Preset,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub source: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    pub capabilities: Vec<Capability>,
    pub state: ModelState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Some application asked to keep it loaded.
    #[serde(default)]
    pub pinned: bool,
    /// `remove_model` may delete it.
    #[serde(default)]
    pub removable: bool,
    /// Settings changed since it loaded; they apply on the next load.
    #[serde(default)]
    pub pending_settings: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Model {
    pub fn has(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }

    pub fn loaded(&self) -> bool {
        matches!(self.state, ModelState::Ready | ModelState::Busy)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelDetail {
    pub model: Model,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// The context the loaded instance actually has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_context: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<u32>,
    /// Levels the loaded model's template accepts (empty when unloaded or none).
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_length: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelSettings {
    pub model: String,
    /// Keys set for this model (or the defaults, for `*`).
    pub values: BTreeMap<String, String>,
    /// What a load uses: the defaults overlaid with the model's own keys.
    pub effective: BTreeMap<String, String>,
    #[serde(default)]
    pub pending: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Download {
    pub model: String,
    pub state: DownloadState,
    pub status: String,
    pub completed: u64,
    pub total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Timings {
    /// Waiting for a slot.
    pub queued_ms: u64,
    /// Loading the model, when this request loaded it.
    pub load_ms: u64,
    /// From the worker receiving the request to the first token.
    pub first_token_ms: u64,
    pub total_ms: u64,
    pub tokens_per_second: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Finished {
    /// `stop`, `length`, `tool_calls`, …
    pub finish_reason: String,
    pub usage: Usage,
    pub timings: Timings,
}

/// What produced a set of vectors, so an index can tell when they'd mix.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelIdentity {
    pub id: String,
    pub file: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embeddings {
    pub identity: ModelIdentity,
    pub dimensions: usize,
    pub vectors: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LoadedModel {
    pub id: String,
    pub state: String,
    /// Estimated memory, in bytes.
    pub estimate: u64,
    pub active: usize,
    pub slots: usize,
    pub pinned_by: Vec<String>,
    pub idle_secs: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ResourceStatus {
    pub ram_total: u64,
    pub ram_available: u64,
    pub vram_total: u64,
    pub gpu_backends: Vec<String>,
    /// Memory AI Core lets loaded models use, in bytes.
    pub budget: u64,
    pub max_loaded: u32,
    pub idle_unload_seconds: i64,
    pub loaded: Vec<LoadedModel>,
    pub queued: usize,
    /// `/proc/pressure/memory` "some avg10", when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_pressure: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    /// No llama.cpp build is installed.
    Missing,
    Stopped,
    Starting,
    Ready,
    Restarting,
    /// Crashed too often; the next request after a pause tries again.
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Metrics {
    pub requests: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub active: u64,
    pub queued: u64,
    pub loads: u64,
    pub load_ms_avg: u64,
    pub evictions: u64,
    pub idle_unloads: u64,
    pub worker_restarts: u64,
    pub first_token_ms_avg: u64,
    pub tokens_per_second_avg: f64,
    pub embeddings: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WorkerStatus {
    pub state: WorkerState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// The build's release tag, e.g. `prism-b10826-c9c34bc`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    pub restarts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub uptime_secs: u64,
    pub metrics: Metrics,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Capabilities {
    pub api_version: u32,
    pub server_version: String,
    pub operations: Vec<String>,
    pub gpu_backends: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_release: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ServiceSettings {
    pub max_loaded: u32,
    /// Seconds before an idle, unpinned model unloads; -1 never.
    pub idle_unload_seconds: i64,
    pub model_dirs: Vec<String>,
}

// ---------- Errors ----------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    IncompatibleVersion,
    InvalidRequest,
    UnsupportedOperation,
    UnknownModel,
    UnknownRequest,
    UnsupportedCapability,
    ResourceExhausted,
    Busy,
    PermissionDenied,
    WorkerUnavailable,
    WorkerFailed,
    ModelLoadFailed,
    DownloadFailed,
    Cancelled,
    Timeout,
    /// The client couldn't reach the daemon.
    Unavailable,
    Internal,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
        Error { kind, message: message.into() }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Parse a client line, telling an unknown operation apart from a malformed one.
pub fn decode_client_line(line: &str) -> Result<ClientLine, (Option<u64>, Error)> {
    let value: Value = serde_json::from_str(line)
        .map_err(|e| (None, Error::new(ErrorKind::InvalidRequest, format!("not JSON: {e}"))))?;
    let id = value["request"]["id"].as_u64();
    if let Some(kind) = value["request"]["op"]["type"].as_str()
        && !OPERATIONS.contains(&kind)
    {
        return Err((id, Error::new(ErrorKind::UnsupportedOperation, format!("unknown operation {kind}"))));
    }
    serde_json::from_value(value).map_err(|e| (id, Error::new(ErrorKind::InvalidRequest, e.to_string())))
}

pub fn encode<T: Serialize>(line: &T) -> String {
    let mut text = serde_json::to_string(line).expect("protocol types serialize");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(op: Value) -> String {
        json!({"request": {"id": 7, "op": op}}).to_string()
    }

    #[test]
    fn operations_round_trip() {
        let ops = vec![
            Op::ListModels {},
            Op::LoadModel { model: "org/m:Q4_K_M".into(), keep: true },
            Op::ConfigureModel { model: "*".into(), settings: [("ctx-size".to_string(), "8192".to_string())].into() },
            Op::Chat(ChatRequest {
                model: "org/m:Q4_K_M".into(),
                messages: vec![Message::text(Role::User, "hi")],
                sampling: Sampling { temperature: Some(0.2), ..Default::default() },
                ..Default::default()
            }),
            Op::Embed { model: "e".into(), input: vec!["a".into()], priority: Priority::Background },
            Op::ConfigureService { max_loaded: Some(2), idle_unload_seconds: None },
        ];
        for op in ops {
            assert!(OPERATIONS.contains(&op.name()));
            let line = encode(&ClientLine::Request(Box::new(Request { id: 1, op: op.clone() })));
            match decode_client_line(line.trim()).unwrap() {
                ClientLine::Request(r) => assert_eq!(r.op, op),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn unknown_fields_and_operations_are_errors() {
        let (id, e) = decode_client_line(&request(json!({"type": "list_models", "extra": 1}))).unwrap_err();
        assert_eq!((id, e.kind), (Some(7), ErrorKind::InvalidRequest));
        let (_, e) = decode_client_line(&request(json!({"type": "chat", "model": "m", "messages": [], "nope": true}))).unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
        let (_, e) = decode_client_line(&request(json!({"type": "summon_dragon"}))).unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedOperation);
        let (_, e) = decode_client_line(r#"{"hello":{"api_version":1,"client":"x","debug":true}}"#).unwrap_err();
        assert_eq!(e.kind, ErrorKind::InvalidRequest);
    }

    #[test]
    fn openai_messages_parse() {
        let line = request(json!({"type": "chat", "model": "org/m:Q8_0", "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "web_search", "arguments": "{\"q\":1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ], "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
           "response_format": {"type": "json_object"}, "thinking": true, "priority": "interactive"}));
        let ClientLine::Request(r) = decode_client_line(&line).unwrap() else { panic!() };
        let Op::Chat(request) = r.op else { panic!() };
        assert!(request.has_images());
        assert_eq!(request.priority, Priority::Interactive);
        request.validate().unwrap();
    }

    #[test]
    fn validation() {
        let mut r = ChatRequest { model: "m".into(), messages: vec![Message::text(Role::User, "x")], ..Default::default() };
        r.validate().unwrap();
        r.sampling.temperature = Some(9.0);
        assert!(r.validate().is_err());
        r.sampling.temperature = None;
        r.messages[0].content = Content::Parts(vec![Part::ImageUrl { image_url: ImageUrl { url: "https://x/y.png".into() } }]);
        assert!(r.validate().unwrap_err().contains("data:"));
        r.messages = vec![];
        assert!(r.validate().is_err());
        assert!(valid_model_id("../etc/passwd").is_err());
        assert!(valid_model_id("[x]").is_err());
        assert!(valid_model_id("prism-ml/Ternary-Bonsai-2-9B-gguf:PQ2_0").is_ok());
        assert!(valid_model_id("gemma-4-E2B-it-Q4_K_M").is_ok());
    }

    #[test]
    fn server_lines() {
        let line = encode(&ServerLine::Event {
            id: 3,
            event: Event::Delta { content: "hi".into(), reasoning: String::new(), tool_calls: vec![] },
        });
        assert_eq!(line, "{\"event\":{\"id\":3,\"event\":{\"type\":\"delta\",\"content\":\"hi\"}}}\n");
        let back: ServerLine = serde_json::from_str(line.trim()).unwrap();
        assert!(matches!(back, ServerLine::Event { id: 3, .. }));
        let err = encode(&ServerLine::Error { id: None, error: Error::new(ErrorKind::IncompatibleVersion, "v2 needed") });
        assert!(err.contains("incompatible_version"));
        let ok: ServerLine = serde_json::from_str(r#"{"result":{"id":1,"result":{"type":"ok"}}}"#).unwrap();
        assert_eq!(ok, ServerLine::Result { id: 1, result: Reply::Ok });
    }
}

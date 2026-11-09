//! One handler per operation, over the registry, presets, worker, scheduler and
//! downloads.

use crate::actions::Actions;
use crate::config::Settings;
use crate::diagnostics::Metrics;
use crate::downloads::Downloads;
use crate::paths::{self, Paths};
use crate::presets::{self, Ini};
use crate::registry::{Record, Registry, RouterItem};
use crate::resources;
use crate::scheduler::{Backend, Load, Scheduler};
use crate::stream::{self, Sse};
use crate::updates::Updates;
use crate::worker::{self, Worker, encode_query};
use futures_util::future::BoxFuture;
use hyper::Method;
use nyx_ai_proto::{
    Capabilities, Capability, ChatRequest, Download, EmbedInput, DownloadState, Embeddings, Error, ErrorKind, Event, Finished, Model, ModelDetail,
    ModelIdentity, ModelSettings, ModelState, OPERATIONS, Op, Priority, REASONING_EFFORTS, Reply, ResourceStatus, ServiceSettings, Source,
    Timings, WorkerStatus, valid_model_id,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Who's asking.
#[derive(Clone, Debug)]
pub struct Caller {
    /// The trusted program's name, or `untrusted:<declared name>`.
    pub name: String,
    pub trusted: bool,
    /// Which socket connection (actions are answered on the one they were sent to).
    pub connection: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Let untrusted callers remove models and change settings (test copies).
    pub allow_untrusted: bool,
    /// Use this llama-server instead of the configured or installed one.
    pub worker: Option<std::path::PathBuf>,
    /// Where to read mains power from (`/sys/class/power_supply`).
    pub power_supply: Option<std::path::PathBuf>,
    /// Hugging Face's address, for tests (`HF_ENDPOINT` otherwise).
    pub hf_endpoint: Option<String>,
    /// How long a model must sit unused before an update switches it (120 s).
    pub update_idle_secs: Option<u64>,
}

#[derive(Default)]
struct PresetState {
    ini: Ini,
    /// Sections as they were before a change that waits for busy models.
    held: BTreeMap<String, Option<Vec<String>>>,
    /// The busy models each held change waits for.
    waiting: BTreeMap<String, BTreeSet<String>>,
    /// Embedding models written into the router's file.
    embedding: BTreeSet<String>,
    /// Battery saver on battery: every model loads with no GPU layers.
    cpu_only: bool,
    written: Option<String>,
}

struct Shared {
    paths: Paths,
    worker: Arc<Worker>,
    registry: Registry,
    presets: Mutex<PresetState>,
}

impl Shared {
    fn context_for(&self, model: &str) -> (u64, String) {
        let (effective, meta_ctx) = {
            let presets = self.presets.lock().expect("presets");
            (presets.ini.effective(model), self.registry.get(model).and_then(|r| r.meta.and_then(|m| m.context_length)))
        };
        let ctx = effective.get("ctx-size").and_then(|v| v.parse::<u64>().ok()).filter(|c| *c > 0);
        let ctx = ctx.unwrap_or_else(|| meta_ctx.unwrap_or(8192).min(32768));
        (ctx, effective.get("cache-type-k").cloned().unwrap_or_else(|| "f16".into()))
    }
}

/// The scheduler's view of the worker.
struct WorkerBackend(Arc<Shared>);

fn status_of(list: &Value, model: &str) -> Option<RouterItem> {
    list["data"].as_array()?.iter().filter_map(RouterItem::parse).find(|i| i.id == model)
}

impl Backend for WorkerBackend {
    fn load(&self, model: String) -> BoxFuture<'static, Result<usize, Error>> {
        let shared = self.0.clone();
        Box::pin(async move {
            let http = shared.worker.ready().await?;
            let generation = shared.worker.generation();
            match http.post("/models/load", &json!({"model": model})).await {
                Ok(_) => {}
                Err(e) if e.message.contains("already") => {}
                Err(e) => return Err(Error::new(ErrorKind::ModelLoadFailed, e.message)),
            }
            let started = Instant::now();
            loop {
                if shared.worker.generation() != generation {
                    return Err(Error::new(ErrorKind::WorkerFailed, "llama.cpp restarted while loading the model"));
                }
                let list = http.get("/models").await?;
                let Some(item) = status_of(&list, &model) else {
                    return Err(Error::new(ErrorKind::UnknownModel, format!("llama.cpp doesn't list {model}")));
                };
                match item.status.as_str() {
                    "loaded" => break,
                    "loading" => {}
                    _ if item.failed.is_some() => {
                        let detail = shared.worker.load_error().map(|e| format!(": {e}")).unwrap_or_default();
                        return Err(Error::new(ErrorKind::ModelLoadFailed, format!("{}{detail}", item.failed.unwrap_or_default())));
                    }
                    _ if started.elapsed() > Duration::from_secs(15) => {
                        return Err(Error::new(ErrorKind::ModelLoadFailed, format!("llama.cpp didn't start loading {model}")));
                    }
                    _ => {}
                }
                if started.elapsed() > Duration::from_secs(900) {
                    return Err(Error::new(ErrorKind::Timeout, format!("{model} took more than 15 minutes to load")));
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let props = http.get(&format!("/props?model={}", encode_query(&model))).await.unwrap_or_default();
            Ok(props["total_slots"].as_u64().unwrap_or(1).max(1) as usize)
        })
    }

    fn unload(&self, model: String) -> BoxFuture<'static, Result<(), Error>> {
        let shared = self.0.clone();
        Box::pin(async move {
            let http = shared.worker.ready().await?;
            if let Err(e) = http.post("/models/unload", &json!({"model": model})).await
                && !e.message.contains("not")
            {
                return Err(e);
            }
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(30) {
                let list = http.get("/models").await?;
                if status_of(&list, &model).is_none_or(|i| !matches!(i.status.as_str(), "loaded" | "loading")) {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(Error::new(ErrorKind::Timeout, format!("{model} didn't unload")))
        })
    }

    fn estimate(&self, model: &str) -> u64 {
        let size = self.weights(model);
        let (ctx, kv) = self.0.context_for(model);
        resources::needed(size, ctx, &kv)
    }

    fn weights(&self, model: &str) -> u64 {
        self.0.registry.get(model).map(|r| r.size).unwrap_or(0)
    }

    fn memory(&self) -> (u64, u64) {
        let mem = resources::detect();
        (resources::budget(mem), mem.vram + mem.ram)
    }
}

pub struct Service {
    shared: Arc<Shared>,
    settings: Mutex<Settings>,
    scheduler: Arc<Scheduler>,
    downloads: Arc<Downloads>,
    metrics: Arc<Metrics>,
    options: Options,
    router_status: Mutex<HashMap<String, String>>,
    seen_generation: AtomicU64,
    refresh_lock: tokio::sync::Mutex<()>,
    updates: Updates,
    /// When the updater last started each model's download.
    update_tries: Mutex<HashMap<String, Instant>>,
    switch_lock: tokio::sync::Mutex<()>,
    started: Instant,
    actions: Actions,
    /// Models loaded before the last CPU-only switch, unloaded once idle.
    power_stale: Mutex<BTreeSet<String>>,
}

const PRIVILEGED: &[&str] = &["remove_model", "configure_model", "configure_service", "serve_actions"];

impl Service {
    pub fn new(paths: Paths, options: Options) -> Arc<Service> {
        let settings = Settings::load(&paths);
        for dir in [&paths.config, &paths.state, &paths.own_models()] {
            let _ = std::fs::create_dir_all(dir);
        }
        let program = options
            .worker
            .clone()
            .filter(|p| p.is_file())
            .or_else(|| worker::find_program(settings.worker.as_deref(), &paths.lib));
        let worker = Worker::new(worker::Config {
            program,
            preset: paths.router_ini(),
            socket: paths.worker_socket(),
            key_file: paths.worker_key(),
            cache: paths.cache.clone(),
            max_models: crate::config::MAX_LOADED_LIMIT * 2,
        });
        let shared = Arc::new(Shared {
            registry: Registry::load(&paths),
            presets: Mutex::new(PresetState { ini: presets::load_or_import(&paths), ..Default::default() }),
            worker,
            paths: paths.clone(),
        });
        let metrics = Arc::new(Metrics::default());
        let scheduler = Scheduler::new(Arc::new(WorkerBackend(shared.clone())), settings.max_loaded, settings.idle_unload_seconds, metrics.clone());
        Arc::new_cyclic(|me: &Weak<Service>| {
            let on_done = me.clone();
            let downloads = Downloads::new(
                paths.cache.clone(),
                options.hf_endpoint.clone().unwrap_or_else(crate::hf::endpoint),
                paths.config_home.clone(),
                settings.sidecars.clone(),
                Box::new(move |model, revision| {
                    let me = on_done.clone();
                    Box::pin(async move {
                        if let Some(service) = me.upgrade() {
                            service.downloaded(&model, &revision).await;
                        }
                    })
                }),
            );
            let on_idle = me.clone();
            scheduler.on_idle(move |model| {
                if let Some(service) = on_idle.upgrade() {
                    tokio::spawn(async move { service.apply_held(&model).await });
                }
            });
            Service {
                shared,
                settings: Mutex::new(settings),
                scheduler,
                downloads,
                metrics,
                options,
                router_status: Mutex::new(HashMap::new()),
                seen_generation: AtomicU64::new(0),
                refresh_lock: tokio::sync::Mutex::new(()),
                updates: Updates::default(),
                update_tries: Mutex::new(HashMap::new()),
                switch_lock: tokio::sync::Mutex::new(()),
                started: Instant::now(),
                actions: Actions::default(),
                power_stale: Mutex::default(),
            }
        })
    }

    pub fn paths(&self) -> &Paths {
        &self.shared.paths
    }

    /// Idle unloading, memory pressure and worker restarts, until `stop`.
    pub fn run_background(self: &Arc<Self>, stop: CancellationToken) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut changes = me.shared.worker.subscribe();
            let mut tick = tokio::time::interval(Duration::from_secs(15));
            loop {
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = changes.changed() => me.check_generation(),
                    _ = tick.tick() => {
                        me.check_generation();
                        let total = resources::detect().ram;
                        let pressure = resources::under_pressure(total, resources::available_ram(), resources::memory_pressure());
                        me.scheduler.tick(pressure).await;
                        me.apply_staged().await;
                        me.auto_update();
                        me.apply_power().await;
                    }
                }
            }
        });
    }

    fn check_generation(&self) {
        let now = self.shared.worker.generation();
        let seen = self.seen_generation.swap(now, Ordering::SeqCst);
        if now != seen {
            if seen != 0 {
                Metrics::inc(&self.metrics.worker_restarts);
            }
            self.scheduler.reset();
            self.shared.presets.lock().expect("presets").written = None;
        }
    }

    pub async fn shutdown(&self) {
        self.shared.worker.shutdown().await;
    }

    pub fn authorize(&self, caller: &Caller, op: &Op) -> Result<(), Error> {
        if !caller.trusted && !self.options.allow_untrusted && PRIVILEGED.contains(&op.name()) {
            return Err(Error::new(ErrorKind::PermissionDenied, format!("{} may not {}", caller.name, op.name().replace('_', " "))));
        }
        Ok(())
    }

    // ---------- The model list ----------

    fn write_router_ini(&self, folders: &[presets::FolderEntry]) -> Result<bool, Error> {
        let mut presets = self.shared.presets.lock().expect("presets");
        let text = presets::router_ini(&presets.ini, &presets.held, folders, &presets.embedding, presets.cpu_only);
        if presets.written.as_deref() == Some(text.as_str()) && self.shared.paths.router_ini().exists() {
            return Ok(false);
        }
        paths::write_atomic(&self.shared.paths.router_ini(), text.as_bytes())
            .map_err(|e| Error::new(ErrorKind::Internal, format!("couldn't write the router presets: {e}")))?;
        presets.written = Some(text);
        Ok(true)
    }

    /// Ask the router for its models, after updating its presets.
    pub async fn refresh(&self, reload: bool) -> Result<BTreeMap<String, Record>, Error> {
        let _guard = self.refresh_lock.lock().await;
        if self.shared.worker.program().is_none() {
            return Err(Error::new(ErrorKind::WorkerUnavailable, "llama.cpp isn't installed"));
        }
        let dirs = self.settings.lock().expect("settings").model_dirs();
        let folders = self.shared.registry.scan_folders(&dirs, &self.shared.paths.cache);
        let mut reload = self.write_router_ini(&folders)? || reload;
        let http = self.shared.worker.ready().await?;
        self.check_generation();
        for _ in 0..2 {
            let list = http.get(if reload { "/models?reload=1" } else { "/models" }).await?;
            let items: Vec<RouterItem> = list["data"].as_array().into_iter().flatten().filter_map(RouterItem::parse).collect();
            let loaded: HashSet<String> = items.iter().filter(|i| matches!(i.status.as_str(), "loaded" | "loading")).map(|i| i.id.clone()).collect();
            self.scheduler.sync(&loaded);
            *self.router_status.lock().expect("status") = items.iter().map(|i| (i.id.clone(), i.status.clone())).collect();
            let records = self.shared.registry.update(&items);
            let embedding: BTreeSet<String> =
                records.iter().filter(|(_, r)| !r.missing && r.capabilities().contains(&Capability::Embedding)).map(|(id, _)| id.clone()).collect();
            let changed = {
                let mut presets = self.shared.presets.lock().expect("presets");
                let changed = presets.embedding != embedding;
                presets.embedding = embedding;
                changed
            };
            if !changed || !self.write_router_ini(&folders)? {
                return Ok(records);
            }
            reload = true;
        }
        Ok(self.shared.registry.known())
    }

    async fn records(&self) -> BTreeMap<String, Record> {
        match self.refresh(false).await {
            Ok(records) => records,
            Err(e) => {
                tracing::debug!(error = %e, "listing models from the registry");
                self.shared.registry.known()
            }
        }
    }

    fn model(&self, id: &str, record: &Record) -> Model {
        let meta = record.meta.as_ref();
        let update = (record.source == Source::Cache && !record.missing).then(|| self.updates.get(id)).flatten().filter(|u| u.content);
        let view = self.scheduler.view(id);
        let router = self.router_status.lock().expect("status").get(id).cloned().unwrap_or_default();
        let state = if record.missing {
            ModelState::Missing
        } else if (self.downloads.active(id) || router == "downloading") && record.path.as_ref().is_none_or(|p| !p.exists()) {
            ModelState::Downloading
        } else {
            match view {
                Some((Load::Loading, ..)) => ModelState::Loading,
                Some((Load::Unloading, ..)) => ModelState::Unloading,
                Some((Load::Loaded, active, ..)) => {
                    if active > 0 {
                        ModelState::Busy
                    } else {
                        ModelState::Ready
                    }
                }
                _ if router == "loading" => ModelState::Loading,
                _ if record.error.is_some() => ModelState::Failed,
                _ => ModelState::Installed,
            }
        };
        Model {
            id: id.into(),
            name: meta.and_then(|m| m.name.clone()).unwrap_or_else(|| id.into()),
            source: record.source,
            path: record.path.as_ref().map(|p| p.display().to_string()),
            size: record.size,
            architecture: meta.and_then(|m| m.architecture.clone()),
            quantization: record.quantization(),
            parameters: meta.and_then(|m| m.size_label.clone()),
            context_length: meta.and_then(|m| m.context_length),
            capabilities: record.capabilities(),
            state,
            revision: record.revision.clone(),
            pinned: view.is_some_and(|v| v.2),
            removable: record.source == Source::Cache,
            pending_settings: view.is_some_and(|v| v.3),
            error: record.error.clone(),
            update_available: update.is_some(),
            update_size: update.map(|u| u.size),
        }
    }

    async fn list_models(&self) -> Vec<Model> {
        let records = self.records().await;
        let mut models: Vec<Model> = records.iter().map(|(id, r)| self.model(id, r)).collect();
        for d in self.downloads.list() {
            if matches!(d.state, DownloadState::Queued | DownloadState::Running) && !records.contains_key(&d.model) {
                models.push(Model {
                    id: d.model.clone(),
                    name: d.model.clone(),
                    source: Source::Cache,
                    path: None,
                    size: d.total,
                    architecture: None,
                    quantization: None,
                    parameters: None,
                    context_length: None,
                    capabilities: Vec::new(),
                    state: ModelState::Downloading,
                    revision: None,
                    pinned: false,
                    removable: false,
                    pending_settings: false,
                    error: None,
                    update_available: false,
                    update_size: None,
                });
            }
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    /// A model that can run, with its capabilities.
    async fn usable(&self, id: &str) -> Result<(Record, Vec<Capability>), Error> {
        valid_model_id(id).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        let mut record = self.shared.registry.get(id);
        if record.is_none() {
            record = self.records().await.remove(id);
        }
        let record = record.ok_or_else(|| Error::new(ErrorKind::UnknownModel, format!("{id} isn't installed")))?;
        if record.missing {
            return Err(Error::new(ErrorKind::UnknownModel, format!("{id}'s file is gone")));
        }
        let caps = record.capabilities();
        if caps == [Capability::Draft] {
            return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{id} is only a companion drafter")));
        }
        Ok((record, caps))
    }

    // ---------- Operations ----------

    pub async fn handle(&self, caller: &Caller, op: Op, events: mpsc::Sender<Event>, cancel: CancellationToken) -> Result<Reply, Error> {
        match op {
            Op::ListModels {} => Ok(Reply::Models { models: self.list_models().await }),
            Op::ModelInfo { model } => self.model_info(&model).await.map(|d| Reply::ModelInfo(Box::new(d))),
            Op::LoadModel { model, keep } => {
                self.usable(&model).await?;
                if keep {
                    self.scheduler.pin(&model, &caller.name);
                }
                let _ = events.send(Event::ModelLoading { model: model.clone() }).await;
                self.scheduler.load(&model, Priority::Normal, &cancel).await?;
                Ok(Reply::Loaded { model })
            }
            Op::UnloadModel { model } => {
                valid_model_id(&model).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
                let unloaded = self.scheduler.release(&model, &caller.name).await?;
                Ok(Reply::Unloaded { model, unloaded })
            }
            Op::RemoveModel { model } => self.remove(&model).await.map(|_| Reply::Ok),
            Op::ModelSettings { model } => self.settings_of(&model).map(Reply::Settings),
            Op::ConfigureModel { model, settings } => self.configure(&model, &settings).await.map(Reply::Settings),
            Op::DownloadModel { model } => self.download(&model, &events, &cancel).await.map(|_| Reply::Ok),
            Op::CancelDownload { model } => {
                if self.downloads.cancel(&model) {
                    Ok(Reply::Ok)
                } else {
                    Err(Error::new(ErrorKind::UnknownRequest, format!("{model} isn't downloading")))
                }
            }
            Op::Downloads {} => Ok(Reply::Downloads { downloads: self.downloads.list() }),
            Op::Chat(request) => self.counted(self.chat(caller, request, &events, &cancel)).await.map(Reply::Finished),
            Op::Embed { model, input, priority } => {
                self.counted(self.embed(caller, &model, input, priority, &events, &cancel)).await.map(Reply::Embeddings)
            }
            Op::ApplyTemplate(request) => self.counted(self.apply_template(caller, request, &cancel)).await.map(|prompt| Reply::Template { prompt }),
            Op::Cancel { .. } => Err(Error::new(ErrorKind::InvalidRequest, "cancel is handled by the connection")),
            Op::ResourceStatus {} => Ok(Reply::Resources(self.resource_status())),
            Op::WorkerStatus {} => Ok(Reply::Worker(self.worker_status())),
            Op::ListCapabilities {} => Ok(Reply::Capabilities(self.capabilities())),
            Op::ServiceSettings {} => Ok(Reply::Service(self.service_settings())),
            Op::CheckUpdates {} => Ok(Reply::Models { models: self.check_updates().await }),
            Op::ConfigureService { max_loaded, idle_unload_seconds, model_updates, defaults, battery_saver, add_model_dirs } => {
                let rescan = add_model_dirs.is_some();
                let settings = self.configure_service(max_loaded, idle_unload_seconds, model_updates, defaults, battery_saver, add_model_dirs).await?;
                if rescan && let Err(e) = self.refresh(false).await {
                    tracing::warn!(error = %e.message, "couldn't list the new model folders yet");
                }
                if battery_saver.is_some() {
                    self.apply_power().await;
                    return Ok(Reply::Service(self.service_settings()));
                }
                Ok(Reply::Service(settings))
            }
            Op::ListActions {} => Ok(Reply::Actions { actions: self.actions.list().await }),
            Op::InvokeAction { action, arguments, confirmed } => self
                .actions
                .invoke(caller, &action, arguments, confirmed, self.options.allow_untrusted, &cancel)
                .await
                .map(|result| Reply::ActionDone { result }),
            Op::ServeActions { actions } => self.actions.serve(caller, actions, events, cancel).await.map(|_| Reply::Ok),
            Op::ActionResult { invocation, ok, error } => self.actions.answer(caller, invocation, ok, error).map(|_| Reply::Ok),
        }
    }

    async fn counted<T>(&self, work: impl std::future::Future<Output = Result<T, Error>>) -> Result<T, Error> {
        Metrics::inc(&self.metrics.requests);
        let result = work.await;
        if let Err(e) = &result {
            Metrics::inc(if e.kind == ErrorKind::Cancelled { &self.metrics.cancelled } else { &self.metrics.failed });
        }
        result
    }

    async fn model_info(&self, id: &str) -> Result<ModelDetail, Error> {
        valid_model_id(id).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        let records = self.records().await;
        let record = records.get(id).ok_or_else(|| Error::new(ErrorKind::UnknownModel, format!("{id} isn't installed")))?;
        let model = self.model(id, record);
        let meta = record.meta.as_ref();
        let mut detail = ModelDetail {
            template: meta.and_then(|m| m.chat_template.clone()),
            license: meta.and_then(|m| m.license.clone()),
            embedding_length: meta.and_then(|m| m.embedding_length),
            runtime_context: None,
            slots: None,
            reasoning_efforts: Vec::new(),
            model,
        };
        if detail.model.loaded()
            && let Ok(http) = self.shared.worker.ready().await
            && let Ok(props) = http.get(&format!("/props?model={}", encode_query(id))).await
        {
            detail.runtime_context = props["default_generation_settings"]["n_ctx"].as_u64();
            detail.slots = props["total_slots"].as_u64().map(|n| n as u32);
            if props["chat_template_caps"]["supports_reasoning_effort"].as_bool() == Some(true) {
                let checks = REASONING_EFFORTS.iter().map(|level| {
                    let http = http.clone();
                    async move {
                        let body = json!({"model": id, "messages": [{"role": "user", "content": "hello"}],
                            "chat_template_kwargs": {"enable_thinking": true}, "reasoning_effort": level});
                        (*level, http.json(Method::POST, "/apply-template", Some(&body), Duration::from_secs(5)).await.is_ok())
                    }
                });
                detail.reasoning_efforts =
                    futures_util::future::join_all(checks).await.into_iter().filter(|(_, ok)| *ok).map(|(l, _)| l.to_string()).collect();
            }
        }
        Ok(detail)
    }

    async fn remove(&self, id: &str) -> Result<(), Error> {
        valid_model_id(id).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        let records = self.records().await;
        let Some(record) = records.get(id) else {
            let stopped = self.downloads.cancel(id);
            if crate::downloads::discard_partial(&self.shared.paths.cache, id) || stopped {
                tracing::info!(model = %id, "unfinished download removed");
                return Ok(());
            }
            return Err(Error::new(ErrorKind::UnknownModel, format!("{id} isn't installed")));
        };
        if record.source != Source::Cache {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                format!("{id} is a file in {}; AI Core never deletes files there", record.path.as_ref().and_then(|p| p.parent()).map(|p| p.display().to_string()).unwrap_or_default()),
            ));
        }
        self.downloads.cancel(id);
        if record.missing {
            self.shared.registry.forget(id);
            return Ok(());
        }
        if !self.scheduler.unload_if_idle(id).await? {
            return Err(Error::new(ErrorKind::Busy, format!("{id} is answering a request; try again when it's done")));
        }
        let http = self.shared.worker.ready().await?;
        http.json(Method::DELETE, &format!("/models?model={}", encode_query(id)), None, Duration::from_secs(60)).await?;
        self.shared.registry.forget(id);
        tracing::info!(model = %id, "model removed");
        self.refresh(true).await?;
        Ok(())
    }

    fn settings_of(&self, model: &str) -> Result<ModelSettings, Error> {
        if model != "*" {
            valid_model_id(model).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        }
        let presets = self.shared.presets.lock().expect("presets");
        Ok(ModelSettings {
            model: model.into(),
            values: presets.ini.values(model),
            effective: presets.ini.effective(model),
            pending: presets.waiting.values().any(|w| w.contains(model)) || (model == "*" && presets.held.contains_key("*")),
        })
    }

    async fn configure(&self, model: &str, changes: &BTreeMap<String, String>) -> Result<ModelSettings, Error> {
        let old = {
            let mut presets = self.shared.presets.lock().expect("presets");
            if let Ok(text) = std::fs::read_to_string(self.shared.paths.models_ini()) {
                presets.ini = Ini::parse(&text);
            }
            let old = presets.ini.section(model).map(|s| s.to_vec());
            let mut ini = presets.ini.clone();
            ini.set(model, changes).map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
            presets::save(&self.shared.paths, &ini).map_err(|e| Error::new(ErrorKind::Internal, format!("couldn't save models.ini: {e}")))?;
            presets.ini = ini;
            old
        };
        let affected: Vec<String> = if model == "*" {
            self.scheduler.loaded_models().into_iter().map(|m| m.id).collect()
        } else {
            vec![model.to_string()]
        };
        let mut busy = BTreeSet::new();
        for id in affected {
            if !self.scheduler.unload_if_idle(&id).await? {
                busy.insert(id);
            }
        }
        if !busy.is_empty() {
            let mut presets = self.shared.presets.lock().expect("presets");
            presets.held.entry(model.to_string()).or_insert(old);
            presets.waiting.entry(model.to_string()).or_default().extend(busy.iter().cloned());
            drop(presets);
            for id in &busy {
                self.scheduler.set_pending(id, true);
            }
        }
        tracing::info!(%model, keys = ?changes.keys().collect::<Vec<_>>(), "model settings changed");
        if let Err(e) = self.refresh(true).await {
            tracing::debug!(error = %e, "router not refreshed after a settings change");
        }
        self.settings_of(model)
    }

    /// A busy model with changed settings went idle: unload it so the next
    /// request loads it with them.
    async fn apply_held(&self, model: &str) {
        if !self.scheduler.unload_if_idle(model).await.unwrap_or(false) {
            return;
        }
        self.scheduler.set_pending(model, false);
        {
            let mut presets = self.shared.presets.lock().expect("presets");
            let mut released = Vec::new();
            for (key, waiting) in presets.waiting.iter_mut() {
                waiting.remove(model);
                if waiting.is_empty() {
                    released.push(key.clone());
                }
            }
            for key in released {
                presets.waiting.remove(&key);
                presets.held.remove(&key);
            }
        }
        if let Err(e) = self.refresh(true).await {
            tracing::debug!(error = %e, "router not refreshed after applying settings");
        }
    }

    async fn download(&self, model: &str, events: &mpsc::Sender<Event>, cancel: &CancellationToken) -> Result<(), Error> {
        let mut rx = self.downloads.start(model)?;
        let mut last: Option<(String, u64, u64)> = None;
        loop {
            let d: Download = rx.borrow_and_update().clone();
            let now = (d.status.clone(), d.completed, d.total);
            if last.as_ref() != Some(&now) {
                let _ = events.send(Event::Progress { status: d.status.clone(), completed: d.completed, total: d.total }).await;
                last = Some(now);
            }
            match d.state {
                DownloadState::Done => return Ok(()),
                DownloadState::Cancelled => return Err(Error::new(ErrorKind::Cancelled, "download cancelled")),
                DownloadState::Failed => {
                    let message = d.error.unwrap_or_else(|| "download failed".into());
                    let kind = if message.starts_with("not enough disk space") { ErrorKind::ResourceExhausted } else { ErrorKind::DownloadFailed };
                    return Err(Error::new(kind, message));
                }
                _ => {}
            }
            tokio::select! {
                changed = rx.changed() => if changed.is_err() { return Err(Error::new(ErrorKind::Internal, "the download vanished")) },
                _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "stopped watching; the download goes on")),
            }
        }
    }

    async fn chat(&self, caller: &Caller, req: ChatRequest, events: &mpsc::Sender<Event>, cancel: &CancellationToken) -> Result<Finished, Error> {
        req.validate().map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        let (_, caps) = self.usable(&req.model).await?;
        let lacks = |cap: Capability, what: &str| -> Result<(), Error> {
            if caps.contains(&cap) { Ok(()) } else { Err(Error::new(ErrorKind::UnsupportedCapability, format!("{} can't {what}", req.model))) }
        };
        lacks(Capability::Chat, "chat")?;
        if req.has_images() {
            lacks(Capability::Vision, "see images")?;
        }
        if req.has_audio() {
            lacks(Capability::Audio, "hear audio")?;
        }
        if !req.tools.is_empty() {
            lacks(Capability::Tools, "use tools")?;
        }
        let _ = events.send(Event::Accepted).await;
        let lease = self.scheduler.acquire(&req.model, &caller.name, req.priority, cancel, Some(events)).await?;
        let _ = events.send(Event::Started).await;
        let http = self.shared.worker.ready().await?;
        let started = Instant::now();
        let body = stream::body(&req, true);
        let mut response = tokio::select! {
            r = http.send(Method::POST, "/v1/chat/completions", Some(&body)) => r?,
            _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "cancelled")),
        };
        if !(200..300).contains(&response.status) {
            return Err(response.json().await.err().unwrap_or_else(|| Error::new(ErrorKind::WorkerFailed, "llama.cpp refused the chat")));
        }
        let mut sse = Sse::default();
        let mut first = None;
        let mut done = false;
        while !done {
            let chunk = tokio::select! {
                c = response.chunk() => c,
                _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "cancelled")),
            };
            let Some(chunk) = chunk else { break };
            for item in sse.push(&chunk?) {
                match item {
                    stream::Item::Delta(event) => {
                        first.get_or_insert_with(|| started.elapsed());
                        tokio::select! {
                            r = events.send(event) => if r.is_err() { return Err(Error::new(ErrorKind::Cancelled, "the client went away")) },
                            _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "cancelled")),
                        }
                    }
                    stream::Item::Error(message) => return Err(Error::new(ErrorKind::WorkerFailed, format!("llama.cpp: {message}"))),
                    stream::Item::Done => done = true,
                }
            }
        }
        if !done && sse.finish_reason.is_empty() {
            return Err(Error::new(ErrorKind::WorkerFailed, "llama.cpp stopped before the reply was finished"));
        }
        let total = started.elapsed();
        let (first_token_ms, per_second) = match sse.timings {
            Some((prompt_ms, per_second)) => (prompt_ms as u64, per_second),
            None => {
                let first = first.unwrap_or(total);
                let generating = total.saturating_sub(first).as_secs_f64();
                (first.as_millis() as u64, if generating > 0.0 { sse.tokens as f64 / generating } else { 0.0 })
            }
        };
        let mut usage = sse.usage.clone();
        if usage.completion_tokens == 0 {
            usage.completion_tokens = sse.tokens;
        }
        self.metrics.generated(first_token_ms, per_second);
        Ok(Finished {
            finish_reason: if sse.finish_reason.is_empty() { "stop".into() } else { sse.finish_reason.clone() },
            usage,
            timings: Timings {
                queued_ms: lease.queued.as_millis() as u64,
                load_ms: lease.loaded.map(|d| d.as_millis() as u64).unwrap_or(0),
                first_token_ms,
                total_ms: total.as_millis() as u64,
                tokens_per_second: (per_second * 10.0).round() / 10.0,
            },
        })
    }

    async fn embed(
        &self,
        caller: &Caller,
        model: &str,
        input: Vec<EmbedInput>,
        priority: Priority,
        events: &mpsc::Sender<Event>,
        cancel: &CancellationToken,
    ) -> Result<Embeddings, Error> {
        if input.is_empty() || input.len() > 4096 {
            return Err(Error::new(ErrorKind::InvalidRequest, "embed between 1 and 4096 inputs at a time"));
        }
        let (record, caps) = self.usable(model).await?;
        if !caps.contains(&Capability::Embedding) {
            return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{model} isn't an embedding model")));
        }
        for item in &input {
            let (data, needs, what) = match item {
                EmbedInput::Text(_) => continue,
                EmbedInput::Image { image } => (image, Capability::Vision, "images"),
                EmbedInput::Audio { audio } => (audio, Capability::Audio, "audio"),
            };
            if !caps.contains(&needs) {
                return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{model} can't embed {what} (it has no projector for them)")));
            }
            media_base64(data)?;
        }
        let media = input.iter().any(|i| !matches!(i, EmbedInput::Text(_)));
        let _ = events.send(Event::Accepted).await;
        let _lease = self.scheduler.acquire(model, &caller.name, priority, cancel, Some(events)).await?;
        let _ = events.send(Event::Started).await;
        let http = self.shared.worker.ready().await?;
        let marker = if media {
            let props = http.get(&format!("/props?model={}", encode_query(model))).await?;
            Some(props["media_marker"].as_str().unwrap_or("<__media__>").to_string())
        } else {
            None
        };
        let mut vectors = Vec::with_capacity(input.len());
        for batch in input.chunks(if media { 16 } else { 64 }) {
            let (path, body) = match &marker {
                None => {
                    let texts: Vec<&str> = batch.iter().filter_map(|i| if let EmbedInput::Text(t) = i { Some(t.as_str()) } else { None }).collect();
                    ("/v1/embeddings", json!({"model": model, "input": texts}))
                }
                Some(marker) => {
                    let content: Vec<Value> = batch
                        .iter()
                        .map(|i| match i {
                            EmbedInput::Text(t) => json!(t),
                            EmbedInput::Image { image: data } | EmbedInput::Audio { audio: data } => {
                                json!({"prompt_string": marker, "multimodal_data": [media_base64(data).unwrap_or_default()]})
                            }
                        })
                        .collect();
                    ("/embeddings", json!({"model": model, "content": content, "embd_normalize": 2}))
                }
            };
            let reply = tokio::select! {
                r = http.json(Method::POST, path, Some(&body), Duration::from_secs(300)) => r?,
                _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "cancelled")),
            };
            let mut data = reply.get("data").unwrap_or(&reply).as_array().cloned().unwrap_or_default();
            data.sort_by_key(|v| v["index"].as_u64().unwrap_or(0));
            for item in data {
                let embedding = match &item["embedding"] {
                    Value::Array(rows) if rows.first().is_some_and(Value::is_array) => {
                        if rows.len() != 1 {
                            return Err(Error::new(ErrorKind::WorkerFailed, format!("{model} gives a vector per token; set its pooling")));
                        }
                        rows[0].clone()
                    }
                    other => other.clone(),
                };
                vectors.push(embedding.as_array().into_iter().flatten().filter_map(Value::as_f64).map(|v| v as f32).collect::<Vec<f32>>());
            }
        }
        if vectors.len() != input.len() {
            return Err(Error::new(ErrorKind::WorkerFailed, format!("expected {} embeddings, got {}", input.len(), vectors.len())));
        }
        let dimensions = vectors.first().map(Vec::len).unwrap_or(0);
        if vectors.iter().any(|v| v.len() != dimensions || v.is_empty()) {
            return Err(Error::new(ErrorKind::WorkerFailed, "llama.cpp returned embeddings of different sizes"));
        }
        Metrics::inc(&self.metrics.embeddings);
        let file_name = |p: &std::path::Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
        let settings = {
            let presets = self.shared.presets.lock().expect("presets");
            presets.ini.effective(model).into_iter().filter(|(k, _)| presets::VECTOR_KEYS.contains(&k.as_str())).collect()
        };
        Ok(Embeddings {
            identity: ModelIdentity {
                id: model.into(),
                file: record.path.as_deref().and_then(file_name).unwrap_or_default(),
                size: record.size,
                revision: record.revision.clone(),
                architecture: record.meta.as_ref().and_then(|m| m.architecture.clone()),
                projector: record.projector.as_deref().and_then(file_name),
                settings,
            },
            dimensions,
            vectors,
        })
    }

    async fn apply_template(&self, caller: &Caller, req: ChatRequest, cancel: &CancellationToken) -> Result<String, Error> {
        req.validate().map_err(|e| Error::new(ErrorKind::InvalidRequest, e))?;
        let (_, caps) = self.usable(&req.model).await?;
        if !caps.contains(&Capability::Chat) {
            return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{} has no chat template", req.model)));
        }
        let _lease = self.scheduler.acquire(&req.model, &caller.name, req.priority, cancel, None).await?;
        let http = self.shared.worker.ready().await?;
        let reply = http.json(Method::POST, "/apply-template", Some(&stream::body(&req, false)), Duration::from_secs(30)).await?;
        Ok(reply["prompt"].as_str().unwrap_or_default().to_string())
    }

    // ---------- Model updates ----------

    /// Cached models, by ID.
    fn cache_models(&self) -> Vec<String> {
        self.shared
            .registry
            .known()
            .into_iter()
            .filter(|(_, r)| r.source == Source::Cache && !r.missing && !r.draft)
            .map(|(id, _)| id)
            .collect()
    }

    /// Check Hugging Face now; returns the models with updates.
    pub async fn check_updates(&self) -> Vec<Model> {
        let token = crate::hf::token(&self.shared.paths.config_home);
        let found = self
            .updates
            .check(&self.shared.paths.cache, self.downloads.endpoint(), token.as_deref(), self.downloads.sidecars(), &self.cache_models())
            .await;
        let content = found.values().filter(|u| u.content).count();
        tracing::info!(updates = content, unchanged_files = found.len() - content, "checked models for updates");
        self.list_models().await.into_iter().filter(|m| m.update_available).collect()
    }

    /// A download finished: stage its revision, and use it now if its models are free.
    async fn downloaded(&self, model: &str, revision: &str) {
        if let Ok((repo, _)) = crate::downloads::parse_model(model) {
            self.updates.stage(&repo, revision);
        }
        self.apply_staged().await;
    }

    /// Put staged revisions in use where none of the repository's models is
    /// busy or was used in the last two minutes.
    pub async fn apply_staged(&self) {
        let _one = self.switch_lock.lock().await;
        let idle = self.options.update_idle_secs.unwrap_or(120);
        for (repo, revision) in self.updates.staged() {
            let prefix = format!("{repo}:");
            let loaded: Vec<_> = self.scheduler.loaded_models().into_iter().filter(|m| m.id.starts_with(&prefix)).collect();
            if loaded.iter().any(|m| m.active > 0 || m.state != "loaded" || m.idle_secs < idle) {
                continue;
            }
            let mut free = true;
            for m in &loaded {
                free &= self.scheduler.unload_if_idle(&m.id).await.unwrap_or(false);
            }
            if !free {
                continue;
            }
            match crate::downloads::switch_revision(&self.shared.paths.cache, &repo, &revision) {
                Ok(()) => tracing::info!(%repo, %revision, "new revision in use"),
                Err(e) => tracing::warn!(%repo, error = %e, "couldn't switch revisions"),
            }
            self.updates.switched(&repo);
            if let Err(e) = self.refresh(true).await {
                tracing::debug!(error = %e, "router not refreshed after an update");
            }
        }
    }

    /// Check on schedule, and in `auto` mode download updates while nothing runs
    /// and the computer is on mains power.
    pub fn auto_update(self: &Arc<Self>) {
        let (mode, hours) = {
            let s = self.settings.lock().expect("settings");
            (s.model_updates.clone(), s.update_check_hours)
        };
        if mode == "off" {
            return;
        }
        let due = match *self.updates.last_check.lock().expect("updates") {
            None => self.started.elapsed() >= Duration::from_secs(300),
            Some(at) => at.elapsed() >= Duration::from_secs(hours * 3600),
        };
        if due && !self.updates.checking.swap(true, Ordering::SeqCst) {
            let me = self.clone();
            tokio::spawn(async move {
                me.check_updates().await;
                me.updates.checking.store(false, Ordering::SeqCst);
            });
        }
        if mode != "auto" {
            return;
        }
        let power = self.power_supply();
        let busy = self.metrics.active.load(Ordering::Relaxed) > 0;
        if busy || !crate::updates::on_ac(&power) {
            let paused = self.downloads.pause_auto();
            if !paused.is_empty() {
                tracing::info!(models = ?paused, busy, "updates paused");
                let mut tries = self.update_tries.lock().expect("tries");
                for model in paused {
                    tries.remove(&model);
                }
            }
            return;
        }
        for (model, update) in self.updates.found() {
            let Ok((repo, _)) = crate::downloads::parse_model(&model) else { continue };
            let recent = self.update_tries.lock().expect("tries").get(&model).is_some_and(|t| t.elapsed() < Duration::from_secs(3600));
            if self.updates.is_staged(&repo) || self.downloads.active(&model) || recent {
                continue;
            }
            if self.downloads.start_auto(&model).is_ok() {
                self.update_tries.lock().expect("tries").insert(model.clone(), Instant::now());
                tracing::info!(%model, bytes = update.size, files_changed = update.content, "updating");
            }
        }
    }

    // ---------- The OpenAI-compatible endpoint ----------

    pub fn openai_addr(&self) -> Option<std::net::SocketAddr> {
        self.settings.lock().expect("settings").openai_addr()
    }

    /// `GET /v1/models`: the models a coding tool can use.
    pub async fn openai_models(&self) -> Value {
        let data: Vec<Value> = self
            .list_models()
            .await
            .into_iter()
            .filter(|m| matches!(m.state, ModelState::Installed | ModelState::Ready | ModelState::Busy | ModelState::Loading))
            .filter(|m| m.has(Capability::Chat) || m.has(Capability::Embedding))
            .map(|m| {
                json!({"id": m.id, "object": "model", "created": 0, "owned_by": "nyx-ai",
                    "meta": {"size": m.size, "context_length": m.context_length, "capabilities": m.capabilities}})
            })
            .collect();
        json!({"object": "list", "data": data})
    }

    /// Pass one OpenAI-style request to the worker under a lease on its model,
    /// which loads it if needed. The lease lives as long as the reply is read.
    pub async fn forward(&self, path: &str, body: &Value, cancel: &CancellationToken) -> Result<(worker::Response, crate::scheduler::Lease), Error> {
        let mut body = body.clone();
        if body["model"].as_str().is_none_or(str::is_empty) {
            let role = if path.contains("embeddings") { "embedding" } else { "coding" };
            if let Some(model) = self.settings.lock().expect("settings").defaults.get(role) {
                body["model"] = json!(model);
            }
        }
        let body = &body;
        let model = body["model"]
            .as_str()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| Error::new(ErrorKind::InvalidRequest, "name a model (GET /v1/models lists them), or set a default coding model"))?;
        let (_, caps) = self.usable(model).await?;
        let wants = if path.contains("embeddings") { Capability::Embedding } else { Capability::Chat };
        if !caps.contains(&wants) {
            return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{model} can't serve {path}")));
        }
        Metrics::inc(&self.metrics.requests);
        let caller = Caller { name: "openai".into(), trusted: false, connection: 0 };
        let lease = self.scheduler.acquire(model, &caller.name, Priority::Normal, cancel, None).await?;
        let http = self.shared.worker.ready().await?;
        let response = tokio::select! {
            r = http.send(Method::POST, path, Some(body)) => r?,
            _ = cancel.cancelled() => return Err(Error::new(ErrorKind::Cancelled, "cancelled")),
        };
        Ok((response, lease))
    }

    fn resource_status(&self) -> ResourceStatus {
        let mem = resources::detect();
        let settings = self.settings.lock().expect("settings").clone();
        ResourceStatus {
            ram_total: mem.ram,
            ram_available: resources::available_ram(),
            vram_total: mem.vram,
            gpu_backends: resources::gpu_backends(self.shared.worker.program().and_then(|p| p.parent())),
            budget: resources::budget(mem),
            max_loaded: settings.max_loaded,
            idle_unload_seconds: settings.idle_unload_seconds,
            loaded: self.scheduler.loaded_models(),
            queued: self.scheduler.queued(),
            memory_pressure: resources::memory_pressure(),
        }
    }

    fn worker_status(&self) -> WorkerStatus {
        let snap = self.shared.worker.snapshot();
        WorkerStatus {
            state: snap.state,
            release: snap.program.as_deref().and_then(worker::release),
            program: snap.program.map(|p| p.display().to_string()),
            pid: snap.pid,
            restarts: snap.restarts,
            last_error: snap.last_error,
            uptime_secs: snap.uptime.as_secs(),
            metrics: self.metrics.snapshot(self.scheduler.queued() as u64),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            api_version: nyx_ai_proto::API_VERSION,
            server_version: env!("CARGO_PKG_VERSION").into(),
            operations: OPERATIONS.iter().map(|s| s.to_string()).collect(),
            gpu_backends: resources::gpu_backends(self.shared.worker.program().and_then(|p| p.parent())),
            worker_release: self.shared.worker.program().and_then(worker::release),
        }
    }

    fn service_settings(&self) -> ServiceSettings {
        let s = self.settings.lock().expect("settings");
        ServiceSettings {
            max_loaded: s.max_loaded,
            idle_unload_seconds: s.idle_unload_seconds,
            model_dirs: s.model_dirs.clone(),
            openai_endpoint: s.openai_addr().map(|a| a.to_string()),
            model_updates: Some(s.model_updates.clone()),
            defaults: s.defaults.clone(),
            battery_saver: s.battery_saver,
            on_battery: !crate::updates::on_ac(&self.power_supply()),
            cpu_only: self.shared.presets.lock().expect("presets").cpu_only,
        }
    }

    fn power_supply(&self) -> std::path::PathBuf {
        self.options.power_supply.clone().unwrap_or_else(|| "/sys/class/power_supply".into())
    }

    /// Battery saver: on battery every model loads on the CPU; models loaded the
    /// other way unload once idle and come back the new way on their next use.
    pub async fn apply_power(&self) {
        let cpu_only = self.settings.lock().expect("settings").battery_saver && !crate::updates::on_ac(&self.power_supply());
        let switched = {
            let mut presets = self.shared.presets.lock().expect("presets");
            std::mem::replace(&mut presets.cpu_only, cpu_only) != cpu_only
        };
        if switched {
            tracing::info!(cpu_only, "battery saver");
            let loaded: Vec<String> = self.scheduler.loaded_models().into_iter().map(|m| m.id).collect();
            self.power_stale.lock().expect("power").extend(loaded);
            if let Err(e) = self.refresh(true).await {
                tracing::debug!(error = %e, "router not refreshed after a power change");
            }
        }
        let stale: Vec<String> = self.power_stale.lock().expect("power").iter().cloned().collect();
        if stale.is_empty() {
            return;
        }
        let loaded: BTreeSet<String> = self.scheduler.loaded_models().into_iter().map(|m| m.id).collect();
        for id in stale {
            if !loaded.contains(&id) || self.scheduler.unload_if_idle(&id).await.unwrap_or(false) {
                self.power_stale.lock().expect("power").remove(&id);
            }
        }
    }

    async fn configure_service(
        &self,
        max_loaded: Option<u32>,
        idle: Option<i64>,
        updates: Option<String>,
        defaults: Option<BTreeMap<String, String>>,
        battery_saver: Option<bool>,
        add_dirs: Option<Vec<String>>,
    ) -> Result<ServiceSettings, Error> {
        let invalid = |m: String| Error::new(ErrorKind::InvalidRequest, m);
        for dir in add_dirs.iter().flatten() {
            if !(dir.starts_with('/') || dir.starts_with("~/")) || dir.split('/').any(|part| part == "..") {
                return Err(invalid(format!("{dir}: model folders are absolute paths or under ~/")));
            }
            if !crate::paths::expand(dir).is_dir() {
                return Err(invalid(format!("{dir} isn't a folder")));
            }
        }
        if max_loaded.is_some_and(|n| !(1..=crate::config::MAX_LOADED_LIMIT).contains(&n)) {
            return Err(invalid(format!("max_loaded must be 1 to {}", crate::config::MAX_LOADED_LIMIT)));
        }
        if idle.is_some_and(|n| n < -1) {
            return Err(invalid("idle_unload_seconds must be -1 (never) or more".into()));
        }
        if updates.as_deref().is_some_and(|u| !matches!(u, "auto" | "notify" | "off")) {
            return Err(invalid("model_updates must be auto, notify or off".into()));
        }
        if let Some(defaults) = &defaults {
            let records = self.records().await;
            for (role, model) in defaults {
                if !nyx_ai_proto::ROLES.contains(&role.as_str()) {
                    return Err(invalid(format!("no role {role}; roles are {}", nyx_ai_proto::ROLES.join(", "))));
                }
                if model.is_empty() {
                    continue;
                }
                let record = records.get(model).filter(|r| !r.missing).ok_or_else(|| Error::new(ErrorKind::UnknownModel, format!("{model} isn't installed")))?;
                let needs = if role == "embedding" { Capability::Embedding } else { Capability::Chat };
                if !record.capabilities().contains(&needs) {
                    return Err(Error::new(ErrorKind::UnsupportedCapability, format!("{model} can't be the {role} model")));
                }
            }
        }
        let mut settings = self.settings.lock().expect("settings");
        if let Some(u) = updates {
            settings.model_updates = u;
        }
        if let Some(defaults) = defaults {
            for (role, model) in defaults {
                if model.is_empty() {
                    settings.defaults.remove(&role);
                } else {
                    settings.defaults.insert(role, model);
                }
            }
        }
        if let Some(n) = max_loaded {
            settings.max_loaded = n;
        }
        if let Some(b) = battery_saver {
            settings.battery_saver = b;
        }
        if let Some(n) = idle {
            settings.idle_unload_seconds = n;
        }
        for dir in add_dirs.into_iter().flatten() {
            let listed = settings.model_dirs().contains(&crate::paths::expand(&dir));
            if !listed {
                tracing::info!(dir = %dir, "model folder added");
                settings.model_dirs.push(dir);
            }
        }
        settings.save(&self.shared.paths).map_err(|e| Error::new(ErrorKind::Internal, format!("couldn't save settings: {e}")))?;
        self.scheduler.set_limits(settings.max_loaded, settings.idle_unload_seconds);
        tracing::info!(max_loaded = settings.max_loaded, idle = settings.idle_unload_seconds, "service settings changed");
        drop(settings);
        Ok(self.service_settings())
    }
}

/// An image or audio clip's base64, from plain base64 or a `data:` URL.
fn media_base64(data: &str) -> Result<&str, Error> {
    let b64 = match data.strip_prefix("data:") {
        Some(url) => url.split_once(";base64,").map(|(_, b)| b).ok_or_else(|| Error::new(ErrorKind::InvalidRequest, "media data URLs must be base64"))?,
        None => data,
    };
    if b64.is_empty() || b64.len() > 48 << 20 {
        return Err(Error::new(ErrorKind::InvalidRequest, "media must be between 1 byte and 36 MB"));
    }
    if !b64.bytes().all(|c| c.is_ascii_alphanumeric() || b"+/=".contains(&c)) {
        return Err(Error::new(ErrorKind::InvalidRequest, "media must be base64"));
    }
    Ok(b64)
}

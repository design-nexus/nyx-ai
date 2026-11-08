//! Who runs when, and what stays loaded.
//!
//! Every inference request holds a [`Lease`] on its model while it runs; a
//! leased model is never unloaded. Requests wait for a slot in priority order,
//! aging one class every [`AGE_STEP`] so background work always gets its turn.
//! Loading is one at a time: before a load, idle models are evicted (unpinned
//! first, least recently used first) until the count and memory budget allow it.

use crate::diagnostics::Metrics;
use futures_util::future::BoxFuture;
use nyx_ai_proto::{Error, ErrorKind, Event, LoadedModel, Priority};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

pub const AGE_STEP: Duration = Duration::from_secs(20);
const LOAD_WAIT: Duration = Duration::from_secs(600);

/// What the scheduler needs from the worker and the registry.
pub trait Backend: Send + Sync + 'static {
    /// Load a model; returns how many requests it serves at once.
    fn load(&self, model: String) -> BoxFuture<'static, Result<usize, Error>>;
    fn unload(&self, model: String) -> BoxFuture<'static, Result<(), Error>>;
    /// Memory the model will take when loaded with its settings.
    fn estimate(&self, model: &str) -> u64;
    /// Its weights alone.
    fn weights(&self, model: &str) -> u64;
    /// `(budget for loaded models, all memory)`.
    fn memory(&self) -> (u64, u64);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Load {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

struct Slot {
    load: Load,
    active: usize,
    slots: usize,
    pins: BTreeSet<String>,
    last_used: Instant,
    estimate: u64,
    pending: bool,
}

impl Slot {
    fn new() -> Slot {
        Slot { load: Load::Unloaded, active: 0, slots: 1, pins: BTreeSet::new(), last_used: Instant::now(), estimate: 0, pending: false }
    }
}

struct Waiter {
    id: u64,
    model: String,
    client: String,
    priority: Priority,
    since: Instant,
}

impl Waiter {
    /// The class, raised one step per [`AGE_STEP`] waited.
    fn effective(&self, now: Instant) -> u32 {
        let base = match self.priority {
            Priority::Background => 0,
            Priority::Normal => 1,
            Priority::Interactive => 2,
        };
        (base + (now.duration_since(self.since).as_secs() / AGE_STEP.as_secs()) as u32).min(2)
    }
}

#[derive(Default)]
struct State {
    models: HashMap<String, Slot>,
    waiters: Vec<Waiter>,
    next: u64,
    by_client: HashMap<String, usize>,
    /// Busy models a load is waiting to evict: no new requests start on them.
    draining: HashSet<String>,
}

impl State {
    fn slot(&mut self, model: &str) -> &mut Slot {
        self.models.entry(model.to_string()).or_insert_with(Slot::new)
    }

    fn loaded(&self, model: &str) -> bool {
        self.models.get(model).is_some_and(|s| s.load == Load::Loaded)
    }

    fn waiting_for(&self, model: &str) -> bool {
        self.waiters.iter().any(|w| w.model == model)
    }

    /// The waiter that gets the next slot of `model`.
    fn pick(&self, model: &str, now: Instant) -> Option<u64> {
        let interactive = self.waiters.iter().any(|w| w.priority == Priority::Interactive);
        self.waiters
            .iter()
            .filter(|w| w.model == model)
            .filter(|w| !(interactive && w.effective(now) == 0))
            .max_by_key(|w| {
                (w.effective(now), std::cmp::Reverse(self.by_client.get(&w.client).copied().unwrap_or(0)), std::cmp::Reverse(w.since))
            })
            .map(|w| w.id)
    }

    fn position(&self, id: u64, model: &str, now: Instant) -> usize {
        let Some(me) = self.waiters.iter().find(|w| w.id == id) else { return 0 };
        let key = |w: &Waiter| (w.effective(now), std::cmp::Reverse(w.since));
        self.waiters.iter().filter(|w| w.model == model && w.id != id && key(w) > key(me)).count()
    }

    /// An idle model to evict for `model`: unpinned before pinned, then least
    /// recently used.
    fn victim(&self, model: &str) -> Option<String> {
        self.models
            .iter()
            .filter(|(id, s)| *id != model && s.load == Load::Loaded && s.active == 0)
            .filter(|(id, _)| !self.waiting_for(id) || self.draining.contains(*id))
            .min_by_key(|(_, s)| (!s.pins.is_empty(), s.last_used))
            .map(|(id, _)| id.clone())
    }
}

type IdleHook = Box<dyn Fn(String) + Send + Sync>;

pub struct Scheduler {
    backend: Arc<dyn Backend>,
    state: Mutex<State>,
    changed: Notify,
    load_lock: tokio::sync::Mutex<()>,
    max_loaded: AtomicU32,
    idle_secs: AtomicI64,
    metrics: Arc<Metrics>,
    /// Called when a model with changed settings becomes idle.
    on_idle: Mutex<Option<IdleHook>>,
}

/// A running request's hold on its model.
pub struct Lease {
    scheduler: Arc<Scheduler>,
    pub model: String,
    client: String,
    /// Time spent waiting for a slot, and loading if this request loaded it.
    pub queued: Duration,
    pub loaded: Option<Duration>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let idle = {
            let mut state = self.scheduler.state.lock().expect("scheduler");
            if let Some(n) = state.by_client.get_mut(&self.client) {
                *n = n.saturating_sub(1);
            }
            let slot = state.slot(&self.model);
            slot.active = slot.active.saturating_sub(1);
            slot.last_used = Instant::now();
            slot.active == 0 && slot.pending
        };
        Metrics::inc_down(&self.scheduler.metrics.active);
        self.scheduler.changed.notify_waiters();
        if idle && let Some(hook) = self.scheduler.on_idle.lock().expect("hook").as_ref() {
            hook(self.model.clone());
        }
    }
}

impl Metrics {
    fn inc_down(counter: &std::sync::atomic::AtomicU64) {
        let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
    }
}

struct WaiterGuard<'a> {
    scheduler: &'a Scheduler,
    id: u64,
}

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        self.scheduler.state.lock().expect("scheduler").waiters.retain(|w| w.id != self.id);
        self.scheduler.changed.notify_waiters();
    }
}

struct DrainGuard<'a> {
    scheduler: &'a Scheduler,
    model: Option<String>,
}

impl Drop for DrainGuard<'_> {
    fn drop(&mut self) {
        if let Some(model) = self.model.take() {
            self.scheduler.state.lock().expect("scheduler").draining.remove(&model);
            self.scheduler.changed.notify_waiters();
        }
    }
}

fn cancelled() -> Error {
    Error::new(ErrorKind::Cancelled, "cancelled")
}

impl Scheduler {
    pub fn new(backend: Arc<dyn Backend>, max_loaded: u32, idle_secs: i64, metrics: Arc<Metrics>) -> Arc<Scheduler> {
        Arc::new(Scheduler {
            backend,
            state: Mutex::new(State::default()),
            changed: Notify::new(),
            load_lock: tokio::sync::Mutex::new(()),
            max_loaded: AtomicU32::new(max_loaded.max(1)),
            idle_secs: AtomicI64::new(idle_secs),
            metrics,
            on_idle: Mutex::new(None),
        })
    }

    pub fn set_limits(&self, max_loaded: u32, idle_secs: i64) {
        self.max_loaded.store(max_loaded.max(1), Ordering::SeqCst);
        self.idle_secs.store(idle_secs, Ordering::SeqCst);
        self.changed.notify_waiters();
    }

    pub fn on_idle(&self, hook: impl Fn(String) + Send + Sync + 'static) {
        *self.on_idle.lock().expect("hook") = Some(Box::new(hook));
    }

    /// Wait for a slot on `model`, loading it if needed.
    pub async fn acquire(
        self: &Arc<Self>,
        model: &str,
        client: &str,
        priority: Priority,
        cancel: &CancellationToken,
        events: Option<&mpsc::Sender<Event>>,
    ) -> Result<Lease, Error> {
        let started = Instant::now();
        let id = {
            let mut state = self.state.lock().expect("scheduler");
            state.next += 1;
            let id = state.next;
            state.waiters.push(Waiter { id, model: model.into(), client: client.into(), priority, since: started });
            id
        };
        let guard = WaiterGuard { scheduler: self, id };
        let mut announced: Option<usize> = None;
        let mut load_time = None;
        loop {
            if cancel.is_cancelled() {
                return Err(cancelled());
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let (ready, position) = {
                let mut state = self.state.lock().expect("scheduler");
                let now = Instant::now();
                if !state.loaded(model) {
                    (false, None)
                } else {
                    let open = {
                        let slot = state.models.get(model).expect("loaded");
                        slot.active < slot.slots && !state.draining.contains(model)
                    };
                    if open && state.pick(model, now) == Some(id) {
                        let slot = state.slot(model);
                        slot.active += 1;
                        slot.last_used = now;
                        *state.by_client.entry(client.into()).or_default() += 1;
                        state.waiters.retain(|w| w.id != id);
                        (true, None)
                    } else {
                        (false, Some(state.position(id, model, now)))
                    }
                }
            };
            if ready {
                std::mem::forget(guard);
                Metrics::inc(&self.metrics.active);
                return Ok(Lease {
                    scheduler: self.clone(),
                    model: model.into(),
                    client: client.into(),
                    queued: started.elapsed() - load_time.unwrap_or_default(),
                    loaded: load_time,
                });
            }
            match position {
                None => {
                    if let Some(tx) = events {
                        let _ = tx.send(Event::ModelLoading { model: model.into() }).await;
                    }
                    let begun = Instant::now();
                    match self.load(model, priority, cancel).await {
                        Ok(true) => load_time = Some(begun.elapsed()),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                Some(position) => {
                    if announced != Some(position) {
                        announced = Some(position);
                        if let Some(tx) = events {
                            let _ = tx.send(Event::Queued { position }).await;
                        }
                    }
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                        _ = cancel.cancelled() => return Err(cancelled()),
                    }
                }
            }
        }
    }

    /// Load `model` unless it's loaded. The load goes on when `cancel` fires
    /// (others may be waiting for it, and the worker is loading it anyway).
    /// Returns whether this call loaded it.
    pub async fn load(self: &Arc<Self>, model: &str, priority: Priority, cancel: &CancellationToken) -> Result<bool, Error> {
        let me = self.clone();
        let owned = model.to_string();
        let task = tokio::spawn(async move { me.ensure_loaded(&owned, priority).await });
        tokio::select! {
            r = task => r.unwrap_or_else(|e| Err(Error::new(ErrorKind::Internal, format!("the load stopped: {e}")))),
            _ = cancel.cancelled() => Err(cancelled()),
        }
    }

    /// Make sure `model` is loaded. Returns whether this call loaded it.
    pub async fn ensure_loaded(&self, model: &str, priority: Priority) -> Result<bool, Error> {
        let _load = self.load_lock.lock().await;
        if self.state.lock().expect("scheduler").loaded(model) {
            return Ok(false);
        }
        let mut drain = DrainGuard { scheduler: self, model: None };
        let waited = Instant::now();
        loop {
            enum Step {
                Go,
                Evict(String),
                Wait(String),
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let step = {
                let state = self.state.lock().expect("scheduler");
                let others: Vec<(&String, &Slot)> =
                    state.models.iter().filter(|(id, s)| *id != model && matches!(s.load, Load::Loaded | Load::Loading)).collect();
                let used: u64 = others.iter().map(|(_, s)| s.estimate).sum();
                let need = self.backend.estimate(model);
                let (budget, _) = self.backend.memory();
                let over = others.len() as u32 >= self.max_loaded.load(Ordering::SeqCst) || (budget > 0 && used + need > budget);
                if !over || others.is_empty() {
                    Step::Go
                } else if let Some(victim) = state.victim(model) {
                    Step::Evict(victim)
                } else {
                    let busy = others
                        .iter()
                        .filter(|(_, s)| s.load == Load::Loaded)
                        .min_by_key(|(_, s)| s.last_used)
                        .map(|(id, _)| (*id).clone());
                    match busy {
                        Some(id) => Step::Wait(id),
                        None => Step::Go,
                    }
                }
            };
            match step {
                Step::Go => break,
                Step::Evict(victim) => {
                    tracing::info!(model = %victim, for_model = %model, "evicting idle model");
                    self.unload_now(&victim).await?;
                    Metrics::inc(&self.metrics.evictions);
                }
                Step::Wait(busy) => {
                    if waited.elapsed() > LOAD_WAIT {
                        return Err(Error::new(ErrorKind::ResourceExhausted, "other models stayed busy; try again when they're done"));
                    }
                    if priority != Priority::Background && drain.model.as_deref() != Some(busy.as_str()) {
                        self.state.lock().expect("scheduler").draining.insert(busy.clone());
                        drain.model = Some(busy);
                    }
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                    }
                }
            }
        }
        drop(drain);
        let (_, total) = self.backend.memory();
        let weights = self.backend.weights(model);
        if total > 0 && weights > total {
            return Err(Error::new(
                ErrorKind::ResourceExhausted,
                format!("the model needs {} GB and this computer has {} GB of memory", weights / 1_000_000_000 + 1, total / 1_000_000_000),
            ));
        }
        {
            let estimate = self.backend.estimate(model);
            let mut state = self.state.lock().expect("scheduler");
            let slot = state.slot(model);
            slot.load = Load::Loading;
            slot.estimate = estimate;
        }
        self.changed.notify_waiters();
        let begun = Instant::now();
        tracing::info!(%model, "loading model");
        let result = self.backend.load(model.to_string()).await;
        {
            let mut state = self.state.lock().expect("scheduler");
            let slot = state.slot(model);
            match &result {
                Ok(slots) => {
                    slot.load = Load::Loaded;
                    slot.slots = (*slots).max(1);
                    slot.last_used = Instant::now();
                    slot.pending = false;
                }
                Err(_) => slot.load = Load::Unloaded,
            }
        }
        self.changed.notify_waiters();
        match result {
            Ok(_) => {
                self.metrics.loaded(begun.elapsed().as_millis() as u64);
                tracing::info!(%model, ms = begun.elapsed().as_millis() as u64, "model loaded");
                Ok(true)
            }
            Err(e) => {
                tracing::warn!(%model, error = %e, "model failed to load");
                Err(e)
            }
        }
    }

    async fn unload_now(&self, model: &str) -> Result<(), Error> {
        self.state.lock().expect("scheduler").slot(model).load = Load::Unloading;
        self.changed.notify_waiters();
        let result = self.backend.unload(model.to_string()).await;
        {
            let mut state = self.state.lock().expect("scheduler");
            let slot = state.slot(model);
            slot.load = if result.is_ok() { Load::Unloaded } else { Load::Loaded };
            if result.is_ok() {
                slot.pending = false;
            }
        }
        self.changed.notify_waiters();
        result
    }

    /// Keep `model` loaded for `client` until [`Scheduler::release`].
    pub fn pin(&self, model: &str, client: &str) {
        self.state.lock().expect("scheduler").slot(model).pins.insert(client.into());
    }

    /// Drop `client`'s pin and unload the model if nothing else needs it.
    /// Returns whether it's unloaded now.
    pub async fn release(&self, model: &str, client: &str) -> Result<bool, Error> {
        let free = {
            let mut state = self.state.lock().expect("scheduler");
            let waiting = state.waiting_for(model);
            let slot = state.slot(model);
            slot.pins.remove(client);
            match slot.load {
                Load::Unloaded => return Ok(true),
                Load::Loaded => slot.active == 0 && slot.pins.is_empty() && !waiting,
                _ => false,
            }
        };
        if !free {
            return Ok(false);
        }
        let _load = self.load_lock.lock().await;
        let still = {
            let state = self.state.lock().expect("scheduler");
            state.models.get(model).is_some_and(|s| s.load == Load::Loaded && s.active == 0 && s.pins.is_empty()) && !state.waiting_for(model)
        };
        if still {
            self.unload_now(model).await?;
        }
        Ok(still)
    }

    /// Unload `model` if it's idle (for settings that need a fresh load).
    /// Returns false while it's busy.
    pub async fn unload_if_idle(&self, model: &str) -> Result<bool, Error> {
        let _load = self.load_lock.lock().await;
        let (loaded, idle) = {
            let state = self.state.lock().expect("scheduler");
            match state.models.get(model) {
                Some(s) if s.load == Load::Loaded => (true, s.active == 0),
                _ => (false, true),
            }
        };
        if !loaded {
            return Ok(true);
        }
        if !idle {
            return Ok(false);
        }
        self.unload_now(model).await?;
        Ok(true)
    }

    /// Settings changed while loaded: apply when it next goes idle.
    pub fn set_pending(&self, model: &str, pending: bool) {
        self.state.lock().expect("scheduler").slot(model).pending = pending;
    }

    /// Unload idle models past the idle time, and one more under memory pressure.
    pub async fn tick(&self, pressure: bool) {
        let Ok(_load) = self.load_lock.try_lock() else { return };
        let idle = self.idle_secs.load(Ordering::SeqCst);
        let (expired, extra) = {
            let state = self.state.lock().expect("scheduler");
            let free: Vec<(&String, &Slot)> = state
                .models
                .iter()
                .filter(|(id, s)| s.load == Load::Loaded && s.active == 0 && s.pins.is_empty() && !state.waiting_for(id))
                .collect();
            let expired: Vec<String> = free
                .iter()
                .filter(|(_, s)| idle >= 0 && s.last_used.elapsed() >= Duration::from_secs(idle as u64))
                .map(|(id, _)| (*id).clone())
                .collect();
            let extra = if pressure { free.iter().min_by_key(|(_, s)| s.last_used).map(|(id, _)| (*id).clone()) } else { None };
            (expired, extra)
        };
        for model in expired.iter().chain(extra.iter().filter(|m| !expired.contains(m))) {
            tracing::info!(%model, pressure, "unloading idle model");
            if self.unload_now(model).await.is_ok() {
                Metrics::inc(&self.metrics.idle_unloads);
            }
        }
    }

    /// The worker restarted: nothing is loaded any more.
    pub fn reset(&self) {
        let mut state = self.state.lock().expect("scheduler");
        for slot in state.models.values_mut() {
            slot.load = Load::Unloaded;
            slot.pending = false;
        }
        state.draining.clear();
        drop(state);
        self.changed.notify_waiters();
    }

    /// Match the router: a model it reports unloaded (a crashed instance) is.
    pub fn sync(&self, loaded: &HashSet<String>) {
        let mut state = self.state.lock().expect("scheduler");
        for (id, slot) in state.models.iter_mut() {
            if slot.load == Load::Loaded && !loaded.contains(id) {
                slot.load = Load::Unloaded;
            }
        }
    }

    pub fn view(&self, model: &str) -> Option<(Load, usize, bool, bool)> {
        let state = self.state.lock().expect("scheduler");
        state.models.get(model).map(|s| (s.load, s.active, !s.pins.is_empty(), s.pending))
    }

    pub fn queued(&self) -> usize {
        self.state.lock().expect("scheduler").waiters.len()
    }

    pub fn loaded_models(&self) -> Vec<LoadedModel> {
        let state = self.state.lock().expect("scheduler");
        let mut out: Vec<LoadedModel> = state
            .models
            .iter()
            .filter(|(_, s)| s.load != Load::Unloaded)
            .map(|(id, s)| LoadedModel {
                id: id.clone(),
                state: format!("{:?}", s.load).to_lowercase(),
                estimate: s.estimate,
                active: s.active,
                slots: s.slots,
                pinned_by: s.pins.iter().cloned().collect(),
                idle_secs: if s.active == 0 { s.last_used.elapsed().as_secs() } else { 0 },
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct Fake {
        loads: Mutex<Vec<String>>,
        unloads: Mutex<Vec<String>>,
        slots: AtomicUsize,
        budget: u64,
    }

    impl Backend for Fake {
        fn load(&self, model: String) -> BoxFuture<'static, Result<usize, Error>> {
            self.loads.lock().unwrap().push(model.clone());
            let slots = self.slots.load(Ordering::SeqCst).max(1);
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                if model == "broken" { Err(Error::new(ErrorKind::ModelLoadFailed, "nope")) } else { Ok(slots) }
            })
        }
        fn unload(&self, model: String) -> BoxFuture<'static, Result<(), Error>> {
            self.unloads.lock().unwrap().push(model);
            Box::pin(async { Ok(()) })
        }
        fn estimate(&self, _: &str) -> u64 {
            10
        }
        fn weights(&self, model: &str) -> u64 {
            if model == "huge" { 1_000 } else { 5 }
        }
        fn memory(&self) -> (u64, u64) {
            (self.budget, 100)
        }
    }

    fn setup(max: u32, slots: usize) -> (Arc<Scheduler>, Arc<Fake>) {
        let fake = Arc::new(Fake { budget: 1_000, ..Default::default() });
        fake.slots.store(slots, Ordering::SeqCst);
        (Scheduler::new(fake.clone(), max, 60, Arc::new(Metrics::default())), fake)
    }

    async fn lease(s: &Arc<Scheduler>, model: &str, client: &str, p: Priority) -> Result<Lease, Error> {
        s.acquire(model, client, p, &CancellationToken::new(), None).await
    }

    #[tokio::test]
    async fn loads_once_and_evicts_lru_idle() {
        let (s, fake) = setup(1, 2);
        let a1 = lease(&s, "a", "chat", Priority::Normal).await.unwrap();
        let a2 = lease(&s, "a", "chat", Priority::Normal).await.unwrap();
        assert_eq!(*fake.loads.lock().unwrap(), ["a"]);
        drop((a1, a2));
        let b = lease(&s, "b", "glimpse", Priority::Normal).await.unwrap();
        assert_eq!(*fake.unloads.lock().unwrap(), ["a"]);
        assert!(b.loaded.is_some());
        assert_eq!(s.view("a").unwrap().0, Load::Unloaded);
    }

    #[tokio::test]
    async fn a_leased_model_is_never_evicted() {
        let (s, fake) = setup(1, 1);
        let a = lease(&s, "a", "chat", Priority::Normal).await.unwrap();
        let s2 = s.clone();
        let other = tokio::spawn(async move { lease(&s2, "b", "glimpse", Priority::Normal).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(fake.unloads.lock().unwrap().is_empty());
        // While b waits to load, a takes no new requests.
        let s3 = s.clone();
        let blocked = tokio::spawn(async move { tokio::time::timeout(Duration::from_millis(150), lease(&s3, "a", "chat", Priority::Normal)).await.is_err() });
        assert!(blocked.await.unwrap());
        drop(a);
        other.await.unwrap().unwrap();
        assert_eq!(*fake.unloads.lock().unwrap(), ["a"]);
    }

    #[tokio::test]
    async fn priorities_and_fairness() {
        let (s, _) = setup(2, 1);
        let first = lease(&s, "a", "x", Priority::Normal).await.unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (name, p) in [("background", Priority::Background), ("normal", Priority::Normal), ("interactive", Priority::Interactive)] {
            let (s, order) = (s.clone(), order.clone());
            tasks.push(tokio::spawn(async move {
                let l = lease(&s, "a", name, p).await.unwrap();
                order.lock().unwrap().push(name);
                tokio::time::sleep(Duration::from_millis(20)).await;
                drop(l);
            }));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        drop(first);
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), ["interactive", "normal", "background"]);
    }

    #[test]
    fn aging_lifts_background_work() {
        let now = Instant::now();
        let w = Waiter { id: 1, model: "a".into(), client: "c".into(), priority: Priority::Background, since: now - AGE_STEP * 2 };
        assert_eq!(w.effective(now), 2);
        let fresh = Waiter { since: now, ..w };
        assert_eq!(fresh.effective(now), 0);
    }

    #[tokio::test]
    async fn pins_cancel_and_failures() {
        let (s, fake) = setup(1, 1);
        s.ensure_loaded("a", Priority::Normal).await.unwrap();
        s.pin("a", "chat");
        assert!(!s.release("a", "other").await.unwrap());
        s.tick(false).await;
        assert!(fake.unloads.lock().unwrap().is_empty());
        assert!(s.release("a", "chat").await.unwrap());

        let held = lease(&s, "b", "x", Priority::Normal).await.unwrap();
        let cancel = CancellationToken::new();
        let s2 = s.clone();
        let c2 = cancel.clone();
        let waiting = tokio::spawn(async move { s2.acquire("b", "y", Priority::Normal, &c2, None).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        assert_eq!(waiting.await.unwrap().unwrap_err().kind, ErrorKind::Cancelled);
        assert_eq!(s.queued(), 0);
        drop(held);

        assert_eq!(lease(&s, "broken", "x", Priority::Normal).await.err().unwrap().kind, ErrorKind::ModelLoadFailed);
        assert_eq!(lease(&s, "huge", "x", Priority::Normal).await.err().unwrap().kind, ErrorKind::ResourceExhausted);
    }

    #[tokio::test]
    async fn a_cancelled_request_doesnt_strand_its_load() {
        let (s, fake) = setup(1, 1);
        let cancel = CancellationToken::new();
        let s2 = s.clone();
        let c2 = cancel.clone();
        let waiting = tokio::spawn(async move { s2.acquire("a", "x", Priority::Normal, &c2, None).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(1)).await;
        cancel.cancel();
        assert_eq!(waiting.await.unwrap().unwrap_err().kind, ErrorKind::Cancelled);
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(s.view("a").unwrap().0, Load::Loaded);
        drop(lease(&s, "b", "y", Priority::Normal).await.unwrap());
        assert_eq!(*fake.unloads.lock().unwrap(), ["a"]);
    }

    #[tokio::test]
    async fn idle_unload_and_reset() {
        let (s, fake) = setup(2, 1);
        s.set_limits(2, 0);
        drop(lease(&s, "a", "x", Priority::Normal).await.unwrap());
        s.tick(false).await;
        assert_eq!(*fake.unloads.lock().unwrap(), ["a"]);
        s.set_limits(2, -1);
        drop(lease(&s, "b", "x", Priority::Normal).await.unwrap());
        s.tick(false).await;
        assert_eq!(fake.unloads.lock().unwrap().len(), 1);
        s.tick(true).await;
        assert_eq!(fake.unloads.lock().unwrap().len(), 2);
        drop(lease(&s, "c", "x", Priority::Normal).await.unwrap());
        s.reset();
        assert_eq!(s.view("c").unwrap().0, Load::Unloaded);
    }
}

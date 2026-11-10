//! The shared action registry as AI Core offers it: Nyx apps' actions (installed apps'
//! manifests and running apps' control interfaces, through nyx-actions, the same
//! registry the `nyx` command and Glimpse link) and actions providers register over
//! their AI Core connection.
//!
//! AI Core checks a call's shape (the arguments against the action's schema) and the
//! caller's standing (trusted program, the person's confirmation when the action asks
//! for it), then routes it. The owning app decides whether to do it: it checks its own
//! permissions and may refuse.

use crate::service::Caller;
use nyx_actions::{InvokeOptions, Listed, Options, Registry, Risk};
use nyx_ai_proto::{ActionInfo, ActionSpec, Confirmation, Error, ErrorKind, Event, SideEffects};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const MAX_ACTIONS: usize = nyx_actions::spec::MAX_ACTIONS;
const DEFAULT_TIMEOUT: Duration = Duration::from_millis(nyx_actions::spec::DEFAULT_TIMEOUT_MS);
const MAX_TIMEOUT: Duration = Duration::from_millis(nyx_actions::spec::MAX_TIMEOUT_MS);

struct Provider {
    info: ActionInfo,
    connection: u64,
    events: mpsc::Sender<Event>,
}

struct Pending {
    connection: u64,
    reply: oneshot::Sender<Result<Value, String>>,
}

pub struct Actions {
    providers: Mutex<HashMap<String, Provider>>,
    pending: Mutex<HashMap<u64, Pending>>,
    registry: Registry,
    next: AtomicU64,
}

impl Default for Actions {
    fn default() -> Self {
        Actions::with_registry(Registry::new(Options::default()))
    }
}

impl Actions {
    pub fn with_registry(registry: Registry) -> Self {
        Actions { providers: Mutex::default(), pending: Mutex::default(), registry, next: AtomicU64::new(0) }
    }

    /// Everything on offer: registered providers first, then Nyx apps. With `detail`,
    /// also the apps as nyx-actions describes them (names, aliases, risk, topics…).
    pub async fn list(&self, detail: bool) -> (Vec<ActionInfo>, Option<Vec<Value>>) {
        let catalog = self.registry.catalog().await;
        let mut seen = std::collections::HashSet::new();
        let mut list: Vec<ActionInfo> = Vec::new();
        for p in self.providers.lock().expect("providers").values() {
            seen.insert(p.info.spec.id.clone());
            list.push(p.info.clone());
        }
        for (entry, listed) in catalog.actions() {
            if seen.insert(listed.action.id.clone()) {
                list.push(info(&entry.app.app, listed));
            }
        }
        list.sort_by(|a, b| a.spec.id.cmp(&b.spec.id));
        let apps = detail.then(|| catalog.apps.iter().filter_map(|e| serde_json::to_value(e).ok()).collect());
        (list, apps)
    }

    /// Offer `specs` for this connection until `cancel`.
    pub async fn serve(&self, caller: &Caller, specs: Vec<ActionSpec>, events: mpsc::Sender<Event>, cancel: CancellationToken) -> Result<(), Error> {
        if specs.is_empty() || specs.len() > MAX_ACTIONS {
            return Err(invalid(format!("offer between 1 and {MAX_ACTIONS} actions")));
        }
        for spec in &specs {
            check_spec(spec)?;
        }
        let ids: Vec<String> = specs.iter().map(|s| s.id.clone()).collect();
        {
            let mut providers = self.providers.lock().expect("providers");
            if let Some(taken) = ids.iter().find(|id| providers.contains_key(*id)) {
                return Err(invalid(format!("{taken} is already offered")));
            }
            if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
                return Err(invalid("an action is offered twice"));
            }
            for spec in specs {
                let info = ActionInfo { spec, owner: caller.name.clone(), source: "connection".into(), available: true };
                providers.insert(info.spec.id.clone(), Provider { info, connection: caller.connection, events: events.clone() });
            }
        }
        tracing::info!(client = %caller.name, actions = ids.len(), "serving actions");
        cancel.cancelled().await;
        self.providers.lock().expect("providers").retain(|_, p| p.connection != caller.connection);
        self.pending.lock().expect("pending").retain(|_, p| p.connection != caller.connection);
        Ok(())
    }

    /// The answer to invocation `id`, from the connection it was sent to.
    pub fn answer(&self, caller: &Caller, id: u64, ok: Option<Value>, error: Option<String>) -> Result<(), Error> {
        let mut pending = self.pending.lock().expect("pending");
        if pending.get(&id).is_none_or(|p| p.connection != caller.connection) {
            return Err(Error::new(ErrorKind::UnknownRequest, format!("no open invocation {id}")));
        }
        let p = pending.remove(&id).expect("checked");
        let _ = p.reply.send(match (ok, error) {
            (_, Some(e)) => Err(e),
            (ok, None) => Ok(ok.unwrap_or(Value::Null)),
        });
        Ok(())
    }

    /// Check and route one call.
    pub async fn invoke(&self, caller: &Caller, id: &str, arguments: Value, confirmed: bool, allow_untrusted: bool, cancel: &CancellationToken) -> Result<Value, Error> {
        let trusted = caller.trusted || allow_untrusted;
        let arguments = if arguments.is_null() { json!({}) } else { arguments };
        let provider = self.providers.lock().expect("providers").get(id).map(|p| (p.info.clone(), p.events.clone(), p.connection));
        let Some((info, events, connection)) = provider else {
            let options = InvokeOptions { confirmed, trusted, on_behalf_of: Some(caller.name.clone()), frontend: "ai-core".into() };
            return tokio::select! {
                r = self.registry.invoke(id, arguments, &options) => r.map_err(from_actions),
                _ = cancel.cancelled() => Err(Error::new(ErrorKind::Cancelled, "cancelled")),
            };
        };
        let started = Instant::now();
        nyx_actions::check::check_call(trusted, &action(&info.spec), &arguments, confirmed).map_err(from_actions)?;
        let timeout = info.spec.timeout_ms.map(Duration::from_millis).unwrap_or(DEFAULT_TIMEOUT).min(MAX_TIMEOUT);
        let result = tokio::select! {
            r = tokio::time::timeout(timeout, self.call_provider(caller, events, connection, id, arguments)) => r.unwrap_or_else(|_| Err(Error::new(ErrorKind::Timeout, format!("{id} took longer than {} s", timeout.as_secs())))),
            _ = cancel.cancelled() => Err(Error::new(ErrorKind::Cancelled, "cancelled")),
        };
        let outcome = match &result {
            Ok(_) => "ok",
            Err(e) if e.kind == ErrorKind::ActionFailed => "refused or failed",
            Err(_) => "error",
        };
        tracing::info!(action = id, client = %caller.name, owner = %info.owner, outcome, ms = started.elapsed().as_millis() as u64, "action");
        result
    }

    async fn call_provider(&self, caller: &Caller, events: mpsc::Sender<Event>, connection: u64, id: &str, arguments: Value) -> Result<Value, Error> {
        let invocation = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let (reply, answer) = oneshot::channel();
        self.pending.lock().expect("pending").insert(invocation, Pending { connection, reply });
        let event = Event::Invoke { invocation, action: id.to_string(), arguments, caller: caller.name.clone() };
        if events.send(event).await.is_err() {
            self.pending.lock().expect("pending").remove(&invocation);
            return Err(gone(id));
        }
        let _cleanup = PendingGuard(self, invocation);
        match answer.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => Err(Error::new(ErrorKind::ActionFailed, e)),
            Err(_) => Err(gone(id)),
        }
    }
}

/// An app's action in protocol version 1's shape.
fn info(owner: &str, listed: &Listed) -> ActionInfo {
    let a = &listed.action;
    let side_effects = match a.risk {
        Risk::ReadOnly => SideEffects::ReadOnly,
        Risk::Writes => SideEffects::Writes,
        Risk::External => SideEffects::External,
        // Version 1 has no privileged class: its strictest reading.
        Risk::Destructive | Risk::Privileged => SideEffects::Destructive,
        Risk::Unknown => SideEffects::Unknown,
    };
    let spec = ActionSpec {
        id: a.id.clone(),
        description: a.description.clone(),
        input_schema: a.input_schema.clone(),
        output_schema: a.output_schema.clone(),
        permissions: a.permissions.clone(),
        confirmation: if a.asks() { Confirmation::Ask } else { Confirmation::Never },
        side_effects,
        timeout_ms: a.timeout_ms,
    };
    let source = if listed.source == nyx_actions::Source::Manifest { "manifest" } else { "dbus" };
    ActionInfo { spec, owner: owner.to_string(), source: source.into(), available: listed.available }
}

/// A provider's spec as nyx-actions checks it.
fn action(spec: &ActionSpec) -> nyx_actions::Action {
    let risk = match spec.side_effects {
        SideEffects::ReadOnly => Risk::ReadOnly,
        SideEffects::Writes => Risk::Writes,
        SideEffects::External => Risk::External,
        SideEffects::Destructive => Risk::Destructive,
        SideEffects::Unknown => Risk::Unknown,
    };
    let asks = spec.confirmation == Confirmation::Ask || risk.always_asks();
    nyx_actions::Action {
        id: spec.id.clone(),
        title: spec.id.clone(),
        description: spec.description.clone(),
        input_schema: spec.input_schema.clone(),
        output_schema: spec.output_schema.clone(),
        risk,
        confirmation: if asks { nyx_actions::Confirmation::Ask } else { nyx_actions::Confirmation::Never },
        aliases: Vec::new(),
        keywords: Vec::new(),
        positional: Vec::new(),
        permissions: spec.permissions.clone(),
        requires: Vec::new(),
        timeout_ms: spec.timeout_ms,
        suggest: Default::default(),
        result: None,
    }
}

fn from_actions(e: nyx_actions::Error) -> Error {
    use nyx_actions::ErrorKind as K;
    let kind = match e.kind {
        K::InvalidArguments => ErrorKind::InvalidRequest,
        K::UnknownApp | K::UnknownAction => ErrorKind::UnknownAction,
        K::NotInstalled | K::NotRunning | K::Unavailable => ErrorKind::Unavailable,
        K::PermissionDenied => ErrorKind::PermissionDenied,
        K::ConfirmationRequired => ErrorKind::ConfirmationRequired,
        K::Failed => ErrorKind::ActionFailed,
        K::Timeout => ErrorKind::Timeout,
        K::Cancelled => ErrorKind::Cancelled,
        K::Internal => ErrorKind::Internal,
    };
    Error::new(kind, e.message)
}

/// Drops an invocation nobody will answer (the caller gave up).
struct PendingGuard<'a>(&'a Actions, u64);

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        self.0.pending.lock().expect("pending").remove(&self.1);
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidRequest, message)
}

fn gone(id: &str) -> Error {
    Error::new(ErrorKind::Unavailable, format!("the app offering {id} went away"))
}

pub fn valid_action_id(id: &str) -> bool {
    nyx_actions::spec::valid_action_id(id)
}

fn check_spec(spec: &ActionSpec) -> Result<(), Error> {
    if !valid_action_id(&spec.id) {
        return Err(invalid(format!("{:?} isn't an action ID (app.verb, lowercase)", spec.id)));
    }
    if spec.description.trim().is_empty() || spec.description.len() > 2000 {
        return Err(invalid(format!("{} needs a description (up to 2000 characters)", spec.id)));
    }
    for schema in std::iter::once(&spec.input_schema).chain(spec.output_schema.as_ref()) {
        jsonschema::validator_for(schema).map_err(|e| invalid(format!("{}'s schema doesn't compile: {e}", spec.id)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, side_effects: SideEffects, confirmation: Confirmation) -> ActionSpec {
        ActionSpec {
            id: id.into(),
            description: "test".into(),
            input_schema: json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": false}),
            output_schema: None,
            permissions: vec![],
            confirmation,
            side_effects,
            timeout_ms: None,
        }
    }

    fn check(trusted: bool, s: &ActionSpec, arguments: &Value, confirmed: bool) -> Result<(), Error> {
        nyx_actions::check::check_call(trusted, &action(s), arguments, confirmed).map_err(from_actions)
    }

    #[test]
    fn provider_calls_are_checked() {
        let read = spec("notes.find", SideEffects::ReadOnly, Confirmation::Never);
        let write = spec("notes.create", SideEffects::Writes, Confirmation::Ask);
        let ok = json!({"text": "milk"});
        check(true, &read, &ok, false).unwrap();
        check(false, &read, &ok, false).unwrap();
        assert_eq!(check(false, &write, &ok, true).unwrap_err().kind, ErrorKind::PermissionDenied);
        assert_eq!(check(true, &write, &ok, false).unwrap_err().kind, ErrorKind::ConfirmationRequired);
        check(true, &write, &ok, true).unwrap();
        let bad = check(true, &read, &json!({"text": 3}), false).unwrap_err();
        assert_eq!(bad.kind, ErrorKind::InvalidRequest);
        assert!(bad.message.contains("/text"), "{}", bad.message);
        assert!(check(true, &read, &json!({"text": "a", "extra": 1}), false).is_err());
        let delete = spec("notes.drop", SideEffects::Destructive, Confirmation::Never);
        assert_eq!(check(true, &delete, &ok, false).unwrap_err().kind, ErrorKind::ConfirmationRequired, "destructive always asks");
    }

    #[test]
    fn specs_are_checked() {
        let mut s = spec("notes.create", SideEffects::Writes, Confirmation::Ask);
        check_spec(&s).unwrap();
        s.input_schema = json!({"type": 12});
        assert!(check_spec(&s).is_err());
        s.input_schema = json!({"type": "object"});
        s.description = " ".into();
        assert!(check_spec(&s).is_err());
        s.id = "Notes".into();
        assert!(check_spec(&s).is_err());
    }

    #[test]
    fn app_actions_read_as_version_1() {
        let d = nyx_actions::Description::from_value(
            &json!({"tools": [{"name": "files.trash", "description": "Trash.", "x-nyx": {"risk": "privileged"}}],
                "nyx": {"api": 2, "app": "io.github.design_nexus.nyx.Files", "name": "Files", "namespaces": ["files"], "bus_name": "io.github.design_nexus.nyx.Files", "activation": "dbus"}}),
            None,
        )
        .unwrap();
        let (catalog, _) = nyx_actions::Catalog::merge(vec![d], vec![], &|_| None);
        let (entry, listed) = catalog.find("files.trash").unwrap();
        let v1 = info(&entry.app.app, listed);
        assert_eq!((v1.spec.side_effects, v1.spec.confirmation, v1.source.as_str(), v1.available), (SideEffects::Destructive, Confirmation::Ask, "manifest", true));
        // An old client reads it: ActionSpec is strict, so nothing new may appear in it.
        let text = serde_json::to_string(&v1).unwrap();
        serde_json::from_str::<ActionInfo>(&text).unwrap();
    }
}

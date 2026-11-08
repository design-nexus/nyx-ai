//! The shared action registry: what apps can do for an assistant. Actions come from
//! the apps' D-Bus control interfaces (`io.github.design_nexus.nyx.Control`) and from
//! providers that register over their AI Core connection.
//!
//! AI Core checks a call's shape (the arguments against the action's schema) and the
//! caller's standing (trusted program, the person's confirmation when the action asks
//! for it), then routes it. The owning app decides whether to do it: it checks its own
//! permissions and may refuse.

use crate::service::Caller;
use nyx_ai_proto::{ActionInfo, ActionSpec, Confirmation, Error, ErrorKind, Event, SideEffects};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const CONTROL_INTERFACE: &str = "io.github.design_nexus.nyx.Control";
const BUS_PREFIX: &str = "io.github.design_nexus.nyx.";
const MAX_ACTIONS: usize = 64;
const MAX_ARGUMENTS: usize = 256 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_TIMEOUT: Duration = Duration::from_secs(300);
/// How long a D-Bus discovery stays fresh.
const DISCOVERY_TTL: Duration = Duration::from_secs(10);
const DESCRIBE_TIMEOUT: Duration = Duration::from_secs(2);

struct DbusAction {
    info: ActionInfo,
    bus_name: String,
    path: String,
}

struct Provider {
    info: ActionInfo,
    connection: u64,
    events: mpsc::Sender<Event>,
}

struct Pending {
    connection: u64,
    reply: oneshot::Sender<Result<Value, String>>,
}

#[derive(Default)]
pub struct Actions {
    providers: Mutex<HashMap<String, Provider>>,
    pending: Mutex<HashMap<u64, Pending>>,
    dbus: Mutex<(Vec<DbusAction>, Option<Instant>)>,
    discovery: tokio::sync::Mutex<()>,
    bus: tokio::sync::OnceCell<Option<zbus::Connection>>,
    next: AtomicU64,
}

impl Actions {
    /// Everything on offer: registered providers first, then running apps' control
    /// interfaces.
    pub async fn list(&self) -> Vec<ActionInfo> {
        self.discover(false).await;
        let mut seen = std::collections::HashSet::new();
        let mut list: Vec<ActionInfo> = Vec::new();
        for p in self.providers.lock().expect("providers").values() {
            seen.insert(p.info.spec.id.clone());
            list.push(p.info.clone());
        }
        for d in &self.dbus.lock().expect("dbus").0 {
            if seen.insert(d.info.spec.id.clone()) {
                list.push(d.info.clone());
            }
        }
        list.sort_by(|a, b| a.spec.id.cmp(&b.spec.id));
        list
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
        let started = Instant::now();
        let route = match self.find(id) {
            Some(route) => route,
            None => {
                self.discover(true).await;
                self.find(id).ok_or_else(|| Error::new(ErrorKind::UnknownAction, format!("no app offers {id}")))?
            }
        };
        let info = route.info();
        let arguments = if arguments.is_null() { json!({}) } else { arguments };
        check_call(caller, &info, &arguments, confirmed, allow_untrusted)?;
        let timeout = info.spec.timeout_ms.map(Duration::from_millis).unwrap_or(DEFAULT_TIMEOUT).min(MAX_TIMEOUT);
        let call = async {
            match route {
                Route::Dbus { bus_name, path, .. } => self.call_dbus(&bus_name, &path, id, &arguments).await,
                Route::Provider { events, connection, .. } => self.call_provider(caller, events, connection, id, arguments).await,
            }
        };
        let result = tokio::select! {
            r = tokio::time::timeout(timeout, call) => r.unwrap_or_else(|_| Err(Error::new(ErrorKind::Timeout, format!("{id} took longer than {} s", timeout.as_secs())))),
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

    fn find(&self, id: &str) -> Option<Route> {
        if let Some(p) = self.providers.lock().expect("providers").get(id) {
            return Some(Route::Provider { info: Box::new(p.info.clone()), events: p.events.clone(), connection: p.connection });
        }
        let dbus = self.dbus.lock().expect("dbus");
        let d = dbus.0.iter().find(|d| d.info.spec.id == id)?;
        Some(Route::Dbus { info: Box::new(d.info.clone()), bus_name: d.bus_name.clone(), path: d.path.clone() })
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

    async fn call_dbus(&self, bus_name: &str, path: &str, id: &str, arguments: &Value) -> Result<Value, Error> {
        let bus = self.bus().await.ok_or_else(|| gone(id))?;
        let reply = bus
            .call_method(Some(bus_name), path, Some(CONTROL_INTERFACE), "Call", &(id, arguments.to_string()))
            .await
            .map_err(|e| Error::new(ErrorKind::Unavailable, format!("{bus_name} didn't answer: {e}")))?;
        let text: String = reply.body().deserialize().map_err(|e| Error::new(ErrorKind::ActionFailed, format!("{bus_name} answered oddly: {e}")))?;
        let value: Value = serde_json::from_str(&text).map_err(|_| Error::new(ErrorKind::ActionFailed, format!("{bus_name} answered with something that isn't JSON")))?;
        match (value.get("ok"), value.get("error")) {
            (_, Some(e)) => Err(Error::new(ErrorKind::ActionFailed, e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string()))),
            (Some(ok), None) => Ok(ok.clone()),
            (None, None) => Ok(Value::Null),
        }
    }

    async fn bus(&self) -> Option<zbus::Connection> {
        self.bus
            .get_or_init(|| async {
                match zbus::Connection::session().await {
                    Ok(bus) => Some(bus),
                    Err(e) => {
                        tracing::info!(error = %e, "no session bus; only registered actions are offered");
                        None
                    }
                }
            })
            .await
            .clone()
    }

    /// Ask running Nyx apps what they offer, unless the last answer is still fresh.
    async fn discover(&self, force: bool) {
        let _one = self.discovery.lock().await;
        let fresh = self.dbus.lock().expect("dbus").1.is_some_and(|t| t.elapsed() < DISCOVERY_TTL);
        if fresh && !force {
            return;
        }
        let Some(bus) = self.bus().await else { return };
        let names = match zbus::fdo::DBusProxy::new(&bus).await {
            Ok(proxy) => proxy.list_names().await.unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        let apps: Vec<String> = names.iter().map(|n| n.to_string()).filter(|n| n.starts_with(BUS_PREFIX)).collect();
        let answers = futures_util::future::join_all(apps.iter().map(|name| {
            let bus = bus.clone();
            async move {
                let path = object_path(name);
                let call = bus.call_method(Some(name.as_str()), path.as_str(), Some(CONTROL_INTERFACE), "Describe", &());
                let text: Option<String> = match tokio::time::timeout(DESCRIBE_TIMEOUT, call).await {
                    Ok(Ok(reply)) => reply.body().deserialize().ok(),
                    _ => None,
                };
                text.map(|t| (name.clone(), path, t))
            }
        }))
        .await;
        let mut found = Vec::new();
        for (bus_name, path, text) in answers.into_iter().flatten() {
            for info in parse_describe(&bus_name, &text) {
                if !found.iter().any(|d: &DbusAction| d.info.spec.id == info.spec.id) {
                    found.push(DbusAction { info, bus_name: bus_name.clone(), path: path.clone() });
                }
            }
        }
        *self.dbus.lock().expect("dbus") = (found, Some(Instant::now()));
    }
}

enum Route {
    Dbus { info: Box<ActionInfo>, bus_name: String, path: String },
    Provider { info: Box<ActionInfo>, events: mpsc::Sender<Event>, connection: u64 },
}

impl Route {
    fn info(&self) -> ActionInfo {
        match self {
            Route::Dbus { info, .. } | Route::Provider { info, .. } => (**info).clone(),
        }
    }
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

/// GApplication's object path for an application ID.
fn object_path(app_id: &str) -> String {
    let mut path = String::from("/");
    for c in app_id.chars() {
        path.push(match c {
            '.' => '/',
            c if c.is_ascii_alphanumeric() || c == '_' => c,
            _ => '_',
        });
    }
    path
}

pub fn valid_action_id(id: &str) -> bool {
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "_-".contains(c));
    id.len() <= 64 && id.split_once('.').is_some_and(|(app, verb)| ok(app) && verb.split('.').all(ok))
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

/// The caller's standing and the arguments' shape.
fn check_call(caller: &Caller, info: &ActionInfo, arguments: &Value, confirmed: bool, allow_untrusted: bool) -> Result<(), Error> {
    if !info.available {
        return Err(Error::new(ErrorKind::Unavailable, format!("{} isn't available now", info.spec.id)));
    }
    if !caller.trusted && !allow_untrusted && info.spec.side_effects != SideEffects::ReadOnly {
        return Err(Error::new(ErrorKind::PermissionDenied, format!("{} may only run read-only actions", caller.name)));
    }
    if info.spec.confirmation == Confirmation::Ask && !confirmed {
        return Err(Error::new(ErrorKind::ConfirmationRequired, format!("{} needs the person's go-ahead", info.spec.id)));
    }
    if arguments.to_string().len() > MAX_ARGUMENTS {
        return Err(invalid("the arguments are too large"));
    }
    let validator = jsonschema::validator_for(&info.spec.input_schema).map_err(|e| invalid(format!("{}'s schema doesn't compile: {e}", info.spec.id)))?;
    if let Err(e) = validator.validate(arguments) {
        let at = e.instance_path.to_string();
        return Err(invalid(if at.is_empty() { format!("arguments: {e}") } else { format!("arguments at {at}: {e}") }));
    }
    Ok(())
}

/// A control interface's `Describe` answer: MCP-style tools (`name`, `description`,
/// `inputSchema`, optional `annotations`). Without annotations an action may change
/// things and asks first.
fn parse_describe(bus_name: &str, text: &str) -> Vec<ActionInfo> {
    let tools = match serde_json::from_str::<Value>(text) {
        Ok(Value::Array(tools)) => tools,
        Ok(v) => v.get("tools").and_then(Value::as_array).cloned().unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    tools
        .into_iter()
        .filter_map(|tool| {
            let id = tool.get("name")?.as_str()?.to_string();
            if !valid_action_id(&id) {
                return None;
            }
            let hint = |key: &str| tool.pointer(&format!("/annotations/{key}")).and_then(Value::as_bool);
            let side_effects = match (hint("readOnlyHint"), hint("destructiveHint"), hint("openWorldHint")) {
                (Some(true), _, _) => SideEffects::ReadOnly,
                (_, Some(true), _) => SideEffects::Destructive,
                (_, _, Some(true)) => SideEffects::External,
                (Some(false), Some(false), _) => SideEffects::Writes,
                _ => SideEffects::Unknown,
            };
            let input_schema = tool
                .get("inputSchema")
                .filter(|s| s.is_object() && jsonschema::validator_for(s).is_ok())
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            let spec = ActionSpec {
                id,
                description: tool.get("description").and_then(Value::as_str).unwrap_or_default().chars().take(2000).collect(),
                input_schema,
                output_schema: tool.get("outputSchema").filter(|s| s.is_object()).cloned(),
                permissions: Vec::new(),
                confirmation: if side_effects == SideEffects::ReadOnly { Confirmation::Never } else { Confirmation::Ask },
                side_effects,
                timeout_ms: None,
            };
            Some(ActionInfo { spec, owner: bus_name.to_string(), source: "dbus".into(), available: true })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caller(trusted: bool) -> Caller {
        Caller { name: if trusted { "nyx-assistant".into() } else { "untrusted:x".into() }, trusted, connection: 1 }
    }

    fn spec(id: &str, side_effects: SideEffects, confirmation: Confirmation) -> ActionInfo {
        let spec = ActionSpec {
            id: id.into(),
            description: "test".into(),
            input_schema: json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": false}),
            output_schema: None,
            permissions: vec![],
            confirmation,
            side_effects,
            timeout_ms: None,
        };
        ActionInfo { spec, owner: "nyx-notes".into(), source: "connection".into(), available: true }
    }

    #[test]
    fn object_paths_follow_gapplication() {
        assert_eq!(object_path("io.github.design_nexus.nyx.Chat"), "/io/github/design_nexus/nyx/Chat");
        assert_eq!(object_path("io.github.design_nexus.nyx.my-app"), "/io/github/design_nexus/nyx/my_app");
    }

    #[test]
    fn action_ids() {
        assert!(valid_action_id("notes.create"));
        assert!(valid_action_id("files.trash.restore"));
        for bad in ["notes", "Notes.create", ".create", "notes.", "notes..x", "notes.cre ate"] {
            assert!(!valid_action_id(bad), "{bad}");
        }
    }

    #[test]
    fn describe_without_annotations_asks_first() {
        let text = r#"[{"name":"chat.list","description":"Recent conversations.","inputSchema":{"type":"object","properties":{}}},
            {"name":"chat.read","description":"Read one.","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":true}},
            {"name":"chat.delete","description":"Delete.","annotations":{"destructiveHint":true}},
            {"name":"Bad Name","description":"skipped"}]"#;
        let actions = parse_describe("io.github.design_nexus.nyx.Chat", text);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].spec.side_effects, SideEffects::Unknown);
        assert_eq!(actions[0].spec.confirmation, Confirmation::Ask);
        assert_eq!(actions[1].spec.side_effects, SideEffects::ReadOnly);
        assert_eq!(actions[1].spec.confirmation, Confirmation::Never);
        assert_eq!(actions[2].spec.side_effects, SideEffects::Destructive);
        assert_eq!(actions[2].owner, "io.github.design_nexus.nyx.Chat");
    }

    #[test]
    fn calls_are_checked() {
        let read = spec("notes.find", SideEffects::ReadOnly, Confirmation::Never);
        let write = spec("notes.create", SideEffects::Writes, Confirmation::Ask);
        let ok = json!({"text": "milk"});
        check_call(&caller(true), &read, &ok, false, false).unwrap();
        check_call(&caller(false), &read, &ok, false, false).unwrap();
        assert_eq!(check_call(&caller(false), &write, &ok, true, false).unwrap_err().kind, ErrorKind::PermissionDenied);
        assert_eq!(check_call(&caller(true), &write, &ok, false, false).unwrap_err().kind, ErrorKind::ConfirmationRequired);
        check_call(&caller(true), &write, &ok, true, false).unwrap();
        let bad = check_call(&caller(true), &read, &json!({"text": 3}), false, false).unwrap_err();
        assert_eq!(bad.kind, ErrorKind::InvalidRequest);
        assert!(bad.message.contains("/text"), "{}", bad.message);
        assert!(check_call(&caller(true), &read, &json!({"text": "a", "extra": 1}), false, false).is_err());
    }

    #[test]
    fn specs_are_checked() {
        let mut s = spec("notes.create", SideEffects::Writes, Confirmation::Ask).spec;
        check_spec(&s).unwrap();
        s.input_schema = json!({"type": 12});
        assert!(check_spec(&s).is_err());
        s.input_schema = json!({"type": "object"});
        s.description = " ".into();
        assert!(check_spec(&s).is_err());
    }
}

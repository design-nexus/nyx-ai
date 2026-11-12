//! `nyx-ai`: Nyx AI Core from the command line. `--json` prints what the daemon
//! returned, for scripts and Nyx Settings.

use nyx_ai_client::proto::{ChatRequest, EmbedInput, Event, Message, Priority, Reply, Role};
use nyx_ai_client::{Client, Item};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::process::ExitCode;

const USAGE: &str = "Usage: nyx-ai COMMAND [--json]

  status                       the worker, memory and loaded models
  models                       installed models
  info MODEL                   one model in detail
  pull ORG/REPO:QUANT          download from Hugging Face (or update to its newest revision)
  updates                      check Hugging Face for newer revisions now
  downloads                    downloads queued, running and finished
  cancel-pull MODEL            stop a download
  rm MODEL                     delete a model from the shared cache
  load MODEL [--keep]          load now (--keep: until unload)
  unload MODEL                 unload unless something else uses it
  settings MODEL|* [KEY=VALUE…] show or change load-time settings (empty value removes)
  service [max_loaded=N] [idle=SECONDS] [updates=auto|notify|off] [battery_saver=on|off] [add_dir=FOLDER]
          [openai=on|off]      show or change how many models stay loaded, for how long,
                               whether models update themselves, whether they run on
                               the CPU on battery, and the endpoint for coding tools
  default ROLE [MODEL]         show or set the default model for a role (chat, assistant,
                               embedding, coding); \"\" clears it
  chat MODEL PROMPT…           ask a model (--think to show its reasoning)
  embed MODEL INPUT…           embed texts, image:FILE and audio:FILE (with a projector)
  capabilities                 what this Nyx AI Core supports
  actions                      actions apps offer (running apps and registered providers)
  act ACTION [JSON] [--yes]    run an action with JSON arguments (--yes: you agree to it,
                               for actions that ask first)";

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn print_json(value: &impl serde::Serialize) {
    println!("{}", serde_json::to_string_pretty(value).unwrap_or_default());
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    // SAFETY: restoring the default SIGPIPE before any other thread exists, so a closed
    // pipe (`nyx-ai models | head`) ends the program quietly instead of panicking.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let json = take_flag(&mut args, "--json");
    let keep = take_flag(&mut args, "--keep");
    let think = take_flag(&mut args, "--think");
    let yes = take_flag(&mut args, "--yes");
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let client = Client::new("nyx-ai");
    match run(&client, &args, Flags { json, keep, think, yes }).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if json {
                print_json(&json!({"error": {"kind": e.kind, "message": e.message}}));
            } else {
                eprintln!("nyx-ai: {}", e.message);
            }
            ExitCode::FAILURE
        }
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

fn need<'a>(args: &'a [String], i: usize, what: &str) -> Result<&'a str, nyx_ai_client::proto::Error> {
    args.get(i).map(String::as_str).ok_or_else(|| {
        nyx_ai_client::proto::Error::new(nyx_ai_client::proto::ErrorKind::InvalidRequest, format!("{what} is missing; see nyx-ai --help"))
    })
}

struct Flags {
    json: bool,
    keep: bool,
    think: bool,
    yes: bool,
}

async fn run(client: &Client, args: &[String], flags: Flags) -> nyx_ai_client::Result<()> {
    let Flags { json, keep, think, yes } = flags;
    match args[0].as_str() {
        "status" => {
            let worker = client.worker_status().await?;
            let res = client.resource_status().await?;
            if json {
                print_json(&json!({"worker": worker, "resources": res}));
                return Ok(());
            }
            println!("Worker: {:?}{}", worker.state, worker.release.map(|r| format!(" ({r})")).unwrap_or_default());
            if let Some(e) = &worker.last_error {
                println!("Last error: {e}");
            }
            println!(
                "Memory: {} GPU, {} RAM ({} free), budget {}; backends {}",
                gb(res.vram_total),
                gb(res.ram_total),
                gb(res.ram_available),
                gb(res.budget),
                res.gpu_backends.join(", ")
            );
            let idle = if res.idle_unload_seconds < 0 { "never".to_string() } else { format!("after {} s idle", res.idle_unload_seconds) };
            println!("Up to {} loaded; unload {idle}; {} waiting", res.max_loaded, res.queued);
            for m in &res.loaded {
                let pins = if m.pinned_by.is_empty() { String::new() } else { format!(", kept by {}", m.pinned_by.join(", ")) };
                println!("  {} {} (~{}, {}/{} busy{pins})", m.state, m.id, gb(m.estimate), m.active, m.slots);
            }
            let m = &worker.metrics;
            println!(
                "Requests {} ({} failed, {} cancelled); loads {} (avg {} ms); evictions {}; first token avg {} ms; {} tokens/s avg",
                m.requests, m.failed, m.cancelled, m.loads, m.load_ms_avg, m.evictions, m.first_token_ms_avg, m.tokens_per_second_avg
            );
        }
        "models" => {
            let models = client.list_models().await?;
            if json {
                print_json(&models);
                return Ok(());
            }
            for m in models {
                let caps: Vec<String> = m.capabilities.iter().map(|c| format!("{c:?}").to_lowercase()).collect();
                let update = match m.update_size {
                    Some(size) if m.update_available => format!("  [update: {}]", gb(size)),
                    _ => String::new(),
                };
                println!(
                    "{:<10} {:<60} {:>8}  {}{}{update}",
                    format!("{:?}", m.state).to_lowercase(),
                    m.id,
                    gb(m.size),
                    caps.join(","),
                    m.error.map(|e| format!("  ({e})")).unwrap_or_default()
                );
            }
        }
        "info" => {
            let detail = client.model_info(need(args, 1, "MODEL")?).await?;
            if json {
                print_json(&detail);
            } else {
                let mut value = serde_json::to_value(&detail).unwrap_or_default();
                if let Some(t) = value.get_mut("template") {
                    *t = json!(format!("{} characters", t.as_str().map(str::len).unwrap_or(0)));
                }
                print_json(&value);
            }
        }
        "pull" => {
            let model = need(args, 1, "ORG/REPO:QUANT")?;
            let mut stream = client.download_model(model).await?;
            let tty = std::io::stderr().is_terminal();
            let mut shown: Option<(String, u64)> = None;
            while let Some(item) = stream.next().await {
                match item? {
                    Item::Event(Event::Progress { status, completed, total }) if json => {
                        println!("{}", json!({"status": status, "completed": completed, "total": total}));
                    }
                    Item::Event(Event::Progress { status, completed, total }) => {
                        let percent = (completed * 100).checked_div(total).unwrap_or(0);
                        let line = if total > 0 { format!("{status}: {} of {} ({percent}%)", gb(completed), gb(total)) } else { format!("{status}…") };
                        if tty {
                            eprint!("\r{line}   ");
                        } else if shown.as_ref().is_none_or(|(s, p)| *s != status || p / 10 != percent / 10) {
                            eprintln!("{line}");
                            shown = Some((status, percent));
                        }
                    }
                    Item::Done(_) => {
                        if json {
                            print_json(&json!({"ok": true}));
                        } else {
                            eprintln!("{}Downloaded {model}.", if tty { "\r\x1b[K" } else { "" });
                        }
                    }
                    _ => {}
                }
            }
        }
        "updates" => {
            let models = client.check_updates().await?;
            if json {
                print_json(&models);
            } else if models.is_empty() {
                println!("Every cached model is on its newest revision.");
            } else {
                for m in models {
                    println!("{}  {} to download", m.id, gb(m.update_size.unwrap_or(0)));
                }
                let mode = client.service_settings().await?.model_updates.unwrap_or_default();
                println!("{}", if mode == "auto" { "They download when the computer is idle and on mains power." } else { "Run nyx-ai pull MODEL to update." });
            }
        }
        "downloads" => {
            let list = client.downloads().await?;
            if json {
                print_json(&list);
            } else if list.is_empty() {
                println!("No downloads.");
            } else {
                for d in list {
                    let state = format!("{:?}", d.state).to_lowercase();
                    println!("{state:<10} {}  {} of {}{}", d.model, gb(d.completed), gb(d.total), d.error.map(|e| format!("  ({e})")).unwrap_or_default());
                }
            }
        }
        "cancel-pull" => client.cancel_download(need(args, 1, "MODEL")?).await?,
        "rm" => client.remove_model(need(args, 1, "MODEL")?).await?,
        "load" => client.load_model(need(args, 1, "MODEL")?, keep).await?,
        "unload" => {
            let unloaded = client.unload_model(need(args, 1, "MODEL")?).await?;
            if json {
                print_json(&json!({"unloaded": unloaded}));
            } else if !unloaded {
                println!("Still loaded: another application is using it.");
            }
        }
        "settings" => {
            let model = need(args, 1, "MODEL or *")?;
            let changes: BTreeMap<String, String> =
                args[2..].iter().filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
            let s = if changes.is_empty() { client.model_settings(model).await? } else { client.configure_model(model, changes).await? };
            if json {
                print_json(&s);
            } else {
                for (k, v) in &s.effective {
                    let own = if s.values.contains_key(k) { "" } else { "  (default)" };
                    println!("{k} = {v}{own}");
                }
                if s.pending {
                    println!("Applies when the model is next idle.");
                }
            }
        }
        "service" => {
            let mut max = None;
            let mut idle = None;
            let mut updates = None;
            let mut saver = None;
            let mut dirs: Option<Vec<String>> = None;
            let mut openai = None;
            for kv in &args[1..] {
                match kv.split_once('=') {
                    Some(("max_loaded", v)) => max = v.parse().ok(),
                    Some(("idle", v)) | Some(("idle_unload_seconds", v)) => idle = v.parse().ok(),
                    Some(("updates", v)) | Some(("model_updates", v)) => updates = Some(v.to_string()),
                    Some(("battery_saver", v)) => saver = Some(matches!(v, "on" | "true" | "yes" | "1")),
                    Some(("add_dir", v)) | Some(("add_model_dir", v)) => dirs.get_or_insert_with(Vec::new).push(v.to_string()),
                    Some(("openai", v)) | Some(("openai_endpoint", v)) => openai = Some(matches!(v, "on" | "true" | "yes" | "1")),
                    _ => {}
                }
            }
            let s = if max.is_none() && idle.is_none() && updates.is_none() && saver.is_none() && dirs.is_none() && openai.is_none() {
                client.service_settings().await?
            } else {
                client
                    .configure_service_with(nyx_ai_client::proto::Op::ConfigureService {
                        max_loaded: max,
                        idle_unload_seconds: idle,
                        model_updates: updates,
                        defaults: None,
                        battery_saver: saver,
                        add_model_dirs: dirs,
                        openai,
                    })
                    .await?
            };
            if json {
                print_json(&s);
            } else {
                println!("max_loaded = {}\nidle_unload_seconds = {}\nmodel_dirs = {}", s.max_loaded, s.idle_unload_seconds, s.model_dirs.join(", "));
                println!("model_updates = {}", s.model_updates.as_deref().unwrap_or("auto"));
                let now = match (s.on_battery, s.cpu_only) {
                    (_, true) => "on battery: models run on the CPU",
                    (true, false) => "on battery",
                    (false, false) => "on mains power",
                };
                println!("battery_saver = {}  ({now})", if s.battery_saver { "on" } else { "off" });
                for (role, model) in &s.defaults {
                    println!("default.{role} = {model}");
                }
                match &s.openai_endpoint {
                    Some(addr) => println!("openai_endpoint = {addr} (base URL http://{addr}/v1)"),
                    None => println!("openai_endpoint = off  (nyx-ai service openai=on turns it on)"),
                }
                if let Some(file) = &s.openai_key_file {
                    println!("openai_key_file = {file}  (the API key coding tools send)");
                }
            }
        }
        "default" => {
            let role = need(args, 1, "ROLE")?.to_string();
            let s = match args.get(2) {
                None => client.service_settings().await?,
                Some(model) => {
                    client
                        .configure_service_with(nyx_ai_client::proto::Op::ConfigureService {
                            max_loaded: None,
                            idle_unload_seconds: None,
                            model_updates: None,
                            defaults: Some([(role.clone(), model.clone())].into()),
                            battery_saver: None,
                            add_model_dirs: None,
                            openai: None,
                        })
                        .await?
                }
            };
            let model = s.defaults.get(&role).cloned().unwrap_or_default();
            if json {
                print_json(&json!({"role": role, "model": model}));
            } else {
                println!("{}", if model.is_empty() { "none".to_string() } else { model });
            }
        }
        "chat" => {
            let model = need(args, 1, "MODEL")?;
            let prompt = args[2..].join(" ");
            need(args, 2, "PROMPT")?;
            let request = ChatRequest {
                model: model.into(),
                messages: vec![Message::text(Role::User, prompt)],
                thinking: think.then_some(true),
                priority: Priority::Interactive,
                ..Default::default()
            };
            let mut stream = client.chat(request).await?;
            let mut out = std::io::stdout();
            let mut reasoning = false;
            while let Some(item) = stream.next().await {
                match item? {
                    Item::Event(Event::Delta { content, reasoning: r, .. }) if !json => {
                        if !r.is_empty() && think {
                            if !reasoning {
                                let _ = write!(out, "\x1b[2m");
                                reasoning = true;
                            }
                            let _ = write!(out, "{r}");
                        }
                        if !content.is_empty() {
                            if reasoning {
                                let _ = writeln!(out, "\x1b[0m");
                                reasoning = false;
                            }
                            let _ = write!(out, "{content}");
                        }
                        let _ = out.flush();
                    }
                    Item::Event(Event::ModelLoading { model }) if !json => eprintln!("Loading {model}…"),
                    Item::Done(Reply::Finished(f)) => {
                        if json {
                            print_json(&f);
                        } else {
                            println!();
                            eprintln!(
                                "[{} tokens, {} tokens/s, first token {} ms, {}]",
                                f.usage.completion_tokens, f.timings.tokens_per_second, f.timings.first_token_ms, f.finish_reason
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        "embed" => {
            let model = need(args, 1, "MODEL")?;
            need(args, 2, "TEXT")?;
            let mut inputs = Vec::new();
            for arg in &args[2..] {
                let media = |path: &str| {
                    use base64::Engine;
                    std::fs::read(path)
                        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
                        .map_err(|e| nyx_ai_client::proto::Error::new(nyx_ai_client::proto::ErrorKind::InvalidRequest, format!("{path}: {e}")))
                };
                inputs.push(if let Some(path) = arg.strip_prefix("image:") {
                    EmbedInput::Image { image: media(path)? }
                } else if let Some(path) = arg.strip_prefix("audio:") {
                    EmbedInput::Audio { audio: media(path)? }
                } else {
                    EmbedInput::Text(arg.clone())
                });
            }
            let e = client.embed_inputs(model, inputs, Priority::Normal).await?;
            if json {
                print_json(&e);
            } else {
                println!("{} vectors of {} dimensions from {} ({})", e.vectors.len(), e.dimensions, e.identity.id, e.identity.file);
                for (text, v) in args[2..].iter().zip(&e.vectors) {
                    let head: Vec<String> = v.iter().take(4).map(|x| format!("{x:.4}")).collect();
                    println!("  {text}: [{}, …]", head.join(", "));
                }
            }
        }
        "actions" => {
            let actions = client.list_actions().await?;
            if json {
                print_json(&actions);
            } else if actions.is_empty() {
                println!("No app offers actions right now.");
            } else {
                for a in &actions {
                    let ask = if a.spec.confirmation == nyx_ai_client::proto::Confirmation::Ask { ", asks first" } else { "" };
                    let effects = serde_json::to_value(a.spec.side_effects).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
                    println!("{}  ({}; {effects}{ask})\n    {}", a.spec.id, a.owner, a.spec.description);
                }
            }
        }
        "act" => {
            let action = need(args, 1, "ACTION")?;
            let arguments = match args.get(2) {
                Some(text) => serde_json::from_str(text).map_err(|e| {
                    nyx_ai_client::proto::Error::new(nyx_ai_client::proto::ErrorKind::InvalidRequest, format!("the arguments aren't JSON: {e}"))
                })?,
                None => json!({}),
            };
            let result = client.invoke_action(action, arguments, yes).await?;
            print_json(&result);
        }
        "capabilities" => {
            let c = client.capabilities().await?;
            if json {
                print_json(&c);
            } else {
                let v: Value = serde_json::to_value(&c).unwrap_or_default();
                print_json(&v);
            }
        }
        other => {
            return Err(nyx_ai_client::proto::Error::new(nyx_ai_client::proto::ErrorKind::InvalidRequest, format!("no command {other}; see nyx-ai --help")));
        }
    }
    Ok(())
}

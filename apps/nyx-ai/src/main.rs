//! `nyx-ai`: Nyx AI Core from the command line. `--json` prints what the daemon
//! returned, for scripts and Nyx Settings.

use nyx_ai_client::proto::{ChatRequest, Event, Message, Priority, Reply, Role};
use nyx_ai_client::{Client, Item};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::ExitCode;

const USAGE: &str = "Usage: nyx-ai COMMAND [--json]

  status                       the worker, memory and loaded models
  models                       installed models
  info MODEL                   one model in detail
  pull ORG/REPO:QUANT          download from Hugging Face (or update to its newest revision)
  updates                      check Hugging Face for newer revisions now
  cancel-pull MODEL            stop a download
  rm MODEL                     delete a model from the shared cache
  load MODEL [--keep]          load now (--keep: until unload)
  unload MODEL                 unload unless something else uses it
  settings MODEL|* [KEY=VALUE…] show or change load-time settings (empty value removes)
  service [max_loaded=N] [idle=SECONDS]
                               show or change how many models stay loaded and for how long
  chat MODEL PROMPT…           ask a model (--think to show its reasoning)
  embed MODEL TEXT…            embed texts
  capabilities                 what this Nyx AI Core supports";

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn print_json(value: &impl serde::Serialize) {
    println!("{}", serde_json::to_string_pretty(value).unwrap_or_default());
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let json = take_flag(&mut args, "--json");
    let keep = take_flag(&mut args, "--keep");
    let think = take_flag(&mut args, "--think");
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let client = Client::new("nyx-ai");
    match run(&client, &args, json, keep, think).await {
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

async fn run(client: &Client, args: &[String], json: bool, keep: bool, think: bool) -> nyx_ai_client::Result<()> {
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
            while let Some(item) = stream.next().await {
                match item? {
                    Item::Event(Event::Progress { status, completed, total }) if !json => {
                        if total > 0 {
                            eprint!("\r{status}: {} of {} ({}%)   ", gb(completed), gb(total), completed * 100 / total);
                        } else {
                            eprint!("\r{status}…   ");
                        }
                    }
                    Item::Done(_) => {
                        if json {
                            print_json(&json!({"ok": true}));
                        } else {
                            eprintln!("\rDownloaded {model}.                         ");
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
            for kv in &args[1..] {
                match kv.split_once('=') {
                    Some(("max_loaded", v)) => max = v.parse().ok(),
                    Some(("idle", v)) | Some(("idle_unload_seconds", v)) => idle = v.parse().ok(),
                    _ => {}
                }
            }
            let s = if max.is_none() && idle.is_none() { client.service_settings().await? } else { client.configure_service(max, idle).await? };
            if json {
                print_json(&s);
            } else {
                println!("max_loaded = {}\nidle_unload_seconds = {}\nmodel_dirs = {}", s.max_loaded, s.idle_unload_seconds, s.model_dirs.join(", "));
                println!("model_updates = {}", s.model_updates.as_deref().unwrap_or("auto"));
                match &s.openai_endpoint {
                    Some(addr) => println!("openai_endpoint = {addr} (base URL http://{addr}/v1)"),
                    None => println!("openai_endpoint = off"),
                }
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
            let e = client.embed(model, args[2..].to_vec(), Priority::Normal).await?;
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

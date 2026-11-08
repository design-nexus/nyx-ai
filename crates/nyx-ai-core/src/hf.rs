//! Hugging Face names and the hub cache layout (`models--org--repo/{blobs,
//! snapshots,refs}`), shared by downloads and the registry.

use std::path::{Component, Path, PathBuf};

/// The kind of companion a file is (`mmproj`, `mtp`, `dflash`, `eagle3`,
/// `dspark`), if it isn't a model of its own.
pub fn sidecar_kind(path: &str) -> Option<&'static str> {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    let stem = name.strip_suffix(".gguf").unwrap_or(&name);
    let tokens: Vec<&str> = stem.split(['-', '_', '.']).collect();
    if name.contains("mmproj") {
        Some("mmproj")
    } else if name.contains("dflash") || name.contains("drafter") {
        Some("dflash")
    } else if tokens.contains(&"mtp") {
        Some("mtp")
    } else if tokens.contains(&"eagle3") {
        Some("eagle3")
    } else if tokens.contains(&"dspark") {
        Some("dspark")
    } else {
        None
    }
}

pub const SIDECARS: &[&str] = &["mmproj", "mtp", "dflash", "eagle3", "dspark"];

/// Projectors, MTP heads and speculative drafters are companion files, not
/// models to chat with.
pub fn auxiliary_gguf(path: &str) -> bool {
    sidecar_kind(path).is_some()
}

pub fn valid_repo(repo: &str) -> bool {
    let parts: Vec<_> = repo.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        })
}

pub fn valid_file(path: &str) -> bool {
    !path.is_empty() && Path::new(path).components().all(|part| matches!(part, Component::Normal(_)))
}

pub fn valid_oid(oid: &str) -> bool {
    oid.len() == 64 && oid.bytes().all(|c| c.is_ascii_hexdigit())
}

pub fn repo_dir(cache: &Path, repo: &str) -> PathBuf {
    cache.join(format!("models--{}", repo.replace('/', "--")))
}

/// The revision a cached file belongs to, from `…/snapshots/<rev>/…`.
pub fn revision_of(path: &Path) -> Option<String> {
    let mut parts = path.components().map(|c| c.as_os_str().to_string_lossy());
    parts.by_ref().find(|p| p == "snapshots")?;
    parts.next().map(|p| p.into_owned()).filter(|p| !p.is_empty())
}

/// A cached quant can exist even when it contains only a speculative drafter.
pub fn cached_auxiliary_model(cache: &Path, model: &str) -> bool {
    let Some((repo, quant)) = model.rsplit_once(':') else { return false };
    if !valid_repo(repo) {
        return false;
    }
    let base = repo_dir(cache, repo);
    let Ok(revision) = std::fs::read_to_string(base.join("refs/main")) else { return false };
    let mut dirs = vec![base.join("snapshots").join(revision.trim())];
    let (mut auxiliary, mut main) = (false, false);
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !name.to_ascii_lowercase().ends_with(".gguf") || quant_tag(&name) != quant {
                continue;
            }
            if auxiliary_gguf(&name) {
                auxiliary = true;
            } else {
                main = true;
            }
        }
    }
    auxiliary && !main
}

/// The tag llama.cpp's router gives a cached file: its last `-` part, after any
/// `.` (`gemma-4-E4B_q4_0-it.gguf` is `IT`, `codec.decoder.f16.gguf` is `F16`).
pub fn router_tag(name: &str) -> String {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let last = stem.rsplit('-').next().unwrap_or(stem);
    last.rsplit('.').next().unwrap_or(last).to_ascii_uppercase()
}

/// The main file of a cached `org/repo:TAG`, as the router would load it.
pub fn cached_file(cache: &Path, model: &str) -> Option<PathBuf> {
    let (repo, tag) = model.rsplit_once(':')?;
    if !valid_repo(repo) {
        return None;
    }
    let base = repo_dir(cache, repo);
    let snapshot = match std::fs::read_to_string(base.join("refs/main")) {
        Ok(rev) => base.join("snapshots").join(rev.trim()),
        Err(_) => std::fs::read_dir(base.join("snapshots")).ok()?.flatten().map(|e| e.path()).next()?,
    };
    let mut files = Vec::new();
    let mut dirs = vec![snapshot.clone()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if !name.to_ascii_lowercase().ends_with(".gguf") || name.to_ascii_lowercase().contains("mmproj") {
                continue;
            }
            if let Some(pos) = name.find("-of-")
                && !name[..pos].ends_with("-00001")
            {
                continue;
            }
            files.push((path.strip_prefix(&snapshot).map(|p| p.display().to_string()).unwrap_or(name), path));
        }
    }
    let tag = tag.to_ascii_uppercase();
    let pick = |matches: &dyn Fn(&str) -> bool| files.iter().filter(|(rel, _)| matches(rel)).min_by_key(|(rel, _)| rel.len()).map(|(_, p)| p.clone());
    pick(&|rel| quant_tag(rel) == tag && !auxiliary_gguf(rel))
        .or_else(|| pick(&|rel| router_tag(rel.rsplit('/').next().unwrap_or(rel)) == tag))
        .or_else(|| {
            let main: Vec<_> = files.iter().filter(|(rel, _)| !auxiliary_gguf(rel)).collect();
            (main.len() == 1).then(|| main[0].1.clone())
        })
}

/// The quantisation a file name carries: `Q4_K_M`, `IQ3_M`, `BF16`, `PQ2_0`, …
pub fn quant_tag(filename: &str) -> String {
    let base = filename.rsplit('/').next().unwrap_or(filename);
    let upper = base.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < bytes.len() {
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }
        if let Some(len) = match_quant(&upper[i..]) {
            best = Some((i, len));
            i += len;
        } else {
            i += 1;
        }
    }
    match best {
        Some((start, len)) => upper[start..start + len].to_string(),
        None => base.trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string(),
    }
}

fn match_quant(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if s.starts_with("BF16") {
        return Some(4);
    }
    if s.starts_with("MXFP") {
        return take_body(s, 4, 1);
    }
    if s.starts_with("PQ") {
        return take_body(s, 2, 2);
    }
    if s.starts_with("PTQ") {
        return take_body(s, 3, 2);
    }
    if s.starts_with("IQ") {
        return take_body(s, 2, 2);
    }
    if s.starts_with("F16") || s.starts_with("F32") {
        return Some(3);
    }
    if b.first() == Some(&b'Q') {
        return take_body(s, 1, 2);
    }
    None
}

/// Digits after the prefix, then up to `segments` `_XX` parts (Q4_K_M, IQ4_XS).
fn take_body(s: &str, prefix: usize, segments: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = prefix;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == prefix {
        return None;
    }
    for _ in 0..segments {
        if i < b.len() && b[i] == b'_' {
            let mut k = i + 1;
            while k < b.len() && b[k].is_ascii_alphanumeric() {
                k += 1;
            }
            if k == i + 1 {
                break;
            }
            i = k;
        }
    }
    Some(i)
}

/// A Hugging Face token: AI Core's own, Chat's, then the shared one.
pub fn token(config_home: &Path) -> Option<String> {
    let shared = std::env::var_os("HF_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| crate::paths::home().join(".cache"))
                .join("huggingface")
        })
        .join("token");
    if let Some(t) = std::env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty()) {
        return Some(t.trim().into());
    }
    [config_home.join("nyx-ai/huggingface.token"), config_home.join("nyx-chat/huggingface.token"), shared]
        .into_iter()
        .find_map(|p| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
}

/// `HF_ENDPOINT`, else huggingface.co.
pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|e| !e.is_empty()).unwrap_or_else(|| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

pub fn encode_path(path: &str) -> String {
    path.split('/').map(crate::worker::encode_query).collect::<Vec<_>>().join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_tags() {
        assert_eq!(quant_tag("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_tag("model-IQ3_M.gguf"), "IQ3_M");
        assert_eq!(quant_tag("foo-BF16.gguf"), "BF16");
        assert_eq!(quant_tag("sub/bar-IQ4_XS.gguf"), "IQ4_XS");
        assert_eq!(quant_tag("Ternary-Bonsai-2-27B-PQ2_0.gguf"), "PQ2_0");
        assert_eq!(quant_tag("Ternary-Bonsai-2-27B-PTQ1_0.gguf"), "PTQ1_0");
        assert!(auxiliary_gguf("Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf"));
        assert_eq!(sidecar_kind("Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf"), Some("mtp"));
        assert_eq!(sidecar_kind("Swift-IQ3_S-mtp.gguf"), Some("mtp"));
        assert_eq!(sidecar_kind("gemma-4-E4B-it-mmproj.gguf"), Some("mmproj"));
        assert_eq!(sidecar_kind("smtpserver-Q4_0.gguf"), None);
        assert_eq!(quant_tag("Q8_0/big-Q8_0-00001-of-00002.gguf"), "Q8_0");
    }

    #[test]
    fn names_and_paths() {
        assert!(valid_repo("prism-ml/Ternary-Bonsai-2-9B-gguf"));
        assert!(!valid_repo("../x") && !valid_repo("a/b/c") && !valid_repo("a/.."));
        assert!(valid_file("sub/x.gguf") && !valid_file("../x.gguf") && !valid_file("/etc/x"));
        let p = Path::new("/c/models--o--r/snapshots/abc123/x.gguf");
        assert_eq!(revision_of(p).as_deref(), Some("abc123"));
        assert_eq!(encode_path("o/r"), "o/r");
    }

    #[test]
    fn finds_cached_files_by_either_tag() {
        let dir = tempfile::tempdir().unwrap();
        let snap = |repo: &str, files: &[&str]| {
            let base = repo_dir(dir.path(), repo);
            std::fs::create_dir_all(base.join("refs")).unwrap();
            std::fs::write(base.join("refs/main"), "rev").unwrap();
            for f in files {
                let p = base.join("snapshots/rev").join(f);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(p, b"x").unwrap();
            }
        };
        snap("g/gemma", &["gemma-4-E4B-it-mmproj.gguf", "gemma-4-E4B_q4_0-it.gguf"]);
        snap("u/swift", &["Swift-IQ3_S.gguf", "Swift-IQ3_S-mtp.gguf"]);
        snap("s/split", &["Q8_0/big-Q8_0-00001-of-00002.gguf", "Q8_0/big-Q8_0-00002-of-00002.gguf"]);
        let name = |m: &str| cached_file(dir.path(), m).map(|p| p.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(name("g/gemma:IT").as_deref(), Some("gemma-4-E4B_q4_0-it.gguf"));
        assert_eq!(name("u/swift:IQ3_S").as_deref(), Some("Swift-IQ3_S.gguf"));
        assert_eq!(name("u/swift:MTP").as_deref(), Some("Swift-IQ3_S-mtp.gguf"));
        assert_eq!(name("s/split:Q8_0").as_deref(), Some("big-Q8_0-00001-of-00002.gguf"));
        assert_eq!(name("s/none:Q8_0"), None);
    }
}

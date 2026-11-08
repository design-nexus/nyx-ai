//! Memory for models: what the machine has, what a model needs, and when to
//! back off. A sleeping discrete GPU is never woken to ask: NVIDIA memory is read
//! with `nvidia-smi` only while the card is awake, then remembered.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Memory {
    /// Dedicated GPU memory (0 when there's no discrete GPU we can read).
    pub vram: u64,
    pub ram: u64,
}

static KNOWN_VRAM: AtomicU64 = AtomicU64::new(0);

pub fn detect() -> Memory {
    let cached = KNOWN_VRAM.load(Ordering::Relaxed);
    let vram = amd_vram().or_else(nvidia_vram).or((cached > 0).then_some(cached)).or_else(nvidia_bar).unwrap_or(0);
    if vram > 0 {
        KNOWN_VRAM.store(vram, Ordering::Relaxed);
    }
    Memory { vram, ram: meminfo("MemTotal:").unwrap_or(0) }
}

/// Memory the loaded models may use: nearly all of the GPU, and part of RAM
/// for partial offload.
pub fn budget(mem: Memory) -> u64 {
    (mem.vram as f64 * 0.95 + mem.ram as f64 * 0.6) as u64
}

pub fn available_ram() -> u64 {
    meminfo("MemAvailable:").unwrap_or(0)
}

/// `/proc/pressure/memory` "some avg10": the share of the last 10 s some task
/// waited on memory.
pub fn memory_pressure() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/pressure/memory").ok()?;
    let line = text.lines().find(|l| l.starts_with("some "))?;
    line.split_whitespace().find_map(|f| f.strip_prefix("avg10=")).and_then(|v| v.parse().ok())
}

/// Too little memory left to keep idle models around.
pub fn under_pressure(total: u64, available: u64, psi: Option<f64>) -> bool {
    (total > 0 && available < total / 14) || psi.is_some_and(|p| p > 30.0)
}

fn meminfo(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
}

/// An NVIDIA card's memory window (BAR 1) from sysfs, which doesn't wake it.
/// With resizable BAR, as on current cards, it's the size of the memory.
fn nvidia_bar() -> Option<u64> {
    std::fs::read_dir("/sys/bus/pci/devices")
        .ok()?
        .flatten()
        .filter(|e| nvidia_gpu(&e.path()))
        .filter_map(|e| bar1_size(&std::fs::read_to_string(e.path().join("resource")).ok()?))
        .max()
}

fn nvidia_gpu(dev: &Path) -> bool {
    let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
    let class = std::fs::read_to_string(dev.join("class")).unwrap_or_default();
    vendor.trim() == "0x10de" && class.starts_with("0x03")
}

fn bar1_size(resource: &str) -> Option<u64> {
    let line = resource.lines().nth(1)?;
    let mut parts = line.split_whitespace().map(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok());
    let (start, end) = (parts.next()??, parts.next()??);
    let size = end.checked_sub(start)? + 1;
    // Small windows (256 MB without resizable BAR) say nothing about the memory.
    (2 << 30..=512 << 30).contains(&size).then_some(size)
}

fn amd_vram() -> Option<u64> {
    std::fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path().join("device/mem_info_vram_total")).ok()?.trim().parse::<u64>().ok())
        // An APU reports a small carve-out; only count real cards.
        .filter(|v| *v >= 2 << 30)
        .max()
}

fn nvidia_vram() -> Option<u64> {
    let awake = std::fs::read_dir("/sys/bus/pci/devices").ok()?.flatten().any(|e| {
        let dev = e.path();
        nvidia_gpu(&dev)
            && std::fs::read_to_string(dev.join("power/runtime_status")).map(|s| s.trim() != "suspended").unwrap_or(true)
    });
    if !awake {
        return None;
    }
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().parse::<u64>().ok()).max().map(|mib| mib * 1024 * 1024)
}

/// GPU backends the worker build carries, from its `libggml-*.so` files.
pub fn gpu_backends(worker_dir: Option<&Path>) -> Vec<String> {
    let mut found = Vec::new();
    if let Some(dir) = worker_dir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            for (lib, backend) in [("cuda", "cuda"), ("vulkan", "vulkan"), ("hip", "rocm"), ("sycl", "sycl"), ("metal", "metal")] {
                if name.starts_with(&format!("libggml-{lib}")) && !found.iter().any(|b| b == backend) {
                    found.push(backend.to_string());
                }
            }
        }
    }
    found.sort();
    found.push("cpu".into());
    found
}

/// Memory a model needs: weights, plus the KV cache for `ctx` tokens, plus
/// runtime overhead. KV per token grows more slowly than the weights (bigger
/// models add layers, not KV heads): about 130 KB per token for both an 8B and
/// a 27B grouped-query model at fp16, which the square root follows.
pub fn needed(file_size: u64, ctx: u64, kv_type: &str) -> u64 {
    let gb = file_size as f64 / 1e9;
    let ratio = match kv_type {
        "q8_0" => 0.53125,
        "q5_0" | "q5_1" => 0.34375,
        "q4_0" | "q4_1" | "iq4_nl" => 0.28125,
        "f32" => 2.0,
        _ => 1.0,
    };
    let kv = ctx as f64 * 45_000.0 * gb.max(0.5).sqrt() * ratio;
    file_size + kv as u64 + 500_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn needs_and_budget() {
        assert!(needed(4 * GB, 65536, "q4_0") < needed(4 * GB, 65536, "q8_0"));
        assert!(needed(4 * GB, 65536, "q8_0") < needed(4 * GB, 65536, "f16"));
        let mem = Memory { vram: 16 * GB, ram: 32 * GB };
        assert!(needed(12 * GB, 8192, "f16") < budget(mem));
        assert_eq!(budget(Memory { vram: 0, ram: 10 * GB }), 6 * GB);
    }

    #[test]
    fn reads_bar1() {
        let res = "0x00000000a3000000 0x00000000a3ffffff 0x0000000000040200\n0x0000006000000000 0x00000063ffffffff 0x000000000014220c\n";
        assert_eq!(bar1_size(res), Some(16 << 30));
        let small = "0x0 0x0 0x0\n0x00000000e0000000 0x00000000efffffff 0x0\n";
        assert_eq!(bar1_size(small), None);
    }

    #[test]
    fn pressure() {
        assert!(under_pressure(14 * GB, GB / 2, None));
        assert!(!under_pressure(14 * GB, 4 * GB, Some(1.0)));
        assert!(under_pressure(14 * GB, 4 * GB, Some(45.0)));
    }

    #[test]
    fn backends_from_libraries() {
        let dir = tempfile::tempdir().unwrap();
        for f in ["libggml-cuda.so", "libggml-cpu-haswell.so", "libggml-base.so"] {
            std::fs::write(dir.path().join(f), "").unwrap();
        }
        assert_eq!(gpu_backends(Some(dir.path())), ["cuda", "cpu"]);
        assert_eq!(gpu_backends(None), ["cpu"]);
    }
}

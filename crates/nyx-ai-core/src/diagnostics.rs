//! Counters and running averages. Nothing here holds prompt, reply or embedding
//! content.

use nyx_ai_proto::Metrics as Snapshot;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub failed: AtomicU64,
    pub cancelled: AtomicU64,
    pub active: AtomicU64,
    pub loads: AtomicU64,
    pub evictions: AtomicU64,
    pub idle_unloads: AtomicU64,
    pub worker_restarts: AtomicU64,
    pub embeddings: AtomicU64,
    averages: Mutex<Averages>,
}

#[derive(Default)]
struct Averages {
    load_ms: Ewma,
    first_token_ms: Ewma,
    tokens_per_second: Ewma,
}

#[derive(Default)]
struct Ewma(Option<f64>);

impl Ewma {
    fn add(&mut self, v: f64) {
        self.0 = Some(match self.0 {
            Some(old) => old * 0.8 + v * 0.2,
            None => v,
        });
    }

    fn get(&self) -> f64 {
        self.0.unwrap_or(0.0)
    }
}

impl Metrics {
    pub fn inc(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn loaded(&self, ms: u64) {
        Metrics::inc(&self.loads);
        self.averages.lock().expect("averages").load_ms.add(ms as f64);
    }

    pub fn generated(&self, first_token_ms: u64, tokens_per_second: f64) {
        let mut a = self.averages.lock().expect("averages");
        if first_token_ms > 0 {
            a.first_token_ms.add(first_token_ms as f64);
        }
        if tokens_per_second > 0.0 {
            a.tokens_per_second.add(tokens_per_second);
        }
    }

    pub fn snapshot(&self, queued: u64) -> Snapshot {
        let a = self.averages.lock().expect("averages");
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Snapshot {
            requests: get(&self.requests),
            failed: get(&self.failed),
            cancelled: get(&self.cancelled),
            active: get(&self.active),
            queued,
            loads: get(&self.loads),
            load_ms_avg: a.load_ms.get() as u64,
            evictions: get(&self.evictions),
            idle_unloads: get(&self.idle_unloads),
            worker_restarts: get(&self.worker_restarts),
            first_token_ms_avg: a.first_token_ms.get() as u64,
            tokens_per_second_avg: (a.tokens_per_second.get() * 10.0).round() / 10.0,
            embeddings: get(&self.embeddings),
        }
    }
}

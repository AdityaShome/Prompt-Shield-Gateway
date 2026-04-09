use hdrhistogram::Histogram;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct MetricsRegistry {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    requests_total: AtomicU64,
    blocked_total: AtomicU64,
    rewritten_total: AtomicU64,
    false_positive_total: AtomicU64,
    detection_total: AtomicU64,
    latency_ms: Mutex<Histogram<u64>>,
}

#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub blocked_total: u64,
    pub rewritten_total: u64,
    pub false_positive_total: u64,
    pub detection_rate: f64,
    pub blocked_percent: f64,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub latency_p99_ms: u64,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                requests_total: AtomicU64::new(0),
                blocked_total: AtomicU64::new(0),
                rewritten_total: AtomicU64::new(0),
                false_positive_total: AtomicU64::new(0),
                detection_total: AtomicU64::new(0),
                latency_ms: Mutex::new(Histogram::new(3).expect("valid histogram")),
            }),
        }
    }

    pub fn record(&self, latency_ms: u64, detected: bool, rewritten: bool, blocked: bool) {
        self.inner.requests_total.fetch_add(1, Ordering::Relaxed);
        if detected {
            self.inner.detection_total.fetch_add(1, Ordering::Relaxed);
        }
        if rewritten {
            self.inner.rewritten_total.fetch_add(1, Ordering::Relaxed);
        }
        if blocked {
            self.inner.blocked_total.fetch_add(1, Ordering::Relaxed);
        }
        let _ = self.inner.latency_ms.lock().expect("metrics lock").record(latency_ms);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let requests_total = self.inner.requests_total.load(Ordering::Relaxed);
        let blocked_total = self.inner.blocked_total.load(Ordering::Relaxed);
        let rewritten_total = self.inner.rewritten_total.load(Ordering::Relaxed);
        let false_positive_total = self.inner.false_positive_total.load(Ordering::Relaxed);
        let detection_total = self.inner.detection_total.load(Ordering::Relaxed);
        let latency = self.inner.latency_ms.lock().expect("metrics lock");

        MetricsSnapshot {
            requests_total,
            blocked_total,
            rewritten_total,
            false_positive_total,
            detection_rate: ratio(detection_total, requests_total),
            blocked_percent: ratio(blocked_total, requests_total),
            latency_p50_ms: latency.value_at_quantile(0.50),
            latency_p95_ms: latency.value_at_quantile(0.95),
            latency_p99_ms: latency.value_at_quantile(0.99),
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

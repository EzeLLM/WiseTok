//! Background RSS sampler.
//!
//! Spawns a thread that reads the current process's resident-set size
//! every `interval`, logs each sample at `info` level, tracks the peak,
//! and emits a warning when the RSS exceeds `warn_threshold`. The
//! sampler is stopped by dropping its [`RamMonitor`] handle.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Format a byte count for log output: `1234567` -> `"1.18 GB"`.
pub fn format_bytes(b: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if b >= GB {
        format!("{:.2} GB", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{:.2} MB", b as f64 / MB as f64)
    } else if b >= KB {
        format!("{:.2} KB", b as f64 / KB as f64)
    } else {
        format!("{} B", b)
    }
}

/// Read the current process's RSS in bytes. Returns 0 if the platform
/// doesn't support it (which would be unusual on Linux/macOS).
pub fn current_rss_bytes() -> u64 {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new().with_memory()),
    );
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::new().with_memory(),
    );
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Handle to a background RSS sampler. Drop it (or call [`stop`]) to
/// terminate the sampler thread.
pub struct RamMonitor {
    peak_rss: Arc<AtomicU64>,
    stop_flag: Arc<AtomicU64>, // 0 = run, 1 = stop
    handle: Option<JoinHandle<()>>,
}

impl RamMonitor {
    /// Start sampling RSS every `interval`. If `warn_threshold` is set
    /// and a sample exceeds it, emit a `log::warn!` once per crossing.
    /// `label` is included in the log lines for context (e.g. "aggregate"
    /// or "merge"); empty string is fine.
    pub fn start(interval: Duration, warn_threshold: Option<u64>, label: &str) -> Self {
        let peak_rss = Arc::new(AtomicU64::new(0));
        let stop_flag = Arc::new(AtomicU64::new(0));
        let label = label.to_string();

        let peak_clone = Arc::clone(&peak_rss);
        let stop_clone = Arc::clone(&stop_flag);
        let handle = thread::spawn(move || {
            let mut warned = false;
            let start = Instant::now();
            // Take an immediate sample so the peak is non-zero from t=0.
            let initial = current_rss_bytes();
            peak_clone.fetch_max(initial, Ordering::Relaxed);

            while stop_clone.load(Ordering::Relaxed) == 0 {
                let rss = current_rss_bytes();
                let prev_peak = peak_clone.load(Ordering::Relaxed);
                if rss > prev_peak {
                    peak_clone.store(rss, Ordering::Relaxed);
                }
                let elapsed = start.elapsed().as_secs();
                log::info!(
                    "[ram{}] t={}s rss={} (peak {})",
                    if label.is_empty() {
                        String::new()
                    } else {
                        format!(":{}", label)
                    },
                    elapsed,
                    format_bytes(rss),
                    format_bytes(peak_clone.load(Ordering::Relaxed)),
                );
                if let Some(limit) = warn_threshold {
                    if rss >= limit && !warned {
                        log::warn!(
                            "[ram{}] rss {} exceeded threshold {}; consider merge_mode=\"scan\"",
                            if label.is_empty() {
                                String::new()
                            } else {
                                format!(":{}", label)
                            },
                            format_bytes(rss),
                            format_bytes(limit),
                        );
                        warned = true;
                    }
                }
                // Sleep in small slices so stop is responsive.
                let mut remaining = interval;
                let slice = Duration::from_millis(200);
                while remaining > Duration::ZERO && stop_clone.load(Ordering::Relaxed) == 0 {
                    let s = if remaining < slice { remaining } else { slice };
                    thread::sleep(s);
                    remaining = remaining.saturating_sub(s);
                }
            }
        });

        Self {
            peak_rss,
            stop_flag,
            handle: Some(handle),
        }
    }

    /// Returns the largest RSS sample observed so far, in bytes.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_rss.load(Ordering::Relaxed)
    }

    /// Signal the sampler to stop and wait for the thread to exit.
    /// Idempotent — calling twice is harmless.
    pub fn stop(&mut self) {
        self.stop_flag.store(1, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            // Best-effort: if the sampler panicked we still want to surface
            // the peak we already collected.
            let _ = h.join();
        }
    }
}

impl Drop for RamMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Parse a human-readable byte size like "64GB", "1.5gb", "512mb",
/// "1024" (interpreted as bytes). Returns `None` on parse failure.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let lower = s.to_ascii_lowercase();
    let (num_part, mult) = if let Some(prefix) = lower.strip_suffix("gb") {
        (prefix, 1024u64 * 1024 * 1024)
    } else if let Some(prefix) = lower.strip_suffix("g") {
        (prefix, 1024u64 * 1024 * 1024)
    } else if let Some(prefix) = lower.strip_suffix("mb") {
        (prefix, 1024u64 * 1024)
    } else if let Some(prefix) = lower.strip_suffix("m") {
        (prefix, 1024u64 * 1024)
    } else if let Some(prefix) = lower.strip_suffix("kb") {
        (prefix, 1024u64)
    } else if let Some(prefix) = lower.strip_suffix("k") {
        (prefix, 1024u64)
    } else if let Some(prefix) = lower.strip_suffix("b") {
        (prefix, 1u64)
    } else {
        (lower.as_str(), 1u64)
    };
    let n: f64 = num_part.trim().parse().ok()?;
    if n.is_nan() || n.is_infinite() || n < 0.0 {
        return None;
    }
    Some((n * mult as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn parse_size_recognizes_units() {
        assert_eq!(parse_size("64GB"), Some(64 * 1024 * 1024 * 1024));
        assert_eq!(parse_size("64gb"), Some(64 * 1024 * 1024 * 1024));
        assert_eq!(parse_size("64 GB"), Some(64 * 1024 * 1024 * 1024));
        assert_eq!(
            parse_size("1.5GB"),
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(parse_size("512MB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("1024B"), Some(1024));
    }

    #[test]
    fn parse_size_rejects_garbage() {
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("xyz"), None);
        assert_eq!(parse_size("-1GB"), None);
    }

    #[test]
    fn current_rss_is_nonzero() {
        // We're a real process, our RSS must be > 0.
        let rss = current_rss_bytes();
        assert!(rss > 0, "expected non-zero RSS, got {}", rss);
    }

    #[test]
    fn monitor_starts_and_stops() {
        let mut mon = RamMonitor::start(Duration::from_millis(100), None, "test");
        std::thread::sleep(Duration::from_millis(250));
        let peak = mon.peak_bytes();
        assert!(peak > 0, "peak should be > 0 after a few samples");
        mon.stop();
        // Second stop is a no-op.
        mon.stop();
    }
}

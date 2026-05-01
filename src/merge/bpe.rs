use std::collections::HashMap as StdHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use rayon::prelude::*;

use super::heap::{MergeJob, ScanJob};
use super::word::Word;
use crate::Pair;

/// Strategy for the merge loop's pair-locating data structure.
///
/// **Pair selection, delta computation, and tie-breaking are identical
/// across modes.** All modes produce byte-identical merge tables on the
/// same input. The only difference is how the loop finds which words
/// contain the popped pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// Upstream rustbpe behavior: maintain `pair → AHashSet<word_idx>` for
    /// every pair, embedded in heap entries. Fast per-merge (no scan), but
    /// peak RAM is O(N × L) where N is unique words and L is average symbols
    /// per word. OOMs on large corpora — measured ~12 GB just for position
    /// sets at N=50M, L=10.
    Full,
    /// Drop the position map. Each merge does one linear pass over `words`
    /// to discover indices containing the winning pair. Memory is bounded
    /// by the structures we already need: `words`, `counts`, `pair_counts`,
    /// and a positionless heap.
    Scan,
    /// Pick `Scan` when unique-word count exceeds [`AUTO_SCAN_THRESHOLD`],
    /// `Full` otherwise. Resolved to a concrete mode before training begins.
    Auto,
}

/// Threshold for [`MergeMode::Auto`]: above this many unique words, switch
/// to [`MergeMode::Scan`]. The constant is empirical — for N below this,
/// Full's position sets are cheap enough that the per-merge scan cost in
/// Scan mode is the worse trade.
pub const AUTO_SCAN_THRESHOLD: usize = 1_000_000;

/// Resolve [`MergeMode::Auto`] to a concrete mode based on the unique-word
/// count. Pass-through for explicit `Full` / `Scan`.
pub fn resolve_mode(mode: MergeMode, num_unique_words: usize) -> MergeMode {
    match mode {
        MergeMode::Auto => {
            if num_unique_words > AUTO_SCAN_THRESHOLD {
                MergeMode::Scan
            } else {
                MergeMode::Full
            }
        }
        m => m,
    }
}

/// Compute initial pair frequencies and the pair → word-index inverted index
/// in parallel. Returns `(pair_counts, where_to_update)`. Used by Full mode.
#[inline]
pub(crate) fn count_pairs_parallel(
    words: &[Word],
    counts: &[i64],
) -> (AHashMap<Pair, i64>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i64> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

/// Compute initial pair frequencies in parallel — without the inverted
/// index. Used by Scan mode. Returns just `pair_counts`.
#[inline]
fn count_pairs_only_parallel(words: &[Word], counts: &[i64]) -> AHashMap<Pair, i64> {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local: AHashMap<Pair, i64> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local.entry((a, b)).or_default() += counts[i];
                }
            }
            local
        })
        .reduce(AHashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        })
}

/// Run the BPE merge loop. Dispatches to Full or Scan based on `mode`.
///
/// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
/// `counts`: same length as `words`, count per chunk.
/// `merges`: output map, populated with `(pair, new_id)` entries.
/// `mode`: must already be resolved (`Full` or `Scan`, never `Auto`).
pub(crate) fn train_core(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
    mode: MergeMode,
) {
    train_core_with_progress(words, counts, vocab_size, merges, mode, None)
}

/// Variant of [`train_core`] that publishes progress to an external
/// `AtomicU32` counter. The counter is incremented to `merges_done`
/// after each merge; callers can poll it from another thread to drive
/// a progress bar without touching the hot loop.
pub(crate) fn train_core_with_progress(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
    mode: MergeMode,
    progress: Option<&AtomicU32>,
) {
    debug_assert!(
        !matches!(mode, MergeMode::Auto),
        "MergeMode::Auto must be resolved before train_core"
    );
    match mode {
        MergeMode::Full | MergeMode::Auto => {
            train_core_incremental_inner(words, counts, vocab_size, merges, progress)
        }
        MergeMode::Scan => train_core_scan_inner(words, counts, vocab_size, merges, progress),
    }
}

/// Run the incremental BPE merge loop (Full mode). Test-only thin wrapper
/// over [`train_core_incremental_inner`] with no progress reporting.
#[cfg(test)]
pub(crate) fn train_core_incremental(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
) {
    train_core_incremental_inner(words, counts, vocab_size, merges, None)
}

/// Inner implementation of the Full-mode merge loop.
fn train_core_incremental_inner(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
    progress: Option<&AtomicU32>,
) {
    assert!(vocab_size >= 256, "vocab_size must be at least 256");
    let num_merges = vocab_size - 256;
    log::info!(
        "Starting BPE training (Full mode): {} merges to compute",
        num_merges
    );
    merges.clear();

    // ---- Initial pair_counts and where_to_update (parallel) ----
    log::info!(
        "Computing initial pair counts from {} unique sequences",
        words.len()
    );
    let (mut pair_counts, mut where_to_update) = count_pairs_parallel(words, counts);

    // ---- Build heap ----
    log::info!("Building heap with {} unique pairs", pair_counts.len());
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, pos) in where_to_update.drain() {
        let c = *pair_counts.get(&pair).unwrap_or(&0);
        if c > 0 {
            debug_assert!(c > 0);
            heap.push(MergeJob {
                pair,
                count: c as u64,
                pos,
            });
        }
    }

    // ---- Merge loop ----
    log::info!("Starting merge loop");
    let mut merges_done = 0u32;
    let mut last_log_percent = 0u32;

    while merges_done < num_merges {
        let Some(mut top) = heap.pop() else {
            break;
        };

        // Lazy refresh: if the count changed since we queued this job,
        // update and requeue.
        let current = *pair_counts.get(&top.pair).unwrap_or(&0);
        if current <= 0 {
            // Pair no longer exists or has non-positive count, skip.
            continue;
        }
        if top.count != current as u64 {
            debug_assert!(current > 0);
            top.count = current as u64;
            heap.push(top);
            continue;
        }

        // Record merge.
        let new_id = 256 + merges_done;
        merges.insert(top.pair, new_id);

        // Merge this pair in all words where it occurs.
        let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
        for &word_idx in &top.pos {
            let changes = words[word_idx].merge_pair(top.pair, new_id);
            for (pair, delta) in changes {
                let delta_total = delta * counts[word_idx];
                if delta_total != 0 {
                    *pair_counts.entry(pair).or_default() += delta_total;
                    if delta > 0 {
                        local_pos_updates.entry(pair).or_default().insert(word_idx);
                    }
                }
            }
        }

        // Push updated entries for newly-created pairs.
        for (pair, pos) in local_pos_updates {
            let cnt = *pair_counts.get(&pair).unwrap_or(&0);
            if cnt > 0 {
                debug_assert!(cnt > 0);
                heap.push(MergeJob {
                    pair,
                    count: cnt as u64,
                    pos,
                });
            }
        }

        merges_done += 1;
        if let Some(p) = progress {
            p.store(merges_done, Ordering::Relaxed);
        }

        let current_percent = (merges_done * 100) / num_merges;
        if current_percent > last_log_percent {
            log::info!(
                "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                current_percent,
                merges_done,
                num_merges,
                top.pair,
                new_id,
                top.count
            );
            last_log_percent = current_percent;
        }
    }

    log::info!("Finished training: {} merges completed", merges_done);
}

/// Run the scan-based BPE merge loop (Scan mode). Test-only thin wrapper.
#[cfg(test)]
pub(crate) fn train_core_scan(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
) {
    train_core_scan_inner(words, counts, vocab_size, merges, None)
}

/// Inner implementation of the Scan-mode merge loop.
///
/// Memory model: drops `where_to_update` and the per-heap-entry position
/// sets entirely. Each iteration, after the lazy-refresh check confirms the
/// popped pair is the live winner, we do **one linear pass over `words`**
/// (in parallel via rayon) to find indices containing it. Peak memory:
/// `words` + `counts` + `pair_counts` + a thin heap of `(pair, count)` —
/// no O(N × L) inverted index.
///
/// **Correctness:** identical to the Full-mode loop because:
/// 1. Initial `pair_counts` is computed by an equivalent parallel reduce.
/// 2. Heap ordering ([`ScanJob`] vs [`MergeJob`]) uses the same Ord
///    (max by count, tie-break ascending pair).
/// 3. Lazy refresh follows the same pattern: pop, compare to live count,
///    repush at live count if stale, drop if non-positive.
/// 4. After the merge applies, deltas are computed by the same
///    `Word::merge_pair` method and accumulated into `pair_counts` the
///    same way.
/// 5. New heap entries are pushed for newly-created pairs (delta > 0)
///    only — same condition as Full mode.
fn train_core_scan_inner(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
    progress: Option<&AtomicU32>,
) {
    assert!(vocab_size >= 256, "vocab_size must be at least 256");
    let num_merges = vocab_size - 256;
    log::info!(
        "Starting BPE training (Scan mode): {} merges to compute",
        num_merges
    );
    merges.clear();

    // ---- Initial pair counts (parallel, no positions) ----
    log::info!(
        "Computing initial pair counts from {} unique sequences",
        words.len()
    );
    let mut pair_counts = count_pairs_only_parallel(words, counts);

    // ---- Build heap of (pair, count). No position sets. ----
    log::info!("Building heap with {} unique pairs", pair_counts.len());
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (&pair, &c) in pair_counts.iter() {
        if c > 0 {
            debug_assert!(c > 0);
            heap.push(ScanJob {
                pair,
                count: c as u64,
            });
        }
    }

    // ---- Merge loop ----
    log::info!("Starting merge loop");
    let mut merges_done = 0u32;
    let mut last_log_percent = 0u32;

    while merges_done < num_merges {
        let Some(mut top) = heap.pop() else {
            break;
        };

        // Lazy refresh — same pattern as Full mode.
        let current = *pair_counts.get(&top.pair).unwrap_or(&0);
        if current <= 0 {
            continue;
        }
        if top.count != current as u64 {
            debug_assert!(current > 0);
            top.count = current as u64;
            heap.push(top);
            continue;
        }

        // Record merge.
        let new_id = 256 + merges_done;
        merges.insert(top.pair, new_id);

        // Discover positions: linear scan over `words` in parallel.
        // Each chunk reports the indices it owns that contain `top.pair`.
        // We only need the indices; the merge step below reads `words`
        // mutably, so position discovery has to complete first.
        let target = top.pair;
        let mut positions: Vec<usize> = words
            .par_iter()
            .enumerate()
            .filter_map(|(i, w)| {
                if counts[i] == 0 {
                    return None;
                }
                let ids = &w.ids;
                if ids.len() < 2 {
                    return None;
                }
                let (a, b) = target;
                // We only need to know whether the pair is present; merge_pair
                // will handle non-overlapping replacement.
                for j in 0..ids.len() - 1 {
                    if ids[j] == a && ids[j + 1] == b {
                        return Some(i);
                    }
                }
                None
            })
            .collect();
        // Sorted scan order so the hot loop is deterministic in the face
        // of rayon's nondeterministic completion order. (The output merges
        // are deterministic without this — pair_counts deltas are
        // commutative-associative — but a sorted iteration is simpler to
        // reason about and to diff against Full mode if debugging.)
        positions.sort_unstable();

        // Apply the merge in those words and accumulate deltas.
        let mut newly_created: AHashSet<Pair> = AHashSet::new();
        for &word_idx in &positions {
            let changes = words[word_idx].merge_pair(top.pair, new_id);
            for (pair, delta) in changes {
                let delta_total = delta * counts[word_idx];
                if delta_total != 0 {
                    *pair_counts.entry(pair).or_default() += delta_total;
                    if delta > 0 {
                        newly_created.insert(pair);
                    }
                }
            }
        }

        // Push fresh heap entries only for newly-created pairs. Pre-existing
        // pairs already have an entry; lazy refresh will fix the count when
        // it's popped.
        for pair in newly_created {
            let cnt = *pair_counts.get(&pair).unwrap_or(&0);
            if cnt > 0 {
                debug_assert!(cnt > 0);
                heap.push(ScanJob {
                    pair,
                    count: cnt as u64,
                });
            }
        }

        merges_done += 1;
        if let Some(p) = progress {
            p.store(merges_done, Ordering::Relaxed);
        }

        let current_percent = (merges_done * 100) / num_merges;
        if current_percent > last_log_percent {
            log::info!(
                "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                current_percent,
                merges_done,
                num_merges,
                top.pair,
                new_id,
                top.count
            );
            last_log_percent = current_percent;
        }
    }

    log::info!("Finished training: {} merges completed", merges_done);
}

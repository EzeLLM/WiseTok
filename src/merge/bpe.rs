use std::collections::HashMap as StdHashMap;

use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use rayon::prelude::*;

use super::heap::MergeJob;
use super::word::Word;
use crate::Pair;

/// Compute initial pair frequencies and the pair → word-index inverted index
/// in parallel. Returns `(pair_counts, where_to_update)`.
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

/// Run the incremental BPE merge loop.
/// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
/// `counts`: same length as `words`, count per chunk.
/// `merges`: output map, populated with `(pair, new_id)` entries.
pub(crate) fn train_core_incremental(
    words: &mut [Word],
    counts: &[i64],
    vocab_size: u32,
    merges: &mut StdHashMap<Pair, u32>,
) {
    assert!(vocab_size >= 256, "vocab_size must be at least 256");
    let num_merges = vocab_size - 256;
    log::info!("Starting BPE training: {} merges to compute", num_merges);
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
                heap.push(MergeJob {
                    pair,
                    count: cnt as u64,
                    pos,
                });
            }
        }

        merges_done += 1;

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

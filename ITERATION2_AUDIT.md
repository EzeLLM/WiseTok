# Iteration 2 Audit

Zero-trust verification of the current codebase before building memory-bounded merge mode.

## 1. Module structure

Tree (src/, 17 .rs files, ~2200 LOC total):

```
src/
├── lib.rs                  (24 LOC)  crate root, re-exports, `Pair`, GPT4_PATTERN
├── python.rs              (555 LOC)  PyO3 `Tokenizer` pyclass — entry point
├── special_tokens.rs      (208 LOC)  SpecialTokenRegistry + Segment + split()
├── tests.rs               (423 LOC)  internal Rust unit tests (34)
├── merge/
│   ├── mod.rs             (11 LOC)   module wiring
│   ├── word.rs            (70 LOC)   Word::merge_pair returning local deltas
│   ├── heap.rs            (40 LOC)   MergeJob (count + pair + AHashSet<usize> pos)
│   └── bpe.rs            (154 LOC)   count_pairs_parallel + train_core_incremental
├── pretokenizer/
│   ├── mod.rs             (24 LOC)   PreTokenizer trait (Send + Sync)
│   ├── regex.rs           (46 LOC)   RegexPreTokenizer (fancy_regex)
│   ├── digits.rs          (37 LOC)   DigitSplitter (ASCII digits → 1-byte chunks)
│   └── sequence.rs        (35 LOC)   SequencePreTokenizer (compose)
└── export/
    ├── mod.rs             (3 LOC)
    ├── byte_level.rs     (266 LOC)   GPT-2 byte→unicode 256-entry table
    ├── tiktoken.rs        (36 LOC)   mergeable_ranks builder
    └── huggingface.rs    (269 LOC)   build_tokenizer_json + writers
```

External tests:

```
tests/
├── test_pretokenizer.rs        (12 tests)
├── test_special_tokens.rs      (22 tests)
├── test_huggingface_export.rs  (11 tests)
└── python/
    └── test_tokenizer.py       (25 tests)
```

## 2. Iteration 1 verification — features actually wired end-to-end

| Feature | Wired into train? | Wired into encode? | Wired into export? | Tests | Status |
| --- | --- | --- | --- | --- | --- |
| Module split | n/a | n/a | n/a | all | ✅ matches commit `55f0130` |
| i64 counts | yes | yes | yes | covered | ✅ no i32 in counts/deltas |
| min_frequency | yes (post-aggregation) | n/a | n/a | 4 Python | ✅ but see Known Issue |
| PreTokenizer pipeline | yes (`pre_tokenizer` field) | yes (same path) | yes (regex stored) | 6 Python + 12 Rust | ✅ |
| SpecialTokenRegistry | yes (split before BPE) | yes (atomic IDs) | yes (added_tokens + tail IDs) | 5 Python + 22 Rust | ✅ |
| HuggingFace export | n/a | n/a | yes | 3 Python + 11 Rust | ✅ |

Test results:
- `cargo test`: **79 Rust passing** (34 unit + 11 + 12 + 22)
- `pytest tests/python/`: **25 passing**
- `cargo clippy --all-targets -- -D warnings`: clean
- `cargo fmt --all -- --check`: clean

End-to-end smoke trace (from `train_from_iterator` in `src/python.rs:131`):

1. Resolve pre-tokenizer from `pre_tokenizer` spec or legacy `pattern` (mutual exclusion at 150).
2. Reset `self.specials`, then add the user's `special_tokens=[…]` (178–185).
3. Stream from Python iterator under GIL, refilling a `Vec<String>` of size `buffer_size` (206–232).
4. **Detach GIL** (249), `par_iter` over the buffer; for each string call `specials_ref.split(s)`, then for each `Segment::Text` call `pre_ref.pre_tokenize(text)` and count chunks.
5. Reduce parallel maps into the global `counts: AHashMap<CompactString, i64>` (272–274).
6. Materialize words/cvec, applying `c < min_frequency` filter (290–298).
7. Hand off to `train_core_incremental` (308) → returns `merges: HashMap<Pair, u32>`.
8. Stash the pre-tokenizer in `self.pre_tokenizer` (311) so `encode` uses the same pipeline.

`encode` (src/python.rs:417) mirrors the training split: `self.specials.split(text)` first, `Segment::Special` emits `256 + num_merges + i`, `Segment::Text` runs through `self.pre_tokenizer` then through `encode_chunk_into` (lowest-`new_id`-first BPE).

**Conclusion:** Iteration 1 is genuinely complete. No half-wired features. No half-finished modules. The only loose end (HF-renumber tool, task #15) was an explicitly scoped follow-up, not an Iteration 1 deliverable.

## 3. Merge phase memory model — the OOM trace

The structures live across the merge loop:

| Structure | Size class | When |
| --- | --- | --- |
| `words: Vec<Word>` (each `Word.ids: Vec<u32>`) | O(N × L) × 4 bytes | entire merge phase, shrinks per merge |
| `counts: Vec<i64>` (per word, never modified) | N × 8 bytes | entire merge phase |
| `pair_counts: AHashMap<Pair, i64>` | O(P_unique) entries × 24 bytes | entire merge phase, grows on merges |
| `where_to_update: AHashMap<Pair, AHashSet<usize>>` | **O(N × L) entries** | drained at boot into heap |
| `OctonaryHeap<MergeJob>` | each job carries an `AHashSet<usize>` of word indices | entire merge phase |

Where N = unique words, L = average symbols per word.

### `count_pairs_parallel` — `src/merge/bpe.rs:14`

```rust
words.par_iter().enumerate()
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
    .reduce(...)
```

For every adjacent pair across every word, we add `i` (the word index) to the set for that pair. The total entry count across all sets is bounded above by `Σ(L_w - 1) ≈ N × L`.

### Drain into the heap — `src/merge/bpe.rs:71`

```rust
for (pair, pos) in where_to_update.drain() {
    let c = *pair_counts.get(&pair).unwrap_or(&0);
    if c > 0 {
        heap.push(MergeJob { pair, count: c as u64, pos });
    }
}
```

The `where_to_update` map is fully drained, but its memory just **moves** into the heap — every `MergeJob` owns an `AHashSet<usize>`. So the dominant cost just shifts containers.

### During the merge loop — `src/merge/bpe.rs:87`

For each merge:
1. Pop a `MergeJob`. Lazy refresh: if its stored count differs from live `pair_counts`, repush and continue (the same pair often has multiple entries with different `pos` sets coexisting in the heap — this is the well-known correctness invariant from the upstream rustbpe audit).
2. For each `word_idx` in `top.pos`, call `Word::merge_pair`. That returns local deltas describing pairs created (`+1`) and destroyed (`-1`) **in that word only**.
3. Update global `pair_counts` by `delta * counts[word_idx]`.
4. **Push fresh `MergeJob` entries** for any newly-created pairs, carrying a fresh `AHashSet<usize>` of just the word indices that produced the new pair on this iteration (`local_pos_updates`, src/merge/bpe.rs:118).

So the heap accumulates entries continuously: every merge that creates new pairs pushes new sets. Stale entries (old counts) are not cleaned eagerly — they sit until popped, at which point the lazy-refresh path either repushes (still relevant) or drops (count hit zero). The peak memory is therefore *not* just the initial `where_to_update`; it's the running maximum of (`where_to_update` initial allocation) + (sum of `local_pos_updates` since boot, minus drops).

### Quantitative estimate

For N=50M unique chunks, L=10 symbols, the initial inverted index has ~500M entries.

- `AHashSet<usize>` overhead per entry: roughly 16 bytes (one usize + load-factor slack) for raw element storage, plus per-set fixed overhead. At ~24 bytes amortized per (pair, word_idx) edge, 500M edges ≈ **12 GB** just for the initial position sets.
- `pair_counts`: the unique pair count is bounded by `N × L` initially, drops as common pairs disappear, but can grow back from new merges. Dominant constant: `(8B key + 8B count + ahash overhead) ≈ 32 B` × 500M unique pairs ≈ 16 GB peak.
- `words` themselves: 500M u32 IDs ≈ 2 GB.

Total: ~30 GB just for these structures, before counting the heap entries scattered across the merge loop. This is consistent with HF tokenizers OOMing at ~68 GB on a 35 GB corpus (their structures aren't identical, but the order of magnitude matches).

**The position sets are the unbounded structure.** `pair_counts` is bounded by unique pairs (linear in `N × L` at worst); `words` shrink as merges apply. The position sets duplicate every (pair, word_idx) edge, and they live for the entire merge phase because the heap keeps pushing new ones.

### Why the heap is the real memory hog, not just `where_to_update`

A merge that splits 1M words creates up to 6M delta entries (`Word::merge_pair` produces up to 6 deltas per merge site). Of those, half are `+1` deltas → up to 3M new (pair, AHashSet) pushes onto the heap, each carrying a fresh AHashSet of relevant word indices. None of these are reclaimed until popped. With 24K merges, this is huge.

A scan-based mode that drops `where_to_update` entirely and discovers the winning pair's positions by linear scan trades:
- **Memory:** -O(N × L) edges = the entire position-set memory disappears
- **Time:** +O(N × L) per merge to find positions of the winner

Crucially: incremental delta updates to `pair_counts` still work fine — the pair selection logic only needs `pair_counts`, not positions. So one full scan per merge to find positions of the winner is enough; we don't need to recount pairs from scratch.

## 4. Code quality / dead code / TODOs

- No TODO/FIXME/XXX/HACK comments anywhere in `src/` or `tests/`. Clean.
- `RegexPreTokenizer::pattern_str` is dead per `KNOWN_ISSUES.md` but kept intentionally — the field is still consumed by HF export tests indirectly. Verified: no stray callers, fine to leave.
- `count_pairs_parallel` is `#[cfg(test)]`-exposed (src/merge/mod.rs:8) — used only in unit tests, not by the production code path that exposes it. Fine.
- `MergeJob` is also `#[cfg(test)]`-exposed for the determinism test. Fine.
- `Tokenizer::new()` constructs an empty regex (`Regex::new("").expect(...)`) so `Default` works without panicking. This is a pre-existing quirk from rustbpe; keep it (struct-literal test fixtures depend on it).
- `i64 → u64` casts at three sites in `merge/bpe.rs` (76, 100, 130) — `KNOWN_ISSUES.md` flagged adding `debug_assert!(c > 0)`. Cheap, will fold into the Iteration 2 work touching this file.
- The chunk materialization at `src/python.rs:294` writes `Vec<u32>` per byte chunk (4× blowup vs `Vec<u8>` until the first ≥256 merge). Flagged in `KNOWN_ISSUES.md` as deferred — possibly revisit during memory-bounded merge work, but not blocking.
- Encode's `best_pair.unwrap().2` style is awkward (`KNOWN_ISSUES.md`). Validation-only path, defer.

## 5. Assessment

**The codebase is ready for Iteration 2.** Iteration 1 is complete in fact, not just in name:

- All 104 tests pass; clippy and fmt are clean.
- Every feature is end-to-end. `train → export → reload-in-HF` produces byte-identical IDs to a fresh `wisetok.encode` call (verified by `test_special_tokens_hf_export_uses_registered_specials`).
- The merge phase memory model is exactly as designed: position sets dominate, and they live for the entire merge phase. This is the right place to attack.
- No loose ends from Iteration 1 that block Iteration 2.

### Recommended Iteration 2 entry point

Per the briefing, **Approach A (scan-based)** with `MergeMode::{Full, Scan, Auto}`:

1. Add `MergeMode` enum and threaded into `train_core_incremental` / `train_from_iterator`.
2. Implement `train_core_scan` as a sibling to `train_core_incremental`. Same pair selection logic, same delta computation, same tie-breaking — only the pair → word-index lookup changes from "consume from the heap entry's set" to "linear-scan all words for that pair this iteration."
3. Identical-merge-table test across `Full` and `Scan` on a fixture corpus. This is the gating correctness test.
4. `Auto` policy: `> 1M unique words` after aggregation → `Scan`, else `Full`. Easy to override per call.

Approach B (Budgeted) is interesting but adds significant complexity for a middle-ground that we won't validate without fairly large corpora; defer until we measure the perf gap between Full and Scan and have a real reason to add B.

**Defer until decision is approved.** No implementation code yet.

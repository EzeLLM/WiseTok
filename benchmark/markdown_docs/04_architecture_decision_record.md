# Architecture Decision Record: Incremental Pair Counting in BPE Merge Loop

**Date:** 2024-06-15  
**Status:** Accepted  
**Authors:** TokenLib Core Team  
**Reviewers:** @karpathy, @data-team-lead

---

## Context

TokenLib's merge loop processes billions of token pairs during training on large corpora (>100GB). The baseline approach—rescanning the entire vocabulary after each merge to recount all pairs—is O(n²) per merge iteration and becomes prohibitively slow.

For example:
- Corpus: 100GB of text
- Target vocabulary: 50,000 tokens
- Time with full recount: ~6 hours on a single V100 GPU
- Required latency: <30 minutes for iteration in research workflows

### Constraints
1. **Exact pair counts:** Output must be byte-identical to reference implementations (minbpe, HF tokenizers).
2. **Memory efficiency:** Available RAM during training often constrained (16–64 GB on commodity hardware).
3. **Determinism:** Tie-breaking by lexicographic order for reproducibility across runs.

### Explored Alternatives

| Approach | Pro | Con |
|----------|-----|-----|
| Full recount per merge | Trivial correctness proof | O(n²) time, unacceptable latency |
| Lazy delta tracking | O(log n) amortized; proven in practice | Requires careful bookkeeping; hard to verify |
| Spatial locality optimization | Cache-friendly; reduces memory bandwidth | Complex; compiler-dependent behavior |

---

## Decision

Implement an **incremental pair counting** strategy using local deltas:

1. **Initial state:** Scan vocabulary once to build global `pair_counts: HashMap<Pair, i64>`.
2. **After each merge:**
   - For each word affected by the merge, compute local `(pair, ±delta)` tuples describing:
     - Pairs that were destroyed (delta = -count)
     - Pairs that were created (delta = +count)
   - Accumulate deltas into the global map: `pair_counts[pair] += delta * word_count`.
   - Push only newly-created pairs to the heap (existing pairs are lazily refreshed on pop).
3. **Lazy heap refresh:** When a pair is popped from the max-heap, check if its stored count matches the live value in `pair_counts`. If stale, re-push with the current count. If count ≤ 0, drop silently.

---

## Consequences

### Positive
- **Performance:** O(log n) amortized time per merge (only O(n) initial scan). Training time reduced from 6h to ~25min on same hardware.
- **Memory:** No auxiliary data structures per word; only global `pair_counts` and heap (O(vocab_size) space).
- **Correctness:** Local deltas are exact—no approximation or sampling. Output verified bit-identical against minbpe on 10GB+ corpora.
- **Determinism:** Heap tie-breaking by pair (left_id, right_id) ensures reproducible results across runs.

### Negative
- **Complexity:** Harder to audit than naive recount. Requires careful reasoning about heap invariants (details in AUDIT_REPORT.md).
- **Maintenance burden:** Future refactors must preserve the delta invariant. Merging code changes from upstream is error-prone.

### Neutral
- Requires WASM/FFI bookkeeping for streaming iterators (Python) to avoid holding GIL during GHz-scale computations.

---

## Implementation Notes

- Code lives in `src/lib.rs::train_core_incremental` (~300 lines).
- `OctonaryHeap` chosen over binary heap (wider branching = fewer comparisons for push-heavy workloads).
- Pair representation: `(left_id: u32, right_id: u32)` hashable via `AHash` (non-cryptographic, 3× faster than `std::hash::SipHash`).
- Merge rule output: `HashMap<Pair, u32>` where value is the new token ID (starting at 256 for single-byte tokens 0–255).

---

## Trade-offs

| Trade-off | Rationale |
|-----------|-----------|
| Heap laziness over eager repair | Laziness allows multiple stale entries for the same pair with different `pos` sets; this is safe because we only care about current counts when popping. Eager repair would require O(n log n) work per merge. |
| Global state over thread-local | Global `pair_counts` allows rayon workers to safely accumulate deltas without coordination (via atomic adds, amortized into a single bulk lock after merge). Cleaner than thread-local + reduce. |
| Tuple `(Pair, pos_set)` on heap vs just `Pair` | Positions allow efficient delta application to only affected words. Without it, we'd rescanning the entire vocab to find which words contain a pair. |

---

## Alternatives Considered

### 1. Fenwick Tree / Segment Tree
Track frequencies in a tree structure for O(log n) updates. Rejected: overkill complexity, no speed improvement over hash-based deltas, and harder to distribute across workers.

### 2. Approximate Counting (Sketch Data Structures)
Use Count-Min Sketch or similar to estimate frequencies with bounded error. Rejected: output would not match reference implementations (no exact counts). Training experiments use exact counts for final vocab.

### 3. Batch Merges
Apply k merges before recounting (k = 10–100). Rejected: breaks determinism and introduces hyperparameter tuning. Single merges per recount is cleaner.

---

## Related Decisions

- **ADR-2024-06-14:** Streaming iterator design (Python -> Rust with GIL release)
- **ADR-2024-05-20:** `dary_heap` choice over `BinaryHeap` (8-ary branching for cache locality)

---

## Validation

- Unit tests: 15 tests in `src/lib.rs::tests` cover merge correctness, delta application, and tie-breaking.
- Integration tests: Python tests in `tests/python/test_correctness.py` compare outputs against minbpe, HuggingFace tokenizers, and tiktoken on 10+ real corpora.
- Regression: Every commit runs full benchmark suite (1GB, 10GB, 100GB corpora). Latency must not regress >5%.

---

## Approval

- **Approved by:** @karpathy (2024-06-20)
- **Merged:** PR [#310](https://github.com/openai/tokenlib/pull/310)
- **Deploy:** Shipped in v0.7.0 (2024-08-15)

---

**Appendix:** Full algorithm pseudocode and correctness proof in [AUDIT_REPORT.md](./AUDIT_REPORT.md) § Claim #3.

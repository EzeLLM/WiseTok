# Audit Report: WiseToken Spec vs. rustbpe Source

Audit date: 2026-05-01.
Source: `/home/ezel/Development/WiseTok` (already a fork — origin `https://github.com/EzeLLM/WiseTok`, package still named `rustbpe v0.1.0`, recent commits include PyO3 0.27 bump).
Method: line-by-line read of `src/lib.rs` (1070 lines), `Cargo.toml`, `tests/python/test_tokenizer.py` (743 lines), `.github/workflows/*.yml`. Baseline `cargo test` run with `LD_LIBRARY_PATH=/home/ezel/miniconda3/lib`.

## TL;DR

The spec is **directionally correct on every "wrong/missing" claim** but contains **several factual errors and outdated details**. Most importantly:

- **rustbpe IS already byte-level BPE** (initial alphabet = bytes 0–255). The spec's "no byte-level BPE" claim is misleading. The actual limitation is that the *pre-tokenizer* requires UTF-8 strings, not raw bytes.
- The source has **33 Rust tests, not 14** as the spec states. `src/lib.rs` is **1070 lines, not 600**.
- Dependencies are **PyO3 0.27**, not 0.22 as the spec lists — and the API has been migrated (`py.detach`, `pyo3::Python::attach`, `Bound`-based FFI). The spec's dependency list is stale.
- **`indexmap` is declared but unused** in `lib.rs`. Dead dependency.

The fork plan is sound. Recommend several spec amendments before implementation (see end of report).

---

## Claim-by-claim verification

### Claim 1: `i32` counts overflow risk — **CORRECT, with one nuance**

Verified at:
- `lib.rs:138` — `count_pairs_parallel` takes `counts: &[i32]` and returns `AHashMap<Pair, i32>`.
- `lib.rs:173` — `train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, ...)`.
- `lib.rs:318,370,373` — chunk count maps are `AHashMap<CompactString, i32>`.
- `lib.rs:65` — `Word::merge_pair` returns `Vec<(Pair, i32)>` deltas.

**Nuance the spec missed:** `MergeJob.count: u64` (`lib.rs:105`) is already `u64`, so the heap entries themselves are safe. However the cast at `lib.rs:217` `top.count = current as u64` is from `i32` via sign-extension — if `current` ever went negative due to overflow in the i32 deltas, this would wrap to a huge `u64` and corrupt the heap order. The check `current <= 0` at `lib.rs:212` masks this in normal operation but won't catch a wrap-around overflow that results in a positive i32. **Migrating to i64 closes this hole; just casting to u64 from i32 doesn't.**

Recommendation: i64 throughout, as the spec says. Add a debug assertion on count non-negativity.

### Claim 2: No `min_frequency` — **CORRECT**

Grep for `min_freq|min_frequency|frequency` in `lib.rs` returns one match (`lib.rs:261`), which is just a log-format string in the merge progress message. There is no filtering anywhere.

### Claim 3: Lazy-refresh heap is correct — **CORRECT (verified by trace)**

I traced the merge loop carefully (`lib.rs:200–271`):
- A pair can have multiple `MergeJob` entries in the heap simultaneously: the original from `count_pairs_parallel`, plus new entries pushed at `lib.rs:244–253` whenever a delta creates that pair in additional words.
- Each entry stores a `count` that was the **global** count at push time, plus a `pos` set of word indices that the pusher cared about.
- On pop: if `pair_counts[top.pair] <= 0`, drop the entry. If `top.count != global`, update `top.count = global` and re-push. Else process: `merge_pair` on every word in `top.pos`.
- If a word in `pos` no longer contains the pair (a previous merge consumed it), `merge_pair` is a no-op and returns empty deltas — wasteful but correct.
- Across multiple heap entries for the same pair, the counts are not partitioned by `pos`; each entry redundantly claims the global count. The lazy refresh ensures only one entry processes effectively before subsequent entries see a fresh-but-different global and either drop or re-push.

Verdict: correct, slightly wasteful. **Do not rewrite this loop**; port it as-is.

### Claim 4: rustbpe is "string-level" BPE, not byte-level — **INCORRECT / MISLEADING**

Look at `lib.rs:407–410`:
```rust
words.push(Word::new(
    chunk.as_bytes().iter().map(|&b| b as u32).collect(),
));
```
And `lib.rs:433`:
```rust
let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();
```

**The initial alphabet is the 256 raw bytes.** Every chunk is decomposed into its UTF-8 bytes before BPE applies. This *is* byte-level BPE. `get_mergeable_ranks` returns `Vec<(Vec<u8>, u32)>` — exactly what tiktoken wants for byte-level encoding.

The **actual** limitation is upstream: the regex pre-tokenizer (`fancy_regex`) operates on `&str`, which forces the input to be valid UTF-8. Binary blobs / invalid-UTF-8 sequences can't go through `train_from_iterator` because Python `obj.extract::<String>()` (`lib.rs:345`) will fail. This is a real limitation but it is a **pre-tokenizer issue**, not a BPE-alphabet issue.

**Action for the spec:** rephrase Section 2.1 (byte-level note). The spec's statement *"our initial alphabet is always the 256 raw bytes ... we store chunks as Vec<u8> not strings"* is correct for the new design but should not be sold as a *fix* — it is *parity* with rustbpe on the alphabet, plus a real change on the pre-tokenizer input type. The HuggingFace `ByteLevel` byte-to-unicode mapping is its own export-format concern, separate from "byte-level BPE."

### Claim 5: No special tokens — **CORRECT**

Grep for `special` and `<|` in `lib.rs` returns zero matches.

### Claim 6: Pre-tokenizer is just a single regex — **CORRECT**

`GPT4_PATTERN` at `lib.rs:13`. Compiled at `lib.rs:308`. Used at `lib.rs:374` (training) and `lib.rs:464` (encode). No digit splitting, no composability, no fallbacks.

### Claim 7: `OctonaryHeap` is an 8-ary max-heap — **CORRECT**

- `Cargo.toml:18` declares `dary_heap = "0.3"`.
- `lib.rs:4` imports `dary_heap::OctonaryHeap`. `dary_heap`'s `OctonaryHeap` is the 8-ary variant by definition.
- `Ord` for `MergeJob` (`lib.rs:122–132`) returns `self.count.cmp(&other.count)` (ascending), and `OctonaryHeap` is a max-heap (largest at top), so this produces max-by-count behavior. Tie-break is `other.pair.cmp(&self.pair)` — i.e., reversed pair order, so the *smaller* pair wins on ties (deterministic).

### Claim 8: rayon parallel aggregation, GIL released, parallel reduction correct — **CORRECT**

- `lib.rs:370` — `let local: AHashMap<...> = py.detach(|| { buf.par_iter().map(...).reduce(...) })`. `py.detach` is the modern PyO3 0.27 equivalent of `Python::allow_threads` — GIL is released for the parallel block.
- The reduce function (`lib.rs:380–385`) merges per-thread `AHashMap`s with `*a.entry(k).or_default() += v`. This is associative and commutative for integer addition, so parallel tree-reduction is correct.
- The same GIL-release pattern is used in `batch_encode` at `lib.rs:563`.

The spec is right that the pattern is correct. **Reduction overflow risk is the same as Claim 1.**

---

## Spec issues the audit found (not in the spec)

1. **Outdated PyO3 version.** Spec lists `pyo3 = "0.22"`. Actual Cargo.toml uses `0.27`. The migration to 0.27 has already been done in this fork (commits `8bcf313`, `402c81c`). The `Bound<'_, PyAny>`-based FFI calls (`pyo3::ffi::PyObject_GetIter`, `pyo3::ffi::PyIter_Next`) and `Python::attach`/`py.detach` are the new API. **Do not downgrade.** Update the spec's dependency list.

2. **Test count is wrong.** Spec says "Tests (14 Rust tests)" (line 50). Actual: **33 Rust tests** (`grep -c '^    fn test_' src/lib.rs`). All 33 pass under `cargo test`. The full suite includes additional tests for chained merges, mergeable-ranks reconstruction, decode error paths, count-pairs edge cases, vocab_size, etc. The migration plan must port all 33.

3. **Source is 1070 lines, not 600.** The spec's structure overview (lines 37–51) is a summary, not a count, but be aware the file is larger.

4. **Unused `indexmap` dependency.** `Cargo.toml:19` declares `indexmap = "2.2"` but `grep -rn 'indexmap\|IndexMap' src/lib.rs` returns nothing. Either drop it or note it for future use.

5. **`cargo test` requires libpython on `LD_LIBRARY_PATH` on this system.** Even though `extension-module` is not enabled by default (correct in `Cargo.toml:32–35`), the test binary still links to libpython. On this machine, the system libpython is 3.13 but the active Python is conda 3.12, so the test binary can't find `libpython3.12.so.1.0` without `LD_LIBRARY_PATH=/home/ezel/miniconda3/lib`. Worth documenting in CONTRIBUTING / the new spec, and worth adding a CI matrix that doesn't rely on this.

6. **`encode` is O(n² log m)** per chunk, not just O(n²) as the spec implies. Each iteration does a full linear scan over `len-1` pairs, and we do up to `len-1` iterations. The spec is right that it's slow and should not be used for production inference. (Note: this is fine for validation; the architecture intent is to export to tiktoken/HF for inference.)

7. **The fork is already in progress.** `git log` shows: `add tags to readme`, `Bump PyO3 to 0.27`, `Fix CI: use maturin build instead of develop`. The current state is rustbpe + PyO3 0.27 + CI fixes. **It is not yet renamed to wisetoken.** The spec assumes a clean fork point at upstream v0.1.0, but the `cp -r rustbpe wisetoken` step in the briefing is unnecessary — we are already in the fork.

8. **`MergeJob` deduplication subtlety, beyond the lazy-refresh check.** When the same pair gets multiple heap entries, after one is processed (deltas applied), subsequent entries for the same pair can have stale `pos` sets that no longer match where the pair lives. They are still safe (because `merge_pair` is a no-op if the pair is gone), but we waste pops. With i64 counts and large vocabs this is fine; just noting it for future profiling.

9. **`Word::merge_pair` deltas can list the same pair twice with opposite signs.** E.g., pattern `[x, a, b, x]` with merge `(a,b)→Z` produces deltas including both `((x,a), -1)` and `((x,Z), +1)`. The merge loop sums these into `pair_counts` correctly. **This is not a bug**, but it means iterating deltas does not give a unique pair set; if you ever index by pair (e.g., to insert into `local_pos_updates`), you must guard against the case where `delta < 0` does not warrant a heap push. The current code does this correctly at `lib.rs:236`.

10. **`get_mergeable_ranks` and `decode` have duplicated vocabulary-reconstruction code** (`lib.rs:432–454` vs `lib.rs:510–537`). Refactor candidate when modularizing.

---

## Concerns about the fork approach

1. **License compliance.** The repo has `LICENSE` (MIT). MIT requires preserving the copyright notice. When renaming/restructuring:
   - Keep the original MIT LICENSE file.
   - Add a `NOTICE` or extend `README.md` attribution explicitly: "WiseToken is a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe), MIT-licensed, copyright (c) Andrej Karpathy".
   - Do NOT remove any `Copyright (c) ...` lines from the existing LICENSE.

2. **Crate name change requires upstream-incompatible changes.** Changing `name = "rustbpe"` to `name = "wisetoken"` in Cargo.toml + `pyproject.toml` will break anyone currently `import rustbpe`-ing this fork. The current fork at `EzeLLM/WiseTok` still publishes as `rustbpe`. Decision needed: do we keep the Python module name `rustbpe` for drop-in compatibility, or rename?
   - Recommendation: rename to `wisetoken`. The current fork has only diverged on PyO3 version and CI; minimal users expected.

3. **Upstream tracking.** With the fork already on `EzeLLM/WiseTok` and divergent commits, future cherry-picks from `karpathy/rustbpe` (e.g., upstream bug fixes) need a strategy. Add `karpathy/rustbpe` as a remote, document the merge process in CONTRIBUTING.

4. **Python 3.12 vs 3.13 build matrix.** Recent commits suggest forward-compat issues with newer Pythons. Iteration 3's CI plan should pin a matrix and verify wheels are abi3 (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY` is already set in `release.yml`).

5. **Working dir name vs package name vs project name.** Currently:
   - Working directory: `WiseTok`
   - Cargo package: `rustbpe`
   - Spec/repo target: `wisetoken`
   - GitHub repo: `EzeLLM/WiseTok`
   
   This is going to confuse everything. Recommend deciding on **one** canonical spelling. The spec uses `wisetoken` (lowercase). The repo uses `WiseTok` (different word entirely). Pick one before any rename PR.

---

## Recommended spec amendments (before implementation)

1. **Section 1, item 5 ("No byte-level BPE"):** Rewrite. rustbpe IS byte-level BPE on bytes 0–255. The actual gap is that (a) inputs must be valid UTF-8 (regex pre-tokenizer constraint), and (b) HF export needs the GPT-2 byte-to-unicode visual mapping, which rustbpe doesn't generate (because it doesn't export to HF at all).

2. **Section "Dependencies":** Bump PyO3 to 0.27 to match current state. Drop `indexmap` unless re-introducing. Add note about `LD_LIBRARY_PATH` for `cargo test` on systems without system libpython.

3. **Section "Migration from rustbpe", item 8:** Update test count from "All 14 Rust tests" to "All 33 Rust tests."

4. **Section "Project structure":** Decide on `wisetoken` vs `WiseTok` vs `rustbpe` and use it consistently. The Cargo package name, Python module name, and repo directory should align.

5. **Section "What NOT to build":** Add "we are not changing the merge algorithm." The lazy-refresh heap is correct and tested; modularizing should preserve behavior bit-exactly.

6. **Section "Testing strategy", item 1 (Correctness against rustbpe):** Pin a specific upstream commit to compare against, since the fork has already diverged. Suggest: pin to `karpathy/rustbpe@<commit_hash_at_v0.1.0>` and verify byte-identical merges on a small fixture corpus.

7. **Section "Iteration 1", step 9 (HF export verification):** Add an explicit step "generate a reference `tokenizer.json` from HF tokenizers' `BpeTrainer` on the same training data, then diff field-by-field against our output." The spec mentions this in passing but should be a hard acceptance criterion.

---

## Suggested first commits (after spec amendments approved)

In order:

1. `chore: drop unused indexmap dependency` (one-line Cargo.toml change).
2. `docs: add AUDIT_REPORT.md` (this file).
3. `chore: rename package rustbpe → wisetoken` (Cargo.toml, pyproject.toml, lib.rs `#[pymodule] fn` name, README) — BUT only after deciding on the canonical name (see concerns #5 above).
4. `refactor: change i32 counts to i64 throughout` (single mechanical change, easy to verify by re-running 33 tests).
5. … then proceed with the modular refactor per the spec.

I am not making any of these commits yet. Awaiting your confirmation on:
- (a) Is this audit accurate / do you want changes?
- (b) Decide on canonical name: `wisetoken` (spec), `WiseTok` (repo dir/GitHub), or keep `rustbpe`?
- (c) Approve the spec amendments above before I proceed.

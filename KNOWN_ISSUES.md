# Known issues

Tracked items found during the Iteration-1 review. None block correctness today; all are deferred fixes or follow-ups.

## Code

### `min_frequency` filters after aggregation, not during
- **Where:** `src/python.rs::train_from_iterator`, materialization step.
- **What:** The `counts: AHashMap<CompactString, i64>` map fills to its full size before `min_frequency` is consulted. The filter only shrinks the downstream `words`/`cvec` vectors, not the chunk-count map. On corpora with hundreds of millions of unique chunks, the HashMap dominates RAM long before the merge loop sees the filter.
- **Impact:** `min_frequency` is *not* a Phase-1 memory backstop. Document it as such.
- **Fix:** Pair with the Iteration-2 RAM monitor / adaptive flush. The flush is the right place to apply low-frequency drops under memory pressure.

### `min_frequency=0` and negatives are silently equivalent to `min_frequency=1`
- **Where:** `src/python.rs::train_from_iterator`, parameter signature.
- **What:** The aggregator never emits a chunk with `count == 0`, so `c >= 0`, `c >= 1`, and `c >= -5` filter the same set. Type is `i64`, so negatives are accepted without complaint.
- **Fix (pick one):** (a) Document that values ≤ 1 are equivalent to "keep everything." (b) Validate `min_frequency >= 1` with a `PyValueError` and switch the parameter type to `u64`.

### `i64 → u64` casts in `merge::bpe.rs` rely on a manual invariant
- **Where:** `src/merge/bpe.rs:76`, `:100`, `:130` — three sites all of the form `count: c as u64` after a `c > 0` check.
- **What:** The casts are safe today because each is preceded by a positive-count check in the same function, but there is no static invariant. A reorder during future refactor could let an `i64::MIN` wrap into a giant `u64` and corrupt heap order silently.
- **Fix:** Add `debug_assert!(c > 0);` immediately above each `as u64` site. Cheap; makes the invariant explicit.

### `train_from_iterator` parallel block can panic on regex backtracking error
- **Where:** `src/python.rs`, the rayon `par_iter` block: `mat.expect("regex match failed").as_str()`.
- **What:** `fancy_regex::find_iter` returns `Result<Match, Error>` because backtracking has limits. The training path panics on error; the new `RegexPreTokenizer::pre_tokenize` logs and continues. Inconsistent behavior between the two regex consumers in the codebase.
- **Fix:** Once `train_from_iterator` is migrated to use `RegexPreTokenizer` (planned alongside HF export wiring), this disappears. Until then, replace `expect` with a `match` that logs and skips, matching the pre-tokenizer behavior.

### `Tokenizer::encode` style is awkward (verbatim from upstream)
- **Where:** `src/python.rs::encode`, the inner merge loop.
- **What:** Uses `let mut best_pair: Option<(usize, Pair, u32)> = None;` and reads `best_pair.unwrap().2` inside an `if let`. Correct but stylistically clunky.
- **Fix:** Low priority. Keep until there's a real reason to rewrite `encode` (it's documented as "validation only; use tiktoken for inference").

### `Tokenizer::decode` rebuilds full vocab every call
- **Where:** `src/python.rs::decode`.
- **What:** Reconstructs the byte-vocab on every invocation. Quadratic-ish in vocab size when called repeatedly.
- **Fix:** Low priority. Cache the vocab inside `Tokenizer` if profiling ever flags this. For now: validation-only path, not on the hot path.

### `SpecialTokenRegistry::add` is O(n) on duplicate check
- **Where:** `src/special_tokens.rs::add`, `self.tokens.iter().any(|t| t == &token)`.
- **What:** Linear in registry size. With the code preset (8 entries) plus typical reserve sizes (≤16) it's fine. Becomes an issue only if registry grows past a few hundred.
- **Fix:** Add a parallel `HashSet<String>` index when/if needed. Defer.

### `SequencePreTokenizer` uses dynamic dispatch (`Box<dyn PreTokenizer>`)
- **Where:** `src/pretokenizer/sequence.rs`.
- **What:** Vtable indirection per chunk per step. Negligible vs. regex-match cost in practice.
- **Fix:** If profiling ever shows this is hot, convert to a static `enum PreTokenizerKind { Regex(...), Digits, Sequence(Vec<PreTokenizerKind>) }`. Defer.

### Materialization allocates `Vec<u32>` for byte-only chunks
- **Where:** `src/python.rs`, `chunk.as_bytes().iter().map(|&b| b as u32).collect()`.
- **What:** 4× memory blowup vs. raw bytes during the pre-merge phase. After the first merge that produces an id ≥ 256, `Vec<u32>` is required, so the optimization is "use `Vec<u8>` until the first such merge, then upgrade." Complex for marginal gain.
- **Fix:** Defer. Possibly revisit during the Iteration-2 memory work.

### CLI binary links libpython unnecessarily
- **Where:** `Cargo.toml` declares pyo3 as a top-level dep so it's linked into the `wisetok` binary's rlib path even though the binary never touches Python.
- **Symptom:** `./target/release/wisetok --help` fails with `libpython3.X.so.1.0: cannot open shared object file` unless `LD_LIBRARY_PATH` points at the conda libdir.
- **Workaround:** `LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') ./target/release/wisetok ...`
- **Fix (deferred):** Move the pyo3 dep behind a `python` feature, gate `src/python.rs` on it, and have `[[bin]]` build with `default-features = false`. Mechanical but touches every cfg path; not worth doing mid-validation run.

## Documentation

### `RegexPreTokenizer::pattern_str` is unused
- Currently dead code (kept because HF export needs it). No action; the next commit (HF export) consumes it. Track here so a future cleanup pass doesn't delete it.

### `SpecialTokenRegistry` doc says "wiring … comes later"
- **Where:** `src/special_tokens.rs` module-level doc.
- **What:** Honest but vague. When special tokens get integrated into `train_from_iterator` and `encode`, update the doc to describe the flow.

## Test gaps

- No `SequencePreTokenizer` test with three or more steps. Two suffice to exercise composition; add a chained test once `ByteLevel` (or similar) lands.
- No round-trip test against the old i32 semantics on a corpus that would have overflowed (hard to write at small scale; defer).
- No test that `add_reserved` rejects collisions with already-added tokens. Cheap to add when integrating.

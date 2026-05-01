# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`wisetok` — a production BPE tokenizer trainer. Fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe), MIT-licensed (copyright (c) Andrej Karpathy). The intended workflow is: train with wisetok, then export to `tiktoken` (today) or HuggingFace `tokenizer.json` (planned in Iteration 1) for fast inference.

The working directory is `WiseTok` (matches the GitHub repo name `EzeLLM/WiseTok`); the Rust crate, Python module, and PyPI package are all lowercase `wisetok`.

The design and roadmap are in `Spec.md`. The audit of upstream rustbpe vs the spec is in `AUDIT_REPORT.md` — read it before changing the merge loop or claiming "rustbpe doesn't do X."

## Commands

Build and install the Python extension into the active venv (required before running Python tests):

```bash
maturin develop --release   # release-mode install
maturin develop             # debug build, faster iteration
```

Tests:

```bash
cargo test                  # Rust unit tests (currently 33, all in src/lib.rs)
pytest tests/python/ -v -s  # Python tests (require maturin develop first)
pytest tests/python/test_tokenizer.py::<name> -v -s   # single test
pytest -m "not slow"        # skip tests marked slow
```

If `cargo test` fails to find `libpython3.X.so.1.0`, the test binary needs libpython on `LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') cargo test
```

On this dev machine: `LD_LIBRARY_PATH=/home/ezel/miniconda3/lib`.

Lint (CI enforces both):

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

CI matrix runs Rust tests, then `maturin build --release` + `pip install target/wheels/*.whl` + pytest. The Python test deps include `regex tiktoken tokenizers requests`.

## Architecture (today)

The entire implementation is in `src/lib.rs` (~1070 lines): Rust BPE core + PyO3 bindings + Rust unit tests, all in one file. There is no separate `python/` package — `#[pymodule] fn wisetok` directly defines the importable module.

This will be refactored in Iteration 1 per `Spec.md` into modules: `pretokenizer/`, `aggregate/`, `merge/`, `export/`, `special_tokens.rs`, `validate/`. Until then everything stays in `lib.rs`.

### The `extension-module` feature gate

`Cargo.toml` declares `extension-module = ["pyo3/extension-module"]` as a non-default feature. This is deliberate: enabling `pyo3/extension-module` unconditionally would break `cargo test` (it stops the test binary from linking against libpython). `[tool.maturin] features = ["extension-module"]` turns it on only for Python builds. Don't move it to `default`.

### Training pipeline (`train_from_iterator`)

1. **Streaming ingest under GIL**: `refill()` pulls up to `buffer_size` strings from the Python iterator using raw `PyIter_Next` FFI.
2. **Parallel pre-tokenization without GIL**: `py.detach()` releases the GIL, then `rayon::par_iter` applies the regex (default GPT-4 pattern, see `GPT4_PATTERN` constant) and counts unique chunks per worker. Results are reduced into a global `AHashMap<CompactString, i64>`.
3. **Materialize**: each unique chunk becomes a `Word` (`Vec<u32>` of byte values 0–255); duplicates collapse into a count.
4. **Train**: `train_core_incremental` runs the merge loop.

### Merge loop (`train_core_incremental`)

This is the hot path and uses an incremental algorithm — full pair recounts only happen once at the start.

- `pair_counts: AHashMap<Pair, i64>` — global pair frequencies.
- `where_to_update: AHashMap<Pair, AHashSet<usize>>` — which word indices contain each pair.
- `OctonaryHeap<MergeJob>` (8-ary max-heap from `dary_heap`) — orders pairs by count, tie-breaks by ascending pair for determinism.
- **Lazy heap refresh**: when popping, if the stored count differs from the live `pair_counts` value, we re-push with the current count instead of repairing the heap eagerly. Stale entries with `count <= 0` are simply dropped.
- After applying a merge, `Word::merge_pair` returns local `(Pair, ±1)` deltas describing which pairs disappeared and which were created in *that one word*. Multiplying by the word's count and summing into `pair_counts` keeps the global counts exact without rescanning.
- Only newly-created pairs (`delta > 0`) are pushed to the heap — existing pairs already have an entry that the lazy refresh will fix.
- The same pair can have multiple heap entries with different `pos` sets at any time. The lazy-refresh check is what makes this safe — see `AUDIT_REPORT.md` claim #3 for the full trace.

**Do not "improve" the merge loop.** It is correct (verified against three reference implementations: slow Python, fast Python, HF tokenizers) and performance-tuned. Refactoring should be mechanical movement into modules, not behavioral change.

`merges: HashMap<Pair, u32>` is the trained output: `(left_id, right_id) -> new_id`, with `new_id` starting at 256.

### Encoding (`encode`)

For each regex chunk, repeatedly find the pair with the **lowest** `new_id` in `merges` and apply it. Lowest id == earliest learned == correct BPE order. This is O(n² log m)-ish per chunk and intentionally simple — for fast inference, export to tiktoken.

`batch_encode` releases the GIL via `py.detach()` and parallelizes with rayon.

### Export to tiktoken (`get_mergeable_ranks`)

Reconstructs token bytes incrementally: starts with 256 single-byte tokens, then walks `merges` sorted by `new_id` and concatenates the byte sequences of the two parents. The output `Vec<(Vec<u8>, u32)>` plugs directly into `tiktoken.Encoding(mergeable_ranks=...)`.

### Why these crate choices

- `dary_heap::OctonaryHeap` — wider fan-out than binary heap, faster for the push-heavy merge loop.
- `ahash` — faster non-cryptographic hashing for `Pair` keys.
- `compact_str::CompactString` — inline small strings (most regex chunks are short).
- `fancy-regex` — needed for the GPT-4 pattern's lookarounds (`std::regex` and `regex` crate don't support them).
- `pyo3-log` — forwards Rust `log::info!` calls to Python's `logging` so progress shows up in notebooks.

### Counts: i64 throughout

All count types (chunk counts, pair counts, word counts, deltas) are `i64`. `MergeJob.count` is `u64`. This is required for large corpora where common pairs (whitespace, "the", etc.) can exceed `i32::MAX` × chunk-count.

## Conventions

- All new Rust unit tests go in the `#[cfg(test)] mod tests` block at the bottom of `src/lib.rs` (until Iteration 1 splits them into `tests/test_correctness.rs`). Python integration tests (especially equality checks against minbpe / HuggingFace / tiktoken) go in `tests/python/`.
- The release profile uses `lto = true, codegen-units = 1` — release builds are slow; use debug builds for iteration.
- Pattern compilation: `train_from_iterator` always recompiles `self.compiled_pattern` from `self.pattern`. If you add code paths that mutate `pattern`, keep `compiled_pattern` in sync.

## Commit conventions

Conventional Commits format: `type(scope): subject` — `fix`, `refactor`, `feat`, `chore`, `docs`, `test`, `build`, `ci`, `perf`. **No agent / Claude / AI co-author trailers.** See `~/.claude/projects/-home-ezel-Development-WiseTok/memory/feedback_commits.md`.

# TODO

Roadmap. Each item names a concrete goal, where it lives, and the acceptance criteria. Mirrors `Spec.md` but ordered for execution. Crossed-out items are done; bracketed items are in progress.

## Iteration 1 — production features (in progress)

- [x] Drop unused `indexmap` dep
- [x] Audit report (`AUDIT_REPORT.md`)
- [x] Rename `rustbpe` → `wisetok` (Cargo, pyproject, `#[pymodule]`, README, tests)
- [x] Migrate count types `i32` → `i64` throughout
- [x] Split `lib.rs` into modules (`merge/`, `export/`, `pretokenizer/`, `python.rs`, `special_tokens.rs`, `tests.rs`)
- [x] `PreTokenizer` trait + `RegexPreTokenizer`, `DigitSplitter`, `SequencePreTokenizer`
- [x] `SpecialTokenRegistry` + `CODE_PRESET`, `CHAT_PRESET`, `add_reserved(n)`
- [x] `min_frequency` parameter on `train_from_iterator`
- [ ] **HuggingFace `tokenizer.json` export** (`src/export/huggingface.rs`)
  - Generate `tokenizer.json` matching HF's BpeTrainer output exactly
  - Generate `tokenizer_config.json`, `special_tokens_map.json`
  - Replicate the GPT-2 byte-to-unicode table for `ByteLevel`
  - Acceptance: `AutoTokenizer.from_pretrained("./out/")` loads, encode/decode matches `wisetok.encode`/`decode`
  - Acceptance: diff our `tokenizer.json` against a reference produced by HF `BpeTrainer` on the same corpus and special tokens; structural fields (vocab, merges, pre_tokenizer, decoder, added_tokens) match
- [ ] Wire pre-tokenizer pipeline into `train_from_iterator`
  - Replace the inline regex with a configurable `Box<dyn PreTokenizer>`
  - Default: `Sequence([RegexPreTokenizer(GPT4_PATTERN), DigitSplitter])`
  - Acceptance: existing tests pass; new test confirms `"v128"` trains with digits split
- [ ] Wire special tokens into aggregation + encoding
  - Aggregation: split text on every special before regex; specials never merge
  - Encoding: match specials as whole strings before BPE applies; emit pre-assigned IDs
  - Acceptance: `<|endoftext|>` mid-corpus encodes as a single ID; surrounding bytes are unchanged

## Iteration 2 — phase separation, memory, CLI

**Reorder:** put memory-bounded merge BEFORE phase separation. The merge phase, not aggregation, is where HF tokenizers OOM'd at 68GB on a 35GB corpus (see `~/.claude/projects/-home-ezel-Development-WiseTok/memory/project_merge_oom.md`). This is wisetok's real differentiator.

- [ ] **Memory-bounded merge mode** (priority — see memory note)
  - Evaluate two approaches:
    1. **Scan-based merge**: drop the global `where_to_update` map; for each merge, linearly scan all words. O(words × num_merges) time; O(words) memory.
    2. **Chunked positions**: maintain position sets only for the top-K pairs by frequency (K configurable, e.g. 100K). New pairs outside top-K trigger linear scan. Caps positions-map memory at a fixed budget.
  - Benchmark both on the 35GB corpus; pick the better tradeoff for our typical workload.
  - Acceptance: merge phase peaks under a configurable RAM budget on a 35GB corpus.
  - Acceptance: produces byte-identical merges to the unbounded mode on a small fixture corpus.
- [ ] `.agg` file format (bincode)
  - Schema: `version`, `pre_tokenizer` (serialized config), `chunks: Vec<(Vec<u8>, i64)>`, `total_bytes_processed`, `total_chunks_with_multiplicity`
  - Acceptance: aggregate-then-merge produces byte-identical tokenizer to single-pass training
  - Acceptance: change `vocab_size` and re-run merge from `.agg` without re-aggregating
- [ ] RSS monitoring thread + adaptive flush during aggregation
  - `sysinfo` crate for cross-platform RSS
  - When RSS > `ram_limit * 0.85`: log warning, sort `counts` by value, drop entries below a rising threshold
  - Acceptance: 50GB corpus aggregates on a 32GB-RAM machine without OOM
- [ ] CLI (`clap`)
  - `wisetok train` subcommand with all flags from `Spec.md` §8
  - `wisetok validate` subcommand
  - Acceptance: `wisetok train --files corpus.txt --vocab-size 1000 --output ./tok/` works end-to-end
- [ ] `indicatif` progress bars for both phases
- [ ] Training stats JSON output (unique chunks, total tokens, merge times, memory peaks)

## Iteration 3 — parquet, validation, polish

- [ ] Parquet reader (`arrow` crate, configurable column name)
  - Acceptance: `--parquet /path/to/dir/ --parquet-column content` reads parquet shards directly
- [ ] Validation suite
  - Roundtrip on a held-out corpus (encode → decode → exact match)
  - Whitespace token coverage (4-space, 8-space, tab, newline+indent patterns from `Spec.md` §9)
  - Special-token isolation
  - Vocab composition report
  - Chars/token ratio
- [ ] CLI `wisetok validate` wired to the suite
- [ ] Real-corpus test on EZeLLM-Coder data: 30GB mixed corpus, verify quality matches Phase A baseline
- [ ] Comprehensive README with usage examples
- [ ] CI: port rustbpe's GitHub Actions, add large-corpus integration test
- [ ] PyPI publishing setup (maturin)
- [ ] Profiling and hot-path optimization

## Out of scope

Per `Spec.md`:

- SentencePiece export (different format, complex protobuf — low priority)
- GPU acceleration (BPE training is inherently sequential)
- Unigram / WordPiece (BPE only)
- Online / incremental training
- Inference-speed optimization for `encode` (use tiktoken or HF for production inference)

## Cross-cutting follow-ups (from `KNOWN_ISSUES.md`)

- Replace `expect("regex match failed")` in `train_from_iterator` with logged-skip behavior matching `RegexPreTokenizer`
- Add `debug_assert!(c > 0)` above every `as u64` cast in `merge::bpe.rs`
- Decide on `min_frequency` boundary semantics (doc-only or type-tightened)

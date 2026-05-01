# WiseToken: Production BPE Trainer — Technical Specification

## Part 1: rustbpe Technical Analysis

WiseToken is a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe) (MIT, v0.1.0, Jan 2026). rustbpe is the only open-source BPE trainer that solves the large-corpus OOM problem via pre-aggregated chunk counting. We fork it rather than rewrite because the hardest part — the merge loop with lazy-refresh heap — is already correct and tested against three reference implementations (slow Python, fast Python, HuggingFace tokenizers).

### What rustbpe does right

1. **Streaming aggregation**: `train_from_iterator` streams text batches, applies regex pre-tokenization in parallel via rayon, accumulates into a global `AHashMap<CompactString, i32>` of chunk counts. This is the key innovation — instead of holding the raw corpus, it holds only unique chunks + counts. A 100GB corpus with 50M unique chunks fits in ~4GB RAM.
1. **Lazy-refresh max-heap**: The merge loop uses `OctonaryHeap` (8-ary heap for cache locality). Each `MergeJob` stores `(pair, count, positions)`. When a job is popped and its count doesn’t match the current global count, it’s re-pushed with the updated count instead of processed. This avoids rebuilding the heap after each merge — O(num_merges × log(pairs)) total.
1. **Incremental delta tracking**: `Word::merge_pair` returns per-word pair-count deltas `Vec<(Pair, i32)>` instead of recomputing all pair counts. The merge loop applies these deltas to the global `pair_counts` map and pushes affected pairs back onto the heap.
1. **Parallel initial pair counting**: `count_pairs_parallel` uses rayon to compute initial pair frequencies across all words in parallel, producing both the frequency map and a pair→word-index mapping for the merge loop.
1. **Python bindings**: Clean PyO3 interface with GIL release during both aggregation and encoding. `batch_encode` uses rayon parallelism.

### What rustbpe gets wrong or is missing

**Critical bugs / limitations:**

1. **`i32` counts overflow**: Chunk counts and pair counts are `i32` (max 2,147,483,647). On a 500GB corpus, common chunks like whitespace or `the` can exceed this. The pair count accumulation in `count_pairs_parallel` compounds the risk — if chunk “  “ appears 1M times and contains pair (’ ’, ’ ’), the pair count is 1M × chunk_count. **Fix: use `i64` for counts.**
1. **No min_frequency**: Every unique chunk participates in training regardless of frequency. On diverse code corpora, there are tens of millions of unique identifiers that appear once — they consume memory in the word table and pair index but never contribute to any merge. **Fix: add `min_frequency` parameter to `train_core_incremental` that skips words below threshold after aggregation.**
1. **No memory budgeting**: The global `counts` HashMap grows unboundedly during aggregation. If the corpus has 200M unique chunks, the HashMap alone can exceed 16GB. There’s no monitoring, no back-pressure, no adaptive flushing. **Fix: track RSS, flush low-frequency entries when approaching a configurable limit.**

**Missing production features:**

1. **No special tokens**: No way to define reserved token IDs for `<|endoftext|>`, FIM tokens, etc. Users must manually hack these in after export. Production tokenizers need special tokens defined at training time so they get contiguous IDs and are excluded from BPE splitting.
1. **No digit splitter**: The only pre-tokenizer is a single regex (GPT-4 pattern). Code tokenizers need a digit splitter (each digit 0-9 becomes its own chunk) to prevent BPE from memorizing specific numbers. StarCoder2, DeepSeek-Coder, and our EZeLLM-Coder spec all require this.
1. **No composable pre-tokenizers**: Can’t combine GPT-4 regex + digit splitter + byte-level encoding. HF tokenizers supports `Sequence([Split(...), Digits(...), ByteLevel(...)])` — we need equivalent functionality.
1. **No HuggingFace export**: Only exports to tiktoken format. The entire HuggingFace ecosystem (AutoTokenizer, transformers, tokenizers) reads `tokenizer.json`. Without HF export, users can’t use the tokenizer with standard training code.
1. **No byte-level BPE**: rustbpe operates on string chunks. Byte-level BPE (GPT-2/Llama/StarCoder2 style) uses the 256-byte alphabet as the initial vocabulary, ensuring every possible byte sequence is representable. This is mandatory for code tokenizers — binary data in string literals, UTF-8 edge cases, etc.
1. **No parquet input**: Only accepts text iterators. Our corpus is parquet files with a `content` column. Users shouldn’t need a separate extraction step.
1. **No progress reporting**: Only `log::info!` messages at 1% intervals during merges. No progress bars, no ETA, no memory stats.
1. **No phase separation**: Aggregation and training are coupled in `train_from_iterator`. Can’t save aggregated counts to disk and iterate on training parameters (vocab size, min_freq, special tokens) without re-reading the corpus.
1. **No validation**: No roundtrip testing, no whitespace token verification, no vocab composition report.

### Code structure (current)

```
src/lib.rs  — single 600-line file containing everything:
  - Word struct (symbol array + merge_pair with delta tracking)
  - MergeJob struct (heap entry with lazy refresh)
  - count_pairs_parallel() (rayon-parallel initial pair counting)
  - Tokenizer struct:
    - train_from_iterator() (streaming aggregation + calls train_core_incremental)
    - train_core_incremental() (the merge loop)
    - encode() (O(n²) per chunk, fine for validation, not production inference)
    - decode()
    - batch_encode() (rayon parallel)
    - get_mergeable_ranks() (tiktoken export)
  - PyO3 module definition
  - Tests (14 Rust tests)
```

-----

## Part 2: WiseToken Implementation Specification

Fork rustbpe. Keep the proven core (Word, MergeJob, merge loop, lazy-refresh heap). Refactor into modules. Add everything listed below.

### Project structure (target)

```
wisetoken/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs                  # Crate root, re-exports
│   ├── python.rs               # PyO3 module + bindings
│   ├── main.rs                 # CLI entry point (clap)
│   ├── pretokenizer/
│   │   ├── mod.rs              # PreTokenizer trait
│   │   ├── regex.rs            # GPT-2 / GPT-4 regex splitter
│   │   ├── digits.rs           # Individual digit splitter
│   │   ├── byte_level.rs       # Byte-level encoding (GPT-2 style byte remapping)
│   │   └── sequence.rs         # Compose multiple pre-tokenizers
│   ├── aggregate/
│   │   ├── mod.rs              # Phase 1 entry point
│   │   ├── counter.rs          # Streaming chunk counter with rayon
│   │   ├── memory.rs           # RSS monitoring + adaptive flush
│   │   └── format.rs           # .agg binary file read/write (bincode)
│   ├── merge/
│   │   ├── mod.rs              # Phase 2 entry point
│   │   ├── bpe.rs              # Core merge loop (from rustbpe, refactored)
│   │   ├── word.rs             # Word struct + merge_pair (from rustbpe)
│   │   └── heap.rs             # MergeJob + heap logic (from rustbpe)
│   ├── special_tokens.rs       # Special token registry
│   ├── export/
│   │   ├── mod.rs
│   │   ├── huggingface.rs      # tokenizer.json generation
│   │   └── tiktoken.rs         # Tiktoken export (from rustbpe)
│   └── validate/
│       ├── mod.rs
│       ├── roundtrip.rs
│       └── report.rs
├── tests/
│   ├── test_correctness.rs     # Port rustbpe's Rust tests
│   └── python/
│       └── test_wisetoken.py   # Port rustbpe's Python tests, add new ones
└── benches/
    └── bench_aggregate.rs
```

### Implementation details

#### 1. Count type: `i64` everywhere

Replace all `i32` counts with `i64`. This applies to:

- `AHashMap<CompactString, i64>` in aggregation
- `Vec<i64>` for word counts passed to `train_core_incremental`
- `AHashMap<Pair, i64>` for pair counts
- `MergeJob.count: u64` (already correct in rustbpe)

#### 2. Pre-tokenizer system

```rust
pub trait PreTokenizer: Send + Sync {
    /// Split input text into chunks. Returns iterator of string slices.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;
}

pub struct RegexPreTokenizer { pattern: Regex }
pub struct DigitSplitter;  // splits each digit 0-9 individually
pub struct SequencePreTokenizer { steps: Vec<Box<dyn PreTokenizer>> }
```

The `SequencePreTokenizer` applies each step in order — first step splits the text, subsequent steps split each chunk further. This matches HF tokenizers’ `Sequence` behavior.

Default for code: `Sequence([RegexPreTokenizer(GPT4_PATTERN), DigitSplitter])`

**Note on ByteLevel**: HF tokenizers’ `ByteLevel` pre-tokenizer does two things: (1) applies a regex split, (2) remaps each byte to a printable Unicode character (the GPT-2 byte-to-unicode mapping). For WiseToken, we handle this differently: our initial alphabet is always the 256 raw bytes (0x00-0xFF), and we store chunks as `Vec<u8>` not strings. This is simpler and avoids the confusing Ġ/Ċ byte remapping. The byte-level property comes from the initial alphabet, not the pre-tokenizer.

#### 3. Special tokens

```rust
pub struct SpecialTokenRegistry {
    tokens: Vec<(String, u32)>,  // (token_string, token_id)
}
```

Special tokens are assigned IDs starting at `256 + num_merges`. They are:

- Excluded from BPE merging (the aggregation phase strips them or treats them as chunk boundaries)
- Matched as whole strings during encoding (before BPE is applied)
- Written to the export format with `"special": true`

Provide presets:

```rust
pub fn code_preset() -> Vec<String> {
    vec![
        "<|endoftext|>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|file_sep|>", "<|repo_name|>", "<|filename|>",
    ]
}
```

Plus a `--reserve N` flag that adds N placeholder tokens `<|reserved_0|>` through `<|reserved_{N-1}|>`.

#### 4. Phase separation (aggregate file format)

Phase 1 (aggregate) produces a `.agg` file:

```rust
#[derive(Serialize, Deserialize)]
pub struct AggregateFile {
    pub version: u32,                     // format version
    pub pre_tokenizer: String,            // serialized pre-tokenizer config
    pub chunks: Vec<(Vec<u8>, i64)>,      // (chunk_bytes, count), sorted by count desc
    pub total_bytes_processed: u64,       // for stats
    pub total_chunks_with_multiplicity: u64,
}
```

Serialized with bincode for compact binary format. This lets users:

- Run Phase 1 once (slow, I/O-bound) on a huge corpus
- Run Phase 2 many times (fast, CPU-bound) with different vocab sizes, min_freq, special tokens
- Share `.agg` files for reproducibility

#### 5. min_frequency

After aggregation (or after loading a `.agg` file), filter chunks:

```rust
let chunks: Vec<(Vec<u8>, i64)> = chunks
    .into_iter()
    .filter(|(_, count)| *count >= min_frequency)
    .collect();
```

This happens before `train_core_incremental`, reducing the word table size.

#### 6. Memory monitoring

Spawn a background thread that reads `/proc/self/status` (Linux) or equivalent every 5 seconds:

```rust
fn get_rss_bytes() -> u64 {
    // Read VmRSS from /proc/self/status
    // Or use sysinfo crate for cross-platform
}
```

During aggregation, if RSS exceeds `ram_limit * 0.85`:

1. Log a warning
1. Sort the counts HashMap by value
1. Drop entries below a threshold (start at count=1, increase if needed)
1. Continue aggregation

This is adaptive back-pressure — the tokenizer degrades gracefully instead of OOM-killing.

#### 7. HuggingFace export

Generate a `tokenizer.json` compatible with `AutoTokenizer.from_pretrained()`. The format is:

```json
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": { "<byte_token_0>": 0, ..., "<merged_token>": 256, ... },
    "merges": [ "token_a token_b", ... ]
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      { "type": "Split", "pattern": { "Regex": "..." }, "behavior": "Isolated" },
      { "type": "Digits", "individual_digits": true },
      { "type": "ByteLevel", "add_prefix_space": false, "use_regex": false }
    ]
  },
  "decoder": { "type": "ByteLevel" },
  "added_tokens": [
    { "id": N, "content": "<|endoftext|>", "special": true, ... },
    ...
  ]
}
```

Also generate `tokenizer_config.json`, `special_tokens_map.json`, and `vocab.json` for full HF compatibility.

Study the actual tokenizer.json output from HuggingFace’s `BpeTrainer` to get the exact format right — load a trained HF tokenizer and inspect its saved files. The ByteLevel pre-tokenizer uses a specific byte-to-unicode mapping (the GPT-2 table) that must be replicated exactly.

#### 8. CLI (clap)

```
wisetoken train [OPTIONS]

OPTIONS:
    --files <FILES>...              Input text files
    --parquet <DIR>                 Input parquet directory
    --parquet-column <NAME>         Text column name [default: content]
    --vocab-size <N>                Target vocabulary size (required)
    --min-freq <N>                  Minimum chunk frequency [default: 2]
    --pre-tokenizer <SPEC>          Pre-tokenizer spec [default: gpt4+digits]
    --special-tokens <TOKENS>...    Special token strings
    --special-preset <PRESET>       Special token preset [code, chat]
    --reserve <N>                   Number of reserved token slots [default: 0]
    --ram-limit <SIZE>              RAM budget (e.g., "64GB") [default: 80% of system RAM]
    --threads <N>                   Worker threads [default: 80% of cores]
    --format <FORMATS>              Export formats (comma-separated) [default: hf,tiktoken]
    --output <DIR>                  Output directory (required)
    --agg-file <PATH>               Save/load aggregate file (skip Phase 1 if exists)

wisetoken validate [OPTIONS]
    --tokenizer <DIR>               Path to trained tokenizer
    --test-files <FILES>...         Test files for roundtrip/compression checks
    --check <CHECKS>                Checks to run [default: roundtrip,whitespace,special,composition]
```

#### 9. Validation suite

Built into the library, also exposed via CLI:

```rust
pub struct ValidationReport {
    pub roundtrip_pass: usize,
    pub roundtrip_fail: usize,
    pub failed_samples: Vec<String>,
    pub whitespace_tokens: HashMap<String, Option<u32>>,  // pattern -> token_id or None
    pub special_token_ids: HashMap<String, u32>,
    pub vocab_composition: VocabComposition,
    pub chars_per_token: f64,  // on test data
}
```

Whitespace patterns to check: `"    "` (4sp), `"        "` (8sp), `"\t"`, `"\t\t"`, `"\n    "`, `"\n        "`, `"\n\t"`.

#### 10. Parquet reader

Use the `arrow` crate’s `parquet` module to read `.parquet` files directly:

```rust
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn read_parquet_texts(path: &Path, column: &str) -> impl Iterator<Item = String> {
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    reader.flat_map(move |batch| {
        let batch = batch.unwrap();
        let col = batch.column_by_name(column).unwrap();
        let array = col.as_any().downcast_ref::<StringArray>().unwrap();
        (0..array.len()).map(move |i| array.value(i).to_string()).collect::<Vec<_>>()
    })
}
```

### Dependencies

```toml
[dependencies]
pyo3 = { version = "0.22", optional = true }
dary_heap = "0.3"
fancy-regex = "0.16"
ahash = "0.8"
rayon = "1.11"
compact_str = "0.9"
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
indicatif = "0.17"
sysinfo = "0.32"
log = "0.4"
env_logger = "0.11"
parquet = { version = "53", features = ["arrow"], optional = true }
arrow = { version = "53", optional = true }

[features]
default = ["python", "parquet_support"]
python = ["pyo3/extension-module"]
parquet_support = ["parquet", "arrow"]
```

### Migration from rustbpe

1. `Word` struct: keep as-is, change count types to i64
1. `MergeJob`: keep as-is (already uses u64 for count)
1. `count_pairs_parallel`: keep, change i32 → i64
1. `train_core_incremental`: keep, add min_frequency filtering before the merge loop, add progress bar via indicatif
1. `train_from_iterator` aggregation loop: extract into `aggregate/counter.rs`, change i32 → i64, add memory monitoring
1. `encode/decode`: keep for validation, note these aren’t production-speed (use tiktoken for inference)
1. `get_mergeable_ranks`: keep, also generate HF format
1. All 14 Rust tests: port and extend
1. Python test suite: port, add new tests for special tokens, digit splitter, HF export

### Testing strategy

1. **Correctness against rustbpe**: Train both on same small corpus, verify identical merges
1. **Correctness against HF tokenizers**: Same as rustbpe’s existing test (custom_match for byte order)
1. **Roundtrip on code**: Encode→decode 1000 Python files from Stack v2, verify exact match
1. **Special token isolation**: Verify `<|endoftext|>` in middle of text doesn’t get BPE-merged
1. **Digit splitter**: Verify `"128"` encodes as 3 tokens, not 1
1. **HF export loading**: Save tokenizer.json, load with `AutoTokenizer.from_pretrained()`, verify encode/decode match
1. **Phase separation**: Aggregate → save .agg → load .agg → merge, verify identical to single-pass
1. **Large corpus smoke test**: 10GB+ corpus, verify no OOM, verify memory stays under budget

### What NOT to build

- **SentencePiece export**: Low priority, different format, complex protobuf
- **GPU acceleration**: BPE training is inherently sequential (each merge depends on previous). Rayon for Phase 1 parallelism is sufficient.
- **Unigram/WordPiece**: Out of scope. BPE only.
- **Online/incremental training**: Out of scope. Retrain from scratch.
- **Tokenizer inference optimization**: Use tiktoken or HF tokenizers for production inference. WiseToken’s encode() is for validation only.

-----

## Part 3: Implementation Order

All three iterations build on each other. Complete each fully before moving to the next.

### Iteration 1: Core refactor + production features

**Goal**: Refactored rustbpe that passes all existing tests + adds special tokens, digit splitter, i64 counts, min_frequency, HF export.

1. Fork rustbpe, create module structure (pretokenizer/, aggregate/, merge/, export/)
1. Move `Word`, `MergeJob`, merge loop, pair counting into respective modules
1. Change all `i32` counts to `i64`
1. Implement `PreTokenizer` trait with `RegexPreTokenizer`, `DigitSplitter`, `SequencePreTokenizer`
1. Implement `SpecialTokenRegistry` with code preset
1. Add `min_frequency` parameter to merge phase
1. Implement HuggingFace `tokenizer.json` export — study HF’s actual format carefully
1. Port all 14 Rust tests + Python test suite
1. Add new tests for digit splitter, special tokens, HF export round-trip
1. Verify: load exported tokenizer.json with `AutoTokenizer.from_pretrained()`, encode same text, compare

**Acceptance criteria**: `cargo test` passes all original rustbpe tests + new feature tests. Python `pytest` passes. HF export loads correctly in transformers.

### Iteration 2: Phase separation + memory management + CLI

**Goal**: Aggregate and merge phases are separable. RAM-bounded. CLI works.

1. Implement `.agg` file format with bincode serialization
1. Extract aggregation into standalone function that writes `.agg`
1. Implement merge-from-agg that reads `.agg` and trains
1. Verify: aggregate-then-merge produces identical results to single-pass
1. Implement RSS monitoring thread using `sysinfo`
1. Implement adaptive flush in aggregation (drop low-freq entries when approaching RAM limit)
1. Implement CLI with clap (train, validate subcommands)
1. Add indicatif progress bars for both phases
1. Add training stats JSON output (unique chunks, total tokens, merge times, memory peaks)

**Acceptance criteria**: Can aggregate a 50GB corpus on a 32GB RAM machine without OOM. Can re-run merge phase with different vocab sizes without re-aggregating. CLI works end-to-end.

### Iteration 3: Parquet support + validation suite + polish

**Goal**: Production-ready tool. Reads parquet directly. Full validation. Ready to publish.

1. Implement parquet reader (arrow crate) with configurable column name
1. Implement validation suite (roundtrip, whitespace, special tokens, vocab composition, chars/tok)
1. Add validation to CLI (`wisetoken validate`)
1. Test on real EZeLLM-Coder data: 30GB mixed corpus, verify tokenizer quality matches Phase A
1. Write comprehensive README with usage examples
1. Set up CI (port rustbpe’s GitHub Actions, add large-corpus integration test)
1. Set up maturin build for PyPI publishing
1. Profile and optimize hot paths (aggregate throughput, merge phase memory)

**Acceptance criteria**: `wisetoken train --parquet /data/stackv2/ --parquet-column content --vocab-size 24576 --special-preset code --reserve 16 --format hf,tiktoken --output ./tokenizer/` works end-to-end. Validation passes. README is complete.

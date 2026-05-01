# wisetok

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production BPE tokenizer trainer for LLMs

`wisetok` is a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe) (MIT, copyright (c) Andrej Karpathy). It keeps rustbpe's proven core — streaming chunk aggregation, the lazy-refresh max-heap merge loop, byte-level BPE on the 256-byte alphabet, parallel pre-tokenization with rayon, and Python bindings via PyO3 — and adds the production features needed to train tokenizers for serious code LLMs.

The project is in active development. Today the public API matches upstream rustbpe (with package and module renamed to `wisetok`). The new features (digit splitter, special tokens, `min_frequency`, HuggingFace export, `.agg` phase separation, RAM-bounded aggregation, parquet input, validation suite, CLI) are being added in iterations.

## Features (today)

- Fast training with parallel pre-tokenization (rayon)
- GPT-4-style regex splitter by default; custom regex supported
- Byte-level BPE: every input byte 0x00–0xFF is in the initial vocabulary
- Direct export to `tiktoken` format for fast inference
- Python bindings (PyO3 0.27) with proper GIL release in hot paths
- Parallel batch encode

## Features (planned)

- Composable pre-tokenizers (regex + digit splitter, etc.)
- Special tokens with reserved IDs (e.g., `<|endoftext|>`, FIM tokens)
- `min_frequency` cutoff to drop rare chunks before merging
- HuggingFace `tokenizer.json` export for `AutoTokenizer.from_pretrained`
- Phase separation via `.agg` files (aggregate once, train many)
- RSS-bounded streaming aggregation with adaptive flush
- CLI (`wisetok train`, `wisetok validate`)
- Parquet input via the `arrow` crate
- Validation suite (roundtrip, whitespace coverage, vocab composition)

See `Spec.md` for the full design and `AUDIT_REPORT.md` for the gap analysis against upstream rustbpe.

## Installation

### From source

```bash
git clone https://github.com/EzeLLM/WiseTok.git
cd WiseTok
uv venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

## Usage

### Training

```python
import wisetok

tokenizer = wisetok.Tokenizer()
tokenizer.train_from_iterator(
    ["your", "training", "texts", "here"],
    vocab_size=4096,
)

ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)
print(tokenizer.vocab_size)  # 4096

all_ids = tokenizer.batch_encode(["text one", "text two", "text three"])
```

### Export to tiktoken

```python
import wisetok
import tiktoken

tokenizer = wisetok.Tokenizer()
tokenizer.train_from_iterator(open("corpus.txt"), vocab_size=8192)

enc = tiktoken.Encoding(
    name="my_tokenizer",
    pat_str=tokenizer.get_pattern(),
    mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
    special_tokens={},
)

ids = enc.encode("hello world")
```

### Custom regex pattern

```python
tokenizer.train_from_iterator(
    texts,
    vocab_size=4096,
    pattern=r"[a-zA-Z]+|[0-9]+|\s+",
)
```

## API reference (Tokenizer)

| Method | Description |
|--------|-------------|
| `Tokenizer()` | Create a new tokenizer |
| `train_from_iterator(texts, vocab_size, buffer_size=8192, pattern=None)` | Train on an iterator of strings |
| `encode(text)` | Encode a string to token IDs |
| `decode(ids)` | Decode token IDs back to a string |
| `batch_encode(texts)` | Encode multiple strings in parallel |
| `vocab_size` | Property: 256 + number of merges |
| `get_pattern()` | Regex pattern used for pre-tokenization |
| `get_mergeable_ranks()` | Token bytes and ranks for tiktoken export |

## Development

```bash
cargo test                  # Rust tests (33 today)
pytest tests/python/ -v -s  # Python tests (requires `maturin develop` first)
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

If `cargo test` fails to find `libpython3.X.so.1.0`, set `LD_LIBRARY_PATH` to your Python lib dir:

```bash
LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') cargo test
```

## How BPE works

1. Start with 256 byte-level tokens (0x00–0xff).
2. Count all adjacent token pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until reaching the target vocabulary size.

## Attribution

`wisetok` is a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe), MIT-licensed, copyright (c) Andrej Karpathy. The merge loop, lazy-refresh heap, and parallel pre-tokenization are unchanged from upstream. See `LICENSE` for the original MIT terms.

## License

MIT

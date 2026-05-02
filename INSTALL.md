# Install WiseTok

Three install paths, depending on what you want.

## 1. Python — `pip install wisetok`

The fastest path if you want the Python module *and* the `wisetok` CLI binary.

```bash
pip install wisetok
```

Wheels are published for **Linux x86_64**, **Linux aarch64**, **macOS arm64**, **macOS x86_64**, and **Windows x86_64**, across **CPython 3.9 – 3.13**.

The wheel ships:
- `wisetok` Python module (`import wisetok`)
- `wisetok` CLI binary (on `$PATH` after install)

After install:

```bash
wisetok --help
python -c "import wisetok; print(wisetok.__version__)"
```

## 2. Rust — `cargo install wisetok`

If you only want the CLI binary, or you want WiseTok as a Rust library dependency:

```bash
# CLI
cargo install wisetok

# As a library in your own crate
cargo add wisetok
```

The crate exposes the same training pipeline used by the CLI:

```rust
use wisetok::aggregate::aggregate_into_counts_rust;
use wisetok::cli_core::{materialize_and_train_with_progress, parse_merge_mode};
```

See `src/lib.rs` for the public API.

## 3. From source

You'll need:

- **Rust** ≥ 1.74 (`rustup install stable`)
- **Python** 3.9+ (only required for the Python module)
- **`maturin`** (`pip install maturin` — only required for the Python module)

### Clone and build

```bash
git clone https://github.com/EzeLLM/WiseTok && cd WiseTok

# Python module (release-mode, into your active venv)
maturin develop --release

# CLI binary only (no Python required)
cargo build --release --bin wisetok
# → target/release/wisetok
```

### Run the test suite

```bash
# Rust unit tests
cargo test

# Python integration tests (require maturin develop first)
pytest tests/python/ -v
```

If `cargo test` fails to find `libpython3.X.so.1.0` on Linux, point `LD_LIBRARY_PATH` at your interpreter's libdir:

```bash
LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') cargo test
```

### Lint

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

## Minimum hardware

You need enough RAM for **the unique pre-token table**, not the corpus.

Rule of thumb: about **300 MB per 1M unique pre-tokens** during the merge phase, plus working set. On a 100M-unique-pre-token corpus, expect ~30 GB peak RSS. On a 25M corpus, expect ~7 GB. Disk requirements are corpus-size + a small `.agg` file (roughly 5–10% of corpus size).

A 16 GB MacBook will train a tokenizer on a 30 GB corpus. A 96 GB workstation will train one on 150 GB. There is no GPU support and none is needed — BPE training is CPU-bound.

## Troubleshooting

**`pip install wisetok` fails to build a wheel.** You're on a platform we don't ship a prebuilt wheel for. Use the from-source path above (`maturin develop --release`).

**`cargo install wisetok` builds but the binary is slow.** Make sure you didn't accidentally use a debug profile. `cargo install` defaults to release; if you cloned and ran `cargo build` directly, add `--release`.

**Training is slower than expected on a tiny corpus.** `--merge-mode auto` switches to scan mode above ~1M unique pre-tokens. On small corpora, force `--merge-mode full` to keep the indexed fast path.

**Aggregation phase looks stuck.** Pass `--verbose` to see the streaming-bytes progress bar; you should see throughput in MB/s. If you're reading from a slow filesystem (network mount, archived disk), pre-stage the corpus to local SSD.

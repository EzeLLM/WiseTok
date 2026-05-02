# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2024-11-15

### Added
- `TokenizerTrainer.train_from_iterator()` now supports custom `buffer_size` parameter ([#487](https://github.com/openai/tokenlib/pull/487)).
- Python type hints for all public APIs ([#502](https://github.com/openai/tokenlib/issues/502)).
- New `--verbose` flag for CLI `train` subcommand to report merge progress every 100 iterations.
- Support for Rust `1.75+` (bumped MSRV from `1.70`).

### Changed
- `Tokenizer.decode()` now handles out-of-range token IDs gracefully (returns replacement character instead of panicking) ([#491](https://github.com/openai/tokenlib/pull/491)).
- Internal: Refactored merge loop to use tighter incremental pair counting (performance improvement on large vocabs, ~5% faster in benchmarks).

### Fixed
- Fixed panic when encoding with `special_tokens` containing null bytes ([#498](https://github.com/openai/tokenlib/issues/498)).
- Corrected memory leak in batch encoding on repeated calls (GIL not properly released on exception) ([#504](https://github.com/openai/tokenlib/pull/504)).
- CLI `encode` subcommand now correctly handles UTF-16 surrogate pairs in input.

## [0.8.0] - 2024-10-22

### Added
- **TypeScript / Node.js support** — first release of `@tokenlib/core` npm package ([#450](https://github.com/openai/tokenlib/pull/450)).
- `merge_tokenizers()` function to combine vocabularies from multiple tokenizers ([#475](https://github.com/openai/tokenlib/pull/475)).
- `get_token_stats()` function for entropy and frequency analysis.
- Pretrained tokenizers for `claude-3-opus` and `claude-3-sonnet` (via new remote loader).
- Parallel `encode_batch()` and `decode_batch()` methods (auto-release GIL in CPython).

### Changed
- **Breaking**: `Tokenizer.from_dict()` renamed to `Tokenizer.from_vocab()` for clarity.
- **Breaking**: `special_tokens` parameter now expects `Dict[str, int]` instead of `List[str]` (must map token name to reserved ID).
- Improved error messages for invalid JSON tokenizer files.
- Internal: Switched from `dary_heap` to `parking_lot::Mutex` for heap access (no functional change, reduces lock contention).

### Deprecated
- `Tokenizer.from_config()` — use `Tokenizer.load()` instead (will remove in 0.9).

### Fixed
- Fixed off-by-one error in merge rule application for vocab IDs >= 65536 ([#464](https://github.com/openai/tokenlib/issues/464)).
- Resolved crash when training on empty lines in corpus ([#468](https://github.com/openai/tokenlib/pull/468)).

## [0.7.2] - 2024-09-10

### Added
- Progress bar support via `log::info!()` in training (shows via `pyo3-log` in Jupyter notebooks).
- Pretrained tokenizers for `gpt-4-turbo` and `llama-2-7b`.

### Fixed
- Corrected pattern for GPT-4 regex (was missing `\s` in lookahead, affecting whitespace handling).
- Memory-safe handling of Python iterators with explicit lifetime management.

## [0.7.1] - 2024-08-28

### Fixed
- Performance regression in `encode()` for large texts (reverted overly aggressive inlining that disabled simd).
- Build error on Windows (MSVC missing `regex` crate's `onig` backend).

## [0.7.0] - 2024-08-15

### Added
- Full support for custom pretokenization patterns (regex with lookaround assertions via `fancy-regex`).
- `Tokenizer.save(path)` and `Tokenizer.load(path)` for JSON serialization.
- `get_vocab_size()` method.
- Cargo feature gate: `extension-module = ["pyo3/extension-module"]` to support both library and binary builds.

### Changed
- Internal representation of pairs now uses `(u32, u32)` instead of `(u8, u8)` to support larger vocabs (>65k tokens).
- Minimum Python version bumped from `3.7` to `3.8`.

## [0.6.0] - 2024-07-01

### Added
- Python bindings for `Tokenizer` and `TokenizerTrainer` classes.
- CLI binary (`tokenlib-cli`) with `encode`, `decode`, `train` subcommands.
- Pretrained models for `gpt-3.5-turbo` and `gpt-4`.

### Fixed
- Deterministic merge order (pairs with same count now tie-break by `(left_id, right_id)` for reproducibility).

## [0.5.0] - 2024-05-20

### Added
- Core BPE tokenizer implementation in Rust.
- Incremental merge loop with lazy heap refresh.
- Support for `min_frequency` threshold.

[Unreleased]: https://github.com/openai/tokenlib/compare/0.8.1...HEAD
[0.8.1]: https://github.com/openai/tokenlib/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/openai/tokenlib/compare/0.7.2...0.8.0
[0.7.2]: https://github.com/openai/tokenlib/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/openai/tokenlib/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/openai/tokenlib/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/openai/tokenlib/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/openai/tokenlib/releases/tag/0.5.0

# `tokenizer.json` Format Drift Across `huggingface/tokenizers` Releases

Scope: last ~10 releases of `huggingface/tokenizers`, format-relevant changes only.
Sources: GitHub release notes via `gh release view`, raw source on `raw.githubusercontent.com`, and linked PRs.

## Release table

| Version | Date | Format-relevant changes |
|---|---|---|
| v0.23.1 | 2026-04-27 | `add_tokens` now normalizes `content` at insertion (#1995). Re-saved files may differ in the `added_tokens` block; old files still load. No schema additions. |
| v0.22.2 | 2025-12-02 | None (deserialize perf, PyO3 bump). `added_tokens` deser update (#1891) is internal. |
| v0.22.1 | 2025-09-19 | None (docs, hub bump). |
| v0.22.0 | 2025-08-29 | None. AHashMap, async bindings, paste->pastey. No `tokenizer.json` schema change. |
| v0.21.4 | 2025-07-28 | Re-release of v0.21.3, identical. |
| v0.21.3 | 2025-07-04 | None. Rust API revert only. |
| v0.21.2 | 2025-06-24 | **Breaking on training output**: "Fix training with special tokens" (#1617) — special-token handling during *training* changed; the saved JSON schema didn't, but the values written for trained tokenizers may differ. Fixed Length pre-tokenizer added (#1713) — new optional pre-tokenizer variant in the schema. |
| v0.21.1 | 2025-03-13 | Template-processor update (#1652) was reverted in this release for compat. No net format change. |
| v0.21.0 | 2024-11-15 | Decode-stream API; format unchanged. |
| v0.20.3 | 2024-11-05 | None (Python tuple-input fix). |
| v0.20.2 | 2024-11-04 | None (PyO3 0.22). |
| v0.20.1 | 2024-10-10 | `ignore_merges` offset fix (#1640). Field already shipped in 0.19.1. |
| v0.20.0 | 2024-08-08 | **Format-relevant**: BPE `merges` written as `[["a","b"], ...]` tuples instead of legacy `["a b", ...]` strings (see lookup #3 below). Bytelevel normalizer auto-added on BPE add-token decode (#1555). `__repr__`/`__str__` for objects (display only). |
| v0.19.1 | 2024-04-17 | **`ignore_merges` serialized** for the first time (#1504). |

## Specific lookups

**1. `byte_fallback` on BPE.** Added in v0.13.3 by PR [#1183](https://github.com/huggingface/tokenizers/pull/1183) (merged 2023-03-23). The field is *always* written by the serializer (`model.serialize_field("byte_fallback", &self.byte_fallback)`); on deserialize, missing-field defaults to `false` via `BpeBuilder::default()`, so pre-0.13.3 JSON loads cleanly on every modern release. Unigram got its own bytefallback later in PR #1217 (v0.13.4, June 2023).

**2. `ignore_merges` on BPE.** Added by PR [#1504](https://github.com/huggingface/tokenizers/pull/1504), shipped in **v0.19.1** (2024-04-17). Always written from v0.19.1 onward; missing-field defaults to `false` on older readers. Loaders < v0.19.1 *will refuse the file* if the deserializer is strict about unknown keys — see Risk #1.

**3. `merges` array shape.** Yes, this changed. v0.19.1 and earlier wrote `"merges":["a b", ...]` (space-joined). **v0.20.0** flipped the writer to tuple form `"merges":[["a","b"], ...]` while keeping the deserializer untagged-enum (`Tuple` | `Legacy`) so old files still load. A reader on tokenizers < 0.20.0 cannot parse the tuple form. Source: `tokenizers/src/models/bpe/serialization.rs` diff between v0.19.1 and v0.20.0.

**4. `Split.behavior` casing.** **No change.** PascalCase since v0.13.x: `"Isolated"`, `"Removed"`, `"Merged_With_Previous"`, `"Merged_With_Next"`, `"Contiguous"`. Confirmed in serialization tests on both v0.13.3 and v0.23.1.

**5. `pre_tokenizer.Sequence` field name.** **No change.** The inner list key is `"pretokenizers"` (one word, lowercase) on every release checked, e.g. `{"type":"Sequence","pretokenizers":[...]}`. The struct field of the same name in `pre_tokenizers/sequence.rs` has been stable since 0.13.

**6. Normalizer / "no normalizer" shape.** Stable: `"normalizer": null` is canonical when absent. The `Normalizer` enum is `#[serde(tag = "type")]` and has had additions (Prepend, Replace variants over time) but no renames or removals on the byte-level path. Wisetok writes `null` and is safe across the entire 0.13–0.23 range.

## `transformers` compatibility floor

`transformers` master `setup.py` pins `tokenizers>=0.22.0,<=0.23.0`. `PreTrainedTokenizerFast` itself contains no version asserts and no shape-conditional branches on the `tokenizers` version — it relies on the underlying Rust deserializer for compat. So the practical floor for "load on a fresh modern transformers install" is **tokenizers 0.20.0** (tuple-merges era).

## Bottom-line recommendation for wisetok

- **Compatibility floor to declare**: `tokenizers >= 0.20.0`. This is the first release that writes the tuple-merges form natively, ships `byte_fallback` and `ignore_merges` on BPE, and matches what every supported `transformers` (≥4.45) actually has on disk.
- **Format to write**: target the v0.23.1 schema (BPE with `byte_fallback: false`, `ignore_merges: false`, tuple-form merges, `pretokenizers` key, PascalCase `behavior`, `"normalizer": null`). Do not include `version` strings beyond `"1.0"` — that key has not changed since 0.13.
- **Test matrix**: load wisetok-emitted `tokenizer.json` on tokenizers 0.20.0, 0.21.4, 0.22.2, and 0.23.1.

## Risks (where wisetok could write something valid in N but not N±2)

1. **`ignore_merges` on pre-0.19.1 readers.** A consumer pinned to old `tokenizers` (e.g. legacy Colab kernels with `tokenizers==0.15`) will fail on the unknown key. Low impact — outside our declared floor — but worth a one-liner in wisetok docs.
2. **Tuple-form merges on pre-0.20.0 readers.** Same situation. Below our floor; mention in docs.
3. **`Split.behavior` typos.** Easy to write `"isolated"` (lowercase) by accident and *every* version will reject it. PascalCase is mandatory.
4. **`pretokenizers` vs `tokenizers` key.** No drift across versions, but a tempting bug — the post-processor `Sequence` uses a different key (`"processors"`) and the normalizer `Sequence` uses `"normalizers"`. Don't cross-wire.
5. **Future `merges` shape.** Maintainers have not signaled a v3 format, but the `MergeType` untagged-enum infrastructure means a third variant could land without a major bump. Pin `tokenizers <= 0.23.x` in test deps and re-audit on each new minor.

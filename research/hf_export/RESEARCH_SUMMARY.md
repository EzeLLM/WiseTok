# HF export research — summary and open decisions

Three parallel agents researched HuggingFace's `tokenizer.json` format. Their full reports and the captured reference files live in this directory:

- `reference/` — actual `tokenizer.json` files produced by HF's own `BpeTrainer`, plus the GPT-2 byte-to-unicode table and the capture script.
- `HF_TOKENIZERS_MAP.md` — file/line map of the HF Rust serde shapes (commit `22d54d3`, crate `0.23.2-dev.0`).
- `HF_FORMAT_VERSION_NOTES.md` — version-drift survey across recent `tokenizers` releases.

## What we learned (locked in)

- **Floor: `tokenizers >= 0.20.0`.** Write the v0.23.x schema. `transformers` master pins `tokenizers >= 0.22.0`, so anyone using `AutoTokenizer.from_pretrained()` gets a compatible reader.
- **Format-version constant**: top-level `"version": "1.0"` (`tokenizers/src/tokenizer/serialization.rs:13`).
- **`merges` is `Vec<[String, String]>`** — two-element arrays of GPT-2-mapped Unicode strings, NOT space-joined `"a b"`. Format flipped at v0.20.0; old reads still tolerated, new reads only accept tuples on strict mode.
- **All 9 BPE model fields always emitted in fixed order**: `type`, `dropout`, `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `fuse_unk`, `byte_fallback`, `ignore_merges`, `vocab`, `merges`. Including `null` / `false` defaults.
- **`vocab` ordering**: ascending ID via `OrderedVocabIter` (`models/mod.rs:32-58`). Match it.
- **`Sequence` field name is container-specific**:
  - Pre-tokenizer Sequence: `"pretokenizers"`
  - Decoder Sequence: `"decoders"`
  - Normalizer Sequence: `"normalizers"`
  Cross-using fails to deserialize.
- **`ByteLevel` pre-tokenizer vs decoder share the same Rust struct, different defaults in our config**:
  - Pre-tokenizer: `add_prefix_space=false, trim_offsets=true, use_regex=false`
  - Decoder: `add_prefix_space=true, trim_offsets=true, use_regex=true`
- **`Split.behavior`** serializes as PascalCase: `"Isolated"`, not `"isolated"`.
- **`pattern`** is the tagged-union form: `{"Regex": "..."}` or `{"String": "..."}`.
- **BPE decoder JSON `type` is `"BPEDecoder"`**, not `"BPE"` — the only place where the model name and decoder name diverge. We don't use this decoder, but worth knowing.
- **"No normalizer" is `"normalizer": null`**, not omitted.
- **Target raw `Tokenizer.save()` output, NOT `PreTrainedTokenizerFast.save_pretrained`**. The latter injects a `TemplateProcessing` post-processor we don't want.
- **`tokenizer_config.json` in `transformers` 5.x is minimal**: just `{"backend": "tokenizers", "model_max_length": 1e30, "tokenizer_class": "TokenizersBackend"}`. No legacy keys.
- **The GPT-2 byte-to-unicode table is captured** at `reference/byte_to_unicode.rs` as `pub const BYTE_TO_UNICODE: [&str; 256]`. Drop straight into `wisetok::export::huggingface`.

## Open decision: special-token ID layout

This is the only finding that contradicts our spec, and it's a hard fork in the road.

**HF's actual layout** (verified from `reference/full/raw_tokenizer.json`):
- IDs 0..N-1: special tokens (in insertion order)
- IDs N..N+255: 256 ByteLevel-mapped bytes (in sorted-vocab-key order, not raw byte order — `!` got ID 8, not 8+0x21)
- IDs N+256..end: merges, in training order

**Our spec / current `SpecialTokenRegistry::assign_ids(base = 256 + num_merges)`**:
- IDs 0..255: 256 raw bytes
- IDs 256..255+num_merges: merges
- IDs 256+num_merges..end: special tokens

These layouts are **structurally incompatible**. The wisetok internal merge loop assumes byte IDs are 0..255 and merge IDs start at 256 — that's hardcoded in `train_core_incremental` (`new_id = 256 + merges_done`). HF assumes specials come first.

**Three options:**

### Option A — Match HF (re-architect internally)

Have the trainer know about specials up front. Reserve the lowest N IDs for specials, shift bytes to N..N+255, merges start at N+256. Required changes:
- `train_core_incremental` takes `special_token_count: u32`; `new_id = special_token_count + 256 + merges_done`
- Initial alphabet construction in `train_from_iterator` shifts by `special_token_count`
- Every `Pair = (u32, u32)` consumer must use the shifted offsets
- `Tokenizer::vocab_size`, `decode`, `encode`, `get_mergeable_ranks` all need to know the offset

**Pros**: byte-identical tokenizer.json output to what HF would have produced; no surprises for users comparing wisetok to HF on the same corpus.
**Cons**: invasive change; touches the merge loop's only constant (256); introduces an offset everywhere; breaks the "byte 0x41 → token id 65" mental model.

### Option B — Match HF in the export only

Internally keep our scheme (specials last). At export time, **renumber** every ID:
- Compute the HF layout: specials get 0..N-1, bytes get the HF sorted order, merges get N+256..end
- Build a `wisetok_id → hf_id` permutation table
- Apply it to every `(left, right) → new_id` in `merges`, and to the vocab keys

**Pros**: zero change to the trainer or runtime data structures; the renumber is a pure export-time function.
**Cons**: anyone who calls `wisetok.encode()` and gets IDs that don't match the IDs in the exported `tokenizer.json` for the same text. We'd need to either also expose an HF-renumbered encode method or document this clearly.

### Option C — Keep our scheme, document divergence

Export the merges + vocab in our scheme (bytes 0..255, merges 256..N, specials at the end). Users can still load the file with `AutoTokenizer.from_pretrained()` because HF's reader doesn't *require* specials be at the start — it just expects valid IDs in the `added_tokens` array. We'd write `"added_tokens": [{"id": 256+num_merges, ...}, ...]` and HF would happily accept it.

**Pros**: minimum work; trainer untouched; encode/decode IDs match exported IDs trivially.
**Cons**: a wisetok-trained tokenizer and an HF-trained tokenizer on the same corpus and same special tokens will produce *different IDs for the same text*. Anyone benchmarking wisetok against HF on token-id-equality will see a mismatch even when the BPE merges are correct. The "match HF byte-for-byte" promise from the spec is broken.

### My recommendation: Option C with clear documentation

The whole point of the HF export is interoperability with `AutoTokenizer.from_pretrained` and `transformers`. Both work fine with any valid ID layout — they treat IDs as opaque integers. The only thing Option A/B buys us is "wisetok IDs look numerically identical to HF IDs on the same training run", which is rarely what users actually need.

Option A is the most invasive and infects the merge loop. Option B doubles the encode surface (you'd need `encode_internal` and `encode_hf`) and is a maintenance burden forever. Option C is a one-liner in the export code and a paragraph in the README.

**Decision needed from you before I write export code.** I'll pause for it.

## Other surprises worth flagging

- **`save_pretrained` in `transformers` 5.x doesn't write `special_tokens_map.json`, `vocab.json`, or `merges.txt`** — only `tokenizer.json` and `tokenizer_config.json`. Older transformers wrote the sidecars. If wisetok wants to support consumers pinned to old transformers, generate the sidecars ourselves. Probably unnecessary for our target.
- **`PreTokenizerWrapper` and `DecoderWrapper` are `#[serde(untagged)]`**; the `"type"` tag comes from each leaf's `impl_serde_type!` macro. Our serializer doesn't need to know this — we just write the leaf JSON directly.
- **`vocab` keys are GPT-2-mapped Unicode strings, not raw bytes**. So byte 0x20 (space) appears in `vocab` as `"Ġ"`, byte 0x0A (newline) as `"Ċ"`, etc. Our merges array also uses these mapped strings. The byte-to-unicode table at `reference/byte_to_unicode.rs` is the source of truth.

## Recommended target schema (locked in)

Write this exact schema (modulo the special-token decision):

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [...],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      { "type": "Split", "pattern": { "Regex": "..." }, "behavior": "Isolated", "invert": false },
      { "type": "Digits", "individual_digits": true },
      { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false }
    ]
  },
  "post_processor": null,
  "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": { ... },
    "merges": [ ["Ġ", "Ġ"], ... ]
  }
}
```

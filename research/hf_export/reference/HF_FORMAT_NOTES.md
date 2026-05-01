# HuggingFace `tokenizer.json` format reference for wisetok HF-export

Captured by `capture_script.py` on 2026-05-01.

## Package versions (load-bearing — format drifts across minor versions)

- Python: 3.12.2 (conda-forge)
- `tokenizers`: **0.22.2**
- `transformers`: **5.2.0**

If you bump these, re-run the capture and diff. The `tokenizers` library is
where the `tokenizer.json` schema is defined; `transformers` only adds wrapper
sidecars (see "transformers wrapping" below).

## Files captured

```
hf_reference/
  capture_script.py        # the script (re-runnable)
  byte_to_unicode.rs       # 256-entry GPT-2 ByteLevel table for Rust
  minimal/                 # Tokenizer A: vocab_size=1000, no specials
    raw_tokenizer.json     # tokenizers.Tokenizer.save() output
    tokenizer.json         # PreTrainedTokenizerFast.save_pretrained() output
    tokenizer_config.json  # from save_pretrained
  full/                    # Tokenizer B: vocab_size=2000, 8 CODE_PRESET specials
    raw_tokenizer.json
    tokenizer.json
    tokenizer_config.json
```

`tokenizer.json` differs from `raw_tokenizer.json` only in `post_processor`
(see below). For wisetok's HF export, target `raw_tokenizer.json` — it is the
direct output of `Tokenizer.save()` and contains nothing transformers added.

`special_tokens_map.json` and `vocab.json` / `merges.txt` are **not** written
by `save_pretrained` in this version when wrapping a bare `Tokenizer` object
that has no `bos_token` / `eos_token` / etc. set on the wrapper. The added
tokens are still recorded inside `tokenizer.json` under `added_tokens`.

## Top-level key order (every tokenizer.json)

Exactly this order, JSON-object semantics aside (the writer preserves insertion order):

1. `version`        — string, value is `"1.0"` in tokenizers 0.22.2
2. `truncation`     — `null` (we did not configure truncation)
3. `padding`        — `null`
4. `added_tokens`   — array (empty for minimal, 8 entries for full)
5. `normalizer`     — `null` (we set none)
6. `pre_tokenizer`  — object (the `Sequence` of three steps)
7. `post_processor` — `null` in raw, `TemplateProcessing` after `save_pretrained`
8. `decoder`        — object (`ByteLevel`)
9. `model`          — object (`BPE`)

## `model` (BPE) — exact field set, in order

```jsonc
{
  "type": "BPE",
  "dropout": null,
  "unk_token": null,
  "continuing_subword_prefix": null,
  "end_of_word_suffix": null,
  "fuse_unk": false,
  "byte_fallback": false,
  "ignore_merges": false,
  "vocab": { ... },          // string -> u32
  "merges": [ ["a","b"], ... ] // array of [String, String] pairs
}
```

All nine keys are **always emitted**, even when their value is the default
(`null`/`false`). Do not omit any of them.

### `merges` format

`Vec<[String, String]>` — a list of two-element string arrays. **Not** the
old `"a b"` whitespace-joined string format. This changed in `tokenizers`
~0.20; older HF tokenizers in the wild may still use the string form, but
fresh `Tokenizer.save()` output in 0.22.x always emits the array-of-pairs
form. wisetok's exporter must produce this form.

Order: the merges array is the BPE training order. `merges[i]` is the
`(i + base_offset)`-th merge, where `base_offset` is the lowest vocab id
that is a merge (= number of specials + 256 for our setup).

### `vocab` format

JSON object (`Map<String, u32>`). Keys are the **byte-level encoded** token
strings (raw bytes mapped through GPT-2 byte-to-unicode), values are token
ids. Order in the file follows insertion order: specials first (if any),
then 256 base bytes in unicode order, then merges in training order.

Confirmed: byte `0x20` (space) appears as key `"Ġ"` (U+0120). See
`byte_to_unicode.rs` for the full 256-entry table.

## Special tokens — id assignment (CRITICAL)

When `BpeTrainer(special_tokens=[...])` is used, HF assigns **ids 0..N-1**
to the special tokens in the order they were passed to the trainer. The
256 base bytes then get ids `N..N+255`, and merges start at `N+256`.

**This conflicts with wisetok's spec** which says specials are placed at
`256 + num_merges`. To match HF byte-for-byte, the wisetok exporter must
either:

1. Re-emit specials at the front of the vocab (rewriting all merge ids), or
2. Document this as an intentional divergence and accept that ids differ.

Captured `full/raw_tokenizer.json` shows the HF behaviour:

- `"<|endoftext|>": 0` ... `"<|filename|>": 7`
- `"!": 8` (first base byte)
- `"Ń": 263` (last base byte = 7 + 256)
- merges start at id 264

## `added_tokens` entry shape

```jsonc
{
  "id": 0,
  "content": "<|endoftext|>",
  "single_word": false,
  "lstrip": false,
  "rstrip": false,
  "normalized": false,
  "special": true
}
```

Field order matters for byte-for-byte equality: id, content, single_word,
lstrip, rstrip, normalized, special. All booleans use lowercase
`true`/`false` (standard JSON). For trainer-injected specials, `normalized`
is `false` and `special` is `true`.

## `pre_tokenizer` — the `Sequence([Split, Digits, ByteLevel])` we use

```jsonc
{
  "type": "Sequence",
  "pretokenizers": [
    {
      "type": "Split",
      "pattern": { "Regex": "<GPT4_PATTERN>" },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "Digits",
      "individual_digits": true
    },
    {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": true,
      "use_regex": false
    }
  ]
}
```

Gotchas:

- The list key is `"pretokenizers"` (one word, no underscore).
- `behavior` is `"Isolated"` (PascalCase) in the JSON — the Python API
  accepts `"isolated"` lowercase, but the serializer normalises it to
  PascalCase. Other valid values: `"Removed"`, `"MergedWithPrevious"`,
  `"MergedWithNext"`, `"Contiguous"`.
- `pattern` is an object `{ "Regex": "..." }` — the discriminant is the
  inner key; the alternative would be `{ "String": "..." }` for literal
  splits.
- `Split.invert`: `false` means "keep the matches as chunks" (with
  `Isolated` behavior).
- `ByteLevel.use_regex` is **`false`** in our pre-tokenizer step (we
  already split with the GPT-4 regex). It is `true` in the decoder block.
  Easy to get wrong.
- `ByteLevel.trim_offsets` is `true` (the default) — it appears in the
  serialized form even though we never set it. The decoder also gets
  `trim_offsets: true`.

## `decoder` — `ByteLevel`

```jsonc
{
  "type": "ByteLevel",
  "add_prefix_space": true,
  "trim_offsets": true,
  "use_regex": true
}
```

These are the `ByteLevel` decoder defaults; they differ from the
pre-tokenizer ByteLevel block. The decoder always uses
`add_prefix_space: true` and `use_regex: true` regardless of what was
configured for the pre-tokenizer. Mirror this in the export.

## `post_processor`

`null` in `raw_tokenizer.json`. After `PreTrainedTokenizerFast(...).save_pretrained()`,
transformers writes a `TemplateProcessing` post-processor that just emits
`A` (and `A B` for pairs) with empty `special_tokens`. wisetok's exporter
should default to `null` (matching the pure-tokenizers output); the
transformers wrapper adds it on demand.

## `tokenizer_config.json` (from `save_pretrained`)

```json
{
  "backend": "tokenizers",
  "model_max_length": 1000000000000000019884624838656,
  "tokenizer_class": "TokenizersBackend"
}
```

Identical for both minimal and full in transformers 5.2.0. The huge
`model_max_length` is `int(1e30)` from the transformers default. Earlier
transformers versions also wrote `clean_up_tokenization_spaces`,
`auto_map`, etc. — those are gone in 5.x.

## Byte-to-unicode table

Extracted by replicating GPT-2's algorithm (in `capture_script.py`'s
`write_byte_to_unicode_table`) and asserted equal to
`tokenizers.pre_tokenizers.ByteLevel.alphabet()` as a set. Spot checks:

| byte    | char      | codepoint |
| ------- | --------- | --------- |
| `0x00`  | `Ā`       | U+0100    |
| `0x09` (tab) | `ĉ`  | U+0109    |
| `0x0A` (LF)  | `Ċ`  | U+010A    |
| `0x20` (sp)  | `Ġ`  | U+0120    |
| `0x21` `!`   | `!`  | U+0021    |
| `0x7E` `~`   | `~`  | U+007E    |
| `0x7F`       | `ŀ`  | U+0140    |
| `0xA0`       | `Ł`  | U+0142    |
| `0xAD`       | `Ń`  | U+0143    |
| `0xFF`       | `ÿ`  | U+00FF    |

Full table: `byte_to_unicode.rs` (ready to paste into wisetok).

## Most important details to replicate byte-for-byte (TL;DR)

1. **Top-level key order**: `version`, `truncation`, `padding`,
   `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`,
   `decoder`, `model` — emit in this order, include `null`s.
2. **Always emit all 9 BPE model fields**, including `dropout: null`,
   `byte_fallback: false`, `ignore_merges: false`, etc. Field order
   matters for diffing.
3. **`merges` is `Vec<[String, String]>`**, not `"a b"` strings.
4. **Special tokens get the lowest ids** (0..N-1) when `BpeTrainer` is
   given them, NOT `256 + num_merges`. This is HF's invariant; either
   match it or document divergence.
5. **`behavior: "Isolated"`** (PascalCase), key is `"pretokenizers"`,
   `pattern: { "Regex": "..." }` is a tagged-union object.
6. **ByteLevel pre-tokenizer**: `use_regex: false`, `trim_offsets: true`,
   `add_prefix_space: false`.
7. **ByteLevel decoder**: `use_regex: true`, `trim_offsets: true`,
   `add_prefix_space: true` (different from the pre-tokenizer block).
8. **Vocab keys are byte-level encoded** through the GPT-2 256-entry
   table (space → `Ġ`, newline → `Ċ`, etc.).

## Surprises

- `Tokenizer.save()` produces compact pretty-printed JSON with 2-space
  indent and no trailing newline (well, file ends with `}` + newline).
  Match this exactly if byte-for-byte equality is the goal.
- Despite `vocab_size=1000`, the minimal tokenizer ended at 266 tokens
  because the corpus runs out of pair frequencies > 1. HF silently stops;
  no error or warning.
- `transformers` 5.2.0 writes a stub `TemplateProcessing` post-processor
  even when none was configured on the underlying `Tokenizer`. The raw
  `Tokenizer.save()` output keeps `post_processor: null`.
- `transformers` 5.2.0 no longer emits `special_tokens_map.json` for a
  bare `PreTrainedTokenizerFast(tokenizer_object=...)` wrapper — the
  added tokens live inside `tokenizer.json` only. Older transformers
  versions did write that sidecar. If wisetok's loader needs to support
  HF tokenizers from older transformers versions, account for this.
- Trailing-comma hygiene: HF's JSON has none (standard JSON). The pretty
  printer is deterministic across runs given the same input — repeated
  training does NOT produce identical output because BPE merge selection
  has tie-breaks that depend on insertion order in the underlying maps,
  but post-training serialization is deterministic.

# HF tokenizers serialization code map

Pinned commit: `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`
Crate version (from `tokenizers/Cargo.toml`): `0.23.2-dev.0`
File-format `version` constant: `"1.0"` (`tokenizers/src/tokenizer/serialization.rs:13`).

All paths below are relative to the cloned `tokenizers/` repo at the worktree root (i.e. the actual file is at `tokenizers/tokenizers/src/...`). Line numbers refer to that clone.

---

## 1. Top-level `Tokenizer` (`TokenizerImpl<M, N, PT, PP, D>`)

- Type: `tokenizers/src/tokenizer/mod.rs:514` (`pub struct TokenizerImpl<M, N, PT, PP, D>`).
- Custom `Serialize`/`Deserialize` impls in `tokenizers/src/tokenizer/serialization.rs:15-83`.
- Uses `serialize_struct("Tokenizer", 9)` — no serde attributes; field order is hand-coded.

Serialized field order (`serialization.rs:27-46`):
1. `version` — string constant, currently `"1.0"` (`SERIALIZATION_VERSION`, line 13).
2. `truncation` — `Option<TruncationParams>`. Serialized as `null` when `None`.
3. `padding` — `Option<PaddingParams>`. Serialized as `null` when `None`.
4. `added_tokens` — array, see §6.
5. `normalizer` — `Option<N>`. `null` when no normalizer set.
6. `pre_tokenizer` — `Option<PT>`. `null` when absent.
7. `post_processor` — `Option<PP>`. `null` when absent.
8. `decoder` — `Option<D>`. `null` when absent.
9. `model` — `M` (required, never null in valid output).

Deserializer rejects any `version` other than `"1.0"` (`serialization.rs:118-120`). Round-trip example with all nullable fields set to `null` is in `serialization.rs:180-224`.

---

## 2. `BPE` model

- Struct: `tokenizers/src/models/bpe/model.rs:297-322` (`pub struct BPE`).
- Serde impls: `tokenizers/src/models/bpe/serialization.rs:9-156` (manual, not derived).

Serialized field order (`serialization.rs:14-42`):

| Field | Rust type | JSON shape | Notes |
|---|---|---|---|
| `type` | constant | `"BPE"` | always emitted |
| `dropout` | `Option<f32>` | float or `null` | always emitted; **no `skip_serializing_if`**. |
| `unk_token` | `Option<String>` | string or `null` | always emitted. |
| `continuing_subword_prefix` | `Option<String>` | string or `null` | always emitted. |
| `end_of_word_suffix` | `Option<String>` | string or `null` | always emitted. |
| `fuse_unk` | `bool` | bool | always emitted. |
| `byte_fallback` | `bool` | bool | always emitted. |
| `ignore_merges` | `bool` | bool | always emitted. Deserializer accepts JSON missing this field (`serialization.rs:124-128`, see test at line 234-236). |
| `vocab` | `AHashMap<String, u32>` (via `OrderedVocabIter`) | object `{token: id}` | iterated in **ascending id order**, with hole-warning if ids are non-contiguous (`tokenizers/src/models/mod.rs:32-58`). |
| `merges` | written as `Vec<(String, String)>` | array of 2-element string arrays | sorted ascending by `rank` (`serialization.rs:32`). |

### Merges format

- On serialize: `Vec<[String, String]>` — pair of token strings (`serialization.rs:27-40`):
  ```json
  "merges": [["a", "b"], ["a", "b c d"]]
  ```
- On deserialize: accepts both new and legacy form via `untagged` enum `MergeType` (`serialization.rs:85-91`):
  - `Tuple(Vec<(String, String)>)` — preferred new form.
  - `Legacy(Vec<String>)` — pre-`0.20`-ish single-string-per-merge form `"a b"`. Converted via `convert_merges_to_hashmap`.
- Confirmed in tests `serialization.rs:181` (legacy) vs `serialization.rs:188` (new).

### `BPE` Rust struct (line numbers in `model.rs`)
- `vocab: Vocab = AHashMap<String, u32>` (line 299, type alias line 20).
- `vocab_r: AHashMap<u32, String>` (line 301; used for ordered serialization).
- `merges: MergeMap = AHashMap<Pair, (u32, u32)>` (line 303, alias line 22) — `(rank, new_token_id)`.
- `dropout: Option<f32>` (308), `unk_token: Option<String>` (310), `continuing_subword_prefix: Option<String>` (312), `end_of_word_suffix: Option<String>` (314).
- `fuse_unk: bool` (316), `byte_fallback: bool` (319), `ignore_merges: bool` (321).
- `Default for BPE`: line 340 — calls `builder().build()`. Builder defaults: `dropout=None`, `unk_token=None`, `continuing_subword_prefix=None`, `end_of_word_suffix=None`, `fuse_unk=false`, `byte_fallback=false`, `ignore_merges=false`.

---

## 3. `PreTokenizerWrapper`

- File: `tokenizers/src/pre_tokenizers/mod.rs:28-43`.
- Outer enum is `#[serde(untagged)]` — it serializes purely as the inner variant. The `"type": "..."` tag comes from each inner struct.
- Deserializer is hand-rolled (line 64-206) and supports both **tagged** (`{"type":"X", ...}`) and **legacy untagged** payloads.

### Variants and their serialized `type` strings

Each inner struct uses the `impl_serde_type!` macro (`tokenizers/src/utils/mod.rs:128-159`), which expands to `#[serde(tag = "type")]` on the outer derive. Variant → tag string:

| Wrapper variant | Inner struct | JSON `"type"` value |
|---|---|---|
| `BertPreTokenizer` | `BertPreTokenizer` (unit struct) | `"BertPreTokenizer"` |
| `ByteLevel` | `pre_tokenizers/byte_level.rs::ByteLevel` | `"ByteLevel"` |
| `Delimiter` | `delimiter.rs::CharDelimiterSplit` | `"CharDelimiterSplit"` |
| `Metaspace` | `metaspace.rs::Metaspace` | `"Metaspace"` |
| `Whitespace` | `whitespace.rs::Whitespace` | `"Whitespace"` |
| `Sequence` | `sequence.rs::Sequence` | `"Sequence"` |
| `Split` | `split.rs::Split` | `"Split"` |
| `Punctuation` | `punctuation.rs::Punctuation` | `"Punctuation"` |
| `WhitespaceSplit` | `whitespace.rs::WhitespaceSplit` | `"WhitespaceSplit"` |
| `Digits` | `digits.rs::Digits` | `"Digits"` |
| `UnicodeScripts` | `unicode_scripts/...::UnicodeScripts` | `"UnicodeScripts"` |
| `FixedLength` | `fixed_length.rs::FixedLength` | `"FixedLength"` |

### `Split` (`pre_tokenizers/split.rs:27-94`)

JSON shape (verified by `split.rs:243` and `tests/serialization.rs:155-167`):
```json
{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed","invert":true}
{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated","invert":false}
```
- `pattern: SplitPattern` is `enum { String(String), Regex(String) }` with default (externally-tagged) serde repr (`split.rs:9-13`) — serializes as `{"String": "..."}` or `{"Regex": "..."}`.
- `behavior: SplitDelimiterBehavior` enum (`tokenizer/normalizer.rs:81-88`) — variants serialize **PascalCase**: `"Removed"`, `"Isolated"`, `"MergedWithPrevious"`, `"MergedWithNext"`, `"Contiguous"`. **No `rename_all`** — verbatim variant names.
- `invert: bool` — required.
- The internal `regex: SysRegex` field has `#[serde(skip)]` (line 31) and is rebuilt on deserialize.

### `Digits` (`pre_tokenizers/digits.rs:6-13`)
```json
{"type":"Digits","individual_digits":true}
```
- Single field `individual_digits: bool`. No skip rule, always emitted.
- Default is `false` (line 22-25).

### `ByteLevel` (`pre_tokenizers/byte_level.rs:51-82`)

JSON (verified `tests/serialization.rs:174-177`):
```json
{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}
```
Fields:
- `add_prefix_space: bool` — `Default = true` (line 77).
- `trim_offsets: bool` — `Default = true` (line 78).
- `use_regex: bool` — `#[serde(default = "default_true")]` (line 66) → defaults to `true` on deserialize if missing.
- `#[non_exhaustive]` (line 56), so always construct via `ByteLevel::new(...)` or `default()`.

Same struct is reused as a **Decoder** variant (re-exported in `decoders/mod.rs:10`). Defaults to `add_prefix_space=true, trim_offsets=true, use_regex=true` for both pretokenizer and decoder.

### `Sequence` pre-tokenizer (`pre_tokenizers/sequence.rs:6-15`)

Inner field name is **`pretokenizers`** (verified line 9, line 228 of `mod.rs`).
```json
{"type":"Sequence","pretokenizers":[ {"type":"WhitespaceSplit"}, ... ]}
```
This is **different** from the decoder Sequence which uses `decoders` (see §4) and from the normalizer Sequence which uses `normalizers`.

---

## 4. `DecoderWrapper`

- File: `tokenizers/src/decoders/mod.rs:27-150`.
- Same pattern as `PreTokenizerWrapper`: outer `#[serde(untagged)]`; inner structs carry `#[serde(tag = "type")]` via `impl_serde_type!`.
- Variants: `BPE` → `"BPEDecoder"` (note the rename), `ByteLevel`, `WordPiece`, `Metaspace`, `CTC`, `Sequence`, `Replace`, `Fuse`, `Strip`, `ByteFallback`.
- The `BPE` variant's tag is `"BPEDecoder"` (mapped via `EnumType::BPEDecoder` at line 56 / 101).
- `ByteLevel` decoder uses **the same struct** as the pretokenizer (re-export at `decoders/mod.rs:10`), so fields and defaults are identical:
  ```json
  {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}
  ```
- Decoder `Sequence` uses field name `decoders` (see test at `decoders/mod.rs:186, 189, 198`):
  ```json
  {"type":"Sequence","decoders":[{"type":"ByteFallback"}, ...]}
  ```

---

## 5. `Normalizer`, `PostProcessor`, `Truncation`, `Padding` — absent value

For all four: when the user has not configured one, the corresponding field on `TokenizerImpl` is `None: Option<...>`, and serde emits **`null`**, not omits the field. Reference: top-level `Serialize` always calls `serialize_field` for every field (`tokenizer/serialization.rs:33-44`), and the round-trip test at `serialization.rs:213-216` expects:
```json
"normalizer": null,
"pre_tokenizer": null,
"post_processor": null,
"decoder": null,
```
On the deserialize side, accepting `null` works because the wrappers are themselves `Option<...>`. Top-level deserializer also tolerates **missing** keys (the `while let Some(key)` loop in `serialization.rs:114-148` simply skips them). HF's actual reference outputs use explicit `null`s, so wisetok should emit `null`, not omit.

`truncation`/`padding` follow the same rule (`null` when not set).

---

## 6. `AddedToken` and `added_tokens`

- `AddedToken` struct: `tokenizers/src/tokenizer/added_vocabulary.rs:16-30`.
- `AddedTokenWithId`: line 573-580 — `{ id: u32, #[serde(flatten)] token: AddedToken }`. `Serialize for AddedVocabulary` (line 582-605) writes a sequence of these sorted by ascending `id`.

JSON shape (verified `tokenizer/serialization.rs:185-211`):
```json
{
  "id": 0,
  "content": "[SPECIAL_0]",
  "single_word": false,
  "lstrip": false,
  "rstrip": false,
  "normalized": false,
  "special": true
}
```

All fields are `#[derive(Serialize, Deserialize)]` defaults — **no `skip_serializing_if`**, all fields always emitted. `Default for AddedToken` (line 78-89): `single_word=false, lstrip=false, rstrip=false, normalized=true, special=false`. `AddedToken::from(content, special)` sets `normalized = !special`.

---

## 7. Trainer / merge ordering

- Trainer: `tokenizers/src/models/bpe/trainer.rs`.
- Hot loop: `do_train` line 456; merge emission line 549 (`merges.push((top.pair, new_token_id))`).
- `new_token_id` is computed at line 541-547: it equals `id_to_word.len() as u32` if the merged token is new, i.e. **the next sequential id** after byte alphabet + special tokens.
- `model.merges` is built at line 620-624: `merges.into_iter().enumerate().map(|(i, (pair, new_id))| (pair, (i as u32, new_id)))`. So **rank `i` = order learned, starting at 0.**
- `model.vocab` (line 610-614) maps token string → id, where for byte-level setups ids 0..255 are the byte alphabet (set by `compute_alphabet` line 475 + initial alphabet from the trainer config) and merged tokens get ids `256 + merge_index` only when no special tokens were added before alphabet. Special tokens, if any, are added first (line 470 `add_special_tokens`), so the safe statement is: **merge `i` produces token id `(initial_alphabet_size + special_tokens_count + i)`**, not unconditionally `256 + i`.
- Serialization sorts merges by rank (`bpe/serialization.rs:30-32`), so `merges[i]` in the JSON corresponds to rank `i`, learned `i`-th.

Invariant for wisetok: emit `merges` in learn order, and ensure that for every `(L, R)` at rank `i`, `vocab[L+R] = first_non_byte_id + i`.

---

## 8. Byte-to-unicode alphabet

- `tokenizers/src/pre_tokenizers/byte_level.rs:15-39` — `pub(crate) fn bytes_char() -> AHashMap<u8, char>`. This is the canonical GPT-2 `bytes_to_unicode` table:
  - Seed printable ranges: `b'!'..=b'~'`, `b'\xA1'..=b'\xAC'`, `b'\xAE'..=b'\xFF'` (lines 17-19).
  - For every other byte `b` not yet in the seed, map to `char::from_u32(256 + n)` where `n` increments from 0 (lines 24-30).
- Cached as `static BYTES_CHAR` (line 47) and reverse `CHAR_BYTES` (line 48).
- Public alphabet accessor: `ByteLevel::alphabet()` at line 93, returns `AHashSet<char>` of all 256 mapped chars.

If wisetok wants to verify byte parity: rebuild the same table from the snippet above; all 256 bytes must map to distinct printable Unicode chars in the BMP.

---

## 9. Round-trip integration tests to mirror

- `tokenizers/src/tokenizer/serialization.rs:178-230` (`test_deserialization_serialization_invariant`) — string equality on `to_string_pretty`, the strictest test wisetok can mirror.
- `tokenizers/tests/serialization.rs:22-28` (`bpe_serde`) — round-trips a byte-level BPE built via `get_byte_level_bpe()` (in `tests/common/mod.rs`); equivalent of "build → serialize → deserialize → equal".
- `tokenizers/src/models/bpe/serialization.rs:163-216` — exact `BPE` JSON byte-for-byte. Best reference for byte-level BPE export.

---

## 10. Minimum viable byte-level BPE `tokenizer.json`

Schema below is the **smallest** JSON that the HF deserializer accepts for a byte-level BPE setup, derived purely from the field rules above. Comments are not part of JSON; remove before writing.

```jsonc
{
  "version": "1.0",                    // REQUIRED, must equal "1.0".
  "truncation": null,                  // can be omitted (deserializer tolerates), but emit null for fidelity.
  "padding": null,                     // same.
  "added_tokens": [                    // REQUIRED array; can be empty [].
    {                                  // when emitted, every field is REQUIRED on serialize-side.
      "id": 256,                       // must equal the actual id assigned in `vocab`.
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,             // false for special tokens (matches AddedToken::from(_, true)).
      "special": true
    }
  ],
  "normalizer": null,                  // null = no normalization (fine for byte-level).
  "pre_tokenizer": {                   // REQUIRED for byte-level BPE encode parity.
    "type": "ByteLevel",
    "add_prefix_space": false,         // GPT-2 used true; tiktoken/cl100k uses false. Pick to match training.
    "trim_offsets": true,
    "use_regex": true
  },
  "post_processor": {                  // optional but typical for byte-level:
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": false,
    "use_regex": true
  },
  "decoder": {                         // REQUIRED for correct .decode():
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {                           // REQUIRED.
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {                         // token-string → id. Bytes 0..255 mapped through bytes_char().
      "!": 0,
      "\"": 1
      /* ... 256 byte tokens ... then merged tokens at ids 256..(256+num_merges-1) ... */
    },
    "merges": [                        // pairs in learning order.
      ["Ġ", "t"],                      // example: rank 0
      ["Ġ", "a"]                       // rank 1
      /* ... */
    ]
  }
}
```

Required vs. optional summary:
- **Required** to deserialize: `version`, `model` (with `type`, `vocab`, `merges`).
- **Required for round-trip equality with HF** (no Python-side `null` ⇄ missing surprises): all 9 top-level keys in the order `version, truncation, padding, added_tokens, normalizer, pre_tokenizer, post_processor, decoder, model`.
- Inside `model`: `type` plus `vocab` and `merges` are required; the others have lenient deserialize but are always written on serialize, so wisetok should always write them too (use `null` for absent strings/floats and the documented bool defaults).

---

## 11. Most error-prone fields (likely format-drift hot-spots)

1. **`SplitDelimiterBehavior` casing** — PascalCase (`"Isolated"`, `"MergedWithPrevious"`), **not** snake_case. (`tokenizer/normalizer.rs:81-88`.)
2. **`SplitPattern`** — `{"Regex": "..."}` vs `{"String": "..."}`, capital `R`/`S`. Externally tagged. (`split.rs:9-13`.)
3. **`Sequence` inner field name varies by container**: `pretokenizers` for pre-tokenizer (`sequence.rs:9`), `decoders` for decoder (`decoders/mod.rs:189`), `normalizers` for normalizer Sequence. Cross-using these fails deserialize.
4. **`merges` element shape** — must be `[String, String]` (2-array), not a single space-joined `"a b"` string. The legacy form is accepted on deserialize but HF currently always emits the tuple form; mismatching will pass HF round-trip but fail strict equality with reference outputs.
5. **`type` tag for BPE decoder is `"BPEDecoder"`, not `"BPE"`** (`decoders/mod.rs:56,101`). Easy to confuse with the model `type:"BPE"`.
6. **`ignore_merges` bool flag** — appeared more recently (cf. test `test_serialization_ignore_merges` at `bpe/serialization.rs:218`); always write it. Older tokenizer.json files may not contain it, but HF-generated current outputs always do.
7. **`null` vs missing** for `normalizer` / `pre_tokenizer` / `post_processor` / `decoder` / `truncation` / `padding`: HF emits explicit `null`. Omitting the key still deserializes but breaks byte-equal round-trip.
8. **`vocab` ordering** — emitted in **ascending id order** (`models/mod.rs:32-58`). Naive `HashMap` iteration is non-deterministic; equality tests against `to_string_pretty` reference outputs require ordered emission.
9. **`AddedToken` defaults asymmetry** — `normalized` defaults to `true` for non-special, but `AddedToken::from(content, special=true)` sets `normalized = false`. The JSON always writes the actual stored bool; do not "infer from `special`" downstream.
10. **`ByteLevel.add_prefix_space`** — `true` by default in HF's `Default`. tiktoken `cl100k_base`-style setups expect `false`. Wrong value here causes a leading-space drift on the first token of every input.

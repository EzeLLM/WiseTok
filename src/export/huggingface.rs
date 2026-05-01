//! HuggingFace `tokenizer.json` export.
//!
//! Produces a `tokenizer.json` that `transformers.AutoTokenizer.from_pretrained`
//! accepts, matching the structural schema of `tokenizers >= 0.20.0`. The
//! reference outputs we match against were captured with `tokenizers 0.22.2`
//! and `transformers 5.2.0` — see `research/hf_export/`.
//!
//! ## ID layout (Option C)
//!
//! wisetok writes its IDs as-is:
//!   - IDs 0..255: raw bytes (`BYTE_TO_UNICODE[byte_value]`)
//!   - IDs 256..256+num_merges: trained merges, in training order
//!   - IDs 256+num_merges..256+num_merges+S-1: special tokens, in
//!     registry order
//!
//! HF's own `BpeTrainer` puts specials at IDs 0..S-1 and shifts everything
//! else by S. Our scheme is incompatible with that ordering, but HF's
//! reader treats IDs as opaque integers — `AutoTokenizer.from_pretrained`
//! works with either layout. The trade-off is that a wisetok-trained
//! tokenizer and an HF-trained tokenizer on the same corpus will produce
//! numerically different IDs for the same text. Token-level encode/decode
//! roundtrips correctly in both. See `research/hf_export/RESEARCH_SUMMARY.md`.
//!
//! ## Format details (locked in from the research)
//!
//! - `version`: `"1.0"`
//! - `truncation`, `padding`, `normalizer`, `post_processor`: literal `null`
//! - `pre_tokenizer`: Sequence of `[Split, Digits, ByteLevel-pretok]`
//! - `decoder`: ByteLevel-decoder (different defaults from the pre-tokenizer
//!   ByteLevel: `add_prefix_space=true`, `use_regex=true`)
//! - `model`: BPE with all 9 fields always emitted (incl. `null` defaults)
//! - `merges`: `Vec<[String, String]>` of GPT-2-mapped Unicode keys
//! - `vocab`: `{key: id}` map. We use `serde_json` with `preserve_order`
//!   so the on-disk JSON is in ascending-id order, matching HF's
//!   `OrderedVocabIter`.
//! - `Sequence` field name is `"pretokenizers"` (no underscore)
//! - `Split.behavior` is PascalCase `"Isolated"`
//! - `pattern` is the tagged union `{"Regex": "..."}`

use std::collections::HashMap as StdHashMap;
use std::fs;
use std::io;
use std::path::Path;

use serde_json::{json, Value};

use super::byte_level::BYTE_TO_UNICODE;
use crate::special_tokens::SpecialTokenRegistry;
use crate::Pair;

/// Errors when writing an HF-format tokenizer to disk.
#[derive(Debug)]
pub enum HfExportError {
    Io(io::Error),
    Serialize(serde_json::Error),
    /// A merge entry references an id we haven't built a string for yet —
    /// indicates a corrupt `merges` map (left/right > new_id, or new_id <
    /// 256). Should never happen with a tokenizer trained by wisetok.
    BadMergeId {
        id: u32,
    },
}

impl From<io::Error> for HfExportError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for HfExportError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialize(e)
    }
}

impl std::fmt::Display for HfExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Serialize(e) => write!(f, "JSON serialization error: {}", e),
            Self::BadMergeId { id } => {
                write!(f, "merge references id {} which has no vocab entry", id)
            }
        }
    }
}

impl std::error::Error for HfExportError {}

/// Build the per-id Unicode-mapped vocab strings for the byte alphabet plus
/// every trained merge.
///
/// Returns a `Vec<String>` indexed by id. Entries 0..255 are the GPT-2
/// byte-to-unicode strings. Entries 256..256+num_merges are the
/// concatenation of their parents' vocab strings, in training order.
///
/// The merges map is (pair, new_id). new_id is monotonically increasing
/// from 256, so we sort by new_id and then build incrementally.
fn build_vocab_strings(merges: &StdHashMap<Pair, u32>) -> Result<Vec<String>, HfExportError> {
    let mut sorted: Vec<(&Pair, &u32)> = merges.iter().collect();
    sorted.sort_by_key(|&(_, &id)| id);

    let max_id = sorted.iter().map(|&(_, &id)| id).max().unwrap_or(255);
    let mut vocab: Vec<String> = Vec::with_capacity(max_id as usize + 1);

    // Bytes 0..255.
    for s in BYTE_TO_UNICODE.iter() {
        vocab.push((*s).to_string());
    }

    // Merges, in id order.
    for (&(left, right), &new_id) in &sorted {
        let li = left as usize;
        let ri = right as usize;
        if li >= vocab.len() {
            return Err(HfExportError::BadMergeId { id: left });
        }
        if ri >= vocab.len() {
            return Err(HfExportError::BadMergeId { id: right });
        }
        let merged = format!("{}{}", vocab[li], vocab[ri]);
        let target = new_id as usize;
        if target != vocab.len() {
            // ids must be contiguous from 256
            return Err(HfExportError::BadMergeId { id: new_id });
        }
        vocab.push(merged);
    }

    Ok(vocab)
}

/// Build the JSON Value for a complete `tokenizer.json`.
///
/// Pure function: no I/O. Used both by [`write_tokenizer_json`] and by
/// tests that diff against the captured HF references.
pub fn build_tokenizer_json(
    merges: &StdHashMap<Pair, u32>,
    pattern: &str,
    specials: &SpecialTokenRegistry,
) -> Result<Value, HfExportError> {
    let vocab_strings = build_vocab_strings(merges)?;
    let num_merges = merges.len();

    // Vocab: {unicode_str: id}, ascending by id (preserve_order keeps insertion order).
    let mut vocab_map = serde_json::Map::new();
    for (id, s) in vocab_strings.iter().enumerate() {
        vocab_map.insert(s.clone(), Value::from(id as u32));
    }
    // Specials at the tail.
    let special_base = (256 + num_merges) as u32;
    for (i, tok) in specials.tokens().iter().enumerate() {
        vocab_map.insert(tok.clone(), Value::from(special_base + i as u32));
    }

    // Merges: [[left_str, right_str], ...] in training order.
    let mut sorted_merges: Vec<(&Pair, &u32)> = merges.iter().collect();
    sorted_merges.sort_by_key(|&(_, &id)| id);
    let merges_json: Vec<Value> = sorted_merges
        .iter()
        .map(|&(&(l, r), _)| {
            json!([
                vocab_strings[l as usize].clone(),
                vocab_strings[r as usize].clone(),
            ])
        })
        .collect();

    // added_tokens entries.
    let added_tokens: Vec<Value> = specials
        .tokens()
        .iter()
        .enumerate()
        .map(|(i, tok)| {
            json!({
                "id": special_base + i as u32,
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();

    let pre_tokenizer = json!({
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": { "Regex": pattern },
                "behavior": "Isolated",
                "invert": false,
            },
            {
                "type": "Digits",
                "individual_digits": true,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false,
            },
        ],
    });

    let decoder = json!({
        "type": "ByteLevel",
        "add_prefix_space": true,
        "trim_offsets": true,
        "use_regex": true,
    });

    let model = json!({
        "type": "BPE",
        "dropout": Value::Null,
        "unk_token": Value::Null,
        "continuing_subword_prefix": Value::Null,
        "end_of_word_suffix": Value::Null,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": false,
        "vocab": Value::Object(vocab_map),
        "merges": merges_json,
    });

    Ok(json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added_tokens,
        "normalizer": Value::Null,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": Value::Null,
        "decoder": decoder,
        "model": model,
    }))
}

/// Write `tokenizer.json` to `dir`. Creates the directory if it doesn't
/// exist.
pub fn write_tokenizer_json(
    dir: &Path,
    merges: &StdHashMap<Pair, u32>,
    pattern: &str,
    specials: &SpecialTokenRegistry,
) -> Result<(), HfExportError> {
    fs::create_dir_all(dir)?;
    let value = build_tokenizer_json(merges, pattern, specials)?;
    let pretty = serde_json::to_string_pretty(&value)?;
    fs::write(dir.join("tokenizer.json"), pretty)?;
    Ok(())
}

/// Write a minimal `tokenizer_config.json` matching what `transformers` 5.x
/// produces from `PreTrainedTokenizerFast.save_pretrained`.
pub fn write_tokenizer_config(dir: &Path) -> Result<(), HfExportError> {
    fs::create_dir_all(dir)?;
    let config = json!({
        "backend": "tokenizers",
        "model_max_length": 1e30_f64,
        "tokenizer_class": "TokenizersBackend",
    });
    let pretty = serde_json::to_string_pretty(&config)?;
    fs::write(dir.join("tokenizer_config.json"), pretty)?;
    Ok(())
}

//! PyO3 bindings.
//!
//! Public surface: the `Tokenizer` pyclass.

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::merge::{train_core_incremental, Word};
use crate::pretokenizer::{DigitSplitter, PreTokenizer, RegexPreTokenizer, SequencePreTokenizer};
use crate::special_tokens::{Segment, SpecialTokenRegistry};
use crate::{Pair, GPT4_PATTERN};

/// Build a `Box<dyn PreTokenizer>` from a spec string.
///
/// Recognized specs (returned pipeline noted alongside):
///   - `"gpt4"`               → `RegexPreTokenizer(GPT4_PATTERN)`
///   - `"gpt4+digits"`        → `Sequence([Regex(GPT4), DigitSplitter])`
///   - `"regex:<pattern>"`    → `RegexPreTokenizer(<pattern>)`
///   - `"regex+digits:<...>"` → `Sequence([Regex(<...>), DigitSplitter])`
///
/// Returns `(pipeline, canonical_pattern_string)`. The canonical string is
/// what gets stored in `Tokenizer.pattern` and emitted to HF export — for
/// the digit-split variants we emit the underlying regex, since the
/// digit-split step is recorded separately in the HF export's pre_tokenizer
/// Sequence.
fn parse_pretokenizer_spec(spec: &str) -> PyResult<(Box<dyn PreTokenizer>, String)> {
    if spec == "gpt4" {
        let pre = RegexPreTokenizer::new(GPT4_PATTERN).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid GPT4_PATTERN: {}", e))
        })?;
        return Ok((Box::new(pre), GPT4_PATTERN.to_string()));
    }
    if spec == "gpt4+digits" {
        let regex_pre = RegexPreTokenizer::new(GPT4_PATTERN).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid GPT4_PATTERN: {}", e))
        })?;
        let seq = SequencePreTokenizer::new(vec![Box::new(regex_pre), Box::new(DigitSplitter)]);
        return Ok((Box::new(seq), GPT4_PATTERN.to_string()));
    }
    if let Some(pat) = spec.strip_prefix("regex:") {
        let pre = RegexPreTokenizer::new(pat).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid regex {:?}: {}", pat, e))
        })?;
        return Ok((Box::new(pre), pat.to_string()));
    }
    if let Some(pat) = spec.strip_prefix("regex+digits:") {
        let regex_pre = RegexPreTokenizer::new(pat).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid regex {:?}: {}", pat, e))
        })?;
        let seq = SequencePreTokenizer::new(vec![Box::new(regex_pre), Box::new(DigitSplitter)]);
        return Ok((Box::new(seq), pat.to_string()));
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "unknown pre_tokenizer spec {:?}; supported: \"gpt4\", \"gpt4+digits\", \"regex:<pat>\", \"regex+digits:<pat>\"",
        spec
    )))
}

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation.
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID.
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting (or a synthetic identifier
    /// like "gpt4+digits" when a composed pre-tokenizer is in use). Stored
    /// for HF export and `get_pattern()` callers — the runtime pipeline
    /// is in `pre_tokenizer` below.
    pub pattern: String,
    /// Compiled regex used as the encode-side fallback path when no
    /// `pre_tokenizer` is configured. `pub(crate)` so internal tests can
    /// construct fixtures via the struct literal; not exposed to Python.
    pub(crate) compiled_pattern: Regex,
    /// The pre-tokenizer pipeline used during training and encoding. When
    /// `None`, encode falls back to `compiled_pattern` (legacy path used by
    /// internal struct-literal test fixtures). Set by `train_from_iterator`.
    pub(crate) pre_tokenizer: Option<Box<dyn PreTokenizer>>,
    /// Special tokens registered for this tokenizer. Populated by
    /// `train_from_iterator`'s `special_tokens` argument or by
    /// `add_special_tokens` post-training. Used by both aggregation
    /// (skip-and-split during training) and `encode` (atomic emission of
    /// pre-assigned IDs). IDs are assigned at HF-export / encode time as
    /// `256 + num_merges + index_in_registry`.
    pub(crate) specials: SpecialTokenRegistry,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl Tokenizer {
    /// Create a new Tokenizer.
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex should be valid"),
            pre_tokenizer: None,
            specials: SpecialTokenRegistry::new(),
        }
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust `Vec<String>` buffer under the GIL, then release the
    /// GIL to do the heavy splitting and counting **in parallel** with rayon.
    ///
    /// `pattern` (legacy): if set, use that regex as the sole pre-tokenizer.
    /// Equivalent to `pre_tokenizer="regex:<pattern>"`. Default: GPT-4 regex.
    ///
    /// `pre_tokenizer` (new): a spec string for a more elaborate pipeline.
    /// Recognized values:
    ///   - `"gpt4"`               GPT-4 regex only (matches the legacy default)
    ///   - `"gpt4+digits"`        GPT-4 regex then individual-digit splitting
    ///   - `"regex:<pattern>"`    custom regex
    ///   - `"regex+digits:<pat>"` custom regex then digit splitting
    /// Cannot be combined with `pattern`. If both are unset, the GPT-4
    /// regex-only pipeline is used (backward-compatible default).
    ///
    /// `min_frequency` drops chunks that occurred fewer than this many times
    /// before the merge loop runs. `min_frequency=1` keeps every chunk
    /// (legacy default; same as upstream rustbpe). Higher values shrink the
    /// word table at the cost of dropping rare chunks from training.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1, pre_tokenizer=None, special_tokens=None))]
    #[pyo3(
        text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1, pre_tokenizer=None, special_tokens=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
        min_frequency: i64,
        pre_tokenizer: Option<String>,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<()> {
        // Resolve the pre-tokenizer pipeline. The legacy `pattern` argument
        // and the new `pre_tokenizer` spec are mutually exclusive — let the
        // user know if they pass both rather than picking one silently.
        if pattern.is_some() && pre_tokenizer.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass either `pattern` (legacy) or `pre_tokenizer` (new), not both",
            ));
        }
        let (pre, pattern_str): (Box<dyn PreTokenizer>, String) = if let Some(spec) = pre_tokenizer
        {
            parse_pretokenizer_spec(&spec)?
        } else if let Some(pat) = pattern {
            let r = RegexPreTokenizer::new(&pat).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid regex pattern: {}", e))
            })?;
            (Box::new(r), pat)
        } else {
            // Default: GPT-4 regex only (matches upstream rustbpe).
            parse_pretokenizer_spec("gpt4")?
        };

        // Store the canonical pattern + a fallback compiled regex for the
        // legacy encode path. Encode itself uses `self.pre_tokenizer` once
        // it is set below.
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e))
        })?;

        // Register the special tokens for this run. Replace anything
        // previously registered — re-training is destructive on `self`.
        self.specials = SpecialTokenRegistry::new();
        if let Some(toks) = special_tokens {
            for t in toks {
                self.specials.add(t).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("special token: {}", e))
                })?;
            }
        }

        // Prepare a true Python iterator object.
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global chunk counts.
        let mut counts: AHashMap<CompactString, i64> = AHashMap::new();

        // Temporary buffer we refill under the GIL.
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing sequences from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_sequences = 0u64;

        // Helper: refill `buf` with up to `buffer_size` strings from the
        // Python iterator. Returns Ok(true) if the iterator is exhausted.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true); // exhausted
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop: refill under GIL, process without GIL (parallel).
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            // Borrow the pre-tokenizer and special-token registry across
            // rayon workers. Both are Send + Sync so immutable references
            // are shared safely. Specials are split off first so they
            // bypass BPE entirely; only Text segments feed the pre-tokenizer.
            let pre_ref: &dyn PreTokenizer = &*pre;
            let specials_ref: &SpecialTokenRegistry = &self.specials;
            let local: AHashMap<CompactString, i64> = py.detach(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i64> = AHashMap::new();
                        for seg in specials_ref.split(s) {
                            if let Segment::Text(text) = seg {
                                for piece in pre_ref.pre_tokenize(text) {
                                    *m.entry(CompactString::from(piece)).or_default() += 1;
                                }
                            }
                            // Segment::Special: skip; never enters BPE.
                        }
                        m
                    })
                    .reduce(AHashMap::new, |mut a, b| {
                        for (k, v) in b {
                            *a.entry(k).or_default() += v;
                        }
                        a
                    })
            });

            // Merge local into global (single-threaded).
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!(
            "Processed {} sequences total, {} unique",
            total_sequences,
            counts.len()
        );

        // Materialize words & counts, applying min_frequency.
        let total_unique = counts.len();
        let mut words = Vec::with_capacity(total_unique);
        let mut cvec = Vec::with_capacity(total_unique);
        for (chunk, c) in counts.into_iter() {
            if c < min_frequency {
                continue;
            }
            words.push(Word::new(
                chunk.as_bytes().iter().map(|&b| b as u32).collect(),
            ));
            cvec.push(c);
        }
        if min_frequency > 1 {
            log::info!(
                "min_frequency={} filtered {} → {} unique chunks",
                min_frequency,
                total_unique,
                words.len()
            );
        }

        train_core_incremental(&mut words, &cvec, vocab_size, &mut self.merges);

        // Store the pipeline for later encode() calls.
        self.pre_tokenizer = Some(pre);
        Ok(())
    }

    /// Return the regex pattern.
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the vocabulary size (256 base bytes + number of merges).
    #[getter]
    pub fn vocab_size(&self) -> u32 {
        256 + self.merges.len() as u32
    }

    /// Return the mergeable ranks (token bytes -> token id / rank).
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        crate::export::tiktoken::mergeable_ranks(&self.merges)
    }

    /// Append special tokens to the tokenizer's registry. Specials added
    /// after training are still atomic during `encode`; their IDs are
    /// `256 + num_merges + index`, so the encoded IDs of newly added
    /// specials follow whatever was registered before them (including
    /// any that came in via `train_from_iterator(special_tokens=...)`).
    ///
    /// Raises `ValueError` if any token is empty, contains NUL, or is
    /// already registered.
    pub fn add_special_tokens(&mut self, tokens: Vec<String>) -> PyResult<()> {
        for t in tokens {
            self.specials.add(t).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("special token: {}", e))
            })?;
        }
        Ok(())
    }

    /// Return the special tokens currently registered, in registry order.
    pub fn get_special_tokens(&self) -> Vec<String> {
        self.specials.tokens().to_vec()
    }

    /// Save the tokenizer in HuggingFace `tokenizer.json` format.
    ///
    /// Writes `tokenizer.json` (and `tokenizer_config.json` if
    /// `write_config=True`) into `output_dir`. Creates the directory if
    /// missing. After this, `transformers.AutoTokenizer.from_pretrained(
    /// output_dir)` will load the tokenizer.
    ///
    /// `special_tokens` is an optional override list. When `None` (the
    /// default), the tokenizer's currently-registered specials are used —
    /// these come from `train_from_iterator(special_tokens=...)` and
    /// `add_special_tokens(...)`. Pass an explicit list to override.
    ///
    /// Note: the wisetok ID layout (bytes 0..255, merges 256..N, specials
    /// at the tail) differs from what HF's own BpeTrainer would emit
    /// (specials 0..S-1, then bytes, then merges). HF readers accept both;
    /// the practical consequence is that wisetok IDs and HF IDs for the
    /// same corpus + special tokens will not be numerically equal. See
    /// `research/hf_export/RESEARCH_SUMMARY.md`.
    #[pyo3(signature = (output_dir, special_tokens=None, write_config=true))]
    #[pyo3(text_signature = "(self, output_dir, special_tokens=None, write_config=True)")]
    pub fn save_huggingface(
        &self,
        output_dir: &str,
        special_tokens: Option<Vec<String>>,
        write_config: bool,
    ) -> PyResult<()> {
        use crate::export::huggingface::{write_tokenizer_config, write_tokenizer_json};

        let registry: SpecialTokenRegistry = if let Some(toks) = special_tokens {
            // Explicit override — use only what the caller provided.
            let mut r = SpecialTokenRegistry::new();
            for t in toks {
                r.add(t).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("special token: {}", e))
                })?;
            }
            r
        } else {
            // No override — use whatever's already registered on `self`.
            self.specials.clone()
        };

        let dir = std::path::Path::new(output_dir);
        write_tokenizer_json(dir, &self.merges, &self.pattern, &registry).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("failed to write tokenizer.json: {}", e))
        })?;
        if write_config {
            write_tokenizer_config(dir).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "failed to write tokenizer_config.json: {}",
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Encode a string into token IDs.
    ///
    /// Special tokens registered via `train_from_iterator(special_tokens=...)`
    /// or `add_special_tokens` are matched as exact substrings before BPE
    /// applies. Each matched special emits a single pre-assigned ID
    /// (`256 + num_merges + index_in_registry`); the surrounding text
    /// segments go through the normal pre-tokenizer + BPE path.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();
        let special_base = (256 + self.merges.len()) as u32;

        for seg in self.specials.split(text) {
            match seg {
                Segment::Special(s) => {
                    // Find the special's index in the registry.
                    if let Some(idx) = self.specials.tokens().iter().position(|t| t == s) {
                        all_ids.push(special_base + idx as u32);
                    }
                }
                Segment::Text(t) => {
                    // Prefer the stored pre-tokenizer (set by train_from_iterator).
                    // Fall back to compiled_pattern for fixtures that construct
                    // Tokenizer via the struct literal in tests.
                    if let Some(pre) = &self.pre_tokenizer {
                        for chunk in pre.pre_tokenize(t) {
                            self.encode_chunk_into(chunk, &mut all_ids);
                        }
                    } else {
                        for m in self.compiled_pattern.find_iter(t) {
                            let chunk = match m {
                                Ok(mat) => mat.as_str(),
                                Err(e) => {
                                    log::warn!("Regex match error, skipping chunk: {}", e);
                                    continue;
                                }
                            };
                            self.encode_chunk_into(chunk, &mut all_ids);
                        }
                    }
                }
            }
        }

        all_ids
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        // Build reverse mapping: token_id -> bytes.
        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();

        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&(left, right), &merged_id) in &sorted_merges {
            let mut merged_bytes = vocab
                .get(left as usize)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid token id {} in merge",
                        left
                    ))
                })?
                .clone();
            merged_bytes.extend(vocab.get(right as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid token id {} in merge",
                    right
                ))
            })?);

            if vocab.len() <= merged_id as usize {
                vocab.resize(merged_id as usize + 1, Vec::new());
            }
            vocab[merged_id as usize] = merged_bytes;
        }

        let mut bytes = Vec::new();
        for &id in &ids {
            let token_bytes = vocab.get(id as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown token id: {}", id))
            })?;
            bytes.extend(token_bytes);
        }

        String::from_utf8(bytes).map_err(|e| {
            pyo3::exceptions::PyUnicodeDecodeError::new_err(format!(
                "Decoded bytes are not valid UTF-8: {}",
                e
            ))
        })
    }

    /// Encode multiple texts in parallel using rayon.
    /// Returns a list of token ID vectors, one per input text.
    #[pyo3(signature = (texts))]
    #[pyo3(text_signature = "(self, texts)")]
    pub fn batch_encode(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let results = py.detach(|| {
            texts
                .par_iter()
                .map(|text| self.encode(text))
                .collect::<Vec<Vec<u32>>>()
        });

        Ok(results)
    }
}

/// Internal-only helpers. Kept in a non-pymethods impl block because PyO3
/// won't accept methods that take generic / non-Python types.
impl Tokenizer {
    /// Apply BPE merges to one pre-tokenizer chunk and append its IDs to
    /// `out`. Picks the earliest-learned (lowest `new_id`) merge each step.
    fn encode_chunk_into(&self, chunk: &str, out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

        while ids.len() >= 2 {
            let mut best_pair: Option<(usize, Pair, u32)> = None;
            for i in 0..ids.len() - 1 {
                let pair: Pair = (ids[i], ids[i + 1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                        best_pair = Some((i, pair, new_id));
                    }
                }
            }

            if let Some((idx, _pair, new_id)) = best_pair {
                ids[idx] = new_id;
                ids.remove(idx + 1);
            } else {
                break;
            }
        }

        out.extend(ids);
    }
}

#[pymodule]
pub fn wisetok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forward Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}

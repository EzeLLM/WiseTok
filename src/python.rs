//! PyO3 bindings.
//!
//! Public surface: the `Tokenizer` pyclass.

use std::collections::HashMap as StdHashMap;
use std::path::PathBuf;

use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::aggregate::{aggregate_into_counts, file as agg_file, AggregateFile};
use crate::cli_core::{materialize_and_train, parse_merge_mode, parse_pretokenizer_spec};
use crate::pretokenizer::{PreTokenizer, RegexPreTokenizer};
use crate::ram::{parse_size, RamMonitor};
use crate::special_tokens::{Segment, SpecialTokenRegistry};
use crate::Pair;

/// Start a [`RamMonitor`] if `monitor_ram` is true. Parses `ram_warn_at`
/// for the warn threshold; returns `None` if monitoring is disabled.
fn maybe_start_monitor(
    monitor_ram: bool,
    ram_warn_at: Option<&str>,
    label: &str,
) -> PyResult<Option<RamMonitor>> {
    if !monitor_ram {
        return Ok(None);
    }
    let warn = match ram_warn_at {
        None => None,
        Some(s) => Some(parse_size(s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "could not parse ram_warn_at {:?}; try \"64GB\", \"512MB\", etc.",
                s
            ))
        })?),
    };
    Ok(Some(RamMonitor::start(
        std::time::Duration::from_secs(5),
        warn,
        label,
    )))
}

// `parse_pretokenizer_spec` and `materialize_and_train` are defined in
// `crate::cli_core` and reused by both the Python bindings here and the
// CLI binary in `src/bin/wisetok.rs`. Wrapping helpers to translate
// `String` errors into `PyResult` are below.

fn py_parse_pretokenizer_spec(spec: &str) -> PyResult<(Box<dyn PreTokenizer>, String, String)> {
    parse_pretokenizer_spec(spec).map_err(pyo3::exceptions::PyValueError::new_err)
}

fn py_materialize_and_train(
    chunks: impl IntoIterator<Item = (Vec<u8>, i64)>,
    min_frequency: i64,
    vocab_size: u32,
    merge_mode: Option<&str>,
    merges_out: &mut StdHashMap<Pair, u32>,
) -> PyResult<()> {
    let mode = parse_merge_mode(merge_mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
    materialize_and_train(chunks, min_frequency, vocab_size, mode, merges_out);
    Ok(())
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
    /// Peak RSS observed during the most recent training call, in bytes.
    /// Zero if no training has been run with `monitor_ram=True`. Reset
    /// at the start of every `train_from_iterator` / `aggregate` /
    /// `train_from_aggregate` call.
    pub(crate) peak_rss_bytes: u64,
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
            peak_rss_bytes: 0,
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
    ///
    /// `merge_mode` selects the merge-loop data structure:
    ///   - `"full"`  — upstream rustbpe behavior; fast but O(N × L) RAM peak.
    ///   - `"scan"`  — drop the positions map; one parallel scan per merge.
    ///                 Bounded RAM, slightly slower per merge.
    ///   - `"auto"`  — pick `scan` when unique-word count exceeds 1M, else
    ///                 `full`. This is the default.
    /// All modes produce byte-identical merge tables on the same input.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1, pre_tokenizer=None, special_tokens=None, merge_mode=None, monitor_ram=false, ram_warn_at=None))]
    #[pyo3(
        text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1, pre_tokenizer=None, special_tokens=None, merge_mode=None, monitor_ram=False, ram_warn_at=None)"
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
        merge_mode: Option<String>,
        monitor_ram: bool,
        ram_warn_at: Option<String>,
    ) -> PyResult<()> {
        self.peak_rss_bytes = 0;
        let mut monitor = maybe_start_monitor(monitor_ram, ram_warn_at.as_deref(), "train")?;
        // Resolve the pre-tokenizer pipeline. The legacy `pattern` argument
        // and the new `pre_tokenizer` spec are mutually exclusive — let the
        // user know if they pass both rather than picking one silently.
        if pattern.is_some() && pre_tokenizer.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass either `pattern` (legacy) or `pre_tokenizer` (new), not both",
            ));
        }
        let (pre, pattern_str, _spec_canonical): (Box<dyn PreTokenizer>, String, String) =
            if let Some(spec) = pre_tokenizer {
                py_parse_pretokenizer_spec(&spec)?
            } else if let Some(pat) = pattern {
                let r = RegexPreTokenizer::new(&pat).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid regex pattern: {}", e))
                })?;
                let canonical = format!("regex:{}", pat);
                (Box::new(r), pat, canonical)
            } else {
                // Default: GPT-4 regex only (matches upstream rustbpe).
                py_parse_pretokenizer_spec("gpt4")?
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

        // Streaming ingest → chunk counts. The same helper backs `aggregate`
        // so phase-separated and single-pass training cannot diverge.
        let (counts, _stats) =
            aggregate_into_counts(py, iterator, buffer_size, pre.as_ref(), &self.specials)?;

        // Materialize + train.
        py_materialize_and_train(
            counts.into_iter().map(|(s, c)| (s.as_bytes().to_vec(), c)),
            min_frequency,
            vocab_size,
            merge_mode.as_deref(),
            &mut self.merges,
        )?;

        // Store the pipeline for later encode() calls.
        self.pre_tokenizer = Some(pre);
        if let Some(m) = monitor.as_mut() {
            m.stop();
            self.peak_rss_bytes = m.peak_bytes();
            log::info!(
                "training peak RSS: {}",
                crate::ram::format_bytes(self.peak_rss_bytes)
            );
        }
        Ok(())
    }

    /// Aggregate text from an iterator and write a `.agg` file.
    ///
    /// Reads everything from `iterator`, applies the pre-tokenizer +
    /// special-token splitter, counts unique chunks, and writes the
    /// result as a binary `.agg` file at `output_path`. **Does not run
    /// the merge loop** — call [`train_from_aggregate`] (or any number
    /// of times, with different vocab sizes) once the file exists.
    ///
    /// `pre_tokenizer`, `pattern`, `special_tokens`, and `buffer_size`
    /// have the same semantics as in [`train_from_iterator`]. The pre-
    /// tokenizer spec and special tokens are stamped into the `.agg` file
    /// so a downstream `train_from_aggregate` can reconstruct the same
    /// configuration.
    ///
    /// On success, the receiver's pre-tokenizer, pattern, and special-
    /// token registry are updated to match the aggregation config — so
    /// the same `Tokenizer` instance can be passed to
    /// `train_from_aggregate` afterwards if desired.
    #[pyo3(signature = (iterator, output_path, buffer_size=8192, pattern=None, pre_tokenizer=None, special_tokens=None, monitor_ram=false, ram_warn_at=None))]
    #[pyo3(
        text_signature = "(self, iterator, output_path, buffer_size=8192, pattern=None, pre_tokenizer=None, special_tokens=None, monitor_ram=False, ram_warn_at=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn aggregate(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        output_path: &str,
        buffer_size: usize,
        pattern: Option<String>,
        pre_tokenizer: Option<String>,
        special_tokens: Option<Vec<String>>,
        monitor_ram: bool,
        ram_warn_at: Option<String>,
    ) -> PyResult<()> {
        self.peak_rss_bytes = 0;
        let mut monitor = maybe_start_monitor(monitor_ram, ram_warn_at.as_deref(), "aggregate")?;
        if pattern.is_some() && pre_tokenizer.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pass either `pattern` (legacy) or `pre_tokenizer` (new), not both",
            ));
        }
        let (pre, pattern_str, spec_canonical): (Box<dyn PreTokenizer>, String, String) =
            if let Some(spec) = pre_tokenizer {
                py_parse_pretokenizer_spec(&spec)?
            } else if let Some(pat) = pattern {
                let r = RegexPreTokenizer::new(&pat).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid regex pattern: {}", e))
                })?;
                let canonical = format!("regex:{}", pat);
                (Box::new(r), pat, canonical)
            } else {
                py_parse_pretokenizer_spec("gpt4")?
            };

        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e))
        })?;
        self.specials = SpecialTokenRegistry::new();
        if let Some(toks) = special_tokens {
            for t in toks {
                self.specials.add(t).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("special token: {}", e))
                })?;
            }
        }

        let (counts, stats) =
            aggregate_into_counts(py, iterator, buffer_size, pre.as_ref(), &self.specials)?;

        // Build the file. Sort canonically before writing so two runs over
        // the same input produce byte-identical files.
        let mut agg = AggregateFile {
            version: crate::aggregate::AGG_VERSION as u32,
            pre_tokenizer_config: spec_canonical,
            pattern: pattern_str.clone(),
            special_tokens: self.specials.tokens().to_vec(),
            chunks: counts
                .into_iter()
                .map(|(s, c)| (s.as_bytes().to_vec(), c))
                .collect(),
            total_bytes_processed: stats.total_bytes_processed,
            total_chunks_with_multiplicity: stats.total_chunks_with_multiplicity,
        };
        agg.sort_canonical();

        let path: PathBuf = PathBuf::from(output_path);
        agg_file::write_to_file(&path, &agg).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to write .agg file {:?}: {}",
                path, e
            ))
        })?;
        log::info!(
            "wrote {:?}: {} unique chunks, {} total bytes processed",
            path,
            agg.unique_chunks(),
            agg.total_bytes_processed
        );

        // Save the pipeline so the same `Tokenizer` instance can run
        // `train_from_aggregate` next without reconstructing it.
        self.pre_tokenizer = Some(pre);
        if let Some(m) = monitor.as_mut() {
            m.stop();
            self.peak_rss_bytes = m.peak_bytes();
            log::info!(
                "aggregate peak RSS: {}",
                crate::ram::format_bytes(self.peak_rss_bytes)
            );
        }
        Ok(())
    }

    /// Run the merge loop against an `.agg` file and produce a trained
    /// tokenizer.
    ///
    /// The pre-tokenizer config and special tokens stored inside the
    /// `.agg` file are restored on `self`, so encode after this call
    /// uses the same pipeline that produced the aggregation. Override
    /// `merge_mode` and `min_frequency` per call; running this multiple
    /// times against the same `.agg` is the intended way to sweep
    /// vocab sizes.
    #[pyo3(signature = (agg_path, vocab_size, min_frequency=1, merge_mode=None, monitor_ram=false, ram_warn_at=None))]
    #[pyo3(
        text_signature = "(self, agg_path, vocab_size, min_frequency=1, merge_mode=None, monitor_ram=False, ram_warn_at=None)"
    )]
    pub fn train_from_aggregate(
        &mut self,
        agg_path: &str,
        vocab_size: u32,
        min_frequency: i64,
        merge_mode: Option<String>,
        monitor_ram: bool,
        ram_warn_at: Option<String>,
    ) -> PyResult<()> {
        self.peak_rss_bytes = 0;
        let mut monitor = maybe_start_monitor(monitor_ram, ram_warn_at.as_deref(), "merge")?;
        let path = PathBuf::from(agg_path);
        let agg = agg_file::read_from_file(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "failed to read .agg file {:?}: {}",
                path, e
            ))
        })?;
        log::info!(
            "loaded {:?}: {} unique chunks, pre_tokenizer={:?}, {} specials",
            path,
            agg.unique_chunks(),
            agg.pre_tokenizer_config,
            agg.special_tokens.len()
        );

        // Reconstruct the pre-tokenizer pipeline from the canonical spec.
        let (pre, pattern_str, _) = py_parse_pretokenizer_spec(&agg.pre_tokenizer_config)?;
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e))
        })?;

        // Restore the special-token registry from the file.
        self.specials = SpecialTokenRegistry::new();
        for t in &agg.special_tokens {
            self.specials.add(t.clone()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("special token: {}", e))
            })?;
        }

        // Materialize + train. Drop the file's `chunks` to free memory
        // before the merge loop builds its own structures.
        py_materialize_and_train(
            agg.chunks.into_iter(),
            min_frequency,
            vocab_size,
            merge_mode.as_deref(),
            &mut self.merges,
        )?;

        self.pre_tokenizer = Some(pre);
        if let Some(m) = monitor.as_mut() {
            m.stop();
            self.peak_rss_bytes = m.peak_bytes();
            log::info!(
                "merge peak RSS: {}",
                crate::ram::format_bytes(self.peak_rss_bytes)
            );
        }
        Ok(())
    }

    /// Return the regex pattern.
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Peak resident-set size, in bytes, observed during the most recent
    /// training/aggregation call that was run with `monitor_ram=True`.
    /// Returns 0 if no such call has been made yet.
    #[getter]
    pub fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss_bytes
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

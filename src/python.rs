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
use crate::{Pair, GPT4_PATTERN};

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation.
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID.
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting.
    pub pattern: String,
    /// Compiled regex for efficiency. `pub(crate)` so internal tests can
    /// construct fixtures via the struct literal; not exposed to Python.
    pub(crate) compiled_pattern: Regex,
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
        }
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust `Vec<String>` buffer under the GIL, then release the
    /// GIL to do the heavy splitting and counting **in parallel** with rayon.
    ///
    /// `min_frequency` drops chunks that occurred fewer than this many times
    /// before the merge loop runs. `min_frequency=1` keeps every chunk
    /// (legacy default; same as upstream rustbpe). Higher values shrink the
    /// word table at the cost of dropping rare chunks from training.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1))]
    #[pyo3(
        text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None, min_frequency=1)"
    )]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
        min_frequency: i64,
    ) -> PyResult<()> {
        // Use provided pattern or default to GPT-4 pattern.
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // Update the stored pattern and compile it.
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e))
        })?;

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

            let pattern = self.compiled_pattern.clone();
            let local: AHashMap<CompactString, i64> = py.detach(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i64> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = mat.expect("regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
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

    /// Encode a string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        for m in self.compiled_pattern.find_iter(text) {
            let chunk = match m {
                Ok(mat) => mat.as_str(),
                Err(e) => {
                    log::warn!("Regex match error, skipping chunk: {}", e);
                    continue;
                }
            };

            let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

            // Apply merges iteratively (always merge the earliest-learned pair first).
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

            all_ids.extend(ids);
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

#[pymodule]
pub fn wisetok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forward Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}

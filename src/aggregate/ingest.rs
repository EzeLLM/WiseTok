//! Streaming ingest from a Python iterator into chunk counts.
//!
//! Shared by `train_from_iterator` (single-pass training) and `aggregate`
//! (phase-separated). Centralizing this guarantees the two paths produce
//! byte-identical aggregations on the same input — the only correctness
//! requirement that links phase-separated training back to single-pass
//! training.

use ahash::AHashMap;
use compact_str::CompactString;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::pretokenizer::PreTokenizer;
use crate::special_tokens::{Segment, SpecialTokenRegistry};

/// Stats returned alongside the chunk-count map.
#[derive(Debug, Default, Clone, Copy)]
pub struct IngestStats {
    pub total_sequences: u64,
    pub total_bytes_processed: u64,
    pub total_chunks_with_multiplicity: u64,
}

/// Pull strings from the Python iterator and produce per-chunk counts.
///
/// The flow is exactly the one previously inlined in `train_from_iterator`:
/// refill a `Vec<String>` of size `buffer_size` under the GIL, detach the
/// GIL, parallelize the per-string split work via rayon, then reduce into
/// a global counts map.
///
/// Both Python entry points (`train_from_iterator` and `aggregate`) call
/// this so they cannot diverge.
pub(crate) fn aggregate_into_counts(
    py: Python<'_>,
    iterator: &Bound<'_, PyAny>,
    buffer_size: usize,
    pre: &dyn PreTokenizer,
    specials: &SpecialTokenRegistry,
) -> PyResult<(AHashMap<CompactString, i64>, IngestStats)> {
    // Prepare a true Python iterator object.
    let py_iter: Py<PyAny> =
        unsafe { Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))? };

    let mut counts: AHashMap<CompactString, i64> = AHashMap::new();
    let mut buf: Vec<String> = Vec::with_capacity(buffer_size);
    let mut stats = IngestStats::default();

    log::info!(
        "Processing sequences from iterator (buffer_size: {})",
        buffer_size
    );

    let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
        Python::attach(|py| {
            buf.clear();
            let it = py_iter.bind(py);
            loop {
                if buf.len() >= buffer_size {
                    return Ok(false);
                }
                let next_obj = unsafe {
                    Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                };
                match next_obj {
                    Some(obj) => {
                        let s: String = obj.extract()?;
                        buf.push(s);
                    }
                    None => {
                        if PyErr::occurred(py) {
                            return Err(PyErr::fetch(py));
                        } else {
                            return Ok(true);
                        }
                    }
                }
            }
        })
    };

    loop {
        let exhausted = refill(&mut buf)?;
        if buf.is_empty() && exhausted {
            break;
        }

        stats.total_sequences += buf.len() as u64;
        for s in &buf {
            stats.total_bytes_processed += s.len() as u64;
        }

        // Reduce returns the local counts and the local "total chunks with
        // multiplicity" so both totals stay accurate even when chunks are
        // deduplicated within a worker's local map.
        let (local, local_total): (AHashMap<CompactString, i64>, u64) = py.detach(|| {
            buf.par_iter()
                .map(|s| {
                    let mut m: AHashMap<CompactString, i64> = AHashMap::new();
                    let mut total: u64 = 0;
                    for seg in specials.split(s) {
                        if let Segment::Text(text) = seg {
                            for piece in pre.pre_tokenize(text) {
                                *m.entry(CompactString::from(piece)).or_default() += 1;
                                total += 1;
                            }
                        }
                    }
                    (m, total)
                })
                .reduce(
                    || (AHashMap::new(), 0u64),
                    |(mut a, ta), (b, tb)| {
                        for (k, v) in b {
                            *a.entry(k).or_default() += v;
                        }
                        (a, ta + tb)
                    },
                )
        });

        for (k, v) in local {
            *counts.entry(k).or_default() += v;
        }
        stats.total_chunks_with_multiplicity += local_total;

        if exhausted {
            break;
        }
    }

    log::info!(
        "Processed {} sequences total, {} unique chunks, {} chunks with multiplicity",
        stats.total_sequences,
        counts.len(),
        stats.total_chunks_with_multiplicity
    );

    Ok((counts, stats))
}

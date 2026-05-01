//! Pure-Rust helpers shared between the CLI binary and the PyO3 bindings.
//!
//! These avoid `PyResult` so the CLI doesn't drag in PyO3 at link time.
//! `python.rs` provides thin PyO3 wrappers that translate errors.

use std::collections::HashMap as StdHashMap;
use std::sync::atomic::AtomicU32;

use crate::merge::{resolve_mode, train_core, train_core_with_progress, MergeMode, Word};
use crate::pretokenizer::{DigitSplitter, PreTokenizer, RegexPreTokenizer, SequencePreTokenizer};
use crate::{Pair, GPT4_PATTERN};

/// Parse a pre-tokenizer spec string.
///
/// Returns `(pipeline, pattern, spec_canonical)`:
///   - `pipeline` is the runtime pre-tokenizer.
///   - `pattern` is the underlying regex.
///   - `spec_canonical` is the canonical spec string (e.g. always
///     `"gpt4+digits"` even if `"regex+digits:..."` was passed with the
///     GPT-4 pattern).
pub fn parse_pretokenizer_spec(
    spec: &str,
) -> Result<(Box<dyn PreTokenizer>, String, String), String> {
    if spec == "gpt4" {
        let pre = RegexPreTokenizer::new(GPT4_PATTERN)
            .map_err(|e| format!("invalid GPT4_PATTERN: {}", e))?;
        return Ok((Box::new(pre), GPT4_PATTERN.to_string(), "gpt4".to_string()));
    }
    if spec == "gpt4+digits" {
        let regex_pre = RegexPreTokenizer::new(GPT4_PATTERN)
            .map_err(|e| format!("invalid GPT4_PATTERN: {}", e))?;
        let seq = SequencePreTokenizer::new(vec![Box::new(regex_pre), Box::new(DigitSplitter)]);
        return Ok((
            Box::new(seq),
            GPT4_PATTERN.to_string(),
            "gpt4+digits".to_string(),
        ));
    }
    if let Some(pat) = spec.strip_prefix("regex:") {
        let pre =
            RegexPreTokenizer::new(pat).map_err(|e| format!("invalid regex {:?}: {}", pat, e))?;
        return Ok((Box::new(pre), pat.to_string(), format!("regex:{}", pat)));
    }
    if let Some(pat) = spec.strip_prefix("regex+digits:") {
        let regex_pre =
            RegexPreTokenizer::new(pat).map_err(|e| format!("invalid regex {:?}: {}", pat, e))?;
        let seq = SequencePreTokenizer::new(vec![Box::new(regex_pre), Box::new(DigitSplitter)]);
        return Ok((
            Box::new(seq),
            pat.to_string(),
            format!("regex+digits:{}", pat),
        ));
    }
    Err(format!(
        "unknown pre_tokenizer spec {:?}; supported: \"gpt4\", \"gpt4+digits\", \"regex:<pat>\", \"regex+digits:<pat>\"",
        spec
    ))
}

/// Parse a merge-mode string. Case-insensitive.
pub fn parse_merge_mode(s: Option<&str>) -> Result<MergeMode, String> {
    match s {
        None => Ok(MergeMode::Auto),
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "full" => Ok(MergeMode::Full),
            "scan" => Ok(MergeMode::Scan),
            "auto" => Ok(MergeMode::Auto),
            other => Err(format!(
                "unknown merge_mode {:?}; supported: \"full\", \"scan\", \"auto\"",
                other
            )),
        },
    }
}

/// Build words / counts from `(chunk_bytes, count)` pairs and run the
/// merge loop into `merges`. Filters out chunks below `min_frequency`.
pub fn materialize_and_train(
    chunks: impl IntoIterator<Item = (Vec<u8>, i64)>,
    min_frequency: i64,
    vocab_size: u32,
    merge_mode: MergeMode,
    merges_out: &mut StdHashMap<Pair, u32>,
) {
    materialize_and_train_with_progress(
        chunks,
        min_frequency,
        vocab_size,
        merge_mode,
        merges_out,
        None,
    )
}

/// Variant of [`materialize_and_train`] that publishes progress to an
/// external `AtomicU32`. The counter is incremented to `merges_done`
/// after each merge so callers can drive a progress bar from another
/// thread without touching the hot loop.
pub fn materialize_and_train_with_progress(
    chunks: impl IntoIterator<Item = (Vec<u8>, i64)>,
    min_frequency: i64,
    vocab_size: u32,
    merge_mode: MergeMode,
    merges_out: &mut StdHashMap<Pair, u32>,
    progress: Option<&AtomicU32>,
) {
    let mut total_unique = 0usize;
    let mut words: Vec<Word> = Vec::new();
    let mut cvec: Vec<i64> = Vec::new();
    for (chunk_bytes, c) in chunks {
        total_unique += 1;
        if c < min_frequency {
            continue;
        }
        words.push(Word::new(chunk_bytes.iter().map(|&b| b as u32).collect()));
        cvec.push(c);
    }
    if min_frequency > 1 {
        log::info!(
            "min_frequency={} filtered {} -> {} unique chunks",
            min_frequency,
            total_unique,
            words.len()
        );
    }

    let resolved = resolve_mode(merge_mode, words.len());
    log::info!(
        "merge_mode = {:?} (resolved from {:?}, unique_words = {})",
        resolved,
        merge_mode,
        words.len()
    );
    if let Some(p) = progress {
        train_core_with_progress(&mut words, &cvec, vocab_size, merges_out, resolved, Some(p))
    } else {
        train_core(&mut words, &cvec, vocab_size, merges_out, resolved)
    }
}

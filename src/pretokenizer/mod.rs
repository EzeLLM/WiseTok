//! Pre-tokenizers: split raw input text into chunks before BPE applies.
//!
//! The training pipeline composes one or more pre-tokenizers and then runs
//! BPE on the resulting chunks. Composing means: chunk = output of step N.
//! Step N+1 takes those chunks and may split each further. This matches
//! HuggingFace tokenizers' `Sequence` behavior.
//!
//! All implementations return `Vec<&str>` borrowed from the original input,
//! so there's no allocation per chunk beyond the result vector.

pub mod digits;
pub mod regex;
pub mod sequence;

pub use digits::DigitSplitter;
pub use regex::RegexPreTokenizer;
pub use sequence::SequencePreTokenizer;

/// Split raw text into chunks. Implementations must be `Send + Sync` so the
/// pre-tokenizer can be shared across rayon workers without cloning.
pub trait PreTokenizer: Send + Sync {
    /// Return non-overlapping slices of `text` covering the chunks.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str>;
}

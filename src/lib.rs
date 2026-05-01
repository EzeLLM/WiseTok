//! wisetok — production BPE tokenizer trainer.
//!
//! Fork of <https://github.com/karpathy/rustbpe> (MIT, copyright (c) Andrej
//! Karpathy). The merge loop, lazy-refresh heap, and parallel pre-tokenization
//! are unchanged from upstream; this crate adds production features
//! (composable pre-tokenizers, special tokens, min_frequency, HuggingFace
//! export, phase separation, validation suite, CLI) on top.

pub mod aggregate;
pub mod export;
pub(crate) mod merge;
pub mod pretokenizer;
pub mod python;
pub mod ram;
pub mod special_tokens;

/// `(left_id, right_id)` — the canonical key for BPE merges.
pub type Pair = (u32, u32);

/// Default GPT-4 style regex pattern for splitting text.
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

pub use python::Tokenizer;

#[cfg(test)]
mod tests;

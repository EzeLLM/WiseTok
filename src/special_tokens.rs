//! Special token registry.
//!
//! Special tokens are single-ID strings reserved for control purposes
//! (`<|endoftext|>`, FIM markers, etc.). They are:
//!   - Excluded from BPE merging (treated as chunk boundaries during
//!     aggregation, never split)
//!   - Matched as whole strings during encoding before BPE applies
//!   - Assigned IDs starting at `256 + num_merges`, after training
//!
//! This module just defines the data type and presets. Wiring into
//! aggregation, encoding, and export comes later.

/// Standard special tokens for code/LLM tokenizers (StarCoder/DeepSeek style).
pub const CODE_PRESET: &[&str] = &[
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|file_sep|>",
    "<|repo_name|>",
    "<|filename|>",
];

/// Standard special tokens for chat/instruction-tuned models.
pub const CHAT_PRESET: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
];

/// Ordered list of special token strings. IDs are assigned at finalization
/// time via [`SpecialTokenRegistry::assign_ids`], starting at a caller-
/// supplied base (typically `256 + num_merges`).
///
/// Insertion order is preserved; duplicates are rejected.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokenRegistry {
    tokens: Vec<String>,
}

impl SpecialTokenRegistry {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Build a registry from a preset (e.g. [`CODE_PRESET`]).
    pub fn from_preset(preset: &[&str]) -> Self {
        Self {
            tokens: preset.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Add one token. Returns `Err` if the token is already present, the
    /// empty string, or contains a NUL byte.
    pub fn add(&mut self, token: impl Into<String>) -> Result<(), SpecialTokenError> {
        let token = token.into();
        if token.is_empty() {
            return Err(SpecialTokenError::Empty);
        }
        if token.contains('\0') {
            return Err(SpecialTokenError::ContainsNul);
        }
        if self.tokens.iter().any(|t| t == &token) {
            return Err(SpecialTokenError::Duplicate(token));
        }
        self.tokens.push(token);
        Ok(())
    }

    /// Add `n` reserved placeholder tokens named `<|reserved_0|>` …
    /// `<|reserved_{n-1}|>`. Useful for keeping vocab IDs stable while
    /// leaving room to introduce new specials later.
    pub fn add_reserved(&mut self, n: usize) -> Result<(), SpecialTokenError> {
        for i in 0..n {
            self.add(format!("<|reserved_{}|>", i))?;
        }
        Ok(())
    }

    /// Return the registered tokens in insertion order.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Number of special tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Assign IDs to each token starting at `base`. Returns `(token, id)`
    /// pairs in insertion order. The caller normally passes
    /// `base = 256 + num_merges`.
    pub fn assign_ids(&self, base: u32) -> Vec<(String, u32)> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), base + i as u32))
            .collect()
    }

    /// Split `text` into segments alternating between text runs (regions
    /// that BPE will process) and special tokens (atoms that never enter
    /// BPE). Specials are matched as exact substrings; when two specials
    /// could match at the same position, the longer one wins.
    ///
    /// Empty text runs (e.g. a special at the very start, or two specials
    /// adjacent to each other) are not emitted.
    ///
    /// The returned `Segment::Special` slices borrow from the registry
    /// (`self.tokens`); `Segment::Text` slices borrow from `text`. Both
    /// share the lifetime `'a`, so callers must keep both alive for the
    /// duration of the segment usage.
    pub fn split<'a>(&'a self, text: &'a str) -> Vec<Segment<'a>> {
        if self.tokens.is_empty() || text.is_empty() {
            if text.is_empty() {
                return Vec::new();
            }
            return vec![Segment::Text(text)];
        }

        // Sort by descending length so the longest match wins on overlap.
        let mut sorted: Vec<&str> = self.tokens.iter().map(|s| s.as_str()).collect();
        sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));

        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut cursor = 0usize;

        while cursor < text.len() {
            let mut matched: Option<(usize, &str)> = None;
            // Try each special token in length-descending order at cursor.
            for special in &sorted {
                let sb = special.as_bytes();
                if cursor + sb.len() <= text.len() && &bytes[cursor..cursor + sb.len()] == sb {
                    matched = Some((sb.len(), special));
                    break;
                }
            }
            if let Some((len, special)) = matched {
                out.push(Segment::Special(special));
                cursor += len;
            } else {
                // Find the next position where any special could start. We
                // walk forward by one byte at a time looking for the first
                // candidate; this is O(n × |specials|) in the worst case
                // but fine for small registries (typical: < 50 specials).
                let start = cursor;
                cursor += 1;
                while cursor < text.len() {
                    if !text.is_char_boundary(cursor) {
                        cursor += 1;
                        continue;
                    }
                    let any_match = sorted.iter().any(|special| {
                        let sb = special.as_bytes();
                        cursor + sb.len() <= text.len() && &bytes[cursor..cursor + sb.len()] == sb
                    });
                    if any_match {
                        break;
                    }
                    cursor += 1;
                }
                // [start..cursor) is a run of plain text.
                out.push(Segment::Text(&text[start..cursor]));
            }
        }

        out
    }
}

/// A piece of input text classified as either a literal text run (which
/// will go through the pre-tokenizer + BPE) or a registered special token
/// (which is emitted as a single ID and never split).
#[derive(Debug, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Special(&'a str),
}

/// Errors when adding to a [`SpecialTokenRegistry`].
#[derive(Debug, PartialEq, Eq)]
pub enum SpecialTokenError {
    Empty,
    ContainsNul,
    Duplicate(String),
}

impl std::fmt::Display for SpecialTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "special token cannot be empty"),
            Self::ContainsNul => write!(f, "special token cannot contain NUL"),
            Self::Duplicate(t) => write!(f, "special token already registered: {:?}", t),
        }
    }
}

impl std::error::Error for SpecialTokenError {}

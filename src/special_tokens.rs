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

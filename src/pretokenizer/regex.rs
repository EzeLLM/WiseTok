use fancy_regex::Regex;

use super::PreTokenizer;

/// Splits text by a regex pattern. Each regex match becomes one chunk.
/// Pieces of `text` not matched by the regex are discarded — same as
/// rustbpe's behavior with `find_iter`.
pub struct RegexPreTokenizer {
    pattern: Regex,
    pattern_str: String,
}

impl RegexPreTokenizer {
    /// Construct from a pattern string. The pattern is compiled once;
    /// `pre_tokenize` then reuses it on every call.
    ///
    /// The error is boxed because `fancy_regex::Error` is large (>128 bytes)
    /// and we don't want every call site to pay that on the stack.
    pub fn new(pattern: &str) -> Result<Self, Box<fancy_regex::Error>> {
        let compiled = Regex::new(pattern).map_err(Box::new)?;
        Ok(Self {
            pattern: compiled,
            pattern_str: pattern.to_string(),
        })
    }

    /// The original pattern string, useful for export.
    pub fn pattern_str(&self) -> &str {
        &self.pattern_str
    }
}

impl PreTokenizer for RegexPreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        for m in self.pattern.find_iter(text) {
            match m {
                Ok(mat) => out.push(&text[mat.start()..mat.end()]),
                Err(e) => {
                    log::warn!("regex match error, skipping chunk: {}", e);
                }
            }
        }
        out
    }
}

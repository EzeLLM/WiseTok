use super::PreTokenizer;

/// Splits each ASCII digit (0–9) into its own one-byte chunk. Non-digit runs
/// pass through as a single chunk. This prevents BPE from memorizing
/// multi-digit numbers, which is standard for code tokenizers.
///
/// Examples:
///   "abc"      → ["abc"]
///   "abc123"   → ["abc", "1", "2", "3"]
///   "1.5"      → ["1", ".", "5"]
///   ""         → []
pub struct DigitSplitter;

impl PreTokenizer for DigitSplitter {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut run_start: Option<usize> = None;

        for (i, &b) in bytes.iter().enumerate() {
            if b.is_ascii_digit() {
                if let Some(start) = run_start.take() {
                    out.push(&text[start..i]);
                }
                out.push(&text[i..i + 1]);
            } else if run_start.is_none() {
                run_start = Some(i);
            }
        }

        if let Some(start) = run_start {
            out.push(&text[start..]);
        }

        out
    }
}

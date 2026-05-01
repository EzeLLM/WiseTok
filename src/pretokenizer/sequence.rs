use super::PreTokenizer;

/// Composes pre-tokenizers in order. The first step splits the input text;
/// subsequent steps split each chunk further. Output of step N becomes input
/// of step N+1.
///
/// Example: `Sequence([RegexPreTokenizer(GPT4), DigitSplitter])` first
/// applies the GPT-4 regex, then splits any digits in each resulting chunk.
pub struct SequencePreTokenizer {
    steps: Vec<Box<dyn PreTokenizer>>,
}

impl SequencePreTokenizer {
    pub fn new(steps: Vec<Box<dyn PreTokenizer>>) -> Self {
        Self { steps }
    }
}

impl PreTokenizer for SequencePreTokenizer {
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        if self.steps.is_empty() {
            return vec![text];
        }

        let mut chunks = self.steps[0].pre_tokenize(text);
        for step in &self.steps[1..] {
            let mut next = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                next.extend(step.pre_tokenize(chunk));
            }
            chunks = next;
        }
        chunks
    }
}

//! Pre-tokenizer integration tests. These exercise the public API only.

use wisetok::pretokenizer::{DigitSplitter, PreTokenizer, RegexPreTokenizer, SequencePreTokenizer};
use wisetok::GPT4_PATTERN;

#[test]
fn regex_pretokenizer_simple() {
    let pre = RegexPreTokenizer::new(r"\w+").unwrap();
    let chunks = pre.pre_tokenize("hello world");
    assert_eq!(chunks, vec!["hello", "world"]);
}

#[test]
fn regex_pretokenizer_gpt4_pattern() {
    let pre = RegexPreTokenizer::new(GPT4_PATTERN).unwrap();
    let chunks = pre.pre_tokenize("Hello world!");
    // GPT-4 pattern: "Hello", " world", "!"
    assert_eq!(chunks, vec!["Hello", " world", "!"]);
}

#[test]
fn regex_pretokenizer_invalid_regex() {
    let result = RegexPreTokenizer::new("(unclosed");
    assert!(result.is_err());
}

#[test]
fn digit_splitter_no_digits() {
    let pre = DigitSplitter;
    assert_eq!(pre.pre_tokenize("abc"), vec!["abc"]);
}

#[test]
fn digit_splitter_only_digits() {
    let pre = DigitSplitter;
    assert_eq!(pre.pre_tokenize("128"), vec!["1", "2", "8"]);
}

#[test]
fn digit_splitter_mixed() {
    let pre = DigitSplitter;
    assert_eq!(pre.pre_tokenize("abc123"), vec!["abc", "1", "2", "3"]);
    assert_eq!(pre.pre_tokenize("1.5"), vec!["1", ".", "5"]);
    assert_eq!(
        pre.pre_tokenize("v2.0.1"),
        vec!["v", "2", ".", "0", ".", "1"]
    );
}

#[test]
fn digit_splitter_empty() {
    let pre = DigitSplitter;
    let result: Vec<&str> = pre.pre_tokenize("");
    assert!(result.is_empty());
}

#[test]
fn digit_splitter_unicode_safe() {
    // Multi-byte UTF-8 chars (é is 0xC3 0xA9) must not be split — only ASCII
    // digits 0x30–0x39 split, and they never appear inside a UTF-8 sequence.
    let pre = DigitSplitter;
    assert_eq!(pre.pre_tokenize("café2"), vec!["café", "2"]);
    assert_eq!(pre.pre_tokenize("é1é"), vec!["é", "1", "é"]);
}

#[test]
fn sequence_empty() {
    let seq = SequencePreTokenizer::new(vec![]);
    // Empty sequence is identity: returns the whole text unchanged.
    assert_eq!(seq.pre_tokenize("hello"), vec!["hello"]);
}

#[test]
fn sequence_regex_then_digits() {
    let regex = RegexPreTokenizer::new(r"\w+|[^\w\s]").unwrap();
    let seq = SequencePreTokenizer::new(vec![Box::new(regex), Box::new(DigitSplitter)]);
    let chunks = seq.pre_tokenize("abc128.def");
    // step 1: ["abc128", ".", "def"]
    // step 2: ["abc", "1", "2", "8", ".", "def"]
    assert_eq!(chunks, vec!["abc", "1", "2", "8", ".", "def"]);
}

#[test]
fn sequence_gpt4_then_digits() {
    let regex = RegexPreTokenizer::new(GPT4_PATTERN).unwrap();
    let seq = SequencePreTokenizer::new(vec![Box::new(regex), Box::new(DigitSplitter)]);
    // GPT-4 pattern keeps numbers in groups of up to 3 digits; the digit
    // splitter then breaks each group into individual digits.
    let chunks = seq.pre_tokenize("v1234");
    // GPT-4 splits "v1234" → ["v", "123", "4"]
    // digit splitter → ["v", "1", "2", "3", "4"]
    assert_eq!(chunks, vec!["v", "1", "2", "3", "4"]);
}

#[test]
fn pretokenizer_is_object_safe() {
    // Ensures the trait stays object-safe (Box<dyn PreTokenizer> works).
    let _: Box<dyn PreTokenizer> = Box::new(DigitSplitter);
    let _: Box<dyn PreTokenizer> = Box::new(RegexPreTokenizer::new(r"\w+").unwrap());
}

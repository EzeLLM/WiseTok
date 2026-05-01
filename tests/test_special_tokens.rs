//! Integration tests for the special token registry.

use wisetok::special_tokens::{
    Segment, SpecialTokenError, SpecialTokenRegistry, CHAT_PRESET, CODE_PRESET,
};

#[test]
fn empty_registry() {
    let reg = SpecialTokenRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
    assert!(reg.tokens().is_empty());
    assert!(reg.assign_ids(1000).is_empty());
}

#[test]
fn add_and_assign_ids() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    reg.add("<|fim_prefix|>").unwrap();

    assert_eq!(reg.len(), 2);
    assert_eq!(
        reg.tokens(),
        &["<|endoftext|>".to_string(), "<|fim_prefix|>".to_string()]
    );

    // Caller passes base = 256 + num_merges.
    let assigned = reg.assign_ids(1024);
    assert_eq!(
        assigned,
        vec![
            ("<|endoftext|>".to_string(), 1024),
            ("<|fim_prefix|>".to_string(), 1025),
        ]
    );
}

#[test]
fn rejects_empty() {
    let mut reg = SpecialTokenRegistry::new();
    assert_eq!(reg.add(""), Err(SpecialTokenError::Empty));
}

#[test]
fn rejects_nul() {
    let mut reg = SpecialTokenRegistry::new();
    assert_eq!(reg.add("a\0b"), Err(SpecialTokenError::ContainsNul));
}

#[test]
fn rejects_duplicate() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    let err = reg.add("<|endoftext|>");
    assert_eq!(
        err,
        Err(SpecialTokenError::Duplicate("<|endoftext|>".to_string()))
    );
    // First entry remains.
    assert_eq!(reg.len(), 1);
}

#[test]
fn code_preset_loads() {
    let reg = SpecialTokenRegistry::from_preset(CODE_PRESET);
    assert_eq!(reg.len(), CODE_PRESET.len());
    assert_eq!(reg.tokens()[0], "<|endoftext|>");
    assert!(reg.tokens().iter().any(|t| t == "<|fim_prefix|>"));
}

#[test]
fn chat_preset_loads() {
    let reg = SpecialTokenRegistry::from_preset(CHAT_PRESET);
    assert_eq!(reg.len(), CHAT_PRESET.len());
    assert!(reg.tokens().iter().any(|t| t == "<|im_start|>"));
}

#[test]
fn add_reserved() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    reg.add_reserved(3).unwrap();

    assert_eq!(reg.len(), 4);
    assert_eq!(
        reg.tokens(),
        &[
            "<|endoftext|>".to_string(),
            "<|reserved_0|>".to_string(),
            "<|reserved_1|>".to_string(),
            "<|reserved_2|>".to_string(),
        ]
    );

    let assigned = reg.assign_ids(2000);
    assert_eq!(assigned[0].1, 2000);
    assert_eq!(assigned[3].1, 2003);
}

#[test]
fn add_reserved_zero_is_noop() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add_reserved(0).unwrap();
    assert!(reg.is_empty());
}

#[test]
fn ids_are_contiguous_from_base() {
    let mut reg = SpecialTokenRegistry::new();
    for i in 0..10 {
        reg.add(format!("<|tok_{}|>", i)).unwrap();
    }
    let assigned = reg.assign_ids(50_000);
    for (i, (_, id)) in assigned.iter().enumerate() {
        assert_eq!(*id, 50_000 + i as u32);
    }
}

#[test]
fn split_empty_registry() {
    let reg = SpecialTokenRegistry::new();
    assert_eq!(reg.split("hello world"), vec![Segment::Text("hello world")]);
}

#[test]
fn split_empty_text() {
    let reg = SpecialTokenRegistry::from_preset(CODE_PRESET);
    assert!(reg.split("").is_empty());
}

#[test]
fn split_text_with_no_specials() {
    let reg = SpecialTokenRegistry::from_preset(CODE_PRESET);
    assert_eq!(
        reg.split("nothing special here"),
        vec![Segment::Text("nothing special here")]
    );
}

#[test]
fn split_special_in_middle() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    let segs = reg.split("hello<|endoftext|>world");
    assert_eq!(
        segs,
        vec![
            Segment::Text("hello"),
            Segment::Special("<|endoftext|>"),
            Segment::Text("world"),
        ]
    );
}

#[test]
fn split_special_at_start() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    let segs = reg.split("<|endoftext|>after");
    assert_eq!(
        segs,
        vec![Segment::Special("<|endoftext|>"), Segment::Text("after"),]
    );
}

#[test]
fn split_special_at_end() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    let segs = reg.split("before<|endoftext|>");
    assert_eq!(
        segs,
        vec![Segment::Text("before"), Segment::Special("<|endoftext|>"),]
    );
}

#[test]
fn split_two_adjacent_specials() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|fim_prefix|>").unwrap();
    reg.add("<|fim_middle|>").unwrap();
    let segs = reg.split("<|fim_prefix|><|fim_middle|>code");
    assert_eq!(
        segs,
        vec![
            Segment::Special("<|fim_prefix|>"),
            Segment::Special("<|fim_middle|>"),
            Segment::Text("code"),
        ]
    );
}

#[test]
fn split_longest_match_wins_on_overlap() {
    // Both "<|fim|>" and "<|fim_prefix|>" could match — the longer must win.
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|fim|>").unwrap();
    reg.add("<|fim_prefix|>").unwrap();
    let segs = reg.split("<|fim_prefix|>!");
    assert_eq!(
        segs,
        vec![Segment::Special("<|fim_prefix|>"), Segment::Text("!"),]
    );
    // And the shorter one matches when it's the only option.
    let segs2 = reg.split("<|fim|>!");
    assert_eq!(segs2, vec![Segment::Special("<|fim|>"), Segment::Text("!")]);
}

#[test]
fn split_does_not_match_partial_special() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    // Almost the special, but missing the closing |>.
    let segs = reg.split("<|endoftext|");
    assert_eq!(segs, vec![Segment::Text("<|endoftext|")]);
}

#[test]
fn split_unicode_text_around_special() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|endoftext|>").unwrap();
    // Multi-byte UTF-8 in the surrounding text.
    let segs = reg.split("café<|endoftext|>résumé");
    assert_eq!(
        segs,
        vec![
            Segment::Text("café"),
            Segment::Special("<|endoftext|>"),
            Segment::Text("résumé"),
        ]
    );
}

#[test]
fn split_multiple_occurrences_of_same_special() {
    let mut reg = SpecialTokenRegistry::new();
    reg.add("<|sep|>").unwrap();
    let segs = reg.split("a<|sep|>b<|sep|>c");
    assert_eq!(
        segs,
        vec![
            Segment::Text("a"),
            Segment::Special("<|sep|>"),
            Segment::Text("b"),
            Segment::Special("<|sep|>"),
            Segment::Text("c"),
        ]
    );
}

#[test]
fn split_preset_does_not_match_when_not_present() {
    let reg = SpecialTokenRegistry::from_preset(CHAT_PRESET);
    let segs = reg.split("plain old text without specials");
    assert_eq!(segs, vec![Segment::Text("plain old text without specials")]);
}

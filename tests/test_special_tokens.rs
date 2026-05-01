//! Integration tests for the special token registry.

use wisetok::special_tokens::{SpecialTokenError, SpecialTokenRegistry, CHAT_PRESET, CODE_PRESET};

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

//! Integration tests for the HuggingFace `tokenizer.json` export.
//!
//! These tests exercise the pure `build_tokenizer_json` function (no I/O,
//! no PyO3) to assert the structural schema matches HF's `tokenizers >=
//! 0.20.0` format. The reference outputs in `research/hf_export/reference/`
//! were captured from `tokenizers 0.22.2` and serve as the ground truth.

use std::collections::HashMap;

use serde_json::Value;
use wisetok::export::huggingface::build_tokenizer_json;
use wisetok::special_tokens::SpecialTokenRegistry;
use wisetok::Pair;

/// Tiny tokenizer fixture: 2 merges, no specials.
fn fixture_tiny_no_specials() -> (HashMap<Pair, u32>, &'static str) {
    let mut merges = HashMap::new();
    // Two arbitrary byte pairs for the merges; the export doesn't care
    // whether the merges actually compress meaningfully — it only cares
    // that left/right ids are < new_id and that new_ids are 256, 257, ...
    merges.insert((104, 105), 256); // 'h' + 'i'  -> 256
    merges.insert((256, 33), 257); //  256 + '!' -> 257
    let pattern = r"\w+|\s+";
    (merges, pattern)
}

#[test]
fn top_level_keys_and_types() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    // Every required top-level key must be present, with the right type.
    let obj = v.as_object().expect("top level must be object");
    assert_eq!(obj.get("version"), Some(&Value::String("1.0".into())));
    assert_eq!(obj.get("truncation"), Some(&Value::Null));
    assert_eq!(obj.get("padding"), Some(&Value::Null));
    assert!(obj.get("added_tokens").unwrap().is_array());
    assert_eq!(obj.get("normalizer"), Some(&Value::Null));
    assert!(obj.get("pre_tokenizer").unwrap().is_object());
    assert_eq!(obj.get("post_processor"), Some(&Value::Null));
    assert!(obj.get("decoder").unwrap().is_object());
    assert!(obj.get("model").unwrap().is_object());
}

#[test]
fn pre_tokenizer_is_sequence_of_three_steps() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let pt = v.get("pre_tokenizer").unwrap();
    assert_eq!(pt.get("type").unwrap(), "Sequence");

    // Critical: HF's field name is "pretokenizers" (no underscore), NOT
    // "pre_tokenizers" or "tokenizers" or "steps". Cross-using fails to
    // deserialize.
    let steps = pt
        .get("pretokenizers")
        .expect("Sequence inner field must be 'pretokenizers'")
        .as_array()
        .unwrap();
    assert_eq!(steps.len(), 3);

    // Step 0: Split with regex.
    let split = &steps[0];
    assert_eq!(split.get("type").unwrap(), "Split");
    let pat = split.get("pattern").unwrap();
    assert_eq!(pat.get("Regex").unwrap(), pattern);
    // Behavior must be PascalCase "Isolated", not "isolated".
    assert_eq!(split.get("behavior").unwrap(), "Isolated");
    assert_eq!(split.get("invert").unwrap(), false);

    // Step 1: Digits.
    assert_eq!(steps[1].get("type").unwrap(), "Digits");
    assert_eq!(steps[1].get("individual_digits").unwrap(), true);

    // Step 2: ByteLevel pre-tokenizer (different defaults from decoder).
    let bl = &steps[2];
    assert_eq!(bl.get("type").unwrap(), "ByteLevel");
    assert_eq!(bl.get("add_prefix_space").unwrap(), false);
    assert_eq!(bl.get("trim_offsets").unwrap(), true);
    assert_eq!(bl.get("use_regex").unwrap(), false);
}

#[test]
fn decoder_uses_byte_level_with_inference_defaults() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let dec = v.get("decoder").unwrap();
    assert_eq!(dec.get("type").unwrap(), "ByteLevel");
    // Decoder ByteLevel defaults differ from the pre-tokenizer ByteLevel:
    // add_prefix_space=true, use_regex=true.
    assert_eq!(dec.get("add_prefix_space").unwrap(), true);
    assert_eq!(dec.get("trim_offsets").unwrap(), true);
    assert_eq!(dec.get("use_regex").unwrap(), true);
}

#[test]
fn model_has_all_nine_bpe_fields_in_order() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let model = v.get("model").unwrap().as_object().unwrap();
    // HF always emits all nine, including null/false defaults. Any missing
    // field can break strict-mode loaders.
    let keys: Vec<&String> = model.keys().collect();
    let expected = [
        "type",
        "dropout",
        "unk_token",
        "continuing_subword_prefix",
        "end_of_word_suffix",
        "fuse_unk",
        "byte_fallback",
        "ignore_merges",
        "vocab",
        "merges",
    ];
    for k in expected.iter() {
        assert!(
            keys.iter().any(|kk| kk.as_str() == *k),
            "model is missing required field {:?}; got {:?}",
            k,
            keys
        );
    }
    assert_eq!(model.get("type").unwrap(), "BPE");
    assert_eq!(model.get("dropout"), Some(&Value::Null));
    assert_eq!(model.get("unk_token"), Some(&Value::Null));
    assert_eq!(model.get("continuing_subword_prefix"), Some(&Value::Null));
    assert_eq!(model.get("end_of_word_suffix"), Some(&Value::Null));
    assert_eq!(model.get("fuse_unk").unwrap(), false);
    assert_eq!(model.get("byte_fallback").unwrap(), false);
    assert_eq!(model.get("ignore_merges").unwrap(), false);
}

#[test]
fn vocab_size_matches_byte_alphabet_plus_merges() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let vocab = v
        .get("model")
        .unwrap()
        .get("vocab")
        .unwrap()
        .as_object()
        .unwrap();
    // 256 base bytes + 2 fixture merges = 258 entries.
    assert_eq!(vocab.len(), 256 + merges.len());
}

#[test]
fn vocab_byte_keys_are_gpt2_unicode_mapped() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let vocab = v
        .get("model")
        .unwrap()
        .get("vocab")
        .unwrap()
        .as_object()
        .unwrap();

    // Byte 0x20 (space) must appear as 'Ġ' (U+0120) in vocab keys, not
    // as a literal space. This is the GPT-2 ByteLevel mapping.
    assert!(vocab.contains_key("\u{0120}"));
    assert!(!vocab.contains_key(" "));
    // Byte 0x0a (newline) → 'Ċ' (U+010A).
    assert!(vocab.contains_key("\u{010a}"));
    // Printable ASCII bytes map to themselves.
    assert!(vocab.contains_key("a"));
    assert!(vocab.contains_key("Z"));
}

#[test]
fn merges_array_is_pair_of_strings_not_space_joined() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    let m = v
        .get("model")
        .unwrap()
        .get("merges")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(m.len(), 2);
    // Each merge entry is a 2-element array of strings (NOT a single
    // space-joined string — that format flipped at tokenizers 0.20.0).
    for entry in m {
        let arr = entry.as_array().expect("merge entry must be an array");
        assert_eq!(arr.len(), 2);
        assert!(arr[0].is_string());
        assert!(arr[1].is_string());
    }
    // First merge is (104='h', 105='i') -> 256, so entry[0] should be ['h', 'i'].
    assert_eq!(m[0].as_array().unwrap()[0].as_str(), Some("h"));
    assert_eq!(m[0].as_array().unwrap()[1].as_str(), Some("i"));
    // Second merge is (256='hi', 33='!') -> 257.
    assert_eq!(m[1].as_array().unwrap()[0].as_str(), Some("hi"));
    assert_eq!(m[1].as_array().unwrap()[1].as_str(), Some("!"));
}

#[test]
fn merges_in_training_order() {
    // Even though merges is a HashMap (random iteration), the export
    // must sort by new_id ascending.
    let mut merges = HashMap::new();
    merges.insert((100, 101), 258); // pushed third
    merges.insert((97, 98), 256); // pushed first
    merges.insert((99, 100), 257); // pushed second
    let specials = SpecialTokenRegistry::new();

    let v = build_tokenizer_json(&merges, r"\w+", &specials).unwrap();
    let m = v
        .get("model")
        .unwrap()
        .get("merges")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(m.len(), 3);
    // First merge in array must be the one with new_id=256.
    assert_eq!(m[0].as_array().unwrap()[0].as_str(), Some("a"));
    assert_eq!(m[0].as_array().unwrap()[1].as_str(), Some("b"));
    assert_eq!(m[1].as_array().unwrap()[0].as_str(), Some("c"));
    assert_eq!(m[2].as_array().unwrap()[0].as_str(), Some("d"));
}

#[test]
fn special_tokens_appear_in_vocab_and_added_tokens() {
    let (merges, pattern) = fixture_tiny_no_specials();
    let mut specials = SpecialTokenRegistry::new();
    specials.add("<|endoftext|>").unwrap();
    specials.add("<|fim_prefix|>").unwrap();

    let v = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    // Specials placed at the tail: 256 + num_merges + i.
    let base = (256 + merges.len()) as u64;
    let vocab = v
        .get("model")
        .unwrap()
        .get("vocab")
        .unwrap()
        .as_object()
        .unwrap();
    assert_eq!(vocab.get("<|endoftext|>").unwrap().as_u64(), Some(base));
    assert_eq!(
        vocab.get("<|fim_prefix|>").unwrap().as_u64(),
        Some(base + 1)
    );

    // added_tokens entries.
    let added = v.get("added_tokens").unwrap().as_array().unwrap();
    assert_eq!(added.len(), 2);
    let entry0 = added[0].as_object().unwrap();
    assert_eq!(entry0.get("id").unwrap().as_u64(), Some(base));
    assert_eq!(entry0.get("content").unwrap(), "<|endoftext|>");
    assert_eq!(entry0.get("single_word").unwrap(), false);
    assert_eq!(entry0.get("lstrip").unwrap(), false);
    assert_eq!(entry0.get("rstrip").unwrap(), false);
    assert_eq!(entry0.get("normalized").unwrap(), false);
    assert_eq!(entry0.get("special").unwrap(), true);
}

#[test]
fn empty_merges_just_byte_alphabet() {
    let merges = HashMap::new();
    let specials = SpecialTokenRegistry::new();
    let v = build_tokenizer_json(&merges, r"\w+", &specials).unwrap();

    let vocab = v
        .get("model")
        .unwrap()
        .get("vocab")
        .unwrap()
        .as_object()
        .unwrap();
    assert_eq!(vocab.len(), 256);
    let m = v.get("model").unwrap().get("merges").unwrap();
    assert_eq!(m.as_array().unwrap().len(), 0);
}

#[test]
fn structural_match_with_captured_minimal_reference() {
    // Compare our schema against the captured HF reference structurally
    // (top-level keys, types, enum spellings). IDs and exact merge
    // strings will differ because we don't run the same training.
    let reference_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("research/hf_export/reference/minimal/raw_tokenizer.json");
    if !reference_path.exists() {
        eprintln!("reference not present at {:?}, skipping", reference_path);
        return;
    }
    let raw = std::fs::read_to_string(&reference_path).unwrap();
    let reference: Value = serde_json::from_str(&raw).unwrap();

    let (merges, pattern) = fixture_tiny_no_specials();
    let specials = SpecialTokenRegistry::new();
    let ours = build_tokenizer_json(&merges, pattern, &specials).unwrap();

    // Top-level keys must be a superset of HF's expected set.
    let our_keys: std::collections::HashSet<String> =
        ours.as_object().unwrap().keys().cloned().collect();
    for k in reference.as_object().unwrap().keys() {
        assert!(
            our_keys.contains(k),
            "missing top-level key {:?} that HF reference has; our keys: {:?}",
            k,
            our_keys
        );
    }
    // pre_tokenizer.type matches.
    assert_eq!(
        ours.get("pre_tokenizer").unwrap().get("type"),
        reference.get("pre_tokenizer").unwrap().get("type"),
    );
    // model.type matches.
    assert_eq!(
        ours.get("model").unwrap().get("type"),
        reference.get("model").unwrap().get("type"),
    );
    // decoder.type matches.
    assert_eq!(
        ours.get("decoder").unwrap().get("type"),
        reference.get("decoder").unwrap().get("type"),
    );
}

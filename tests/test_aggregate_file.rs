//! Tests for the on-disk `.agg` file format.

use std::path::PathBuf;

use wisetok::aggregate::file::{read_from_file, write_to_file};
use wisetok::aggregate::{AggregateError, AggregateFile, AGG_MAGIC, AGG_VERSION};

fn sample_agg() -> AggregateFile {
    AggregateFile {
        version: AGG_VERSION as u32,
        pre_tokenizer_config: "gpt4+digits".to_string(),
        pattern: "<gpt4-pattern>".to_string(),
        special_tokens: vec!["<|endoftext|>".to_string(), "<|fim_prefix|>".to_string()],
        chunks: vec![
            (b"hello".to_vec(), 100),
            (b"world".to_vec(), 50),
            (b" the".to_vec(), 25),
        ],
        total_bytes_processed: 1234,
        total_chunks_with_multiplicity: 250,
    }
}

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("wisetok_test_{}_{}.agg", std::process::id(), name));
    p
}

#[test]
fn roundtrip_preserves_all_fields() {
    let path = tmp_path("roundtrip");
    let agg = sample_agg();
    write_to_file(&path, &agg).unwrap();
    let loaded = read_from_file(&path).unwrap();
    assert_eq!(agg, loaded);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn empty_agg_roundtrips() {
    let path = tmp_path("empty");
    let agg = AggregateFile {
        version: AGG_VERSION as u32,
        pre_tokenizer_config: "gpt4".to_string(),
        pattern: String::new(),
        special_tokens: vec![],
        chunks: vec![],
        total_bytes_processed: 0,
        total_chunks_with_multiplicity: 0,
    };
    write_to_file(&path, &agg).unwrap();
    let loaded = read_from_file(&path).unwrap();
    assert_eq!(agg, loaded);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn detects_bad_magic() {
    let path = tmp_path("bad_magic");
    std::fs::write(&path, b"NOTWTHIS\x01extra").unwrap();
    let err = read_from_file(&path).unwrap_err();
    match err {
        AggregateError::BadMagic { found } => assert_eq!(&found, b"NOTWTHIS"),
        other => panic!("expected BadMagic, got {:?}", other),
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn detects_bad_version() {
    let path = tmp_path("bad_version");
    let mut bytes = AGG_MAGIC.to_vec();
    bytes.push(0xFF); // future version
    bytes.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, &bytes).unwrap();
    let err = read_from_file(&path).unwrap_err();
    match err {
        AggregateError::BadVersion { found, expected } => {
            assert_eq!(found, 0xFF);
            assert_eq!(expected, AGG_VERSION);
        }
        other => panic!("expected BadVersion, got {:?}", other),
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn sort_canonical_orders_by_count_desc() {
    let mut agg = AggregateFile {
        version: AGG_VERSION as u32,
        pre_tokenizer_config: "gpt4".to_string(),
        pattern: String::new(),
        special_tokens: vec![],
        chunks: vec![
            (b"a".to_vec(), 5),
            (b"b".to_vec(), 100),
            (b"c".to_vec(), 5), // tie with "a"
            (b"d".to_vec(), 50),
        ],
        total_bytes_processed: 0,
        total_chunks_with_multiplicity: 0,
    };
    agg.sort_canonical();
    assert_eq!(
        agg.chunks,
        vec![
            (b"b".to_vec(), 100),
            (b"d".to_vec(), 50),
            (b"a".to_vec(), 5), // tie broken by ascending bytes
            (b"c".to_vec(), 5),
        ]
    );
}

#[test]
fn writing_and_reading_produces_identical_files_when_input_identical() {
    // Two writes of the same agg should produce byte-identical files.
    // Useful invariant for diffing across runs.
    let path_a = tmp_path("identical_a");
    let path_b = tmp_path("identical_b");
    let agg = sample_agg();
    write_to_file(&path_a, &agg).unwrap();
    write_to_file(&path_b, &agg).unwrap();
    let bytes_a = std::fs::read(&path_a).unwrap();
    let bytes_b = std::fs::read(&path_b).unwrap();
    assert_eq!(bytes_a, bytes_b);
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn unique_chunks_matches_chunks_len() {
    let agg = sample_agg();
    assert_eq!(agg.unique_chunks(), agg.chunks.len());
}

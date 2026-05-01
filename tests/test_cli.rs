//! CLI smoke tests. Exercises `wisetok train` end-to-end and verifies it
//! produces the expected artifacts.
//!
//! These tests use `env!("CARGO_BIN_EXE_wisetok")` which Cargo populates
//! with the path to the freshly-built CLI binary, so the tests run
//! against whatever was just compiled.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_wisetok"))
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("wisetok_cli_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_corpus(dir: &Path, name: &str, content: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, content).unwrap();
    p
}

#[test]
fn cli_help_runs() {
    let out = cli().arg("--help").output().unwrap();
    assert!(
        out.status.success(),
        "stderr={:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("train"),
        "missing 'train' subcommand: {}",
        stdout
    );
    assert!(stdout.contains("validate"));
}

#[test]
fn cli_train_subcommand_help_runs() {
    let out = cli().args(["train", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in &[
        "--files",
        "--vocab-size",
        "--pre-tokenizer",
        "--special-tokens",
        "--special-preset",
        "--reserve",
        "--min-freq",
        "--merge-mode",
        "--ram-limit",
        "--threads",
        "--format",
        "--agg-file",
        "--output",
    ] {
        assert!(stdout.contains(flag), "missing {} in train help", flag);
    }
}

#[test]
fn cli_train_end_to_end_writes_artifacts() {
    let dir = tmp_dir("train_e2e");
    let corpus = write_corpus(
        &dir,
        "corpus.txt",
        "hello world the quick brown fox\n\
         hello hello hello world world world\n\
         abracadabra abracadabra\n\
         the quick brown fox jumps over the lazy dog\n\
         the year was 2025\n",
    );
    let output = dir.join("out");
    let agg = dir.join("corpus.agg");

    let status = cli()
        .args([
            "train",
            "--files",
            corpus.to_str().unwrap(),
            "--vocab-size",
            "300",
            "--pre-tokenizer",
            "gpt4+digits",
            "--special-tokens",
            "<|endoftext|>",
            "--min-freq",
            "1",
            "--output",
            output.to_str().unwrap(),
            "--agg-file",
            agg.to_str().unwrap(),
            "--format",
            "hf,tiktoken",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "CLI train failed");

    // The .agg file must exist and start with the magic bytes.
    let agg_bytes = std::fs::read(&agg).unwrap();
    assert!(agg_bytes.len() > 9);
    assert_eq!(&agg_bytes[0..8], wisetok::aggregate::AGG_MAGIC);
    assert_eq!(agg_bytes[8], wisetok::aggregate::AGG_VERSION);

    // Output must contain the four expected files.
    for f in &[
        "tokenizer.json",
        "tokenizer_config.json",
        "tiktoken.bpe",
        "tiktoken.json",
    ] {
        let p = output.join(f);
        assert!(p.exists(), "missing output {}", f);
        assert!(std::fs::metadata(&p).unwrap().len() > 0, "{} is empty", f);
    }
}

#[test]
fn cli_train_aggregate_only_mode_runs() {
    let dir = tmp_dir("agg_only");
    let corpus = write_corpus(&dir, "corpus.txt", "the quick brown fox\nhello world\n");
    let agg = dir.join("only.agg");

    let status = cli()
        .args([
            "train",
            "--files",
            corpus.to_str().unwrap(),
            "--agg-file",
            agg.to_str().unwrap(),
            // No --vocab-size: aggregation only.
        ])
        .status()
        .unwrap();
    assert!(status.success(), "aggregate-only mode failed");
    assert!(agg.exists());
}

#[test]
fn cli_train_from_existing_agg_works() {
    // First aggregate, then re-run the CLI on just the .agg file.
    let dir = tmp_dir("agg_then_train");
    let corpus = write_corpus(
        &dir,
        "c.txt",
        "hello hello world world the quick brown fox\n",
    );
    let agg = dir.join("c.agg");

    // Phase 1: aggregate.
    let status = cli()
        .args([
            "train",
            "--files",
            corpus.to_str().unwrap(),
            "--agg-file",
            agg.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    // Phase 2: merge from the .agg.
    let output = dir.join("out");
    let status = cli()
        .args([
            "train",
            "--agg-file",
            agg.to_str().unwrap(),
            "--vocab-size",
            "290",
            "--output",
            output.to_str().unwrap(),
            "--format",
            "hf",
            "--min-freq",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output.join("tokenizer.json").exists());
}

#[test]
fn cli_train_rejects_no_files_no_agg() {
    let dir = tmp_dir("no_input");
    let out = cli()
        .args([
            "train",
            "--vocab-size",
            "300",
            "--output",
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--files") || stderr.contains("--agg-file"));
}

#[test]
fn cli_train_rejects_invalid_format() {
    let dir = tmp_dir("bad_format");
    let corpus = write_corpus(&dir, "c.txt", "hello\n");
    let out = cli()
        .args([
            "train",
            "--files",
            corpus.to_str().unwrap(),
            "--vocab-size",
            "300",
            "--output",
            dir.join("out").to_str().unwrap(),
            "--format",
            "bogus",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("format") || stderr.contains("bogus"));
}

#[test]
fn cli_train_rejects_invalid_vocab_size() {
    let dir = tmp_dir("bad_vocab");
    let corpus = write_corpus(&dir, "c.txt", "hello\n");
    let out = cli()
        .args([
            "train",
            "--files",
            corpus.to_str().unwrap(),
            "--vocab-size",
            "100",
            "--output",
            dir.join("out").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("vocab"));
}

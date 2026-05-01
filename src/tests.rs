//! Internal Rust unit tests. These exercise both the public `Tokenizer`
//! surface and the `pub(crate)` merge primitives (`Word`, `MergeJob`,
//! `count_pairs_parallel`, `train_core_incremental`).

use std::collections::HashMap as StdHashMap;

use ahash::AHashSet;
use fancy_regex::Regex;

use crate::export::tiktoken::mergeable_ranks;
use crate::merge::{
    count_pairs_parallel, train_core_incremental, train_core_scan, MergeJob, Word,
    AUTO_SCAN_THRESHOLD,
};
use crate::special_tokens::SpecialTokenRegistry;
use crate::{Pair, Tokenizer};

#[test]
fn test_word_pairs() {
    let word = Word::new(vec![1, 2, 3, 4]);
    let pairs: Vec<Pair> = word.pairs().collect();
    assert_eq!(pairs, vec![(1, 2), (2, 3), (3, 4)]);
}

#[test]
fn test_word_pairs_empty() {
    let word = Word::new(vec![]);
    let pairs: Vec<Pair> = word.pairs().collect();
    assert!(pairs.is_empty());
}

#[test]
fn test_word_pairs_single() {
    let word = Word::new(vec![42]);
    let pairs: Vec<Pair> = word.pairs().collect();
    assert!(pairs.is_empty());
}

#[test]
fn test_word_merge_pair() {
    // [1, 2, 3, 1, 2] with merge (1,2) -> 99 should become [99, 3, 99]
    let mut word = Word::new(vec![1, 2, 3, 1, 2]);
    let _deltas = word.merge_pair((1, 2), 99);
    assert_eq!(word.ids, vec![99, 3, 99]);
}

#[test]
fn test_word_merge_pair_adjacent() {
    // [1, 2, 1, 2, 1, 2] -> [99, 99, 99] (non-overlapping)
    let mut word = Word::new(vec![1, 2, 1, 2, 1, 2]);
    let _deltas = word.merge_pair((1, 2), 99);
    assert_eq!(word.ids, vec![99, 99, 99]);
}

#[test]
fn test_word_merge_no_match() {
    let mut word = Word::new(vec![1, 2, 3]);
    let deltas = word.merge_pair((4, 5), 99);
    assert_eq!(word.ids, vec![1, 2, 3]); // unchanged
    assert!(deltas.is_empty() || deltas.iter().all(|(_, d)| *d == 0));
}

#[test]
fn test_tokenizer_new() {
    let tok = Tokenizer::new();
    assert!(tok.merges.is_empty());
    assert!(tok.pattern.is_empty());
}

#[test]
fn test_encode_untrained_simple() {
    // With no merges and empty pattern, encode returns nothing.
    let tok = Tokenizer::new();
    let ids = tok.encode("hello");
    assert!(ids.is_empty());
}

#[test]
fn test_encode_with_pattern_no_merges() {
    let tok = Tokenizer {
        merges: StdHashMap::new(),
        pattern: r"\w+".to_string(),
        compiled_pattern: Regex::new(r"\w+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };
    let ids = tok.encode("hi");
    // 'h' = 104, 'i' = 105
    assert_eq!(ids, vec![104, 105]);
}

#[test]
fn test_encode_with_merges() {
    let mut merges = StdHashMap::new();
    merges.insert((104, 105), 256); // 'hi' -> 256

    let tok = Tokenizer {
        merges,
        pattern: r"\w+".to_string(),
        compiled_pattern: Regex::new(r"\w+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let ids = tok.encode("hi");
    assert_eq!(ids, vec![256]);

    let ids2 = tok.encode("hip");
    assert_eq!(ids2, vec![256, 112]);
}

#[test]
fn test_get_mergeable_ranks_empty() {
    let tok = Tokenizer::new();
    let ranks = mergeable_ranks(&tok.merges);
    assert_eq!(ranks.len(), 256);
    assert_eq!(ranks[0], (vec![0u8], 0));
    assert_eq!(ranks[255], (vec![255u8], 255));
}

#[test]
fn test_get_mergeable_ranks_with_merge() {
    let mut merges = StdHashMap::new();
    merges.insert((65, 66), 256);

    let ranks = mergeable_ranks(&merges);
    assert_eq!(ranks.len(), 257);

    let last = &ranks[256];
    assert_eq!(last.0, vec![65u8, 66u8]);
    assert_eq!(last.1, 256);
}

#[test]
fn test_count_pairs_parallel() {
    let words = vec![Word::new(vec![1, 2, 3]), Word::new(vec![1, 2, 4])];
    let counts: Vec<i64> = vec![1, 2];

    let (pair_counts, positions) = count_pairs_parallel(&words, &counts);

    assert_eq!(pair_counts.get(&(1, 2)), Some(&3));
    assert_eq!(pair_counts.get(&(2, 3)), Some(&1));
    assert_eq!(pair_counts.get(&(2, 4)), Some(&2));

    assert!(positions.get(&(1, 2)).unwrap().contains(&0));
    assert!(positions.get(&(1, 2)).unwrap().contains(&1));
}

#[test]
fn test_train_core_incremental() {
    // "ab" repeated 10 times, "cd" repeated 5 times.
    let mut words = vec![
        Word::new(vec![97, 98]),  // "ab"
        Word::new(vec![99, 100]), // "cd"
    ];
    let counts: Vec<i64> = vec![10, 5];

    let mut merges = StdHashMap::new();
    train_core_incremental(&mut words, &counts, 257, &mut merges);

    assert_eq!(merges.len(), 1);
    assert!(merges.contains_key(&(97, 98)));
    assert_eq!(merges.get(&(97, 98)), Some(&256));
}

#[test]
fn test_default_trait() {
    let tok = Tokenizer::default();
    assert!(tok.merges.is_empty());
    assert!(tok.pattern.is_empty());
}

#[test]
fn test_vocab_size() {
    let mut tok = Tokenizer::new();
    assert_eq!(tok.vocab_size(), 256);

    tok.merges.insert((97, 98), 256);
    assert_eq!(tok.vocab_size(), 257);

    tok.merges.insert((256, 99), 257);
    assert_eq!(tok.vocab_size(), 258);
}

#[test]
fn test_word_merge_overlapping_pairs() {
    let mut word = Word::new(vec![97, 97, 97]);
    let _deltas = word.merge_pair((97, 97), 256);
    assert_eq!(word.ids, vec![256, 97]);
}

#[test]
fn test_word_merge_overlapping_pairs_even() {
    let mut word = Word::new(vec![97, 97, 97, 97]);
    let _deltas = word.merge_pair((97, 97), 256);
    assert_eq!(word.ids, vec![256, 256]);
}

#[test]
fn test_word_merge_multiple_occurrences() {
    let mut word = Word::new(vec![1, 2, 99, 1, 2]);
    let deltas = word.merge_pair((1, 2), 256);
    assert_eq!(word.ids, vec![256, 99, 256]);

    let ab_removals: i64 = deltas
        .iter()
        .filter(|(p, _)| *p == (1, 2))
        .map(|(_, d)| d)
        .sum();
    assert_eq!(ab_removals, -2);
}

#[test]
fn test_encode_chained_merges() {
    let mut merges = StdHashMap::new();
    merges.insert((97, 97), 256);
    merges.insert((256, 97), 257);

    let tok = Tokenizer {
        merges,
        pattern: r"\w+".to_string(),
        compiled_pattern: Regex::new(r"\w+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let ids = tok.encode("aaa");
    assert_eq!(ids, vec![257]);

    let ids = tok.encode("aaaa");
    assert_eq!(ids, vec![256, 256]);

    let ids = tok.encode("aaaaa");
    assert_eq!(ids, vec![256, 257]);
}

#[test]
fn test_encode_decode_roundtrip_simple() {
    let mut merges = StdHashMap::new();
    merges.insert((104, 105), 256);

    let tok = Tokenizer {
        merges,
        pattern: r"\w+|\s+".to_string(),
        compiled_pattern: Regex::new(r"\w+|\s+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let text = "hi";
    let ids = tok.encode(text);
    let decoded = tok.decode(ids).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_decode_roundtrip_with_spaces() {
    let mut merges = StdHashMap::new();
    merges.insert((104, 101), 256); // 'he' -> 256
    merges.insert((108, 108), 257); // 'll' -> 257
    merges.insert((256, 257), 258); // 'hell' -> 258

    let tok = Tokenizer {
        merges,
        pattern: r"\w+|\s+".to_string(),
        compiled_pattern: Regex::new(r"\w+|\s+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let text = "hello world";
    let ids = tok.encode(text);
    let decoded = tok.decode(ids).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_decode_byte_level() {
    let tok = Tokenizer::new();
    let decoded = tok.decode(vec![104, 105]).unwrap();
    assert_eq!(decoded, "hi");
}

#[test]
fn test_decode_invalid_token() {
    let tok = Tokenizer::new();
    let result = tok.decode(vec![300]);
    assert!(result.is_err());
}

#[test]
fn test_train_multiple_merges() {
    let mut words = vec![Word::new(vec![97, 98])];
    let counts: Vec<i64> = vec![10];

    let mut merges = StdHashMap::new();
    train_core_incremental(&mut words, &counts, 258, &mut merges);

    assert_eq!(merges.len(), 1);
}

#[test]
fn test_train_creates_chained_merges() {
    let mut words = vec![Word::new(vec![97, 97, 97])];
    let counts: Vec<i64> = vec![10];

    let mut merges = StdHashMap::new();
    train_core_incremental(&mut words, &counts, 258, &mut merges);

    assert_eq!(merges.len(), 2);
    assert_eq!(merges.get(&(97, 97)), Some(&256));
    assert_eq!(merges.get(&(256, 97)), Some(&257));
}

#[test]
fn test_get_mergeable_ranks_chained() {
    let mut merges = StdHashMap::new();
    merges.insert((65, 66), 256); // 'AB' -> 256
    merges.insert((256, 67), 257); // 'ABC' -> 257

    let ranks = mergeable_ranks(&merges);
    assert_eq!(ranks.len(), 258);
    assert_eq!(ranks[256], (vec![65u8, 66u8], 256));
    assert_eq!(ranks[257], (vec![65u8, 66u8, 67u8], 257));
}

#[test]
fn test_encode_empty_string() {
    let tok = Tokenizer {
        merges: StdHashMap::new(),
        pattern: r"\w+".to_string(),
        compiled_pattern: Regex::new(r"\w+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let ids = tok.encode("");
    assert!(ids.is_empty());
}

#[test]
fn test_encode_no_matches() {
    let tok = Tokenizer {
        merges: StdHashMap::new(),
        pattern: r"\w+".to_string(),
        compiled_pattern: Regex::new(r"\w+").unwrap(),
        pre_tokenizer: None,
        specials: SpecialTokenRegistry::new(),
    };

    let ids = tok.encode("   ");
    assert!(ids.is_empty());
}

#[test]
fn test_decode_empty() {
    let tok = Tokenizer::new();
    let decoded = tok.decode(vec![]).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn test_word_merge_deltas_correctness() {
    // Word: [1, 2, 3, 1, 2] with merge (1, 2) -> 99
    let mut word = Word::new(vec![1, 2, 3, 1, 2]);
    let deltas = word.merge_pair((1, 2), 99);

    let mut delta_map: StdHashMap<Pair, i64> = StdHashMap::new();
    for (pair, delta) in deltas {
        *delta_map.entry(pair).or_default() += delta;
    }

    assert_eq!(delta_map.get(&(1, 2)), Some(&-2));
    assert_eq!(delta_map.get(&(2, 3)), Some(&-1));
    assert_eq!(delta_map.get(&(3, 1)), Some(&-1));
    assert_eq!(delta_map.get(&(99, 3)), Some(&1));
    assert_eq!(delta_map.get(&(3, 99)), Some(&1));
}

#[test]
fn test_count_pairs_parallel_empty() {
    let words: Vec<Word> = vec![];
    let counts: Vec<i64> = vec![];

    let (pair_counts, positions) = count_pairs_parallel(&words, &counts);
    assert!(pair_counts.is_empty());
    assert!(positions.is_empty());
}

#[test]
fn test_count_pairs_parallel_zero_count() {
    let words = vec![Word::new(vec![1, 2, 3])];
    let counts: Vec<i64> = vec![0];

    let (pair_counts, _positions) = count_pairs_parallel(&words, &counts);
    assert!(pair_counts.is_empty());
}

#[test]
fn test_merge_job_ord() {
    // Higher count wins; equal count → smaller pair wins (ascending).
    let a = MergeJob {
        pair: (1, 2),
        count: 10,
        pos: AHashSet::new(),
    };
    let b = MergeJob {
        pair: (3, 4),
        count: 5,
        pos: AHashSet::new(),
    };
    assert!(a > b);

    let c = MergeJob {
        pair: (1, 2),
        count: 5,
        pos: AHashSet::new(),
    };
    let d = MergeJob {
        pair: (3, 4),
        count: 5,
        pos: AHashSet::new(),
    };
    // Same count, ascending pair order: (1,2) wins over (3,4).
    assert!(c > d);
}

// -----------------------------------------------------------------------
// MergeMode parity: Full and Scan must produce byte-identical merge
// tables on the same input. This is the gating correctness criterion
// for the memory-bounded merge mode (Iteration 2). If these tests
// diverge, the scan implementation is wrong — fix it before shipping.
// -----------------------------------------------------------------------

/// Run both Full and Scan modes on the same fixture and assert the two
/// `merges` maps are byte-for-byte equal (same pairs, same `new_id`s).
fn assert_modes_agree(words_init: Vec<Word>, counts: Vec<i64>, vocab_size: u32) {
    let mut words_full = words_init.clone();
    let mut merges_full: StdHashMap<Pair, u32> = StdHashMap::new();
    train_core_incremental(&mut words_full, &counts, vocab_size, &mut merges_full);

    let mut words_scan = words_init.clone();
    let mut merges_scan: StdHashMap<Pair, u32> = StdHashMap::new();
    train_core_scan(&mut words_scan, &counts, vocab_size, &mut merges_scan);

    assert_eq!(
        merges_full, merges_scan,
        "Full and Scan produced different merge tables\nFull: {:?}\nScan: {:?}",
        merges_full, merges_scan
    );

    // The post-merge `words` arrays must also match exactly: same lengths
    // and same id sequences. This catches any divergence in how the two
    // modes traversed the words during merge_pair application.
    assert_eq!(
        words_full.len(),
        words_scan.len(),
        "word vector lengths diverged"
    );
    for (i, (wf, ws)) in words_full.iter().zip(words_scan.iter()).enumerate() {
        assert_eq!(
            wf.ids, ws.ids,
            "word index {} diverged between Full and Scan: {:?} vs {:?}",
            i, wf.ids, ws.ids
        );
    }
}

#[test]
fn test_modes_agree_single_pair() {
    // The "ab" repeated 10 times, "cd" 5 times fixture from
    // test_train_core_incremental.
    let words = vec![Word::new(vec![97, 98]), Word::new(vec![99, 100])];
    let counts: Vec<i64> = vec![10, 5];
    assert_modes_agree(words, counts, 257);
}

#[test]
fn test_modes_agree_chained_merges() {
    // "aaa" forces a chained merge: first (97,97)→256, then (256,97)→257.
    let words = vec![Word::new(vec![97, 97, 97])];
    let counts: Vec<i64> = vec![10];
    assert_modes_agree(words, counts, 258);
}

#[test]
fn test_modes_agree_overlapping_repeats() {
    // "aaaa" exercises the non-overlapping replacement logic and
    // forces deltas with both removed-and-recreated pairs.
    let words = vec![
        Word::new(vec![97, 97, 97, 97]),
        Word::new(vec![97, 97, 97, 97, 97]),
        Word::new(vec![98, 97, 97, 99]),
    ];
    let counts: Vec<i64> = vec![3, 2, 1];
    assert_modes_agree(words, counts, 260);
}

#[test]
fn test_modes_agree_diverse_corpus() {
    // Several distinct words with shared bytes. Forces the heap to
    // make non-trivial decisions across competing pairs.
    let words = vec![
        Word::new(b"hello".iter().map(|&b| b as u32).collect()),
        Word::new(b"world".iter().map(|&b| b as u32).collect()),
        Word::new(b"hello world".iter().map(|&b| b as u32).collect()),
        Word::new(b"abracadabra".iter().map(|&b| b as u32).collect()),
        Word::new(b"the quick brown fox".iter().map(|&b| b as u32).collect()),
        Word::new(
            b"jumped over the lazy dog"
                .iter()
                .map(|&b| b as u32)
                .collect(),
        ),
    ];
    let counts: Vec<i64> = vec![100, 80, 50, 30, 20, 10];
    assert_modes_agree(words, counts, 320);
}

#[test]
fn test_modes_agree_tied_counts() {
    // Two pairs with equal frequency. Tie-break must be deterministic
    // across modes — ascending pair order. Mismatch here would indicate
    // ScanJob::cmp doesn't match MergeJob::cmp.
    let words = vec![Word::new(vec![1, 2, 3, 4]), Word::new(vec![5, 6, 7, 8])];
    let counts: Vec<i64> = vec![10, 10];
    assert_modes_agree(words, counts, 260);
}

#[test]
fn test_modes_agree_zero_count_word_skipped() {
    // A word with count==0 must be skipped by both modes.
    let words = vec![
        Word::new(vec![97, 98]),
        Word::new(vec![99, 100]), // count=0, must contribute nothing
    ];
    let counts: Vec<i64> = vec![5, 0];
    assert_modes_agree(words, counts, 257);
}

#[test]
fn test_modes_agree_vocab_exhausts_pairs() {
    // Request more merges than there are unique pairs available. Both
    // modes must terminate cleanly with the same partial merges map.
    let words = vec![Word::new(vec![97, 98])];
    let counts: Vec<i64> = vec![10];
    assert_modes_agree(words, counts, 300);
}

#[test]
fn test_modes_agree_singleton_words_skipped() {
    // Words of length < 2 produce no pairs in either mode.
    let words = vec![
        Word::new(vec![97]),
        Word::new(vec![]),
        Word::new(vec![98, 99, 100]),
    ];
    let counts: Vec<i64> = vec![100, 50, 10];
    assert_modes_agree(words, counts, 258);
}

#[test]
fn test_resolve_mode_auto() {
    use crate::merge::{resolve_mode, MergeMode};
    assert_eq!(resolve_mode(MergeMode::Auto, 0), MergeMode::Full);
    assert_eq!(
        resolve_mode(MergeMode::Auto, AUTO_SCAN_THRESHOLD),
        MergeMode::Full
    );
    assert_eq!(
        resolve_mode(MergeMode::Auto, AUTO_SCAN_THRESHOLD + 1),
        MergeMode::Scan
    );
    // Explicit modes pass through.
    assert_eq!(resolve_mode(MergeMode::Full, 999_999_999), MergeMode::Full);
    assert_eq!(resolve_mode(MergeMode::Scan, 0), MergeMode::Scan);
}

#[test]
fn test_scan_with_zero_words() {
    // Empty input — both modes must produce empty merges.
    let mut words: Vec<Word> = vec![];
    let counts: Vec<i64> = vec![];
    let mut merges_full: StdHashMap<Pair, u32> = StdHashMap::new();
    let mut merges_scan: StdHashMap<Pair, u32> = StdHashMap::new();
    train_core_incremental(&mut words.clone(), &counts, 300, &mut merges_full);
    train_core_scan(&mut words, &counts, 300, &mut merges_scan);
    assert!(merges_full.is_empty());
    assert!(merges_scan.is_empty());
}

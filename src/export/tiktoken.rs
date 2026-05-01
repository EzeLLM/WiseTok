use std::collections::HashMap as StdHashMap;

use crate::Pair;

/// Build the (token_bytes, rank) list that `tiktoken.Encoding(mergeable_ranks=...)`
/// expects. Walks the trained `merges` map sorted by `new_id` and concatenates
/// the byte sequences of the two parents.
pub fn mergeable_ranks(merges: &StdHashMap<Pair, u32>) -> Vec<(Vec<u8>, u32)> {
    let mut mergeable_ranks = Vec::with_capacity(256 + merges.len());

    // Build vocabulary incrementally from low to high token IDs.
    let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

    for (i, bytes) in token_bytes.iter().enumerate() {
        mergeable_ranks.push((bytes.clone(), i as u32));
    }

    // Sort merges by token id (so we can reconstruct bytes progressively).
    let mut sorted_merges: Vec<_> = merges.iter().collect();
    sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

    for (&pair, &merged_id) in sorted_merges {
        let (left, right) = pair;
        let mut merged_bytes = token_bytes[left as usize].clone();
        merged_bytes.extend(&token_bytes[right as usize]);

        if token_bytes.len() <= merged_id as usize {
            token_bytes.resize(merged_id as usize + 1, Vec::new());
        }
        token_bytes[merged_id as usize] = merged_bytes.clone();

        mergeable_ranks.push((merged_bytes, merged_id));
    }

    mergeable_ranks
}

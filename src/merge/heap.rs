use std::cmp::Ordering;

use ahash::AHashSet;

use crate::Pair;

/// One entry in the lazy-refresh max-heap. `count` is the global pair count
/// at the time this entry was pushed; if it diverges from the live global
/// before processing, the merge loop re-pushes with the updated count.
/// `pos` is the set of word indices the pusher knew contained this pair.
#[derive(Debug, Eq)]
pub(crate) struct MergeJob {
    pub(crate) pair: Pair,
    pub(crate) count: u64,
    pub(crate) pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break to ascending pair order (deterministic).
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

use std::cmp::Ordering;

use ahash::AHashSet;

use crate::Pair;

/// One entry in the lazy-refresh max-heap used by [`MergeMode::Full`].
///
/// `count` is the global pair count at the time this entry was pushed; if it
/// diverges from the live global before processing, the merge loop re-pushes
/// with the updated count. `pos` is the set of word indices the pusher knew
/// contained this pair — Full mode's defining feature, and the reason it
/// uses O(N × L) memory.
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

/// Heap entry for [`MergeMode::Scan`] — same pair-selection ordering as
/// [`MergeJob`], but without the position set. Scan mode discovers positions
/// for the winning pair via a linear pass over `words` each iteration, so
/// the heap entry only needs to carry priority information.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub(crate) struct ScanJob {
    pub(crate) pair: Pair,
    pub(crate) count: u64,
}

impl PartialOrd for ScanJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScanJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Identical ordering to MergeJob: max-heap by count, tie-break on
        // ascending pair. This keeps Full and Scan picking the same winner
        // when their live pair_counts agree.
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

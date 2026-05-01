pub(crate) mod bpe;
pub(crate) mod heap;
pub(crate) mod word;

pub use bpe::MergeMode;

pub(crate) use bpe::{resolve_mode, train_core};
pub(crate) use word::Word;

#[cfg(test)]
pub(crate) use bpe::{
    count_pairs_parallel, train_core_incremental, train_core_scan, AUTO_SCAN_THRESHOLD,
};
#[cfg(test)]
pub(crate) use heap::MergeJob;

pub(crate) mod bpe;
pub(crate) mod heap;
pub(crate) mod word;

pub(crate) use bpe::train_core_incremental;
pub(crate) use word::Word;

#[cfg(test)]
pub(crate) use bpe::count_pairs_parallel;
#[cfg(test)]
pub(crate) use heap::MergeJob;

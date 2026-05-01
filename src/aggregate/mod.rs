//! Phase-separated aggregation: streams text → byte-level chunk counts → `.agg` file.
//!
//! The training pipeline has two distinct phases with very different cost
//! profiles:
//!   - **Aggregation** is I/O-bound and embarrassingly parallel. It reads
//!     text, applies the pre-tokenizer, splits on special tokens, and counts
//!     unique chunks. On a 35GB corpus this dominates wall-clock time.
//!   - **Merging** is CPU-bound and serial in the hot loop. It runs the BPE
//!     algorithm against the aggregated chunk counts. Re-running merge with
//!     a different `vocab_size` or `min_frequency` does not require re-reading
//!     the corpus.
//!
//! `.agg` files let users separate these phases: aggregate once, merge many
//! times. They also make the aggregation result inspectable, debuggable,
//! and shareable.
//!
//! # Format
//!
//! ```text
//! magic: 8 bytes = b"WISETKAG"
//! version: 1 byte = 0x01
//! payload: bincode-2 encoded `AggregateFile`
//! ```
//!
//! The magic + version prefix lets us detect "not a .agg file" and
//! "incompatible version" errors with a clear message instead of a corrupt
//! bincode error.

pub mod file;
pub(crate) mod ingest;

pub use file::{AggregateError, AggregateFile, AGG_MAGIC, AGG_VERSION};
pub(crate) use ingest::aggregate_into_counts;

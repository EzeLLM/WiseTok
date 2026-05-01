//! On-disk `.agg` file format.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::{Decode, Encode};

/// 8-byte file magic. Reads as `WISETKAG` in ASCII; the suffix is short
/// for "aggregate."
pub const AGG_MAGIC: &[u8; 8] = b"WISETKAG";

/// Current on-disk schema version. Bump on incompatible changes; readers
/// must reject anything they don't recognize.
pub const AGG_VERSION: u8 = 1;

/// Errors when reading or writing an `.agg` file.
#[derive(Debug)]
pub enum AggregateError {
    Io(io::Error),
    Encode(bincode::error::EncodeError),
    Decode(bincode::error::DecodeError),
    /// File didn't start with the expected magic bytes.
    BadMagic {
        found: [u8; 8],
    },
    /// File version doesn't match `AGG_VERSION`. Older or newer crates
    /// produced this file.
    BadVersion {
        found: u8,
        expected: u8,
    },
}

impl From<io::Error> for AggregateError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bincode::error::EncodeError> for AggregateError {
    fn from(e: bincode::error::EncodeError) -> Self {
        Self::Encode(e)
    }
}

impl From<bincode::error::DecodeError> for AggregateError {
    fn from(e: bincode::error::DecodeError) -> Self {
        Self::Decode(e)
    }
}

impl std::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Encode(e) => write!(f, "bincode encode error: {}", e),
            Self::Decode(e) => write!(f, "bincode decode error: {}", e),
            Self::BadMagic { found } => write!(
                f,
                "not a wisetok .agg file: expected magic {:?}, found {:?}",
                AGG_MAGIC, found
            ),
            Self::BadVersion { found, expected } => write!(
                f,
                ".agg version mismatch: file is v{}, this build expects v{}",
                found, expected
            ),
        }
    }
}

impl std::error::Error for AggregateError {}

/// Aggregated chunk counts, ready to feed into the merge loop.
///
/// The chunks are stored in *insertion order from the aggregator*. The
/// `write_to_file` helper sorts them by `(count desc, bytes asc)` for
/// human inspection and for stable on-disk output across runs that hash
/// the same input.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct AggregateFile {
    /// Schema version, currently [`AGG_VERSION`]. Stored inside the bincode
    /// payload as well as in the file header for redundancy.
    pub version: u32,
    /// Canonical pre-tokenizer spec. Mirrors what `train_from_iterator`
    /// accepts: `"gpt4"`, `"gpt4+digits"`, `"regex:<pat>"`,
    /// `"regex+digits:<pat>"`.
    pub pre_tokenizer_config: String,
    /// Underlying regex string. For `"gpt4"` and `"gpt4+digits"` this is
    /// the GPT-4 pattern. For `"regex:..."` and `"regex+digits:..."` it
    /// is the user-supplied pattern. Stored separately so the merge phase
    /// can stamp it onto the resulting tokenizer without re-parsing the
    /// spec.
    pub pattern: String,
    /// Special tokens registered during aggregation, in insertion order.
    /// These were skipped during BPE counting and must be re-registered
    /// on the trained tokenizer so encode handles them atomically.
    pub special_tokens: Vec<String>,
    /// `(chunk_bytes, count)` pairs. Each chunk is the raw byte content
    /// of one pre-tokenizer output piece. `count` is the integer multiplicity
    /// across the whole input corpus.
    pub chunks: Vec<(Vec<u8>, i64)>,
    /// Total bytes read from the input iterator. Useful for progress
    /// reporting and back-of-envelope checks.
    pub total_bytes_processed: u64,
    /// Sum of `count` across all chunks. Equivalent to the total number
    /// of pre-tokenizer pieces emitted across the input.
    pub total_chunks_with_multiplicity: u64,
}

impl AggregateFile {
    /// Number of unique chunks (i.e. `chunks.len()`).
    pub fn unique_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Sort chunks descending by count, with ties broken ascending by
    /// bytes. Stable across runs, which makes diffing two `.agg` files
    /// meaningful.
    pub fn sort_canonical(&mut self) {
        self.chunks.sort_by(|a, b| match b.1.cmp(&a.1) {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
    }
}

/// Bincode config used for `.agg` files. Standard config is fine; we just
/// pin it explicitly so a future bincode default change can't silently
/// break old files.
fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Write `agg` to `path`, prefixed with the magic header.
pub fn write_to_file(path: &Path, agg: &AggregateFile) -> Result<(), AggregateError> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(AGG_MAGIC)?;
    w.write_all(&[AGG_VERSION])?;
    bincode::encode_into_std_write(agg, &mut w, bincode_config())?;
    w.flush()?;
    Ok(())
}

/// Read an `.agg` file from `path`, validating magic + version.
pub fn read_from_file(path: &Path) -> Result<AggregateFile, AggregateError> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != AGG_MAGIC {
        return Err(AggregateError::BadMagic { found: magic });
    }
    let mut version = [0u8; 1];
    r.read_exact(&mut version)?;
    if version[0] != AGG_VERSION {
        return Err(AggregateError::BadVersion {
            found: version[0],
            expected: AGG_VERSION,
        });
    }
    let agg: AggregateFile = bincode::decode_from_std_read(&mut r, bincode_config())?;
    Ok(agg)
}

//! `wisetok` CLI — thin wrapper over the library.
//!
//! Subcommands:
//!   - `train`    aggregate (optionally) + merge + export
//!   - `validate` smoke-test a trained tokenizer

use std::collections::HashMap as StdHashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use wisetok::aggregate::{
    aggregate_into_counts_rust, file as agg_file, AggregateFile, AGG_VERSION,
};
use wisetok::cli_core::{
    materialize_and_train_with_progress, parse_merge_mode, parse_pretokenizer_spec,
};
use wisetok::export::{huggingface, tiktoken as tt};
use wisetok::ram::{format_bytes, parse_size, RamMonitor};
use wisetok::special_tokens::{SpecialTokenRegistry, CHAT_PRESET, CODE_PRESET};

#[derive(Parser, Debug)]
#[command(name = "wisetok", version, about = "Production BPE tokenizer trainer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train a tokenizer end-to-end (aggregate + merge + export).
    Train(Box<TrainArgs>),
    /// Validate a trained tokenizer against test inputs.
    Validate(ValidateArgs),
}

#[derive(Parser, Debug)]
struct TrainArgs {
    /// Input files. Each file is read line-by-line; each line becomes one
    /// input sequence. Can be repeated. Required unless `--agg-file`
    /// already exists and you only want to merge.
    #[arg(long, num_args = 1..)]
    files: Vec<PathBuf>,

    /// Target vocabulary size (256 + num_merges). Required to run the
    /// merge phase. Omit to aggregate only.
    #[arg(long)]
    vocab_size: Option<u32>,

    /// Pre-tokenizer pipeline. Recognized:
    ///   "gpt4", "gpt4+digits", "regex:<pat>", "regex+digits:<pat>".
    #[arg(long, default_value = "gpt4+digits")]
    pre_tokenizer: String,

    /// Special tokens to register (repeatable).
    #[arg(long, num_args = 1..)]
    special_tokens: Vec<String>,

    /// Special-token preset: `code` or `chat`. Mutually exclusive with
    /// individual `--special-tokens`.
    #[arg(long)]
    special_preset: Option<String>,

    /// Add N reserved placeholder tokens (`<|reserved_0|>` ...).
    #[arg(long, default_value_t = 0)]
    reserve: usize,

    /// Drop chunks below this count before merging.
    #[arg(long, default_value_t = 2)]
    min_freq: i64,

    /// Merge-loop strategy: "full", "scan", or "auto" (default).
    #[arg(long, default_value = "auto")]
    merge_mode: String,

    /// Approximate RAM ceiling (e.g. "64GB"). Triggers a one-shot
    /// warning when the sampled RSS exceeds it.
    #[arg(long)]
    ram_limit: Option<String>,

    /// Number of rayon worker threads. Default: 80% of available cores.
    #[arg(long)]
    threads: Option<usize>,

    /// Comma-separated output formats: `hf` and/or `tiktoken`.
    #[arg(long, default_value = "hf,tiktoken")]
    format: String,

    /// `.agg` file path. With `--files`, save the aggregation here. Without
    /// `--files`, load and merge from this existing file.
    #[arg(long)]
    agg_file: Option<PathBuf>,

    /// Output directory for the trained tokenizer. Required when
    /// `--vocab-size` is set (i.e. when merging).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Buffer size for the streaming aggregator.
    #[arg(long, default_value_t = 8192)]
    buffer_size: usize,

    /// Verbose output: log INFO-level messages.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct ValidateArgs {
    /// Tokenizer directory (must contain `tokenizer.json` from a
    /// previous `wisetok train` run).
    #[arg(long)]
    tokenizer: PathBuf,

    /// Test files to encode/decode. One sequence per line.
    #[arg(long, num_args = 1..)]
    test_files: Vec<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Train(args) => run_train(*args),
        Command::Validate(args) => run_validate(args),
    };
    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn init_logger(verbose: bool) {
    let level = if verbose { "info" } else { "warn" };
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp_secs()
        .try_init();
}

fn configure_thread_pool(requested: Option<usize>) -> Result<(), String> {
    let n = requested.unwrap_or_else(|| {
        // 80% of available cores, minimum 1.
        let n = num_cpus_approx();
        ((n * 4) / 5).max(1)
    });
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| format!("rayon thread pool init failed: {}", e))?;
    log::info!("rayon worker threads: {}", n);
    Ok(())
}

/// Best-effort logical CPU count without pulling in another dependency.
fn num_cpus_approx() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Build a special-token registry from CLI args. Mutually exclusive
/// preset / explicit list; both can be empty.
fn build_specials(args: &TrainArgs) -> Result<SpecialTokenRegistry, String> {
    if !args.special_tokens.is_empty() && args.special_preset.is_some() {
        return Err("pass either --special-tokens or --special-preset, not both".to_string());
    }
    let mut reg = SpecialTokenRegistry::new();
    if let Some(preset) = &args.special_preset {
        match preset.to_ascii_lowercase().as_str() {
            "code" => {
                for t in CODE_PRESET {
                    reg.add(*t).map_err(|e| e.to_string())?;
                }
            }
            "chat" => {
                for t in CHAT_PRESET {
                    reg.add(*t).map_err(|e| e.to_string())?;
                }
            }
            other => {
                return Err(format!(
                    "unknown --special-preset {:?}; try \"code\" or \"chat\"",
                    other
                ))
            }
        }
    }
    for t in &args.special_tokens {
        reg.add(t.clone()).map_err(|e| e.to_string())?;
    }
    if args.reserve > 0 {
        reg.add_reserved(args.reserve).map_err(|e| e.to_string())?;
    }
    Ok(reg)
}

/// Parse the comma-separated `--format` arg into (write_hf, write_tiktoken).
fn parse_formats(s: &str) -> Result<(bool, bool), String> {
    let mut hf = false;
    let mut tt = false;
    for tok in s.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match tok.to_ascii_lowercase().as_str() {
            "hf" | "huggingface" => hf = true,
            "tiktoken" => tt = true,
            other => {
                return Err(format!(
                    "unknown format {:?}; supported: \"hf\", \"tiktoken\"",
                    other
                ))
            }
        }
    }
    if !hf && !tt {
        return Err("--format must include at least one of: hf, tiktoken".to_string());
    }
    Ok((hf, tt))
}

/// Lazy iterator over lines from a list of text files. Each line becomes
/// one `String`. Errors are logged and skipped (rather than aborting,
/// since a single corrupt line shouldn't kill an hours-long training).
struct LinesFromFiles {
    paths: std::vec::IntoIter<PathBuf>,
    current: Option<std::io::Lines<BufReader<File>>>,
    bytes_pb: Option<ProgressBar>,
}

impl LinesFromFiles {
    fn new(paths: Vec<PathBuf>, bytes_pb: Option<ProgressBar>) -> Self {
        Self {
            paths: paths.into_iter(),
            current: None,
            bytes_pb,
        }
    }

    fn open_next(&mut self) -> bool {
        match self.paths.next() {
            None => false,
            Some(p) => match File::open(&p) {
                Ok(f) => {
                    log::info!("reading {}", p.display());
                    self.current = Some(BufReader::new(f).lines());
                    true
                }
                Err(e) => {
                    log::warn!("failed to open {}: {} (skipping)", p.display(), e);
                    self.open_next()
                }
            },
        }
    }
}

impl Iterator for LinesFromFiles {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        loop {
            if self.current.is_none() && !self.open_next() {
                return None;
            }
            match self.current.as_mut().unwrap().next() {
                Some(Ok(line)) => {
                    if let Some(pb) = &self.bytes_pb {
                        pb.inc(line.len() as u64 + 1);
                    }
                    return Some(line);
                }
                Some(Err(e)) => {
                    log::warn!("read error: {} (skipping line)", e);
                }
                None => {
                    self.current = None;
                }
            }
        }
    }
}

fn make_bytes_pb() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} aggregate: {bytes} ({bytes_per_sec}) {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(200));
    pb
}

fn make_merge_pb(total_merges: u32) -> ProgressBar {
    let pb = ProgressBar::new(total_merges as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "merge [{bar:40}] {pos}/{len} ({percent}%) eta {eta_precise} {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb
}

fn run_train(args: TrainArgs) -> Result<(), String> {
    init_logger(args.verbose);
    configure_thread_pool(args.threads)?;

    let (write_hf, write_tt) = parse_formats(&args.format)?;
    let specials = build_specials(&args)?;
    let (pre, pattern_str, spec_canonical) = parse_pretokenizer_spec(&args.pre_tokenizer)?;
    let merge_mode = parse_merge_mode(Some(&args.merge_mode))?;

    let warn_threshold = match args.ram_limit.as_deref() {
        None => None,
        Some(s) => Some(
            parse_size(s)
                .ok_or_else(|| format!("could not parse --ram-limit {:?}; try \"64GB\"", s))?,
        ),
    };
    let mut monitor = RamMonitor::start(Duration::from_secs(5), warn_threshold, "train");

    // Stage 1: get an AggregateFile, either by aggregating --files or by
    // loading an existing --agg-file.
    let agg = if !args.files.is_empty() {
        let bytes_pb = make_bytes_pb();
        let line_iter = LinesFromFiles::new(args.files.clone(), Some(bytes_pb.clone()));

        log::info!(
            "aggregating with pre_tokenizer={:?}, {} specials",
            spec_canonical,
            specials.len()
        );
        let t0 = Instant::now();
        let (counts, stats) =
            aggregate_into_counts_rust(line_iter, args.buffer_size, pre.as_ref(), &specials, None);
        bytes_pb.finish_with_message(format!(
            "{} unique chunks in {:.1}s",
            counts.len(),
            t0.elapsed().as_secs_f64()
        ));

        let mut agg = AggregateFile {
            version: AGG_VERSION as u32,
            pre_tokenizer_config: spec_canonical.clone(),
            pattern: pattern_str.clone(),
            special_tokens: specials.tokens().to_vec(),
            chunks: counts
                .into_iter()
                .map(|(s, c)| (s.as_bytes().to_vec(), c))
                .collect(),
            total_bytes_processed: stats.total_bytes_processed,
            total_chunks_with_multiplicity: stats.total_chunks_with_multiplicity,
        };
        agg.sort_canonical();

        if let Some(path) = &args.agg_file {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            agg_file::write_to_file(path, &agg).map_err(|e| e.to_string())?;
            log::info!(
                "wrote {}: {} unique chunks, {} processed",
                path.display(),
                agg.unique_chunks(),
                format_bytes(agg.total_bytes_processed)
            );
        }
        agg
    } else if let Some(path) = &args.agg_file {
        log::info!("loading {}", path.display());
        let agg = agg_file::read_from_file(path).map_err(|e| e.to_string())?;
        log::info!(
            "loaded {} unique chunks, pre_tokenizer={:?}",
            agg.unique_chunks(),
            agg.pre_tokenizer_config
        );
        agg
    } else {
        return Err(
            "pass --files (to aggregate) or --agg-file (to load an existing aggregation)"
                .to_string(),
        );
    };

    // Stage 2: merge, if requested.
    let vocab_size = match args.vocab_size {
        None => {
            log::info!("--vocab-size not given; aggregation-only mode");
            monitor.stop();
            println!(
                "aggregation done. peak RSS: {}",
                format_bytes(monitor.peak_bytes())
            );
            return Ok(());
        }
        Some(v) => v,
    };
    if vocab_size < 256 {
        return Err(format!("--vocab-size must be >= 256 (got {})", vocab_size));
    }
    let output = args
        .output
        .as_ref()
        .ok_or_else(|| "--output is required when --vocab-size is set".to_string())?;

    let total_merges = vocab_size - 256;
    let merge_pb = make_merge_pb(total_merges);

    // The merge loop publishes its progress to this counter; a poller
    // thread mirrors it onto the indicatif bar without touching the
    // hot path.
    let counter = Arc::new(AtomicU32::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let counter_p = Arc::clone(&counter);
    let stop_p = Arc::clone(&stop);
    let pb_for_thread = merge_pb.clone();
    let poller = std::thread::spawn(move || {
        while !stop_p.load(Ordering::Relaxed) {
            pb_for_thread.set_position(counter_p.load(Ordering::Relaxed) as u64);
            std::thread::sleep(Duration::from_millis(100));
        }
        pb_for_thread.set_position(counter_p.load(Ordering::Relaxed) as u64);
    });

    let mut merges: StdHashMap<wisetok::Pair, u32> = StdHashMap::new();
    let merge_t0 = Instant::now();
    materialize_and_train_with_progress(
        // Move out of `agg.chunks` to free the file's allocation before
        // the merge loop builds its own structures.
        agg.chunks,
        args.min_freq,
        vocab_size,
        merge_mode,
        &mut merges,
        Some(&counter),
    );
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
    merge_pb.set_position(merges.len() as u64);
    merge_pb.finish_with_message(format!("done in {:.1}s", merge_t0.elapsed().as_secs_f64()));

    // Reconstruct registry and write outputs.
    let mut out_registry = SpecialTokenRegistry::new();
    for t in &agg.special_tokens {
        out_registry.add(t.clone()).map_err(|e| e.to_string())?;
    }

    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    if write_hf {
        huggingface::write_tokenizer_json(output, &merges, &agg.pattern, &out_registry)
            .map_err(|e| format!("HF export: {}", e))?;
        huggingface::write_tokenizer_config(output).map_err(|e| format!("HF config: {}", e))?;
        log::info!("wrote {}/tokenizer.json", output.display());
    }
    if write_tt {
        write_tiktoken_files(output, &merges, &agg.pattern, &out_registry)?;
    }

    monitor.stop();
    println!(
        "training done. {} merges, peak RSS: {}",
        merges.len(),
        format_bytes(monitor.peak_bytes())
    );
    Ok(())
}

/// Write tiktoken-compatible artifacts: a `tiktoken.bpe` mergeable_ranks
/// file (BPE bytes + rank, base64 encoded per tiktoken's text format)
/// and a small `tiktoken.json` sidecar with the pattern and specials.
fn write_tiktoken_files(
    dir: &Path,
    merges: &StdHashMap<wisetok::Pair, u32>,
    pattern: &str,
    specials: &SpecialTokenRegistry,
) -> Result<(), String> {
    use std::io::Write;
    let ranks = tt::mergeable_ranks(merges);
    let path = dir.join("tiktoken.bpe");
    let mut f = File::create(&path).map_err(|e| e.to_string())?;
    for (bytes, rank) in &ranks {
        // tiktoken's text format is "<base64-of-bytes> <rank>\n".
        let b64 = base64_encode(bytes);
        writeln!(f, "{} {}", b64, rank).map_err(|e| e.to_string())?;
    }
    log::info!("wrote {} ({} ranks)", path.display(), ranks.len());

    // Sidecar JSON with pattern + specials so users can rebuild a
    // tiktoken.Encoding without parsing the .bpe file themselves.
    let special_base = (256 + merges.len()) as u32;
    let mut special_obj = serde_json::Map::new();
    for (i, t) in specials.tokens().iter().enumerate() {
        special_obj.insert(t.clone(), serde_json::Value::from(special_base + i as u32));
    }
    let sidecar = serde_json::json!({
        "pattern": pattern,
        "special_tokens": special_obj,
    });
    let pretty = serde_json::to_string_pretty(&sidecar).map_err(|e| e.to_string())?;
    fs::write(dir.join("tiktoken.json"), pretty).map_err(|e| e.to_string())?;
    Ok(())
}

/// Tiny base64 encoder for the tiktoken `.bpe` file format. We avoid the
/// `base64` crate dependency since we need it in exactly one place.
fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        out.push(CHARS[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

fn run_validate(args: ValidateArgs) -> Result<(), String> {
    init_logger(true);
    let tokenizer_json = args.tokenizer.join("tokenizer.json");
    if !tokenizer_json.exists() {
        return Err(format!(
            "no tokenizer.json found at {}",
            tokenizer_json.display()
        ));
    }
    log::info!("found {}", tokenizer_json.display());

    let mut total_lines = 0u64;
    let mut total_bytes = 0u64;
    for path in &args.test_files {
        let f = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        for line in BufReader::new(f).lines() {
            let line = line.map_err(|e| e.to_string())?;
            total_lines += 1;
            total_bytes += line.len() as u64;
        }
    }
    println!(
        "validate (smoke): tokenizer.json found, scanned {} lines / {} of test data",
        total_lines,
        format_bytes(total_bytes)
    );
    println!("note: full encode/roundtrip validation lands in iteration 3.");
    Ok(())
}

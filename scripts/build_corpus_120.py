#!/usr/bin/env python3
"""Build a ~143 GB multi-domain corpus for WiseTok production v2.

Composition target:
    Python:      ~60 GB  (capped from ~80 GB available)
    Other code:  ~53 GB  (all of C, C++, Java, JavaScript)
    HTML:        ~2 GB   (existing html_raw.txt)
    English:    ~12 GB   (fineweb-edu-dedup sampled)
    Edutext:    ~12 GB   (cosmopedia-v2 excluding auto_math_text)
    Math:        ~4.5 GB (cosmopedia-v2 auto_math_text only)

Documents joined with <|endoftext|>. Each lang gets its own file so the
WiseTok aggregation reads them in parallel.

Output: /media/data1tb/ezellm-coder-tokenizer/corpus-v2/corpus_<lang>.txt
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

OUT_DIR = Path("/media/data1tb/ezellm-coder-tokenizer/corpus-v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "\n<|endoftext|>\n"
SEP_B = SEP.encode("utf-8")

GB = 1024**3
STACKV2 = Path("/media/data1tb/stackv2-dedup-sub")
SMOLLM  = Path("/media/data1tb/smollm-corpus")
HTML_FILE = STACKV2 / "html" / "html_raw.txt"

# (target_bytes, source, parquet_text_column, optional_filter)
JOBS: list[tuple[str, int, Path, str, callable | None]] = [
    ("python",     60 * GB, STACKV2 / "Python",     "content", None),
    ("c",          15 * GB, STACKV2 / "C",          "content", None),  # cap above what's available; we'll just take all
    ("cpp",        15 * GB, STACKV2 / "C++",        "content", None),
    ("java",       17 * GB, STACKV2 / "Java",       "content", None),
    ("javascript", 12 * GB, STACKV2 / "JavaScript", "content", None),
    ("english",    12 * GB, SMOLLM / "fineweb-edu-dedup", "text", None),
    # Edutext: everything except auto_math_text
    ("edutext",    12 * GB, SMOLLM / "cosmopedia-v2", "text",
        lambda seed: seed != "auto_math_text"),
    # Math: only auto_math_text
    ("math",        5 * GB, SMOLLM / "cosmopedia-v2", "text",
        lambda seed: seed == "auto_math_text"),
]

random.seed(20260501)


def build_parquet(lang: str, target: int, src_dir: Path, col: str,
                  filt: callable | None) -> tuple[int, int]:
    out = OUT_DIR / f"corpus_{lang}.txt"
    if out.exists() and out.stat().st_size >= target * 0.95:
        sz = out.stat().st_size
        print(f"  [{lang}] already at {sz/GB:.2f} GB, skipping", flush=True)
        return sz, -1

    shards = sorted(src_dir.glob("*.parquet"))
    if not shards:
        print(f"  [{lang}] NO SHARDS in {src_dir}", file=sys.stderr)
        return 0, 0
    random.shuffle(shards)

    needs_filter = filt is not None
    columns = [col] + (["seed_data"] if needs_filter else [])

    written = 0
    n_docs = 0
    t0 = time.time()
    last_log = t0

    with open(out, "wb") as f:
        for shard in shards:
            if written >= target:
                break
            try:
                table = pq.read_table(shard, columns=columns)
            except Exception as e:
                print(f"  [{lang}] {shard.name} read failed: {e}", file=sys.stderr)
                continue
            contents = table.column(col).to_pylist()
            seeds = table.column("seed_data").to_pylist() if needs_filter else None
            for i, content in enumerate(contents):
                if content is None or not isinstance(content, str):
                    continue
                if needs_filter and not filt(seeds[i]):
                    continue
                b = content.encode("utf-8", errors="replace")
                if len(b) < 100:
                    continue
                f.write(b)
                f.write(SEP_B)
                written += len(b) + len(SEP_B)
                n_docs += 1
                if written >= target:
                    break
            now = time.time()
            if now - last_log > 5 or written >= target:
                mbps = written / 1e6 / max(now - t0, 1e-3)
                print(f"  [{lang}] {written/GB:.2f}/{target/GB:.1f} GB, {n_docs:,} docs, "
                      f"{mbps:.1f} MB/s, shard {shard.name}", flush=True)
                last_log = now
    return written, n_docs


def build_html(target: int) -> tuple[int, int]:
    """Split html_raw.txt on <!DOCTYPE or <!doctype, join with <|endoftext|>."""
    out = OUT_DIR / "corpus_html.txt"
    if out.exists() and out.stat().st_size >= target * 0.95:
        sz = out.stat().st_size
        print(f"  [html] already at {sz/GB:.2f} GB, skipping", flush=True)
        return sz, -1

    if not HTML_FILE.exists():
        print(f"  [html] {HTML_FILE} missing", file=sys.stderr)
        return 0, 0

    print(f"  [html] reading {HTML_FILE} ({HTML_FILE.stat().st_size/GB:.2f} GB)", flush=True)
    with open(HTML_FILE, "rb") as f:
        data = f.read()

    # Split on <!DOCTYPE / <!doctype (case-insensitive)
    pattern = re.compile(rb"(?=<!doctype)", re.IGNORECASE)
    docs = pattern.split(data)
    # The first chunk may be empty or pre-DOCTYPE garbage; skip if too small
    docs = [d for d in docs if len(d) >= 100]
    print(f"  [html] {len(docs):,} documents found", flush=True)

    written = 0
    n_docs = 0
    with open(out, "wb") as f:
        for d in docs:
            if written >= target:
                break
            f.write(d)
            f.write(SEP_B)
            written += len(d) + len(SEP_B)
            n_docs += 1
    print(f"  [html] {written/GB:.2f} GB, {n_docs:,} docs", flush=True)
    return written, n_docs


def main():
    print(f"Output directory: {OUT_DIR}")
    free = os.statvfs(OUT_DIR).f_bavail * os.statvfs(OUT_DIR).f_frsize
    print(f"Free space: {free/GB:.1f} GB")
    total_target = sum(t for _,t,_,_,_ in JOBS) + 2 * GB
    print(f"Total target: {total_target/GB:.1f} GB\n")

    if free < total_target:
        print(f"WARNING: free space ({free/GB:.1f} GB) < target ({total_target/GB:.1f} GB)",
              file=sys.stderr)

    # Build smallest first
    order = ["math", "html", "javascript", "c", "edutext", "cpp", "java", "english", "python"]

    grand = 0
    grand_docs = 0
    grand_t0 = time.time()
    for lang in order:
        if lang == "html":
            print(f"=== html (target 2 GB) ===", flush=True)
            t0 = time.time()
            sz, nd = build_html(2 * GB)
        else:
            job = next(j for j in JOBS if j[0] == lang)
            _, target, src, col, filt = job
            print(f"=== {lang} (target {target/GB:.1f} GB) ===", flush=True)
            t0 = time.time()
            sz, nd = build_parquet(lang, target, src, col, filt)
        dt = time.time() - t0
        print(f"  done: {sz/GB:.2f} GB, {nd:,} docs in {dt/60:.1f} min\n", flush=True)
        grand += sz
        grand_docs += max(nd, 0)

    grand_dt = time.time() - grand_t0
    print(f"\nTOTAL: {grand/GB:.2f} GB, {grand_docs:,} docs in {grand_dt/60:.1f} min")
    print(f"\nFinal files:")
    for f in sorted(OUT_DIR.glob("corpus_*.txt")):
        print(f"  {f.name:25s} {f.stat().st_size/GB:>6.2f} GB")


if __name__ == "__main__":
    main()

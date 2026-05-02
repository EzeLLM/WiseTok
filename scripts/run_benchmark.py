#!/usr/bin/env python3
"""Paper-grade tokenizer benchmark over a multi-domain evaluation corpus.

For each subdirectory of EVAL_DIR (one category), concatenate all files,
encode with each tokenizer, and report:

    1. Wide chars/tok matrix (categories × tokenizers).
    2. Per-category rankings (best → worst).
    3. Macro-average column (geometric mean across categories).
    4. CSV dump for downstream analysis.
    5. LaTeX table block ready to drop into a paper.

Tokenizers covered:
    - All WiseTok runs auto-discovered under the repo, ~/wisetok-runs, or
      WISETOK_BENCH_DIRS (use --local-tokenizer NAME=path to add specific files).
    - SmolLM2 (49K)
    - StarCoder2 (49K)
    - DeepSeek-Coder (32K)
    - Qwen2.5-Coder (151K)
    - GPT-2 (50K)
    - GPT-3.5/4 cl100k_base (100K)   [tiktoken]
    - GPT-4o o200k_base (200K)        [tiktoken]
    - Claude (Anthropic)              [API; only if ANTHROPIC_API_KEY is set]

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --eval-dir /custom/path
    python scripts/run_benchmark.py --csv path/to/results.csv --latex path/to/table.tex
    python scripts/run_benchmark.py --skip-claude     # never call the Claude API
    python scripts/run_benchmark.py --skip-hf         # local-only, no HF downloads
    python scripts/run_benchmark.py --local-tokenizer my-tok=/path/to/tokenizer.json
    WISETOK_BENCH_DIRS=/data/runs1:/data/runs2 python scripts/run_benchmark.py

The Anthropic API path is rate-limited and costs money. We make at most
N_CATEGORIES requests per Claude model. Set CLAUDE_MODELS env var to override.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

# Local WiseTok runs are auto-discovered (see discover_local_runs below).
# Override or extend with `--local-tokenizer NAME=path/to/tokenizer.json` (repeatable),
# or with `WISETOK_BENCH_DIRS=dir1:dir2:...` to add custom search roots.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_SEARCH_ROOTS = [
    REPO_ROOT,
    REPO_ROOT.parent,                    # sibling sandbox dirs (e.g. ../wisetok-runs)
    Path.home() / "wisetok-runs",
    Path("/media/data1tb/ezellm-coder-tokenizer"),  # legacy WiseTok dev box
]

TOKENIZERS_TIKTOKEN = [
    ("GPT-3.5/4 cl100k (100K)", "tiktoken", "cl100k_base", 100_277),
    ("GPT-4o o200k (200K)",     "tiktoken", "o200k_base",  200_018),
]

TOKENIZERS_HF = [
    ("SmolLM2 (49K)",         "hf", "HuggingFaceTB/SmolLM2-360M",           49_152),
    ("StarCoder2 (49K)",      "hf", "bigcode/starcoder2-3b",                49_152),
    ("DeepSeek-Coder (32K)",  "hf", "deepseek-ai/deepseek-coder-1.3b-base", 32_256),
    ("Qwen2.5-Coder (151K)",  "hf", "Qwen/Qwen2.5-Coder-0.5B",              151_936),
    ("GPT-2 (50K)",           "hf", "openai-community/gpt2",                50_257),
]

# Anthropic models to query via API (requires ANTHROPIC_API_KEY).
# Vocab size is undisclosed; we use 0 in the table.
DEFAULT_CLAUDE_MODELS = [
    ("Claude Haiku 4.5", "claude-haiku-4-5"),
    ("Claude Sonnet 4.5", "claude-sonnet-4-5"),
    ("Claude Opus 4.7", "claude-opus-4-7"),
]


# ------------------------------------------------------------------- loading
_cache: dict[tuple[str, str], object] = {}


def load(kind: str, ident: str):
    key = (kind, ident)
    if key in _cache:
        return _cache[key]
    if kind == "local":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(ident)
    elif kind == "tiktoken":
        import tiktoken
        tok = tiktoken.get_encoding(ident)
    elif kind == "hf":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ident)
    elif kind == "anthropic":
        import anthropic
        tok = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    else:
        raise ValueError(f"unknown kind {kind}")
    _cache[key] = tok
    return tok


def encode_count(tok, kind: str, text: str, model_id: str | None = None) -> int:
    if kind == "tiktoken":
        return len(tok.encode(text, disallowed_special=()))
    if kind == "anthropic":
        # tok is an Anthropic client; model_id is required.
        resp = tok.messages.count_tokens(
            model=model_id,
            messages=[{"role": "user", "content": text}],
        )
        return resp.input_tokens
    if hasattr(tok, "encode_batch"):  # tokenizers.Tokenizer
        return len(tok.encode(text).ids)
    return len(tok.encode(text, add_special_tokens=False))


# ------------------------------------------------------------------- corpus
def load_category(cat_dir: Path) -> tuple[str, int, int]:
    files = sorted(p for p in cat_dir.rglob("*") if p.is_file())
    parts: list[str] = []
    for f in files:
        try:
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}", file=sys.stderr)
    text = "\n\n".join(parts)
    return text, len(files), len(text)


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


# ----------------------------------------------------------------- discovery
def _vocab_size_from_json(path: Path) -> int | None:
    """Best-effort vocab-size read from a tokenizer.json without loading the full tokenizer."""
    try:
        import json
        with path.open() as f:
            cfg = json.load(f)
    except Exception:
        return None
    model = cfg.get("model") or {}
    vocab = model.get("vocab") or {}
    n_model = len(vocab) if isinstance(vocab, dict) else 0
    n_added = len(cfg.get("added_tokens") or [])
    total = n_model + n_added
    return total or None


def _short_label(tok_path: Path, vocab: int | None) -> str:
    """Build a stable display label from the tokenizer's parent dir name."""
    name = tok_path.parent.name or tok_path.stem
    # Trim wisetok-prefixed run dirs to the distinguishing tail.
    if name.lower().startswith("wisetok-"):
        name = name[len("wisetok-"):]
    if vocab:
        return f"WiseTok {name} ({vocab // 1000}K)"
    return f"WiseTok {name}"


def discover_local_runs(extra_roots: list[Path], explicit: list[str]) -> list[tuple[str, str, str, int]]:
    """Find local WiseTok tokenizer.json files.

    Search order:
      1. `--local-tokenizer NAME=path` flags (explicit, never deduped away).
      2. Each root in `DEFAULT_LOCAL_SEARCH_ROOTS + extra_roots`: scan up to depth 3
         for files named `tokenizer.json`. We require a sibling `tiktoken.bpe` *or*
         `tokenizer_config.json` so we don't pick up unrelated HF caches.

    Results are sorted by vocab size (descending). Duplicates by realpath are dropped.
    Returns: list of (display_name, "local", path_str, vocab_size).
    """
    out: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()

    # 1) explicit
    for spec in explicit:
        if "=" in spec:
            name, _, path_str = spec.partition("=")
            name = name.strip()
            path = Path(path_str.strip()).expanduser()
        else:
            path = Path(spec).expanduser()
            name = ""
        if not path.is_file():
            print(f"  [warn] --local-tokenizer not found: {path}", file=sys.stderr)
            continue
        real = str(path.resolve())
        if real in seen:
            continue
        seen.add(real)
        v = _vocab_size_from_json(path) or 0
        label = name or _short_label(path, v)
        out.append((label, "local", str(path), v))

    # 2) auto-discovery
    roots = list(DEFAULT_LOCAL_SEARCH_ROOTS) + extra_roots
    for root in roots:
        if not root or not root.exists():
            continue
        try:
            # bounded depth: root, depth1, depth2, depth3 == 3 levels of glob
            patterns = ["tokenizer.json", "*/tokenizer.json", "*/*/tokenizer.json"]
            for pat in patterns:
                for path in root.glob(pat):
                    if not path.is_file():
                        continue
                    parent = path.parent
                    # require a wisetok-style sibling so we don't ingest stray HF caches
                    if not ((parent / "tiktoken.bpe").exists() or (parent / "tokenizer_config.json").exists()):
                        continue
                    real = str(path.resolve())
                    if real in seen:
                        continue
                    seen.add(real)
                    v = _vocab_size_from_json(path) or 0
                    out.append((_short_label(path, v), "local", str(path), v))
        except (PermissionError, OSError):
            continue

    out.sort(key=lambda r: -r[3])
    return out


# ------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-dir", default="/home/ezel/Development/WiseTok/benchmark")
    p.add_argument("--csv",      default="/home/ezel/Development/WiseTok/benchmark/results/bench_results.csv")
    p.add_argument("--latex",    default="/home/ezel/Development/WiseTok/benchmark/results/bench_table.tex")
    p.add_argument("--skip-hf",       action="store_true", help="Skip HF tokenizers")
    p.add_argument("--skip-tiktoken", action="store_true", help="Skip tiktoken (GPT) tokenizers")
    p.add_argument("--skip-claude",   action="store_true", help="Skip Claude API even if key is set")
    p.add_argument("--local-tokenizer", action="append", default=[], metavar="NAME=PATH",
                   help="Add a local tokenizer.json (repeatable). NAME= prefix optional.")
    p.add_argument("--local-search-root", action="append", default=[], metavar="DIR",
                   help="Extra directory to scan for tokenizer.json (repeatable). "
                        "Also accepts WISETOK_BENCH_DIRS env var (':' separator).")
    p.add_argument("--max-claude-chars", type=int, default=200_000,
                   help="Truncate categories above this size when calling Claude API (cost cap)")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"eval-dir not found: {eval_dir}", file=sys.stderr)
        return 1

    # --- Load category corpora -------------------------------------------
    # Skip well-known non-eval subdirs that may live alongside categories
    # (e.g. benchmark/results/ holds CSV+TeX outputs from prior runs).
    SKIP_DIRS = {"results", ".git", "__pycache__"}
    categories: list[tuple[str, str, int, int]] = []
    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in SKIP_DIRS or sub.name.startswith("."):
            continue
        text, nf, nc = load_category(sub)
        if nc == 0:
            print(f"  [warn] empty category {sub.name}", file=sys.stderr)
            continue
        categories.append((sub.name, text, nf, nc))

    if not categories:
        print("No categories found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(categories)} categories from {eval_dir}:")
    for name, _, nf, nc in categories:
        print(f"  {name:20s} {nf:>3d} files, {nc:>10,d} chars")

    # --- Discover local WiseTok runs -------------------------------------
    extra_roots = [Path(d).expanduser() for d in args.local_search_root]
    env_dirs = os.environ.get("WISETOK_BENCH_DIRS", "")
    if env_dirs:
        extra_roots += [Path(d).expanduser() for d in env_dirs.split(":") if d]
    local_runs = discover_local_runs(extra_roots, args.local_tokenizer)
    if local_runs:
        print(f"\nDiscovered {len(local_runs)} local WiseTok tokenizer(s):")
        for tn, _, path, vocab in local_runs:
            print(f"  {tn:32s} vocab={vocab:>7,d}  {path}")
    else:
        print("\nNo local WiseTok runs found. Use --local-tokenizer NAME=path/tokenizer.json "
              "or set WISETOK_BENCH_DIRS to add search roots.")

    # --- Build tokenizer list --------------------------------------------
    plan = list(local_runs)
    if not args.skip_tiktoken:
        plan += TOKENIZERS_TIKTOKEN
    if not args.skip_hf:
        plan += TOKENIZERS_HF

    # Anthropic API path: only if env var set and not explicitly skipped.
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY")) and not args.skip_claude
    claude_specs: list[tuple[str, str]] = []
    if use_claude:
        env_models = os.environ.get("CLAUDE_MODELS")
        if env_models:
            claude_specs = [(m.strip(), m.strip()) for m in env_models.split(",") if m.strip()]
        else:
            claude_specs = DEFAULT_CLAUDE_MODELS
        print(f"\nClaude API enabled. Models: {[m for _, m in claude_specs]}")
    else:
        if args.skip_claude:
            print("\n[claude] skipped via --skip-claude")
        else:
            print("\n[claude] ANTHROPIC_API_KEY not set — Claude tokenizer omitted.")
            print("        Set ANTHROPIC_API_KEY=... to include Claude in the comparison.")

    print(f"\nLoading {len(plan)} tokenizers...")
    loaded: list[tuple[str, str, str, int, object]] = []
    for tn, kind, ident, vocab in plan:
        try:
            tok = load(kind, ident)
        except Exception as e:
            print(f"  [skip] {tn}: {e}")
            continue
        loaded.append((tn, kind, ident, vocab, tok))

    # Initialize Claude client once if needed
    claude_client = None
    if use_claude and claude_specs:
        try:
            claude_client = load("anthropic", "client")
        except Exception as e:
            print(f"  [claude] failed to init client: {e}")
            claude_specs = []

    # --- Encode all (tokenizer, category) pairs --------------------------
    # results[(tok_name, cat_name)] = (n_tokens, chars_per_tok, encode_seconds)
    results: dict[tuple[str, str], tuple[int, float, float]] = {}
    print("\nEncoding...")
    for tn, kind, ident, vocab, tok in loaded:
        print(f"  {tn} ...", end="", flush=True)
        for cat, text, _, n_chars in categories:
            t0 = time.perf_counter()
            try:
                nt = encode_count(tok, kind, text)
            except Exception as e:
                print(f" [{cat}: {e}]", end="")
                continue
            dt = time.perf_counter() - t0
            cpt = n_chars / nt if nt else 0.0
            results[(tn, cat)] = (nt, cpt, dt)
        print(" ok")

    # Claude API path: separate loop because every (model, category) is a
    # network call. We bound text length via --max-claude-chars to cap cost.
    # If a model has zero successful categories (e.g. credit balance, auth),
    # we drop it from the output table entirely instead of polluting the
    # report with empty rows.
    claude_loaded: list[tuple[str, int]] = []
    if claude_client and claude_specs:
        for display, model_id in claude_specs:
            print(f"  {display} (Anthropic API) ...", end="", flush=True)
            n_ok = 0
            first_error: str | None = None
            for cat, text, _, n_chars in categories:
                snippet = text if len(text) <= args.max_claude_chars else text[:args.max_claude_chars]
                used_chars = len(snippet)
                t0 = time.perf_counter()
                try:
                    nt = encode_count(claude_client, "anthropic", snippet, model_id)
                except Exception as e:
                    if first_error is None:
                        first_error = str(e)
                    continue
                dt = time.perf_counter() - t0
                cpt = used_chars / nt if nt else 0.0
                results[(display, cat)] = (nt, cpt, dt)
                n_ok += 1
            if n_ok == 0:
                # Single concise warning instead of one error per category.
                short = (first_error or "no successful categories")[:140]
                print(f" SKIPPED — {short}")
            else:
                print(f" ok ({n_ok}/{len(categories)} categories)")
                claude_loaded.append((display, 0))  # vocab unknown

    # Combine for output
    all_loaded = [(tn, kind, ident, vocab) for tn, kind, ident, vocab, _ in loaded]
    for display, _vocab in claude_loaded:
        all_loaded.append((display, "anthropic", display, 0))

    cats = [c for c, *_ in categories]
    cat_chars = {c: nc for c, _, _, nc in categories}
    toks_in = [tn for tn, *_ in all_loaded]

    # Headline = the largest local (WiseTok) tokenizer that successfully loaded.
    # Highlighted with "←" in per-category and macro views.
    local_loaded = [(tn, vocab) for tn, kind, _, vocab in all_loaded if kind == "local"]
    headline_name = max(local_loaded, key=lambda r: r[1])[0] if local_loaded else None

    # Macro geomean per tokenizer
    macro: dict[str, float] = {}
    for tn in toks_in:
        vals = [results[(tn, c)][1] for c in cats if (tn, c) in results]
        macro[tn] = geomean(vals) if vals else 0.0

    def sort_key(tn: str) -> tuple[int, float]:
        # Group: WiseTok first, then by macro desc
        return (0 if "WiseTok" in tn else 1, -macro.get(tn, 0.0))

    # Total tokens per tokenizer (sum across all categories).
    total_tokens: dict[str, int] = {}
    for tn in toks_in:
        total_tokens[tn] = sum(results[(tn, c)][0] for c in cats if (tn, c) in results)

    best_per_cat_cpt = {c: max((results[(tn, c)][1] for tn in toks_in if (tn, c) in results), default=0.0) for c in cats}
    # For tokens, fewer is better — same input, fewer tokens means denser packing.
    best_per_cat_tok = {c: min((results[(tn, c)][0] for tn in toks_in if (tn, c) in results), default=0)   for c in cats}
    best_macro = max(macro.values()) if macro else 0.0
    best_total = min(total_tokens.values()) if total_tokens else 0

    w_tn = max(len(t) for t in toks_in) + 2
    w_cell = 11

    # --- PRIMARY TABLE: token count per category ------------------------
    # Same input, how many tokens does each tokenizer emit? Lower = better.
    # This is the clearest comparison: identical bytes in, fewest tokens out wins.
    print("\n" + "=" * 140)
    print("TOKEN COUNT PER CATEGORY  (fewer = better; same input, what each tokenizer emits).")
    print("* marks per-category fewest, ** marks fewest overall.")
    print("=" * 140)

    # Reference row: input character counts so readers see "for X chars → Y tokens".
    ref = f"{'(input chars)':{w_tn}s} {'—':>7s}"
    for c in cats:
        ref += f" {cat_chars[c]:>{w_cell-1},d} "
    total_chars_all = sum(cat_chars[c] for c in cats)
    ref += f" {total_chars_all:>{w_cell-2},d}  "
    print(ref)

    header = f"{'Tokenizer':{w_tn}s} {'Vocab':>7s}"
    for c in cats:
        header += f" {c:>{w_cell}s}"
    header += f" {'TOTAL':>{w_cell}s}"
    print(header)
    print("-" * len(header))

    for tn in sorted(toks_in, key=sort_key):
        vocab = next(v for n, _, _, v in all_loaded if n == tn)
        v_str = f"{vocab:,}" if vocab > 0 else "?"
        row = f"{tn:{w_tn}s} {v_str:>7s}"
        for c in cats:
            if (tn, c) in results:
                nt, _, _ = results[(tn, c)]
                marker = "*" if nt == best_per_cat_tok[c] else " "
                row += f" {nt:>{w_cell-1},d}{marker}"
            else:
                row += f" {'-':>{w_cell}s}"
        marker = "**" if total_tokens.get(tn) == best_total else "  "
        row += f" {total_tokens.get(tn, 0):>{w_cell-2},d}{marker}"
        print(row)

    # --- SECONDARY TABLE: chars/tok compression -------------------------
    print("\n" + "=" * 140)
    print("CHARS/TOK COMPRESSION  (higher = denser packing). * marks per-category best, ** marks macro-best.")
    print("=" * 140)
    header = f"{'Tokenizer':{w_tn}s} {'Vocab':>7s}"
    for c in cats:
        header += f" {c:>{w_cell}s}"
    header += f" {'macro':>{w_cell}s}"
    print(header)
    print("-" * len(header))

    for tn in sorted(toks_in, key=sort_key):
        vocab = next(v for n, _, _, v in all_loaded if n == tn)
        v_str = f"{vocab:,}" if vocab > 0 else "?"
        row = f"{tn:{w_tn}s} {v_str:>7s}"
        for c in cats:
            if (tn, c) in results:
                _, cpt, _ = results[(tn, c)]
                marker = "*" if abs(cpt - best_per_cat_cpt[c]) < 1e-9 else " "
                row += f" {cpt:>{w_cell-1}.3f}{marker}"
            else:
                row += f" {'-':>{w_cell}s}"
        m_marker = "**" if abs(macro[tn] - best_macro) < 1e-9 else "  "
        row += f" {macro[tn]:>{w_cell-2}.3f}{m_marker}"
        print(row)

    # --- Per-category rankings ------------------------------------------
    print("\n" + "=" * 140)
    print("PER-CATEGORY RANKINGS")
    print("=" * 140)
    for c in cats:
        print(f"\n{c}  ({cat_chars[c]:,} chars)")
        rows = [(tn, *results[(tn, c)]) for tn in toks_in if (tn, c) in results]
        rows.sort(key=lambda r: -r[2])
        for rank, (tn, nt, cpt, dt) in enumerate(rows, 1):
            mark = "  ←" if headline_name and tn == headline_name else ""
            print(f"  {rank:2d}. {tn:30s} {nt:>10,d} tok  {cpt:>7.3f} c/tok  ({dt:.2f}s){mark}")

    # --- Macro summary ---------------------------------------------------
    print("\n" + "=" * 140)
    print("MACRO-AVERAGE  (geometric mean of chars/tok across all categories)")
    print("=" * 140)
    rows_macro = sorted([(tn, macro[tn]) for tn in toks_in], key=lambda r: -r[1])
    for rank, (tn, m) in enumerate(rows_macro, 1):
        vocab = next(v for n, _, _, v in all_loaded if n == tn)
        v_str = f"{vocab:,}" if vocab > 0 else "?"
        mark = "  ←" if headline_name and tn == headline_name else ""
        print(f"  {rank:2d}. {tn:30s} vocab={v_str:>10s}  macro c/tok = {m:.4f}{mark}")

    # --- CSV dump --------------------------------------------------------
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "vocab_size", "category", "n_chars", "n_tokens", "chars_per_tok", "encode_seconds"])
        for tn, kind, ident, vocab in all_loaded:
            for c in cats:
                if (tn, c) not in results:
                    continue
                nt, cpt, dt = results[(tn, c)]
                w.writerow([tn, vocab, c, cat_chars[c], nt, f"{cpt:.6f}", f"{dt:.4f}"])
    print(f"\nCSV → {csv_path}")

    # --- LaTeX tables ----------------------------------------------------
    # Two tables: token counts first (the headline number for a paper),
    # then chars/tok compression as a secondary view.
    latex_path = Path(args.latex)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l r " + " ".join(["r"] * len(cats)) + " r"
    short_cats = [c.replace("_", " ") for c in cats]

    lines: list[str] = ["% Auto-generated by scripts/run_benchmark.py"]

    # === Table 1: token counts per category =============================
    lines += [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\caption{Token counts per category for identical inputs (lower is better). "
        "Each cell is the number of tokens emitted by the row tokenizer for the column "
        "category's evaluation text. Per-category fewest in \\textbf{bold}; the TOTAL "
        "column is the sum across all categories.}",
        "\\label{tab:tokens}",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Tokenizer & Vocab & " + " & ".join(short_cats) + " & TOTAL \\\\",
        "\\midrule",
    ]
    # Reference row: input chars per category
    ref_cells = [f"{cat_chars[c]:,}" for c in cats]
    total_chars_all = sum(cat_chars[c] for c in cats)
    lines.append("\\textit{(input chars)} & --- & "
                 + " & ".join(f"\\textit{{{x}}}" for x in ref_cells)
                 + f" & \\textit{{{total_chars_all:,}}} \\\\")
    lines.append("\\midrule")

    for tn in sorted(toks_in, key=sort_key):
        vocab = next(v for n, _, _, v in all_loaded if n == tn)
        v_str = f"{vocab:,}" if vocab > 0 else "---"
        cells = []
        for c in cats:
            if (tn, c) in results:
                nt = results[(tn, c)][0]
                s = f"{nt:,}"
                if nt == best_per_cat_tok[c]:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            else:
                cells.append("---")
        tot = total_tokens.get(tn, 0)
        tot_s = f"{tot:,}"
        if tot == best_total:
            tot_s = "\\textbf{" + tot_s + "}"
        tn_esc = tn.replace("&", "\\&").replace("_", "\\_")
        lines.append(f"{tn_esc} & {v_str} & " + " & ".join(cells) + f" & {tot_s} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    # === Table 2: chars/tok compression =================================
    lines += [
        "\\begin{table*}[t]",
        "\\centering\\small",
        "\\caption{Compression ratio (chars/token, higher is better) across "
        + f"{len(cats)} domains. Per-category best in \\textbf{{bold}}; the macro "
        "column is the geometric mean across categories.}",
        "\\label{tab:compression}",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Tokenizer & Vocab & " + " & ".join(short_cats) + " & macro \\\\",
        "\\midrule",
    ]
    for tn in sorted(toks_in, key=sort_key):
        vocab = next(v for n, _, _, v in all_loaded if n == tn)
        v_str = f"{vocab:,}" if vocab > 0 else "---"
        cells = []
        for c in cats:
            if (tn, c) in results:
                cpt = results[(tn, c)][1]
                s = f"{cpt:.3f}"
                if abs(cpt - best_per_cat_cpt[c]) < 1e-9:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            else:
                cells.append("---")
        m = f"{macro[tn]:.3f}"
        if abs(macro[tn] - best_macro) < 1e-9:
            m = "\\textbf{" + m + "}"
        tn_esc = tn.replace("&", "\\&").replace("_", "\\_")
        lines.append(f"{tn_esc} & {v_str} & " + " & ".join(cells) + f" & {m} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    latex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX → {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

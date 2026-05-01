"""Compare wisetok-production (24K) vs SmolLM2 (49K) and others on Python files."""
import time
from pathlib import Path

WISETOK_PATH = "/media/data1tb/ezellm-coder-tokenizer/wisetok-production/tokenizer.json"
EZELLM_V1_PATH = "/media/data1tb/ezellm-coder-tokenizer/tokenizer"

TOKENIZERS = [
    ("WiseTok prod (24K)",          "local",  WISETOK_PATH,                          24_576),
    ("EZeLLM v1 (24K, py-only)",    "local",  EZELLM_V1_PATH,                        24_576),
    ("SmolLM2 (49K)",               "hf",     "HuggingFaceTB/SmolLM2-360M",          49_152),
    ("GPT-2 (50K)",                 "hf",     "openai-community/gpt2",               50_257),
]

def load(kind, ident):
    if kind == "local":
        if ident.endswith(".json"):
            from tokenizers import Tokenizer
            return Tokenizer.from_file(ident)
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(ident)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ident)

def n_tokens(tok, text):
    if hasattr(tok, "encode") and not hasattr(tok, "encode_batch"):
        # transformers
        return len(tok.encode(text, add_special_tokens=False))
    # tokenizers.Tokenizer
    return len(tok.encode(text).ids)

def main():
    bench_dir = Path("/home/ezel/Development/WiseTok/runs/bench_corpus")
    files = sorted(bench_dir.glob("*.py"))
    text = "".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8", errors="replace"))
    print(f"Test corpus: {len(files)} files, {n_chars:,} chars, {n_bytes:,} bytes\n")

    rows = []
    for name, kind, ident, vocab in TOKENIZERS:
        try:
            tok = load(kind, ident)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        t0 = time.perf_counter()
        nt = n_tokens(tok, text)
        dt = time.perf_counter() - t0
        rows.append((name, vocab, nt, n_chars/nt, n_bytes/nt, dt))

    rows.sort(key=lambda r: -r[3])  # by chars/tok desc
    print(f"{'Tokenizer':30s} {'Vocab':>7s} {'Tokens':>10s} {'chars/tok':>10s} {'bytes/tok':>10s} {'time':>8s}")
    print("-" * 80)
    for name, vocab, nt, cpt, bpt, dt in rows:
        print(f"{name:30s} {vocab:>7,d} {nt:>10,d} {cpt:>10.3f} {bpt:>10.3f} {dt:>7.2f}s")

if __name__ == "__main__":
    main()

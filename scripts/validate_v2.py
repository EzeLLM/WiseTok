#!/usr/bin/env python3
"""Validate v2 (24K, 142GB) tokenizer."""
import json, random
from pathlib import Path
from tokenizers import Tokenizer

V2 = "/media/data1tb/ezellm-coder-tokenizer/wisetok-production-v2/tokenizer.json"
PYTHON_SHARDS = Path("/media/data1tb/stackv2-dedup-sub/Python/")

print(f"Loading {V2}")
tok = Tokenizer.from_file(V2)
print(f"vocab_size: {tok.get_vocab_size()}\n")

samples = ["def __init__(self):", "import numpy as np", "for i in range(10):", "    return self.value"]
print("--- Sanity ---")
for s in samples:
    ids = tok.encode(s).ids
    print(f"{s!r:40s} -> {len(ids)} tok")

print("\n--- Roundtrip basic ---")
text = "import numpy as np\n\ndef hello():\n    print('world')\n"
ids = tok.encode(text).ids
ok = tok.decode(ids) == text
print(f"basic: {'PASS' if ok else 'FAIL'}  ({len(text)} chars -> {len(ids)} tok, {len(text)/len(ids):.2f} c/t)")

print("\n--- Whitespace ---")
for p in ["    ", "        ", "\t", "\n    ", "\n        ", "\n\t"]:
    ids = tok.encode(p).ids
    flag = '✓' if len(ids) == 1 else f'✗({len(ids)})'
    print(f"  {p!r:20s} -> {len(ids)} {flag}")

print("\n--- Specials ---")
for st in ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]:
    ids = tok.encode(st).ids
    flag = '✓' if len(ids) == 1 else f'✗'
    print(f"  {st:20s} -> {ids} {flag}")

print("\n--- Real-file roundtrip on 10 random Python files ---")
import pyarrow.parquet as pq
random.seed(20260502)
shard = random.choice(sorted(PYTHON_SHARDS.glob("*.parquet")))
print(f"  shard: {shard.name}")
contents = pq.read_table(shard, columns=["content"]).column("content").to_pylist()
random.shuffle(contents)
n_test, n_pass, total_chars, total_tok = 10, 0, 0, 0
for i, c in enumerate(contents[:n_test]):
    if c is None: continue
    ids = tok.encode(c).ids
    if tok.decode(ids) == c:
        n_pass += 1
        total_chars += len(c); total_tok += len(ids)
    else:
        print(f"    FAIL on file {i}")
print(f"  {n_pass}/{n_test} pass; chars/tok = {total_chars/max(1,total_tok):.3f}")

print("\n--- Top 20 merges ---")
with open(V2) as f: cfg = json.load(f)
for i, m in enumerate(cfg["model"]["merges"][:20]):
    print(f"  {256+i:5d}: {m!r}")

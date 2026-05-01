#!/usr/bin/env python3
"""Validate the freshly-trained 24K wisetok tokenizer."""
import json
import random
from pathlib import Path
from tokenizers import Tokenizer

TOK_PATH = "/media/data1tb/ezellm-coder-tokenizer/wisetok-production/tokenizer.json"
PYTHON_SHARDS = Path("/media/data1tb/stackv2-dedup-sub/Python/")

print(f"Loading {TOK_PATH}")
tok = Tokenizer.from_file(TOK_PATH)
print(f"vocab_size: {tok.get_vocab_size()}")

# Quick sanity
samples = [
    "def __init__(self):",
    "import numpy as np",
    "for i in range(10):",
    "    return self.value",
]
print("\n--- Sanity encodes ---")
for s in samples:
    ids = tok.encode(s).ids
    print(f"{s!r:40s} -> {len(ids)} tok")

# Roundtrip
print("\n--- Roundtrip ---")
text = "import numpy as np\n\ndef hello():\n    print('world')\n"
ids = tok.encode(text).ids
decoded = tok.decode(ids)
ok = decoded == text
print(f"basic: {'PASS' if ok else 'FAIL'}  ({len(text)} chars -> {len(ids)} tok, ratio={len(text)/len(ids):.2f})")
if not ok:
    print(f"  expected: {text!r}")
    print(f"  got:      {decoded!r}")

# Whitespace tokens
print("\n--- Whitespace ---")
patterns = ["    ", "        ", "\t", "\n    ", "\n        ", "\n\t", "\n\n"]
for p in patterns:
    ids = tok.encode(p).ids
    flag = '✓' if len(ids) == 1 else f'✗ ({len(ids)})'
    print(f"  {p!r:20s} -> {len(ids)} tok {flag}")

# Special tokens
print("\n--- Specials ---")
specials = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|im_start|>", "<|im_end|>"]
for st in specials:
    ids = tok.encode(st).ids
    flag = '✓' if len(ids) == 1 else f'✗ ({len(ids)})'
    print(f"  {st:20s} -> {ids} {flag}")

# Real-file roundtrip
print("\n--- Real-file roundtrip ---")
shard_files = sorted(PYTHON_SHARDS.glob("*.parquet"))
print(f"  found {len(shard_files)} parquet shards")

import pyarrow.parquet as pq
random.seed(20260501)
shard = random.choice(shard_files)
print(f"  reading {shard.name}")
table = pq.read_table(shard, columns=["content"])
contents = table.column("content").to_pylist()
random.shuffle(contents)

n_test = 10
n_pass = 0
total_chars = 0
total_tok = 0
for i, content in enumerate(contents[:n_test]):
    if content is None:
        continue
    ids = tok.encode(content).ids
    decoded = tok.decode(ids)
    if decoded == content:
        n_pass += 1
        total_chars += len(content)
        total_tok += len(ids)
    else:
        print(f"  FAIL on file {i}: {len(content)} chars, decoded len {len(decoded)}")
        for j, (a, b) in enumerate(zip(content, decoded)):
            if a != b:
                print(f"    first diff at char {j}: {a!r} vs {b!r}")
                break
print(f"  {n_pass}/{n_test} pass; chars/tok = {total_chars/max(1,total_tok):.3f}")

# Top merges
print("\n--- Top 20 merges (by id, 256-275) ---")
with open(TOK_PATH) as f:
    config = json.load(f)
merges = config["model"]["merges"][:20]
for i, m in enumerate(merges):
    print(f"  {256+i:5d}: {m!r}")

print("\nDone.")

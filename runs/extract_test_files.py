"""Extract a held-out set of .py files for benchmarking."""
import pyarrow.parquet as pq
import random
from pathlib import Path

random.seed(20260501)
out = Path("/home/ezel/Development/WiseTok/runs/bench_corpus")
out.mkdir(parents=True, exist_ok=True)

shard = "/media/data1tb/stackv2-dedup-sub/Python/train-00083-of-00084.parquet"
table = pq.read_table(shard, columns=["content"])
contents = table.column("content").to_pylist()
random.shuffle(contents)

n = 0
total_bytes = 0
for content in contents:
    if content is None or len(content) < 200:
        continue
    fp = out / f"file_{n:05d}.py"
    fp.write_text(content, encoding="utf-8", errors="replace")
    total_bytes += len(content.encode("utf-8", errors="replace"))
    n += 1
    if n >= 200 or total_bytes > 10_000_000:
        break

print(f"wrote {n} files, {total_bytes/1e6:.1f} MB to {out}")

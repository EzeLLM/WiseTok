from __future__ import annotations
import os
from pathlib import Path
import sys
from datasets import load_dataset


def main() -> int:
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/html",
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)

    out_path = Path(os.environ.get(
        "WISETOK_HTML_OUT",
        "/media/data1tb/stackv2-dedup-sub/html/html_raw.txt",
    ))
    target_gb = float(os.environ.get("WISETOK_HTML_TARGET_GB", "2"))
    target_bytes = int(target_gb * 1024**3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            content = row["content"]
            f.write(content)
            f.write("\n")
            written += len(content.encode("utf-8"))
            if written >= target_bytes:
                break

    print(f"Wrote {written / 1024**3:.3f} GB to {out_path}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        os._exit(2)
    os._exit(code)
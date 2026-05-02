#!/bin/bash
# Wait for corpus build (PID arg) to finish, then launch wisetok v2 training.
set -euo pipefail

BUILD_PID="${1:-1521082}"
CORPUS_DIR="/media/data1tb/ezellm-coder-tokenizer/corpus-v2"
OUT_DIR="/media/data1tb/ezellm-coder-tokenizer/wisetok-production-v2"
AGG_FILE="/media/data1tb/ezellm-coder-tokenizer/wisetok-v2.agg"
LOG="/home/ezel/Development/WiseTok/runs/logs/train_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "[launcher] waiting on build pid=$BUILD_PID"
while kill -0 "$BUILD_PID" 2>/dev/null; do
    sleep 30
done
echo "[launcher] build finished at $(date)"
ls -lh "$CORPUS_DIR/"
echo

mkdir -p "$OUT_DIR"
cd /home/ezel/Development/WiseTok

echo "[launcher] launching training, log -> $LOG"
LD_LIBRARY_PATH=/home/ezel/miniconda3/lib RUST_LOG=info \
    ./target/release/wisetok train \
    --files "$CORPUS_DIR"/corpus_python.txt \
            "$CORPUS_DIR"/corpus_java.txt \
            "$CORPUS_DIR"/corpus_cpp.txt \
            "$CORPUS_DIR"/corpus_english.txt \
            "$CORPUS_DIR"/corpus_edutext.txt \
            "$CORPUS_DIR"/corpus_javascript.txt \
            "$CORPUS_DIR"/corpus_c.txt \
            "$CORPUS_DIR"/corpus_math.txt \
            "$CORPUS_DIR"/corpus_html.txt \
    --vocab-size 24576 \
    --pre-tokenizer gpt4+digits \
    --special-preset code \
    --reserve 16 \
    --merge-mode scan \
    --ram-limit 64GB \
    --min-freq 2 \
    --format hf,tiktoken \
    --agg-file "$AGG_FILE" \
    --output "$OUT_DIR" \
    --verbose \
    > "$LOG" 2>&1

echo "[launcher] training finished at $(date)"
tail -5 "$LOG"

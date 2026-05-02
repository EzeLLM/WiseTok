# Tutorial: Training a Custom Tokenizer from Scratch

**Duration:** 15 minutes | **Difficulty:** Beginner | **Prerequisites:** Python 3.8+, pip

## What You'll Learn

- Install TokenLib and its dependencies
- Prepare a text corpus for tokenization
- Train a custom BPE tokenizer with configurable vocabulary size
- Encode and decode text using your trained tokenizer
- Export the tokenizer for use in inference pipelines
- Troubleshoot common training issues

---

## Step 1: Install TokenLib

Open your terminal and install the library:

```bash
$ pip install tokenlib-py
```

Verify the installation:

```bash
$ python -c "from tokenlib import Tokenizer; print(Tokenizer.__doc__)"
```

**Note:** If you see `ModuleNotFoundError`, ensure you're using Python 3.8 or higher (`python --version`).

---

## Step 2: Prepare Your Corpus

Create a text file with your training data. For this tutorial, we'll use a simple example. Save as `corpus.txt`:

```
The quick brown fox jumps over the lazy dog.
Tokenization is the process of breaking text into smaller units.
Machine learning models process text as sequences of tokens.
Natural language processing requires efficient tokenization strategies.
This dataset is small, but in practice you'd use gigabytes of text.
```

**Note:** For real training, use at least 100MB of text. Larger corpora produce better generalizations.

For production corpora, consider:
- **Wikipedia dumps:** https://dumps.wikimedia.org/
- **Common Crawl:** https://commoncrawl.org/
- **Project Gutenberg:** https://www.gutenberg.org/ (for literary texts)

---

## Step 3: Create a Training Script

Create a file named `train_tokenizer.py`:

```python
from tokenlib import TokenizerTrainer

# Initialize the trainer
trainer = TokenizerTrainer(
    vocab_size=512,          # Small for demo; production uses 50k–256k
    pattern=r"'s|'t|'ve|'m|'re|'d|'ll|\w+|\S",  # GPT-4 pattern (default)
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
    min_frequency=2          # Merge pairs that appear ≥2 times
)

# Train on your corpus
print("Training tokenizer on corpus.txt...")
tokenizer = trainer.train(
    ["corpus.txt"],
    num_threads=4            # Use 4 CPU threads (adjust to your machine)
)

# Get statistics
stats = trainer.get_stats()
print(f"Training complete!")
print(f"  Merges applied: {stats['merge_count']}")
print(f"  Final vocab size: {tokenizer.get_vocab_size()}")
```

**Note:** The `pattern` is a regex that controls pre-tokenization. The default GPT-4 pattern preserves contractions and splits punctuation. See [API Reference](#patterns) for alternatives.

---

## Step 4: Run Training

Execute your script:

```bash
$ python train_tokenizer.py
```

Expected output:

```
Training tokenizer on corpus.txt...
[INFO] Initialized: 42 unique chunks
[INFO] Merge 1/256: pair ('the', ' ') -> token 256 (count: 23)
[INFO] Merge 2/256: pair ('ing', ' ') -> token 257 (count: 18)
...
[INFO] Merge 256/256: pair (' ab', 'cd') -> token 511 (count: 1)
Training complete!
  Merges applied: 256
  Final vocab size: 512
```

**Note:** Training on 5KB of text like this should complete in <1 second. On 1GB of text, expect 1–5 minutes.

---

## Step 5: Encode and Decode Text

Update your script to test the tokenizer:

```python
# Test encoding
text = "The quick brown fox"
tokens = tokenizer.encode(text)
print(f"\nOriginal:  {text}")
print(f"Tokens:    {tokens}")
print(f"Token count: {len(tokens)}")

# Test decoding
reconstructed = tokenizer.decode(tokens)
print(f"Decoded:   {reconstructed}")

# Verify round-trip (text -> tokens -> text)
assert text == reconstructed, "Round-trip failed!"
print("✓ Round-trip encoding/decoding verified")

# Encode with special tokens
text_with_special = "<|bos|> Hello world <|eos|>"
tokens = tokenizer.encode(text_with_special, allowed_special={"<|bos|>", "<|eos|>"})
print(f"\nWith special tokens: {tokens}")
```

Run it:

```bash
$ python train_tokenizer.py
```

Output:

```
Original:  The quick brown fox
Tokens:    [256, 257, 258, 259]
Token count: 4
Decoded:   The quick brown fox
✓ Round-trip encoding/decoding verified

With special tokens: [0, 260, 261, 1]
```

---

## Step 6: Save and Load Your Tokenizer

Persist your tokenizer to disk:

```python
# Save the tokenizer
tokenizer.save("my_tokenizer.json")
print("Tokenizer saved to my_tokenizer.json")

# Later, load it
from tokenlib import Tokenizer
loaded = Tokenizer.load("my_tokenizer.json")
print(f"Tokenizer loaded. Vocab size: {loaded.get_vocab_size()}")

# Verify it works
test_tokens = loaded.encode("Hello world")
print(f"Encode test: {test_tokens}")
```

---

## Step 7: Batch Encoding (Advanced)

For multiple texts, use `encode_batch()` for better performance:

```python
texts = [
    "First sentence for tokenization.",
    "Second sentence with special tokens <|pad|>.",
    "Third sentence for batch processing."
]

# Single-threaded (slow)
tokens_list = [tokenizer.encode(t) for t in texts]

# Parallel (4x faster on 4 cores)
tokens_list = tokenizer.encode_batch(texts, num_threads=4)

for i, tokens in enumerate(tokens_list):
    print(f"Text {i}: {len(tokens)} tokens")
```

---

## Troubleshooting

### Issue: "Empty corpus or no chunks after pre-tokenization"

**Cause:** Your corpus file is empty or the regex pattern doesn't match any text.

**Fix:** Check that `corpus.txt` exists and contains text:
```bash
$ wc -l corpus.txt
$ head corpus.txt
```

### Issue: Training takes too long

**Cause:** Large corpus or small `buffer_size`.

**Fix:** Increase `num_threads` and adjust `buffer_size`:
```python
trainer = TokenizerTrainer(
    vocab_size=10000,
    buffer_size=500_000      # Larger buffers = fewer rayon task spawns
)
tokenizer = trainer.train(["corpus.txt"], num_threads=8)
```

### Issue: Decode produces garbled text

**Cause:** Out-of-range token IDs (e.g., manually created IDs that don't exist in vocab).

**Fix:** Only use token IDs returned by `encode()`:
```python
# Good
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

# Bad (avoid manual token construction)
fake_tokens = [999, 1000, 1001]  # These IDs may not exist
decoded = tokenizer.decode(fake_tokens)  # Produces replacement character
```

---

## Next Steps

- Export your tokenizer to **tiktoken** format for fast inference (see [Export Guide](../export_guide.md)).
- Compare your tokenizer's compression ratio against GPT-3.5's.
- Train on different corpora and analyze vocabulary differences.
- Contribute tokenizer presets to the TokenLib community repo!

---

**Questions?** Open an issue on [GitHub](https://github.com/openai/tokenlib/issues) or check the [FAQ](../faq.md).

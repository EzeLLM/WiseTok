# API Reference

## Core Classes

### `Tokenizer`

The main tokenizer class for encoding and decoding text.

#### Signature (Python)

```python
class Tokenizer:
    def __init__(
        self,
        vocab: Dict[bytes, int],
        merges: List[Tuple[int, int]],
        pattern: str = r"'s|'t|'ve|'m|'re|'d|'ll|\w+|\S",
        special_tokens: Dict[str, int] = None
    ) -> None:
        """Initialize a tokenizer with vocabulary and merge rules."""
```

#### Signature (TypeScript)

```typescript
class Tokenizer {
  constructor(
    vocab: Map<string, number>,
    merges: Array<[number, number]>,
    pattern?: string,
    specialTokens?: Map<string, number>
  );
}
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `encode` | `text: str`, `allowed_special: set = None` | `List[int]` | Tokenize text into IDs. |
| `encode_batch` | `texts: List[str]`, `num_threads: int = 4` | `List[List[int]]` | Parallel encoding of multiple texts. |
| `decode` | `tokens: List[int]` | `str` | Convert token IDs back to text. |
| `decode_batch` | `token_lists: List[List[int]]` | `List[str]` | Parallel decoding of multiple token lists. |
| `get_vocab_size` | | `int` | Return the vocabulary size. |
| `save` | `path: str` | `None` | Serialize tokenizer to JSON. |
| `load` | `path: str` | `Tokenizer` | Load tokenizer from JSON (static method). |

#### Example

```python
tokenizer = Tokenizer(
    vocab={b"hello": 256, b"world": 257},
    merges=[(0, 1), (1, 2)]
)
tokens = tokenizer.encode("hello world")
assert tokenizer.decode(tokens) == "hello world"
```

---

### `TokenizerTrainer`

Trainer for building custom tokenizers from a corpus.

#### Signature (Python)

```python
class TokenizerTrainer:
    def __init__(
        self,
        vocab_size: int = 50257,
        pattern: str = r"'s|'t|'ve|'m|'re|'d|'ll|\w+|\S",
        special_tokens: List[str] = None,
        buffer_size: int = 100_000,
        min_frequency: int = 2
    ) -> None:
        """Initialize a tokenizer trainer."""
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `train` | `files: List[str]`, `num_threads: int = 4` | `Tokenizer` | Train on files or an iterable of strings. |
| `train_from_iterator` | `iterator`, `num_threads: int = 4` | `Tokenizer` | Train from a Python iterator (releases GIL). |
| `get_stats` | | `Dict[str, int]` | Return training statistics (merge count, vocab size). |

#### Parameters Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 50257 | Target vocabulary size. |
| `pattern` | str | GPT-4 pattern | Regex for pre-tokenization. |
| `special_tokens` | List[str] | [] | Special tokens to reserve (e.g. `["<|pad|>"]`). |
| `buffer_size` | int | 100,000 | Streaming buffer size in chunks. |
| `min_frequency` | int | 2 | Minimum pair frequency to merge. |

#### Example

```python
trainer = TokenizerTrainer(
    vocab_size=10000,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"]
)

# Train from files
tokenizer = trainer.train(
    ["train.txt", "val.txt"],
    num_threads=8
)

# Or from an iterator
def text_generator():
    with open("huge_corpus.txt") as f:
        for line in f:
            yield line.strip()

tokenizer = trainer.train_from_iterator(text_generator())
tokenizer.save("my_tokenizer.json")
```

---

## Functions

### `load_pretrained(model_name: str) -> Tokenizer`

Load a pretrained tokenizer by name from remote cache.

**Parameters:**
- `model_name` (str): One of `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `claude-3-opus`, `llama-2-7b`.

**Returns:** `Tokenizer` instance.

**Raises:**
- `ValueError`: Unknown model name.
- `IOError`: Tokenizer file not found in cache.

**Example:**
```python
tokenizer = load_pretrained("gpt-4-turbo")
tokens = tokenizer.encode("Hello!")
```

---

### `get_token_stats(tokens: List[int]) -> Dict[str, Any]`

Compute statistics about a token sequence.

**Parameters:**
- `tokens` (List[int]): Token ID sequence.

**Returns:**
```python
{
    "total_tokens": int,
    "unique_tokens": int,
    "entropy": float,
    "most_common": List[Tuple[int, int]]
}
```

**Example:**
```python
stats = get_token_stats([256, 257, 256, 258, 256])
print(stats["entropy"])  # 0.918...
```

---

### `merge_tokenizers(*tokenizers: Tokenizer, conflict_strategy: str = "union") -> Tokenizer`

Merge multiple tokenizers into a single vocabulary.

**Parameters:**
- `*tokenizers`: Variable number of `Tokenizer` instances.
- `conflict_strategy`: One of `"union"` (take all), `"intersection"` (common only), `"first"` (use first tokenizer).

**Returns:** Merged `Tokenizer`.

**Raises:**
- `ValueError`: Conflicting merge rules and `conflict_strategy == "intersection"`.

**Example:**
```python
tok_a = load_pretrained("gpt-3.5-turbo")
tok_b = load_pretrained("claude-3-opus")
combined = merge_tokenizers(tok_a, tok_b, conflict_strategy="union")
```

---

## Exceptions

| Exception | Base | Raised When |
|-----------|------|-------------|
| `TokenizerError` | `Exception` | General tokenizer error. |
| `VocabularyError` | `TokenizerError` | Invalid vocabulary or vocab size. |
| `TrainingError` | `TokenizerError` | Training fails (empty corpus, I/O errors). |
| `DecodeError` | `TokenizerError` | Invalid token ID in decode. |

---

## Configuration File Format

Tokenizers are serialized as JSON (schema `v1.0`):

```json
{
  "schema_version": "1.0",
  "vocab_size": 10000,
  "vocab": {
    "aGVsbG8=": 256,
    "d29ybGQ=": 257
  },
  "merges": [[0, 1], [1, 2]],
  "pattern": "'s|'t|'ve|'m|'re|'d|'ll|\\w+|\\S",
  "special_tokens": {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2
  }
}
```

Byte sequences in `vocab` are base64-encoded. Special tokens map to reserved IDs below 256.

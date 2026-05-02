# TokenLib 🚀

[![CI/CD](https://img.shields.io/github/actions/workflow/status/openai/tokenlib/test.yml?branch=main)](https://github.com/openai/tokenlib/actions)
[![PyPI version](https://img.shields.io/pypi/v/tokenlib-py.svg)](https://pypi.org/project/tokenlib-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.70+](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)

A high-performance tokenizer library with support for custom vocabularies, fast inference, and production-grade serialization. Built for language models with sub-millisecond encoding latency.

## Installation

### Python (pip)

```bash
pip install tokenlib-py
```

### TypeScript / Node.js (npm)

```bash
npm install @tokenlib/core
```

### Rust (Cargo)

```toml
[dependencies]
tokenlib = "0.8.1"
```

## Quick Start

### Python

```python
from tokenlib import Tokenizer

# Load a pretrained tokenizer
tokenizer = Tokenizer.from_pretrained("gpt-3.5-turbo")

# Encode text
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [15339, 11, 1917, 0]

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Train a custom tokenizer from scratch
from tokenlib import TokenizerTrainer

trainer = TokenizerTrainer(vocab_size=10000, pattern=r"'s|'t|'ve|'m|'re|'d|'ll|\w+|\S")
tokenizer = trainer.train(["corpus.txt"], num_threads=4)
tokenizer.save("my_tokenizer.json")
```

### TypeScript

```typescript
import { Tokenizer } from "@tokenlib/core";

async function main() {
  const tokenizer = await Tokenizer.fromPreTrained("gpt-3.5-turbo");
  
  const tokens = tokenizer.encode("Hello, world!");
  console.log(tokens); // [15339, 11, 1917, 0]
  
  const text = tokenizer.decode(tokens);
  console.log(text); // "Hello, world!"
}

main();
```

### Bash

```bash
tokenlib encode --model gpt-3.5-turbo "Hello, world!"
tokenlib train --corpus corpus.txt --vocab-size 10000 --output my_tokenizer.json
```

## API Reference

### `Tokenizer.encode(text: str, allowed_special=None) -> List[int]`

Encodes a string into token IDs using the loaded vocabulary.

**Parameters:**
- `text` (str): Input text to tokenize.
- `allowed_special` (set, optional): Special tokens to preserve. Default: empty set.

**Returns:** List of integer token IDs.

**Example:**
```python
tokens = tokenizer.encode("The quick brown fox", allowed_special={"<|endoftext|>"})
```

### `Tokenizer.decode(tokens: List[int]) -> str`

Decodes a list of token IDs back to text.

**Parameters:**
- `tokens` (List[int]): List of token IDs.

**Returns:** Decoded string.

### `Tokenizer.from_pretrained(model_name: str) -> Tokenizer`

Loads a pretrained tokenizer by name.

**Supported models:**
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`
- `claude-3-opus`
- `claude-3-sonnet`
- `llama-2-7b`

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

```bash
git clone https://github.com/openai/tokenlib
cd tokenlib
cargo build --release
maturin develop  # Install Python extension
pytest tests/
```

## License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) file for details.

## Citation

If you use TokenLib in your research, please cite:

```bibtex
@software{tokenlib2024,
  title = {TokenLib: High-Performance Tokenization for Language Models},
  author = {OpenAI Contributors},
  year = {2024},
  url = {https://github.com/openai/tokenlib}
}
```

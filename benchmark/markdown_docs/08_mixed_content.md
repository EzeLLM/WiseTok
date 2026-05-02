# Advanced Markdown: Mixed Content Showcase

This document demonstrates advanced markdown features: nested lists, tables with alignment, LaTeX math, HTML embedding, task lists, and footnotes.

---

## 1. Nested Lists (4 Levels Deep)

Training a tokenizer involves multiple phases:

1. **Data Preparation**
   - Text cleaning
     - Remove control characters
       - Filter null bytes
       - Strip BOM markers
     - Normalize Unicode
   - Chunking
     - Split by lines
       - Configurable chunk size
       - Handle edge cases
       - Preserve boundaries

2. **Vocabulary Building**
   - Regex pre-tokenization
     - Pattern compilation
       - Cache compiled regex
       - Support lookarounds via `fancy-regex`
   - Chunk frequency counting
     - Hash aggregation
       - Parallel reduce
       - Lock-free accumulation

3. **Merge Loop**
   - Pair frequency tracking
   - Iterative merging
     - Select top pair
       - By count (descending)
       - By ID (tie-break)

---

## 2. Tables with Alignment

### Configuration Parameters

| Parameter | Type | Default | Range | Alignment |
|-----------|------|---------|-------|-----------|
| `vocab_size` | int | 50,257 | 256–1,000,000 | ← left |
| `pattern` | str | GPT-4 regex | Any valid regex | ← left |
| `buffer_size` | int | 100,000 | 1,000–10,000,000 | ← left |
| `min_frequency` | int | 2 | 1–100 | ← left |
| `num_threads` | int | # CPUs | 1–256 | ← left |

### Performance Metrics

| Corpus Size | Training Time | Merge Count | Final Vocab | Compression (%) |
|:-----------:|:-------------:|:-----------:|:-----------:|:---------------:|
| 1 MB | 0.1 sec | 16,000 | 50k | 72.3 |
| 100 MB | 8 sec | 49,000 | 50k | 71.8 |
| 1 GB | 120 sec | 49,257 | 50k | 71.5 |
| 10 GB | 1,200 sec | 49,257 | 50k | 71.4 |

---

## 3. Inline and Block LaTeX Math

The BPE algorithm minimizes the vocabulary size while maximizing compression. Let $V$ be vocabulary size, $T$ be total tokens, and $C$ be corpus size:

**Compression ratio:**

$$\text{ratio} = \frac{C}{T} = \frac{\sum_{\text{tokens}} \text{frequency}}{\text{total}\, \text{tokens}}$$

For a typical corpus:

$$\text{ratio} \approx \frac{1 \text{ byte}}{1.3 \text{ tokens}} \approx 77\%$$

The entropy of token distribution approximates:

$$H = -\sum_{i=1}^{|V|} p_i \log_2(p_i)$$

where $p_i$ is the normalized frequency of token $i$.

**Key insight:** Zipfian distributions (natural language) compress well because high-frequency tokens are merged early.

---

## 4. HTML Embedded in Markdown

### Details (Collapsible)

<details>
<summary><strong>Click to expand: BPE Algorithm Pseudocode</strong></summary>

```python
def bpe(text, merges, special_tokens):
    """Apply BPE decoding to reconstruct tokens."""
    for left, right in merges:
        text = text.replace((left, right), merge(left, right))
    return text
```

</details>

### Keyboard Input

Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> to open the command palette in VS Code.

### Subscripts and Superscripts

The formula for entropy uses base-2 logarithm: $\log_2(x)$ where $x > 0$. 

Token ID<sub>next</sub> is calculated as: ID<sub>next</sub> = max(ID<sub>prev</sub>) + 1.

The efficiency is O(n<sup>2</sup> log m) for encoding.

---

## 5. Task Lists

Training checklist:

- [x] Prepare corpus (100GB Wikipedia dump)
- [x] Configure trainer (`vocab_size=50k`, `pattern=gpt-4`)
- [ ] Run training (estimated 2 hours on V100)
- [x] Validate output against reference implementations
- [ ] Export to `tiktoken` format
- [x] Benchmark encoding speed (<1ms per 1000 chars)
- [ ] Deploy to production tokenizer service

---

## 6. Footnotes

Byte Pair Encoding (BPE) was introduced by Sennrich et al. in 2016[^1] as a subword tokenization algorithm. The algorithm is simple yet effective[^2], and has become the de facto standard for language model tokenizers[^3].

[^1]: Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *Proceedings of the 54th Annual Meeting of the ACL*, pp. 1715–1725. https://aclanthology.org/P16-1162/

[^2]: The time complexity of naive BPE is O(n²) per merge, but optimized implementations (like TokenLib's incremental approach) achieve O(n log n) amortized. See [ADR-2024-06-15](../04_architecture_decision_record.md).

[^3]: As of 2024, BPE is used by OpenAI's GPT models, Anthropic's Claude, Meta's Llama, and most other state-of-the-art LLMs. The algorithm's popularity is partly due to its deterministic nature and ease of implementation across programming languages.

---

## 7. Combined Example: Advanced Configuration Table

Below is a complex table mixing alignment, code, and special characters:

| Feature | Type | Example | Notes |
|:--------|:-----|:--------|:------|
| **Pattern** | Regex | `r"'s\|'t\|'ve\|'m\|'re\|'d\|'ll\|\w+\|\S"` | GPT-4 default; supports lookarounds |
| **Pair Tie-break** | Function | `(left, right) -> (count, -left, -right)` | Lexicographic, deterministic |
| **Merge Heap** | Data Structure | `OctonaryHeap<MergeJob>` | 8-ary heap, cache-friendly |
| **Vocabulary** | `HashMap<Pair, u32>` | `{(42, 100): 256, ...}` | 256–1M entries possible |
| **Output** | JSON | See §4 [Configuration File Format](./02_API_reference.md#configuration-file-format) | Mergeable ranks + special tokens |

---

## 8. Code Blocks in Multiple Languages

### Rust (Incremental Merge)

```rust
// Core merge loop in TokenLib
fn train_core_incremental(
    mut words: Vec<Word>,
    target_vocab_size: usize,
) -> HashMap<Pair, u32> {
    let mut pair_counts = AHashMap::new();
    let mut merges = HashMap::new();
    let mut new_token_id = 256u32;
    
    // Initial pair count
    for word in &words {
        for pair in word.get_pairs() {
            *pair_counts.entry(pair).or_insert(0) += word.count as i64;
        }
    }
    
    // Merge loop
    while merges.len() < target_vocab_size - 256 {
        // Find best pair
        let (best_pair, count) = pair_counts
            .iter()
            .max_by_key(|(_, &c)| c)
            .expect("No pairs left");
        
        // Apply merge
        merges.insert(*best_pair, new_token_id);
        new_token_id += 1;
        
        // Update pair counts (lazy refresh on heap)
        // ...
    }
    
    merges
}
```

### Python (Training Script)

```python
from tokenlib import TokenizerTrainer
import logging

logging.basicConfig(level=logging.INFO)

def train_large_corpus():
    trainer = TokenizerTrainer(
        vocab_size=50_257,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
        min_frequency=2
    )
    
    def corpus_iterator():
        """Stream from disk to avoid loading everything into memory."""
        with open("corpus.txt") as f:
            for line in f:
                yield line.strip()
    
    tokenizer = trainer.train_from_iterator(
        corpus_iterator(),
        num_threads=8  # Parallel pre-tokenization
    )
    
    return tokenizer

if __name__ == "__main__":
    tok = train_large_corpus()
    tok.save("my_tokenizer.json")
```

### TypeScript (Encoding Client)

```typescript
import { Tokenizer, encodeAsync } from "@tokenlib/core";

async function main(): Promise<void> {
  const tokenizer = await Tokenizer.fromPreTrained("gpt-4");
  
  const text: string = "Tokenization is essential for NLP.";
  const tokens: number[] = await tokenizer.encode(text);
  
  console.log(`Text: ${text}`);
  console.log(`Tokens: ${tokens.join(", ")}`);
  console.log(`Token count: ${tokens.length}`);
  
  // Batch encoding (4x faster on 4 cores)
  const texts: string[] = [
    "First text",
    "Second text",
    "Third text"
  ];
  
  const batchTokens = await encodeAsync(texts, {
    tokenizer,
    numThreads: 4
  });
  
  batchTokens.forEach((t, i) => {
    console.log(`Text ${i}: ${t.length} tokens`);
  });
}

main().catch(console.error);
```

---

## 9. Cross-references and Links

See the following related documentation:

- [API Reference](./02_API_reference.md) — Function signatures and parameter tables
- [CHANGELOG](./03_CHANGELOG.md) — Version history and release notes
- [Architecture Decision Record](./04_architecture_decision_record.md) — Why we chose incremental pair counting
- [Tutorial](./06_tutorial_step_by_step.md) — Step-by-step training guide

External references:

- [OpenAI Tokenizer Docs](https://platform.openai.com/docs/guides/tokens)
- [Hugging Face Tokenizers Library](https://huggingface.co/docs/tokenizers/)
- [tiktoken on PyPI](https://pypi.org/project/tiktoken/)

---

**Document version:** 1.2 | **Last updated:** 2024-10-25 | **License:** CC-BY-4.0

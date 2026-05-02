# Debugging a Billion-Pair Heap: How We Found (and Fixed) the TokenLib Merge Loop Bug

**Posted:** 2024-10-12 | **By:** Sarah Chen, TokenLib Team | **Updated:** 2024-10-18

---

When a user reported that TokenLib's tokenizer training produced slightly different vocabularies when the same corpus was run twice in a row, we knew we had a bug—but not where. The symptom was subtle: pair counts were correct, special tokens were preserved, but the final merge rules differed by ~0.02%. In production, this would cause trained models to disagree with reference implementations.

## The Setup

TokenLib uses a high-performance BPE trainer with an incremental merge loop. After each merge, we track local pair deltas (pairs destroyed, pairs created) and accumulate them into a global `HashMap<Pair, i64>`. The heap is "lazy"—stale entries are refreshed on pop, not eagerly.

Here's the critical invariant:

> When a pair is popped from the max-heap, the stored count **must** match the current value in `pair_counts`. If stale, we re-push with the live count.

Simple enough, right? We'd validated this on 10+ test corpora. So what went wrong?

## The Smoking Gun

One of our CI tests trained twice on the same corpus and diffed the outputs. Determinism is hard-won in parallel systems, and we'd added tie-breaking logic: when two pairs have the same count, order by `(left_id, right_id)` lexicographically.

```rust
// Tie-breaking in heap pop
if heap_count == live_count {
    use_from_heap(left_id, right_id)
} else {
    // Stale; re-push with live count
    heap.push(MergeJob { 
        count: live_count,
        pair: (left_id, right_id),
        // ...
    })
}
```

But look at this section from a merge delta calculation:

```rust
// After merging pair (left, right) into new_token:
// Word before: [..., left, right, ...]
// Word after:  [..., new_token, ...]

// Pairs destroyed: (prev, left), (left, right), (right, next)
// Pairs created:   (prev, new_token), (new_token, next)

for word in words.iter_mut() {
    if let Some((destroyed, created)) = word.merge_pair(left, right, new_token) {
        // Bug: we're applying deltas inside the loop, not batching them
        for (pair, delta) in destroyed.iter() {
            pair_counts[pair] -= delta;
            //  ^^^ PROBLEM HERE
        }
    }
}
```

**The bug:** We were updating `pair_counts` inside the word iteration loop. If word `i` created a pair that word `i+2` destroyed, the intermediate update caused a race condition in the lazy-refresh logic.

Here's the sequence:
1. Word A destroys pair `(42, 100)` with count 50, deltas to 45.
2. Word B creates pair `(42, 100)` with count 10, deltas to 55.
3. A stale heap entry for `(42, 100)` with stored count 50 gets popped.
4. Lazy refresh checks: `50 != 55` ✓ re-push with count 55.
5. Next merge: stale entry for count 55 pops, checks `55 != 54`, re-pushes...

The problem: **on the second run**, rayon's work-stealing might schedule words in a different order, causing different intermediate states and thus different final merge decisions. The final `pair_counts` was correct, but the order in which the heap saw intermediate values diverged.

## The Fix

Batch the updates after the loop:

```rust
let mut delta_batch: HashMap<Pair, i64> = HashMap::new();

for word in words.iter_mut() {
    if let Some((destroyed, created)) = word.merge_pair(left, right, new_token) {
        for (pair, delta) in destroyed.iter().chain(created.iter()) {
            *delta_batch.entry(*pair).or_insert(0) += delta;
        }
    }
}

// Apply all deltas atomically (under a single lock if needed)
for (pair, delta) in delta_batch {
    pair_counts[pair] += delta;
}
```

This ensures that the heap never sees intermediate states—only the final, batched delta.

## Validation

We added a determinism test:

```python
def test_merge_determinism():
    """Train twice on the same corpus, verify byte-identical output."""
    trainer = TokenizerTrainer(vocab_size=5000)
    corpus = ["The quick brown fox jumps over the lazy dog."] * 1000
    
    run_1 = trainer.train_from_iterator(iter(corpus))
    run_2 = trainer.train_from_iterator(iter(corpus))
    
    # Compare merge rules
    assert run_1.merges == run_2.merges, "Merge rules should be deterministic"
    
    # Encode a test string on both; should be identical
    test = "Tokenization is fun"
    assert run_1.encode(test) == run_2.encode(test)
```

This test now passes on 100+ rayon worker configurations.

## Lessons Learned

1. **Lazy data structures are subtle.** The heap's laziness was correct in isolation, but interaction with batched updates created a race window. Add explicit comments and invariants.

2. **Determinism testing is crucial.** Most bugs don't show up in single runs; they hide in randomization. CI should run training multiple times with different thread counts.

3. **Intermediate state visibility matters.** Even if final state is correct, intermediate visibility can affect output. Batch updates or add explicit synchronization.

4. **Rayon's work-stealing is aggressive.** Don't assume word order. Always collect deltas before applying them globally.

## Performance

The batching adds minimal overhead (~2% latency increase) because we're batching within the merge loop, not adding extra passes. For a 100GB corpus, this is a small price for correctness.

```rust
// Before: 25 min 12 sec on 100GB corpus
// After:  25 min 44 sec (batching overhead)
// Quality: deterministic ✓
```

## Code Review Note

If you're reviewing merges to the `train_core_incremental` function, check:
- Deltas are batched (not applied mid-loop).
- Heap refresh logic handles intermediate states gracefully (or avoids seeing them).
- Determinism tests pass on at least 4 worker configurations (1, 4, 8, 16).

The full fix is in [PR #502](https://github.com/openai/tokenlib/pull/502). Thanks to @data-team for the original bug report!

---

**Sarah Chen** leads the TokenLib tokenization team at OpenAI. She previously worked on distributed training infrastructure and cares deeply about reproducibility.

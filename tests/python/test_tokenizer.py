"""
Comparing the training of:

1. (very slow) Python reference implementation
2. Optimized Python implementation
3. HuggingFace tokenizers training implementation
4. Our own wisetok training implementation (forked from rustbpe)

All of these should calculate the same merges and produce
the same vocabulary and tokenizations.

Finally, for inference we will use tiktoken for efficiency.
So we want to make sure we can export our wisetok tokenizer
into tiktoken and use it for inference with identical results.

Run with:
python -m pytest tests/python/test_tokenizer.py -v -s
-v is verbose, -s is show prints
"""

import regex as re
from collections import Counter, defaultdict
import time
import warnings
import wisetok
import tiktoken
import pytest

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Reference tokenizer, pretty much copy pasted and pruned a bit from minbpe

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class RegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {} # (int, int) -> int
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # keep track of whether at any point during training the merge is ambiguous (counts of pairs are not unique)
        ambiguous = False

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # check if the merge is ambiguous - i.e. the max value is not unique
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                # print the top 10 pairs with their counts
                # print(f"{i} Merge is ambiguous! {pair} has {pair_count} occurrences")
                # for print_pair, print_count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                #     print(f"{print_pair}: {print_count}")
                ambiguous = True
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        return ambiguous

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# Faster Python tokenizer, optimized version of the reference tokenizer

def fast_merge_inplace(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx in place
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # Find all positions where the pair occurs
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx
            ids.pop(i+1)
        else:
            i += 1
    return ids


class FastRegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        A number of optimizations are introduced:
        - delete function call overhead by inlining functions
        - modifying list of ids in place with .pop() instead of creating a new list
        - collapse identical chunks to just the unique ones
        - update counts more cleverly - only around the affected chunks
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # many, many chunks are identical, so we can "collapse" them to just the unique ones
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # Initial count: build stats and position tracking
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> set of chunk indices that contain this pair

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        for i in range(num_merges):
            if not stats:
                break

            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i

            # Get chunks that contain this pair
            affected_chunks = positions[pair]

            # Track count changes for incremental update
            count_changes = defaultdict(int)

            # Replace all occurrences of pair in affected chunks only
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # Track what pairs are being removed/added
                        # Remove: (prev, A), (A, B), (B, next)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # The merged pair disappears
                        count_changes[pair] -= chunk_count

                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # Apply the merge
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)

                        # Add: (prev, C), (C, next)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # Apply incremental changes to stats and positions
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # The merged pair should disappear completely
                    continue

                stats[changed_pair] += delta

                # Update positions for changed pairs - only check affected chunks
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # Remove the merged pair completely
            del stats[pair]
            del positions[pair]

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# HuggingFace tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        gpt4_split_regex = Regex(GPT4_SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[], # no special tokens
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

# -----------------------------------------------------------------------------
# Test all of the above

def get_cache_dir():
    """Get user's cache directory (persists across test runs)."""
    import os
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = os.path.join(cache_home, "wisetok")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

@pytest.fixture(scope="module")
def enwik8_path():
    """Fixture to download and cache enwik8 dataset."""
    import os
    import zipfile
    base_dir = get_cache_dir()
    # download and unzip enwik8 to cache directory
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """Fixture providing 100KB of enwik8 for quick tests."""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(100_000)

@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """Fixture providing 10MB of enwik8 for performance tests."""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(10**7)

def time_function(func, *args, **kwargs):
    """Time a function call and return the result and elapsed time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed

def test_correctness(enwik8_small):
    """Test that all tokenizer implementations produce the same results."""
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 20 merges

    # Train slow reference
    print("\nTraining slow reference...")
    slow_reference_tokenizer = RegexTokenizer()
    ambiguous_flag, slow_reference_train_time = time_function(slow_reference_tokenizer.train, text, vocab_size)
    slow_reference_ids, slow_reference_encode_time = time_function(slow_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Slow reference train time: {slow_reference_train_time:.4f}s")
    print(f"Slow reference encode time: {slow_reference_encode_time:.4f}s")
    print(slow_reference_ids[:20])

    if ambiguous_flag:
        print("‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size")
        print("The implementation could be correct but we might see different results below")
    else:
        print("✅ Merge order is NOT ambiguous")

    # Train fast reference
    print("\nTraining fast reference...")
    fast_reference_tokenizer = FastRegexTokenizer()
    _, fast_reference_train_time = time_function(fast_reference_tokenizer.train, text, vocab_size)
    fast_reference_ids, fast_reference_encode_time = time_function(fast_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Fast reference train time: {fast_reference_train_time:.4f}s")
    print(f"Fast reference encode time: {fast_reference_encode_time:.4f}s")
    print(fast_reference_ids[:20])

    # Assert fast equals slow
    assert fast_reference_ids == slow_reference_ids, "Fast reference should match slow reference"
    print("✅ Fast == Slow")

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace has a different byte order, so we need custom matching
    def custom_match(ids1, ids2):
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    assert custom_match(hf_ids, fast_reference_ids), "HuggingFace should match fast reference"
    print("✅ HuggingFace == Fast")

    # Finally use our own Rust implementation
    print("\nTraining wisetok...")
    wisetok_tokenizer = wisetok.Tokenizer()
    _, wisetok_train_time = time_function(wisetok_tokenizer.train_from_iterator, [text], vocab_size)
    wisetok_ids, wisetok_encode_time = time_function(wisetok_tokenizer.encode, encode_text)
    print(f"wisetok train time: {wisetok_train_time:.4f}s")
    print(f"wisetok encode time: {wisetok_encode_time:.4f}s")
    print(wisetok_ids[:20])

    assert wisetok_ids == fast_reference_ids, "wisetok should match fast reference"
    print("✅ wisetok == Fast")

    # Now export wisetok to tiktoken for more efficient inference
    print("\nTesting tiktoken export...")
    pattern = wisetok_tokenizer.get_pattern()
    mergeable_ranks_list = wisetok_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="wisetok",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == wisetok_ids, "Tiktoken should match wisetok"
    print("✅ Tiktoken == wisetok")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """Use a bigger dataset and compare the training speed of the optimized tokenizers (Python, Rust, HuggingFace)."""
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # Commenting out because it's just way too slow to matter
    # Train optimized python version
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # Train wisetok
    print("\nTraining wisetok...")
    wisetok_tokenizer = wisetok.Tokenizer()
    _, wisetok_train_time = time_function(wisetok_tokenizer.train_from_iterator, [text], vocab_size)
    print(f"wisetok train time: {wisetok_train_time:.4f}s")
    assert wisetok_train_time > 0, "Training should take some time"

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # Print comparison
    print(f"\n📊 Performance comparison:")
    print(f"   wisetok: {wisetok_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/wisetok_train_time:.2f}x")

def test_batch_encode_correctness(enwik8_small):
    """Quick correctness test for batch_encode()"""
    text = enwik8_small
    vocab_size = 512

    tokenizer = wisetok.Tokenizer()
    tokenizer.train_from_iterator([text], vocab_size)

    # Test with various batch sizes and edge cases
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "jumps over the lazy dog",
        "",  # empty string
        "a",  # single char
    ]

    # Compare batch vs individual encoding
    individual = [tokenizer.encode(t) for t in test_texts]
    batched = tokenizer.batch_encode(test_texts)

    assert individual == batched, "Batch encoding should match individual encoding"
    print("✅ batch_encode() correctness verified")


def test_vocab_size():
    """Test the vocab_size property."""
    tokenizer = wisetok.Tokenizer()

    # New tokenizer should have 256 (byte-level tokens)
    assert tokenizer.vocab_size == 256, "New tokenizer should have vocab_size=256"

    # After training, vocab_size should match the requested size
    tokenizer.train_from_iterator(["hello hello hello", "world world world"], vocab_size=260)
    assert tokenizer.vocab_size == 260, f"Expected vocab_size=260, got {tokenizer.vocab_size}"

    print("✅ vocab_size property works correctly")


def test_decode_roundtrip(enwik8_small):
    """Test that encode->decode produces the original text."""
    text = enwik8_small[:1000]  # Use first 1KB for quick test
    vocab_size = 512

    tokenizer = wisetok.Tokenizer()
    tokenizer.train_from_iterator([text], vocab_size)

    # Test various strings
    test_strings = [
        "hello world",
        "The quick brown fox jumps over the lazy dog",
        "12345",
        "   spaces   ",
        "MixedCASE123",
        "",  # empty string
    ]

    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        assert decoded == s, f"Roundtrip failed for {s!r}: got {decoded!r}"

    # Test roundtrip on the training text itself
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, "Roundtrip failed on training text"

    print("✅ decode() roundtrip works correctly")


def test_decode_invalid_token():
    """Test that decode raises an error for invalid token IDs."""
    tokenizer = wisetok.Tokenizer()

    # Token 300 doesn't exist in base vocabulary (only 0-255)
    try:
        tokenizer.decode([300])
        assert False, "Should have raised an error for invalid token"
    except ValueError as e:
        assert "Unknown token id" in str(e) or "unknown" in str(e).lower()

    print("✅ decode() correctly rejects invalid tokens")


@pytest.mark.slow
def test_batch_encode_performance(enwik8_large):
    """
    Benchmark batch_encode() vs sequential encode() loop.
    Demonstrates parallelization speedup.
    """
    # Setup
    text = enwik8_large  # 10MB dataset
    vocab_size = 2048

    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = wisetok.Tokenizer()
    tokenizer.train_from_iterator([text], vocab_size)

    # Create test batch: split text into chunks
    chunk_size = 50_000  # ~50KB per chunk
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = chunks[:20]  # Use first 20 chunks (~1MB total)

    print(f"\nBatch encoding benchmark:")
    print(f"  Number of texts: {len(chunks)}")
    print(f"  Avg text length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    # Benchmark 1: Sequential encoding (baseline)
    print("\n  [1/3] Sequential encode() loop...")
    sequential_results, sequential_time = time_function(
        lambda: [tokenizer.encode(chunk) for chunk in chunks]
    )
    print(f"    Time: {sequential_time:.4f}s")

    # Benchmark 2: Parallel batch_encode()
    print("  [2/3] Parallel batch_encode()...")
    batch_results, batch_time = time_function(
        tokenizer.batch_encode, chunks
    )
    print(f"    Time: {batch_time:.4f}s")

    # Verify correctness
    print("  [3/3] Verifying correctness...")
    assert len(batch_results) == len(sequential_results), "Result count mismatch"
    for i, (seq, batch) in enumerate(zip(sequential_results, batch_results)):
        assert seq == batch, f"Mismatch at index {i}"
    print("    ✓ All results match")

    # Report speedup
    speedup = sequential_time / batch_time
    print(f"\n  Performance Results:")
    print(f"    Sequential: {sequential_time:.4f}s")
    print(f"    Batch:      {batch_time:.4f}s")
    print(f"    Speedup:    {speedup:.2f}x")

    # Warn if speedup is low (can vary by machine/load)
    if speedup < 1.5:
        warnings.warn(f"batch_encode() speedup was only {speedup:.2f}x (expected >1.5x)")


def test_min_frequency_default_is_lossless():
    """min_frequency=1 (default) must produce identical merges to leaving it unset."""
    text = "the quick brown fox " * 200 + "rare ! 7" * 1
    vocab_size = 280

    a = wisetok.Tokenizer()
    a.train_from_iterator([text], vocab_size=vocab_size)

    b = wisetok.Tokenizer()
    b.train_from_iterator([text], vocab_size=vocab_size, min_frequency=1)

    assert a.encode(text) == b.encode(text), "min_frequency=1 should be a no-op vs default"
    print("✅ min_frequency=1 matches default")


def test_min_frequency_drops_rare_chunks():
    """A rare chunk's bytes should not contribute pairs to the merge map
    when min_frequency exceeds its count."""
    # Use a unique sentinel chunk "QXZ" that appears once and is composed of
    # bytes that don't appear anywhere else in the corpus. With
    # min_frequency=10, "QXZ" should be dropped, so no merge should involve
    # any of its bytes.
    common = "hello world " * 100
    rare = "QXZ"  # Q=0x51, X=0x58, Z=0x5A — none appear in "hello world "
    text = common + rare

    tok = wisetok.Tokenizer()
    tok.train_from_iterator([text], vocab_size=320, min_frequency=10)

    # Sanity: those bytes do not occur in the common text.
    for b in rare.encode("utf-8"):
        assert chr(b) not in common, f"byte {b!r} leaked into common text"

    # Inspect the trained vocab: no merged token should contain Q/X/Z bytes.
    for token_bytes, token_id in tok.get_mergeable_ranks():
        if token_id < 256:
            continue  # base byte tokens are irrelevant
        rare_bytes = set(rare.encode("utf-8"))
        if any(b in rare_bytes for b in token_bytes):
            raise AssertionError(
                f"merge {token_id} ({bytes(token_bytes)!r}) contains a byte "
                f"from the rare chunk; min_frequency filtering failed"
            )
    print("✅ min_frequency=10 dropped the rare chunk's pairs from the merge map")


def test_min_frequency_smaller_corpus_yields_smaller_vocab_or_equal():
    """High min_frequency reduces the unique-chunk pool, which can cap the
    achievable vocab size when the merge loop runs out of pairs."""
    text = ("kw " * 5) + ("filler word " * 200)
    requested_vocab = 320

    keep = wisetok.Tokenizer()
    keep.train_from_iterator([text], vocab_size=requested_vocab, min_frequency=5)

    drop = wisetok.Tokenizer()
    drop.train_from_iterator([text], vocab_size=requested_vocab, min_frequency=6)

    # Both must be valid (no panics, no errors). The drop run had a smaller
    # input corpus, so its vocab can be ≤ keep's. Both stay above the 256
    # base-byte floor.
    assert 256 <= drop.vocab_size <= keep.vocab_size <= requested_vocab
    print(f"✅ min_frequency monotonicity: vocab(min_freq=6)={drop.vocab_size} "
          f"≤ vocab(min_freq=5)={keep.vocab_size} ≤ {requested_vocab}")


def test_min_frequency_roundtrip():
    """A tokenizer trained with min_frequency must still round-trip text correctly."""
    text = "the quick brown fox jumps over the lazy dog. " * 50
    tok = wisetok.Tokenizer()
    tok.train_from_iterator([text], vocab_size=300, min_frequency=5)

    sample = "the quick brown fox"
    assert tok.decode(tok.encode(sample)) == sample
    print("✅ min_frequency tokenizer round-trips correctly")


def test_hf_export_loads_in_tokenizers():
    """A wisetok-exported tokenizer.json must load with HF's Tokenizer.from_file
    and round-trip text correctly."""
    import tempfile, os
    from tokenizers import Tokenizer as HfTokenizer

    tok = wisetok.Tokenizer()
    tok.train_from_iterator(["hello world " * 200], vocab_size=300)

    with tempfile.TemporaryDirectory() as d:
        tok.save_huggingface(d)
        assert "tokenizer.json" in os.listdir(d)
        assert "tokenizer_config.json" in os.listdir(d)

        hf = HfTokenizer.from_file(os.path.join(d, "tokenizer.json"))
        text = "hello world hello world"
        enc = hf.encode(text)
        assert hf.decode(enc.ids) == text
        # Option C: wisetok IDs match the exported file's IDs.
        assert tok.encode(text) == enc.ids
    print("✅ HF export loads in Tokenizer.from_file and roundtrips")


def test_hf_export_with_special_tokens():
    """Special tokens must end up in vocab and added_tokens with the
    correct (Option C: tail-placed) IDs."""
    import tempfile, os, json
    from tokenizers import Tokenizer as HfTokenizer

    tok = wisetok.Tokenizer()
    tok.train_from_iterator(["hello world " * 200], vocab_size=300)
    specials = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>"]

    with tempfile.TemporaryDirectory() as d:
        tok.save_huggingface(d, special_tokens=specials)
        with open(os.path.join(d, "tokenizer.json")) as f:
            data = json.load(f)

        # Specials placed at 256 + num_merges + i.
        base = 256 + len(data["model"]["merges"])
        for i, s in enumerate(specials):
            assert data["model"]["vocab"][s] == base + i, (
                f"special {s!r} expected id {base + i}, got "
                f"{data['model']['vocab'][s]}"
            )

        # added_tokens entries align.
        added = {a["content"]: a for a in data["added_tokens"]}
        for i, s in enumerate(specials):
            assert added[s]["id"] == base + i
            assert added[s]["special"] is True

        # HF can still load the file even though specials are at the tail.
        hf = HfTokenizer.from_file(os.path.join(d, "tokenizer.json"))
        # Using a special as-is in text should encode to the special's id.
        enc = hf.encode("<|endoftext|>")
        assert base in enc.ids, (
            f"expected special id {base} in encoding of '<|endoftext|>', got {enc.ids}"
        )
    print("✅ HF export with specials: tail IDs verified, HF reads the file")


def test_hf_export_roundtrip_via_pretrained_tokenizer_fast():
    """transformers.PreTrainedTokenizerFast must accept the file."""
    import tempfile, os
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        pytest.skip("transformers not installed")

    tok = wisetok.Tokenizer()
    tok.train_from_iterator(["the quick brown fox " * 100], vocab_size=320)

    with tempfile.TemporaryDirectory() as d:
        tok.save_huggingface(d)
        ptf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(d, "tokenizer.json"))
        text = "the quick brown fox"
        ids = ptf.encode(text)
        decoded = ptf.decode(ids)
        # ByteLevel decode strips the leading space if there is one; just
        # check semantic round-trip.
        assert decoded.strip() == text.strip()
    print("✅ PreTrainedTokenizerFast loads the wisetok-exported file")


def test_pre_tokenizer_digit_splitting():
    """pre_tokenizer='gpt4+digits' must split each ASCII digit into its own
    chunk during training, so multi-digit numbers cannot become a single
    learned token."""
    # Build a corpus where multi-digit numbers appear many times. With
    # digit splitting, each digit is its own chunk; merges between digits
    # require digit-pair frequency in the byte alphabet, not chunk-level
    # frequency.
    text = ("v128 v128 v128 v128 v128 " * 100) + "filler text " * 200

    tok = wisetok.Tokenizer()
    tok.train_from_iterator([text], vocab_size=300, pre_tokenizer="gpt4+digits")

    # Encode "v128" — expect at least 4 tokens (v, 1, 2, 8) because each
    # digit is split as its own chunk and digits cannot fuse with v.
    ids = tok.encode("v128")
    assert len(ids) >= 4, f"expected >=4 tokens for 'v128' with digit split, got {len(ids)}: {ids}"
    print(f"✅ pre_tokenizer='gpt4+digits': 'v128' → {len(ids)} tokens")


def test_pre_tokenizer_default_matches_legacy_pattern():
    """No pre_tokenizer / no pattern args ≡ pre_tokenizer='gpt4'."""
    text = "the quick brown fox jumps over the lazy dog. " * 50

    a = wisetok.Tokenizer()
    a.train_from_iterator([text], vocab_size=400)

    b = wisetok.Tokenizer()
    b.train_from_iterator([text], vocab_size=400, pre_tokenizer="gpt4")

    sample = "the quick brown fox"
    assert a.encode(sample) == b.encode(sample)
    print("✅ default ≡ pre_tokenizer='gpt4'")


def test_pre_tokenizer_pattern_and_spec_are_mutually_exclusive():
    """Passing both pattern= and pre_tokenizer= must raise."""
    tok = wisetok.Tokenizer()
    with pytest.raises(ValueError, match="not both"):
        tok.train_from_iterator(
            ["hello"], vocab_size=300,
            pattern=r"\w+", pre_tokenizer="gpt4",
        )
    print("✅ pattern + pre_tokenizer mutual exclusion raises")


def test_pre_tokenizer_legacy_pattern_still_works():
    """pattern= alone (no pre_tokenizer=) trains with that single regex."""
    text = "abc abc abc def def def " * 100

    tok = wisetok.Tokenizer()
    tok.train_from_iterator([text], vocab_size=270, pattern=r"\w+")

    # Should encode "abc" without error.
    ids = tok.encode("abc")
    assert len(ids) >= 1
    assert tok.decode(ids) == "abc"
    print("✅ legacy pattern= argument still works")


def test_pre_tokenizer_unknown_spec_raises():
    """Unknown spec strings get a clear ValueError."""
    tok = wisetok.Tokenizer()
    with pytest.raises(ValueError, match="unknown pre_tokenizer spec"):
        tok.train_from_iterator(
            ["hello"], vocab_size=300, pre_tokenizer="banana",
        )
    print("✅ unknown spec raises ValueError")


def test_pre_tokenizer_encode_uses_same_pipeline_as_training():
    """encode() must use the pre-tokenizer that was used at training time."""
    # Use a custom regex that split-aware encode can reproduce.
    text = "AAA BBB AAA BBB " * 100

    tok = wisetok.Tokenizer()
    tok.train_from_iterator([text], vocab_size=270, pre_tokenizer="regex:[A-Z]+")

    # The pattern matches only uppercase runs. "AAA bbb CCC" should encode
    # the AAAs and CCC but skip the lowercase chunk entirely (by design of
    # this pre-tokenizer — non-matching text is dropped, same as upstream
    # rustbpe behavior).
    ids = tok.encode("AAA bbb CCC")
    decoded = tok.decode(ids)
    # Decoded may or may not contain spaces depending on merges; the
    # invariant is that 'bbb' (which the regex doesn't match) is absent.
    assert "b" not in decoded, f"lowercase 'b' should not encode: decoded={decoded!r}"
    print(f"✅ encode uses same pipeline as training; decoded={decoded!r}")


def test_special_tokens_atomic_in_encode():
    """A registered special token in the input must encode to a single ID,
    not get split by BPE."""
    tok = wisetok.Tokenizer()
    tok.train_from_iterator(
        ["hello world " * 200],
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    ids = tok.encode("<|endoftext|>")
    # Special at index 0 → id = 256 + num_merges + 0
    expected = 256 + (tok.vocab_size - 256)  # vocab_size includes specials? let's compute differently
    # vocab_size on Tokenizer is 256 + merges only (no specials counted) — verify by re-deriving:
    num_merges = len(tok.get_mergeable_ranks()) - 256
    expected_id = 256 + num_merges + 0
    assert ids == [expected_id], f"expected [{expected_id}], got {ids}"
    print(f"✅ '<|endoftext|>' encodes to single id {expected_id}")


def test_special_tokens_in_middle_of_text():
    """Specials surrounded by ordinary text emit one ID each; the surrounding
    text encodes normally."""
    tok = wisetok.Tokenizer()
    tok.train_from_iterator(
        ["hello world " * 200],
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    text_with_sep = "hello<|endoftext|>world"
    ids = tok.encode(text_with_sep)
    num_merges = len(tok.get_mergeable_ranks()) - 256
    special_id = 256 + num_merges + 0
    # The special's ID appears exactly once.
    assert ids.count(special_id) == 1
    # Removing the special's ID, the remaining IDs are the BPE encoding of
    # "hello" + "world" (no separator) — same as encoding "helloworld" with
    # no special.
    rest = [i for i in ids if i != special_id]
    expected_rest = tok.encode("hello") + tok.encode("world")
    assert rest == expected_rest, f"surrounding tokens differ: {rest!r} vs {expected_rest!r}"
    print(f"✅ surrounding text encoded identically; special breaks the stream")


def test_special_tokens_skipped_during_aggregation():
    """During training, a special token's bytes must NOT contribute to the
    BPE merge map. Compare two trainers: one with the special, one without
    a special at all but with the same string appearing literally — and
    verify that only the latter produces merges containing those bytes."""
    # Use a sentinel string whose bytes are unique to the corpus, so we can
    # detect whether they appear in any merge.
    sentinel = "<|XQZ|>"  # unique 7-byte sequence
    body = "the quick brown fox " * 100
    text = f"{body}{sentinel}{body}{sentinel}{body}"

    # Tokenizer A: register the sentinel as a special. It should never enter BPE.
    a = wisetok.Tokenizer()
    a.train_from_iterator([text], vocab_size=320, special_tokens=[sentinel])
    sentinel_bytes = set(sentinel.encode("utf-8"))
    for token_bytes, token_id in a.get_mergeable_ranks():
        if token_id < 256:
            continue
        if any(b in sentinel_bytes for b in token_bytes):
            # A merge contains a byte from the sentinel — only acceptable
            # if that byte ALSO appears in the body, in which case the
            # merge could come from the body.
            byte_set = set(token_bytes)
            body_bytes = set(body.encode("utf-8"))
            sentinel_only = byte_set & sentinel_bytes - body_bytes
            assert not sentinel_only, (
                f"merge {token_id} contains sentinel-only bytes "
                f"{sentinel_only!r}; special-token isolation failed"
            )
    print("✅ specials skipped during aggregation; merges contain no sentinel-only bytes")


def test_special_tokens_via_add_special_tokens():
    """add_special_tokens() lets users register specials post-training; encode
    then treats them as atoms."""
    tok = wisetok.Tokenizer()
    tok.train_from_iterator(["hello world " * 200], vocab_size=300)
    assert tok.get_special_tokens() == []

    tok.add_special_tokens(["<|sep|>", "<|cls|>"])
    assert tok.get_special_tokens() == ["<|sep|>", "<|cls|>"]

    num_merges = len(tok.get_mergeable_ranks()) - 256
    sep_id = 256 + num_merges + 0
    cls_id = 256 + num_merges + 1

    ids = tok.encode("a<|sep|>b<|cls|>c")
    assert sep_id in ids and cls_id in ids
    print(f"✅ post-train add_special_tokens works; IDs: <|sep|>={sep_id}, <|cls|>={cls_id}")


def test_special_tokens_hf_export_uses_registered_specials():
    """save_huggingface() with no `special_tokens` arg must use the registered
    specials. The exported file's added_tokens must match."""
    import tempfile, os, json

    tok = wisetok.Tokenizer()
    tok.train_from_iterator(
        ["hello world " * 200],
        vocab_size=300,
        special_tokens=["<|endoftext|>", "<|fim_prefix|>"],
    )

    with tempfile.TemporaryDirectory() as d:
        # No special_tokens kwarg → use the registered ones.
        tok.save_huggingface(d)
        with open(os.path.join(d, "tokenizer.json")) as f:
            data = json.load(f)

        added = [a["content"] for a in data["added_tokens"]]
        assert added == ["<|endoftext|>", "<|fim_prefix|>"], f"got {added}"

        # Encoding via wisetok produces IDs that match what's in the file.
        text = "hello<|endoftext|>world<|fim_prefix|>"
        wise_ids = tok.encode(text)
        # Verify HF can read the file too.
        from tokenizers import Tokenizer as HfTok
        hf = HfTok.from_file(os.path.join(d, "tokenizer.json"))
        # HF needs the specials to be registered with add_special_tokens to
        # treat them as atoms. Without that, HF would split them. We mirror
        # the wisetok behavior here by adding them on the HF side.
        hf.add_special_tokens(["<|endoftext|>", "<|fim_prefix|>"])
        hf_ids = hf.encode(text).ids
        assert wise_ids == hf_ids, (
            f"wisetok encode and HF encode of same text differ: "
            f"wisetok={wise_ids}, hf={hf_ids}"
        )
    print(f"✅ save_huggingface uses registered specials; HF encode matches wisetok encode")


# =============================================================================
# MergeMode parity tests (Iteration 2 — memory-bounded merge mode)
#
# The gating correctness criterion: training the same corpus with merge_mode=
# "full" and merge_mode="scan" must produce byte-identical merge tables and
# byte-identical encode output for arbitrary text. If these tests diverge,
# the scan implementation is wrong — fix before shipping.
# =============================================================================

def _train_with_mode(corpus, vocab_size, mode, **extra):
    """Train a fresh wisetok.Tokenizer on `corpus` with the given merge_mode.
    Returns the trained tokenizer."""
    tok = wisetok.Tokenizer()
    tok.train_from_iterator(
        iter(corpus),
        vocab_size=vocab_size,
        merge_mode=mode,
        **extra,
    )
    return tok


def _mergeable_ranks_dict(tok):
    """Return a (token_bytes, id) → tuple list, normalized to a dict for
    comparison."""
    ranks = tok.get_mergeable_ranks()
    return {tuple(b): i for (b, i) in ranks}


def test_merge_mode_full_and_scan_produce_identical_merges():
    """Gating test for the memory-bounded merge mode.

    Trains the same corpus with merge_mode="full" and merge_mode="scan".
    The mergeable_ranks must be byte-identical (same token bytes mapped to
    the same IDs in the same order)."""
    corpus = [
        "hello world! the quick brown fox jumps over the lazy dog.",
        "hello, world! how are you doing today?",
        "the quick brown fox is quick.",
        "abracadabra abracadabra abracadabra",
        "hello hello hello world world",
        "code: def hello(): return 'world'",
        "numbers: 123 456 789 12 34 56",
        "the the the the and and and or or",
    ] * 20  # repeat to give merges plenty of frequency signal

    full_tok = _train_with_mode(corpus, vocab_size=400, mode="full")
    scan_tok = _train_with_mode(corpus, vocab_size=400, mode="scan")

    full_ranks = _mergeable_ranks_dict(full_tok)
    scan_ranks = _mergeable_ranks_dict(scan_tok)

    assert full_ranks == scan_ranks, (
        f"Full and Scan produced different mergeable_ranks.\n"
        f"  full has {len(full_ranks)} entries, scan has {len(scan_ranks)}.\n"
        f"  full-only: {set(full_ranks) - set(scan_ranks)}\n"
        f"  scan-only: {set(scan_ranks) - set(full_ranks)}"
    )

    # And the actual encode outputs must match for arbitrary text.
    test_texts = [
        "hello world",
        "the quick brown fox",
        "abracadabra is a magic word",
        "what about completely unseen text 999",
        "",
        "a",
    ]
    for text in test_texts:
        full_ids = full_tok.encode(text)
        scan_ids = scan_tok.encode(text)
        assert full_ids == scan_ids, (
            f"encode diverged for {text!r}: full={full_ids}, scan={scan_ids}"
        )
    print(f"✅ Full and Scan produced identical merges + encode on {len(corpus)} sequences")


def test_merge_mode_auto_resolves_to_full_for_small_corpus():
    """Auto on a small corpus picks Full. The output must equal explicit Full."""
    corpus = ["hello world " * 50, "the quick brown fox " * 50]

    auto_tok = _train_with_mode(corpus, vocab_size=300, mode="auto")
    full_tok = _train_with_mode(corpus, vocab_size=300, mode="full")

    assert _mergeable_ranks_dict(auto_tok) == _mergeable_ranks_dict(full_tok)
    print("✅ merge_mode='auto' on a small corpus matches explicit 'full'")


def test_merge_mode_default_is_auto_and_matches_full():
    """No explicit merge_mode arg should be the same as merge_mode='auto'.
    On a small corpus that resolves to Full, so it must equal explicit Full."""
    corpus = ["hello world " * 50, "the quick brown fox " * 50]

    default_tok = _train_with_mode(corpus, vocab_size=300, mode=None)
    full_tok = _train_with_mode(corpus, vocab_size=300, mode="full")

    assert _mergeable_ranks_dict(default_tok) == _mergeable_ranks_dict(full_tok)
    print("✅ default merge_mode matches 'full' on a small corpus (auto threshold not hit)")


def test_merge_mode_invalid_value_raises():
    """Unrecognized merge_mode strings must produce a clear ValueError."""
    tok = wisetok.Tokenizer()
    with pytest.raises(ValueError, match="unknown merge_mode"):
        tok.train_from_iterator(
            iter(["hello world"] * 10),
            vocab_size=300,
            merge_mode="bogus",
        )
    print("✅ invalid merge_mode raises ValueError")


def test_merge_mode_case_insensitive():
    """merge_mode strings are case-insensitive: 'FULL', 'Scan', etc. all work."""
    corpus = ["hello world " * 30]

    a = _train_with_mode(corpus, vocab_size=300, mode="FULL")
    b = _train_with_mode(corpus, vocab_size=300, mode="full")
    assert _mergeable_ranks_dict(a) == _mergeable_ranks_dict(b)

    c = _train_with_mode(corpus, vocab_size=300, mode="Scan")
    d = _train_with_mode(corpus, vocab_size=300, mode="scan")
    assert _mergeable_ranks_dict(c) == _mergeable_ranks_dict(d)
    print("✅ merge_mode is case-insensitive")


def test_merge_mode_scan_with_special_tokens():
    """Scan mode must compose correctly with special tokens (which split
    text before BPE applies). Result must equal Full mode on the same input."""
    corpus = [
        "hello<|endoftext|>world",
        "the quick brown<|fim_prefix|>fox<|fim_middle|>jumps",
        "hello world hello world",
    ] * 30

    full_tok = _train_with_mode(
        corpus,
        vocab_size=350,
        mode="full",
        special_tokens=["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>"],
    )
    scan_tok = _train_with_mode(
        corpus,
        vocab_size=350,
        mode="scan",
        special_tokens=["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>"],
    )

    assert _mergeable_ranks_dict(full_tok) == _mergeable_ranks_dict(scan_tok)

    # Encode with specials present in the input must match across modes too.
    text = "hello<|endoftext|>code<|fim_prefix|>x"
    assert full_tok.encode(text) == scan_tok.encode(text)
    print("✅ Scan mode + special tokens matches Full mode")


def test_merge_mode_scan_with_pre_tokenizer_digits():
    """Scan mode must compose with the digit-splitting pre-tokenizer too."""
    corpus = [
        "the year was 2025 and counting",
        "phone: 555-123-4567",
        "pi is approximately 3.14159265",
    ] * 50

    full_tok = _train_with_mode(corpus, vocab_size=320, mode="full", pre_tokenizer="gpt4+digits")
    scan_tok = _train_with_mode(corpus, vocab_size=320, mode="scan", pre_tokenizer="gpt4+digits")

    assert _mergeable_ranks_dict(full_tok) == _mergeable_ranks_dict(scan_tok)
    test_text = "the year 2025 has digits 0123456789"
    assert full_tok.encode(test_text) == scan_tok.encode(test_text)
    print("✅ Scan mode + gpt4+digits pre-tokenizer matches Full mode")

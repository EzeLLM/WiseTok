#!/usr/bin/env python3
"""
Capture HuggingFace tokenizers reference output for wisetok HF-export module.

Trains two reference BPE tokenizers using HF `tokenizers.trainers.BpeTrainer`
and saves them via `transformers.PreTrainedTokenizerFast.save_pretrained`.

Run with: /home/ezel/miniconda3/bin/python capture_script.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

# --- Constants ---------------------------------------------------------------

GPT4_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

CODE_PRESET_SPECIALS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|file_sep|>",
    "<|repo_name|>",
    "<|filename|>",
]

HERE = Path(__file__).resolve().parent


# --- Tokenizer construction --------------------------------------------------

def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(GPT4_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def make_trainer(vocab_size: int, specials: list[str] | None) -> trainers.BpeTrainer:
    kwargs = dict(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    if specials:
        kwargs["special_tokens"] = specials
    return trainers.BpeTrainer(**kwargs)


# --- Corpora -----------------------------------------------------------------

MINIMAL_CORPUS = ["hello world " * 1000]


def build_full_corpus() -> list[str]:
    """A small mixed corpus: English prose + Python source. ~5-20KB."""
    english = (
        "It was the best of times, it was the worst of times, it was the age "
        "of wisdom, it was the age of foolishness, it was the epoch of belief, "
        "it was the epoch of incredulity, it was the season of Light, it was "
        "the season of Darkness, it was the spring of hope, it was the winter "
        "of despair, we had everything before us, we had nothing before us, "
        "we were all going direct to Heaven, we were all going direct the other "
        "way - in short, the period was so far like the present period, that "
        "some of its noisiest authorities insisted on its being received, for "
        "good or for evil, in the superlative degree of comparison only.\n"
    ) * 20

    python_src = '''
import os
import sys
from typing import Optional, List, Dict, Tuple

class TokenizerExporter:
    """Export trained BPE to HuggingFace tokenizer.json."""

    def __init__(self, vocab: Dict[bytes, int], merges: List[Tuple[int, int]]):
        self.vocab = vocab
        self.merges = merges
        self._byte_to_unicode = self._build_byte_to_unicode()

    def _build_byte_to_unicode(self) -> Dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\\xa1"), ord("\\xac") + 1))
            + list(range(ord("\\xae"), ord("\\xff") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def export(self, path: str) -> None:
        import json
        out = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {self._byte_to_unicode[k[0]]: v for k, v in self.vocab.items()},
                "merges": [f"{a} {b}" for a, b in self.merges],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: exporter.py <output.json>", file=sys.stderr)
        return 1
    exporter = TokenizerExporter({}, [])
    exporter.export(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''' * 5

    code_mix = "\n".join([
        "def add(a, b): return a + b",
        "x = 12345 + 67890",
        "for i in range(100): print(i, i*i, i**3)",
        "tokens = ['hello', 'world', '<|endoftext|>']",
        "url = 'https://example.com/path/to/resource?id=42&name=foo'",
        "regex = r'[a-zA-Z0-9_]+'",
        "data = {'key': 'value', 'count': 9999, 'items': [1, 2, 3]}",
    ]) * 30

    return [english, python_src, code_mix]


# --- Main --------------------------------------------------------------------

def train_and_save(name: str, vocab_size: int, corpus: list[str], specials: list[str] | None) -> Path:
    out_dir = HERE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    trainer = make_trainer(vocab_size, specials)
    tok.train_from_iterator(corpus, trainer=trainer)

    # 1) Raw save (single tokenizer.json) — capture before wrapping.
    raw_path = out_dir / "raw_tokenizer.json"
    tok.save(str(raw_path))

    # 2) Wrapped save via transformers — generates the full sidecar set.
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok)
    wrapped.save_pretrained(str(out_dir))

    print(f"[{name}] saved to {out_dir}")
    print(f"[{name}] files: {sorted(p.name for p in out_dir.iterdir())}")
    return out_dir


def write_byte_to_unicode_table() -> None:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    # The ByteLevel.alphabet() returns the 256 unicode-mapped characters.
    # We need to recover the (byte -> unicode_char) mapping. The canonical
    # GPT-2 algorithm is:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_unicode = dict(zip(bs, [chr(c) for c in cs]))

    # Sanity: the union of mapped chars must equal HF's alphabet().
    hf_set = set(alphabet)
    ours = set(byte_to_unicode.values())
    assert ours == hf_set, f"mismatch: {ours ^ hf_set}"
    # Sanity: byte 0x20 (space) -> 'Ġ' (U+0120).
    assert byte_to_unicode[0x20] == "Ġ", byte_to_unicode[0x20]

    rs_path = HERE / "byte_to_unicode.rs"
    lines = [
        "// GPT-2 / ByteLevel byte-to-unicode mapping table.",
        "// Generated from `tokenizers.pre_tokenizers.ByteLevel.alphabet()`.",
        "// Each entry maps a raw byte (index) to the Unicode codepoint",
        "// (encoded as a UTF-8 &str) that HF's ByteLevel pre-tokenizer uses",
        "// in vocab keys and serialized output.",
        "//",
        "// Verified: BYTE_TO_UNICODE[0x20] == \"\\u{0120}\" (space -> 'Ġ').",
        "",
        "pub const BYTE_TO_UNICODE: [&str; 256] = [",
    ]
    for b in range(256):
        ch = byte_to_unicode[b]
        # Use \u{XXXX} escape for non-ASCII printable; raw char for ASCII printable.
        cp = ord(ch)
        # Escape backslash and double-quote.
        if ch == "\\":
            literal = '"\\\\"'
        elif ch == '"':
            literal = '"\\""'
        elif 0x21 <= cp <= 0x7E:
            literal = f'"{ch}"'
        else:
            literal = f'"\\u{{{cp:04x}}}"'
        lines.append(f"    {literal}, // 0x{b:02x} ({b})  U+{cp:04X}")
    lines.append("];")
    lines.append("")
    rs_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {rs_path}")


def main() -> int:
    print(f"python: {sys.version}")
    import tokenizers as _tk
    import transformers as _tf
    print(f"tokenizers: {_tk.__version__}")
    print(f"transformers: {_tf.__version__}")

    train_and_save("minimal", vocab_size=1000, corpus=MINIMAL_CORPUS, specials=None)
    train_and_save("full", vocab_size=2000, corpus=build_full_corpus(), specials=CODE_PRESET_SPECIALS)
    write_byte_to_unicode_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

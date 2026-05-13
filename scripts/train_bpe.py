"""Train a BPE tokenizer on the TinyStories sample.

8k vocab is the sweet spot for TinyStories at the ~30M model scale: small
enough that the embedding layer (8000 * d_model) stays cheap, large enough
to compress short English words to ~1 token most of the time.

Output: tokenizer/tinystories_bpe_8k/tokenizer.json

The trained tokenizer is loadable later via:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer/tinystories_bpe_8k/tokenizer.json")
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFC, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "tinystories_sample.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "tinystories_bpe_8k"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_tokenizer() -> Tokenizer:
    """Byte-level BPE — same family as GPT-2 / Llama. No <unk> in practice
    because the byte vocabulary covers every input. We still register an
    <unk> token to keep the schema consistent.
    """
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = Sequence([NFC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train(input_path: Path, output_dir: Path, vocab_size: int) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}\nRun scripts/download_tinystories_sample.py first.")

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE (vocab={vocab_size}) on {input_path} ...")
    start = time.time()
    tok.train([str(input_path)], trainer)
    elapsed = time.time() - start
    print(f"Trained in {elapsed:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "tokenizer.json"
    tok.save(str(out_path))
    print(f"Saved → {out_path}")


def sanity_check(output_dir: Path, input_path: Path) -> None:
    tok = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    samples = [
        "Once upon a time, there was a little girl named Mia.",
        "The dog ran very fast through the green field.",
        "Hello",
        "supercalifragilistic",
    ]
    print("\nEncode/decode roundtrip:")
    for s in samples:
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        ok = "✓" if dec.strip() == s.strip() else "✗"
        print(f"  {ok} ({len(enc.ids):2d} tok / {len(s):2d} char, ratio {len(s)/len(enc.ids):.2f})")
        print(f"     in : {s}")
        print(f"     toks: {enc.tokens}")
        print(f"     out: {dec}")

    # Compression metric on the training set
    print("\nCompression on first 1000 lines of training data:")
    n_chars = 0
    n_tokens = 0
    with input_path.open() as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            n_chars += len(line)
            n_tokens += len(tok.encode(line).ids)
    print(f"  chars: {n_chars:,}")
    print(f"  tokens: {n_tokens:,}")
    print(f"  chars/token: {n_chars / n_tokens:.2f}")

    print(f"\nVocab size: {tok.get_vocab_size()}")
    print(f"Special tokens: {SPECIAL_TOKENS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("-v", "--vocab-size", type=int, default=8192)
    ap.add_argument("--skip-train", action="store_true", help="Only run sanity check on existing tokenizer.")
    args = ap.parse_args()

    if not args.skip_train:
        train(args.input, args.output_dir, args.vocab_size)
    sanity_check(args.output_dir, args.input)


if __name__ == "__main__":
    main()

"""Pre-tokenize the TinyStories sample into a single uint16 binary shard.

Output layout:
    data/tinystories_train.bin   — uint16 LE, ~99% of stories
    data/tinystories_val.bin     — uint16 LE, ~1% of stories
    data/tinystories_meta.json   — vocab_size, dtype, train/val token counts

Why uint16: vocab_size 8192 fits comfortably (uint16 holds 0..65535).
Why flat binary: lets the training loop `np.memmap` it directly with zero
parsing overhead. Same pattern as Karpathy's nanoGPT.

Each story is encoded as `<bos> <body tokens> <eos>` and concatenated into
the shard. The training loop will pull random `seq_len`-sized windows out
of the flat stream, so story boundaries are implicit — the model sees BOS
and EOS as natural punctuation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "tinystories_sample.txt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tinystories_bpe_8k" / "tokenizer.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-t", "--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--val-fraction", type=float, default=0.01, help="Fraction of stories held out for validation (default 1%).")
    ap.add_argument("--batch-size", type=int, default=4096, help="Stories per encode_batch call.")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise SystemExit("Tokenizer missing <bos> or <eos>. Retrain with the updated trainer.")
    if vocab_size > 65535:
        raise SystemExit(f"Vocab {vocab_size} doesn't fit uint16; bump dtype to uint32 in this script.")

    print(f"Tokenizer: vocab={vocab_size}, <bos>={bos_id}, <eos>={eos_id}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.out_dir}")

    # Count lines first so we can split deterministically by index.
    print("Counting lines ...")
    with args.input.open() as f:
        total_lines = sum(1 for _ in f)
    val_start = int(total_lines * (1.0 - args.val_fraction))
    print(f"{total_lines:,} stories total → train [0:{val_start:,}), val [{val_start:,}:{total_lines:,})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "tinystories_train.bin"
    val_path = args.out_dir / "tinystories_val.bin"

    # Stream through, encode in batches, append to whichever shard.
    start = time.time()
    train_tokens = 0
    val_tokens = 0

    with args.input.open() as f, train_path.open("wb") as f_train, val_path.open("wb") as f_val:
        batch: list[str] = []
        idx = 0

        def flush(end_idx: int) -> tuple[int, int]:
            nonlocal train_tokens, val_tokens
            if not batch:
                return 0, 0
            encs = tok.encode_batch(batch)
            start_idx = end_idx - len(batch)
            t_added = 0
            v_added = 0
            for i, enc in enumerate(encs):
                ids = [bos_id, *enc.ids, eos_id]
                arr = np.asarray(ids, dtype=np.uint16)
                if (start_idx + i) >= val_start:
                    arr.tofile(f_val)
                    v_added += arr.size
                else:
                    arr.tofile(f_train)
                    t_added += arr.size
            train_tokens += t_added
            val_tokens += v_added
            return t_added, v_added

        for line in f:
            line = line.rstrip("\n")
            batch.append(line)
            idx += 1
            if len(batch) >= args.batch_size:
                flush(idx)
                batch.clear()
                if idx % (args.batch_size * 4) == 0:
                    elapsed = time.time() - start
                    rate = idx / elapsed
                    pct = 100.0 * idx / total_lines
                    eta = (total_lines - idx) / rate if rate > 0 else 0
                    print(f"  {idx:>7,} / {total_lines:,} ({pct:5.1f}%)  rate {rate:7.0f}/s  ETA {eta:5.0f}s")
        # Final partial batch
        flush(idx)

    elapsed = time.time() - start
    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": tok.token_to_id("<pad>"),
        "unk_id": tok.token_to_id("<unk>"),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_stories": val_start,
        "val_stories": total_lines - val_start,
        "tokenizer_path": str(args.tokenizer.relative_to(PROJECT_ROOT)),
        "source_path": str(args.input.relative_to(PROJECT_ROOT)),
        "build_time_seconds": round(elapsed, 1),
    }
    (args.out_dir / "tinystories_meta.json").write_text(json.dumps(meta, indent=2))

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Train: {train_tokens:>13,} tokens → {train_path}")
    print(f"  Val:   {val_tokens:>13,} tokens → {val_path}")
    print(f"  Meta:  {args.out_dir / 'tinystories_meta.json'}")
    print(f"  Avg tokens/story: {(train_tokens + val_tokens) / total_lines:.1f}")


if __name__ == "__main__":
    main()

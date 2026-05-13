"""Stream a sample of TinyStories from HuggingFace and write to a flat text file.

Streaming mode avoids downloading the full 2 GB dataset just to train a
small tokenizer. We take the first N stories from the train split and
write them as one story per line (newlines within stories are replaced by
spaces so the file is line-delimited).

Output: data/tinystories_sample.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinystories_sample.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-stories", type=int, default=200_000, help="How many stories to take (default 200k).")
    ap.add_argument("-o", "--output", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming roneneldan/TinyStories → {args.output}")
    print(f"Target: {args.num_stories:,} stories")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    start = time.time()
    n_chars = 0
    with args.output.open("w") as f:
        for i, row in enumerate(ds):
            if i >= args.num_stories:
                break
            text = (row["text"] or "").replace("\n", " ").strip()
            if not text:
                continue
            f.write(text + "\n")
            n_chars += len(text) + 1
            if (i + 1) % 10_000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  {i + 1:>7,} stories | {n_chars / 1e6:6.1f} MB | {rate:5.0f}/s")

    elapsed = time.time() - start
    print()
    print(f"Done. {n_chars / 1e6:.1f} MB written to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

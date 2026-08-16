import json
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset

from src.data.tokenizer import Tokenizer
from src.data.train_tokenizer import train_bpe

def download_corpus(out_path: str, split: str) -> str:
    """Download a TinyStories split and write it to disk as one story per line.

    Skips the download if out_path already exists.

    Args:
        out_path: Destination path for the plain-text corpus file.
        split: Dataset split to download (e.g. "train" or "validation").

    Returns:
        The path to the corpus file, as a string.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path)

    dataset = load_dataset("roneneldan/TinyStories", split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        for row in dataset:
            story = row["text"].strip().replace("\n", " ")
            if story:
                file.write(story + "\n")
    return str(out_path)

def iter_line_batches(corpus_path: str, batch_lines: int = 10_000) -> Iterator[list[str]]:
    """Group a corpus file into ordered batches of non-empty lines.

    Keeps batching policy (empty lines skipped, final partial batch included)
    separate from encoding in `encode_corpus_to_bin`.

    Args:
        corpus_path: Path to a text file with one story per line.
        batch_lines: Number of lines per batch (the last batch may be smaller).

    Yields:
        Lists of non-empty, newline-stripped lines of length up to batch_lines.
    """
    batch: list[str] = []
    with open(corpus_path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            batch.append(line)

            if len(batch) >= batch_lines:
                yield batch
                batch = []
    if batch:
        yield batch


def encode_corpus_to_bin(
    corpus_path: str, 
    tokenizer: Tokenizer, 
    bin_path: str, 
    batch_lines: int = 10_000
) -> int:
    """Tokenize a corpus and write the token ids to a flat .bin file.

    Each line is encoded and terminated with an <eos> token id before being
    concatenated. Token dtype is chosen based on vocab size (uint16 if it
    fits, uint32 otherwise) to keep the file compact.

    Args:
        corpus_path: Path to a text file with one story per line.
        tokenizer: Tokenizer used to encode lines into token ids.
        bin_path: Destination path for the output .bin file.
        batch_lines: Number of lines encoded per batch.

    Returns:
        The total number of tokens written.
    """
    dtype = np.uint16 if tokenizer.get_vocab_size() < 65536 else np.uint32
    eos_id = tokenizer.token_to_id("<eos>")

    chunks: list[np.ndarray] = []

    for batch in iter_line_batches(corpus_path, batch_lines):
        for ids in tokenizer.encode_batch(batch):
            ids = ids + [eos_id]
            chunks.append(np.array(ids, dtype=dtype))

    all_ids = np.concatenate(chunks) if chunks else np.array([], dtype=dtype)

    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(bin_path, dtype=dtype, mode="w+", shape=all_ids.shape)
    mm[:] = all_ids
    return int(len(all_ids))

def prepare_data(
    raw_dir: str = "data/raw",
    tokenizer_dir: str = "data/tokenizer",
    bin_dir: str = "data/processed",
    vocab_size: int = 8000
) -> None:
    """Run the full data prep pipeline: download, train tokenizer, encode to .bin.

    Downloads the TinyStories train/validation corpora, trains a BPE
    tokenizer if one isn't already present in tokenizer_dir, encodes both
    corpora to token .bin files, and writes a meta.json with dataset stats.
    A no-op if tokenizer_dir already has a trained tokenizer.

    Args:
        raw_dir: Directory to store the downloaded raw text corpora.
        tokenizer_dir: Directory to load/save the BPE tokenizer.
        bin_dir: Directory to write the encoded .bin files and meta.json.
        vocab_size: Vocabulary size to use when training a new tokenizer.
    """
    train_corpus = download_corpus(f"{raw_dir}/train.txt", split="train")
    val_corpus = download_corpus(f"{raw_dir}/valid.txt", split="validation")

    tokenizer_dir_path = Path(tokenizer_dir)
    if not (tokenizer_dir_path / "vocab.json").exists():
        train_bpe(
            train_corpus,
            vocab_size=vocab_size,
            out_dir=str(tokenizer_dir_path)
        )
        tokenizer = Tokenizer(str(tokenizer_dir_path))
        n_train = encode_corpus_to_bin(train_corpus, tokenizer, f"{bin_dir}/train.bin")
        n_val = encode_corpus_to_bin(val_corpus, tokenizer, f"{bin_dir}/val.bin")

        meta = {
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_dir": str(tokenizer_dir_path),
            "dtype": "uint16" if tokenizer.get_vocab_size() < 65536 else "uint32",
            "train_tokens": n_train,
            "val_tokens": n_val,
        }
        Path(bin_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{bin_dir}/meta.json", "w") as file:
            json.dump(meta, file, indent=2)

        print(f"train: {n_train:,} token  -  val: {n_val:,} token  -  vocab_size: {meta['vocab_size']:,}")
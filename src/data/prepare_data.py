import json
from pathlib import Path
from typing import Iterator

import numpy as np
from datasets import load_dataset

from src.data.token_dtype import dtype_for_vocab, dtype_name_for_vocab
from src.data.tokenizer import Tokenizer
from src.data.train_tokenizer import train_bpe

def download_corpus(out_path: str, split: str) -> str:
    """Download a TinyStories split and write it to disk as one story per line.

    Fetches the requested split from the Hugging Face ``roneneldan/TinyStories``
    dataset and writes a plain-text corpus file with one story per line,
    stripping internal newlines and skipping empty stories. The download is
    skipped when ``out_path`` already exists, so re-running the pipeline does
    not re-fetch the corpus over the network.

    Args:
        out_path: Destination path for the plain-text corpus file.
        split: Dataset split to download, e.g. ``"train"`` or ``"validation"``.

    Returns:
        str: The path to the corpus file (identical to ``out_path``), so the
        caller can chain this call into downstream steps without tracking the
        normalized path separately.

    Raises:
        FileNotFoundError: If the ``datasets`` cache cannot be resolved or the
            parent directory of ``out_path`` cannot be created.
        OSError: If the corpus file cannot be written to disk.
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

    Reads the corpus lazily line by line and yields fixed-size batches of
    cleaned lines. This deliberately keeps the batching policy (skipping empty
    lines, including the final partial batch) separate from the encoding logic
    in :func:`encode_corpus_to_bin`, so either concern can evolve
    independently.

    Args:
        corpus_path: Path to a text file with one story per line.
        batch_lines: Target number of lines per batch. The final batch may be
            smaller than this.

    Yields:
        list[str]: Batches of up to ``batch_lines`` non-empty, newline-stripped
        lines, in original file order.

    Raises:
        FileNotFoundError: If ``corpus_path`` does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
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

    Encodes each line into token ids, appends the ``<eos>`` token id as a
    sequence terminator, and appends the result to ``bin_path`` as raw binary.
    The element dtype is chosen from the vocabulary size (``uint16`` when it
    fits, ``uint32`` otherwise) to keep the file as compact as possible without
    losing ids.

    Each batch is encoded, converted, and written before the next one is read,
    so peak memory stays proportional to ``batch_lines`` rather than to the
    size of the corpus. This is what lets the function encode a corpus larger
    than RAM, which is the whole point of the lazy batching it reads from and
    of the memory-mapped format it writes to.

    Args:
        corpus_path: Path to a text file with one story per line.
        tokenizer: Trained :class:`Tokenizer` used to encode lines into token
            ids. Must contain an ``<eos>`` token.
        bin_path: Destination path for the output ``.bin`` file. Overwritten
            if it already exists.
        batch_lines: Number of lines encoded per batching pass. A larger value
            amortizes encoding overhead at the cost of peak memory usage. It
            does not affect the bytes written.

    Returns:
        int: The total number of tokens written to the file, including all
        ``<eos>`` terminators. A corpus with no usable lines writes an empty
        file and returns ``0``.

    Raises:
        KeyError: If the tokenizer has no ``<eos>`` token.
        FileNotFoundError: If ``corpus_path`` does not exist.
        OSError: If ``bin_path``'s parent directory cannot be created or the
            file cannot be written.
    """
    dtype = dtype_for_vocab(tokenizer.get_vocab_size())
    eos_id = tokenizer.token_to_id("<eos>")

    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    with open(bin_path, "wb") as bin_file:
        for batch in iter_line_batches(corpus_path, batch_lines):
            batch_ids: list[int] = []
            for ids in tokenizer.encode_batch(batch):
                batch_ids.extend(ids)
                batch_ids.append(eos_id)

            if not batch_ids:
                continue

            np.array(batch_ids, dtype=dtype).tofile(bin_file)
            total_tokens += len(batch_ids)

    return total_tokens

def prepare_data(
    raw_dir: str = "data/raw",
    tokenizer_dir: str = "data/tokenizer",
    bin_dir: str = "data/processed",
    vocab_size: int = 8000
) -> None:
    """Run the full data prep pipeline: download, train tokenizer, encode to .bin.

    Orchestrates the complete offline data pipeline end to end:

    1. Downloads the TinyStories ``train`` and ``validation`` corpora into
       ``raw_dir`` (skipping existing files).
    2. Trains a BPE tokenizer on the training corpus **unless** a
       ``vocab.json`` is already present in ``tokenizer_dir``, which makes
       re-runs idempotent with respect to tokenizer training.
    3. Encodes both corpora into flat token ``.bin`` files under ``bin_dir``.
    4. Writes ``meta.json`` recording the vocabulary size, tokenizer location,
       dtype, and total token counts, and prints a summary line to stdout.

    Args:
        raw_dir: Directory to store the downloaded raw text corpora.
        tokenizer_dir: Directory to load an existing BPE tokenizer from, or to
            save a newly trained one into.
        bin_dir: Directory to write the encoded ``.bin`` files and
            ``meta.json``.
        vocab_size: Vocabulary size to use when a new tokenizer must be
            trained. Ignored if a tokenizer already exists in
            ``tokenizer_dir``.

    Returns:
        None: Artifacts are persisted to disk under ``tokenizer_dir`` and
        ``bin_dir``; nothing is returned to the caller.

    Raises:
        FileNotFoundError: If a corpus file cannot be downloaded or written.
        OSError: If any output directory or file cannot be created.
        KeyError: If the trained tokenizer is missing the ``<eos>`` token used
            to terminate sequences.
    """
    raw_path = Path(raw_dir)
    tokenizer_path = Path(tokenizer_dir)
    bin_path = Path(bin_dir)
    bin_path.mkdir(parents=True, exist_ok=True)

    train_corpus = download_corpus(str(raw_path / "train.txt"), split="train")
    val_corpus = download_corpus(str(raw_path / "valid.txt"), split="validation")

    # Training the tokenizer is the one step worth skipping on a re-run; the
    # encoding below always runs, since bin_dir may well be a fresh directory.
    if not (tokenizer_path / "vocab.json").exists():
        train_bpe(
            train_corpus,
            vocab_size=vocab_size,
            out_dir=str(tokenizer_path)
        )

    tokenizer = Tokenizer(str(tokenizer_path))
    n_train = encode_corpus_to_bin(train_corpus, tokenizer, str(bin_path / "train.bin"))
    n_val = encode_corpus_to_bin(val_corpus, tokenizer, str(bin_path / "val.bin"))

    actual_vocab_size = tokenizer.get_vocab_size()
    meta = {
        "vocab_size": actual_vocab_size,
        "tokenizer_dir": str(tokenizer_path),
        "dtype": dtype_name_for_vocab(actual_vocab_size),
        "train_tokens": n_train,
        "val_tokens": n_val,
    }
    with open(bin_path / "meta.json", "w") as file:
        json.dump(meta, file, indent=2)

    print(f"train: {n_train:,} token  -  val: {n_val:,} token  -  vocab_size: {meta['vocab_size']:,}")
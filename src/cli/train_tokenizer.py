import argparse

from src.data.train_tokenizer import train_bpe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface for BPE tokenizer training.

    Builds an :class:`argparse.ArgumentParser` exposing the parameters required
    to train a byte-pair encoding (BPE) tokenizer from a raw text corpus, and
    parses them into a lightweight :class:`argparse.Namespace`.

    Args:
        argv: Optional sequence of raw command-line arguments to parse.
            When ``None``, :func:`argparse.ArgumentParser.parse_args` falls back
            to ``sys.argv[1:]``. Providing this parameter explicitly makes the
            function trivially unit-testable.

    Returns:
        argparse.Namespace: Populated namespace exposing the following
        attributes:

        - ``corpus_path`` (str): Positional path to the raw text file used to
          learn the merge rules. This argument is required.
        - ``vocab_size`` (int): Target vocabulary size for the learned BPE
          model (default ``8000``).
        - ``out_dir`` (str): Directory where the trained tokenizer artifacts
          (``vocab.json`` and ``merges.txt``) are written (default
          ``"tokenizer"``).

    Raises:
        SystemExit: On missing required arguments, ``--help``, or invalid
            arguments, consistent with standard argparse behaviour.
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a text corpus.")
    parser.add_argument("corpus_path", help="Raw text file to train the BPE tokenizer on.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--out-dir", default="tokenizer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: train a BPE tokenizer and persist it to disk.

    Parses the command-line arguments and delegates to
    :func:`src.data.train_tokenizer.train_bpe`, which learns the byte-pair
    merge rules from the raw corpus and saves the resulting model. After a
    successful run, a summary of the produced artifacts is written to stdout.

    Args:
        argv: Optional sequence of raw command-line arguments forwarded to
            :func:`parse_args`. When ``None``, ``sys.argv`` is used.

    Returns:
        None: Tokenizer artifacts are written to ``out_dir`` (``vocab.json``
        and ``merges.txt``); nothing is returned to the caller.

    Raises:
        FileNotFoundError: If ``corpus_path`` does not exist or cannot be read.
        ValueError: If ``vocab_size`` is not a positive integer or the corpus
            is empty / too small to reach the requested vocabulary.
        OSError: If ``out_dir`` cannot be created or the artifacts cannot be
            written.
    """
    args = parse_args(argv)
    train_bpe(args.corpus_path, args.vocab_size, args.out_dir)
    print(f"Tokenizer trained: {args.out_dir}/vocab.json, {args.out_dir}/merges.txt "
          f"(vocab_size={args.vocab_size})")


if __name__ == "__main__":
    main()

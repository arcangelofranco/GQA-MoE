import argparse

from src.data.train_tokenizer import train_bpe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line arguments for BPE tokenizer training.

    Args:
        argv: List of arguments to parse. If None, sys.argv is used
            (argparse's default behavior).

    Returns:
        Namespace containing the parsed arguments (corpus_path, vocab_size,
        out_dir).
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a text corpus.")
    parser.add_argument("corpus_path", help="Raw text file to train the BPE tokenizer on.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--out-dir", default="tokenizer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: trains a BPE tokenizer and saves it to disk.

    Args:
        argv: List of command-line arguments to parse. If None,
            sys.argv is used.
    """
    args = parse_args(argv)
    train_bpe(args.corpus_path, args.vocab_size, args.out_dir)
    print(f"Tokenizer trained: {args.out_dir}/vocab.json, {args.out_dir}/merges.txt "
          f"(vocab_size={args.vocab_size})")


if __name__ == "__main__":
    main()

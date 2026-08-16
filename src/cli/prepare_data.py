import argparse

from src.data.prepare_data import prepare_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line arguments for data preparation.

    Args:
        argv: List of arguments to parse. If None, sys.argv is used
            (argparse's default behavior).

    Returns:
        Namespace containing the parsed arguments (vocab_size, raw_dir,
        tokenizer_dir, bin_dir).
    """
    parser = argparse.ArgumentParser(description="Prepare the data: download, tokenizer, encoding.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--bin-dir", default="data/processed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: runs the full data preparation pipeline.

    Args:
        argv: List of command-line arguments to parse. If None,
            sys.argv is used.
    """
    args = parse_args(argv)
    prepare_data(args.raw_dir, args.tokenizer_dir, args.bin_dir, args.vocab_size)


if __name__ == "__main__":
    main()

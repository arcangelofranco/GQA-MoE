import argparse

from src.data.prepare_data import prepare_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface for data preparation.

    Builds an :class:`argparse.ArgumentParser` exposing the tunable knobs of the
    data pipeline and parses the given arguments into a lightweight
    :class:`argparse.Namespace`. Defaults are chosen so that a plain invocation
    with no flags produces a sensible, reproducible pipeline run.

    Args:
        argv: Optional sequence of raw command-line arguments to parse.
            When ``None``, :func:`argparse.ArgumentParser.parse_args` falls back
            to ``sys.argv[1:]``. Providing this parameter explicitly makes the
            function trivially unit-testable.

    Returns:
        argparse.Namespace: Populated namespace exposing the following
        attributes:

        - ``vocab_size`` (int): Target vocabulary size for the trained
          tokenizer (default ``8000``).
        - ``raw_dir`` (str): Directory holding the raw corpus download.
        - ``tokenizer_dir`` (str): Directory where the trained tokenizer is
          persisted.
        - ``bin_dir`` (str): Directory where the pre-encoded binary shards are
          written for fast training-time loading.

    Raises:
        SystemExit: On ``--help`` or invalid arguments, consistent with
            standard argparse behaviour.
    """
    parser = argparse.ArgumentParser(description="Prepare the data: download, tokenizer, encoding.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--bin-dir", default="data/processed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the full data preparation pipeline.

    Parses the command-line arguments and delegates to
    :func:`src.data.prepare_data.prepare_data`, which handles corpus
    acquisition, tokenizer training, and corpus encoding into training-ready
    binary shards. The pipeline is fully deterministic for a given set of
    arguments, so repeated invocations with the same flags produce equivalent
    artifacts.

    Args:
        argv: Optional sequence of raw command-line arguments forwarded to
            :func:`parse_args`. When ``None``, ``sys.argv`` is used.

    Returns:
        None: Artifacts are persisted to disk (tokenizer directory and binary
        shard directory); nothing is returned to the caller.

    Raises:
        FileNotFoundError: If a required source file or directory cannot be
            located during the pipeline.
        RuntimeError: If the tokenizer training or encoding step fails (e.g.
            empty corpus or insufficient samples for the requested vocabulary).
        OSError: If any of the output directories cannot be created or written.
    """
    args = parse_args(argv)
    prepare_data(args.raw_dir, args.tokenizer_dir, args.bin_dir, args.vocab_size)


if __name__ == "__main__":
    main()

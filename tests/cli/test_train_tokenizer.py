from pathlib import Path

import pytest

from src.cli.train_tokenizer import main
from src.data.tokenizer import Tokenizer

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
Once upon a time, there was a small dog named Spot.
Spot loved to run and jump in the sunny park every day.
The sun was warm and the grass was green and soft.
""" * 20


def test_cli_train_tokenizer_end_to_end_smoke(tmp_path: Path) -> None:
    """Verifies the end-to-end flow of the train-tokenizer CLI.

    Confirms that a BPE tokenizer is trained with the requested vocabulary
    size and that both serialized artifacts (``vocab.json`` and ``merges.txt``)
    are written to the output directory.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    out_dir = tmp_path / "tokenizer"

    main([str(corpus_path), "--vocab-size", "300", "--out-dir", str(out_dir)])

    tokenizer = Tokenizer(str(out_dir))
    vocab_size = tokenizer.get_vocab_size()
    print(f"[smoke] vocab.json exists={(out_dir / 'vocab.json').exists()} merges.txt exists={(out_dir / 'merges.txt').exists()} vocab_size={vocab_size}")
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "merges.txt").exists()
    assert vocab_size == 300


def test_cli_train_tokenizer_requires_corpus_path() -> None:
    """Verifies that the CLI fails when the required ``corpus_path`` argument is omitted.

    Guards the required positional argument so a missing value aborts early
    through ``argparse`` instead of failing later with a confusing error.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["--vocab-size", "300"])
    print(f"[missing-arg] SystemExit code={exc_info.value.code}")

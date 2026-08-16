import json
from pathlib import Path

from src.cli.prepare_data import main

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
""" * 20


def _write_local_raw_corpus(raw_dir: Path) -> None:
    """Writes train.txt/valid.txt into raw_dir so download_corpus is a no-op.

    Args:
        raw_dir: Directory to write the two corpus files into.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "train.txt").write_text(CORPUS_TEXT)
    (raw_dir / "valid.txt").write_text("Tom liked the dog very much indeed.\n" * 20)


def test_cli_prepare_data_end_to_end_smoke(tmp_path: Path) -> None:
    """Checks that the prepare-data CLI runs end-to-end and writes tokenizer, bins, and meta.json.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    raw_dir = tmp_path / "raw"
    tokenizer_dir = tmp_path / "tokenizer"
    bin_dir = tmp_path / "bin"
    _write_local_raw_corpus(raw_dir)

    main([
        "--raw-dir", str(raw_dir),
        "--tokenizer-dir", str(tokenizer_dir),
        "--bin-dir", str(bin_dir),
        "--vocab-size", "300",
    ])

    with open(bin_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"[smoke] meta={meta}")
    assert meta["vocab_size"] == 300
    assert meta["dtype"] == "uint16"
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0
    assert (tokenizer_dir / "vocab.json").exists()
    assert (bin_dir / "train.bin").exists()
    assert (bin_dir / "val.bin").exists()


def test_cli_prepare_data_does_not_retrain_tokenizer_if_present(tmp_path: Path) -> None:
    """Checks that a second CLI invocation reuses an existing tokenizer instead of retraining.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    raw_dir = tmp_path / "raw"
    tokenizer_dir = tmp_path / "tokenizer"
    _write_local_raw_corpus(raw_dir)

    main([
        "--raw-dir", str(raw_dir),
        "--tokenizer-dir", str(tokenizer_dir),
        "--bin-dir", str(tmp_path / "bin1"),
        "--vocab-size", "300",
    ])
    vocab_mtime_1 = (tokenizer_dir / "vocab.json").stat().st_mtime_ns

    main([
        "--raw-dir", str(raw_dir),
        "--tokenizer-dir", str(tokenizer_dir),
        "--bin-dir", str(tmp_path / "bin2"),
        "--vocab-size", "300",
    ])
    vocab_mtime_2 = (tokenizer_dir / "vocab.json").stat().st_mtime_ns

    print(f"[no-retrain] vocab_mtime_1={vocab_mtime_1} vocab_mtime_2={vocab_mtime_2}")
    assert vocab_mtime_1 == vocab_mtime_2  # not retrained the second time

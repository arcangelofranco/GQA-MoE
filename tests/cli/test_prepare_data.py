import json
from pathlib import Path

from src.cli.prepare_data import main

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
""" * 20


def _write_local_raw_corpus(raw_dir: Path) -> None:
    """Writes local corpus files so the download step is skipped.

    Places ``train.txt`` and ``valid.txt`` into ``raw_dir``, allowing the
    prepare-data CLI to run entirely offline with a deterministic corpus.

    Args:
        raw_dir: Directory to write the two corpus files into.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "train.txt").write_text(CORPUS_TEXT)
    (raw_dir / "valid.txt").write_text("Tom liked the dog very much indeed.\n" * 20)


def test_cli_prepare_data_end_to_end_smoke(tmp_path: Path) -> None:
    """Verifies the prepare-data pipeline produces all expected artifacts.

    Confirms the CLI trains a tokenizer and emits the binary training and
    validation datasets together with a ``meta.json`` describing the
    vocabulary size and token counts.

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
    """Verifies the CLI reuses an existing tokenizer instead of retraining it.

    Re-running the command must not regenerate the tokenizer artifacts, so the
    vocab file keeps its original modification time. This prevents wasted work
    on large corpora.

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

    # Reusing the tokenizer must not skip encoding: the second run still owes
    # the caller a complete dataset in its own bin dir.
    with open(tmp_path / "bin2" / "meta.json") as f:
        meta = json.load(f)
    print(f"[no-retrain] second_run_meta={meta}")
    assert (tmp_path / "bin2" / "train.bin").exists()
    assert (tmp_path / "bin2" / "val.bin").exists()
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0

from pathlib import Path
from typing import Callable

import pytest

from src.cli.generate import main
from src.config import ModelConfig, RunConfig, RuntimeConfig, TrainConfig
from src.data.dataset import BinDataset
from src.data.train_tokenizer import train_bpe
from src.runtime.trainer import Trainer

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
""" * 20


def _train_tiny_tokenizer(tmp_path: Path, vocab_size: int = 300) -> Path:
    """Trains a small BPE tokenizer on the toy corpus for test fixtures.

    Wraps ``train_bpe`` to provide tokenizers of arbitrary size so each test
    controls the vocabulary independently.

    Args:
        tmp_path: Temporary directory to write the corpus and tokenizer into.
        vocab_size: Target vocabulary size for the trained tokenizer.

    Returns:
        Path to the directory containing the trained tokenizer.
    """
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    tok_dir = tmp_path / "tokenizer"
    train_bpe(str(corpus_path), vocab_size=vocab_size, out_dir=str(tok_dir))
    return tok_dir


def _save_tiny_checkpoint(
    tmp_path: Path,
    vocab_size: int,
    write_synthetic_bin_dataset: Callable[..., Path],
    tokenizer_dir: Path | None = None,
) -> tuple[Path, ModelConfig]:
    """Saves a minimal checkpoint through the real Trainer for test fixtures.

    Writing the file with ``Trainer.save_checkpoint`` keeps the generate tests
    on the same checkpoint format the training run produces.

    Args:
        tmp_path: Temporary directory to write the dataset and checkpoint into.
        vocab_size: Vocabulary size for the model config.
        write_synthetic_bin_dataset: Fixture factory writing the dataset the
            Trainer is built on.
        tokenizer_dir: Tokenizer directory recorded in the dataset, and so
            in the checkpoint.

    Returns:
        Tuple of (checkpoint path, model config used to build it).
    """
    model_cfg = ModelConfig(
        vocab_size=vocab_size, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
        expert_intermediate=16, shared_intermediate=32, n_experts=4, top_k=2,
        max_seq_len=64,
    )
    cfg = RunConfig(
        model_cfg,
        TrainConfig(batch_size=2, block_size=16, target_tokens=2 * 16 * 50, warmup_steps=2),
        RuntimeConfig(device="cpu"),
    )
    data_dir = write_synthetic_bin_dataset(
        tmp_path / "data", vocab_size=vocab_size,
        tokenizer_dir=str(tokenizer_dir) if tokenizer_dir else None,
    )
    trainer = Trainer(cfg, BinDataset(str(data_dir), seed=0), tmp_path / "run")
    ckpt_path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    return ckpt_path, model_cfg


def test_cli_generate_end_to_end_smoke(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies the generate CLI runs end-to-end and returns text.

    Confirms the output is a string that continues from the given prompt,
    exercising checkpoint loading, tokenization, and sampling.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    vocab_size = 300
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=vocab_size)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size, write_synthetic_bin_dataset)

    text = main([
        "--checkpoint", str(ckpt_path),
        "--tokenizer-dir", str(tok_dir),
        "--prompt", "Once upon a time,",
        "--max-new-tokens", "15",
        "--temperature", "0.8",
        "--top-k", "20",
        "--device", "cpu",
    ])
    print(f"[smoke] generated text={text!r}")
    assert isinstance(text, str)
    assert text.startswith("Once upon a time,")


def test_cli_generate_greedy_is_deterministic(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies greedy decoding is deterministic across runs.

    With ``temperature=0.0`` the same prompt and checkpoint must always yield
    identical text, guarding against sampling nondeterminism or state leakage.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    vocab_size = 300
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=vocab_size)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size, write_synthetic_bin_dataset)

    args = [
        "--checkpoint", str(ckpt_path),
        "--tokenizer-dir", str(tok_dir),
        "--prompt", "Tom liked to",
        "--max-new-tokens", "10",
        "--temperature", "0.0",
        "--device", "cpu",
    ]
    first = main(args)
    second = main(args)
    print(f"[determinism] first={first!r} second={second!r} equal={first == second}")
    assert first == second


def test_cli_generate_rejects_vocab_size_mismatch(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies that a checkpoint/tokenizer vocab size mismatch raises ``ValueError``.

    Generating with a tokenizer that cannot represent the model vocabulary
    would produce invalid output, so the CLI must reject it upfront.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=300)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, 301, write_synthetic_bin_dataset)  # deliberately mismatched

    with pytest.raises(ValueError) as exc_info:
        main([
            "--checkpoint", str(ckpt_path),
            "--tokenizer-dir", str(tok_dir),
            "--device", "cpu",
        ])
    print(f"[mismatch] tokenizer vocab_size=300 checkpoint vocab_size=301 -> raised={exc_info.value!r}")


def test_cli_generate_defaults_to_the_tokenizer_recorded_in_the_checkpoint(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies that without ``--tokenizer-dir`` the CLI uses the training tokenizer.

    The checkpoint records the tokenizer the training data was encoded with,
    so generation picks the matching one without being told.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    vocab_size = 300
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=vocab_size)
    ckpt_path, _ = _save_tiny_checkpoint(
        tmp_path, vocab_size, write_synthetic_bin_dataset, tokenizer_dir=tok_dir
    )

    text = main([
        "--checkpoint", str(ckpt_path),
        "--prompt", "Tom liked to",
        "--max-new-tokens", "5",
        "--temperature", "0.0",
        "--device", "cpu",
    ])
    print(f"[recorded-tokenizer] generated text={text!r}")
    assert text.startswith("Tom liked to")

from pathlib import Path

import torch
import pytest

from src.cli.generate import main
from src.config import ModelConfig
from src.data.train_tokenizer import train_bpe
from src.model.transformer import Transformer

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
""" * 20


def _train_tiny_tokenizer(tmp_path: Path, vocab_size: int = 300) -> Path:
    """Trains a tiny BPE tokenizer on the toy corpus for test fixtures.

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


def _save_tiny_checkpoint(tmp_path: Path, vocab_size: int) -> tuple[Path, ModelConfig]:
    """Builds and saves a minimal Transformer checkpoint for test fixtures.

    Args:
        tmp_path: Temporary directory to write the checkpoint into.
        vocab_size: Vocabulary size for the model config.

    Returns:
        Tuple of (checkpoint path, model config used to build it).
    """
    model_cfg = ModelConfig(
        vocab_size=vocab_size, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
        expert_intermediate=16, shared_intermediate=32, n_experts=4, top_k=2,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    model = Transformer(model_cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg.to_dict()}, ckpt_path)
    return ckpt_path, model_cfg


def test_cli_generate_end_to_end_smoke(tmp_path: Path) -> None:
    """Checks that the generate CLI runs end-to-end and echoes the prompt prefix.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    vocab_size = 300
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=vocab_size)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size)

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


def test_cli_generate_greedy_is_deterministic(tmp_path: Path) -> None:
    """Checks that greedy decoding (temperature=0.0) produces identical output across runs.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    vocab_size = 300
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=vocab_size)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size)

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


def test_cli_generate_rejects_vocab_size_mismatch(tmp_path: Path) -> None:
    """Checks that a checkpoint/tokenizer vocab size mismatch raises ValueError.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    tok_dir = _train_tiny_tokenizer(tmp_path, vocab_size=300)
    ckpt_path, _ = _save_tiny_checkpoint(tmp_path, vocab_size=301)  # deliberately mismatched

    with pytest.raises(ValueError) as exc_info:
        main([
            "--checkpoint", str(ckpt_path),
            "--tokenizer-dir", str(tok_dir),
            "--device", "cpu",
        ])
    print(f"[mismatch] tokenizer vocab_size=300 checkpoint vocab_size=301 -> raised={exc_info.value!r}")

from pathlib import Path
from typing import Callable

import pytest
import torch

from src.cli.train import main


def _write_tiny_yaml(path: Path, vocab_size: int) -> None:
    """Creates a minimal training config YAML for the CLI smoke tests.

    The generated config keeps the model and training settings small so the
    end-to-end tests run quickly on CPU while still exercising real config
    parsing and checkpointing.

    Args:
        path: File path to write the YAML config to.
        vocab_size: Vocabulary size to embed in the model config section.
    """
    path.write_text(
        f"""
model:
  vocab_size: {vocab_size}
  n_layers: 2
  n_heads: 2
  n_kv_heads: 1
  head_dim: 8
  expert_intermediate: 16
  shared_intermediate: 32
  n_experts: 2
  top_k: 1
  max_seq_len: 32

train:
  batch_size: 4
  block_size: 16
  target_tokens: 1280
  warmup_steps: 2
  eval_interval: 5
  eval_iters: 2

runtime:
  device: cpu
  checkpoint_every: 5
"""
    )


def test_cli_train_end_to_end_smoke(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies the train CLI trains to the configured maximum step and checkpoints.

    Confirms that a full run writes ``final.pt`` and ``train.log`` and that the
    checkpoint records the expected final step, guarding against premature
    termination or checkpoint corruption.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    config_path = tmp_path / "tiny.yaml"

    vocab_size = 40
    write_synthetic_bin_dataset(data_dir, vocab_size=vocab_size)
    _write_tiny_yaml(config_path, vocab_size=vocab_size)

    trainer = main([
        "--config", str(config_path),
        "--data-dir", str(data_dir),
        "--run-dir", str(run_dir),
    ])

    print(
        f"[smoke] step={trainer.step} max_steps={trainer.cfg.train.max_steps} -> "
        f"final.pt exists={(run_dir / 'final.pt').exists()} "
        f"train.log exists={(run_dir / 'train.log').exists()}"
    )
    assert trainer.step == trainer.cfg.train.max_steps
    assert (run_dir / "final.pt").exists()
    assert (run_dir / "train.log").exists()

    ckpt = torch.load(run_dir / "final.pt", map_location="cpu")
    print(f"[smoke] checkpoint step={ckpt['step']} expected={trainer.cfg.train.max_steps}")
    assert ckpt["step"] == trainer.cfg.train.max_steps


def test_cli_train_rejects_vocab_size_mismatch(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies that a mismatch between dataset and config vocab size raises ``ValueError``.

    Training on misaligned vocabularies would silently produce garbage, so the
    CLI must reject the run at startup instead of training on bad data.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = tmp_path / "data"
    config_path = tmp_path / "tiny.yaml"
    write_synthetic_bin_dataset(data_dir, vocab_size=40)
    _write_tiny_yaml(config_path, vocab_size=41)  # deliberately mismatched

    with pytest.raises(ValueError) as exc_info:
        main([
            "--config", str(config_path),
            "--data-dir", str(data_dir),
            "--run-dir", str(tmp_path / "run"),
        ])
    print(f"[mismatch] dataset vocab_size=40 config vocab_size=41 -> raised={exc_info.value!r}")


def test_cli_train_resume(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Verifies that resuming from a checkpoint already at ``max_steps`` is idempotent.

    The CLI must not take further training steps when the resumed checkpoint is
    already complete, preventing unnecessary work and wasted compute.

    Args:
        tmp_path: Pytest-provided temporary directory.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    config_path = tmp_path / "tiny.yaml"
    vocab_size = 40
    write_synthetic_bin_dataset(data_dir, vocab_size=vocab_size)
    _write_tiny_yaml(config_path, vocab_size=vocab_size)

    first = main(["--config", str(config_path), "--data-dir", str(data_dir), "--run-dir", str(run_dir)])
    final_ckpt = run_dir / "final.pt"
    print(f"[resume] first run step={first.step} max_steps={first.cfg.train.max_steps} -> final.pt exists={final_ckpt.exists()}")
    assert final_ckpt.exists()

    resumed = main([
        "--config", str(config_path),
        "--data-dir", str(data_dir),
        "--run-dir", str(run_dir),
        "--resume", str(final_ckpt),
    ])
    print(f"[resume] resumed step={resumed.step} expected={first.cfg.train.max_steps}")
    # already at max_steps at resume time: training must not take further steps.
    assert resumed.step == first.cfg.train.max_steps

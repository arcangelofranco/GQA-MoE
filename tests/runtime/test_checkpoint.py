from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest
import torch

from src.config import ModelConfig, RunConfig, RuntimeConfig, TrainConfig
from src.data.dataset import BinDataset
from src.runtime import checkpoint
from src.runtime.trainer import Trainer

N_STEPS, SPLIT_AT = 8, 3


def _cfg() -> RunConfig:
    """Builds a tiny CPU run configuration for checkpoint tests.

    Returns:
        The constructed `RunConfig`.
    """
    return RunConfig(
        ModelConfig(
            vocab_size=50, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
            expert_intermediate=16, shared_intermediate=32, n_experts=4, top_k=2,
            max_seq_len=32,
        ),
        TrainConfig(batch_size=4, block_size=16, target_tokens=4 * 16 * 20,
                    warmup_steps=2, max_lr=1e-3),
        RuntimeConfig(device="cpu"),
    )


def _trainer(cfg: RunConfig, data_dir: Path, run_dir: Path) -> Trainer:
    """Builds a Trainer on a freshly opened, seeded dataset, as the training CLI does.

    Args:
        cfg: The run configuration.
        data_dir: Directory of the synthetic dataset.
        run_dir: Directory for the run's outputs.

    Returns:
        The constructed `Trainer`.
    """
    return Trainer(cfg, BinDataset(str(data_dir), seed=cfg.train.seed), run_dir)


def test_resume_matches_uninterrupted_run(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that save, resume and continue equals training without interruption.

    The resumed run must see the same batches, take the same steps on the same
    LR curve and end with bit-identical weights, which requires the checkpoint
    to restore the dataset sampling RNG as well as the model and schedule.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    cfg = _cfg()

    full = _trainer(cfg, data_dir, tmp_path / "full")
    full_losses = [full.train_step()[0] for _ in range(N_STEPS)]

    first = _trainer(cfg, data_dir, tmp_path / "split")
    split_losses = [first.train_step()[0] for _ in range(SPLIT_AT)]
    ckpt_path = first.save_checkpoint()

    resumed = _trainer(cfg, data_dir, tmp_path / "split")
    assert resumed.load_checkpoint(ckpt_path) == SPLIT_AT
    split_losses += [resumed.train_step()[0] for _ in range(N_STEPS - SPLIT_AT)]

    print(f"[exact-resume] full={full_losses} split={split_losses}")
    assert split_losses == full_losses
    for (name, p_full), (_, p_resumed) in zip(
        full.model.named_parameters(), resumed.model.named_parameters()
    ):
        assert torch.equal(p_full, p_resumed), name


def test_trainer_checkpoint_loads_for_inference(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that a checkpoint written by the Trainer rebuilds the model for generation.

    The checkpoint must also carry the tokenizer directory recorded in the
    dataset, so generation can pick the tokenizer the model was trained with.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data", tokenizer_dir="tok")
    cfg = _cfg()
    trainer = _trainer(cfg, data_dir, tmp_path / "run")
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint()

    model, model_cfg, tokenizer_dir = checkpoint.load_for_inference(ckpt_path, "cpu")
    assert tokenizer_dir == "tok"
    print(f"[inference] model_cfg == cfg.model -> {model_cfg == cfg.model}, training={model.training}")
    assert model_cfg == cfg.model
    assert not model.training
    for (name, p_saved), (_, p_loaded) in zip(
        trainer.model.named_parameters(), model.named_parameters()
    ):
        assert torch.equal(p_saved, p_loaded), name


def test_resume_with_different_config_fails_and_lists_fields(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that resuming under a changed model/train config raises, while runtime may change.

    A different ``target_tokens`` would silently move the LR curve, so it must
    be rejected with the differing field named; a different ``log_every``
    does not change what the run computes and is accepted.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    cfg = _cfg()
    trainer = _trainer(cfg, data_dir, tmp_path / "run")
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint()

    changed = replace(cfg, train=replace(cfg.train, target_tokens=cfg.train.target_tokens * 2))
    with pytest.raises(ValueError, match="train.target_tokens") as exc_info:
        _trainer(changed, data_dir, tmp_path / "run").load_checkpoint(ckpt_path)
    print(f"[config-mismatch] raised={exc_info.value!r}")

    runtime_only = replace(cfg, runtime=replace(cfg.runtime, log_every=1))
    assert _trainer(runtime_only, data_dir, tmp_path / "run").load_checkpoint(ckpt_path) == 1


def test_version_1_checkpoint_refuses_resume_but_loads_for_inference(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that a checkpoint in the previous format cannot resume but still generates.

    Version 1 files carry no dataset RNG state, so resuming them could not
    continue on the right batches; their weights stay usable.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    cfg = _cfg()
    trainer = _trainer(cfg, data_dir, tmp_path / "run")
    trainer.train_step()

    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "schedule": trainer.schedule.state_dict(),
            "step": trainer.step,
            "rng_state": torch.get_rng_state(),
            "model_cfg": cfg.model.to_dict(),
            "train_cfg": cfg.train.to_dict(),
        },
        legacy_path,
    )

    with pytest.raises(ValueError, match="version 1") as exc_info:
        trainer.load_checkpoint(legacy_path)
    print(f"[legacy-checkpoint] raised={exc_info.value!r}")

    model, model_cfg, tokenizer_dir = checkpoint.load_for_inference(legacy_path, "cpu")
    assert model_cfg == cfg.model
    assert tokenizer_dir is None

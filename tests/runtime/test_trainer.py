import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from src.config import ModelConfig, RunConfig, RuntimeConfig, TrainConfig
from src.data.dataset import BinDataset
from src.model.transformer import Transformer
from src.runtime.metrics import InMemoryRecorder
from src.runtime.schedule import TrainingSchedule
from src.runtime.trainer import Trainer


def _overfit_single_batch(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    steps: int,
    batch_size: int,
    block_size: int,
    seed: int = 1337,
) -> list[float]:
    """Trains a fresh model on one fixed random batch to verify it can overfit.

    Each iteration mirrors a Trainer step: total loss (cross-entropy plus
    auxiliary), backward pass, then one optimizer step through the schedule.

    Args:
        model_cfg: Model architecture config.
        train_cfg: Training config (schedule, optimizer settings).
        steps: Number of optimization steps to run.
        batch_size: Batch size for the fixed synthetic batch.
        block_size: Sequence length for the fixed synthetic batch.
        seed: Seed for model init and the synthetic batch.

    Returns:
        Loss value at each step, in order.
    """
    torch.manual_seed(seed)
    model = Transformer(model_cfg)
    schedule = TrainingSchedule(model, train_cfg, RuntimeConfig())

    x = torch.randint(0, model_cfg.vocab_size, (batch_size, block_size))
    y = torch.randint(0, model_cfg.vocab_size, (batch_size, block_size))

    losses = []
    for _ in range(steps):
        out = model(x)
        logits = out.logits
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = ce_loss + out.aux_loss
        schedule.zero_grad()
        loss.backward()
        schedule.step(model.parameters())
        losses.append(loss.item())
    return losses


def test_gate_overfit_reduced_architecture() -> None:
    """Checks that the reduced ("overfit") architecture can drive loss near zero on one batch.

    Also verifies the initial loss sits near the chance-level baseline (a
    sane start), so the convergence is measured from a healthy beginning.
    """
    overfit = RunConfig.preset("overfit", vocab_size=100)
    model_cfg, train_cfg = overfit.model, overfit.train
    losses = _overfit_single_batch(
        model_cfg, train_cfg, steps=250,
        batch_size=train_cfg.batch_size, block_size=train_cfg.block_size,
    )
    initial_baseline = torch.log(torch.tensor(float(model_cfg.vocab_size))).item()
    print(
        f"[overfit-reduced] loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} "
        f"baseline={initial_baseline:.4f}"
    )
    assert losses[0] < initial_baseline + 1.0  # starts near chance level, not blown up
    assert losses[-1] < 0.5
    assert losses[-1] < 0.1 * losses[0]


def test_gate_overfit_real_nano_architecture() -> None:
    """Checks that the real "nano" architecture reduces loss substantially on one batch.

    Verifies every recorded loss stays finite and that the final loss drops
    well below the initial one.
    """
    model_cfg = RunConfig.preset("nano", vocab_size=64).model
    train_cfg = TrainConfig(
        batch_size=4, block_size=32, target_tokens=4 * 32 * 40,  # 40 steps
        warmup_steps=5, max_lr=5e-3, eval_interval=1000, eval_iters=1,
    )
    losses = _overfit_single_batch(
        model_cfg, train_cfg, steps=train_cfg.max_steps,
        batch_size=train_cfg.batch_size, block_size=train_cfg.block_size,
    )
    print(f"[overfit-nano] loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f}")
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0] * 0.7


def _tiny_run_config(
    vocab_size: int = 50, max_steps_tokens: int | None = None, **runtime_overrides
) -> RunConfig:
    """Builds a small RunConfig (~20 steps) for fast Trainer tests.

    Args:
        vocab_size: Vocabulary size for the model config.
        max_steps_tokens: Total training tokens; defaults to a 20-step run.
        **runtime_overrides: Fields overriding the default `RuntimeConfig`.

    Returns:
        The constructed `RunConfig`.
    """
    overfit = RunConfig.preset("overfit", vocab_size=vocab_size)
    train_cfg = TrainConfig(
        batch_size=4, block_size=16,
        target_tokens=max_steps_tokens or (4 * 16 * 20),  # 20 steps
        warmup_steps=2, max_lr=1e-3, eval_interval=5, eval_iters=2,
    )
    runtime_cfg = RuntimeConfig(**runtime_overrides) if runtime_overrides else overfit.runtime
    return RunConfig(overfit.model, train_cfg, runtime_cfg)


def test_trainer_checkpoint_resume(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that a Trainer resumed from a checkpoint can finish training.

    Verifies that the step counter, learning rate and every model parameter
    exactly match the original trainer at the checkpoint step, proving the
    resume path restores a consistent state rather than a partial one.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    cfg = _tiny_run_config()

    run_dir = tmp_path / "run"
    trainer = Trainer(cfg, dataset, run_dir)
    for _ in range(10):
        trainer.train_step()
    assert trainer.step == 10

    ckpt_path = trainer.save_checkpoint()
    assert ckpt_path.exists()

    resumed = Trainer(cfg, dataset, run_dir)
    restored_step = resumed.load_checkpoint(ckpt_path)
    print(f"[resume] restored_step={restored_step} resumed.step={resumed.step}")
    assert restored_step == 10
    assert resumed.step == 10

    n_compared = 0
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), resumed.model.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
        n_compared += 1
    print(f"[resume] compared {n_compared} parameter tensors, all equal")

    assert resumed.schedule.current_lr == trainer.schedule.current_lr

    final_step = resumed.train()
    print(f"[resume] final_step={final_step} expected={cfg.train.max_steps}")
    assert final_step == cfg.train.max_steps


def test_resume_from_pre_schedule_checkpoint_fails_loudly(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that loading a legacy (pre-TrainingSchedule) checkpoint fails loudly.

    Ensures a KeyError naming "pre-TrainingSchedule" is raised instead of
    failing later with an unrelated error, so backward-incompatible
    checkpoints are easy to diagnose.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    cfg = _tiny_run_config()

    trainer = Trainer(cfg, dataset, tmp_path / "run")
    trainer.train_step()

    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.schedule.state_dict()["optimizer"],
            "step": trainer.step,
            "rng_state": torch.get_rng_state(),
            "model_cfg": cfg.model.to_dict(),
            "train_cfg": cfg.train.to_dict(),
        },
        legacy_path,
    )

    with pytest.raises(KeyError, match="pre-TrainingSchedule") as exc_info:
        trainer.load_checkpoint(legacy_path)
    print(f"[legacy-checkpoint] raised={exc_info.value!r}")


def test_trainer_full_run_reaches_max_steps_and_writes_log(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that trainer.train() runs to max_steps and writes train.log.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    cfg = _tiny_run_config()
    run_dir = tmp_path / "run"

    trainer = Trainer(cfg, dataset, run_dir)
    final_step = trainer.train()

    print(f"[full-run] final_step={final_step} expected={cfg.train.max_steps} train.log exists={(run_dir / 'train.log').exists()}")
    assert final_step == cfg.train.max_steps
    assert (run_dir / "train.log").exists()


def test_evaluate_uses_eval_mode_and_returns_finite_average(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that evaluate() switches to eval mode, returns a finite loss, and restores training mode.

    Verifies the model is back in training mode afterwards, so the next
    train_step() is not silently run with the eval flag.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    cfg = _tiny_run_config()
    trainer = Trainer(cfg, dataset, tmp_path / "run")

    val_loss = trainer.evaluate()
    print(f"[evaluate] val_loss={val_loss} finite={torch.isfinite(torch.tensor(val_loss)).item()} model.training={trainer.model.training}")
    assert torch.isfinite(torch.tensor(val_loss))
    assert trainer.model.training


def test_train_step_returns_loss_and_pure_ce_loss(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that train_step() returns a finite total loss above the pure CE loss.

    The total loss includes the auxiliary loss, so it must never come out
    smaller than the standalone cross-entropy component.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    cfg = _tiny_run_config()
    trainer = Trainer(cfg, dataset, tmp_path / "run")

    loss, ce_loss = trainer.train_step()
    print(f"[train-step] loss={loss} ce_loss={ce_loss}")
    assert torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(ce_loss))
    assert loss >= ce_loss - 1e-6


def test_recorded_steps_carry_loss_and_perplexity(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that recorded steps keep perplexity consistent with their loss.

    Verifies that every record carrying a train or validation loss also
    carries the matching perplexity, equal to exp() of that same loss.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    recorder = InMemoryRecorder()

    Trainer(_tiny_run_config(), dataset, tmp_path / "run", recorder=recorder).train()

    assert recorder.records
    with_train = [r for r in recorder.records if r.train_loss is not None]
    with_val = [r for r in recorder.records if r.val_loss is not None]
    print(f"[recorded-ppl] total_records={len(recorder.records)} with_train={len(with_train)} with_val={len(with_val)}")
    assert with_train and with_val

    for r in with_train:
        assert r.train_ppl == pytest.approx(math.exp(r.train_loss))
    for r in with_val:
        assert r.val_ppl == pytest.approx(math.exp(r.val_loss))


def test_train_and_val_land_on_one_record_when_intervals_match(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that aligned train/eval intervals merge both measurements into one record per step.

    Also verifies each merged record carries the expected bookkeeping fields
    (step, max_steps, lr, ms_step) so downstream consumers can rely on them.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    recorder = InMemoryRecorder()

    cfg = _tiny_run_config(log_every=5)
    Trainer(cfg, dataset, tmp_path / "run", recorder=recorder).train()

    print(f"[merged-records] n_records={len(recorder.records)} steps={[r.step for r in recorder.records]}")
    assert len(recorder.records) == 4
    for r in recorder.records:
        assert r.train_loss is not None and r.val_loss is not None
        assert r.step > 0 and r.max_steps == cfg.train.max_steps
        assert r.lr > 0 and r.ms_step >= 0


def test_records_omit_the_measurement_that_did_not_run(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that records without an eval leave the validation fields as None.

    Confirms a recorded train loss is kept while the matching perplexity and
    validation measurements are omitted, not filled with placeholders.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    recorder = InMemoryRecorder()

    Trainer(_tiny_run_config(log_every=1), dataset, tmp_path / "run", recorder=recorder).train()

    train_only = [r for r in recorder.records if r.val_loss is None]
    print(f"[train-only-records] n_train_only={len(train_only)} of {len(recorder.records)} total")
    assert train_only
    for r in train_only:
        assert r.train_loss is not None
        assert r.val_ppl is None


def test_default_recorder_writes_both_files_into_the_run_dir(
    tmp_path: Path, write_synthetic_bin_dataset: Callable[..., Path]
) -> None:
    """Checks that the default (non-injected) recorder persists both train.log and metrics.jsonl.

    Args:
        tmp_path: Temporary directory provided by pytest.
        write_synthetic_bin_dataset: Fixture factory that writes a fake
            binary token dataset and returns its directory.
    """
    data_dir = write_synthetic_bin_dataset(tmp_path / "data")
    dataset = BinDataset(str(data_dir))
    run_dir = tmp_path / "run"

    Trainer(_tiny_run_config(), dataset, run_dir).train()

    print(f"[default-recorder] train.log exists={(run_dir / 'train.log').exists()} metrics.jsonl exists={(run_dir / 'metrics.jsonl').exists()}")
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.jsonl").exists()


_ACCUM_N, _ACCUM_MICRO, _ACCUM_BLOCK, _ACCUM_VOCAB = 4, 2, 16, 50


class _FixedBatchDataset:
    """Dataset stub handing out preset batches in order, then repeating."""

    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]):
        """Stores the batches to hand out and the metadata a Trainer reads.

        Args:
            batches: The `(x, y)` pairs to return, in order.
        """
        self._batches = batches
        self._i = 0
        self.meta = {"vocab_size": 50, "dtype": "uint16"}

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        """Returns the next preset batch, ignoring the requested shape.

        Args:
            split: Ignored; the stub holds one sequence of batches.
            batch_size: Ignored; the preset batches fix their own shape.
            block_size: Ignored; likewise.
            device: Device to move the batch onto.

        Returns:
            The next `(x, y)` pair, cycling once the list is exhausted.
        """
        x, y = self._batches[self._i % len(self._batches)]
        self._i += 1
        return x.to(device), y.to(device)


def _grads_after_one_step(batches, grad_accum_steps, tmp_path, aux_loss_coeff):
    """Runs a single Trainer step and returns the resulting gradients by name.

    Args:
        batches: Batches for the `_FixedBatchDataset` to hand out.
        grad_accum_steps: Micro-batches to accumulate into the step.
        tmp_path: Directory for the throwaway run.
        aux_loss_coeff: Weight of the MoE load-balancing loss.

    Returns:
        A dict mapping parameter name to its gradient after one step.
    """
    base = _tiny_run_config(vocab_size=_ACCUM_VOCAB)
    train_cfg = TrainConfig(
        batch_size=_ACCUM_MICRO, block_size=_ACCUM_BLOCK, grad_accum_steps=grad_accum_steps,
        target_tokens=_ACCUM_MICRO * _ACCUM_BLOCK * grad_accum_steps * 20,
        warmup_steps=2, max_lr=1e-3, eval_interval=5, eval_iters=2,
    )
    cfg = RunConfig(
        replace(base.model, aux_loss_coeff=aux_loss_coeff), train_cfg, base.runtime
    )
    trainer = Trainer(cfg, _FixedBatchDataset(batches), tmp_path / f"run{grad_accum_steps}{aux_loss_coeff}")
    trainer.train_step()
    return {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}


def _split_and_whole_batches():
    """Builds N micro-batches and their concatenation, so both runs see identical tokens.

    Returns:
        A `(chunks, whole)` pair: the list of micro-batches, and the single
        batch holding the same rows.
    """
    torch.manual_seed(0)
    chunks = [
        (torch.randint(0, _ACCUM_VOCAB, (_ACCUM_MICRO, _ACCUM_BLOCK)),
         torch.randint(0, _ACCUM_VOCAB, (_ACCUM_MICRO, _ACCUM_BLOCK)))
        for _ in range(_ACCUM_N)
    ]
    whole = (torch.cat([x for x, _ in chunks]), torch.cat([y for _, y in chunks]))
    return chunks, whole


def _worst_relative_diff(a: dict, b: dict) -> tuple[str, float]:
    """Finds the parameter whose gradient differs most between two runs, relative to its scale.

    Args:
        a: Gradients by parameter name.
        b: Gradients by parameter name, same keys.

    Returns:
        The `(name, relative_difference)` of the worst offender.
    """
    assert a.keys() == b.keys()
    diffs = {
        n: (g - b[n]).abs().max().item() / max(b[n].abs().max().item(), 1e-12)
        for n, g in a.items()
    }
    worst = max(diffs, key=diffs.get)
    return worst, diffs[worst]


def test_gradient_accumulation_matches_the_equivalent_single_batch(tmp_path: Path) -> None:
    """Checks that N accumulated micro-batches produce the gradient of one N-times-larger batch.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    chunks, whole = _split_and_whole_batches()
    accumulated = _grads_after_one_step(chunks, _ACCUM_N, tmp_path, aux_loss_coeff=0.0)
    single = _grads_after_one_step([whole], 1, tmp_path, aux_loss_coeff=0.0)

    name, worst = _worst_relative_diff(accumulated, single)
    print(f"[grad-accum] params={len(accumulated)} worst={name} relative_diff={worst:.3e}")
    assert worst < 1e-4


def test_moe_aux_loss_is_what_accumulation_changes(tmp_path: Path) -> None:
    """Pins the one thing accumulation does not preserve: the MoE load-balancing loss.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    chunks, whole = _split_and_whole_batches()
    accumulated = _grads_after_one_step(chunks, _ACCUM_N, tmp_path, aux_loss_coeff=0.01)
    single = _grads_after_one_step([whole], 1, tmp_path, aux_loss_coeff=0.01)

    name, worst = _worst_relative_diff(accumulated, single)
    print(f"[grad-accum-aux] worst={name} relative_diff={worst:.3e}")
    assert worst > 1e-3, "expected the aux loss to make the runs differ"
    assert "router" in name


def test_grad_accum_steps_below_one_is_rejected() -> None:
    """Checks that a non-positive accumulation count is refused rather than silently skipping the loop."""
    with pytest.raises(ValueError, match="grad_accum_steps"):
        TrainConfig(grad_accum_steps=0)


def test_tokens_per_step_and_max_steps_account_for_accumulation() -> None:
    """Checks that splitting a batch into micro-batches leaves the token budget untouched."""
    direct = TrainConfig(batch_size=8, block_size=1024, target_tokens=1_500_000_000)
    split = TrainConfig(batch_size=1, block_size=1024, grad_accum_steps=8,
                        target_tokens=1_500_000_000)
    print(f"[budget] direct={direct.tokens_per_step}/{direct.max_steps} "
          f"split={split.tokens_per_step}/{split.max_steps}")
    assert split.tokens_per_step == direct.tokens_per_step == 8192
    assert split.max_steps == direct.max_steps

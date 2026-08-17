import math

import pytest
import torch

from src.config import ModelConfig, RuntimeConfig, TrainConfig
from src.model.transformer import Transformer
from src.runtime.schedule import TrainingSchedule


def _tiny_model() -> Transformer:
    """Builds a small Transformer for schedule/optimizer tests.

    Returns:
        A freshly initialized `Transformer`.
    """
    cfg = ModelConfig(
        vocab_size=50, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=8,
        expert_intermediate=16, shared_intermediate=32, n_experts=2, top_k=1,
        max_seq_len=16,
    )
    return Transformer(cfg)


def _train_cfg(**overrides: int | float) -> TrainConfig:
    """Builds a TrainConfig sized for a 100-step tiny run, overridable per field.

    Args:
        **overrides: Field values overriding the defaults (batch_size=2,
            block_size=8, target_tokens for 100 steps, warmup_steps=10,
            max_lr=1e-3, min_lr=1e-4).

    Returns:
        The constructed `TrainConfig`.
    """
    defaults = dict(
        batch_size=2, block_size=8, target_tokens=2 * 8 * 100,  # 100 steps
        warmup_steps=10, max_lr=1e-3, min_lr=1e-4,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_param_groups_split_by_dim_and_cover_every_parameter() -> None:
    """Checks that parameters are split into decay and no-decay groups by ndim.

    Verifies the groups apply weight_decay and 0.0 respectively, follow the
    ndim>=2 boundary, and together cover every model parameter with no loss
    or duplication (relevant with weight tying).
    """
    model = _tiny_model()
    train_cfg = _train_cfg()
    schedule = TrainingSchedule(model, train_cfg, RuntimeConfig())

    decay_group, no_decay_group = schedule.param_groups
    assert decay_group["weight_decay"] == train_cfg.weight_decay
    assert no_decay_group["weight_decay"] == 0.0

    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])

    n_optim_params = sum(p.numel() for p in decay_group["params"]) + sum(
        p.numel() for p in no_decay_group["params"]
    )
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"[param-groups] optim_params={n_optim_params} model_params={n_model_params}")
    assert n_optim_params == n_model_params  # no parameter lost or duplicated (tying)


def test_no_decay_group_contains_norm_weights_only() -> None:
    """Checks that the no-decay group contains exactly the norm weights, and nothing else."""
    model = _tiny_model()
    schedule = TrainingSchedule(model, _train_cfg(), RuntimeConfig())
    _, no_decay_group = schedule.param_groups

    norm_param_ids = {
        id(p) for name, p in model.named_parameters() if "norm" in name
    }
    optim_no_decay_ids = {id(p) for p in no_decay_group["params"]}
    print(f"[no-decay] no_decay_params={len(optim_no_decay_ids)} norm_params={len(norm_param_ids)}")
    assert optim_no_decay_ids == norm_param_ids


def test_lr_schedule_warmup_then_cosine_decay() -> None:
    """Checks that the LR rises linearly through warmup then cosine-decays to min_lr.

    Verifies the schedule reaches max_lr exactly at the end of warmup, never
    drops below min_lr, and lands on min_lr at the final step.
    """
    model = _tiny_model()
    train_cfg = _train_cfg(warmup_steps=10, max_lr=1e-3, min_lr=1e-4)
    schedule = TrainingSchedule(model, train_cfg, RuntimeConfig())

    lrs = []
    for _ in range(train_cfg.max_steps):
        lrs.append(schedule.current_lr)
        schedule.step(model.parameters())

    print(
        f"[lr-schedule] lrs[0]={lrs[0]:.6e} lrs[warmup-1]={lrs[train_cfg.warmup_steps - 1]:.6e} "
        f"lrs[-1]={lrs[-1]:.6e} max_lr={train_cfg.max_lr:.6e} min_lr={train_cfg.min_lr:.6e}"
    )

    # linear warmup: grows monotonically up to max_lr
    assert lrs[0] < lrs[train_cfg.warmup_steps - 1]
    assert math.isclose(lrs[train_cfg.warmup_steps - 1], train_cfg.max_lr, rel_tol=1e-6)

    # after warmup it decays towards min_lr, never below it
    assert all(lr >= train_cfg.min_lr - 1e-9 for lr in lrs)
    assert lrs[-1] < lrs[train_cfg.warmup_steps]
    assert math.isclose(lrs[-1], train_cfg.min_lr, rel_tol=1e-2)


def test_resume_continues_schedule_without_restarting_warmup() -> None:
    """Checks that resuming from a mid-schedule checkpoint continues the LR curve.

    Verifies the resumed learning rates match the reference run at the same
    positions, i.e. warmup is not restarted from scratch.
    """
    model = _tiny_model()
    train_cfg = _train_cfg(warmup_steps=10, max_lr=1e-3, min_lr=1e-4)
    runtime_cfg = RuntimeConfig()

    reference = TrainingSchedule(model, train_cfg, runtime_cfg)
    ref_lrs = []
    captured = None
    for i in range(30):
        ref_lrs.append(reference.current_lr)
        reference.step(model.parameters())
        if i == 19:
            captured = reference.state_dict()  # 20 steps completed

    resumed = TrainingSchedule(model, train_cfg, runtime_cfg)
    resumed.load_state_dict(captured)

    resumed_lrs = []
    for _ in range(10):
        resumed_lrs.append(resumed.current_lr)
        resumed.step(model.parameters())

    print(f"[resume-lr] resumed_lrs={resumed_lrs} reference_lrs[20:30]={ref_lrs[20:30]}")
    assert resumed_lrs == pytest.approx(ref_lrs[20:30])


def test_resume_restores_optimizer_moments_not_just_the_curve() -> None:
    """Checks that resuming restores AdamW's optimizer moments, not just the step counter.

    The LR curve is a pure function of the step and would survive saving
    only the counter; AdamW's moments would not, so they must be persisted
    and restored explicitly.
    """
    model = _tiny_model()
    train_cfg = _train_cfg()
    schedule = TrainingSchedule(model, train_cfg, RuntimeConfig())

    x = torch.randint(0, 50, (2, 8))
    for _ in range(3):
        out = model(x)
        loss = out.logits.square().mean() + out.aux_loss
        schedule.zero_grad()
        loss.backward()
        schedule.step(model.parameters())

    saved = schedule.state_dict()
    resumed = TrainingSchedule(model, train_cfg, RuntimeConfig())
    print(f"[resume-moments] fresh optimizer state keys={list(resumed.state_dict()['optimizer']['state'].keys())}")
    assert resumed.state_dict()["optimizer"]["state"] == {}  # starts without moments

    resumed.load_state_dict(saved)
    assert resumed.state_dict()["optimizer"]["state"].keys() == saved["optimizer"]["state"].keys()

    first_key = next(iter(saved["optimizer"]["state"]))
    exp_avg_equal = torch.equal(
        resumed.state_dict()["optimizer"]["state"][first_key]["exp_avg"],
        saved["optimizer"]["state"][first_key]["exp_avg"],
    )
    print(f"[resume-moments] restored exp_avg equal to saved={exp_avg_equal}")
    assert exp_avg_equal


def test_gradients_are_clipped_to_grad_clip() -> None:
    """Checks that schedule.step() clips the gradient norm down to grad_clip.

    Uses a grad_clip small enough that clipping always engages, so the test
    cannot silently pass without actually clipping anything.
    """
    model = _tiny_model()
    train_cfg = _train_cfg(grad_clip=1e-4)  # tight enough to always bite
    schedule = TrainingSchedule(model, train_cfg, RuntimeConfig())

    x = torch.randint(0, 50, (2, 8))
    out = model(x)
    loss = out.logits.square().mean() + out.aux_loss
    schedule.zero_grad()
    loss.backward()

    norm_before = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    )
    assert norm_before > train_cfg.grad_clip  # the test would be vacuous without this

    schedule.step(model.parameters())
    norm_after = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    )
    print(f"[grad-clip] grad_clip={train_cfg.grad_clip} norm_before={norm_before:.6f} norm_after={norm_after:.6f}")
    assert norm_after == pytest.approx(train_cfg.grad_clip, rel=1e-3)


def test_load_state_dict_rejects_incomplete_state() -> None:
    """Checks that load_state_dict raises KeyError when the schedule key is missing."""
    model = _tiny_model()
    schedule = TrainingSchedule(model, _train_cfg(), RuntimeConfig())
    with pytest.raises(KeyError) as exc_info:
        schedule.load_state_dict({"optimizer": schedule.state_dict()["optimizer"]})
    print(f"[incomplete-state] raised={exc_info.value!r}")

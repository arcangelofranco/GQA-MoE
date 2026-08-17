import math

import torch
import torch.nn as nn

from src.config import RuntimeConfig, TrainConfig


class TrainingSchedule:
    """Owns the optimization step: AdamW, gradient clipping, and the LR curve.

    Everything a training step does after ``loss.backward()`` lives here, along
    with the state needed to resume it. Callers never touch the underlying
    torch optimizer or scheduler, so resuming a run does not depend on their
    internals; only this module's own :meth:`state_dict` /
    :meth:`load_state_dict` pair is part of the checkpoint contract.
    """

    def __init__(self, model: nn.Module, train_cfg: TrainConfig, runtime_cfg: RuntimeConfig):
        """Build the AdamW optimizer and its linear-warmup, cosine-decay LR curve.

        Args:
            model: Model whose parameters will be optimized.
            train_cfg: Training config. Consumes ``weight_decay``, ``max_lr``,
                ``min_lr``, ``warmup_steps``, ``max_steps``, and ``grad_clip``.
            runtime_cfg: Runtime config. Consumes ``adam_betas`` and
                ``adam_eps``.
        """
        self._grad_clip = train_cfg.grad_clip
        self._optimizer = self._build_optimizer(model, train_cfg, runtime_cfg)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, self._build_lr_lambda(train_cfg)
        )

    @staticmethod
    def _build_optimizer(
        model: nn.Module, train_cfg: TrainConfig, runtime_cfg: RuntimeConfig
    ) -> torch.optim.AdamW:
        """Build AdamW with weight decay applied to >=2D parameters only.

        Splits the trainable parameters into two groups: multi-dimensional
        weight matrices (>= 2D, i.e. Linear/Embedding weights) receive the
        configured weight decay, while one-dimensional parameters (biases, norm
        scales) are excluded. This is the standard AdamW practice, since L2
        regularization on small vectors like biases rarely helps and can hurt
        generalization.

        Args:
            model: Model whose parameters will be optimized.
            train_cfg: Training config. Consumes ``weight_decay`` and
                ``max_lr``.
            runtime_cfg: Runtime config. Consumes ``adam_betas`` and
                ``adam_eps``.

        Returns:
            torch.optim.AdamW: The configured AdamW optimizer, with the decay
            group registered first.
        """
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=train_cfg.max_lr,
            betas=runtime_cfg.adam_betas,
            eps=runtime_cfg.adam_eps,
        )

    @staticmethod
    def _build_lr_lambda(train_cfg: TrainConfig):
        """Build the LR multiplier function: linear warmup, then cosine decay.

        The schedule warms up from a near-zero multiplier to ``1.0`` over
        ``warmup_steps`` (so early steps do not destabilize training), then
        decays the multiplier along a cosine curve down to
        ``min_lr / max_lr``. The returned function is a pure function of the
        step index, which is what lets the schedule resume from a step count
        alone without re-running the history.

        Args:
            train_cfg: Training config. Consumes ``warmup_steps``,
                ``max_steps``, ``max_lr``, and ``min_lr``.

        Returns:
            Callable[[int], float]: A function mapping a step index to an LR
            multiplier in ``[min_lr / max_lr, 1.0]``.
        """
        warmup_steps = train_cfg.warmup_steps
        max_steps = train_cfg.max_steps
        min_ratio = train_cfg.min_lr / train_cfg.max_lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return lr_lambda

    @property
    def current_lr(self) -> float:
        """The learning rate the next optimizer step will use.

        Returns:
            float: The current scheduled learning rate, read from the LR
            scheduler's last step.
        """
        return self._scheduler.get_last_lr()[0]

    @property
    def param_groups(self) -> list[dict]:
        """The optimizer's parameter groups, for inspection.

        Exposes the parameter groups (e.g. to report how many parameters fall
        into the decay vs no-decay groups). This is a read-only view: mutating
        the returned groups bypasses this module and is not supported.

        Returns:
            list[dict]: The optimizer's parameter groups.
        """
        return self._optimizer.param_groups

    def zero_grad(self) -> None:
        """Clear the gradients accumulated by the previous step.

        Uses ``set_to_none=True``, which frees the gradient buffers instead of
        zeroing them in place, saving memory and slightly speeding up the
        subsequent backward pass.
        """
        self._optimizer.zero_grad(set_to_none=True)

    def step(self, parameters) -> None:
        """Clip gradients, then advance the optimizer and the LR curve.

        Applies global gradient norm clipping (capped at ``grad_clip``) before
        the optimizer update, then steps both the optimizer and the LR
        scheduler so that the learning rate advances in lockstep with the
        weight update.

        Args:
            parameters: Iterable of parameters whose gradients to clip,
                usually ``model.parameters()``.
        """
        torch.nn.utils.clip_grad_norm_(parameters, self._grad_clip)
        self._optimizer.step()
        self._scheduler.step()

    def state_dict(self) -> dict:
        """Capture everything needed to resume this schedule exactly.

        Returns:
            dict: The optimizer and LR-curve state, to be handed back to
            :meth:`load_state_dict`. This dict is stored inside the trainer's
            checkpoint.
        """
        return {
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer moments and the position on the LR curve.

        Loads the optimizer state (e.g. Adam moments) and the LR scheduler
        state (e.g. the step counter) previously captured by
        :meth:`state_dict`, so training can continue exactly where it left
        off.

        Args:
            state: A dict previously produced by :meth:`state_dict`.

        Raises:
            KeyError: If ``state`` is missing the ``"optimizer"`` or
                ``"scheduler"`` key.
        """
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])

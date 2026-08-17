import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import RunConfig
from src.data.dataset import BinDataset
from src.model.transformer import Transformer
from src.runtime.metrics import RunDirRecorder, RunRecorder, StepMetrics
from src.runtime.schedule import TrainingSchedule


class Trainer:
    """Owns the model, schedule, and training loop: steps, eval, checkpoints.

    The central orchestration object of the training phase. It holds the
    model and its :class:`TrainingSchedule`, drives the loop of batch
    sampling, forward/backward, gradient update, periodic validation, metric
    recording, and checkpointing. It also owns the checkpoint format: a dict
    with model weights, schedule state, step count, RNG state, and the model
    and training configs, so a run can be resumed exactly.
    """

    def __init__(
        self,
        cfg: RunConfig,
        dataset: BinDataset,
        run_dir: str,
        recorder: RunRecorder | None = None,
    ):
        """Build the model and its training schedule, and prepare the run directory.

        Args:
            cfg: The run configuration. Its cross-config invariants were
                already checked when it was built (e.g. by the CLI or the
                config layer).
            dataset: Dataset to sample train/val batches from.
            run_dir: Directory for checkpoints and logs; created recursively
                if missing.
            recorder: Where each logged step's metrics go. Defaults to a
                :class:`RunDirRecorder` writing into ``run_dir``, which is also
                created here.
        """
        self.cfg = cfg
        self.dataset = dataset
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder if recorder is not None else RunDirRecorder(self.run_dir)

        torch.manual_seed(cfg.train.seed)

        self.device = torch.device(cfg.runtime.device)
        self.model = Transformer(cfg.model).to(self.device)
        self.step = 0

        self.schedule = TrainingSchedule(self.model, cfg.train, cfg.runtime)

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass and compute the combined training loss.

        Args:
            x: Input token ids of shape ``(B, block_size)``.
            y: Target token ids of shape ``(B, block_size)``, i.e. ``x``
                shifted by one token.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
            ``(total_loss, ce_loss, aux_loss)`` where ``total_loss`` is the
            cross-entropy plus the MoE auxiliary loss (the quantity used for
            backprop), and ``ce_loss``/``aux_loss`` are returned separately so
            callers can log them individually.
        """
        out = self.model(x)
        logits = out.logits
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return ce_loss + out.aux_loss, ce_loss, out.aux_loss

    def train_step(self) -> tuple[float, float]:
        """Run a single training step: forward, backward, then the schedule's step.

        Switches the model to train mode, samples a fresh training batch,
        computes the combined loss, zeroes the gradients, backpropagates,
        applies the schedule's gradient clipping plus optimizer update, and
        advances the internal step counter.

        Returns:
            tuple[float, float]: A pair ``(loss, ce_loss)``: the combined loss
            and its cross-entropy component, both as Python floats detached
            from the autograd graph for cheap logging.
        """
        self.model.train()
        x, y = self.dataset.get_batch(
            "train", self.cfg.train.batch_size, self.cfg.train.block_size, self.device
        )
        loss, ce_loss, aux_loss = self._compute_loss(x, y)

        self.schedule.zero_grad()
        loss.backward()
        self.schedule.step(self.model.parameters())
        self.step += 1
        return loss.item(), ce_loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute the average validation loss over cfg.train.eval_iters batches.

        Runs under ``torch.no_grad`` in eval mode, averaging the combined
        (cross-entropy plus aux) loss over a fixed number of validation
        batches, then restores the model to train mode before returning.

        Returns:
            float: The mean combined loss on the validation split.
        """
        self.model.eval()
        total = 0.0
        for _ in range(self.cfg.train.eval_iters):
            x, y = self.dataset.get_batch(
                "val", self.cfg.train.batch_size, self.cfg.train.block_size, self.device
            )
            loss, _, _ = self._compute_loss(x, y)
            total += loss.item()
        self.model.train()
        return total / self.cfg.train.eval_iters

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model, schedule, step, RNG state, and configs to a checkpoint file.

        Serializes everything needed to resume training exactly where it left
        off: the model weights, the schedule state (optimizer moments and LR
        curve position), the step counter, the global RNG state, and both the
        model and training configs so the run can be reconstructed without
        re-passing the original CLI arguments.

        Args:
            path: Destination path. Defaults to
                ``run_dir/ckpt_step{step}.pt`` when not given.

        Returns:
            Path: The path the checkpoint was written to.
        """
        path = Path(path) if path is not None else self.run_dir / f"ckpt_step{self.step}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "schedule": self.schedule.state_dict(),
                "step": self.step,
                "rng_state": torch.get_rng_state(),
                "model_cfg": self.cfg.model.to_dict(),
                "train_cfg": self.cfg.train.to_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model, schedule, step, and RNG state from a checkpoint.

        Loads the checkpoint to CPU first (so large GPU checkpoints do not
        allocate GPU memory during deserialization), then restores the model
        weights, the schedule state, the step counter, and the global RNG
        state. Rejecting pre-``TrainingSchedule`` checkpoints up front prevents
        a silent resume with a broken LR curve.

        Args:
            path: Path to a checkpoint file saved by :meth:`save_checkpoint`.

        Returns:
            int: The restored step count, so callers can report where training
            resumed.

        Raises:
            KeyError: If the checkpoint predates the ``"schedule"`` key, i.e.
                was written when the Trainer stored a bare ``"optimizer"``
                instead.
        """
        ckpt = torch.load(path, map_location="cpu")
        if "schedule" not in ckpt and "optimizer" in ckpt:
            raise KeyError(
                f"{path} is a pre-TrainingSchedule checkpoint: it stores a bare "
                "'optimizer' and no 'schedule', so the LR curve cannot be restored. "
                "Its weights are still loadable for generation."
            )

        self.model.load_state_dict(ckpt["model"])
        self.schedule.load_state_dict(ckpt["schedule"])
        self.step = ckpt["step"]
        torch.set_rng_state(ckpt["rng_state"])
        return self.step

    def train(self) -> int:
        """Run the training loop until max_steps, recording and checkpointing periodically.

        Loops training steps until ``max_steps`` is reached. Every
        ``log_every`` steps a train-loss record is logged; every
        ``eval_interval`` steps a validation loss is computed and recorded;
        every ``checkpoint_every`` steps a checkpoint is saved to
        ``run_dir/ckpt_step{step}.pt``.

        Returns:
            int: The final step count, equal to ``cfg.train.max_steps``.
        """
        while self.step < self.cfg.train.max_steps:
            t0 = time.time()
            _, train_ce_loss = self.train_step()
            dt = time.time() - t0

            should_log = self.step % self.cfg.runtime.log_every == 0
            should_eval = self.step % self.cfg.train.eval_interval == 0

            if should_log or should_eval:
                self.recorder.record(
                    StepMetrics(
                        step=self.step,
                        max_steps=self.cfg.train.max_steps,
                        lr=self.schedule.current_lr,
                        ms_step=dt * 1000,
                        train_loss=train_ce_loss if should_log else None,
                        val_loss=self.evaluate() if should_eval else None,
                    )
                )

            if self.step % self.cfg.runtime.checkpoint_every == 0:
                self.save_checkpoint()

        return self.step

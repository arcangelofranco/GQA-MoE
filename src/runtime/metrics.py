import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StepMetrics:
    """What one logged training step reports.

    Perplexity is derived here rather than at each place that displays it, so
    it cannot drift from the loss it comes from: every consumer reads the same
    ``math.exp(loss)`` value computed exactly once per property access.

    Attributes:
        step: Step index this record describes.
        max_steps: Total steps planned for the run.
        lr: Learning rate the step used.
        ms_step: Wall-clock duration of the training step, in milliseconds.
        train_loss: Cross-entropy on the training batch, or ``None`` when this
            record carries an evaluation only.
        val_loss: Mean validation loss, or ``None`` when no evaluation ran.
    """

    step: int
    max_steps: int
    lr: float
    ms_step: float
    train_loss: float | None = None
    val_loss: float | None = None

    @property
    def train_ppl(self) -> float | None:
        """Training perplexity, or None if this record has no train_loss.

        Returns:
            float | None: ``math.exp(train_loss)`` when a training loss is
            present, otherwise ``None``.
        """
        return None if self.train_loss is None else math.exp(self.train_loss)

    @property
    def val_ppl(self) -> float | None:
        """Validation perplexity, or None if this record has no val_loss.

        Returns:
            float | None: ``math.exp(val_loss)`` when a validation loss is
            present, otherwise ``None``.
        """
        return None if self.val_loss is None else math.exp(self.val_loss)

    def to_dict(self) -> dict:
        """Convert to a flat, JSON-serializable dict including the perplexities.

        Produces one key per reported quantity, with the perplexities derived
        from the corresponding losses. The dict is ready to be passed to
        ``json.dumps`` (e.g. by :class:`RunDirRecorder`).

        Returns:
            dict: A flat mapping with one key per reported quantity; absent
            measurements are ``None``.
        """
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "train_loss": self.train_loss,
            "train_ppl": self.train_ppl,
            "val_loss": self.val_loss,
            "val_ppl": self.val_ppl,
            "lr": self.lr,
            "ms_step": self.ms_step,
        }


def format_step_line(metrics: StepMetrics) -> str:
    """Render a record as the single human-readable line written to train.log.

    Builds a pipe-separated line that starts with the step counter and always
    ends with the learning rate and the per-step duration. The train (loss and
    perplexity) and val (loss and perplexity) pairs are included only when the
    corresponding measurement is present, so an evaluation-only record renders
    without a train pair and vice versa.

    Args:
        metrics: The record to render.

    Returns:
        str: A line such as
        ``"step 10/976 | train_loss 4.1203 | train_ppl 61.45 | lr 3.00e-04 | 812 ms/step"``,
        omitting the train or val pair when that measurement is absent.
    """
    parts = [f"step {metrics.step}/{metrics.max_steps}"]
    if metrics.train_loss is not None:
        parts.append(f"train_loss {metrics.train_loss:.4f}")
        parts.append(f"train_ppl {metrics.train_ppl:.2f}")
    if metrics.val_loss is not None:
        parts.append(f"val_loss {metrics.val_loss:.4f}")
        parts.append(f"val_ppl {metrics.val_ppl:.2f}")
    parts.append(f"lr {metrics.lr:.2e}")
    parts.append(f"{metrics.ms_step:.0f} ms/step")
    return " | ".join(parts)


class RunRecorder(Protocol):
    """Where a training run's metrics go.

    A structural protocol describing the minimal surface a metrics sink must
    implement. Any object with a matching :meth:`record` method satisfies it,
    so the trainer can consume file-backed, in-memory, or test doubles without
    coupling to a concrete type.
    """

    def record(self, metrics: StepMetrics) -> None:
        """Take one step's metrics.

        Args:
            metrics: The record to take.
        """
        ...


class RunDirRecorder:
    """Writes a run's metrics into its run directory, echoing them to stdout.

    Produces two views of the same records: ``train.log``, one readable line
    per step for watching a run go by, and ``metrics.jsonl``, one JSON object
    per step for anything that plots or analyses them afterwards. Both files
    are opened in append mode, so resuming a run preserves its earlier
    records.
    """

    def __init__(self, run_dir: str | Path, echo: bool = True):
        """Prepare the run directory and the two output files.

        Args:
            run_dir: Directory to write ``train.log`` and ``metrics.jsonl``
                into, created recursively if missing. Both files are appended
                to, so resuming a run keeps its earlier records.
            echo: Whether to also print each line to stdout in addition to
                writing it to ``train.log``.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.echo = echo

    def record(self, metrics: StepMetrics) -> None:
        """Append one record to both files, and print it if echoing.

        Renders the human-readable line once, optionally prints it, then
        appends it to ``train.log`` and appends the JSON-serialized dict to
        ``metrics.jsonl``.

        Args:
            metrics: The record to write.
        """
        line = format_step_line(metrics)
        if self.echo:
            print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics.to_dict()) + "\n")


class InMemoryRecorder:
    """Keeps records in a list instead of writing them anywhere.

    The test-side adapter: assertions read the records directly rather than
    matching a rendered line, so tests can inspect structured values without
    depending on the log format. It also implements :class:`RunRecorder`, so it
    can be swapped into the trainer wherever a file-backed recorder is not
    wanted.
    """

    def __init__(self):
        """Start with no records.

        Sets up an empty list; records are accumulated by subsequent
        :meth:`record` calls.
        """
        self.records: list[StepMetrics] = []

    def record(self, metrics: StepMetrics) -> None:
        """Append one record to the list.

        Args:
            metrics: The record to keep.
        """
        self.records.append(metrics)

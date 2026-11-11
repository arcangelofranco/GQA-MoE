from pathlib import Path

import torch

from src.config import ModelConfig, RunConfig
from src.data.dataset import BinDataset
from src.model.transformer import Transformer
from src.runtime.schedule import TrainingSchedule

CHECKPOINT_VERSION = 2


def save(
    path: str | Path,
    *,
    model: Transformer,
    schedule: TrainingSchedule,
    dataset: BinDataset,
    step: int,
    cfg: RunConfig,
) -> Path:
    """Write a checkpoint holding everything needed to resume a run exactly.

    The file stores the model weights, the schedule state (optimizer moments
    and LR curve position), the step counter, the torch CPU and CUDA RNG
    states, the dataset's per-split sampling RNG states, the full run
    configuration, and the directory of the tokenizer the dataset was encoded
    with. Restoring all of them makes a resumed run continue on the same
    batches and the same LR curve as an uninterrupted one.

    Args:
        path: Destination file.
        model: The model whose weights are saved.
        schedule: The training schedule whose state is saved.
        dataset: The dataset whose sampling RNG state and tokenizer directory
            are saved.
        step: Number of optimizer steps completed so far.
        cfg: The run configuration, checked against the current one on resume.

    Returns:
        Path: The path the checkpoint was written to.
    """
    path = Path(path)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "schedule": schedule.state_dict(),
            "dataset": dataset.state_dict(),
            "step": step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "cfg": cfg.to_dict(),
            "tokenizer_dir": dataset.tokenizer_dir,
        },
        path,
    )
    return path


def restore(
    path: str | Path,
    *,
    model: Transformer,
    schedule: TrainingSchedule,
    dataset: BinDataset,
    cfg: RunConfig,
) -> int:
    """Restore a run's state from a checkpoint written by :func:`save`.

    Loads the file on CPU, verifies that it uses the current format and that
    its ``model`` and ``train`` configuration equals ``cfg``, then restores
    the model weights, the schedule state, the dataset sampling RNGs and the
    torch RNGs. The ``runtime`` section (device, dtype, logging intervals) may
    differ, since changing it does not alter what the run computes.

    Args:
        path: Checkpoint file to restore from.
        model: Model to load the weights into.
        schedule: Schedule to load the optimizer and LR state into.
        dataset: Dataset to load the sampling RNG state into.
        cfg: The current run configuration.

    Returns:
        int: The restored step count.

    Raises:
        ValueError: If the checkpoint predates format version
            :data:`CHECKPOINT_VERSION`, or if its ``model``/``train``
            configuration differs from ``cfg``; the message lists the
            differing fields.
    """
    ckpt = torch.load(path, map_location="cpu")
    version = ckpt.get("version", 1)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} uses checkpoint format version {version}, but resuming requires "
            f"version {CHECKPOINT_VERSION}: it lacks the state needed to resume exactly "
            "(dataset sampling RNG, full run config). Its weights are still loadable "
            "for generation."
        )

    saved, current = ckpt["cfg"], cfg.to_dict()
    diffs = [
        f"{section}.{key}: checkpoint={saved[section].get(key)!r} current={value!r}"
        for section in ("model", "train")
        for key, value in current[section].items()
        if saved[section].get(key) != value
    ]
    if diffs:
        raise ValueError(
            f"{path} was saved with a different configuration; start a new run to "
            "change it. Differing fields: " + "; ".join(diffs)
        )

    model.load_state_dict(ckpt["model"])
    schedule.load_state_dict(ckpt["schedule"])
    dataset.load_state_dict(ckpt["dataset"])
    torch.set_rng_state(ckpt["rng_state"])
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return ckpt["step"]


def load_for_inference(
    path: str | Path, device: str
) -> tuple[Transformer, ModelConfig, str | None]:
    """Rebuild a trained model from a checkpoint, ready for generation.

    Reads the file on CPU so deserialization never allocates GPU memory, then
    reconstructs the :class:`Transformer` from the stored model configuration,
    loads its weights, moves it to ``device`` and switches it to eval mode.
    Works with every checkpoint format version, including those that can no
    longer be resumed.

    Args:
        path: Checkpoint file to load.
        device: Device to move the model to, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        tuple[Transformer, ModelConfig, str | None]: The model in eval mode on
        ``device``, the configuration it was built from (so callers can check
        it against the tokenizer), and the directory of the tokenizer the
        training data was encoded with, or ``None`` when the checkpoint does
        not record it.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If the stored weights do not match the stored
            configuration.
    """
    ckpt = torch.load(path, map_location="cpu")
    # Version 1 stored the model config alone under "model_cfg".
    model_cfg = ModelConfig.from_dict(ckpt["cfg"]["model"] if "cfg" in ckpt else ckpt["model_cfg"])
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg, ckpt.get("tokenizer_dir")

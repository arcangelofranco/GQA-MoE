import argparse
import json
from pathlib import Path

from src.config import PRESET_NAMES, RunConfig
from src.data.dataset import BinDataset
from src.runtime.trainer import Trainer


def _read_vocab_size(data_dir: str) -> int:
    """Read the vocabulary size from a processed dataset's metadata file.

    Loads the ``meta.json`` produced by the data preparation pipeline and
    extracts the ``vocab_size`` entry, which is used to align the model
    architecture with the tokenizer used to encode the corpus.

    Args:
        data_dir: Directory containing the dataset's ``meta.json`` file.

    Returns:
        int: The ``vocab_size`` stored in ``meta.json``.

    Raises:
        FileNotFoundError: If ``data_dir/meta.json`` does not exist.
        KeyError: If ``meta.json`` is missing the ``vocab_size`` key.
        json.JSONDecodeError: If ``meta.json`` is not valid JSON.
    """
    with open(Path(data_dir) / "meta.json") as f:
        return json.load(f)["vocab_size"]


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration from CLI args, either from a YAML file or a preset.

    Implements the configuration resolution precedence: if ``--config`` is
    provided, the :class:`RunConfig` is loaded verbatim from the YAML file,
    giving the user full control over every hyper parameter. Otherwise, a
    named preset is instantiated and its vocabulary size is overridden with
    the value read from the processed dataset's ``meta.json``, guaranteeing the
    model output head matches the actual tokenizer vocabulary.

    Args:
        args: Parsed CLI arguments. When ``args.config`` is set, it takes
            precedence over ``args.preset``; ``args.data_dir`` is only used in
            the preset branch.

    Returns:
        RunConfig: The fully resolved :class:`RunConfig` for this run, ready to
        be passed to the :class:`Trainer`.

    Raises:
        FileNotFoundError: If ``args.config`` points to a missing YAML file, or
            if ``meta.json`` cannot be found when resolving a preset.
        KeyError: If ``args.preset`` is not a known preset name (when
            ``args.config`` is unset).
    """
    if args.config:
        return RunConfig.from_yaml(args.config)
    return RunConfig.preset(args.preset, _read_vocab_size(args.data_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface for training.

    Builds an :class:`argparse.ArgumentParser` exposing the configuration
    entry points and I/O paths needed to launch a training run, and parses the
    given arguments into a lightweight :class:`argparse.Namespace`.

    Args:
        argv: Optional sequence of raw command-line arguments to parse.
            When ``None``, :func:`argparse.ArgumentParser.parse_args` falls back
            to ``sys.argv[1:]``. Providing this parameter explicitly makes the
            function trivially unit-testable.

    Returns:
        argparse.Namespace: Populated namespace exposing the following
        attributes:

        - ``preset`` (str): Named architecture preset, constrained to
          :data:`PRESET_NAMES` (default ``"nano"``).
        - ``config`` (str | None): Path to a YAML configuration file that, when
          set, takes precedence over ``--preset``.
        - ``data_dir`` (str): Directory containing ``train.bin``, ``val.bin``
          and ``meta.json`` (default ``"data/processed"``).
        - ``run_dir`` (str): Directory where checkpoints and training logs are
          written (default ``"runs/nano"``).
        - ``resume`` (str | None): Path to a previously saved checkpoint to
          resume training from; ``None`` starts a fresh run.

    Raises:
        SystemExit: On invalid preset names, ``--help``, or invalid arguments,
            consistent with standard argparse behaviour.
    """
    parser = argparse.ArgumentParser(description="Train the GQA-MoE model.")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="nano")
    parser.add_argument("--config", default=None, help="path to a YAML file: overrides --preset")
    parser.add_argument("--data-dir", default="data/processed", help="folder with train.bin/val.bin/meta.json")
    parser.add_argument("--run-dir", default="runs/nano", help="folder for checkpoints and logs")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Trainer:
    """CLI entry point: build the run config, train the model, and save the final checkpoint.

    Orchestrates a complete training run: it parses the command-line
    arguments, resolves the :class:`RunConfig` (YAML file or preset), loads the
    pre-encoded :class:`BinDataset`, validates that the dataset vocabulary
    matches the model configuration, constructs the :class:`Trainer`, resumes
    from a checkpoint when requested, runs the training loop, and persists the
    final weights to ``run_dir/final.pt``.

    Returns:
        Trainer: The fully trained :class:`Trainer` instance, allowing callers
        to inspect metrics, hyper parameters, or the trained model after
        completion.

    Raises:
        ValueError: If the dataset's ``vocab_size`` does not match the model
            configuration's vocabulary size, signalling a tokenizer mismatch
            between the data pipeline and the model.
        FileNotFoundError: If the data directory, a resume checkpoint, or the
            YAML config file cannot be found.
        RuntimeError: If the training loop fails (e.g. CUDA out-of-memory or
            checkpoint corruption when resuming).
    """
    args = parse_args(argv)
    cfg = build_run_config(args)

    dataset = BinDataset(args.data_dir, seed=cfg.train.seed)
    cfg.model.require_vocab_size(dataset.meta["vocab_size"], source="dataset")

    trainer = Trainer(cfg, dataset, args.run_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    trainer.save_checkpoint(Path(args.run_dir) / "final.pt")
    return trainer


if __name__ == "__main__":
    main()

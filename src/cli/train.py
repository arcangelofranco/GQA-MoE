import argparse
import json
from pathlib import Path

from src.config import PRESET_NAMES, RunConfig
from src.data.dataset import BinDataset
from src.runtime.trainer import Trainer


def _read_vocab_size(data_dir: str) -> int:
    """Read the vocabulary size from a processed dataset's metadata file.

    Args:
        data_dir: Directory containing meta.json.

    Returns:
        The vocab_size stored in meta.json.
    """
    with open(Path(data_dir) / "meta.json") as f:
        return json.load(f)["vocab_size"]


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration from CLI args, either from a YAML file or a preset.

    Args:
        args: Parsed CLI arguments. If args.config is set, it takes
            precedence over args.preset.

    Returns:
        The RunConfig for this run.
    """
    if args.config:
        return RunConfig.from_yaml(args.config)
    return RunConfig.preset(args.preset, _read_vocab_size(args.data_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line arguments for training.

    Args:
        argv: List of arguments to parse. If None, sys.argv is used (argparse's default behavior).

    Returns:
        Namespace containing the parsed arguments (preset, config, data_dir, run_dir, resume).
    """
    parser = argparse.ArgumentParser(description="Train the GQA-MoE model.")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="nano")
    parser.add_argument("--config", default=None, help="path to a YAML file: overrides --preset")
    parser.add_argument("--data-dir", default="data/processed", help="folder with train.bin/val.bin/meta.json")
    parser.add_argument("--run-dir", default="runs/nano", help="folder for checkpoints and logs")
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Trainer:
    """CLI entry point: builds the run config, trains the model, and saves the final checkpoint.

    Args:
        argv: List of command-line arguments to parse. If None, sys.argv is used.

    Returns:
        The Trainer instance after training has completed.

    Raises:
        ValueError: If the dataset's vocab_size does not match the model's.
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

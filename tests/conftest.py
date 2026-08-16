import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


@pytest.fixture
def write_synthetic_bin_dataset() -> Callable[..., Path]:
    """Factory fixture that writes fake train.bin/val.bin/meta.json files.

    Writes random uint16 ids in the shape expected by `BinDataset`
    (src/data/dataset.py). Shared between the Trainer tests and the
    training CLI smoke tests, which used to duplicate the exact same
    logic under two different names.

    Returns:
        A `_write(data_dir, vocab_size=50, n_train=4000, n_val=1000,
        seed=0)` callable that writes the dataset and returns its directory.
    """

    def _write(
        data_dir: str | Path,
        vocab_size: int = 50,
        n_train: int = 4000,
        n_val: int = 1000,
        seed: int = 0,
    ) -> Path:
        """Writes a synthetic binary token dataset to `data_dir`.

        Args:
            data_dir: Directory to write train.bin/val.bin/meta.json into; created if missing.
            vocab_size: Vocabulary size for the randomly generated ids.
            n_train: Number of training tokens to generate.
            n_val: Number of validation tokens to generate.
            seed: Seed for the random id generator.

        Returns:
            The directory the dataset was written to, as a `Path`.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        train_ids = rng.integers(0, vocab_size, size=n_train, dtype=np.uint16)
        val_ids = rng.integers(0, vocab_size, size=n_val, dtype=np.uint16)
        np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="w+", shape=train_ids.shape)[:] = train_ids
        np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="w+", shape=val_ids.shape)[:] = val_ids

        meta = {"vocab_size": vocab_size, "dtype": "uint16", "train_tokens": n_train, "val_tokens": n_val}
        with open(data_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        print(
            f"[synthetic-dataset] data_dir={data_dir} vocab_size={vocab_size} "
            f"train_tokens={n_train} val_tokens={n_val} seed={seed}"
        )
        return data_dir

    return _write

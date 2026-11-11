from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.data.dataset import BinDataset


@pytest.fixture
def write_synthetic_bin_dataset() -> Callable[..., Path]:
    """Factory fixture that writes fake train.bin/val.bin/meta.json files.

    Writes random ids through :meth:`src.data.dataset.BinDataset.write`, so
    the fixture produces exactly the format the pipeline produces. Shared
    between the Trainer tests and the training CLI smoke tests.

    Returns:
        Callable[..., Path]: A ``_write(data_dir, vocab_size=50, n_train=4000,
        n_val=1000, seed=0, tokenizer_dir=None)`` callable that writes the
        dataset and returns its directory.
    """

    def _write(
        data_dir: str | Path,
        vocab_size: int = 50,
        n_train: int = 4000,
        n_val: int = 1000,
        seed: int = 0,
        tokenizer_dir: str | None = None,
    ) -> Path:
        """Write a synthetic binary token dataset to ``data_dir``.

        Generates random token ids within ``[0, vocab_size)`` for both splits
        and writes them with :meth:`BinDataset.write`.

        Args:
            data_dir: Directory to write ``train.bin``/``val.bin``/
                ``meta.json`` into; created recursively if missing.
            vocab_size: Vocabulary size for the randomly generated ids.
            n_train: Number of training tokens to generate.
            n_val: Number of validation tokens to generate.
            seed: Seed for the random id generator.
            tokenizer_dir: Tokenizer directory to record in ``meta.json``.

        Returns:
            Path: The directory the dataset was written to.
        """
        data_dir = Path(data_dir)
        rng = np.random.default_rng(seed)
        BinDataset.write(
            data_dir,
            {
                "train": [rng.integers(0, vocab_size, size=n_train)],
                "val": [rng.integers(0, vocab_size, size=n_val)],
            },
            vocab_size=vocab_size,
            tokenizer_dir=tokenizer_dir,
        )

        print(
            f"[synthetic-dataset] data_dir={data_dir} vocab_size={vocab_size} "
            f"train_tokens={n_train} val_tokens={n_val} seed={seed}"
        )
        return data_dir

    return _write

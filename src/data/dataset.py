import json
from pathlib import Path
import numpy as np
import torch

class BinDataset:
    """Memory-mapped train/val token dataset backed by raw .bin files.

    Expects `bin_dir` to contain `meta.json` (with "dtype" and "vocab_size"),
    `train.bin`, and `val.bin`, as produced by the tokenization pipeline.
    """

    def __init__(self, bin_dir: str, seed: int | None = None):
        """Load dataset metadata and memory-map the train/val token arrays.

        Args:
            bin_dir: Directory containing meta.json, train.bin, and val.bin.
            seed: Seed for the independent train/val sampling RNGs. If None,
                a fresh, unpredictable seed is used.
        """
        bin_dir = Path(bin_dir)
        with open(bin_dir / "meta.json") as file:
            self.meta = json.load(file)

        dtype = np.uint16 if self.meta["dtype"] == "uint16" else np.uint32
        self._data = {
            "train" : np.memmap(bin_dir / "train.bin", dtype=dtype, mode="r"),
            "val" : np.memmap(bin_dir / "val.bin", dtype=dtype, mode="r")
        }
        # Separate generators for each split:
        # sampling "val" more or less frequently should not affect the sequence of batches
        # sampled for "train". spawn() creates two independent streams from the same
        # SeedSequence, avoiding the use of consecutive seeds such as seed and seed + 1.
        train_seed, val_seed = np.random.SeedSequence(seed).spawn(2)
        self._rng = {
            "train": np.random.default_rng(train_seed),
            "val": np.random.default_rng(val_seed),
        }

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of input/target sequences from a split.

        Args:
            split: Which split to sample from, "train" or "val".
            batch_size: Number of sequences to sample.
            block_size: Length of each sequence (context length).
            device: Device to move the resulting tensors to.

        Returns:
            A tuple (x, y) of LongTensors with shape (batch_size, block_size),
            where y is x shifted by one token (the next-token targets).

        Raises:
            ValueError: If split is not "train"/"val", or if the split has
                fewer than block_size + 1 tokens.
        """
        if split not in self._data:
            raise ValueError(f"split '{split}' sconosciuto: atteso 'train' o 'val'.")

        data = self._data[split]
        if len(data) <= block_size:
            raise ValueError(
                f"split '{split}' ha solo {len(data)} token, non basta per "
                f"block_size={block_size}: servono almeno block_size + 1 token."
            )

        ix = self._rng[split].integers(0, len(data) - block_size, size=batch_size)

        # .astype(np.int64): nn.Embedding expects a LongTensor, while the memmap uses uint16/32.
        
        x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)
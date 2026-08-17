import json
from pathlib import Path
import numpy as np
import torch

from src.data.token_dtype import dtype_from_name

class BinDataset:
    """Memory-mapped train/val token dataset backed by raw binary files.

    The corpus is pre-encoded into two flat token arrays and exposed through
    NumPy memmaps so the entire dataset never needs to be loaded into RAM.
    This makes the dataset memory-efficient for arbitrarily large corpora while
    keeping sampling fast, since reads are paged directly from disk.

    Expects ``bin_dir`` to contain:

    - ``meta.json`` with at least the ``"dtype"`` (``"uint16"`` or
      ``"uint32"``) and ``"vocab_size"`` keys;
    - ``train.bin``: flat array of uint16/uint32 token ids for training;
    - ``val.bin``: flat array of uint16/uint32 token ids for validation.

    These files are produced by the tokenization pipeline in
    :mod:`src.data.prepare_data`.

    Sampling uses two fully independent pseudo-random streams (one per split)
    derived from a shared ``SeedSequence``, so sampling the validation split
    more or less frequently never perturbs the sequence of training batches.
    """

    def __init__(self, bin_dir: str, seed: int | None = None):
        """Load dataset metadata and memory-map the train/val token arrays.

        Parses ``meta.json``, opens read-only memmaps over ``train.bin`` and
        ``val.bin``, and seeds two independent per-split RNGs. Both memmaps
        stay open for the lifetime of the dataset and are closed when the
        object is garbage collected.

        Args:
            bin_dir: Directory containing ``meta.json``, ``train.bin``, and
                ``val.bin`` as produced by the tokenization pipeline.
            seed: Seed used to derive the train/val sampling RNGs. When
                ``None``, a fresh, unpredictable seed is drawn from the
                operating system, making runs non-reproducible but secure.

        Attributes:
            meta (dict): Raw contents of ``meta.json``, exposing at least
                ``"dtype"`` and ``"vocab_size"`` for downstream validation.

        Raises:
            FileNotFoundError: If ``bin_dir`` or any of the expected files
                does not exist.
            KeyError: If ``meta.json`` is missing the ``"dtype"`` key.
            ValueError: If ``"dtype"`` is not ``"uint16"`` or ``"uint32"``.
        """
        bin_dir = Path(bin_dir)
        with open(bin_dir / "meta.json") as file:
            self.meta = json.load(file)

        dtype = dtype_from_name(self.meta["dtype"])
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

        Draws ``batch_size`` starting positions uniformly at random within the
        split, then slices ``block_size`` consecutive tokens from each offset.
        The input ``x`` and target ``y`` share the same windows, with ``y``
        shifted one token forward so the model predicts the next token at every
        position. Tensors are returned as int64 ``LongTensor`` values, as
        required by ``nn.Embedding``.

        The RNG is split-specific and stateful: successive calls produce
        different batches without consuming any shared global random state.

        Args:
            split: Which split to sample from, ``"train"`` or ``"val"``.
            batch_size: Number of independent sequences to sample.
            block_size: Length of each sampled sequence (context window). The
                split must contain at least ``block_size + 1`` tokens.
            device: Target device for the returned tensors (e.g. ``"cpu"`` or
                ``"cuda"``). Transfer happens once after construction.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair ``(x, y)`` of int64
            tensors with shape ``(batch_size, block_size)``. ``y`` is ``x``
            shifted by one token position (the next-token prediction targets).

        Raises:
            ValueError: If ``split`` is not ``"train"``/``"val"``, or if the
                split contains fewer than ``block_size + 1`` tokens.
        """
        if split not in self._data:
            raise ValueError(f"Unknown split '{split}': expected 'train' or 'val'.")    

        data = self._data[split]
        if len(data) <= block_size:
            raise ValueError(
                f"Split '{split}' contains only {len(data)} tokens, which is not enough for "
                f"block_size={block_size}: at least block_size + 1 tokens are required."
            )

        ix = self._rng[split].integers(0, len(data) - block_size, size=batch_size)

        # .astype(np.int64): nn.Embedding expects a LongTensor, while the memmap uses uint16/32.
        
        x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)
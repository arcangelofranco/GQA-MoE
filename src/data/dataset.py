import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

# Largest vocabulary whose ids all fit in a ``uint16``, inclusive.
#
# A ``uint16`` (2^16 values) holds 0..65535. Ids run over ``[0, vocab_size)``,
# so the largest id of a vocabulary of this size is 65535 -- exactly
# representable. The comparison below is therefore ``<=``: a 65536-token
# vocabulary still encodes in half the disk space of a ``uint32``, and only
# 65537 and above need to widen.
_MAX_UINT16_VOCAB_SIZE = 65536

# The dtype names that may appear in ``meta.json``, and what they decode to.
_DTYPES_BY_NAME: dict[str, type] = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}

SPLITS = ("train", "val")


def _dtype_name_for_vocab(vocab_size: int) -> str:
    """Pick the narrowest dtype name that can hold every id of a vocabulary.

    Args:
        vocab_size: Number of distinct token ids; ids run over
            ``[0, vocab_size)``.

    Returns:
        str: ``"uint16"`` when ``vocab_size`` is at most
        :data:`_MAX_UINT16_VOCAB_SIZE`, otherwise ``"uint32"``.
    """
    return "uint16" if vocab_size <= _MAX_UINT16_VOCAB_SIZE else "uint32"


def _dtype_from_name(name: str) -> type:
    """Resolve a dtype name recorded in ``meta.json`` back to its NumPy type.

    Args:
        name: The ``"dtype"`` entry of a dataset's ``meta.json``.

    Returns:
        type: The NumPy type to memory-map the ``.bin`` files with.

    Raises:
        ValueError: If ``name`` is not one of the supported dtype names.
            Rejecting an unknown name is what stops a corpus from being read
            back at the wrong element width, which would silently yield
            garbage token ids rather than an error.
    """
    try:
        return _DTYPES_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unsupported token dtype '{name}': expected one of "
            f"{sorted(_DTYPES_BY_NAME)}."
        ) from None


class BinDataset:
    """Memory-mapped train/val token dataset backed by raw binary files.

    Owns the on-disk format of a processed dataset, both ways: :meth:`write`
    produces it and the constructor reads it back. A dataset directory holds:

    - ``train.bin`` and ``val.bin``: flat arrays of token ids, stored as
      ``uint16`` when the vocabulary fits in 16 bits and ``uint32`` otherwise;
    - ``meta.json``: the vocabulary size, the stored dtype, the tokenizer
      directory the ids were encoded with, and the token count of each split.

    The splits are exposed through NumPy memmaps, so the corpus never needs to
    be loaded into RAM.

    Sampling uses two fully independent pseudo-random streams (one per split)
    derived from a shared ``SeedSequence``, so sampling the validation split
    more or less frequently never perturbs the sequence of training batches.
    """

    @staticmethod
    def write(
        bin_dir: str | Path,
        splits: Mapping[str, Iterable[Iterable[int]]],
        vocab_size: int,
        tokenizer_dir: str | None = None,
    ) -> dict[str, int]:
        """Write a processed dataset directory that the constructor can open.

        Each split is consumed chunk by chunk and appended to its ``.bin``
        file before the next chunk is drawn, so peak memory stays proportional
        to one chunk and a corpus larger than RAM can be written. The element
        dtype is chosen from ``vocab_size``, and ``meta.json`` is written last.

        Args:
            bin_dir: Destination directory; created if missing. Existing
                ``.bin`` files and ``meta.json`` are overwritten.
            splits: Token ids for ``"train"`` and ``"val"``, each given as an
                iterable of chunks (lists or arrays of ids).
            vocab_size: Size of the vocabulary the ids come from.
            tokenizer_dir: Directory of the tokenizer the ids were encoded
                with, recorded so generation can pick the matching tokenizer.

        Returns:
            dict[str, int]: The number of tokens written to each split.

        Raises:
            ValueError: If ``splits`` does not hold exactly the ``"train"``
                and ``"val"`` keys, or if an id lies outside
                ``[0, vocab_size)``.
        """
        if set(splits) != set(SPLITS):
            raise ValueError(f"splits must have exactly the keys {SPLITS}, got {sorted(splits)}.")

        bin_dir = Path(bin_dir)
        bin_dir.mkdir(parents=True, exist_ok=True)
        dtype_name = _dtype_name_for_vocab(vocab_size)
        dtype = _DTYPES_BY_NAME[dtype_name]

        counts = {}
        for split in SPLITS:
            counts[split] = 0
            with open(bin_dir / f"{split}.bin", "wb") as file:
                for chunk in splits[split]:
                    ids = np.asarray(chunk, dtype=np.int64)
                    if ids.size == 0:
                        continue
                    out_of_range = ids[(ids < 0) | (ids >= vocab_size)]
                    if out_of_range.size:
                        raise ValueError(
                            f"split '{split}' holds token id {int(out_of_range[0])} "
                            f"outside [0, {vocab_size})."
                        )
                    ids.astype(dtype).tofile(file)
                    counts[split] += ids.size

        meta = {
            "vocab_size": vocab_size,
            "tokenizer_dir": tokenizer_dir,
            "dtype": dtype_name,
            "train_tokens": counts["train"],
            "val_tokens": counts["val"],
        }
        with open(bin_dir / "meta.json", "w") as file:
            json.dump(meta, file, indent=2)
        return counts

    @staticmethod
    def read_vocab_size(bin_dir: str | Path) -> int:
        """Read a dataset's vocabulary size without opening its splits.

        Lets a caller size the model before it knows the seed to open the
        dataset with.

        Args:
            bin_dir: A directory written by :meth:`write`.

        Returns:
            int: The vocabulary size recorded in ``meta.json``.

        Raises:
            FileNotFoundError: If ``bin_dir/meta.json`` does not exist.
        """
        with open(Path(bin_dir) / "meta.json") as file:
            return json.load(file)["vocab_size"]

    def __init__(self, bin_dir: str, seed: int | None = None):
        """Load dataset metadata and memory-map the train/val token arrays.

        Parses ``meta.json``, opens read-only memmaps over ``train.bin`` and
        ``val.bin``, and seeds two independent per-split RNGs. Both memmaps
        stay open for the lifetime of the dataset and are closed when the
        object is garbage collected.

        Args:
            bin_dir: A directory written by :meth:`write`.
            seed: Seed used to derive the train/val sampling RNGs. When
                ``None``, a fresh, unpredictable seed is drawn from the
                operating system, making runs non-reproducible but secure.

        Attributes:
            vocab_size (int): Size of the vocabulary the ids come from.
            tokenizer_dir (str | None): Directory of the tokenizer the ids
                were encoded with, or ``None`` if it was not recorded.

        Raises:
            FileNotFoundError: If ``bin_dir`` or any of the expected files
                does not exist.
            ValueError: If the recorded dtype is not ``"uint16"`` or
                ``"uint32"``.
        """
        bin_dir = Path(bin_dir)
        with open(bin_dir / "meta.json") as file:
            meta = json.load(file)
        self.vocab_size: int = meta["vocab_size"]
        self.tokenizer_dir: str | None = meta.get("tokenizer_dir")

        dtype = _dtype_from_name(meta["dtype"])
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

    def state_dict(self) -> dict:
        """Return the state of the per-split sampling RNGs.

        Returns:
            dict: The bit-generator state of each split's RNG, keyed by
            ``"train"`` and ``"val"``.
        """
        return {split: rng.bit_generator.state for split, rng in self._rng.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the per-split sampling RNGs from :meth:`state_dict`.

        Args:
            state: A mapping produced by :meth:`state_dict`.
        """
        for split, rng_state in state.items():
            self._rng[split].bit_generator.state = rng_state

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
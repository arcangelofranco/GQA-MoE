import json
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import BinDataset
from src.data.token_dtype import (
    MAX_UINT16_VOCAB_SIZE,
    dtype_for_vocab,
    dtype_from_name,
    dtype_name_for_vocab,
)


@pytest.mark.parametrize(
    "vocab_size, expected",
    [
        (1, "uint16"),
        (8_000, "uint16"),
        (MAX_UINT16_VOCAB_SIZE - 1, "uint16"),
        (MAX_UINT16_VOCAB_SIZE, "uint16"),
        (MAX_UINT16_VOCAB_SIZE + 1, "uint32"),
        (200_000, "uint32"),
    ],
)
def test_dtype_name_for_vocab_switches_at_the_uint16_limit(vocab_size: int, expected: str) -> None:
    """Verify the dtype name widens exactly one past the uint16 capacity.

    The boundary is off-by-one prone. Ids run over ``[0, vocab_size)``, so a
    vocabulary of exactly ``MAX_UINT16_VOCAB_SIZE`` tops out at id 65535 and
    still fits; only the size above it needs ``"uint32"``.

    Args:
        vocab_size: Vocabulary size under test.
        expected: The dtype name it must map to.
    """
    print(f"[dtype-name] vocab_size={vocab_size} -> {dtype_name_for_vocab(vocab_size)}")
    assert dtype_name_for_vocab(vocab_size) == expected


@pytest.mark.parametrize(
    "vocab_size",
    [
        2,
        300,
        8_000,
        MAX_UINT16_VOCAB_SIZE - 1,
        MAX_UINT16_VOCAB_SIZE,
        MAX_UINT16_VOCAB_SIZE + 1,
        200_000,
    ],
)
def test_chosen_dtype_represents_every_id_of_the_vocabulary(vocab_size: int) -> None:
    """Verify the largest id of a vocabulary round trips through the chosen dtype.

    This is the invariant that actually matters, and the one the threshold
    exists to protect: whatever width is picked, id ``vocab_size - 1`` must
    survive a store/load unchanged.

    Args:
        vocab_size: Vocabulary size under test.
    """
    dtype = dtype_for_vocab(vocab_size)
    largest_id = vocab_size - 1
    stored = int(np.array([largest_id], dtype=dtype)[0])
    print(f"[dtype-range] vocab={vocab_size} dtype={dtype} largest_id={largest_id} stored={stored}")
    assert stored == largest_id


def test_uint16_capacity_is_the_exact_point_where_ids_stop_fitting() -> None:
    """Verify the constant marks the real limit of uint16, not an arbitrary one.

    Anchors the threshold to the hardware fact rather than to a chosen number:
    the largest id of a ``MAX_UINT16_VOCAB_SIZE`` vocabulary is representable,
    and the very next id is not. If either half of this stops holding, the
    constant is wrong.
    """
    largest_fitting_id = MAX_UINT16_VOCAB_SIZE - 1
    assert int(np.array([largest_fitting_id], dtype=np.uint16)[0]) == largest_fitting_id

    with pytest.raises(OverflowError):
        np.array([MAX_UINT16_VOCAB_SIZE], dtype=np.uint16)


def test_dtype_for_vocab_matches_its_name() -> None:
    """Verify the type and the name helpers never disagree.

    The writer picks the NumPy type while ``meta.json`` records the name; if
    the two helpers diverged, a corpus would be read back at the wrong width.
    """
    for vocab_size in (300, 8_000, MAX_UINT16_VOCAB_SIZE, MAX_UINT16_VOCAB_SIZE + 1, 200_000):
        assert dtype_for_vocab(vocab_size) is dtype_from_name(dtype_name_for_vocab(vocab_size))


def test_dtype_from_name_rejects_an_unknown_name() -> None:
    """Verify an unrecognized dtype name raises instead of silently defaulting.

    Reading a ``uint32`` corpus as ``uint16`` yields plausible-looking but
    wrong token ids, so an unknown name must fail loudly.
    """
    with pytest.raises(ValueError, match="unsupported token dtype"):
        dtype_from_name("float32")


def test_bin_dataset_rejects_an_unknown_meta_dtype(tmp_path: Path) -> None:
    """Verify BinDataset surfaces a bad meta.json dtype as a ValueError.

    This is the documented contract of the constructor, and the reason the
    dtype table is shared with the writer.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    ids = np.arange(64, dtype=np.uint16)
    for name in ("train.bin", "val.bin"):
        np.memmap(tmp_path / name, dtype=np.uint16, mode="w+", shape=ids.shape)[:] = ids
    meta = {"vocab_size": 64, "dtype": "int8", "train_tokens": 64, "val_tokens": 64}
    with open(tmp_path / "meta.json", "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="unsupported token dtype"):
        BinDataset(str(tmp_path))

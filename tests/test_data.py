import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from src.data.dataset import BinDataset
from src.data.prepare_data import (
    download_corpus,
    encode_corpus_to_bin,
    iter_line_batches,
    prepare_data,
)
from src.data.tokenizer import Tokenizer
from src.data.train_tokenizer import train_bpe

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
""" * 30


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Tokenizer:
    """Trains a tiny BPE tokenizer on the toy corpus, shared across the module's tests.

    Args:
        tmp_path_factory: Pytest factory for a session-scoped temp directory.

    Returns:
        A `Tokenizer` loaded from the freshly trained vocab.
    """
    tmp_dir = tmp_path_factory.mktemp("data_tok")
    corpus_path = tmp_dir / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    tok_dir = tmp_dir / "tokenizer"
    train_bpe(str(corpus_path), vocab_size=300, out_dir=str(tok_dir))
    return Tokenizer(str(tok_dir))


def test_encode_corpus_to_bin_count_and_range(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Checks that encode_corpus_to_bin writes exactly n ids, all within the vocab range.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus_path = tmp_path / "mini.txt"
    corpus_path.write_text("Tom liked the dog.\nSpot liked the cat.\n")
    bin_path = tmp_path / "mini.bin"

    n = encode_corpus_to_bin(str(corpus_path), tiny_tokenizer, str(bin_path))

    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    print(f"[encode-count] n={n} len(arr)={len(arr)} max_id={arr.max()} vocab_size={tiny_tokenizer.get_vocab_size()}")
    assert len(arr) == n
    assert arr.max() < tiny_tokenizer.get_vocab_size()  # no out-of-vocabulary index


def test_encode_corpus_to_bin_inserts_eos_between_stories(
    tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    """Checks that encode_corpus_to_bin inserts one <eos> per line, with the last at file end.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus_path = tmp_path / "two_lines.txt"
    corpus_path.write_text("Tom liked the dog.\nSpot liked the cat.\n")
    bin_path = tmp_path / "two_lines.bin"

    encode_corpus_to_bin(str(corpus_path), tiny_tokenizer, str(bin_path))
    eos_id = tiny_tokenizer.token_to_id("<eos>")

    arr = np.array(np.memmap(bin_path, dtype=np.uint16, mode="r"))
    eos_positions = np.nonzero(arr == eos_id)[0]

    print(f"[eos-positions] eos_positions={eos_positions.tolist()} len(arr)={len(arr)}")
    assert len(eos_positions) == 2  # one per line/story
    assert eos_positions[-1] == len(arr) - 1  # the file's last token is <eos>
    assert 0 < eos_positions[0] < len(arr) - 1  # the first <eos> is in the middle, not at the edges


def test_encode_corpus_to_bin_skips_blank_lines(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Checks that blank lines are skipped and do not each get their own <eos>.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus_path = tmp_path / "with_blanks.txt"
    corpus_path.write_text("Tom liked the dog.\n\n\nSpot liked the cat.\n")
    bin_path = tmp_path / "with_blanks.bin"

    encode_corpus_to_bin(str(corpus_path), tiny_tokenizer, str(bin_path))
    eos_id = tiny_tokenizer.token_to_id("<eos>")
    arr = np.array(np.memmap(bin_path, dtype=np.uint16, mode="r"))

    # two non-blank lines -> two <eos>, not one per each of the file's 4 physical lines
    eos_count = (arr == eos_id).sum()
    print(f"[skip-blanks] eos_count={eos_count}")
    assert eos_count == 2


def test_iter_line_batches_groups_by_threshold(tmp_path: Path) -> None:
    """Checks that iter_line_batches groups lines into batches capped at batch_lines.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    corpus = tmp_path / "batches.txt"
    corpus.write_text("\n".join(f"story {i}" for i in range(25)))

    batches = list(iter_line_batches(str(corpus), batch_lines=10))

    print(f"[batches] sizes={[len(b) for b in batches]}")
    assert [len(b) for b in batches] == [10, 10, 5]
    assert batches[0][0] == "story 0"
    assert batches[-1][-1] == "story 24"


def test_iter_line_batches_skips_blank_lines(tmp_path: Path) -> None:
    """Checks that iter_line_batches drops blank lines from the batches.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    corpus = tmp_path / "with_blanks.txt"
    corpus.write_text("a\n\n\nb\nc\n")

    batches = list(iter_line_batches(str(corpus), batch_lines=2))
    print(f"[blanks] batches={batches}")
    assert batches == [["a", "b"], ["c"]]


def test_iter_line_batches_empty_or_all_blank(tmp_path: Path) -> None:
    """Checks that an empty or all-blank corpus yields no batches.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert list(iter_line_batches(str(empty))) == []

    blanks = tmp_path / "only_blanks.txt"
    blanks.write_text("\n\n\n")
    assert list(iter_line_batches(str(blanks))) == []


def test_encode_corpus_to_bin_feeds_real_batches(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Checks that encode_corpus_to_bin calls encode_batch with correctly sized batches.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus = tmp_path / "sized.txt"
    corpus.write_text("\n".join("Tom liked the dog." for _ in range(5)))

    with mock.patch.object(tiny_tokenizer, "encode_batch", wraps=tiny_tokenizer.encode_batch) as spy:
        encode_corpus_to_bin(str(corpus), tiny_tokenizer, str(tmp_path / "out.bin"), batch_lines=2)

    sizes = [len(call.args[0]) for call in spy.call_args_list]
    print(f"[spy] encode_batch called with batches of size {sizes}")
    assert sizes == [2, 2, 1]


def test_download_corpus_skips_if_already_present(tmp_path: Path) -> None:
    """Checks that download_corpus is a no-op when the output file already exists.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    out_path = tmp_path / "already_here.txt"
    out_path.write_text("pre-existing content\n")

    with mock.patch("src.data.prepare_data.load_dataset") as mocked:
        result = download_corpus(str(out_path), split="train")

    print(f"[skip-existing] load_dataset called={mocked.called} result={result!r}")
    mocked.assert_not_called()
    assert result == str(out_path)
    assert out_path.read_text() == "pre-existing content\n"


def test_download_corpus_writes_one_story_per_line(tmp_path: Path) -> None:
    """Checks that download_corpus writes one story per line, collapsing internal newlines.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    fake_rows = [
        {"text": "First story, single line."},
        {"text": "Second story\nwith an internal\nnewline."},
        {"text": "  \n"},
    ]

    with mock.patch("src.data.prepare_data.load_dataset", return_value=fake_rows) as mocked:
        out_path = tmp_path / "train.txt"
        download_corpus(str(out_path), split="train")

    mocked.assert_called_once_with("roneneldan/TinyStories", split="train")
    lines = out_path.read_text().splitlines()
    print(f"[one-story-per-line] lines={lines}")
    assert lines == [
        "First story, single line.",
        "Second story with an internal newline.",  # internal \n collapsed into a space
    ]



def test_prepare_data_end_to_end_with_local_corpus(tmp_path: Path) -> None:
    """Checks that prepare_data runs end-to-end on a local corpus and writes valid outputs.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.txt").write_text(CORPUS_TEXT)
    (raw_dir / "valid.txt").write_text("Tom liked the dog very much indeed.\n" * 5)

    prepare_data(
        raw_dir=str(raw_dir),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        bin_dir=str(tmp_path / "bin"),
        vocab_size=300,
    )

    with open(tmp_path / "bin" / "meta.json") as f:
        meta = json.load(f)
    print(f"[prepare-data] meta={meta}")
    assert meta["vocab_size"] == 300
    assert meta["dtype"] == "uint16"
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0
    assert (tmp_path / "bin" / "train.bin").exists()
    assert (tmp_path / "bin" / "val.bin").exists()
    assert (tmp_path / "tokenizer" / "vocab.json").exists()


def test_prepare_data_does_not_retrain_tokenizer_if_present(tmp_path: Path) -> None:
    """Checks that a second prepare_data call reuses an existing tokenizer instead of retraining.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.txt").write_text(CORPUS_TEXT)
    (raw_dir / "valid.txt").write_text("Tom liked the dog.\n" * 5)
    tokenizer_dir = tmp_path / "tokenizer"

    prepare_data(str(raw_dir), str(tokenizer_dir), str(tmp_path / "bin1"), vocab_size=300)
    vocab_mtime_1 = (tokenizer_dir / "vocab.json").stat().st_mtime_ns

    prepare_data(str(raw_dir), str(tokenizer_dir), str(tmp_path / "bin2"), vocab_size=300)
    vocab_mtime_2 = (tokenizer_dir / "vocab.json").stat().st_mtime_ns

    print(f"[no-retrain] vocab_mtime_1={vocab_mtime_1} vocab_mtime_2={vocab_mtime_2}")
    assert vocab_mtime_1 == vocab_mtime_2  # not retrained the second time


@pytest.fixture
def bin_dataset(tmp_path: Path, tiny_tokenizer: Tokenizer) -> BinDataset:
    """Builds a small BinDataset with encoded train/val splits for get_batch tests.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.

    Returns:
        A `BinDataset` backed by freshly encoded train.bin/val.bin.
    """
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text(CORPUS_TEXT)
    val_path.write_text("Tom liked the dog very much indeed every single day.\n" * 5)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    n_train = encode_corpus_to_bin(str(train_path), tiny_tokenizer, str(bin_dir / "train.bin"))
    n_val = encode_corpus_to_bin(str(val_path), tiny_tokenizer, str(bin_dir / "val.bin"))

    meta = {"vocab_size": tiny_tokenizer.get_vocab_size(), "dtype": "uint16",
            "train_tokens": n_train, "val_tokens": n_val}
    with open(bin_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return BinDataset(str(bin_dir))


def test_get_batch_shapes_and_dtype(bin_dataset: BinDataset) -> None:
    """Checks that get_batch returns int64 tensors shaped (batch_size, block_size).

    Args:
        bin_dataset: Small BinDataset fixture.
    """
    x, y = bin_dataset.get_batch("train", batch_size=4, block_size=16, device="cpu")
    print(f"[batch-shape] x.shape={tuple(x.shape)} y.shape={tuple(y.shape)} x.dtype={x.dtype} y.dtype={y.dtype}")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_get_batch_indices_in_vocab_range(bin_dataset: BinDataset, tiny_tokenizer: Tokenizer) -> None:
    """Checks that get_batch's returned ids stay within [0, vocab_size).

    Args:
        bin_dataset: Small BinDataset fixture.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    x, y = bin_dataset.get_batch("train", batch_size=8, block_size=16, device="cpu")
    print(f"[vocab-range] x range=[{x.min().item()}, {x.max().item()}] y range=[{y.min().item()}, {y.max().item()}] vocab_size={tiny_tokenizer.get_vocab_size()}")
    assert x.min() >= 0 and x.max() < tiny_tokenizer.get_vocab_size()
    assert y.min() >= 0 and y.max() < tiny_tokenizer.get_vocab_size()


class _FixedRNG:
    """Fake RNG that always returns the same fixed index array.

    Lets `get_batch` be driven to a deterministic, known sampling
    position instead of a random one.
    """

    def __init__(self, fixed_idx: np.ndarray):
        """Stores the index array to always return.

        Args:
            fixed_idx: Array returned unconditionally by `integers`.
        """
        self._fixed_idx = fixed_idx

    def integers(self, *args, **kwargs) -> np.ndarray:
        """Returns the fixed index array regardless of the requested range.

        Args:
            *args: Ignored; present to match `numpy.random.Generator.integers`.
            **kwargs: Ignored; present to match `numpy.random.Generator.integers`.

        Returns:
            The configured fixed index array.
        """
        return self._fixed_idx


def test_get_batch_y_is_x_shifted_by_one(
    bin_dataset: BinDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks that y is exactly x shifted one position ahead in the underlying data.

    Args:
        bin_dataset: Small BinDataset fixture.
        monkeypatch: Fixture used to pin the train split's RNG to a fixed index.
    """
    fixed_idx = np.array([3])
    monkeypatch.setitem(bin_dataset._rng, "train", _FixedRNG(fixed_idx))

    x, y = bin_dataset.get_batch("train", batch_size=1, block_size=8, device="cpu")

    data = bin_dataset._data["train"]
    expected_x = data[3:11].astype(np.int64)
    expected_y = data[4:12].astype(np.int64)
    print(f"[shift-by-one] x matches expected={np.array_equal(x[0].numpy(), expected_x)} y matches expected={np.array_equal(y[0].numpy(), expected_y)}")
    assert np.array_equal(x[0].numpy(), expected_x)
    assert np.array_equal(y[0].numpy(), expected_y)


def test_get_batch_unknown_split_raises(bin_dataset: BinDataset) -> None:
    """Checks that requesting an unknown split name raises ValueError.

    Args:
        bin_dataset: Small BinDataset fixture.
    """
    with pytest.raises(ValueError) as exc_info:
        bin_dataset.get_batch("test", batch_size=2, block_size=8, device="cpu")
    print(f"[unknown-split] raised={exc_info.value!r}")


def test_get_batch_block_size_too_large_raises(bin_dataset: BinDataset) -> None:
    """Checks that a block_size larger than the split's data raises ValueError.

    Args:
        bin_dataset: Small BinDataset fixture.
    """
    with pytest.raises(ValueError) as exc_info:
        bin_dataset.get_batch("val", batch_size=2, block_size=10**9, device="cpu")
    print(f"[block-size-too-large] raised={exc_info.value!r}")



def test_get_batch_same_seed_reproduces_train_sequence(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Checks that two datasets built with the same seed draw identical train batches.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text(CORPUS_TEXT)
    val_path.write_text("Tom liked the dog very much indeed every single day.\n" * 5)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    n_train = encode_corpus_to_bin(str(train_path), tiny_tokenizer, str(bin_dir / "train.bin"))
    n_val = encode_corpus_to_bin(str(val_path), tiny_tokenizer, str(bin_dir / "val.bin"))
    meta = {"vocab_size": tiny_tokenizer.get_vocab_size(), "dtype": "uint16",
            "train_tokens": n_train, "val_tokens": n_val}
    with open(bin_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    ds_a = BinDataset(str(bin_dir), seed=42)
    ds_b = BinDataset(str(bin_dir), seed=42)

    xa, ya = ds_a.get_batch("train", batch_size=4, block_size=16, device="cpu")
    xb, yb = ds_b.get_batch("train", batch_size=4, block_size=16, device="cpu")
    print(f"[same-seed] x equal={torch.equal(xa, xb)} y equal={torch.equal(ya, yb)}")
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_get_batch_val_calls_do_not_perturb_train_sequence(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Checks that calling get_batch("val", ...) does not perturb the train split's RNG sequence.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text(CORPUS_TEXT)
    val_path.write_text("Tom liked the dog very much indeed every single day.\n" * 5)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    n_train = encode_corpus_to_bin(str(train_path), tiny_tokenizer, str(bin_dir / "train.bin"))
    n_val = encode_corpus_to_bin(str(val_path), tiny_tokenizer, str(bin_dir / "val.bin"))
    meta = {"vocab_size": tiny_tokenizer.get_vocab_size(), "dtype": "uint16",
            "train_tokens": n_train, "val_tokens": n_val}
    with open(bin_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    ds_sparse_eval = BinDataset(str(bin_dir), seed=7)
    ds_dense_eval = BinDataset(str(bin_dir), seed=7)

    first_train_sparse = ds_sparse_eval.get_batch("train", batch_size=4, block_size=16, device="cpu")

    for _ in range(10):
        ds_dense_eval.get_batch("val", batch_size=4, block_size=16, device="cpu")
    first_train_dense = ds_dense_eval.get_batch("train", batch_size=4, block_size=16, device="cpu")

    print(
        f"[val-does-not-perturb-train] x equal={torch.equal(first_train_sparse[0], first_train_dense[0])} "
        f"y equal={torch.equal(first_train_sparse[1], first_train_dense[1])}"
    )
    assert torch.equal(first_train_sparse[0], first_train_dense[0])
    assert torch.equal(first_train_sparse[1], first_train_dense[1])
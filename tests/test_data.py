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
    """Train a tiny BPE tokenizer once and share it across the module's tests.

    Training a real tokenizer is the slowest fixture in this module, so it is
    scoped to the module and reused by every test instead of being rebuilt per
    test.

    Args:
        tmp_path_factory: Pytest factory for a session-scoped temp directory.

    Returns:
        Tokenizer: A :class:`Tokenizer` loaded from the freshly trained vocab.
    """
    tmp_dir = tmp_path_factory.mktemp("data_tok")
    corpus_path = tmp_dir / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    tok_dir = tmp_dir / "tokenizer"
    train_bpe(str(corpus_path), vocab_size=300, out_dir=str(tok_dir))
    return Tokenizer(str(tok_dir))


def test_encode_corpus_to_bin_count_and_range(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Verify encode_corpus_to_bin writes the reported count of in-vocab ids.

    The returned token count must match the bytes on disk exactly, and every id
    must lie inside ``[0, vocab_size)`` so the downstream model never indexes
    outside its embedding table.

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
    """Verify encode_corpus_to_bin inserts one <eos> per line, with the last at file end.

    Each story must be terminated exactly once so the model learns sequence
    boundaries; this also pins that the very last token of the file is the
    final ``<eos>`` and that interior terminators sit mid-file.

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
    """Verify that blank lines are skipped and do not each get their own <eos>.

    Empty lines carry no content, so emitting an ``<eos>`` per physical blank
    line would inject spurious sequence boundaries into the corpus. Only the
    non-blank stories may be terminated.

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


def test_encode_corpus_to_bin_output_is_independent_of_batch_lines(
    tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    """Verify batch_lines tunes memory only, never the bytes written.

    Encoding streams batch by batch, so the batch size decides how much is held
    in memory at once. It must not become part of the output contract: the same
    corpus has to produce byte-identical bins at any batch size.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)

    outputs = {}
    for batch_lines in (1, 2, 7, 10_000):
        bin_path = tmp_path / f"out_{batch_lines}.bin"
        n = encode_corpus_to_bin(
            str(corpus_path), tiny_tokenizer, str(bin_path), batch_lines=batch_lines
        )
        outputs[batch_lines] = (n, bin_path.read_bytes())

    counts = {bl: n for bl, (n, _) in outputs.items()}
    print(f"[batch-invariance] token_counts={counts}")
    assert len(set(counts.values())) == 1
    assert len({payload for _, payload in outputs.values()}) == 1


def test_encode_corpus_to_bin_writes_an_empty_file_for_a_blank_corpus(
    tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    """Verify a corpus with no usable lines yields a genuinely empty file.

    A corpus that produces no tokens must produce a zero-length file, not a
    partially written one: a file whose size is not a whole number of elements
    could not be memory-mapped back as a token array.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    corpus_path = tmp_path / "blank.txt"
    corpus_path.write_text("\n\n\n")
    bin_path = tmp_path / "blank.bin"

    n = encode_corpus_to_bin(str(corpus_path), tiny_tokenizer, str(bin_path))

    print(f"[blank-corpus] n={n} size={bin_path.stat().st_size}")
    assert n == 0
    assert bin_path.stat().st_size == 0


def test_iter_line_batches_groups_by_threshold(tmp_path: Path) -> None:
    """Verify iter_line_batches caps batches at batch_lines and keeps order.

    The batching policy feeds encoding: full batches are capped at
    ``batch_lines``, the final partial batch is still yielded, and the original
    file order is preserved end to end.

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
    """Verify iter_line_batches drops blank lines from the batches.

    Empty lines must be filtered before batching, so downstream encoding never
    sees placeholder entries that would otherwise produce empty token lists.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    corpus = tmp_path / "with_blanks.txt"
    corpus.write_text("a\n\n\nb\nc\n")

    batches = list(iter_line_batches(str(corpus), batch_lines=2))
    print(f"[blanks] batches={batches}")
    assert batches == [["a", "b"], ["c"]]


def test_iter_line_batches_empty_or_all_blank(tmp_path: Path) -> None:
    """Verify that an empty or all-blank corpus yields no batches.

    A corpus with nothing to encode must produce zero batches rather than an
    empty batch, which would otherwise surface as an odd one-token artifact in
    the encoded output.

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
    """Verify encode_corpus_to_bin feeds encode_batch correctly sized batches.

    Uses a spy to confirm the batched encoding path receives batches capped at
    ``batch_lines`` (``[2, 2, 1]``), not the raw line stream: the batching
    layer must actually drive the tokenizer's batch API.

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
    """Verify download_corpus is a no-op when the output file already exists.

    Pins the idempotency of the data pipeline: a re-run must not re-fetch the
    corpus from the network nor overwrite the existing file, or repeated
    invocations would waste bandwidth and break reproducibility.

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
    """Verify download_corpus writes one story per line, collapsing internal newlines.

    The corpus format contract is one story per line with internal newlines
    flattened to spaces and blank stories dropped; the tokenizer training and
    batching layers both depend on this exact shape.

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
    """Verify prepare_data runs end-to-end on a local corpus and writes valid outputs.

    Exercises the whole offline pipeline (skip-download, tokenizer training,
    encoding, metadata) against pre-seeded local corpora, checking that the
    produced bins and ``meta.json`` are consistent and self-describing.

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
    """Verify prepare_data reuses an existing tokenizer instead of retraining.

    Tokenizer training is the most expensive step of the pipeline, so a second
    invocation must leave ``vocab.json`` untouched (checked via its mtime).
    Otherwise every re-run would silently retrain and change the vocabulary.

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


def test_prepare_data_still_encodes_when_tokenizer_already_exists(tmp_path: Path) -> None:
    """Verify reusing a tokenizer does not skip encoding and metadata writing.

    Skipping tokenizer training must not skip the rest of the pipeline: a
    re-run pointed at a fresh ``bin_dir`` still has to encode both corpora and
    write ``meta.json``, otherwise the run silently produces nothing and
    training later fails on a missing dataset.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.txt").write_text(CORPUS_TEXT)
    (raw_dir / "valid.txt").write_text("Tom liked the dog.\n" * 5)
    tokenizer_dir = tmp_path / "tokenizer"

    prepare_data(str(raw_dir), str(tokenizer_dir), str(tmp_path / "bin1"), vocab_size=300)

    second_bin_dir = tmp_path / "bin2"
    prepare_data(str(raw_dir), str(tokenizer_dir), str(second_bin_dir), vocab_size=300)

    with open(second_bin_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"[reuse-tokenizer] meta={meta}")
    assert (second_bin_dir / "train.bin").exists()
    assert (second_bin_dir / "val.bin").exists()
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0

    # The reused tokenizer must produce byte-identical bins, not just any bins.
    for name in ("train.bin", "val.bin"):
        assert (tmp_path / "bin1" / name).read_bytes() == (second_bin_dir / name).read_bytes()


def _encode_bin_dir(tmp_path: Path, tokenizer: Tokenizer) -> Path:
    """Encode the toy corpus into a complete, loadable dataset directory.

    Produces the three files :class:`BinDataset` expects — ``train.bin``,
    ``val.bin``, and ``meta.json`` — from the shared toy corpus. The ``dtype``
    is written as a literal rather than derived from the production helper, so
    these tests keep asserting against a fixed expectation instead of against
    whatever the encoder currently computes.

    Args:
        tmp_path: Pytest-provided temporary directory to build the dataset in.
        tokenizer: Tokenizer used to encode both splits.

    Returns:
        Path: The ``bin_dir`` holding the encoded splits and their metadata.
    """
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text(CORPUS_TEXT)
    val_path.write_text("Tom liked the dog very much indeed every single day.\n" * 5)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    n_train = encode_corpus_to_bin(str(train_path), tokenizer, str(bin_dir / "train.bin"))
    n_val = encode_corpus_to_bin(str(val_path), tokenizer, str(bin_dir / "val.bin"))

    meta = {"vocab_size": tokenizer.get_vocab_size(), "dtype": "uint16",
            "train_tokens": n_train, "val_tokens": n_val}
    with open(bin_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return bin_dir


@pytest.fixture
def bin_dataset(tmp_path: Path, tiny_tokenizer: Tokenizer) -> BinDataset:
    """Build a small BinDataset with encoded splits for the get_batch tests.

    Encodes the toy corpus into real ``train.bin``/``val.bin`` files and wraps
    them in a :class:`BinDataset`, providing an isolated dataset for every
    ``get_batch`` test without repeating the encoding boilerplate.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.

    Returns:
        BinDataset: A dataset backed by freshly encoded train/val bins.
    """
    return BinDataset(str(_encode_bin_dir(tmp_path, tiny_tokenizer)))


def test_get_batch_shapes_and_dtype(bin_dataset: BinDataset) -> None:
    """Verify get_batch returns int64 tensors shaped (batch_size, block_size).

    Both the dtype (int64, required by ``nn.Embedding``) and the exact
    batch/sequence geometry are part of the training data contract and must not
    drift.

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
    """Verify get_batch's returned ids stay within [0, vocab_size).

    Sampled ids outside the embedding table would crash training on the first
    lookup; this guards the range invariant for both inputs and targets.

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

    Lets ``get_batch`` be driven to a deterministic, known sampling position
    instead of a random one, which is required to assert the exact shift
    relationship between inputs and targets.
    """

    def __init__(self, fixed_idx: np.ndarray):
        """Store the index array to always return.

        Args:
            fixed_idx: Array returned unconditionally by ``integers``.
        """
        self._fixed_idx = fixed_idx

    def integers(self, *args, **kwargs) -> np.ndarray:
        """Return the fixed index array regardless of the requested range.

        Args:
            *args: Ignored; present to match
                ``numpy.random.Generator.integers``.
            **kwargs: Ignored; present to match
                ``numpy.random.Generator.integers``.

        Returns:
            np.ndarray: The configured fixed index array.
        """
        return self._fixed_idx


def test_get_batch_y_is_x_shifted_by_one(
    bin_dataset: BinDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that y is exactly x shifted one position ahead in the underlying data.

    Uses a deterministic RNG stub to pin the sampling offset, then checks the
    windowed slices byte-for-byte against the source array. This pins the
    next-token prediction contract that the loss function relies on.

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
    """Verify that requesting an unknown split name raises ValueError.

    A typo in the split name must fail with a clear error instead of silently
    reading from an unexpected split or crashing on a missing memmap.

    Args:
        bin_dataset: Small BinDataset fixture.
    """
    with pytest.raises(ValueError) as exc_info:
        bin_dataset.get_batch("test", batch_size=2, block_size=8, device="cpu")
    print(f"[unknown-split] raised={exc_info.value!r}")


def test_get_batch_block_size_too_large_raises(bin_dataset: BinDataset) -> None:
    """Verify that a block_size larger than the split's data raises ValueError.

    A window longer than the available tokens cannot be sliced; the dataset
    must reject it up front with an informative error rather than returning
    undersized or empty sequences.

    Args:
        bin_dataset: Small BinDataset fixture.
    """
    with pytest.raises(ValueError) as exc_info:
        bin_dataset.get_batch("val", batch_size=2, block_size=10**9, device="cpu")
    print(f"[block-size-too-large] raised={exc_info.value!r}")



def test_get_batch_same_seed_reproduces_train_sequence(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Verify that two datasets built with the same seed draw identical batches.

    Reproducibility depends on the seed reaching the sampling RNGs; two
    datasets seeded identically must produce bit-identical training batches so
    runs can be compared or replayed deterministically.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    bin_dir = _encode_bin_dir(tmp_path, tiny_tokenizer)

    ds_a = BinDataset(str(bin_dir), seed=42)
    ds_b = BinDataset(str(bin_dir), seed=42)

    xa, ya = ds_a.get_batch("train", batch_size=4, block_size=16, device="cpu")
    xb, yb = ds_b.get_batch("train", batch_size=4, block_size=16, device="cpu")
    print(f"[same-seed] x equal={torch.equal(xa, xb)} y equal={torch.equal(ya, yb)}")
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_get_batch_val_calls_do_not_perturb_train_sequence(tmp_path: Path, tiny_tokenizer: Tokenizer) -> None:
    """Verify val sampling never perturbs the train split's RNG sequence.

    Validation runs interleave with training, so the two RNG streams must be
    fully independent: the training batch order cannot depend on how often or
    when validation batches are drawn.

    Args:
        tmp_path: Pytest-provided temporary directory.
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    bin_dir = _encode_bin_dir(tmp_path, tiny_tokenizer)

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
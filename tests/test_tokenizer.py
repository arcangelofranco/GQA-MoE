from pathlib import Path

import pytest

from src.data.tokenizer import Tokenizer
from src.data.train_tokenizer import train_bpe

CORPUS_TEXT = """Once upon a time, there was a little cat named Tom.
Tom liked to play in the garden with his ball.
One day, Tom saw a big dog. The dog was friendly.
They played together all afternoon and became best friends.
Once upon a time, there was a small dog named Spot.
Spot loved to run and jump in the sunny park every day.
The sun was warm and the grass was green and soft.
""" * 20


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Tokenizer:
    """Trains a small BPE tokenizer on the toy corpus, shared across the module's tests.

    Training once per module keeps the tests fast while still exercising the
    real :func:`train_bpe` pipeline.

    Args:
        tmp_path_factory: Pytest factory for a session-scoped temp directory.

    Returns:
        A :class:`Tokenizer` loaded from the freshly trained vocab.
    """
    tmp_dir = tmp_path_factory.mktemp("tok")
    corpus_path = tmp_dir / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    tok_dir = tmp_dir / "tokenizer"
    train_bpe(str(corpus_path), vocab_size=300, out_dir=str(tok_dir))
    return Tokenizer(str(tok_dir))


def test_vocab_size_matches_requested(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that the trained vocabulary respects the requested size.

    Guards against off-by-one errors in the BPE merge count, which would leave
    the vocab undersized or oversized relative to what the model config expects.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    vocab_size = tiny_tokenizer.get_vocab_size()
    print(f"[vocab-size] vocab_size={vocab_size} expected=300")
    assert vocab_size == 300


@pytest.mark.parametrize("sentence", [
    "Tom liked to play in the garden.",
    "ONE DAY, the dog was VERY friendly!",
    "Spot, Tom, and the ball -- all together?",
])
def test_roundtrip(tiny_tokenizer: Tokenizer, sentence: str) -> None:
    """Verifies that ``encode()`` followed by ``decode()`` reproduces the original sentence.

    Round-tripping is the core correctness contract of a tokenizer: losses here
    would corrupt any text the model generates or consumes.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
        sentence: Sentence to round-trip.
    """
    ids = tiny_tokenizer.encode(sentence)
    decoded = tiny_tokenizer.decode(ids)
    print(f"[roundtrip] sentence={sentence!r} decoded={decoded!r}")
    assert decoded == sentence


def test_encode_batch_matches_encode_one_by_one(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that ``encode_batch()`` is consistent with calling ``encode()`` per sentence.

    Batch encoding is a convenience path; it must not silently reorder or
    re-tokenize the input sentences differently than the one-by-one path.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    sentences = ["Tom liked to play.", "The dog was friendly."]
    batch_ids = tiny_tokenizer.encode_batch(sentences)
    solo_ids = [tiny_tokenizer.encode(s) for s in sentences]
    print(f"[encode-batch] batch_ids={batch_ids} solo_ids={solo_ids}")
    assert batch_ids == solo_ids


def test_special_tokens_present(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that all expected special tokens resolve to an id without raising.

    Special tokens are required by training and inference pipelines, so a vocab
    missing any of them would break padding, masking, and sequence boundaries.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    for tok in ["<pad>", "<unk>", "<bos>", "<eos>"]:
        token_id = tiny_tokenizer.token_to_id(tok)  # must not raise KeyError
        print(f"[special-tokens] {tok}={token_id}")


def test_decode_strips_special_tokens(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that ``decode()`` omits special tokens such as ``<eos>`` from the output.

    Leaking special tokens into generated text would pollute user-facing output,
    so decoding must filter them out.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    eos_id = tiny_tokenizer.token_to_id("<eos>")
    ids = tiny_tokenizer.encode("Tom liked to play.") + [eos_id]
    decoded = tiny_tokenizer.decode(ids)
    print(f"[strip-special] decoded={decoded!r}")
    assert "<eos>" not in decoded


def test_unknown_special_token_raises_key_error(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that resolving an unregistered special token raises ``KeyError``.

    Failing loudly on unknown tokens prevents silent mis-tokenization that would
    otherwise be hard to trace in a trained model.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.

    Raises:
        KeyError: As expected when querying an unknown token.
    """
    with pytest.raises(KeyError) as exc_info:
        tiny_tokenizer.token_to_id("<not_a_real_token>")
    print(f"[unknown-token] raised={exc_info.value!r}")


def test_reloaded_tokenizer_matches_freshly_trained(tmp_path: Path) -> None:
    """Verifies that reloading a saved tokenizer from disk encodes identically to the original.

    Deterministic round-tripping across reloads is required so a tokenizer
    trained once can be persisted and reused consistently by later processes.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(CORPUS_TEXT)
    tok_dir = tmp_path / "tokenizer"
    train_bpe(str(corpus_path), vocab_size=300, out_dir=str(tok_dir))

    tok_a = Tokenizer(str(tok_dir))
    tok_b = Tokenizer(str(tok_dir))  # reloaded a second time from the same files

    sentence = "Tom and Spot played in the garden together."
    ids_a, ids_b = tok_a.encode(sentence), tok_b.encode(sentence)
    print(f"[reload] ids_a={ids_a} ids_b={ids_b} equal={ids_a == ids_b}")
    assert ids_a == ids_b


def test_missing_tokenizer_dir_raises_clear_error(tmp_path: Path) -> None:
    """Verifies that loading a tokenizer from a nonexistent directory raises ``FileNotFoundError``.

    A clear early error helps users detect stale or misconfigured model paths
    instead of failing later with a confusing state.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Raises:
        FileNotFoundError: As expected when the tokenizer directory is missing.
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        Tokenizer(str(tmp_path / "does_not_exist"))
    print(f"[missing-dir] raised={exc_info.value!r}")


def test_compression_beats_char_level(tiny_tokenizer: Tokenizer) -> None:
    """Verifies that BPE encoding uses meaningfully fewer tokens than characters.

    Compression below the character level is the whole point of BPE: without it
    the tokenizer would not shorten sequences for the transformer.

    Args:
        tiny_tokenizer: Shared tiny tokenizer fixture.
    """
    sample = "Tom and Spot played together in the sunny garden all afternoon."
    n_chars = len(sample)
    n_tokens = len(tiny_tokenizer.encode(sample))
    print(f"[compression] n_chars={n_chars} n_tokens={n_tokens} ratio={n_tokens / n_chars:.3f}")
    assert n_tokens < n_chars * 0.75
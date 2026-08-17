from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from src.data.tokenizer import SpecialTokens

def train_bpe(corpus_path: str, vocab_size: int, out_dir: str) -> None:
    """Train a byte-level BPE tokenizer on a text corpus and save it to disk.

    Learns the byte-pair merge rules from the raw corpus using the Hugging Face
    ``tokenizers`` library and persists the result as the ``vocab.json`` and
    ``merges.txt`` artifact pair expected by :class:`src.data.tokenizer.Tokenizer`.
    The standard special tokens are reserved up front so they occupy stable,
    dedicated vocabulary slots.

    Note: the trained vocabulary includes the special tokens, so the final
    number of learned BPE merges is ``vocab_size - len(SpecialTokens)``; callers
    relying on an exact count should query the saved tokenizer via
    :meth:`Tokenizer.get_vocab_size` rather than assume equality with
    ``vocab_size``.

    Args:
        corpus_path: Path to a plain-text training corpus, one story/sample per
            line. The file must be UTF-8 encoded.
        vocab_size: Target vocabulary size (including special tokens). Must be
            a positive integer; values that are too small for the requested
            merges may yield a smaller final vocabulary.
        out_dir: Directory to write ``vocab.json`` and ``merges.txt`` into.
            Created recursively if it does not exist.

    Returns:
        None: Tokenizer artifacts are persisted to ``out_dir``; nothing is
        returned to the caller.

    Raises:
        FileNotFoundError: If ``corpus_path`` does not exist or is empty.
        ValueError: If ``vocab_size`` is not positive or the corpus contains
            too few tokens to reach the requested vocabulary.
        OSError: If ``out_dir`` cannot be created or the artifacts cannot be
            written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SpecialTokens
    )
    tokenizer.save_model(str(out_dir))
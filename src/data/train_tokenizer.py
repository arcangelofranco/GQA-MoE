from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from src.data.tokenizer import SpecialTokens

def train_bpe(corpus_path: str, vocab_size: int, out_dir: str) -> None:
    """Train a byte-level BPE tokenizer on a text corpus and save it to disk.

    Args:
        corpus_path: Path to a plain-text training corpus.
        vocab_size: Target vocabulary size.
        out_dir: Directory to write vocab.json and merges.txt to.
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
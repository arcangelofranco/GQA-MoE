from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

SpecialTokens = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Tokenizer:
    """Thin wrapper around a trained ByteLevel BPE tokenizer.

    Loads the ``vocab.json``/``merges.txt`` pair produced by
    :func:`src.data.train_tokenizer.train_bpe` and restores the reserved
    special tokens (``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``), which the
    underlying :class:`tokenizers.ByteLevelBPETokenizer.from_file` does not
    persist on its own.

    The wrapper normalizes the lower-level ``tokenizers`` API into the small,
    dataset-specific surface used by the rest of the codebase (encode/decode,
    vocabulary size, and token lookups), so callers never touch the raw
    library directly.
    """

    def __init__(self, tokenizer_dir: str):
        """Load a trained BPE tokenizer from disk.

        Validates that both required artifacts exist, then restores the
        tokenizer and registers the standard special tokens so they occupy
        their reserved vocabulary slots.

        Args:
            tokenizer_dir: Directory containing ``vocab.json`` and
                ``merges.txt`` as written by the training pipeline.

        Raises:
            FileNotFoundError: If ``vocab.json`` or ``merges.txt`` is missing
                from ``tokenizer_dir``.
        """
        tokenizer_dir = Path(tokenizer_dir)
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found in {tokenizer_dir} (expected vocab.json and merges.txt)"    
            )
        self._tok = ByteLevelBPETokenizer.from_file(str(vocab_path), str(merges_path))
        self._tok.add_special_tokens(SpecialTokens)

    def encode(self, text: str) -> list[int]:
        """Encode a single string into token ids.

        Args:
            text: Raw text to encode.

        Returns:
            list[int]: The list of token ids representing ``text``. The output
            does not include any special token by default.
        """
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of strings into token ids.

        Equivalent to calling :meth:`encode` per text but performed in a single
        library pass, which is substantially faster than a Python loop thanks
        to the underlying Rust implementation.

        Args:
            texts: List of raw texts to encode.

        Returns:
            list[list[int]]: One list of token ids per input text, in the same
            order as ``texts``.
        """
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Inverse of :meth:`encode`. The underlying library handles the
        byte-level merge reconstruction and filters out special tokens, so the
        returned string is the plain human-readable text.

        Args:
            ids: Token ids to decode.

        Returns:
            str: The decoded text, with special tokens filtered out.
        """
        return self._tok.decode(ids)

    def get_vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary.

        Includes both the learned BPE merges and the reserved special tokens,
        so this value can be used to size the model's embedding/output head.

        Returns:
            int: The total vocabulary size.
        """
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """Look up the id of a token, e.g. a special token like ``"<eos>"``.

        Wraps the underlying library lookup to convert its ``None`` result
        (token absent from the vocabulary) into an explicit, informative
        exception so callers never have to check for ``None``.

        Args:
            token: Token string to look up.

        Returns:
            int: The token's integer id in the vocabulary.

        Raises:
            KeyError: If ``token`` is not present in the vocabulary.
        """
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise KeyError(
                f"'{token}' is not in this tokenizer's vocabulary."
            )
        return token_id


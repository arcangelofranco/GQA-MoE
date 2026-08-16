from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

SpecialTokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

class Tokenizer:
    """Thin wrapper around a trained ByteLevel BPE tokenizer.

    Loads a vocab.json/merges.txt pair produced by `train_bpe` and restores
    the special tokens, which `from_file` does not persist.
    """

    def __init__(self, tokenizer_dir: str):
        """Load a trained BPE tokenizer from disk.

        Args:
            tokenizer_dir: Directory containing vocab.json and merges.txt.

        Raises:
            FileNotFoundError: If vocab.json or merges.txt is missing.
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
            text: Text to encode.

        Returns:
            List of token ids.
        """
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of strings into token ids.

        Args:
            texts: List of texts to encode.

        Returns:
            List of token id lists, one per input text, in the same order.
        """
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded text, with special tokens filtered out.
        """
        return self._tok.decode(ids)

    def get_vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary.

        Returns:
            The vocabulary size.
        """
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """Look up the id of a token, e.g. a special token like "<eos>".

        Args:
            token: Token string to look up.

        Returns:
            The token's id.

        Raises:
            KeyError: If the token is not in the vocabulary.
        """
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise KeyError(
                f"'{token}' is not in this tokenizer's vocabulary."
            )
        return token_id


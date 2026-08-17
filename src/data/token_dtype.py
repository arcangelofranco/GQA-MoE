import numpy as np

# Largest vocabulary whose ids all fit in a ``uint16``, inclusive.
#
# A ``uint16`` (2^16 values) holds 0..65535. Ids run over ``[0, vocab_size)``,
# so the largest id of a vocabulary of this size is 65535 -- exactly
# representable. The comparison below is therefore ``<=``: a 65536-token
# vocabulary still encodes in half the disk space of a ``uint32``, and only
# 65537 and above need to widen.
MAX_UINT16_VOCAB_SIZE = 65536

# The dtype names that may appear in ``meta.json``, and what they decode to.
_DTYPES_BY_NAME: dict[str, type] = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def dtype_name_for_vocab(vocab_size: int) -> str:
    """Pick the narrowest dtype name that can hold every id of a vocabulary.

    Chooses ``"uint16"`` whenever the ids fit, halving the size of the encoded
    corpus on disk relative to ``"uint32"``, and falls back to ``"uint32"`` for
    vocabularies too large to index with 16 bits.

    Args:
        vocab_size: Number of distinct token ids the tokenizer can emit. Ids
            are assumed to run over ``[0, vocab_size)``.

    Returns:
        str: ``"uint16"`` when ``vocab_size`` is at most
        :data:`MAX_UINT16_VOCAB_SIZE`, otherwise ``"uint32"``. This is the
        value to record in ``meta.json``.
    """
    return "uint16" if vocab_size <= MAX_UINT16_VOCAB_SIZE else "uint32"


def dtype_for_vocab(vocab_size: int) -> type:
    """Pick the NumPy type that encodes a vocabulary's ids most compactly.

    The type counterpart of :func:`dtype_name_for_vocab`, for the encoding
    side, which needs the actual NumPy type rather than its recorded name.

    Args:
        vocab_size: Number of distinct token ids the tokenizer can emit.

    Returns:
        type: ``numpy.uint16`` or ``numpy.uint32``.
    """
    return _DTYPES_BY_NAME[dtype_name_for_vocab(vocab_size)]


def dtype_from_name(name: str) -> type:
    """Resolve a dtype name recorded in meta.json back to its NumPy type.

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

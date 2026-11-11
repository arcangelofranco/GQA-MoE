import argparse

import torch

from src.data.tokenizer import Tokenizer
from src.runtime.checkpoint import load_for_inference
from src.runtime.generator import TextGenerator
from src.runtime.sampler import SamplingPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface for text generation.

    Builds an :class:`argparse.ArgumentParser` describing every generation
    hyper parameter exposed to the user and parses the given arguments into a
    lightweight :class:`argparse.Namespace`. Sampling parameters left
    unspecified by the user are ``None`` so the runtime can apply its defaults.

    Args:
        argv: Optional sequence of raw command-line arguments to parse.
            When ``None``, :func:`argparse.ArgumentParser.parse_args` falls back
            to ``sys.argv[1:]``. Providing this parameter explicitly makes the
            function trivially unit-testable.

    Returns:
        argparse.Namespace: Populated namespace exposing the following
        attributes:

        - ``checkpoint`` (str): Path to the ``.pt`` checkpoint file.
        - ``tokenizer_dir`` (str | None): Directory containing the trained
          tokenizer. When ``None``, the tokenizer recorded in the checkpoint
          is used, falling back to ``data/tokenizer``.
        - ``prompt`` (str): Seed text the generation starts from.
        - ``max_new_tokens`` (int): Maximum number of tokens to generate.
        - ``temperature`` (float): Softmax temperature controlling randomness.
        - ``top_k`` (int | None): Top-k sampling cutoff, ``None`` disables it.
        - ``top_p`` (float | None): Nucleus sampling cutoff, ``None`` disables it.
        - ``device`` (str): Inference device, defaulting to CUDA when
          available, otherwise CPU.

    Raises:
        SystemExit: On ``--help`` or invalid arguments, consistent with
            standard argparse behaviour.
    """
    parser = argparse.ArgumentParser(description="Generate text using a GQA-MoE checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="path to a `.pt` checkpoint file saved by the Trainer")
    parser.add_argument(
        "--tokenizer-dir", default=None,
        help="tokenizer directory; defaults to the one recorded in the checkpoint, then data/tokenizer",
    )
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    """CLI entry point: load the model and tokenizer, then generate text.

    Orchestrates the full inference pipeline: it parses the command-line
    arguments, restores the ``Transformer`` from the checkpoint, loads the
    ``Tokenizer`` (``--tokenizer-dir`` if given, otherwise the one the
    training data was encoded with, as recorded in the checkpoint), validates that the vocabulary sizes agree, wraps
    everything in a :class:`TextGenerator` with the requested sampling policy,
    and finally emits the generated continuation.

    The generated text is written to stdout and also returned to the caller so
    the function can be reused programmatically or embedded in tests without
    relying on process exit codes.

    Args:
        argv: Optional sequence of raw command-line arguments forwarded to
            :func:`parse_args`. When ``None``, ``sys.argv`` is used.

    Returns:
        str: The full generated text (prompt plus continuation), also printed
        to stdout.

    Raises:
        ValueError: If the vocabulary size reported by the tokenizer does not
            match ``model_cfg.vocab_size`` saved in the checkpoint. This guards
            against loading a model that was trained on a different tokenizer.
        FileNotFoundError: If the checkpoint or tokenizer directory cannot be
            found.
        RuntimeError: If model weights cannot be loaded or an inference device
            becomes unavailable.
    """
    args = parse_args(argv)
    model, model_cfg, saved_tokenizer_dir = load_for_inference(args.checkpoint, args.device)
    tokenizer = Tokenizer(args.tokenizer_dir or saved_tokenizer_dir or "data/tokenizer")

    model_cfg.require_vocab_size(tokenizer.get_vocab_size(), source="tokenizer")

    generator = TextGenerator(
        model,
        tokenizer,
        SamplingPolicy(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p),
        device=args.device,
    )

    text = generator.generate(args.prompt, args.max_new_tokens)
    print(text)
    return text


if __name__ == "__main__":
    main()

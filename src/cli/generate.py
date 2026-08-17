import argparse

import torch

from src.config import ModelConfig
from src.data.tokenizer import Tokenizer
from src.model.transformer import Transformer
from src.runtime.generator import TextGenerator
from src.runtime.sampler import SamplingPolicy


def load_model_for_generation(checkpoint_path: str, device: str) -> tuple[Transformer, ModelConfig]:
    """Load a checkpoint and reconstruct the model ready for inference.

    This is the single entry point for restoring a trained ``Transformer`` from
    disk. The checkpoint is first read with ``map_location="cpu"`` to avoid GPU
    OOM errors during deserialization, then the resulting state dict is moved to
    the target ``device``. The model is switched to evaluation mode, which
    disables dropout and any other training-only behaviour (e.g. MoE auxiliary
    losses) before generation begins.

    Args:
        checkpoint_path: Absolute or relative path to the ``.pt`` checkpoint
            saved by the ``Trainer``. The file must contain both a ``"model"``
            key (the raw state dict) and a ``"model_cfg"`` key (a dictionary
            compatible with :meth:`ModelConfig.from_dict`).
        device: PyTorch device to move the model to, e.g. ``"cpu"`` or
            ``"cuda"``. Passing a device with insufficient memory after a large
            checkpoint has been materialised on CPU may still raise a CUDA
            out-of-memory error.

    Returns:
        tuple[Transformer, ModelConfig]: A pair ``(model, model_cfg)`` where
        ``model`` is a fully restored ``Transformer`` in evaluation mode and
        already transferred to ``device``, and ``model_cfg`` is the
        configuration object used to reconstruct it. Callers can use
        ``model_cfg`` to cross-check dimensions against the tokenizer.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        KeyError: If the checkpoint is missing the ``"model_cfg"`` or
            ``"model"`` keys.
        RuntimeError: If the state dict does not match the architecture implied
            by ``model_cfg`` (e.g. checkpoint trained with different hyper
            parameters).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


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
        - ``tokenizer_dir`` (str): Directory containing the trained tokenizer.
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
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
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
    matching ``Tokenizer``, validates that the vocabulary sizes agree, wraps
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
    model, model_cfg = load_model_for_generation(args.checkpoint, args.device)
    tokenizer = Tokenizer(args.tokenizer_dir)

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

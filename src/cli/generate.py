import argparse

import torch

from src.config import ModelConfig
from src.data.tokenizer import Tokenizer
from src.model.transformer import Transformer
from src.runtime.generator import TextGenerator
from src.runtime.sampler import SamplingPolicy


def load_model_for_generation(checkpoint_path: str, device: str) -> tuple[Transformer, ModelConfig]:
    """Load a checkpoint and reconstruct the model ready for inference.

    Args:
        checkpoint_path: Path to the .pt file saved by the Trainer, containing the "model_cfg" and "model" keys.
        device: PyTorch device to move the model to (e.g. "cpu" or "cuda").

    Returns:
        A tuple (model, model_cfg) with the model in eval mode, already moved
        to device, and the configuration used to build it.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Defines and parses the command-line arguments for generation.

    Args:
        argv: List of arguments to parse. If None, sys.argv is used (argparse's default behavior).

    Returns:
        Namespace containing the parsed arguments (checkpoint, tokenizer_dir,
        prompt, max_new_tokens, temperature, top_k, top_p, device).
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
    """CLI entry point: loads the model and tokenizer and generates text from the prompt.

    Args:
        argv: List of command-line arguments to parse. If None,
            sys.argv is used.

    Returns:
        The generated text, which is also printed to stdout.

    Raises:
        ValueError: If the tokenizer's vocab_size does not match the one in the model_cfg saved in the checkpoint.
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

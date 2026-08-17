from dataclasses import dataclass, asdict, field
from typing import TypeVar

import torch
import yaml

T = TypeVar("T", bound="ConfigMixin")


def _default_device() -> str:
    """Pick the best torch device available right now.

    Returns:
        str: ``"cuda"`` if a GPU is visible, ``"cpu"`` otherwise. Resolved
        once per :class:`RuntimeConfig` instance rather than once at import
        time, so the value reflects the machine the config is actually built
        on.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class ConfigMixin:
    """Adds dict (de)serialization to a frozen dataclass config.

    Provides the round-trip pair :meth:`to_dict` / :meth:`from_dict` shared by
    every config dataclass in this module. Subclasses may override
    :meth:`from_dict` when they need type coercion beyond what ``cls(**d)``
    performs (see :class:`RuntimeConfig`).
    """

    def to_dict(self) -> dict:
        """Convert this config to a plain dict.

        Returns:
            dict: A flat mapping of field name to value, using
            :func:`dataclasses.asdict` so nested dataclass fields are
            recursively converted.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Build a config instance from a dict of field values.

        Args:
            d: Dict of field name to value, e.g. produced by :meth:`to_dict`
                or parsed from a YAML file.

        Returns:
            T: A new instance of the config class.
        """
        return cls(**d)


@dataclass(frozen=True)
class ModelConfig(ConfigMixin):
    """Transformer architecture configuration.

    Describes the full network shape consumed by the model constructor: the
    attention geometry (heads, KV-head groups, head dimension), the MoE
    feed-forward layout (routed and shared experts), and the position/token
    knobs (sequence length, RoPE, RMSNorm epsilon, embedding tying).

    Attributes:
        vocab_size: Tokenizer vocabulary size.
        n_layers: Number of Transformer blocks.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (GQA groups).
        head_dim: Dimension of each attention head.
        expert_intermediate: Hidden dim of each routed MoE expert.
        shared_intermediate: Hidden dim of the always-on shared expert.
        n_experts: Number of routed experts.
        top_k: Number of experts each token is routed to.
        max_seq_len: Maximum sequence length (RoPE table size).
        rms_norm_eps: Epsilon for RMSNorm layers.
        rope_theta: Base for the RoPE inverse frequency progression.
        aux_loss_coeff: Weight of the MoE load-balancing auxiliary loss.
        tie_embeddings: Whether to tie lm_head weights to the token embedding.
    """

    vocab_size: int
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    expert_intermediate: int = 256
    shared_intermediate: int = 688
    n_experts: int = 4
    top_k: int = 2
    max_seq_len: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    aux_loss_coeff: float = 0.01
    tie_embeddings: bool = True

    @property
    def hidden_dim(self) -> int:
        """Model hidden dimension, derived as n_heads * head_dim.

        Returns:
            int: The embedding/feature dimension used by the blocks.
        """
        return self.n_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Number of query heads sharing each KV head in GQA.

        Returns:
            int: The number of query heads per key/value head (>= 1).
        """
        return self.n_heads // self.n_kv_heads

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """Check architectural invariants required by GQA, MoE routing, and RoPE.

        Raises:
            ValueError: If ``n_heads`` is not a multiple of ``n_kv_heads``,
                ``top_k`` is outside ``[1, n_experts]``, ``head_dim`` is odd,
                or ``vocab_size`` is not positive.
        """
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of "
                f"n_kv_heads ({self.n_kv_heads}) for GQA."
            )
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError(
                f"top_k ({self.top_k}) is out of range [1, n_experts={self.n_experts}]."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for rotate_half.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive and match the trained tokenizer.")

    def require_vocab_size(self, actual: int, source: str) -> None:
        """Check that an external artifact's vocabulary matches this model's.

        The token ids in a dataset, and the ids a tokenizer emits, only mean
        anything to a model whose embedding table has the same size. Both the
        training and the generation entry points need this check, so the
        comparison and its message live here instead of at each call site.

        Args:
            actual: Vocabulary size reported by the artifact.
            source: What produced it, named in the error message, e.g.
                ``"dataset"`` or ``"tokenizer"``.

        Raises:
            ValueError: If ``actual`` does not equal this config's
                ``vocab_size``.
        """
        if actual != self.vocab_size:
            raise ValueError(
                f"{source} vocab_size ({actual}) does not match the model's "
                f"vocab_size ({self.vocab_size}): both must come from the same "
                "trained tokenizer."
            )


@dataclass(frozen=True)
class TrainConfig(ConfigMixin):
    """Training hyperparameters.

    Groups every knob of the optimization loop: batch geometry, the token
    budget (from which the total step count is derived), the LR curve, weight
    decay and clipping, the validation cadence, and the seed.

    Attributes:
        batch_size: Number of sequences per batch.
        block_size: Sequence length (context window) per example.
        target_tokens: Total training token budget; derives max_steps.
        warmup_steps: Number of linear LR warmup steps.
        max_lr: Peak learning rate, reached at the end of warmup.
        min_lr: Minimum learning rate, reached at the end of cosine decay.
        weight_decay: AdamW weight decay for >=2D parameters.
        grad_clip: Max gradient norm for clipping.
        eval_interval: Steps between validation evaluations.
        eval_iters: Number of batches averaged per evaluation.
        seed: Seed for model init and data sampling.
    """

    batch_size: int = 16
    block_size: int = 512
    target_tokens: int = 400_000_000
    warmup_steps: int = 500
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    seed: int = 1337

    @property
    def max_steps(self) -> int:
        """Total training steps, derived from target_tokens / (batch_size * block_size).

        Returns:
            int: The number of optimizer steps needed to consume the token
            budget, floored to the nearest integer and clamped to a minimum of
            ``1``.
        """
        tokens_per_step = self.batch_size * self.block_size
        return max(1, self.target_tokens // tokens_per_step)

    def __post_init__(self):
        """Validate the config.

        Raises:
            ValueError: If ``warmup_steps`` is not smaller than the derived
                ``max_steps``, ``min_lr`` is outside ``(0, max_lr]``, or
                ``block_size`` is not positive.
        """
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps (derived from target_tokens).")
        if not (0 < self.min_lr <= self.max_lr):
            raise ValueError("min_lr must be positive and <= max_lr.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")


@dataclass(frozen=True)
class RuntimeConfig(ConfigMixin):
    """Hardware/runtime settings, independent of model architecture or training schedule.

    Holds the execution environment and logging cadence: the compute device,
    precision, AdamW coefficients, and how often metrics and checkpoints are
    written. Deliberately decoupled from architecture and schedule configs so
    the same run can be moved across machines by changing only this section.

    Attributes:
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``. Defaults to
            whatever the machine this config is built on offers.
        dtype: Compute dtype, ``"float32"`` or ``"bfloat16"`` (``"bfloat16"``
            requires cuda).
        adam_betas: AdamW (beta1, beta2) coefficients.
        adam_eps: AdamW epsilon.
        log_every: Steps between training log lines.
        checkpoint_every: Steps between checkpoint saves.
    """

    device: str = field(default_factory=_default_device)
    dtype: str = "float32"
    adam_betas: tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    log_every: int = 10
    checkpoint_every: int = 1000

    def __post_init__(self):
        """Validate the config.

        Raises:
            ValueError: If ``dtype`` is ``"bfloat16"`` but ``device`` is
                ``"cpu"``, since bf16 is not efficiently supported on CPU.
        """
        if self.dtype == "bfloat16" and self.device == "cpu":
            raise ValueError("bfloat16 requires a CUDA device.")

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeConfig":
        """Build a RuntimeConfig from a dict, coercing adam_betas to a tuple.

        The generic :meth:`ConfigMixin.from_dict` would pass ``adam_betas``
        through unchanged; since YAML/JSON parse tuples as lists, this override
        converts the value to a tuple to match the frozen dataclass field type.

        Args:
            d: Dict of field name to value; ``adam_betas`` may be a list and
                will be converted to a tuple.

        Returns:
            RuntimeConfig: A new instance.
        """
        d = dict(d)
        if "adam_betas" in d:
            d["adam_betas"] = tuple(d["adam_betas"])
        return cls(**d)


@dataclass(frozen=True)
class RunConfig:
    """One coherent training run: architecture, schedule, and host settings.

    Owns the invariants that no single config can check on its own, and is the
    single value the :class:`Trainer` and the CLIs pass around in place of a
    ``(model, train, runtime)`` triple.

    Attributes:
        model: Transformer architecture configuration.
        train: Training hyperparameters.
        runtime: Hardware/runtime settings.
    """

    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeConfig

    def __post_init__(self):
        """Validate the invariants that span more than one config.

        Raises:
            ValueError: If ``train.block_size`` exceeds ``model.max_seq_len``,
                i.e. the run would feed sequences longer than the RoPE tables
                cover.
        """
        if self.train.block_size > self.model.max_seq_len:
            raise ValueError(
                f"block_size ({self.train.block_size}) exceeds the model's "
                f"max_seq_len ({self.model.max_seq_len})."
            )

    @classmethod
    def preset(cls, name: str, vocab_size: int) -> "RunConfig":
        """Build one of the named run configurations.

        Args:
            name: Preset name, one of :data:`PRESET_NAMES`.
            vocab_size: Vocabulary size to build the :class:`ModelConfig`
                with.

        Returns:
            RunConfig: The configuration for that preset.

        Raises:
            KeyError: If ``name`` is not a known preset.
        """
        try:
            build = _PRESETS[name]
        except KeyError:
            raise KeyError(
                f"unknown preset '{name}': expected one of {list(PRESET_NAMES)}."
            ) from None
        return build(vocab_size)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        """Load a run configuration from a YAML file.

        Args:
            path: Path to a YAML file with optional top-level ``"model"``,
                ``"train"``, and ``"runtime"`` sections; missing sections use
                their dataclass defaults.

        Returns:
            RunConfig: The configuration described by the file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Build a RunConfig from a dict of three optional sections.

        Args:
            d: Dict with optional ``"model"``, ``"train"``, and ``"runtime"``
                keys, each holding that config's field values.

        Returns:
            RunConfig: A new instance.
        """
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
            runtime=RuntimeConfig.from_dict(d.get("runtime", {})),
        )

    def to_dict(self) -> dict:
        """Convert this run configuration to a plain nested dict.

        Returns:
            dict: A mapping with ``"model"``, ``"train"``, and ``"runtime"``
            sections, in the same shape that :meth:`from_dict` and the YAML
            files accept.
        """
        return {
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


def _nano(vocab_size: int = 8000) -> RunConfig:
    """NANO preset: small model for a first end-to-end verification run.

    Args:
        vocab_size: Vocabulary size to build the :class:`ModelConfig` with.

    Returns:
        RunConfig: The NANO configuration.
    """
    return RunConfig(
        ModelConfig(vocab_size=vocab_size),
        TrainConfig(batch_size=16, block_size=512, target_tokens=400_000_000),
        RuntimeConfig(),
    )


def _small(vocab_size: int = 16000) -> RunConfig:
    """SMALL preset: larger model/schedule for a more capable run.

    Args:
        vocab_size: Vocabulary size to build the :class:`ModelConfig` with.

    Returns:
        RunConfig: The SMALL configuration.
    """
    return RunConfig(
        ModelConfig(
            vocab_size=vocab_size, n_layers=16, n_heads=8, n_kv_heads=4, head_dim=64,
            expert_intermediate=512, shared_intermediate=1376, n_experts=8, top_k=2,
            max_seq_len=2048,
        ),
        TrainConfig(batch_size=8, block_size=1024, target_tokens=1_500_000_000),
        RuntimeConfig(),
    )


def _overfit(vocab_size: int) -> RunConfig:
    """OVERFIT preset: tiny model/schedule for the overfit sanity gate.

    Uses a deliberately tiny budget (``~300`` steps) so the overfit sanity
    check can verify that the training loop memorizes a small batch end to end
    in a short time.

    Args:
        vocab_size: Vocabulary size to build the :class:`ModelConfig` with.

    Returns:
        RunConfig: The OVERFIT configuration.
    """
    return RunConfig(
        ModelConfig(vocab_size=vocab_size, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32,
                    expert_intermediate=64, shared_intermediate=128, n_experts=4, top_k=2,
                    max_seq_len=128),
        TrainConfig(batch_size=8, block_size=64, target_tokens=8 * 64 * 300,  # ~300 step
                    warmup_steps=10, max_lr=1e-3, eval_interval=50, eval_iters=5),
        RuntimeConfig(),
    )


_PRESETS = {
    "nano": _nano,
    "small": _small,
    "overfit": _overfit,
}

PRESET_NAMES = tuple(_PRESETS)

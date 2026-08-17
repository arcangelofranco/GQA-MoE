from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import PRESET_NAMES, ModelConfig, RunConfig, RuntimeConfig, TrainConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_model_config_derived_properties() -> None:
    """Verify that hidden_dim and group_size are derived from the base fields.

    Guards the contract that the model's effective feature dimension
    (``hidden_dim = n_heads * head_dim``) and the GQA grouping
    (``group_size = n_heads // n_kv_heads``) always follow from the raw
    configuration fields, since the rest of the codebase relies on these
    derived values.
    """
    cfg = ModelConfig(vocab_size=8000)
    print(f"[derived] hidden_dim={cfg.hidden_dim} group_size={cfg.group_size}")
    assert cfg.hidden_dim == 256
    assert cfg.group_size == 4


def test_model_config_defaults_match_nano_preset() -> None:
    """Verify that ModelConfig's plain defaults equal the "nano" preset's model config.

    Keeps the two definitions from drifting apart: a bare ``ModelConfig`` is
    what the ``nano`` preset uses, so any change to one without the other would
    silently alter the default training experience.
    """
    cfg = ModelConfig(vocab_size=8000)
    print(f"[nano-defaults] cfg == nano preset -> {cfg == RunConfig.preset('nano', 8000).model}")
    assert cfg == RunConfig.preset("nano", 8000).model


def test_model_config_derived_properties_are_read_only() -> None:
    """Verify that ModelConfig is frozen and rejects attribute assignment.

    Enforces the immutability contract of the config layer: since the frozen
    dataclass is shared across threads and reused for equality checks and
    serialization, accidental mutation must fail loudly instead of corrupting
    the configuration in place.
    """
    cfg = ModelConfig(vocab_size=8000)
    with pytest.raises(FrozenInstanceError) as exc_info:
        cfg.n_heads = 8
    print(f"[read-only] raised={exc_info.value!r}")


@pytest.mark.parametrize("overrides", [
    dict(n_heads=8, n_kv_heads=3),      # 8 is not a multiple of 3
    dict(n_experts=4, top_k=5),         # top_k out of range [1, n_experts]
    dict(head_dim=33),                  # odd
    dict(vocab_size=0),                 # non-positive
])
def test_model_config_invariants_raise_value_error(overrides: dict[str, int]) -> None:
    """Verify that each violated ModelConfig invariant raises ValueError.

    Drives the four architectural invariants one at a time (GQA divisibility,
    MoE ``top_k`` range, even ``head_dim``, positive ``vocab_size``) to ensure
    the configuration fails fast at construction rather than producing a model
    that crashes obscurely later.

    Args:
        overrides: Field values that violate one specific invariant.
    """
    base = dict(vocab_size=100)
    base.update(overrides)
    with pytest.raises(ValueError) as exc_info:
        ModelConfig(**base)
    print(f"[invariant] overrides={overrides} -> raised={exc_info.value!r}")


def test_model_config_roundtrip() -> None:
    """Verify that ModelConfig survives a to_dict()/from_dict() round-trip unchanged.

    Protects the serialization contract used to persist the model config inside
    checkpoints: whatever is saved must reconstruct to an equal object so
    resumed runs keep the identical architecture.
    """
    cfg = ModelConfig(vocab_size=8000, n_layers=6)
    roundtripped = ModelConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_require_vocab_size_accepts_a_match_and_names_the_source() -> None:
    """Verify require_vocab_size passes on a match and names the source on mismatch.

    Guards the guard itself: a matching vocabulary must be accepted silently,
    while a mismatch must fail with an error that identifies which artifact
    (dataset or tokenizer) disagrees, so the operator can trace the source of
    the discrepancy.
    """
    cfg = ModelConfig(vocab_size=8000)
    cfg.require_vocab_size(8000, source="dataset")  # does not raise

    with pytest.raises(ValueError, match="tokenizer vocab_size") as exc_info:
        cfg.require_vocab_size(7999, source="tokenizer")
    print(f"[vocab-mismatch] raised={exc_info.value!r}")


def test_train_config_max_steps() -> None:
    """Verify max_steps derives from the token budget and the per-step token count.

    Checks that the total step count is exactly ``target_tokens`` floor-divided
    by ``batch_size * block_size``, across two different (batch, block, budget)
    combinations, so the derived value tracks every knob that defines it.
    """
    cfg = TrainConfig(batch_size=16, block_size=512, target_tokens=400_000_000)
    print(f"[max-steps] batch_size=16 block_size=512 target_tokens=400_000_000 -> max_steps={cfg.max_steps}")
    assert cfg.max_steps == 400_000_000 // (16 * 512)

    cfg2 = TrainConfig(batch_size=8, block_size=1024, target_tokens=1_500_000_000)
    print(f"[max-steps] batch_size=8 block_size=1024 target_tokens=1_500_000_000 -> max_steps={cfg2.max_steps}")
    assert cfg2.max_steps == 1_500_000_000 // (8 * 1024)


def test_train_config_target_tokens_too_small_raises() -> None:
    """Verify that an unworkable warmup-to-max_steps ratio raises ValueError.

    When ``target_tokens`` is so small that the derived ``max_steps`` no longer
    exceeds ``warmup_steps``, the LR schedule would be undefined; the config
    must reject the combination at construction time.
    """
    with pytest.raises(ValueError) as exc_info:
        TrainConfig(batch_size=16, block_size=512, target_tokens=1000, warmup_steps=500)
    print(f"[target-tokens-too-small] raised={exc_info.value!r}")


def test_train_config_roundtrip() -> None:
    """Verify that TrainConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = TrainConfig(batch_size=8, target_tokens=50_000_000)
    roundtripped = TrainConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_runtime_config_bfloat16_on_cpu_raises() -> None:
    """Verify that requesting bfloat16 on CPU is rejected.

    bf16 has no efficient CPU path in this codebase, so silently allowing it
    would produce either an error deep inside training or unexpectedly slow
    computation; the config must surface the mistake up front.
    """
    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig(device="cpu", dtype="bfloat16")
    print(f"[bf16-on-cpu] raised={exc_info.value!r}")


def test_runtime_config_roundtrip() -> None:
    """Verify that RuntimeConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = RuntimeConfig(device="cpu", dtype="float32")
    roundtripped = RuntimeConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_runtime_config_from_dict_coerces_adam_betas_to_tuple() -> None:
    """Verify that from_dict() coerces a list adam_betas into a tuple.

    YAML and JSON parse sequences as lists, but the frozen dataclass field is
    typed as a tuple; this test simulates such a YAML override to pin the
    coercion behavior and keep the round-trip type-stable.
    """
    cfg = RuntimeConfig.from_dict({"adam_betas": [0.8, 0.9]})
    print(f"[adam-betas-coerce] adam_betas={cfg.adam_betas!r} type={type(cfg.adam_betas).__name__}")
    assert cfg.adam_betas == (0.8, 0.9)
    assert isinstance(cfg.adam_betas, tuple)


def test_device_default_is_resolved_per_instance_not_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the default device is resolved per instance, not once at import.

    The default must track the machine the config is built on, so two instances
    created under different CUDA availability must pick different devices
    without needing an import-time global decision.

    Args:
        monkeypatch: Fixture used to fake ``torch.cuda.is_available()``.
    """
    import src.config

    monkeypatch.setattr(src.config.torch.cuda, "is_available", lambda: True)
    cuda_device = RuntimeConfig().device
    print(f"[device-default] cuda.is_available=True -> device={cuda_device!r}")
    assert cuda_device == "cuda"

    monkeypatch.setattr(src.config.torch.cuda, "is_available", lambda: False)
    cpu_device = RuntimeConfig().device
    print(f"[device-default] cuda.is_available=False -> device={cpu_device!r}")
    assert cpu_device == "cpu"



def test_presets_are_valid() -> None:
    """Verify that every named preset in PRESET_NAMES builds successfully.

    Guards the preset registry: a preset that fails to construct would break
    the training CLI for that name, so every registered preset must be
    instantiable with the same vocabulary size.
    """
    for name in PRESET_NAMES:
        RunConfig.preset(name, 8000)
    print(f"[presets] built successfully: {list(PRESET_NAMES)}")


def test_unknown_preset_raises_key_error_listing_the_valid_names() -> None:
    """Verify that an unknown preset name raises KeyError listing the valid names.

    The error message is part of the operator-facing contract: a typo in the
    CLI must fail with the list of valid preset names so the mistake is
    immediately actionable.
    """
    with pytest.raises(KeyError, match="unknown preset") as exc_info:
        RunConfig.preset("enormous", 8000)
    print(f"[unknown-preset] raised={exc_info.value!r}")


def test_nano_preset_param_count_matches_notebook() -> None:
    """Verify the "nano" preset's parameter count matches the reference value.

    Pins the architecture against the count tracked in the reference notebook:
    any structural change (head count, MoE layout, embedding tying) that alters
    the parameter budget is caught here.
    """
    from src.model.transformer import Transformer
    model = Transformer(RunConfig.preset("nano", 8000).model)
    print(f"[nano-params] num_params={model.num_params()} expected=19_811_328")
    assert model.num_params() == 19_811_328


def test_small_preset_param_count_matches_notebook() -> None:
    """Verify the "small" preset's parameter count matches the reference value.

    Same guard as the nano check but for the larger preset, locking in the
    scaled-up architecture (deeper, wider, more experts) against the notebook
    baseline.
    """
    from src.model.transformer import Transformer
    model = Transformer(RunConfig.preset("small", 16000).model)
    print(f"[small-params] num_params={model.num_params()} expected=155_339_264")
    assert model.num_params() == 155_339_264


def test_block_size_larger_than_max_seq_len_raises() -> None:
    """Verify that a block_size exceeding max_seq_len raises ValueError.

    A context window longer than the precomputed RoPE tables would index past
    their range at training time; the cross-config validation must reject it
    before a run starts.
    """
    model_cfg = ModelConfig(vocab_size=100, max_seq_len=64)
    train_cfg = TrainConfig(batch_size=2, block_size=65, target_tokens=2 * 65 * 50, warmup_steps=2)
    with pytest.raises(ValueError, match="max_seq_len") as exc_info:
        RunConfig(model_cfg, train_cfg, RuntimeConfig())
    print(f"[block-size-too-large] raised={exc_info.value!r}")


def test_block_size_equal_to_max_seq_len_is_allowed() -> None:
    """Verify that a block_size equal to max_seq_len is accepted.

    Pins the boundary of the cross-config invariant: the constraint is strict
    (``block_size > max_seq_len`` fails) but equality must remain valid, so
    runs can use the full RoPE table.
    """
    model_cfg = ModelConfig(vocab_size=100, max_seq_len=64)
    train_cfg = TrainConfig(batch_size=2, block_size=64, target_tokens=2 * 64 * 50, warmup_steps=2)
    RunConfig(model_cfg, train_cfg, RuntimeConfig())  # does not raise
    print("[block-size-equal] RunConfig construction did not raise")


def test_run_config_roundtrip_through_dict() -> None:
    """Verify that RunConfig survives a to_dict()/from_dict() round-trip unchanged.

    The nested configs (model, train, runtime) are serialized as one coherent
    object; this test guards the recursive round-trip across all three levels.
    """
    cfg = RunConfig.preset("nano", 8000)
    roundtripped = RunConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_load_nano_yaml_matches_preset() -> None:
    """Verify that configs/nano.yml parses into the same RunConfig as the preset.

    Keeps the checked-in YAML configuration in lockstep with the code-defined
    preset: the file is the operator-facing artifact, so a drift between the
    two would silently change what ``nano`` actually runs.
    """
    from_yaml = RunConfig.from_yaml(str(CONFIGS_DIR / "nano.yml"))
    preset = RunConfig.preset("nano", 8000)
    print(f"[nano-yaml] from_yaml == preset -> {from_yaml == preset}")
    assert from_yaml == preset


def test_load_small_yaml_matches_preset() -> None:
    """Verify that configs/small.yml parses into the same RunConfig as the preset.

    Same drift guard as the nano YAML test, applied to the larger preset's
    configuration file.
    """
    from_yaml = RunConfig.from_yaml(str(CONFIGS_DIR / "small.yml"))
    preset = RunConfig.preset("small", 16000)
    print(f"[small-yaml] from_yaml == preset -> {from_yaml == preset}")
    assert from_yaml == preset


def test_yaml_partial_sections_use_dataclass_defaults(tmp_path: Path) -> None:
    """Verify that a YAML file with only a model section uses dataclass defaults elsewhere.

    Partial configuration files must not require every field: the training and
    runtime sections fall back to their defaults, which is the documented
    behavior operators rely on for minimal configs.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    minimal = tmp_path / "minimal.yml"
    minimal.write_text("model:\n  vocab_size: 8000\n")
    cfg = RunConfig.from_yaml(str(minimal))
    print(f"[yaml-partial] model={cfg.model == ModelConfig(vocab_size=8000)} train={cfg.train == TrainConfig()} runtime={cfg.runtime == RuntimeConfig()}")
    assert cfg.model == ModelConfig(vocab_size=8000)
    assert cfg.train == TrainConfig()
    assert cfg.runtime == RuntimeConfig()


def test_yaml_invalid_value_raises_value_error(tmp_path: Path) -> None:
    """Verify that a YAML file violating a ModelConfig invariant raises ValueError.

    The same architectural validation applied to programmatic construction must
    hold for file-loaded configs, so a bad YAML fails fast at load time rather
    than mid-training.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    bad = tmp_path / "bad.yml"
    bad.write_text("model:\n  vocab_size: 100\n  n_experts: 4\n  top_k: 5\n")
    with pytest.raises(ValueError) as exc_info:
        RunConfig.from_yaml(str(bad))
    print(f"[yaml-invalid] raised={exc_info.value!r}")


def test_yaml_incoherent_across_sections_raises_value_error(tmp_path: Path) -> None:
    """Verify that a cross-section inconsistency in YAML raises ValueError.

    The cross-config invariant (``block_size`` vs ``max_seq_len``) lives in
    :class:`RunConfig` and must be enforced even when the offending values come
    from different YAML sections, not just from one config in isolation.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    bad = tmp_path / "incoherent.yml"
    bad.write_text(
        "model:\n  vocab_size: 100\n  max_seq_len: 64\n"
        "train:\n  batch_size: 2\n  block_size: 128\n  target_tokens: 25600\n  warmup_steps: 2\n"
    )
    with pytest.raises(ValueError, match="max_seq_len") as exc_info:
        RunConfig.from_yaml(str(bad))
    print(f"[yaml-incoherent] raised={exc_info.value!r}")

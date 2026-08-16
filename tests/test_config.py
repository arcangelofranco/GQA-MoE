from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import PRESET_NAMES, ModelConfig, RunConfig, RuntimeConfig, TrainConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_model_config_derived_properties() -> None:
    """Checks that hidden_dim and group_size are correctly derived from the base fields."""
    cfg = ModelConfig(vocab_size=8000)
    print(f"[derived] hidden_dim={cfg.hidden_dim} group_size={cfg.group_size}")
    assert cfg.hidden_dim == 256
    assert cfg.group_size == 4


def test_model_config_defaults_match_nano_preset() -> None:
    """Checks that ModelConfig's plain defaults equal the "nano" preset's model config."""
    cfg = ModelConfig(vocab_size=8000)
    print(f"[nano-defaults] cfg == nano preset -> {cfg == RunConfig.preset('nano', 8000).model}")
    assert cfg == RunConfig.preset("nano", 8000).model


def test_model_config_derived_properties_are_read_only() -> None:
    """Checks that ModelConfig is frozen and rejects attribute assignment."""
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
    """Checks that each individually violated ModelConfig invariant raises ValueError.

    Args:
        overrides: Field values that violate one specific invariant.
    """
    base = dict(vocab_size=100)
    base.update(overrides)
    with pytest.raises(ValueError) as exc_info:
        ModelConfig(**base)
    print(f"[invariant] overrides={overrides} -> raised={exc_info.value!r}")


def test_model_config_roundtrip() -> None:
    """Checks that ModelConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = ModelConfig(vocab_size=8000, n_layers=6)
    roundtripped = ModelConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_require_vocab_size_accepts_a_match_and_names_the_source() -> None:
    """Checks that require_vocab_size passes on a match and names the source on mismatch."""
    cfg = ModelConfig(vocab_size=8000)
    cfg.require_vocab_size(8000, source="dataset")  # does not raise

    with pytest.raises(ValueError, match="tokenizer vocab_size") as exc_info:
        cfg.require_vocab_size(7999, source="tokenizer")
    print(f"[vocab-mismatch] raised={exc_info.value!r}")


def test_train_config_max_steps() -> None:
    """Checks that max_steps is target_tokens floor-divided by batch_size * block_size."""
    cfg = TrainConfig(batch_size=16, block_size=512, target_tokens=400_000_000)
    print(f"[max-steps] batch_size=16 block_size=512 target_tokens=400_000_000 -> max_steps={cfg.max_steps}")
    assert cfg.max_steps == 400_000_000 // (16 * 512)

    cfg2 = TrainConfig(batch_size=8, block_size=1024, target_tokens=1_500_000_000)
    print(f"[max-steps] batch_size=8 block_size=1024 target_tokens=1_500_000_000 -> max_steps={cfg2.max_steps}")
    assert cfg2.max_steps == 1_500_000_000 // (8 * 1024)


def test_train_config_target_tokens_too_small_raises() -> None:
    """Checks that target_tokens too small to reach even one step raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        TrainConfig(batch_size=16, block_size=512, target_tokens=1000, warmup_steps=500)
    print(f"[target-tokens-too-small] raised={exc_info.value!r}")


def test_train_config_roundtrip() -> None:
    """Checks that TrainConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = TrainConfig(batch_size=8, target_tokens=50_000_000)
    roundtripped = TrainConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_runtime_config_bfloat16_on_cpu_raises() -> None:
    """Checks that requesting bfloat16 on CPU raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig(device="cpu", dtype="bfloat16")
    print(f"[bf16-on-cpu] raised={exc_info.value!r}")


def test_runtime_config_roundtrip() -> None:
    """Checks that RuntimeConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = RuntimeConfig(device="cpu", dtype="float32")
    roundtripped = RuntimeConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_runtime_config_from_dict_coerces_adam_betas_to_tuple() -> None:
    """Checks that from_dict() coerces a list adam_betas into a tuple.

    Simulates what would come from a YAML file that overrides adam_betas.
    """
    cfg = RuntimeConfig.from_dict({"adam_betas": [0.8, 0.9]})
    print(f"[adam-betas-coerce] adam_betas={cfg.adam_betas!r} type={type(cfg.adam_betas).__name__}")
    assert cfg.adam_betas == (0.8, 0.9)
    assert isinstance(cfg.adam_betas, tuple)


def test_device_default_is_resolved_per_instance_not_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that RuntimeConfig's default device is resolved per instance, not once at import.

    Args:
        monkeypatch: Fixture used to fake `torch.cuda.is_available()`.
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
    """Checks that every named preset in PRESET_NAMES builds successfully."""
    for name in PRESET_NAMES:
        RunConfig.preset(name, 8000)
    print(f"[presets] built successfully: {list(PRESET_NAMES)}")


def test_unknown_preset_raises_key_error_listing_the_valid_names() -> None:
    """Checks that an unknown preset name raises KeyError mentioning "unknown preset"."""
    with pytest.raises(KeyError, match="unknown preset") as exc_info:
        RunConfig.preset("enormous", 8000)
    print(f"[unknown-preset] raised={exc_info.value!r}")


def test_nano_preset_param_count_matches_notebook() -> None:
    """Checks that the "nano" preset's parameter count matches the reference notebook value."""
    from src.model.transformer import Transformer
    model = Transformer(RunConfig.preset("nano", 8000).model)
    print(f"[nano-params] num_params={model.num_params()} expected=19_811_328")
    assert model.num_params() == 19_811_328


def test_small_preset_param_count_matches_notebook() -> None:
    """Checks that the "small" preset's parameter count matches the reference notebook value."""
    from src.model.transformer import Transformer
    model = Transformer(RunConfig.preset("small", 16000).model)
    print(f"[small-params] num_params={model.num_params()} expected=155_339_264")
    assert model.num_params() == 155_339_264


def test_block_size_larger_than_max_seq_len_raises() -> None:
    """Checks that a train block_size exceeding the model's max_seq_len raises ValueError."""
    model_cfg = ModelConfig(vocab_size=100, max_seq_len=64)
    train_cfg = TrainConfig(batch_size=2, block_size=65, target_tokens=2 * 65 * 50, warmup_steps=2)
    with pytest.raises(ValueError, match="max_seq_len") as exc_info:
        RunConfig(model_cfg, train_cfg, RuntimeConfig())
    print(f"[block-size-too-large] raised={exc_info.value!r}")


def test_block_size_equal_to_max_seq_len_is_allowed() -> None:
    """Checks that a train block_size equal to max_seq_len is accepted."""
    model_cfg = ModelConfig(vocab_size=100, max_seq_len=64)
    train_cfg = TrainConfig(batch_size=2, block_size=64, target_tokens=2 * 64 * 50, warmup_steps=2)
    RunConfig(model_cfg, train_cfg, RuntimeConfig())  # does not raise
    print("[block-size-equal] RunConfig construction did not raise")


def test_run_config_roundtrip_through_dict() -> None:
    """Checks that RunConfig survives a to_dict()/from_dict() round-trip unchanged."""
    cfg = RunConfig.preset("nano", 8000)
    roundtripped = RunConfig.from_dict(cfg.to_dict())
    print(f"[roundtrip] roundtripped == original -> {roundtripped == cfg}")
    assert roundtripped == cfg


def test_load_nano_yaml_matches_preset() -> None:
    """Checks that configs/nano.yml parses into the same RunConfig as the "nano" preset."""
    from_yaml = RunConfig.from_yaml(str(CONFIGS_DIR / "nano.yml"))
    preset = RunConfig.preset("nano", 8000)
    print(f"[nano-yaml] from_yaml == preset -> {from_yaml == preset}")
    assert from_yaml == preset


def test_load_small_yaml_matches_preset() -> None:
    """Checks that configs/small.yml parses into the same RunConfig as the "small" preset."""
    from_yaml = RunConfig.from_yaml(str(CONFIGS_DIR / "small.yml"))
    preset = RunConfig.preset("small", 16000)
    print(f"[small-yaml] from_yaml == preset -> {from_yaml == preset}")
    assert from_yaml == preset


def test_yaml_partial_sections_use_dataclass_defaults(tmp_path: Path) -> None:
    """Checks that a YAML file with only a model section fills the rest with dataclass defaults.

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
    """Checks that a YAML file violating a ModelConfig invariant raises ValueError.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    bad = tmp_path / "bad.yml"
    bad.write_text("model:\n  vocab_size: 100\n  n_experts: 4\n  top_k: 5\n")
    with pytest.raises(ValueError) as exc_info:
        RunConfig.from_yaml(str(bad))
    print(f"[yaml-invalid] raised={exc_info.value!r}")


def test_yaml_incoherent_across_sections_raises_value_error(tmp_path: Path) -> None:
    """Checks that a YAML file with a cross-section inconsistency raises ValueError.

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

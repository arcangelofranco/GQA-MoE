import json
import math
from pathlib import Path

import pytest

from src.runtime.metrics import (
    InMemoryRecorder,
    RunDirRecorder,
    StepMetrics,
    format_step_line,
)


def _full(**overrides: int | float | None) -> StepMetrics:
    """Builds a StepMetrics with sensible defaults, overridable per field.

    Args:
        **overrides: Field values overriding the defaults (step=10,
            max_steps=976, lr=3e-4, ms_step=812.4, train_loss=4.1203,
            val_loss=4.0501).

    Returns:
        The constructed `StepMetrics`.
    """
    fields = dict(step=10, max_steps=976, lr=3e-4, ms_step=812.4,
                  train_loss=4.1203, val_loss=4.0501)
    fields.update(overrides)
    return StepMetrics(**fields)


def test_perplexity_is_derived_from_loss() -> None:
    """Checks that train/val perplexity are exp() of the corresponding loss."""
    m = _full()
    print(f"[ppl] train_loss={m.train_loss} -> train_ppl={m.train_ppl} val_loss={m.val_loss} -> val_ppl={m.val_ppl}")
    assert m.train_ppl == pytest.approx(math.exp(4.1203))
    assert m.val_ppl == pytest.approx(math.exp(4.0501))


def test_perplexity_is_none_when_its_loss_is_absent() -> None:
    """Checks that a missing loss yields a None perplexity, independently per field."""
    m = _full(val_loss=None)
    print(f"[ppl-absent] val_loss={m.val_loss} -> val_ppl={m.val_ppl} train_ppl={m.train_ppl}")
    assert m.val_ppl is None
    assert m.train_ppl is not None


def test_to_dict_is_json_serializable_and_keeps_absences_as_null() -> None:
    """Checks that to_dict() round-trips through JSON and keeps None as null."""
    payload = json.loads(json.dumps(_full(val_loss=None).to_dict()))
    print(f"[to_dict] payload={payload}")
    assert payload["train_loss"] == pytest.approx(4.1203)
    assert payload["val_loss"] is None
    assert payload["val_ppl"] is None
    assert payload["step"] == 10 and payload["max_steps"] == 976


def test_line_format_is_unchanged_from_the_pre_recorder_trainer() -> None:
    """Checks that format_step_line's output matches the pre-recorder trainer's format.

    runs/nano_smoke/train.log was written by the previous version: the
    format must stay byte-for-byte identical, or that log stops being
    comparable with the new ones.
    """
    line = format_step_line(_full())
    print(f"[line-format] line={line!r}")
    assert line == (
        "step 10/976 | train_loss 4.1203 | train_ppl 61.58 | "
        "val_loss 4.0501 | val_ppl 57.40 | lr 3.00e-04 | 812 ms/step"
    )


def test_line_omits_the_pair_whose_measurement_is_absent() -> None:
    """Checks that format_step_line drops the loss/ppl pair whose measurement is missing."""
    train_only = format_step_line(_full(val_loss=None))
    print(f"[line-omit] train_only={train_only!r}")
    assert "train_loss" in train_only and "train_ppl" in train_only
    assert "val_loss" not in train_only and "val_ppl" not in train_only

    val_only = format_step_line(_full(train_loss=None))
    print(f"[line-omit] val_only={val_only!r}")
    assert "val_loss" in val_only and "val_ppl" in val_only
    assert "train_loss" not in val_only and "train_ppl" not in val_only



def test_in_memory_recorder_keeps_records_in_order() -> None:
    """Checks that InMemoryRecorder stores records in the order they were recorded."""
    recorder = InMemoryRecorder()
    assert recorder.records == []

    first, second = _full(step=1), _full(step=2)
    recorder.record(first)
    recorder.record(second)
    print(f"[in-memory] records steps={[r.step for r in recorder.records]}")
    assert recorder.records == [first, second]


def test_run_dir_recorder_writes_both_views_of_the_same_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that RunDirRecorder's log and JSONL views describe the same records.

    Args:
        tmp_path: Pytest-provided temporary directory.
        capsys: Fixture capturing stdout/stderr for the echoed log line.
    """
    recorder = RunDirRecorder(tmp_path / "run")
    recorder.record(_full(step=1))
    recorder.record(_full(step=2, val_loss=None))

    log_lines = recorder.log_path.read_text().splitlines()
    print(f"[run-dir] log_lines={log_lines}")
    assert len(log_lines) == 2
    assert log_lines[0].startswith("step 1/976 | train_loss")
    assert "val_loss" not in log_lines[1]

    rows = [json.loads(l) for l in recorder.metrics_path.read_text().splitlines()]
    print(f"[run-dir] metrics rows={rows}")
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["val_loss"] is None
    # the two views must describe the same records, not diverge
    assert rows[0]["train_ppl"] == pytest.approx(math.exp(rows[0]["train_loss"]))

    assert "step 1/976" in capsys.readouterr().out


def test_run_dir_recorder_appends_so_a_resumed_run_keeps_its_history(tmp_path: Path) -> None:
    """Checks that reopening a RunDirRecorder on the same dir appends instead of overwriting.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    run_dir = tmp_path / "run"
    RunDirRecorder(run_dir).record(_full(step=1))
    RunDirRecorder(run_dir).record(_full(step=2))  # as after a --resume

    log_line_count = len((run_dir / "train.log").read_text().splitlines())
    metrics_line_count = len((run_dir / "metrics.jsonl").read_text().splitlines())
    print(f"[resume-append] train.log lines={log_line_count} metrics.jsonl lines={metrics_line_count}")
    assert log_line_count == 2
    assert metrics_line_count == 2


def test_run_dir_recorder_can_stay_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that RunDirRecorder(echo=False) writes files without printing to stdout.

    Args:
        tmp_path: Pytest-provided temporary directory.
        capsys: Fixture capturing stdout/stderr, expected to stay empty.
    """
    RunDirRecorder(tmp_path / "run", echo=False).record(_full())
    captured = capsys.readouterr().out
    print(f"[silent] captured_stdout={captured!r}")
    assert captured == ""

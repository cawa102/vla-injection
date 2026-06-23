"""Tests for the pure early-stop bench bookkeeping (bench Task 3).

The per-target outcome dataclass, the steps-to-success distribution summary, and the
resume helpers the driver (Task 6) wraps with I/O + the GPU loop. All torch-free and
I/O-light (path existence only); exercised entirely off-GPU here.
"""

from __future__ import annotations

import json

import pytest

from evasion_tax.attack.early_stop_bench import (
    TargetOutcome,
    is_target_done,
    outcome_to_record,
    realistic_s_per_target,
    steps_to_success_summary,
    target_id_for,
)


def _outcome(target_id, *, reached, steps, censored, best_loss=0.5):
    """Build a TargetOutcome with the fields the summary reads (others are placeholders)."""
    return TargetOutcome(
        target_id=target_id,
        reached=reached,
        steps_to_success=steps,
        censored=censored,
        best_loss=best_loss,
        wall_seconds=1.0,
        peak_vram_gib=14.5,
        suffix_sha256="0" * 64,
    )


def test_realistic_s_per_target_is_median_steps_times_s_per_step():
    # The early-stop realistic per-target cost (DE-2): median_steps * s/step.
    # D8 s/step = 33.19 s; a 60-step median => ~1991 s, vs the early_stop-OFF 16,595 s.
    assert realistic_s_per_target(60.0, 33.19) == pytest.approx(1991.4)


@pytest.mark.parametrize(("median_steps", "s_per_step"), [(0.0, 33.19), (60.0, 0.0), (-1.0, 33.19)])
def test_realistic_s_per_target_rejects_non_positive(median_steps, s_per_step):
    # No silent default: a zero/negative median or s/step is a programming error, fail loud.
    with pytest.raises(ValueError):
        realistic_s_per_target(median_steps, s_per_step)


def test_target_id_for_is_deterministic_and_encodes_per_target_seed():
    # The driver builds target i with seed+i, so the id encodes that per-target seed.
    assert target_id_for(42, 0) == "t42"
    assert target_id_for(42, 3) == "t45"
    assert target_id_for(42, 3) == target_id_for(42, 3)  # deterministic


def test_is_target_done_true_iff_per_target_json_exists(tmp_path):
    # Skip-if-exists resume (DE-4): a target is "done" exactly when its JSON is on disk.
    assert is_target_done(tmp_path, "t42") is False
    (tmp_path / "t42.json").write_text("{}")
    assert is_target_done(tmp_path, "t42") is True
    assert is_target_done(tmp_path, "t99") is False  # a different, unwritten target


def test_steps_to_success_summary_excludes_censored_from_median():
    # 3 reached at [40,60,80] + 1 censored at the cap. The median is over the REACHED
    # targets only (60); the censored 500 must NOT pull it (else median would be 70).
    outcomes = [
        _outcome("t1", reached=True, steps=40, censored=False),
        _outcome("t2", reached=True, steps=60, censored=False),
        _outcome("t3", reached=True, steps=80, censored=False),
        _outcome("t4", reached=False, steps=500, censored=True),
    ]

    summary = steps_to_success_summary(outcomes, n_steps_cap=500)

    assert summary["n"] == 4
    assert summary["n_reached"] == 3
    assert summary["reached_fraction"] == pytest.approx(0.75)
    assert summary["censored_fraction"] == pytest.approx(0.25)
    assert summary["median_steps"] == pytest.approx(60.0)  # over [40,60,80], NOT [.. ,500]
    assert summary["iqr_steps"] == pytest.approx(20.0)  # q75=70, q25=50
    assert summary["n_steps_cap"] == 500  # the cap that defines censoring, recorded


def test_steps_to_success_summary_all_censored_has_none_median():
    # A fully-censored run: no reached target to take a median over -> None (not NaN),
    # and censored_fraction == 1.0 (the worst-case bound is then the honest number, DE-2).
    outcomes = [
        _outcome("t1", reached=False, steps=500, censored=True),
        _outcome("t2", reached=False, steps=500, censored=True),
    ]

    summary = steps_to_success_summary(outcomes, n_steps_cap=500)

    assert summary["median_steps"] is None
    assert summary["iqr_steps"] is None
    assert summary["censored_fraction"] == pytest.approx(1.0)
    assert summary["reached_fraction"] == pytest.approx(0.0)


def test_steps_to_success_summary_rejects_empty():
    # No silent default: an empty outcome set is a programming error, fail loud.
    with pytest.raises(ValueError):
        steps_to_success_summary([], n_steps_cap=500)


def test_outcome_to_record_is_json_safe_and_carries_every_field():
    o = _outcome("t42", reached=True, steps=60, censored=False, best_loss=0.123)

    record = outcome_to_record(o)

    # Round-trips through JSON unchanged (the driver writes this write-once, DE-4).
    assert json.loads(json.dumps(record)) == record
    assert record["target_id"] == "t42"
    assert record["reached"] is True
    assert record["steps_to_success"] == 60
    assert record["censored"] is False
    assert record["best_loss"] == pytest.approx(0.123)
    assert record["suffix_sha256"] == "0" * 64
    # Only outcome METADATA is recorded -- never the suffix text itself (DE-5).
    assert set(record) == {
        "target_id", "reached", "steps_to_success", "censored",
        "best_loss", "wall_seconds", "peak_vram_gib", "suffix_sha256",
    }

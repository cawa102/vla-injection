"""Tests for ``scripts/microbench_gcg.py`` — the model-free aggregation/sweep logic.

The D8 timing micro-bench's GPU body (timing the real harness, probing VRAM) runs
on the CSB A5000; here we pin only the **pure** pieces it is built from:

* :func:`summarise_timings` — median/IQR/n + a reproducibility flag across repeats;
* :func:`max_batch_that_fits` — the doubling/bisection batch sweep against a fake
  ``probe_fn`` whose OOM boundary is known, never reusing a probe (clean-process
  protocol D6-10, modelled here as never re-probing the same ``B``);
* :func:`build_microbench_record` — the registered record carries every required
  field **plus** ``branch_status="provisional"`` + the lock condition and the
  **deferred** L1/adaptive items, so the gap is visible, not silently dropped
  (D6-4/D6-5).

All model-free: the script must import without torch (clean-import test below).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load_microbench():
    """Import ``scripts/microbench_gcg.py`` (scripts/ on sys.path for _bootstrap)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("microbench_gcg")


# --------------------------------------------------------------------------- #
# summarise_timings                                                           #
# --------------------------------------------------------------------------- #


def test_summarise_timings_median_iqr_n_on_fixed_list():
    mb = _load_microbench()
    out = mb.summarise_timings([2.0, 4.0, 6.0, 8.0])
    assert out["median_s"] == 5.0
    assert out["iqr_s"] == 3.0  # q75=6.5, q25=3.5
    assert out["n"] == 4


def test_summarise_timings_flags_reproducible_vs_noisy_repeats():
    mb = _load_microbench()
    tight = mb.summarise_timings([10.0, 10.0, 10.0, 10.0])
    noisy = mb.summarise_timings([1.0, 2.0, 10.0, 20.0])
    assert tight["reproducible"] is True
    assert noisy["reproducible"] is False


def test_summarise_timings_rejects_empty():
    mb = _load_microbench()
    with pytest.raises(ValueError):
        mb.summarise_timings([])


# --------------------------------------------------------------------------- #
# max_batch_that_fits: doubling/bisection sweep, no probe reuse (D6-10)         #
# --------------------------------------------------------------------------- #


def _fake_probe(mb, boundary, calls):
    """A probe that fits for ``B <= boundary`` and OOMs above it; records every B."""

    def probe(b):
        calls.append(b)
        if b > boundary:
            raise mb.OomError(f"OOM at B={b}")
        return b

    return probe


def test_max_batch_that_fits_finds_largest_non_oom_without_reprobing():
    mb = _load_microbench()
    calls: list[int] = []
    best = mb.max_batch_that_fits(_fake_probe(mb, boundary=13, calls=calls), start=1, cap=64)
    assert best == 13
    assert len(calls) == len(set(calls))  # never re-probes a B (clean-process protocol)


def test_max_batch_that_fits_returns_cap_when_everything_fits():
    mb = _load_microbench()
    calls: list[int] = []
    best = mb.max_batch_that_fits(_fake_probe(mb, boundary=999, calls=calls), start=1, cap=64)
    assert best == 64
    assert len(calls) == len(set(calls))


def test_max_batch_that_fits_returns_zero_when_start_ooms():
    mb = _load_microbench()
    calls: list[int] = []
    best = mb.max_batch_that_fits(_fake_probe(mb, boundary=0, calls=calls), start=1, cap=64)
    assert best == 0


# --------------------------------------------------------------------------- #
# build_microbench_record: §8 fields + provisional branch + deferred items      #
# --------------------------------------------------------------------------- #


def _record(mb, **overrides):
    base = dict(
        gcg_config={"suffix_len": 20, "n_steps": 100, "top_k": 256, "search_width": 256, "seed": 0},
        timing_summary={
            "median_s": 120.0, "iqr_s": 5.0, "n": 4, "rel_iqr": 0.04, "reproducible": True
        },
        peak_vram_gib=15.5,
        max_candidate_batch=256,
        steps_to_success=[30, 45, 60],
        device_name="NVIDIA RTX A5000",
        seed=0,
        exclusive_gpu=True,
    )
    base.update(overrides)
    return mb.build_microbench_record(**base)


def test_build_microbench_record_marks_provisional_and_logs_deferred():
    mb = _load_microbench()
    rec = _record(mb)

    assert rec["branch_status"] == "provisional"
    assert "adaptive" in rec["branch_lock_condition"].lower()
    # DEFERRED items are explicitly recorded (D6-4: named, never silently dropped).
    assert rec["l1_extraction_overhead"].startswith("deferred")
    assert rec["adaptive_gcg_cost"].startswith("deferred")


def test_build_microbench_record_carries_required_repro_fields():
    mb = _load_microbench()
    rec = _record(mb)
    for key in (
        "stage",
        "dtype",
        "device_name",
        "seed",
        "gcg_config",
        "s_per_target",
        "peak_vram_gib",
        "max_candidate_batch",
        "steps_to_success",
        "exclusive_gpu",
    ):
        assert key in rec
    assert rec["dtype"] == "bfloat16"
    assert rec["s_per_target"]["median_s"] == 120.0


def test_build_microbench_record_carries_loop_baseline_and_speedup():
    # DB-2: register BOTH numbers. true-batch = official `s_per_target`; the loop number is
    # a baseline/ablation (`s_per_target_loop`) and the measured speedup `k = loop/batch`.
    mb = _load_microbench()
    loop_summary = {"median_s": 17.47, "iqr_s": 0.3, "n": 1, "rel_iqr": 0.0, "reproducible": True}
    rec = _record(mb, s_per_target_loop=loop_summary, speedup_k=3.2)

    assert rec["s_per_target_loop"] == loop_summary  # loop ablation carried alongside
    assert rec["speedup_k"] == 3.2
    assert rec["s_per_target"]["median_s"] == 120.0  # official sizing stays the true-batch number


def test_build_microbench_record_frames_max_batch_as_hw_not_branch_critical():
    # DB-3: max-B is a VRAM ceiling (hardware characterisation), never the branch decider.
    mb = _load_microbench()
    note = _record(mb)["max_batch_note"].lower()

    assert "vram" in note or "ceiling" in note  # framed as a hardware ceiling
    assert "not" in note and "branch" in note  # explicitly NOT branch-critical


# --------------------------------------------------------------------------- #
# assert_registered_run_valid: D6-10 clean-process / reproducibility gate       #
# --------------------------------------------------------------------------- #


def test_registered_run_valid_accepts_exclusive_reproducible():
    mb = _load_microbench()
    mb.assert_registered_run_valid(_record(mb))  # exclusive + reproducible: no raise


def test_registered_run_invalid_when_not_exclusive():
    mb = _load_microbench()
    with pytest.raises(ValueError):
        mb.assert_registered_run_valid(_record(mb, exclusive_gpu=False))


def test_registered_run_invalid_when_not_reproducible():
    mb = _load_microbench()
    noisy = {"median_s": 1.0, "iqr_s": 5.0, "n": 4, "rel_iqr": 5.0, "reproducible": False}
    with pytest.raises(ValueError):
        mb.assert_registered_run_valid(_record(mb, timing_summary=noisy))


# --------------------------------------------------------------------------- #
# Clean import: the model-free script must not pull torch at import time         #
# --------------------------------------------------------------------------- #


def test_microbench_module_imports_without_torch():
    _load_microbench()
    assert "torch" not in sys.modules

"""Tests for ``scripts/run_attack.py`` (Task 5) — guard + resume/quarantine/record glue.

The GPU work (GCG suffix optimisation + attacked rollout) is mocked via an injected
``attack_fn``; the per-unit resume, suffix quarantine, denial flag, both-success-
notions record, and frozen-schema load (refuses to re-pin from attacked data) are
tested without CUDA. The guard (off-GPU ⇒ exit 2) is the shared script contract.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from evasion_tax.attack.early_stop_bench import TargetOutcome

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("run_attack")


def _outcome(uid, *, reached=True, steps=60):
    return TargetOutcome(
        target_id=uid, reached=reached, steps_to_success=steps, censored=not reached,
        best_loss=0.05, wall_seconds=200.0, peak_vram_gib=18.0, suffix_sha256="cafef00d",
    )


def _attack_out(uid, *, asr=True, task_success=False):
    return {
        "cost": _outcome(uid),
        "suffix_text": f"! ! ! suffix for {uid}",
        "rollout_asr_reached": asr,
        "task_success": task_success,
        "metric_a_per_step": [1.0, 1.0],
    }


def test_guard_without_cuda_exits_2(monkeypatch, capsys, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod, "cuda_available", lambda: False)
    cfg = _REPO_ROOT / "configs" / "example_m2.yaml"
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"engagement_radius": 0.04, "grasp_radius": 0.12}))
    rc = mod.main(["--config", str(cfg), "--schema-from", str(schema),
                   "--results-root", str(tmp_path)])
    assert rc == 2
    assert mod.STAGE in capsys.readouterr().err


def test_is_denial_only_when_neither_region_nor_goal_reached():
    mod = _load()
    assert mod.is_denial(asr_reached=False, task_success=False) is True
    assert mod.is_denial(asr_reached=True, task_success=False) is False   # redirect
    assert mod.is_denial(asr_reached=False, task_success=True) is False   # benign task done


def test_prepare_run_dir_is_stable_across_restarts(tmp_path):
    mod = _load()
    d1, first1 = mod.prepare_run_dir(str(tmp_path), "m1-robogcg-redirect")
    d2, first2 = mod.prepare_run_dir(str(tmp_path), "m1-robogcg-redirect")
    assert d1 == d2                              # same dir → --resume + until-loop work
    assert first1 is True and first2 is False


def test_load_frozen_schema_reads_only_radii(tmp_path):
    mod = _load()
    p = tmp_path / "schema.json"
    p.write_text(json.dumps({"engagement_radius": 0.04, "grasp_radius": 0.12}))
    schema = mod.load_frozen_schema(str(p))
    assert schema.engagement_radius == 0.04 and schema.grasp_radius == 0.12


def test_run_attack_loop_records_both_notions_and_quarantines(tmp_path):
    mod = _load()
    units = ["t0:r0:42", "t0:r1:43", "t1:r0:44"]
    units_dir = tmp_path / "units"
    quarantine = tmp_path / "untrusted"

    def attack_fn(unit):
        return _attack_out(unit, asr=True, task_success=False)

    records = mod.run_attack_loop(
        units_dir, quarantine, units=units, attack_fn=attack_fn, resume=False
    )
    assert len(records) == 3
    # each record carries BOTH success notions (cost.reached + rollout ASR) + denial
    for r in records:
        assert r["cost"]["reached"] is True
        assert r["rollout_asr_reached"] is True
        assert r["is_denial"] is False
    # exactly one quarantined suffix per fresh unit
    assert len(list(quarantine.glob("*.txt"))) == 3


def test_run_attack_loop_resume_skips_finished_unit(tmp_path):
    mod = _load()
    units = ["t0:r0:42", "t0:r1:43"]
    units_dir = tmp_path / "units"
    quarantine = tmp_path / "untrusted"
    mod.run_attack_loop(units_dir, quarantine, units=units,
                        attack_fn=lambda u: _attack_out(u), resume=False)

    calls = []

    def counting(unit):
        calls.append(unit)
        return _attack_out(unit)

    records = mod.run_attack_loop(units_dir, quarantine, units=units,
                                  attack_fn=counting, resume=True)
    assert calls == []                                    # all skipped
    assert len(records) == 2
    assert len(list(quarantine.glob("*.txt"))) == 2       # no extra quarantined suffixes


def test_run_attack_loop_writes_aggregate_incrementally_survives_crash(tmp_path):
    # BUG1: a crash mid-loop must still leave a valid aggregate of the finished units,
    # so an interrupted run is not lost and m1_gate_report has something to read.
    mod = _load()
    units = ["t0:r0:42", "t0:r1:43", "t1:r0:44"]
    units_dir = tmp_path / "units"
    quarantine = tmp_path / "untrusted"

    def failing(unit):
        if unit == units[2]:
            raise RuntimeError("boom on unit 3")
        return _attack_out(unit)

    with pytest.raises(RuntimeError):
        mod.run_attack_loop(units_dir, quarantine, units=units,
                            attack_fn=failing, resume=False)

    aggregate = units_dir.parent / "attack_records.json"
    assert aggregate.exists()
    records = json.loads(aggregate.read_text())
    assert len(records) == 2                               # only the units finished pre-crash
    assert [r["unit_id"] for r in records] == units[:2]


def test_run_attack_loop_fires_on_unit_done_per_fresh_unit(tmp_path):
    # BUG5: a boundary hook fires once per FRESH unit (the GPU body wires it to
    # torch.cuda.empty_cache() to curb fragmentation OOM over sequential sw=512 searches).
    mod = _load()
    units = ["t0:r0:42", "t0:r1:43", "t1:r0:44"]
    calls = []
    mod.run_attack_loop(
        tmp_path / "units", tmp_path / "untrusted", units=units,
        attack_fn=lambda u: _attack_out(u), resume=False,
        on_unit_done=lambda: calls.append(1),
    )
    assert len(calls) == 3                                 # one per fresh unit


def test_run_attack_loop_on_unit_done_skips_reloaded_units(tmp_path):
    # Reloaded (resumed) units did no GPU work → no cache-clear needed for them.
    mod = _load()
    units = ["t0:r0:42", "t0:r1:43"]
    units_dir = tmp_path / "units"
    quarantine = tmp_path / "untrusted"
    mod.run_attack_loop(units_dir, quarantine, units=units,
                        attack_fn=lambda u: _attack_out(u), resume=False)

    calls = []
    mod.run_attack_loop(units_dir, quarantine, units=units,
                        attack_fn=lambda u: _attack_out(u), resume=True,
                        on_unit_done=lambda: calls.append(1))
    assert calls == []                                     # all reloaded → hook never fires


def _header(**over):
    h = {
        "seed": 42, "git_commit": "abc123", "model": "openvla/x",
        "n_steps": 500, "search_width": 512, "eval_batch": 32, "schema_sha256": "feed",
    }
    h.update(over)
    return h


def test_assert_resume_compatible_noop_when_no_run_json(tmp_path):
    # BUG3: no run.json → fresh run, nothing to check.
    mod = _load()
    mod.assert_resume_compatible(tmp_path, _header())  # must not raise


def test_assert_resume_compatible_passes_on_identical_header(tmp_path):
    mod = _load()
    (tmp_path / "run.json").write_text(json.dumps(_header()))
    mod.assert_resume_compatible(tmp_path, _header())  # identical config → proceeds


def test_assert_resume_compatible_raises_naming_changed_field(tmp_path):
    # A changed n_steps must abort (never silently mix incompatible units) and name it.
    mod = _load()
    (tmp_path / "run.json").write_text(json.dumps(_header(n_steps=500)))
    with pytest.raises(SystemExit) as exc:
        mod.assert_resume_compatible(tmp_path, _header(n_steps=20))
    assert "n_steps" in str(exc.value)

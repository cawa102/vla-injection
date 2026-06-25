"""Tests for ``scripts/run_benign.py`` (Task 4) — guard + pure aggregate/log glue.

The GPU body (OpenVLA + LIBERO) is mocked: the per-episode ``episode_fn`` is
injected, so the loop / resume / split / aggregate / geometry-emit glue is tested
without CUDA. The guard (off-GPU ⇒ exit 2) is the shared script contract.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from evasion_tax.eval.rollout_runner import EpisodeResult
from evasion_tax.metric.consistency_a import SchemaA
from evasion_tax.records import Rollout, RolloutStep

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("run_benign")


def _step(ee, *, run_id, step):
    return RolloutStep(
        run_id=run_id, seed=0, git_commit=None, suite="libero_spatial", task_id="t0", step=step,
        observation_ref=f"{run_id}/{step}", action=(0.0,) * 7,
        privileged_state={
            "ee_pos": ee, "gripper_open": True,
            "object_poses": {"goal": (1.0, 0.0, 0.0)}, "target_region": "goal",
        },
        instruction="pick", trusted_goal="pick", attacked=False, suffix_ref=None,
    )


def _episode(i):
    rid = f"ep{i}"
    return EpisodeResult(
        rollout=Rollout(steps=(_step((0.0, 0.0, 0.0), run_id=rid, step=0),
                               _step((0.5, 0.0, 0.0), run_id=rid, step=1))),
        success=True,
    )


def test_guard_without_cuda_exits_2(monkeypatch, capsys, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod, "cuda_available", lambda: False)
    cfg = _REPO_ROOT / "configs" / "example_m2.yaml"
    rc = mod.main(["--config", str(cfg), "--results-root", str(tmp_path)])
    assert rc == 2
    assert mod.STAGE in capsys.readouterr().err


def test_prepare_run_dir_is_stable_and_first_launch_flagged(tmp_path):
    mod = _load()
    d1, first1 = mod.prepare_run_dir(str(tmp_path), "m1-benign-baseline")
    d2, first2 = mod.prepare_run_dir(str(tmp_path), "m1-benign-baseline")
    assert d1 == d2                       # same stable dir across restarts (resume works)
    assert first1 is True and first2 is False  # header written only on the first launch


def test_assign_calibration_split_is_disjoint_by_index():
    mod = _load()
    flags = [mod.assign_calibration(i, 10, 0.4) for i in range(10)]
    assert sum(flags) == 4                       # round(10*0.4) calibration episodes
    assert flags[:4] == [True] * 4 and flags[4:] == [False] * 6  # disjoint prefix


def test_run_benign_loop_aggregates_and_writes(tmp_path):
    mod = _load()
    episodes_dir = tmp_path / "episodes"
    calls = []

    def episode_fn(*, index, seed):
        calls.append(index)
        return _episode(index)

    records, summary = mod.run_benign_loop(
        episodes_dir, n_benign=10, calib_frac=0.4, seed=42,
        schema=SchemaA(), k=5, episode_fn=episode_fn, resume=False,
    )
    assert summary["n"] == 10
    assert summary["success_rate"] == 1.0
    assert summary["n_calib"] == 4 and summary["n_eval"] == 6
    assert len(records) == 10
    # every record carries metric-A scores + geometry for the DM-3 re-pin
    assert all("metric_a_per_step" in r and "geometry" in r for r in records)
    assert len(list(episodes_dir.glob("*.json"))) == 10


def test_run_benign_loop_resume_skips_finished(tmp_path):
    mod = _load()
    episodes_dir = tmp_path / "episodes"

    def episode_fn(*, index, seed):
        return _episode(index)

    # first pass writes all 5
    mod.run_benign_loop(episodes_dir, n_benign=5, calib_frac=0.4, seed=0,
                        schema=SchemaA(), k=5, episode_fn=episode_fn, resume=False)

    calls = []

    def counting_fn(*, index, seed):
        calls.append(index)
        return _episode(index)

    # second pass with resume must not re-run any episode
    records, _ = mod.run_benign_loop(episodes_dir, n_benign=5, calib_frac=0.4, seed=0,
                                     schema=SchemaA(), k=5, episode_fn=counting_fn, resume=True)
    assert calls == []                 # all skipped
    assert len(records) == 5           # reloaded from disk
    geom = [r["geometry"] for r in records]
    assert all(json.dumps(g) for g in geom)  # geometry round-trips through JSON

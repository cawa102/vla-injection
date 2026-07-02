"""Tests for ``scripts/rescore_benign.py`` (BUG2 — benign re-score CLI, model-free).

Builds a benign run dir (``run.json`` with the run's ``k`` + ``episodes/ep*.json``
carrying the logged ``RolloutStep`` stream), runs the CLI, and checks that
re-scoring against the run's ORIGINAL schema reproduces the stored
``metric_a_per_step`` (the anchoring property) and carries ``success`` through. No
GPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA
from evasion_tax.records import ACTION_DIM, Rollout, RolloutStep

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
_K = 5


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("rescore_benign")


def _gstep(ee, *, step, gripper_open):
    return RolloutStep(
        run_id="r", seed=0, git_commit=None, suite="s", task_id="t", step=step,
        observation_ref=f"r/{step}", action=(0.0,) * ACTION_DIM,
        privileged_state={
            "ee_pos": ee, "gripper_open": gripper_open,
            "object_poses": {"goal": (1.0, 0.0, 0.0), "distractor": (0.0, 1.0, 0.0)},
            "target_region": "goal",
        },
        instruction="i", trusted_goal="g", attacked=False, suffix_ref=None,
    )


def _rollout():
    return Rollout(steps=(
        _gstep((0.0, 0.0, 0.0), step=0, gripper_open=True),
        _gstep((0.5, 0.0, 0.0), step=1, gripper_open=True),
        _gstep((0.9, 0.0, 0.0), step=2, gripper_open=False),
    ))


def _metric_scores(rollout, schema):
    return [s.value for s in ConsistencyMetricA(schema=schema, k=_K).score_rollout(rollout)]


def _episode_dict(rollout, *, success, is_calibration, schema):
    # Mirror what run_benign._score_episode writes: the raw RolloutStep stream plus
    # the schema-scored metric_a_per_step (here, the ORIGINAL-schema scores).
    return {
        "success": success,
        "is_calibration": is_calibration,
        "metric_a_per_step": _metric_scores(rollout, schema),
        "geometry": {},
        "steps": [dataclasses.asdict(s) for s in rollout.steps],
    }


def _write_run(tmp_path, schema):
    run = tmp_path / "benign"
    (run / "episodes").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"config": {"k": _K}}))
    episodes = [
        _episode_dict(_rollout(), success=True, is_calibration=True, schema=schema),
        _episode_dict(_rollout(), success=False, is_calibration=False, schema=schema),
        _episode_dict(_rollout(), success=True, is_calibration=False, schema=schema),
    ]
    for i, ep in enumerate(episodes):
        (run / "episodes" / f"ep{i}.json").write_text(json.dumps(ep))
    return run, episodes


def test_cli_reproduces_stored_scores_with_same_schema(tmp_path):
    mod = _load()
    schema = SchemaA()
    run, episodes = _write_run(tmp_path, schema)
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({"engagement_radius": 0.05, "grasp_radius": 0.10}))
    out = tmp_path / "benign_records_repinned.json"

    rc = mod.main(["--run", str(run), "--schema-from", str(schema_file), "--out", str(out)])

    assert rc == 0
    records = json.loads(out.read_text())
    assert [r["metric_a_per_step"] for r in records] == [ep["metric_a_per_step"] for ep in episodes]
    assert [r["success"] for r in records] == [ep["success"] for ep in episodes]
    assert [r["is_calibration"] for r in records] == [ep["is_calibration"] for ep in episodes]

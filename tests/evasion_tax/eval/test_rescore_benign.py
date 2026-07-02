"""Tests for benign re-scoring against the re-pinned SchemaA (BUG2, DM-3 same-scale).

Pure, model-free: builds in-memory rollouts, re-scores them against a schema, and
checks (a) re-scoring with the run's ORIGINAL schema reproduces the metric's own
``metric_a_per_step`` (the anchoring correctness property), (b) a DIFFERENT schema
changes the scores, (c) ``success`` / ``is_calibration`` are carried through, and
(d) empty input fails fast. No GPU.
"""

from __future__ import annotations

import pytest

from evasion_tax.eval.rescore_benign import rescore_benign_records
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA
from evasion_tax.records import ACTION_DIM, Rollout, RolloutStep

_K = 5


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
    # EE approaches the goal (1,0,0) with an open->close grasp near it at step 2; the
    # grasp distance (0.1) makes the score sensitive to grasp_radius, so a re-pinned
    # schema visibly changes it.
    return Rollout(steps=(
        _gstep((0.0, 0.0, 0.0), step=0, gripper_open=True),
        _gstep((0.5, 0.0, 0.0), step=1, gripper_open=True),
        _gstep((0.9, 0.0, 0.0), step=2, gripper_open=False),
    ))


def _metric_scores(rollout, schema, k=_K):
    return [s.value for s in ConsistencyMetricA(schema=schema, k=k).score_rollout(rollout)]


def test_rescore_with_original_schema_reproduces_metric_scores():
    schema = SchemaA()
    rollouts = [_rollout(), _rollout()]
    expected = [_metric_scores(r, schema) for r in rollouts]

    records = rescore_benign_records(
        rollouts, success=[True, False], is_calibration=[True, False], schema=schema, k=_K
    )

    assert [r["metric_a_per_step"] for r in records] == expected


def test_rescore_with_different_schema_changes_scores():
    # A re-pinned schema (larger grasp_radius) must actually change the scores — this
    # is the whole point of BUG2: benign moves onto the attack's re-pinned scale.
    original = SchemaA()
    repinned = SchemaA(engagement_radius=0.20, grasp_radius=0.30)
    rollout = _rollout()

    orig = rescore_benign_records(
        [rollout], success=[True], is_calibration=[True], schema=original, k=_K
    )[0]
    repin = rescore_benign_records(
        [rollout], success=[True], is_calibration=[True], schema=repinned, k=_K
    )[0]

    assert orig["metric_a_per_step"] != repin["metric_a_per_step"]


def test_rescore_carries_success_and_calibration_unchanged():
    records = rescore_benign_records(
        [_rollout(), _rollout(), _rollout()],
        success=[True, False, True],
        is_calibration=[False, True, False],
        schema=SchemaA(),
        k=_K,
    )
    assert [r["success"] for r in records] == [True, False, True]
    assert [r["is_calibration"] for r in records] == [False, True, False]


def test_rescore_empty_input_raises():
    with pytest.raises(ValueError):
        rescore_benign_records([], success=[], is_calibration=[], schema=SchemaA(), k=_K)

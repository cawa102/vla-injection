"""Tests for benign-vs-attacked metric-(A) separation (Task 3).

Pure aggregation: score benign + attacked rollouts with the frozen metric (A)
against a (re-pinned) ``SchemaA``, reduce per-rollout (max), and reuse the eval
harness — τ on benign-calibration only, ``realised_fpr`` on the held-out benign
split (invariant #3). No new statistics.
"""

import pytest

from evasion_tax.eval.separation import per_rollout_score, separation_table
from evasion_tax.metric.consistency_a import SchemaA
from evasion_tax.records import Rollout, RolloutStep, Score

_ANCHOR = (1.0, 0.0, 0.0)


def _step(ee, *, run_id, step):
    return RolloutStep(
        run_id=run_id,
        seed=0,
        git_commit=None,
        suite="libero_spatial",
        task_id="t0",
        step=step,
        observation_ref=f"{run_id}/{step}",
        action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        privileged_state={
            "ee_pos": ee,
            "gripper_open": True,  # no grasp transition -> P3 = 0
            "object_poses": {"goal": _ANCHOR},  # single object -> no distractor (P2 = 0)
            "target_region": "goal",
        },
        instruction="pick the goal",
        trusted_goal="pick the goal",
        attacked=False,
        suffix_ref=None,
    )


def _toward(i):
    """A benign rollout: EE moves toward the anchor -> P1 ~ 0 -> score ~ 0."""
    rid = f"benign{i}"
    return Rollout(steps=(_step((0.0, 0.0, 0.0), run_id=rid, step=0),
                          _step((0.5, 0.0, 0.0), run_id=rid, step=1)))


def _away(i):
    """An attacked rollout: EE moves away from the anchor -> P1 ~ 1 -> score ~ 1."""
    rid = f"attacked{i}"
    return Rollout(steps=(_step((0.5, 0.0, 0.0), run_id=rid, step=0),
                          _step((0.0, 0.0, 0.0), run_id=rid, step=1)))


def test_per_rollout_score_is_max_over_steps():
    assert per_rollout_score([Score(0.1, 0), Score(0.9, 1), Score(0.3, 2)]) == pytest.approx(0.9)


def test_clearly_separated_gives_high_tpr():
    calib = [_toward(i) for i in range(4)]
    benign_eval = [_toward(i) for i in range(4, 8)]
    attacked = [_away(i) for i in range(4)]

    table = separation_table(
        calib, benign_eval, attacked, schema=SchemaA(), fpr=0.05, trusted_goal="pick the goal"
    )
    row = table.rows[0]
    op = row.operating_points[0]
    assert op.fpr_target == pytest.approx(0.05)
    assert op.tpr == pytest.approx(1.0)       # attacked (~1) all exceed tau
    assert op.realised_fpr == pytest.approx(0.0)  # benign (~0) never fire on held-out
    assert row.auc == pytest.approx(1.0)


def test_tau_from_calibration_fpr_from_heldout(): # invariant #3: splits stay separate
    calib = [_toward(i) for i in range(6)]
    benign_eval = [_toward(i) for i in range(6, 10)]
    attacked = [_away(i) for i in range(3)]
    table = separation_table(calib, benign_eval, attacked, schema=SchemaA(), fpr=0.05)
    op = table.rows[0].operating_points[0]
    assert op.n_benign == len(benign_eval)        # realised_fpr measured on held-out
    assert op.n_benign_calib == len(calib)        # tau calibrated on calibration split
    assert op.n_attacked == len(attacked)


def test_degenerate_benign_equals_attacked_is_chance_no_crash():
    # attacked stream identical to benign -> ~chance AUC, no separation, no crash.
    calib = [_toward(i) for i in range(4)]
    benign_eval = [_toward(i) for i in range(4, 8)]
    attacked = [_toward(i) for i in range(8, 12)]  # benign-like
    table = separation_table(calib, benign_eval, attacked, schema=SchemaA(), fpr=0.05)
    row = table.rows[0]
    assert row.auc == pytest.approx(0.5)
    assert row.operating_points[0].tpr == pytest.approx(0.0)


def test_empty_attacked_raises():
    calib = [_toward(0)]
    benign_eval = [_toward(1)]
    with pytest.raises(ValueError, match="attacked"):
        separation_table(calib, benign_eval, [], schema=SchemaA(), fpr=0.05)

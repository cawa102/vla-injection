"""Tests for the goal-agnostic action-anomaly baseline (Task 8).

This is the **mandatory** M2 comparison detector (plan invariant #4 / playbook
§5): a fully model-free OOD score on the action stream that **never sees the
goal**. Its purpose is to show that metric (A)'s *goal-conditioning* buys
detection power that mere out-of-distribution anomaly does not. These tests pin:

* one score per step, all in ``[0, 1]``;
* far-out-of-distribution action streams score higher than in-distribution ones;
* the score is **goal-agnostic** — identical when only ``trusted_goal`` /
  ``target_region`` change (the property that distinguishes it from metric (A));
* the score is **causal** — score at ``t`` is unchanged by future steps;
* it calibrates through the *same* ``calibrate`` baselines must reuse.
"""

import dataclasses

import numpy as np
import pytest

from t7.baselines.anomaly import BenignActionStats, goal_agnostic_anomaly_score
from t7.detector.calibrate import calibrate
from t7.detector.decide import rollout_fires
from t7.records import Rollout, RolloutStep, Score


def make_step(action, *, step=0, trusted_goal="reach the cube", target_region="cube"):
    """A minimal RolloutStep carrying just what the anomaly baseline reads.

    The anomaly baseline must ignore everything goal-related, so we still set a
    goal / privileged target to prove it is not consulted.
    """
    return RolloutStep(
        run_id="test",
        seed=0,
        git_commit=None,
        suite="libero_spatial",
        task_id="t0",
        step=step,
        observation_ref=f"obs/{step}",
        action=tuple(float(x) for x in action),
        privileged_state={
            "ee_pos": (0.0, 0.0, 0.0),
            "gripper_open": True,
            "object_poses": {target_region: (0.0, 0.0, 0.0)},
            "target_region": target_region,
        },
        instruction=trusted_goal,
        trusted_goal=trusted_goal,
        attacked=False,
        suffix_ref=None,
    )


def rollout_from_actions(actions, **step_kwargs):
    return Rollout(
        steps=tuple(make_step(a, step=i, **step_kwargs) for i, a in enumerate(actions))
    )


def benign_rollout(n_steps, *, seed, scale=0.05):
    """A benign rollout: small zero-mean action deltas (typical LIBERO motion)."""
    rng = np.random.default_rng(seed)
    actions = rng.normal(0.0, scale, size=(n_steps, 7))
    return rollout_from_actions(actions)


def benign_stats_from(n_rollouts, n_steps, *, seed, scale=0.05):
    rollouts = [benign_rollout(n_steps, seed=seed + i, scale=scale) for i in range(n_rollouts)]
    return BenignActionStats.from_rollouts(rollouts), rollouts


# --------------------------------------------------------------------------- #
# BenignActionStats                                                           #
# --------------------------------------------------------------------------- #


def test_benign_stats_from_rollouts_has_per_dim_mean_and_std():
    stats, _ = benign_stats_from(20, 30, seed=0)
    assert len(stats.mean) == 7
    assert len(stats.std) == 7
    # Zero-mean benign data → mean near 0, std near the generating scale.
    assert np.allclose(stats.mean, 0.0, atol=0.02)
    assert np.allclose(stats.std, 0.05, atol=0.02)


def test_benign_stats_rejects_empty_rollouts():
    with pytest.raises(ValueError):
        BenignActionStats.from_rollouts([])


def test_benign_stats_is_immutable():
    stats, _ = benign_stats_from(5, 10, seed=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.mean = (0,) * 7  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Shape / range                                                               #
# --------------------------------------------------------------------------- #


def test_returns_one_score_per_step_in_unit_interval():
    stats, _ = benign_stats_from(20, 30, seed=2)
    roll = benign_rollout(15, seed=999)
    scores = goal_agnostic_anomaly_score(roll, benign_stats=stats)
    assert len(scores) == len(roll)
    assert all(isinstance(s, Score) for s in scores)
    assert all(0.0 <= s.value <= 1.0 for s in scores)
    assert [s.window_end for s in scores] == list(range(len(roll)))


def test_empty_rollout_returns_no_scores():
    stats, _ = benign_stats_from(5, 10, seed=3)
    assert goal_agnostic_anomaly_score(Rollout(steps=()), benign_stats=stats) == []


# --------------------------------------------------------------------------- #
# Core behaviour: OOD scores higher than in-distribution                       #
# --------------------------------------------------------------------------- #


def test_out_of_distribution_actions_score_higher_than_benign():
    stats, benign = benign_stats_from(40, 30, seed=4)
    # An attacked-like stream: actions pushed many std out (a redirected action).
    ood = rollout_from_actions(np.full((30, 7), 1.0))  # ~20 std out at scale 0.05
    benign_max = max(s.value for s in goal_agnostic_anomaly_score(benign[0], benign_stats=stats))
    ood_max = max(s.value for s in goal_agnostic_anomaly_score(ood, benign_stats=stats))
    assert ood_max > benign_max
    assert ood_max > 0.9  # clearly flagged


def test_action_at_benign_mean_scores_low():
    stats, _ = benign_stats_from(40, 30, seed=5)
    at_mean = rollout_from_actions([stats.mean])  # exactly the benign mean
    (score,) = goal_agnostic_anomaly_score(at_mean, benign_stats=stats)
    assert score.value < 0.5


def test_degenerate_benign_with_no_variation_abstains_with_zero_scores():
    # Benign actions never vary → no distribution to be anomalous against; the
    # baseline honestly abstains (all scores 0) rather than firing arbitrarily.
    constant = [rollout_from_actions(np.zeros((10, 7))) for _ in range(5)]
    stats = BenignActionStats.from_rollouts(constant)
    assert all(s == 0.0 for s in stats.std)
    ood = rollout_from_actions(np.full((4, 7), 9.0))
    scores = goal_agnostic_anomaly_score(ood, benign_stats=stats)
    assert [s.value for s in scores] == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Goal-agnostic: ignores goal / privileged target (the defining property)      #
# --------------------------------------------------------------------------- #


def test_score_is_independent_of_trusted_goal_and_target_region():
    stats, _ = benign_stats_from(20, 30, seed=6)
    actions = np.random.default_rng(7).normal(0.3, 0.1, size=(12, 7))
    roll_a = rollout_from_actions(actions, trusted_goal="reach the cube", target_region="cube")
    roll_b = rollout_from_actions(actions, trusted_goal="reach the plate", target_region="plate")
    sa = [s.value for s in goal_agnostic_anomaly_score(roll_a, benign_stats=stats)]
    sb = [s.value for s in goal_agnostic_anomaly_score(roll_b, benign_stats=stats)]
    assert sa == sb  # only the actions matter; the goal is never read


# --------------------------------------------------------------------------- #
# Causality                                                                    #
# --------------------------------------------------------------------------- #


def test_score_at_t_is_unaffected_by_future_steps():
    stats, _ = benign_stats_from(20, 30, seed=8)
    actions = np.random.default_rng(9).normal(0.2, 0.2, size=(10, 7))
    full = rollout_from_actions(actions)
    truncated = rollout_from_actions(actions[:6])
    full_scores = goal_agnostic_anomaly_score(full, benign_stats=stats)
    trunc_scores = goal_agnostic_anomaly_score(truncated, benign_stats=stats)
    for t in range(len(truncated)):
        assert full_scores[t].value == trunc_scores[t].value


# --------------------------------------------------------------------------- #
# Reuses the shared calibrate (invariant #4)                                   #
# --------------------------------------------------------------------------- #


def test_calibrates_through_shared_calibrate_and_flags_attack():
    stats, benign = benign_stats_from(200, 20, seed=10)
    calib = [goal_agnostic_anomaly_score(r, benign_stats=stats) for r in benign]
    thr = calibrate(calib, target_per_rollout_fpr=0.05)

    ood = rollout_from_actions(np.full((20, 7), 1.0))
    ood_scores = goal_agnostic_anomaly_score(ood, benign_stats=stats)
    assert rollout_fires(ood_scores, thr.tau).hold

    # And the benign calibration set stays at or below its budget.
    fired = sum(rollout_fires(c, thr.tau).hold for c in calib)
    assert fired / len(calib) <= 0.05


def test_does_not_mutate_rollout_or_stats():
    stats, _ = benign_stats_from(10, 10, seed=11)
    mean_before = stats.mean
    roll = benign_rollout(8, seed=12)
    actions_before = roll.actions().copy()
    goal_agnostic_anomaly_score(roll, benign_stats=stats)
    assert stats.mean == mean_before
    assert np.array_equal(roll.actions(), actions_before)

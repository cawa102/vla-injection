"""Tests for consistency metric (A) — frozen schema + causal scorer (Task 5).

These tests are the executable form of the frozen annotation schema
(``docs/plans/metric-a-annotation-schema.md``). They use *only* synthetic
fixtures (no attack output exists at freeze time — circularity guard). Geometry
is hand-checkable: a single goal object the end-effector approaches, plus a
distractor it can engage "by mistake".
"""

from __future__ import annotations

import dataclasses

import pytest

from t7.metric.consistency_a import (
    ConsistencyMetricA,
    GoalAnchor,
    PrivilegedGoalResolver,
    SchemaA,
    Semantics,
)
from t7.metric.state import PrivilegedState
from t7.records import Rollout, RolloutStep, Score

# --- Hand-checkable object layout (matches fixtures_state geometry) -------------
GOAL = "red_block"
WRONG = "blue_bowl"
GOAL_POS = (0.50, 0.10, 0.05)
WRONG_POS = (0.50, -0.30, 0.05)
OBJECTS = {GOAL: GOAL_POS, WRONG: WRONG_POS}

R = 0.05  # engagement_radius default
RG = 0.10  # grasp_radius default


def _priv(ee_pos, *, gripper_open=True, target_region: str | None = GOAL, objects=None):
    """A privileged_state dict in the normalised PrivilegedState schema."""
    return {
        "ee_pos": ee_pos,
        "gripper_open": gripper_open,
        "object_poses": dict(OBJECTS if objects is None else objects),
        "target_region": target_region,
    }


def _step(ee_pos, *, step, gripper_open=True, target_region: str | None = GOAL, objects=None):
    return RolloutStep(
        run_id="t",
        seed=0,
        git_commit=None,
        suite="libero_spatial",
        task_id="task0",
        step=step,
        observation_ref="o",
        action=(0.0,) * 7,
        privileged_state=_priv(
            ee_pos,
            gripper_open=gripper_open,
            target_region=target_region,
            objects=objects,
        ),
        instruction="i",
        trusted_goal="pick up the red block",
        attacked=False,
        suffix_ref=None,
    )


def _rollout(steps):
    return Rollout(steps=tuple(steps))


def _metric(k=5, combination="max"):
    return ConsistencyMetricA(schema=SchemaA(combination=combination), k=k)


def _anchor():
    return GoalAnchor(position=GOAL_POS, object_name=GOAL)


# ============================================================================ #
# SchemaA / records — defaults + immutability                                  #
# ============================================================================ #


def test_schema_defaults_match_frozen_doc():
    s = SchemaA()
    assert s.engagement_radius == pytest.approx(0.05)
    assert s.grasp_radius == pytest.approx(0.10)
    assert s.combination == "max"
    assert s.primitives == (
        "progress",
        "distractor_engagement",
        "grasp_appropriateness",
    )


def test_schema_and_semantics_and_anchor_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SchemaA().combination = "noisy_or"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Semantics(0.0, 0.0, 0.0).progress = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _anchor().object_name = "x"  # type: ignore[misc]


def test_schema_rejects_unknown_combination():
    with pytest.raises(ValueError):
        ConsistencyMetricA(schema=SchemaA(combination="bogus"), k=5)


def test_metric_rejects_bad_k():
    with pytest.raises(ValueError):
        ConsistencyMetricA(schema=SchemaA(), k=0)


# ============================================================================ #
# PrivilegedGoalResolver                                                        #
# ============================================================================ #


def test_resolver_returns_anchor_at_target_region_object():
    state = PrivilegedState(
        ee_pos=(0, 0, 0), gripper_open=True, object_poses=OBJECTS, target_region=GOAL
    )
    anchor = PrivilegedGoalResolver().resolve(_step((0, 0, 0), step=0), state)
    assert anchor is not None
    assert anchor.object_name == GOAL
    assert anchor.position == pytest.approx(GOAL_POS)


@pytest.mark.parametrize("bad_target", [None, "not_a_known_object"])
def test_resolver_unresolvable_returns_none(bad_target):
    state = PrivilegedState(
        ee_pos=(0, 0, 0),
        gripper_open=True,
        object_poses=OBJECTS,
        target_region=bad_target,
    )
    assert PrivilegedGoalResolver().resolve(_step((0, 0, 0), step=0), state) is None


# ============================================================================ #
# P1 — progress / directional alignment                                        #
# ============================================================================ #


def test_p1_low_when_moving_toward_goal():
    # start far, end nearer the goal (but still outside r): displacement ~ toward.
    prefix = (
        _step((0.20, 0.10, 0.30), step=0),
        _step((0.50, 0.10, 0.20), step=1),
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.progress < 0.5


def test_p1_high_when_moving_away_from_goal():
    prefix = (
        _step((0.50, 0.10, 0.20), step=0),
        _step((0.20, 0.10, 0.40), step=1),
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.progress > 0.5


def test_p1_zero_when_at_goal_regardless_of_direction():
    # end within r of the goal → transit direction irrelevant.
    prefix = (
        _step((0.20, 0.10, 0.30), step=0),
        _step((0.50, 0.10, 0.08), step=1),  # d_end = 0.03 < r
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.progress == pytest.approx(0.0)


def test_p1_zero_when_no_motion_and_not_at_goal():
    prefix = (
        _step((0.20, 0.10, 0.30), step=0),
        _step((0.20, 0.10, 0.30), step=1),  # identical → ‖Δ‖ < ε
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.progress == pytest.approx(0.0)


# ============================================================================ #
# P2 — distractor engagement                                                   #
# ============================================================================ #


def test_p2_high_when_engaging_distractor_far_from_goal():
    # end hovering just above the wrong object.
    prefix = (
        _step((0.40, -0.10, 0.20), step=0),
        _step((0.50, -0.30, 0.07), step=1),  # d_dist ~0.02 <= r, d_goal ~0.4 > r
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.distractor_engagement > 0.9


def test_p2_zero_when_near_goal():
    prefix = (
        _step((0.40, 0.10, 0.20), step=0),
        _step((0.50, 0.10, 0.07), step=1),  # near goal, far from distractor
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.distractor_engagement == pytest.approx(0.0)


def test_p2_zero_when_no_distractors():
    only_goal = {GOAL: GOAL_POS}
    prefix = (
        _step((0.40, -0.10, 0.20), step=0, objects=only_goal),
        _step((0.50, -0.30, 0.07), step=1, objects=only_goal),
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.distractor_engagement == pytest.approx(0.0)


# ============================================================================ #
# P3 — grasp-event appropriateness                                             #
# ============================================================================ #


def test_p3_zero_without_a_grasp_event():
    # gripper stays open throughout → no open->close transition.
    prefix = (
        _step((0.50, 0.10, 0.20), step=0, gripper_open=True),
        _step((0.50, 0.10, 0.07), step=1, gripper_open=True),
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.grasp_appropriateness == pytest.approx(0.0)


def test_p3_low_for_grasp_at_goal():
    prefix = (
        _step((0.50, 0.10, 0.20), step=0, gripper_open=True),
        _step((0.50, 0.10, 0.05), step=1, gripper_open=False),  # grasp AT goal (d=0)
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.grasp_appropriateness == pytest.approx(0.0)


def test_p3_high_for_grasp_at_distractor():
    prefix = (
        _step((0.50, -0.30, 0.20), step=0, gripper_open=True),
        _step((0.50, -0.30, 0.05), step=1, gripper_open=False),  # grasp at distractor
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    assert sem.grasp_appropriateness > 0.9


# ============================================================================ #
# Combination rule (decision 2)                                                #
# ============================================================================ #


def test_combination_max_is_default_and_takes_worst_primitive():
    # craft a prefix that trips P2 strongly; max == P2.
    prefix = (
        _step((0.40, -0.10, 0.20), step=0),
        _step((0.50, -0.30, 0.07), step=1),
    )
    m = _metric(combination="max")
    sem = m.extract_semantics(prefix, _anchor())
    score = m.score(1, _rollout(prefix), "g")
    assert score.value == pytest.approx(
        max(sem.progress, sem.distractor_engagement, sem.grasp_appropriateness)
    )


def test_combination_noisy_or_and_mean_match_formulas():
    prefix = (
        _step((0.40, -0.10, 0.20), step=0),
        _step((0.50, -0.30, 0.07), step=1),
    )
    sem = _metric().extract_semantics(prefix, _anchor())
    ps = (sem.progress, sem.distractor_engagement, sem.grasp_appropriateness)

    noisy = ConsistencyMetricA(SchemaA(combination="noisy_or"), k=5).score(
        1, _rollout(prefix), "g"
    )
    expected_noisy = 1.0 - (1 - ps[0]) * (1 - ps[1]) * (1 - ps[2])
    assert noisy.value == pytest.approx(expected_noisy)

    mean = ConsistencyMetricA(SchemaA(combination="weighted_mean"), k=5).score(
        1, _rollout(prefix), "g"
    )
    assert mean.value == pytest.approx(sum(ps) / 3.0)


# ============================================================================ #
# score / score_rollout — shape, range, window_end                            #
# ============================================================================ #


def test_score_returns_valid_score_with_window_end_equal_step_index():
    prefix = (
        _step((0.20, 0.10, 0.30), step=0),
        _step((0.50, 0.10, 0.20), step=1),
    )
    score = _metric().score(1, _rollout(prefix), "g")
    assert isinstance(score, Score)
    assert 0.0 <= score.value <= 1.0
    assert score.window_end == 1


def test_score_rollout_has_one_score_per_step_in_order():
    steps = [_step((0.20 + 0.05 * i, 0.10, 0.30), step=i) for i in range(6)]
    scores = _metric().score_rollout(_rollout(steps))
    assert [s.window_end for s in scores] == list(range(6))


def test_unresolvable_goal_scores_zero():
    steps = [
        _step((0.50, -0.30, 0.07), step=0, target_region=None),
        _step((0.50, -0.30, 0.07), step=1, target_region=None),
    ]
    scores = _metric().score_rollout(_rollout(steps))
    assert all(s.value == pytest.approx(0.0) for s in scores)


# ============================================================================ #
# Causality (invariant #1) + window length k                                   #
# ============================================================================ #


def test_score_is_causal_future_steps_do_not_change_score_at_t():
    base = [_step((0.20 + 0.05 * i, 0.10, 0.30), step=i) for i in range(4)]
    future = base + [_step((0.50, -0.30, 0.07), step=4)]  # later veers to distractor
    s_short = _metric().score(3, _rollout(base), "g")
    s_long = _metric().score(3, _rollout(future), "g")
    assert s_short.value == pytest.approx(s_long.value)


def test_k_controls_how_many_past_steps_inform_the_score():
    # A long approach then a single away-step at the end. With k=2 only the last
    # away-step pair is seen (progress high); with a large k the net trend over
    # the whole approach dominates (progress lower).
    steps = [_step((0.20 + 0.04 * i, 0.10, 0.30), step=i) for i in range(6)]
    steps.append(_step((0.20, 0.10, 0.45), step=6))  # jump away from goal
    roll = _rollout(steps)
    s_k2 = ConsistencyMetricA(SchemaA(), k=2).score(6, roll, "g")
    s_k7 = ConsistencyMetricA(SchemaA(), k=7).score(6, roll, "g")
    # Small k sees only the final away-step pair (progress high); large k lets the
    # long approach dampen the net displacement → strictly lower.
    assert s_k2.value > s_k7.value


def test_score_uses_exactly_the_causal_window_steps():
    # Pins window membership for score(t=4, k=3): exactly steps[2:5], not [1:5]/[3:5].
    steps = [_step((0.20 + 0.05 * i, 0.10, 0.30), step=i) for i in range(6)]
    roll = _rollout(steps)
    m = _metric(k=3)
    sem = m.extract_semantics(roll.steps[2:5], _anchor())
    expected = max(
        sem.progress, sem.distractor_engagement, sem.grasp_appropriateness
    )
    assert m.score(4, roll, "g").value == pytest.approx(expected)


# ============================================================================ #
# Monitoring ceiling (non-causal) — separately labelled method                 #
# ============================================================================ #


def test_monitoring_ceiling_returns_one_score_per_step():
    steps = [_step((0.20 + 0.05 * i, 0.10, 0.30), step=i) for i in range(5)]
    scores = _metric().score_rollout_monitoring_ceiling(_rollout(steps))
    assert [s.window_end for s in scores] == list(range(5))


def test_monitoring_ceiling_is_a_true_upper_bound_on_causal():
    # By construction the ceiling at t is the max causal score over a centred
    # neighbourhood, so ceiling[t] >= causal[t] for every t.
    steps = [
        _step((0.30, 0.10, 0.30), step=0),
        _step((0.40, 0.10, 0.25), step=1),
        _step((0.45, 0.10, 0.20), step=2),
        _step((0.50, -0.20, 0.10), step=3),
        _step((0.50, -0.30, 0.07), step=4),
    ]
    roll = _rollout(steps)
    m = _metric(k=3)
    causal = m.score_rollout(roll)
    ceiling = m.score_rollout_monitoring_ceiling(roll)
    assert all(c.value >= s.value for c, s in zip(ceiling, causal))


def test_monitoring_ceiling_can_see_future_deviation_that_causal_misses():
    # Benign up to step 2, then veers to the distractor at steps 3-4. The causal
    # score at step 2 cannot see the future; the non-causal ceiling can.
    steps = [
        _step((0.30, 0.10, 0.30), step=0),
        _step((0.40, 0.10, 0.25), step=1),
        _step((0.45, 0.10, 0.20), step=2),
        _step((0.50, -0.20, 0.10), step=3),
        _step((0.50, -0.30, 0.07), step=4),
    ]
    roll = _rollout(steps)
    causal = _metric(k=3).score(2, roll, "g").value
    ceiling = _metric(k=3).score_rollout_monitoring_ceiling(roll)[2].value
    assert ceiling > causal


# ============================================================================ #
# extract_semantics — auditable in isolation (anti-circularity)                #
# ============================================================================ #


def test_extract_semantics_returns_zeroed_on_unresolved_anchor():
    prefix = (_step((0.50, -0.30, 0.07), step=0),)
    sem = _metric().extract_semantics(prefix, None)
    assert sem == Semantics(0.0, 0.0, 0.0)

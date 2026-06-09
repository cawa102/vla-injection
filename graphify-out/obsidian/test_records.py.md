---
source_file: "tests/evasion_tax/test_records.py"
type: "code"
community: "Rollout Records Tests"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Rollout_Records_Tests
---

# test_records.py

## Connections
- [[Build a RolloutStep with sensible defaults; action defaults to a 7-vector.]] - `defined_in` [EXTRACTED]
- [[Decision (shared record)]] - `references` [EXTRACTED]
- [[Rollout_9]] - `defined_in` [EXTRACTED]
- [[Rollout (shared record)]] - `references` [EXTRACTED]
- [[RolloutStep_3]] - `defined_in` [EXTRACTED]
- [[RolloutStep (shared record)]] - `references` [EXTRACTED]
- [[Score (shared record)]] - `references` [EXTRACTED]
- [[TargetActionSpec (shared record)]] - `references` [EXTRACTED]
- [[Tests for the core immutable data records (Task 2).  Covers frozen-record immut]] - `rationale_for` [EXTRACTED]
- [[make_rollout()_2]] - `contains` [EXTRACTED]
- [[make_step()_1]] - `contains` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[test_actions_has_shape_n_by_7()]] - `contains` [EXTRACTED]
- [[test_actions_values_match_steps()]] - `contains` [EXTRACTED]
- [[test_decision_fields_and_immutability()]] - `contains` [EXTRACTED]
- [[test_prefix_window_clamped_at_start_when_t_less_than_k_minus_1()]] - `contains` [EXTRACTED]
- [[test_prefix_window_k_less_than_one_raises()]] - `contains` [EXTRACTED]
- [[test_prefix_window_mid_rollout_returns_exactly_k_steps_ending_at_t()]] - `contains` [EXTRACTED]
- [[test_prefix_window_never_includes_future_steps()]] - `contains` [EXTRACTED]
- [[test_prefix_window_returns_tuple()]] - `contains` [EXTRACTED]
- [[test_prefix_window_t_out_of_range_raises()]] - `contains` [EXTRACTED]
- [[test_prefix_window_t_zero_returns_only_first_step()]] - `contains` [EXTRACTED]
- [[test_rollout_is_immutable()]] - `contains` [EXTRACTED]
- [[test_rollout_len()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_action_accepts_numpy_array()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_action_length_seven_accepted_and_stored_as_float_tuple()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_action_non_numeric_raises()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_action_too_long_raises()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_action_too_short_raises()]] - `contains` [EXTRACTED]
- [[test_rolloutstep_is_immutable()]] - `contains` [EXTRACTED]
- [[test_score_accepts_value_in_range()]] - `contains` [EXTRACTED]
- [[test_score_is_immutable()]] - `contains` [EXTRACTED]
- [[test_score_rejects_value_above_one()]] - `contains` [EXTRACTED]
- [[test_score_rejects_value_below_zero()]] - `contains` [EXTRACTED]
- [[test_target_spec_dim_out_of_range_raises()]] - `contains` [EXTRACTED]
- [[test_target_spec_is_immutable()]] - `contains` [EXTRACTED]
- [[test_target_spec_low_above_high_raises()]] - `contains` [EXTRACTED]
- [[test_target_spec_mismatched_low_high_dims_lengths_raise()]] - `contains` [EXTRACTED]
- [[test_target_spec_persistence_steps_less_than_one_raises()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_false_outside_region()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_inclusive_bounds()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_true_inside_region()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_accepts_numpy_array()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_persistence_greater_than_n_is_false()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_persistence_override()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_requires_consecutive_hits()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_run_of_exactly_persistence_true()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_scattered_hits_false()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_step_agrees_with_reached_window()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_step_first_completion_after_reset()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_step_none_when_never_sustained()]] - `contains` [EXTRACTED]
- [[test_target_spec_reached_window_step_returns_completion_index()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Rollout_Records_Tests

## 📄 Source

`tests/evasion_tax/test_records.py`

```python
"""Tests for the core immutable data records (Task 2).

Covers: frozen-record immutability, action-length validation/coercion, causal
start-clamped prefix windows, target-action-region reach + consecutive-persistence
window scoring, and Score value-range validation.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from evasion_tax.records import Decision, Rollout, RolloutStep, Score, TargetActionSpec


def make_step(step: int, action=None, **overrides) -> RolloutStep:
    """Build a RolloutStep with sensible defaults; action defaults to a 7-vector."""
    if action is None:
        action = (float(step),) * 7
    fields = {
        "run_id": "run-1",
        "seed": 0,
        "git_commit": "abc123",
        "suite": "libero_spatial",
        "task_id": "task-0",
        "step": step,
        "observation_ref": f"obs/{step}.png",
        "action": action,
        "privileged_state": {"ee_pose": [0.0] * 7},
        "instruction": "pick up the bowl",
        "trusted_goal": "pick up the bowl",
        "attacked": False,
        "suffix_ref": None,
    }
    fields.update(overrides)
    return RolloutStep(**fields)


# --------------------------------------------------------------------------- #
# RolloutStep                                                                  #
# --------------------------------------------------------------------------- #


def test_rolloutstep_is_immutable():
    step = make_step(0)
    with pytest.raises(FrozenInstanceError):
        step.step = 5  # type: ignore[misc]


def test_rolloutstep_action_length_seven_accepted_and_stored_as_float_tuple():
    step = make_step(0, action=[1, 2, 3, 4, 5, 6, 7])
    assert step.action == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert isinstance(step.action, tuple)
    assert all(isinstance(x, float) for x in step.action)


def test_rolloutstep_action_accepts_numpy_array():
    step = make_step(0, action=np.arange(7, dtype=np.float64))
    assert step.action == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert isinstance(step.action, tuple)


def test_rolloutstep_action_too_short_raises():
    with pytest.raises(ValueError, match="length 7"):
        make_step(0, action=(1.0, 2.0, 3.0))


def test_rolloutstep_action_too_long_raises():
    with pytest.raises(ValueError, match="length 7"):
        make_step(0, action=tuple(range(8)))


def test_rolloutstep_action_non_numeric_raises():
    with pytest.raises(ValueError):
        make_step(0, action=("a", "b", "c", "d", "e", "f", "g"))


# --------------------------------------------------------------------------- #
# Rollout.prefix_window — causal + start-clamped                              #
# --------------------------------------------------------------------------- #


def make_rollout(n: int) -> Rollout:
    return Rollout(steps=tuple(make_step(i) for i in range(n)))


def test_rollout_len():
    assert len(make_rollout(5)) == 5


def test_prefix_window_clamped_at_start_when_t_less_than_k_minus_1():
    rollout = make_rollout(10)
    # t=1, k=4 -> start = max(0, 1-4+1) = 0 -> indices 0..1 (only available prefix).
    window = rollout.prefix_window(t=1, k=4)
    assert [s.step for s in window] == [0, 1]


def test_prefix_window_t_zero_returns_only_first_step():
    rollout = make_rollout(10)
    window = rollout.prefix_window(t=0, k=5)
    assert [s.step for s in window] == [0]


def test_prefix_window_mid_rollout_returns_exactly_k_steps_ending_at_t():
    rollout = make_rollout(10)
    # t=6, k=3 -> indices 4,5,6.
    window = rollout.prefix_window(t=6, k=3)
    assert [s.step for s in window] == [4, 5, 6]
    assert len(window) == 3


def test_prefix_window_never_includes_future_steps():
    rollout = make_rollout(10)
    window = rollout.prefix_window(t=5, k=20)
    # Start clamped to 0; end is exactly t -> no index > 5.
    assert [s.step for s in window] == [0, 1, 2, 3, 4, 5]
    assert max(s.step for s in window) == 5


def test_prefix_window_returns_tuple():
    rollout = make_rollout(10)
    assert isinstance(rollout.prefix_window(t=3, k=2), tuple)


def test_prefix_window_t_out_of_range_raises():
    rollout = make_rollout(5)
    with pytest.raises(ValueError, match="out of range"):
        rollout.prefix_window(t=5, k=2)
    with pytest.raises(ValueError, match="out of range"):
        rollout.prefix_window(t=-1, k=2)


def test_prefix_window_k_less_than_one_raises():
    rollout = make_rollout(5)
    with pytest.raises(ValueError, match="k"):
        rollout.prefix_window(t=2, k=0)


def test_rollout_is_immutable():
    rollout = make_rollout(3)
    with pytest.raises(FrozenInstanceError):
        rollout.steps = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Rollout.actions                                                             #
# --------------------------------------------------------------------------- #


def test_actions_has_shape_n_by_7():
    rollout = make_rollout(4)
    actions = rollout.actions()
    assert actions.shape == (4, 7)


def test_actions_values_match_steps():
    rollout = make_rollout(3)
    actions = rollout.actions()
    # make_step uses action = (step,)*7.
    np.testing.assert_array_equal(actions[2], np.full(7, 2.0))


# --------------------------------------------------------------------------- #
# TargetActionSpec                                                            #
# --------------------------------------------------------------------------- #


def test_target_spec_reached_true_inside_region():
    spec = TargetActionSpec(
        dims=(0, 2), low=(0.0, -1.0), high=(1.0, 1.0), persistence_steps=2
    )
    # dim 0 in [0,1], dim 2 in [-1,1]; other dims unconstrained.
    assert spec.reached((0.5, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0)) is True


def test_target_spec_reached_false_outside_region():
    spec = TargetActionSpec(
        dims=(0, 2), low=(0.0, -1.0), high=(1.0, 1.0), persistence_steps=2
    )
    # dim 0 = 2.0 is above high.
    assert spec.reached((2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) is False


def test_target_spec_reached_inclusive_bounds():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=1)
    assert spec.reached((0.0, 0, 0, 0, 0, 0, 0)) is True
    assert spec.reached((1.0, 0, 0, 0, 0, 0, 0)) is True


def test_target_spec_reached_window_requires_consecutive_hits():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=3)
    # Hits at 0,1,2 then miss, then hit -> scattered; longest run = 3 -> True.
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    actions = [in_region, in_region, in_region, out_region, in_region]
    assert spec.reached_window(actions) is True


def test_target_spec_reached_window_scattered_hits_false():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=3)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    # Hits never form a run of 3 consecutive.
    actions = [in_region, in_region, out_region, in_region, out_region, in_region]
    assert spec.reached_window(actions) is False


def test_target_spec_reached_window_run_of_exactly_persistence_true():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=2)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    actions = [out_region, in_region, in_region, out_region]
    assert spec.reached_window(actions) is True


def test_target_spec_reached_window_persistence_override():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=5)
    in_region = (0.5,) + (0.0,) * 6
    actions = [in_region, in_region]
    # Default persistence 5 -> False; override to 2 -> True.
    assert spec.reached_window(actions) is False
    assert spec.reached_window(actions, persistence=2) is True


def test_target_spec_reached_window_persistence_greater_than_n_is_false():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=10)
    in_region = (0.5,) + (0.0,) * 6
    actions = [in_region, in_region, in_region]
    assert spec.reached_window(actions) is False


def test_target_spec_reached_window_accepts_numpy_array():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=2)
    actions = np.zeros((3, 7))
    actions[:, 0] = 0.5
    assert spec.reached_window(actions) is True


def test_target_spec_reached_window_step_returns_completion_index():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=3)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    # First run of 3 completes at index 2.
    actions = [in_region, in_region, in_region, out_region, in_region]
    assert spec.reached_window_step(actions) == 2


def test_target_spec_reached_window_step_first_completion_after_reset():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=2)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    # Run resets at index 1; the first run of 2 completes at index 3.
    actions = [in_region, out_region, in_region, in_region, in_region]
    assert spec.reached_window_step(actions) == 3


def test_target_spec_reached_window_step_none_when_never_sustained():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=3)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    actions = [in_region, in_region, out_region, in_region, out_region]
    assert spec.reached_window_step(actions) is None


def test_target_spec_reached_window_step_agrees_with_reached_window():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=2)
    in_region = (0.5,) + (0.0,) * 6
    out_region = (5.0,) + (0.0,) * 6
    for actions in (
        [in_region, in_region],
        [in_region, out_region, in_region],
        [out_region, out_region],
    ):
        assert (spec.reached_window_step(actions) is not None) == spec.reached_window(actions)


def test_target_spec_mismatched_low_high_dims_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        TargetActionSpec(dims=(0, 1), low=(0.0,), high=(1.0, 1.0), persistence_steps=1)
    with pytest.raises(ValueError, match="same length"):
        TargetActionSpec(dims=(0, 1), low=(0.0, 0.0), high=(1.0,), persistence_steps=1)


def test_target_spec_dim_out_of_range_raises():
    with pytest.raises(ValueError, match="0..6"):
        TargetActionSpec(dims=(7,), low=(0.0,), high=(1.0,), persistence_steps=1)
    with pytest.raises(ValueError, match="0..6"):
        TargetActionSpec(dims=(-1,), low=(0.0,), high=(1.0,), persistence_steps=1)


def test_target_spec_persistence_steps_less_than_one_raises():
    with pytest.raises(ValueError, match="persistence_steps"):
        TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=0)


def test_target_spec_low_above_high_raises():
    with pytest.raises(ValueError, match="low"):
        TargetActionSpec(dims=(0,), low=(1.0,), high=(0.0,), persistence_steps=1)


def test_target_spec_is_immutable():
    spec = TargetActionSpec(dims=(0,), low=(0.0,), high=(1.0,), persistence_steps=1)
    with pytest.raises(FrozenInstanceError):
        spec.persistence_steps = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Score / Decision                                                            #
# --------------------------------------------------------------------------- #


def test_score_accepts_value_in_range():
    assert Score(value=0.0, window_end=0).value == 0.0
    assert Score(value=1.0, window_end=3).value == 1.0
    assert Score(value=0.5, window_end=2).value == 0.5


def test_score_rejects_value_below_zero():
    with pytest.raises(ValueError, match=r"\[0, ?1\]"):
        Score(value=-0.1, window_end=0)


def test_score_rejects_value_above_one():
    with pytest.raises(ValueError, match=r"\[0, ?1\]"):
        Score(value=1.1, window_end=0)


def test_score_is_immutable():
    score = Score(value=0.5, window_end=1)
    with pytest.raises(FrozenInstanceError):
        score.value = 0.9  # type: ignore[misc]


def test_decision_fields_and_immutability():
    decision = Decision(hold=True, step=4)
    assert decision.hold is True
    assert decision.step == 4
    with pytest.raises(FrozenInstanceError):
        decision.hold = False  # type: ignore[misc]
```


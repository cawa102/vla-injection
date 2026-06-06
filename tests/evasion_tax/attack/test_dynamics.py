"""Tests for the action→state dynamics seam (playbook §4b-(II)).

The metric-A oracle scores the *privileged-state trajectory* an action sequence
induces, not the raw actions — so the idealized attacker needs a dynamics model.
The optimiser is model-free, so this seam is the swap point:

* :class:`SyntheticDynamics` — a local kinematic integrator that emits a
  :class:`~evasion_tax.records.Rollout` whose ``privileged_state`` dicts match the
  :class:`~evasion_tax.metric.state.SyntheticStateAdapter` schema (so metric A can score
  it on the 8 GB host);
* :class:`RealDynamics` — the LIBERO-sim backend, a GPU-only stub.

These tests pin the integrator's determinism + arithmetic, the emitted schema,
the gripper mapping, and the scenario validation.
"""

import dataclasses

import numpy as np
import pytest

from evasion_tax.attack.dynamics import AttackScenario, Dynamics, RealDynamics, SyntheticDynamics
from evasion_tax.metric.state import SyntheticStateAdapter
from evasion_tax.records import ACTION_DIM, Rollout

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #


def make_scenario(*, seed=0, n_steps=4, target_region="cube"):
    """A minimal reach-the-cube scene with one distractor."""
    return AttackScenario(
        task_id="t0",
        trusted_goal="reach the cube",
        seed=seed,
        init_ee_pos=(0.0, 0.0, 0.0),
        gripper_open0=True,
        object_poses={"cube": (1.0, 0.0, 0.0), "block": (-1.0, 0.0, 0.0)},
        target_region=target_region,
        n_steps=n_steps,
    )


def const_actions(n_steps, vec):
    """An ``(n_steps, 7)`` array repeating ``vec`` (a length-7 action)."""
    return np.tile(np.asarray(vec, dtype=float), (n_steps, 1))


# --------------------------------------------------------------------------- #
# AttackScenario validation                                                   #
# --------------------------------------------------------------------------- #


def test_scenario_rejects_non_positive_n_steps():
    with pytest.raises(ValueError):
        make_scenario(n_steps=0)


def test_scenario_rejects_target_region_not_in_objects():
    with pytest.raises(ValueError):
        make_scenario(target_region="ghost")


def test_scenario_rejects_bad_position_length():
    with pytest.raises(ValueError):
        AttackScenario(
            task_id="t",
            trusted_goal="g",
            seed=0,
            init_ee_pos=(0.0, 0.0),  # type: ignore[arg-type]  # length 2 (runtime check)
            gripper_open0=True,
            object_poses={"cube": (1.0, 0.0, 0.0)},
            target_region="cube",
            n_steps=3,
        )


def test_scenario_is_immutable():
    s = make_scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_steps = 9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# SyntheticDynamics: seam conformance + structure                             #
# --------------------------------------------------------------------------- #


def test_synthetic_dynamics_conforms_to_protocol():
    assert isinstance(SyntheticDynamics(), Dynamics)


def test_synthetic_dynamics_returns_rollout_of_n_steps():
    dyn = SyntheticDynamics()
    s = make_scenario(n_steps=5)
    roll = dyn.rollout(s, const_actions(5, (0.1, 0, 0, 0, 0, 0, 1.0)))
    assert isinstance(roll, Rollout)
    assert len(roll.steps) == 5


def test_synthetic_dynamics_is_deterministic():
    dyn = SyntheticDynamics()
    s = make_scenario()
    a = const_actions(4, (0.2, -0.1, 0, 0, 0, 0, 1.0))
    assert dyn.rollout(s, a) == dyn.rollout(s, a)


def test_synthetic_dynamics_rejects_wrong_action_shape():
    dyn = SyntheticDynamics()
    s = make_scenario(n_steps=4)
    with pytest.raises(ValueError):
        dyn.rollout(s, const_actions(3, (0.1,) * ACTION_DIM))  # 3 != n_steps
    with pytest.raises(ValueError):
        dyn.rollout(s, np.zeros((4, 5)))  # width 5 != 7


# --------------------------------------------------------------------------- #
# SyntheticDynamics: emitted privileged-state schema + arithmetic             #
# --------------------------------------------------------------------------- #


def test_emitted_privileged_state_is_adapter_consumable():
    dyn = SyntheticDynamics()
    roll = dyn.rollout(make_scenario(), const_actions(4, (0.1, 0, 0, 0, 0, 0, 1.0)))
    adapter = SyntheticStateAdapter()
    for step in roll.steps:
        ps = adapter.to_privileged_state(step.privileged_state)  # must not raise
        assert ps.target_region == "cube"
        assert "cube" in ps.object_poses and "block" in ps.object_poses


def test_ee_pos_integrates_position_deltas_cumulatively():
    dyn = SyntheticDynamics(pos_scale=0.1)
    s = make_scenario(n_steps=3)
    # constant +x delta of 1.0 → ee_x grows by pos_scale each step.
    roll = dyn.rollout(s, const_actions(3, (1.0, 0, 0, 0, 0, 0, 1.0)))
    xs = [step.privileged_state["ee_pos"][0] for step in roll.steps]
    assert xs == pytest.approx([0.1, 0.2, 0.3])


def test_gripper_open_maps_from_gripper_dim():
    dyn = SyntheticDynamics(gripper_open_threshold=0.0)
    s = make_scenario(n_steps=2)
    # step 0 gripper dim = +1 (open), step 1 gripper dim = -1 (closed).
    actions = np.array(
        [[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, -1.0]], dtype=float
    )
    roll = dyn.rollout(s, actions)
    assert roll.steps[0].privileged_state["gripper_open"] is True
    assert roll.steps[1].privileged_state["gripper_open"] is False


def test_attacked_flag_defaults_true_and_is_overridable():
    dyn = SyntheticDynamics()
    s = make_scenario()
    a = const_actions(4, (0.1, 0, 0, 0, 0, 0, 1.0))
    assert all(step.attacked for step in dyn.rollout(s, a).steps)
    assert all(not step.attacked for step in dyn.rollout(s, a, attacked=False).steps)


# --------------------------------------------------------------------------- #
# RealDynamics: GPU-only stub                                                 #
# --------------------------------------------------------------------------- #


def test_real_dynamics_conforms_to_protocol():
    assert isinstance(RealDynamics(), Dynamics)


def test_real_dynamics_is_a_gpu_only_stub():
    s = make_scenario()
    with pytest.raises(NotImplementedError):
        RealDynamics().rollout(s, const_actions(4, (0.1,) * ACTION_DIM))

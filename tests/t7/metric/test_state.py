"""Tests for the privileged-state adapter + synthetic fixtures (Task 4)."""

import dataclasses

import fixtures_state
import pytest

from t7.metric.state import PrivilegedState, SyntheticStateAdapter

# --- SyntheticStateAdapter: fixtures produce well-formed PrivilegedState -------


@pytest.mark.parametrize(
    "fixture_fn",
    [
        fixtures_state.approaching_goal_fixture,
        fixtures_state.near_wrong_object_fixture,
    ],
)
def test_adapter_builds_well_formed_state_from_each_fixture(fixture_fn):
    adapter = SyntheticStateAdapter()
    state = adapter.to_privileged_state(fixture_fn())

    assert isinstance(state, PrivilegedState)
    assert len(state.ee_pos) == 3
    assert all(isinstance(v, float) for v in state.ee_pos)
    assert isinstance(state.gripper_open, bool)
    assert isinstance(state.object_poses, dict)
    for name, pos in state.object_poses.items():
        assert isinstance(name, str)
        assert len(pos) == 3
        assert all(isinstance(c, float) for c in pos)


def test_approaching_goal_fixture_targets_an_existing_object():
    state = fixtures_state.approaching_goal_state()
    assert state.target_region in state.object_poses


def test_near_wrong_object_state_differs_from_approaching_goal_state():
    near = fixtures_state.near_wrong_object_state()
    approaching = fixtures_state.approaching_goal_state()
    # Same named goal, but the end-effector is somewhere else.
    assert near.target_region == approaching.target_region
    assert near.ee_pos != approaching.ee_pos


@pytest.mark.parametrize("gripper_open", [True, False])
def test_gripper_open_and_closed_variants(gripper_open):
    state = fixtures_state.approaching_goal_state(gripper_open=gripper_open)
    assert state.gripper_open is gripper_open


def test_object_poses_is_real_dict_of_three_tuples_and_target_region_can_be_none():
    adapter = SyntheticStateAdapter()
    fixture = {
        "ee_pos": (0.0, 0.0, 0.0),
        "gripper_open": True,
        "object_poses": {"thing": (1.0, 2.0, 3.0)},
        "target_region": None,
    }
    state = adapter.to_privileged_state(fixture)
    assert state.object_poses == {"thing": (1.0, 2.0, 3.0)}
    assert state.target_region is None


# --- SyntheticStateAdapter: rejects malformed input ----------------------------


def test_adapter_rejects_missing_required_key():
    adapter = SyntheticStateAdapter()
    incomplete = {
        "ee_pos": (0.0, 0.0, 0.0),
        "gripper_open": True,
        "object_poses": {},
        # "target_region" missing
    }
    with pytest.raises((KeyError, ValueError)):
        adapter.to_privileged_state(incomplete)


def test_adapter_does_not_mutate_its_input():
    adapter = SyntheticStateAdapter()
    fixture = fixtures_state.approaching_goal_fixture()
    before = dict(fixture)
    adapter.to_privileged_state(fixture)
    assert fixture == before


# --- PrivilegedState: immutability + boundary validation -----------------------


def test_state_is_immutable():
    state = fixtures_state.approaching_goal_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.gripper_open = False  # type: ignore[misc]


def test_bad_ee_pos_length_raises():
    with pytest.raises(ValueError):
        PrivilegedState(
            ee_pos=(0.0, 0.0),  # length 2
            gripper_open=True,
            object_poses={},
            target_region=None,
        )


def test_bad_object_pose_value_length_raises():
    with pytest.raises(ValueError):
        PrivilegedState(
            ee_pos=(0.0, 0.0, 0.0),
            gripper_open=True,
            object_poses={"thing": (1.0, 2.0)},  # length 2
            target_region=None,
        )


def test_ee_pos_is_coerced_to_float_tuple():
    state = PrivilegedState(
        ee_pos=(0, 1, 2),  # ints
        gripper_open=True,
        object_poses={"thing": (1, 2, 3)},
        target_region=None,
    )
    assert state.ee_pos == (0.0, 1.0, 2.0)
    assert all(isinstance(v, float) for v in state.ee_pos)
    assert state.object_poses["thing"] == (1.0, 2.0, 3.0)
    assert all(isinstance(c, float) for c in state.object_poses["thing"])

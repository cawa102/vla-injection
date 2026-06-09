---
source_file: "tests/evasion_tax/metric/test_state_libero.py"
type: "code"
community: "LIBERO State Adapter Tests"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/LIBERO_State_Adapter_Tests
---

# test_state_libero.py

## Connections
- [[A throwaway step; ``PrivilegedGoalResolver`` is step-agnostic (anchors on state)]] - `defined_in` [EXTRACTED]
- [[LiberoStateAdapter]] - `references` [EXTRACTED]
- [[PrivilegedGoalResolver]] - `references` [EXTRACTED]
- [[RolloutStep_2]] - `defined_in` [EXTRACTED]
- [[Tests for the concrete LIBERO StateAdapter (``state_libero.py``).  Run against F]] - `rationale_for` [EXTRACTED]
- [[_dummy_step()]] - `contains` [EXTRACTED]
- [[_load()]] - `contains` [EXTRACTED]
- [[extract_ee_pos]] - `references` [EXTRACTED]
- [[extract_object_poses]] - `references` [EXTRACTED]
- [[gripper_open_from_qpos]] - `references` [EXTRACTED]
- [[libero_obs_goal_opendrawer.json]] - `shares_data_with` [EXTRACTED]
- [[libero_obs_spatial0.json]] - `shares_data_with` [EXTRACTED]
- [[opendrawer()]] - `contains` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[spatial0()]] - `contains` [EXTRACTED]
- [[target_region_from_obj_of_interest]] - `references` [EXTRACTED]
- [[test_adapter_builds_resolvable_state_spatial0()]] - `contains` [EXTRACTED]
- [[test_adapter_opendrawer_abstain_case()]] - `contains` [EXTRACTED]
- [[test_adapter_rejects_non_mapping()]] - `contains` [EXTRACTED]
- [[test_consistency_a.py]] - `semantically_similar_to` [INFERRED]
- [[test_extract_ee_pos_from_real_obs()]] - `contains` [EXTRACTED]
- [[test_extract_ee_pos_missing_raises()]] - `contains` [EXTRACTED]
- [[test_gripper_closed_when_qpos_near_zero()]] - `contains` [EXTRACTED]
- [[test_gripper_open_true_on_real_open_value()]] - `contains` [EXTRACTED]
- [[test_gripper_threshold_constant()]] - `contains` [EXTRACTED]
- [[test_object_poses_excludes_relative_and_robot_keys()]] - `contains` [EXTRACTED]
- [[test_resolver_abstains_on_region_without_pose()]] - `contains` [EXTRACTED]
- [[test_resolver_resolves_plate_anchor()]] - `contains` [EXTRACTED]
- [[test_target_region_empty_is_none()]] - `contains` [EXTRACTED]
- [[test_target_region_is_last_obj_of_interest_binary()]] - `contains` [EXTRACTED]
- [[test_target_region_unary_predicate()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/LIBERO_State_Adapter_Tests

## 📄 Source

`tests/evasion_tax/metric/test_state_libero.py`

```python
"""Tests for the concrete LIBERO StateAdapter (``state_libero.py``).

Run against FROZEN real-LIBERO obs fixtures (captured state-only from a live
``ControlEnv``; see ``fixtures/PROVENANCE.md``) — no LIBERO/robosuite import here,
so the suite stays in the core ``.venv``.
"""

import json
from pathlib import Path

import pytest

from evasion_tax.metric.consistency_a import PrivilegedGoalResolver
from evasion_tax.metric.state import PrivilegedState
from evasion_tax.metric.state_libero import (
    GRIPPER_OPEN_SUM_THRESHOLD,
    LiberoStateAdapter,
    extract_ee_pos,
    extract_object_poses,
    gripper_open_from_qpos,
    target_region_from_obj_of_interest,
)
from evasion_tax.records import RolloutStep

_FIX = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIX / name).read_text())


@pytest.fixture
def spatial0() -> dict:
    return _load("libero_obs_spatial0.json")


@pytest.fixture
def opendrawer() -> dict:
    return _load("libero_obs_goal_opendrawer.json")


# --- extract_ee_pos ----------------------------------------------------------
def test_extract_ee_pos_from_real_obs(spatial0):
    ee = extract_ee_pos(spatial0["obs"])
    assert len(ee) == 3
    assert ee == pytest.approx(
        (-0.2084646605658239, -8.939910281987004e-18, 1.1732794757296403)
    )


def test_extract_ee_pos_missing_raises():
    with pytest.raises((KeyError, ValueError)):
        extract_ee_pos({})


# --- gripper_open_from_qpos --------------------------------------------------
def test_gripper_open_true_on_real_open_value():
    # Panda open reset: sum|qpos| = 0.0417 > 0.04
    assert gripper_open_from_qpos([0.020833, -0.020833]) is True


def test_gripper_closed_when_qpos_near_zero():
    assert gripper_open_from_qpos([0.0, 0.0]) is False


def test_gripper_threshold_constant():
    assert GRIPPER_OPEN_SUM_THRESHOLD == 0.04


# --- extract_object_poses ----------------------------------------------------
def test_object_poses_excludes_relative_and_robot_keys(spatial0):
    poses = extract_object_poses(spatial0["obs"])
    assert set(poses) == {
        "akita_black_bowl_1",
        "akita_black_bowl_2",
        "cookies_1",
        "glazed_rim_porcelain_ramekin_1",
        "plate_1",
    }
    assert not any("_to_" in name for name in poses)  # no relative deltas
    assert "robot0_eef" not in poses
    for pos in poses.values():
        assert len(pos) == 3
        assert all(isinstance(v, float) for v in pos)


# --- target_region_from_obj_of_interest --------------------------------------
def test_target_region_is_last_obj_of_interest_binary(spatial0):
    assert target_region_from_obj_of_interest(spatial0["obj_of_interest"]) == "plate_1"


def test_target_region_unary_predicate(opendrawer):
    assert (
        target_region_from_obj_of_interest(opendrawer["obj_of_interest"])
        == "wooden_cabinet_1_middle_region"
    )


def test_target_region_empty_is_none():
    assert target_region_from_obj_of_interest([]) is None


# --- LiberoStateAdapter ------------------------------------------------------
def test_adapter_builds_resolvable_state_spatial0(spatial0):
    adapter = LiberoStateAdapter(spatial0["obj_of_interest"])
    state = adapter.to_privileged_state(spatial0["obs"])
    assert isinstance(state, PrivilegedState)
    assert state.gripper_open is True
    assert state.target_region == "plate_1"
    assert state.target_region in state.object_poses  # resolvable case


def test_adapter_opendrawer_abstain_case(opendrawer):
    adapter = LiberoStateAdapter(opendrawer["obj_of_interest"])
    state = adapter.to_privileged_state(opendrawer["obs"])
    assert state.target_region == "wooden_cabinet_1_middle_region"
    # abstract region has no pose obs -> not in object_poses -> metric abstains
    assert state.target_region not in state.object_poses


def test_adapter_rejects_non_mapping(spatial0):
    adapter = LiberoStateAdapter(spatial0["obj_of_interest"])
    with pytest.raises(TypeError):
        adapter.to_privileged_state([1, 2, 3])


# --- metric integration ------------------------------------------------------
def _dummy_step() -> RolloutStep:
    """A throwaway step; ``PrivilegedGoalResolver`` is step-agnostic (anchors on state)."""
    return RolloutStep(
        run_id="t", seed=0, git_commit=None, suite="libero_spatial", task_id="0",
        step=0, observation_ref="x", action=(0.0,) * 7, privileged_state={},
        instruction="i", trusted_goal="g", attacked=False, suffix_ref=None,
    )


def test_resolver_resolves_plate_anchor(spatial0):
    adapter = LiberoStateAdapter(spatial0["obj_of_interest"])
    state = adapter.to_privileged_state(spatial0["obs"])
    anchor = PrivilegedGoalResolver().resolve(_dummy_step(), state)
    assert anchor is not None
    assert anchor.object_name == "plate_1"
    assert anchor.position == state.object_poses["plate_1"]


def test_resolver_abstains_on_region_without_pose(opendrawer):
    adapter = LiberoStateAdapter(opendrawer["obj_of_interest"])
    state = adapter.to_privileged_state(opendrawer["obs"])
    assert PrivilegedGoalResolver().resolve(_dummy_step(), state) is None
```


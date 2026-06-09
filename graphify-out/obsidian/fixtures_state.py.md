---
source_file: "tests/evasion_tax/metric/fixtures_state.py"
type: "code"
community: "Synthetic State Fixtures"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Synthetic_State_Fixtures
---

# fixtures_state.py

## Connections
- [[Benign 'approaching the goal object' class`PrivilegedState`.]] - `defined_in` [EXTRACTED]
- [[Build a fixture dict with the standard two-object layout.]] - `defined_in` [EXTRACTED]
- [[Fixture dict end-effector hovering above the wrong (distractor) object.]] - `defined_in` [EXTRACTED]
- [[Fixture dict end-effector hovering just above the goal object.]] - `defined_in` [EXTRACTED]
- [[Near the wrong object' class`PrivilegedState`.]] - `defined_in` [EXTRACTED]
- [[PrivilegedState_1]] - `references` [EXTRACTED]
- [[PrivilegedState_2]] - `defined_in` [EXTRACTED]
- [[Reusable synthetic privileged-state fixtures (Task 4).  These are not pytest t]] - `rationale_for` [EXTRACTED]
- [[SyntheticStateAdapter]] - `references` [EXTRACTED]
- [[_base_fixture_dict()]] - `contains` [EXTRACTED]
- [[approaching_goal_fixture()]] - `contains` [EXTRACTED]
- [[approaching_goal_state()]] - `contains` [EXTRACTED]
- [[near_wrong_object_fixture()]] - `contains` [EXTRACTED]
- [[near_wrong_object_state()]] - `contains` [EXTRACTED]
- [[test_state.py]] - `references` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Synthetic_State_Fixtures

## 📄 Source

`tests/evasion_tax/metric/fixtures_state.py`

```python
"""Reusable synthetic privileged-state fixtures (Task 4).

These are *not* pytest tests — they are plain builder functions returning the
fixture ``dict`` shape that :class:`evasion_tax.metric.state.SyntheticStateAdapter`
consumes, plus convenience builders that return the resulting
:class:`evasion_tax.metric.state.PrivilegedState`. They are import-safe (no pytest-only
constructs) so Task 5's metric tests can reuse them by importing this module
by name (the test directory is on ``sys.path`` under importlib mode).

Geometry is deliberately trivial and hand-checkable: a single goal object the
end-effector approaches, plus a distractor object it can be near "by mistake".
"""

from __future__ import annotations

from evasion_tax.metric.state import PrivilegedState, SyntheticStateAdapter

# Fixed object layout reused across fixtures so coordinates are comparable.
_GOAL_OBJECT = "red_block"
_WRONG_OBJECT = "blue_bowl"
_GOAL_POS = (0.50, 0.10, 0.05)
_WRONG_POS = (0.50, -0.30, 0.05)

_ADAPTER = SyntheticStateAdapter()


def _base_fixture_dict(
    ee_pos: tuple[float, float, float],
    *,
    gripper_open: bool,
    target_region: str | None,
) -> dict:
    """Build a fixture dict with the standard two-object layout."""
    return {
        "ee_pos": ee_pos,
        "gripper_open": gripper_open,
        "object_poses": {
            _GOAL_OBJECT: _GOAL_POS,
            _WRONG_OBJECT: _WRONG_POS,
        },
        "target_region": target_region,
    }


def approaching_goal_fixture(*, gripper_open: bool = True) -> dict:
    """Fixture dict: end-effector hovering just above the goal object."""
    ee_pos = (_GOAL_POS[0], _GOAL_POS[1], _GOAL_POS[2] + 0.02)
    return _base_fixture_dict(
        ee_pos, gripper_open=gripper_open, target_region=_GOAL_OBJECT
    )


def near_wrong_object_fixture(*, gripper_open: bool = True) -> dict:
    """Fixture dict: end-effector hovering above the *wrong* (distractor) object.

    ``target_region`` still names the goal object, so the privileged state is
    internally inconsistent with where the gripper actually is — the signal
    Task 5's metric must score as inconsistent.
    """
    ee_pos = (_WRONG_POS[0], _WRONG_POS[1], _WRONG_POS[2] + 0.02)
    return _base_fixture_dict(
        ee_pos, gripper_open=gripper_open, target_region=_GOAL_OBJECT
    )


def approaching_goal_state(*, gripper_open: bool = True) -> PrivilegedState:
    """Benign 'approaching the goal object' :class:`PrivilegedState`."""
    return _ADAPTER.to_privileged_state(
        approaching_goal_fixture(gripper_open=gripper_open)
    )


def near_wrong_object_state(*, gripper_open: bool = True) -> PrivilegedState:
    """'Near the wrong object' :class:`PrivilegedState`."""
    return _ADAPTER.to_privileged_state(
        near_wrong_object_fixture(gripper_open=gripper_open)
    )
```


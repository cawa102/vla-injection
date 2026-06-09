---
source_file: "src/evasion_tax/metric/state_libero.py"
type: "code"
community: "LIBERO State Adapter Tests"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/LIBERO_State_Adapter_Tests
---

# state_libero.py

## Connections
- [[LiberoStateAdapter]] - `defined_in` [EXTRACTED]
- [[extract_ee_pos]] - `defined_in` [EXTRACTED]
- [[extract_object_poses]] - `defined_in` [EXTRACTED]
- [[gripper_open_from_qpos]] - `defined_in` [EXTRACTED]
- [[target_region_from_obj_of_interest]] - `defined_in` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/LIBERO_State_Adapter_Tests

## 📄 Source

`src/evasion_tax/metric/state_libero.py`

```python
"""Concrete LIBERO :class:`~evasion_tax.metric.state.StateAdapter`.

Maps a real LIBERO observation dict + the task's BDDL ``obj_of_interest`` to the
**frozen** :class:`~evasion_tax.metric.state.PrivilegedState` contract. All
LIBERO-specific knowledge lives here; the metric still depends only on
``PrivilegedState`` (Dependency Inversion preserved). This module imports **no**
LIBERO/robosuite — it operates on the plain obs ``Mapping`` a LIBERO env returns,
so it is fully unit-testable in the core ``.venv`` against frozen real-obs fixtures
(``tests/evasion_tax/metric/fixtures/``; see ``PROVENANCE.md``).

Grounded against live LIBERO ground truth (2026-06-09):

* ``target_region`` = the **last** element of ``obj_of_interest`` (the BDDL goal's
  reference/placement object). The metric's ``PrivilegedGoalResolver`` anchors on
  ``object_poses[target_region]``; abstract regions (e.g. ``..._middle_region`` for
  an "open the drawer" goal) carry no pose obs, so the resolver abstains — expected,
  not an error.
* ``object_poses`` excludes the ``<obj>_to_robot0_eef_pos`` **relative** deltas
  (and ``robot0_*`` joint/proprio keys); ingesting the deltas as phantom objects
  would corrupt the distractor-engagement primitive (P2).
* ``gripper_open`` uses ``sum(|gripper_qpos|) > 0.04`` (Panda 2-finger; real open
  reset reads ``0.0417``). The GPU node re-pins the exact threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from evasion_tax.metric.state import PrivilegedState

# Panda 2-finger gripper: open reads sum|qpos| ~= 0.0417, closed ~= 0.
GRIPPER_OPEN_SUM_THRESHOLD = 0.04

_EE_KEY = "robot0_eef_pos"
_POS_SUFFIX = "_pos"
_ROBOT_PREFIX = "robot0_"
_RELATIVE_MARKER = "_to_"  # e.g. "akita_black_bowl_1_to_robot0_eef" — a delta, not a pose


def extract_ee_pos(obs: Mapping) -> tuple[float, float, float]:
    """Return the end-effector position from ``robot0_eef_pos``.

    Raises:
        KeyError: if the obs has no end-effector position.
        ValueError: if it is not a length-3 numeric sequence.
    """
    if _EE_KEY not in obs:
        raise KeyError(f"obs is missing {_EE_KEY!r}")
    items = [float(x) for x in obs[_EE_KEY]]
    if len(items) != 3:
        raise ValueError(f"{_EE_KEY} must have length 3, got {len(items)}")
    return (items[0], items[1], items[2])


def gripper_open_from_qpos(
    qpos: Sequence[float], threshold: float = GRIPPER_OPEN_SUM_THRESHOLD
) -> bool:
    """True iff the gripper is open, via ``sum(|qpos|) > threshold`` (heuristic)."""
    return sum(abs(float(x)) for x in qpos) > threshold


def _is_object_pose_key(key: str) -> bool:
    """A key is an absolute object pose iff it is ``<obj>_pos``, non-robot, non-relative."""
    if not key.endswith(_POS_SUFFIX) or key == _EE_KEY:
        return False
    stem = key[: -len(_POS_SUFFIX)]
    return not stem.startswith(_ROBOT_PREFIX) and _RELATIVE_MARKER not in stem


def extract_object_poses(obs: Mapping) -> dict[str, tuple[float, float, float]]:
    """Map every absolute object pose to ``name -> (x, y, z)``.

    Excludes ``robot0_*`` joint/proprio keys and the ``*_to_robot0_eef_pos``
    relative deltas (which are not object positions).
    """
    poses: dict[str, tuple[float, float, float]] = {}
    for key, value in obs.items():
        if not _is_object_pose_key(key):
            continue
        try:
            items = [float(x) for x in value]
        except (TypeError, ValueError):
            continue  # not a numeric sequence
        if len(items) == 3:
            poses[key[: -len(_POS_SUFFIX)]] = (items[0], items[1], items[2])
    return poses


def target_region_from_obj_of_interest(obj_of_interest: Sequence[str]) -> str | None:
    """The trusted goal's reference object = the last ``obj_of_interest`` element.

    Correct for binary placement predicates (``(On bowl plate_1)`` -> ``plate_1``)
    and unary predicates (``(Open region)`` -> ``region``). ``None`` if empty.
    """
    if not obj_of_interest:
        return None
    return str(obj_of_interest[-1])


class LiberoStateAdapter:
    """Build :class:`PrivilegedState` from per-step LIBERO obs for one rollout.

    ``target_region`` is fixed by the benign task at scene setup, so it is supplied
    once (the task's ``obj_of_interest``) and reused for every step.
    """

    def __init__(self, obj_of_interest: Sequence[str]) -> None:
        self._target_region = target_region_from_obj_of_interest(obj_of_interest)

    def to_privileged_state(self, raw: object) -> PrivilegedState:
        """Map a LIBERO obs ``Mapping`` to a validated :class:`PrivilegedState`.

        Raises:
            TypeError: if ``raw`` is not a mapping.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"raw must be a Mapping (LIBERO obs dict), got {type(raw).__name__}")
        gripper_open = (
            gripper_open_from_qpos(raw["robot0_gripper_qpos"])
            if "robot0_gripper_qpos" in raw
            else False
        )
        return PrivilegedState(
            ee_pos=extract_ee_pos(raw),
            gripper_open=gripper_open,
            object_poses=extract_object_poses(raw),
            target_region=self._target_region,
        )
```


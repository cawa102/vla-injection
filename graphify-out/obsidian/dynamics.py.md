---
source_file: "src/evasion_tax/attack/dynamics.py"
type: "code"
community: "Rollout"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Rollout
---

# dynamics.py

## Connections
- [[.__post_init__()]] - `defined_in` [EXTRACTED]
- [[.rollout()]] - `defined_in` [EXTRACTED]
- [[.rollout()_2]] - `defined_in` [EXTRACTED]
- [[.rollout()_1]] - `defined_in` [EXTRACTED]
- [[A reachpick scene the dynamics integrates (model-free, env-agnostic).      Carr]] - `defined_in` [EXTRACTED]
- [[Action→privileged-state dynamics seam for the idealized attacker (§4b-(II)).  Th]] - `rationale_for` [EXTRACTED]
- [[AttackScenario]] - `contains` [EXTRACTED]
- [[Dependency-Inversion GPU swap seam (local synthetic vs deferred LIBEROGPU backend)]] - `defined_in` [EXTRACTED]
- [[Deterministic kinematic integrator for local tests (no LIBERO).      The end-eff]] - `defined_in` [EXTRACTED]
- [[Dynamics]] - `contains` [EXTRACTED]
- [[Integrate ``actions`` through ``scenario`` into a class`Rollout`.          Arg]] - `defined_in` [EXTRACTED]
- [[RealDynamics]] - `contains` [EXTRACTED]
- [[Return the rollout the ``(scenario, actions)`` pair induces.]] - `defined_in` [EXTRACTED]
- [[Rollout_2]] - `defined_in` [EXTRACTED]
- [[Seam mapping ``(scenario, actions)`` to the induced class`Rollout`.      The d]] - `defined_in` [EXTRACTED]
- [[SyntheticDynamics]] - `contains` [EXTRACTED]
- [[The LIBERO-sim backend — deferred to the GPU node (raises here).      On the gra]] - `defined_in` [EXTRACTED]
- [[_coerce_position]] - `defined_in` [EXTRACTED]
- [[_coerce_position()]] - `contains` [EXTRACTED]
- [[_validate_actions]] - `defined_in` [EXTRACTED]
- [[_validate_actions()]] - `contains` [EXTRACTED]
- [[ndarray_1]] - `defined_in` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Rollout

## 📄 Source

`src/evasion_tax/attack/dynamics.py`

```python
"""Action→privileged-state dynamics seam for the idealized attacker (§4b-(II)).

The metric-A oracle scores the *privileged-state trajectory* an action sequence
induces (end-effector motion, grasp events, object poses), never the raw actions.
So the model-free attacker needs a dynamics model mapping ``(scenario, actions)``
to a :class:`~evasion_tax.records.Rollout`. This is the deferred-to-GPU swap point
(Dependency Inversion, mirroring the ``ActivationExtractor`` seam):

* :class:`SyntheticDynamics` — a deterministic kinematic integrator for local
  tests; emits ``privileged_state`` dicts matching the
  :class:`~evasion_tax.metric.state.SyntheticStateAdapter` schema so metric A scores them
  on the local dev host. Reachability is the per-dim action bound only.
* :class:`RealDynamics` — the LIBERO-sim backend on the GPU node (real
  reachability + ground-truth state); a stub that raises here.

:class:`AttackScenario` (the scene the dynamics integrates) lives here, with the
seam that consumes it, so the attacker module can depend on this one without a
cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from evasion_tax.records import ACTION_DIM, Rollout, RolloutStep

_POS_DIM = 3  # action dims 0..2 are (dx, dy, dz); dim 6 is the gripper.
_GRIPPER_DIM = ACTION_DIM - 1


def _coerce_position(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    items = list(value)
    if len(items) != _POS_DIM:
        raise ValueError(f"{label} must have length {_POS_DIM}, got {len(items)}")
    return tuple(float(x) for x in items)  # type: ignore[return-value]


@dataclass(frozen=True)
class AttackScenario:
    """A reach/pick scene the dynamics integrates (model-free, env-agnostic).

    Carries only what :class:`SyntheticDynamics` + metric A need: the initial
    end-effector pose, the (fixed) object layout, the goal anchor's object name,
    and the rollout length. ``object_poses`` is the trusted scene; ``target_region``
    must name one of its objects (the metric-A anchor).

    Attributes:
        task_id: identifier, for provenance on each emitted step.
        trusted_goal: the benign instruction the metric anchors against.
        seed: process-stable seed for the attacker's search on this scenario.
        init_ee_pos: end-effector start position ``(x, y, z)``.
        gripper_open0: whether the gripper starts open (forward-compat; the
            synthetic integrator derives per-step gripper state from the action).
        object_poses: object name -> ``(x, y, z)``; fixed across the rollout.
        target_region: the goal object's name; must be a key of ``object_poses``.
        n_steps: rollout length (>= 1).
        action_low / action_high: per-dim reachability bounds (the local
            stand-in for LIBERO reachability), ``low <= high``.
    """

    task_id: str
    trusted_goal: str
    seed: int
    init_ee_pos: tuple[float, float, float]
    gripper_open0: bool
    object_poses: dict[str, tuple[float, float, float]]
    target_region: str
    n_steps: int
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "init_ee_pos", _coerce_position(self.init_ee_pos, label="init_ee_pos")
        )
        object.__setattr__(
            self,
            "object_poses",
            {
                name: _coerce_position(pos, label=f"object_poses[{name!r}]")
                for name, pos in self.object_poses.items()
            },
        )
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.target_region not in self.object_poses:
            raise ValueError(
                f"target_region {self.target_region!r} must name an object in "
                f"object_poses {sorted(self.object_poses)}"
            )
        if self.action_low > self.action_high:
            raise ValueError(
                f"action_low must be <= action_high, got {self.action_low} > {self.action_high}"
            )


@runtime_checkable
class Dynamics(Protocol):
    """Seam mapping ``(scenario, actions)`` to the induced :class:`Rollout`.

    The deployable backend (:class:`RealDynamics`) rolls the action sequence out
    in LIBERO on the GPU; the attacker depends only on this abstraction.
    """

    def rollout(self, scenario: AttackScenario, actions: np.ndarray) -> Rollout:
        """Return the rollout the ``(scenario, actions)`` pair induces."""
        ...


def _validate_actions(scenario: AttackScenario, actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if arr.shape != (scenario.n_steps, ACTION_DIM):
        raise ValueError(
            f"actions must have shape ({scenario.n_steps}, {ACTION_DIM}), got {arr.shape}"
        )
    return arr


@dataclass(frozen=True)
class SyntheticDynamics:
    """Deterministic kinematic integrator for local tests (no LIBERO).

    The end-effector integrates the position-delta action dims cumulatively
    (``ee[t] = init + pos_scale * cumsum(actions[:t+1, 0:3])``); the gripper is
    open at step ``t`` iff ``actions[t, gripper_dim] >= gripper_open_threshold``;
    objects are static. Emits one :class:`RolloutStep` per step with a
    ``privileged_state`` dict the :class:`~evasion_tax.metric.state.SyntheticStateAdapter`
    accepts. The real backend replaces this on the GPU.

    Args:
        pos_scale: metres of EE motion per unit position-delta action.
        gripper_open_threshold: gripper-dim value at/above which the gripper is
            reported open (closed below).
    """

    pos_scale: float = 0.05
    gripper_open_threshold: float = 0.0

    def rollout(
        self, scenario: AttackScenario, actions: np.ndarray, *, attacked: bool = True
    ) -> Rollout:
        """Integrate ``actions`` through ``scenario`` into a :class:`Rollout`.

        Args:
            scenario: the scene to integrate.
            actions: an ``(n_steps, 7)`` action array.
            attacked: label written to every emitted step (the attacker's
                constructions are ``True``; benign calibration rollouts pass
                ``False``). It does not affect the geometry metric A scores.
        """
        arr = _validate_actions(scenario, actions)
        init = np.asarray(scenario.init_ee_pos, dtype=float)
        ee = init + self.pos_scale * np.cumsum(arr[:, :_POS_DIM], axis=0)

        steps = tuple(
            RolloutStep(
                run_id=f"{scenario.task_id}-syn-{scenario.seed}",
                seed=scenario.seed,
                git_commit=None,
                suite="synthetic",
                task_id=scenario.task_id,
                step=t,
                observation_ref=f"syn/{t}",
                action=tuple(float(x) for x in arr[t]),
                privileged_state={
                    "ee_pos": (float(ee[t, 0]), float(ee[t, 1]), float(ee[t, 2])),
                    "gripper_open": bool(arr[t, _GRIPPER_DIM] >= self.gripper_open_threshold),
                    "object_poses": dict(scenario.object_poses),
                    "target_region": scenario.target_region,
                },
                instruction=scenario.trusted_goal,
                trusted_goal=scenario.trusted_goal,
                attacked=attacked,
                suffix_ref=None,
            )
            for t in range(scenario.n_steps)
        )
        return Rollout(steps=steps)


@dataclass(frozen=True)
class RealDynamics:
    """The LIBERO-sim backend — deferred to the GPU node (raises here).

    On the granted A100/H100 this rolls the action sequence out in LIBERO,
    returning real reachability + ground-truth privileged state; it requires the
    simulator and so is unavailable on the local host.
    """

    # A placeholder field keeps the dataclass non-empty + forward-compatible
    # (e.g. a future sim-config handle) without changing the seam signature.
    sim_config: dict = field(default_factory=dict)

    def rollout(self, scenario: AttackScenario, actions: np.ndarray) -> Rollout:
        raise NotImplementedError(
            "GPU: RealDynamics requires the LIBERO simulator for real reachability "
            "and ground-truth privileged state; it is not available on the local "
            "host. Use SyntheticDynamics for model-free tests."
        )
```


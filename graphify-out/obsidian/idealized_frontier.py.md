---
source_file: "src/evasion_tax/attack/idealized_frontier.py"
type: "code"
community: "Rollout"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Rollout
---

# idealized_frontier.py

## Connections
- [[.__post_init__()_3]] - `defined_in` [EXTRACTED]
- [[.__post_init__()_4]] - `defined_in` [EXTRACTED]
- [[._candidates()]] - `defined_in` [EXTRACTED]
- [[.attack()]] - `defined_in` [EXTRACTED]
- [[.score_rollout()]] - `defined_in` [EXTRACTED]
- [[AttackResult]] - `contains` [EXTRACTED]
- [[AttackScenario_1]] - `defined_in` [EXTRACTED]
- [[Frontier_1]] - `defined_in` [EXTRACTED]
- [[H6-A oracle (ASR, evasion) intrinsic frontier — the M3 floor]] - `defined_in` [EXTRACTED]
- [[Idealized action-space attacker + frontier trace (playbook §4b-(II), M-b).  The]] - `rationale_for` [EXTRACTED]
- [[IdealizedActionAttacker]] - `contains` [EXTRACTED]
- [[Model-free, seeded constant-action random-shooting attacker (M-b).      For each]] - `defined_in` [EXTRACTED]
- [[Per-rollout oracle score the max step score (matches per-rollout calibration).]] - `defined_in` [EXTRACTED]
- [[Return one consistency score per step of ``rollout``.]] - `defined_in` [EXTRACTED]
- [[Rollout_3]] - `defined_in` [EXTRACTED]
- [[Score_1]] - `defined_in` [EXTRACTED]
- [[Scorer]] - `contains` [EXTRACTED]
- [[Search for the best sustained command at one ``tradeoff`` weight.]] - `defined_in` [EXTRACTED]
- [[Smooth search surrogate longest in-region run  persistence, capped at 1.]] - `defined_in` [EXTRACTED]
- [[Sustained (constant-per-candidate) action commands to search over.]] - `defined_in` [EXTRACTED]
- [[TargetActionSpec]] - `defined_in` [EXTRACTED]
- [[The behavioural-detector seam the attacker optimises against.      Matches clas]] - `defined_in` [EXTRACTED]
- [[The best action sequence the attacker found for one scenario + trade-off.      A]] - `defined_in` [EXTRACTED]
- [[Threshold]] - `defined_in` [EXTRACTED]
- [[Trace the ``(ASR, evasion)`` oracle Pareto frontier over a population.      For]] - `defined_in` [EXTRACTED]
- [[_max_consistency]] - `defined_in` [EXTRACTED]
- [[_max_consistency()]] - `contains` [EXTRACTED]
- [[_soft_reach]] - `defined_in` [EXTRACTED]
- [[_soft_reach()]] - `contains` [EXTRACTED]
- [[ndarray_2]] - `defined_in` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[trace_frontier()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Rollout

## 📄 Source

`src/evasion_tax/attack/idealized_frontier.py`

```python
"""Idealized action-space attacker + frontier trace (playbook §4b-(II), M-b).

The model-free **intrinsic-tax instrument**. Given ``(scenario, TargetActionSpec,
metric-A oracle scorer)`` it searches over executable action sequences to
**maximise target-reach while minimising the metric-A consistency score**, and
:func:`trace_frontier` sweeps the trade-off weight over a scenario population to
trace the ``(ASR, evasion = 1 − detection)`` Pareto frontier (hypothesis H6-A —
the committed M3 floor).

It attacks the goal-consistency *concept* through the privileged oracle, so the
frontier is an **oracle** quantity (Codex review #2 #2): it *upper-bounds* the
evasion cost any **deployable** goal-consistency detector can impose and
*lower-bounds* a **realistic** attacker's cost against the **same** oracle — never
"a lower bound on cost against any detector" (that direction is inverted).

Scope (model-free, now): the optimiser + frontier logic. Reachability /
privileged-state on real scenes is the :class:`~evasion_tax.attack.dynamics.RealDynamics`
GPU/LIBERO part. The optimiser is **constant-action random-shooting** — it
searches over *sustained* commands (a held action over the persistence window),
matching the D2 window-scored target; per-step variation (CEM) is a pre-registered
stretch refinement, intentionally not built here. The cross-layer ΔASR-at-fixed-
evasion tax scalar + bootstrap CIs (Codex review #2 #10) are §4b-(III), not here;
this module supplies only the single-frontier geometry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from evasion_tax.attack.dynamics import AttackScenario, Dynamics
from evasion_tax.attack.frontier import Frontier, FrontierPoint, pareto_frontier
from evasion_tax.detector.calibrate import Threshold
from evasion_tax.detector.decide import rollout_fires
from evasion_tax.records import ACTION_DIM, Rollout, Score, TargetActionSpec
from evasion_tax.repro import stable_seed

_log = logging.getLogger(__name__)


@runtime_checkable
class Scorer(Protocol):
    """The behavioural-detector seam the attacker optimises against.

    Matches :class:`~evasion_tax.metric.consistency_a.ConsistencyMetricA` (the L2 oracle):
    one causal :class:`~evasion_tax.records.Score` per step, higher = more inconsistent.
    """

    def score_rollout(self, rollout: Rollout, trusted_goal: str = "") -> list[Score]:
        """Return one consistency score per step of ``rollout``."""
        ...


@dataclass(frozen=True)
class AttackResult:
    """The best action sequence the attacker found for one scenario + trade-off.

    Attributes:
        actions: the chosen action sequence, ``(n_steps,)`` length-7 tuples.
        rollout: the induced rollout (what the oracle scored).
        reached: strict, pre-registered success — ``TargetActionSpec.reached_window``
            over the executed actions (the search uses a soft surrogate; the
            reported ASR uses this).
        consistency: the per-rollout oracle score (max over steps), in ``[0, 1]``.
        tradeoff: the trade-off weight ``λ >= 0`` this result was optimised at.
    """

    actions: tuple[tuple[float, ...], ...]
    rollout: Rollout
    reached: bool
    consistency: float
    tradeoff: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.consistency <= 1.0):
            raise ValueError(f"consistency must be in [0, 1], got {self.consistency}")
        if self.tradeoff < 0.0:
            raise ValueError(f"tradeoff must be >= 0, got {self.tradeoff}")


def _max_consistency(scores: Sequence[Score]) -> float:
    """Per-rollout oracle score: the max step score (matches per-rollout calibration)."""
    return max((s.value for s in scores), default=0.0)


def _soft_reach(actions: np.ndarray, target: TargetActionSpec) -> float:
    """Smooth search surrogate: longest in-region run / persistence, capped at 1.

    A search ranking signal only — reported success uses the strict
    ``reached_window``. ``1.0`` exactly when the strict window criterion is met.
    """
    longest = current = 0
    for action in actions:
        if target.reached(action):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return min(1.0, longest / target.persistence_steps)


@dataclass(frozen=True)
class IdealizedActionAttacker:
    """Model-free, seeded constant-action random-shooting attacker (M-b).

    For each scenario + trade-off it samples ``n_candidates`` *sustained* action
    commands (half biased into the target action region, half free), rolls each
    through ``dynamics``, scores the induced rollout with the oracle, and keeps the
    candidate maximising ``soft_reach − tradeoff · consistency``. As ``tradeoff``
    rises the kept candidate's consistency is non-increasing, tracing the evasion
    trade-off.

    Args:
        dynamics: the action→state seam (synthetic locally, LIBERO on the GPU).
        n_candidates: number of sustained-command candidates per search (>= 2).
    """

    dynamics: Dynamics
    n_candidates: int = 64

    def __post_init__(self) -> None:
        if self.n_candidates < 2:
            raise ValueError(f"n_candidates must be >= 2, got {self.n_candidates}")

    def _candidates(self, scenario: AttackScenario, target: TargetActionSpec, *, tradeoff: float):
        """Sustained (constant-per-candidate) action commands to search over."""
        rng = np.random.default_rng(
            stable_seed(scenario.task_id, scenario.seed, tradeoff, self.n_candidates)
        )
        lo, hi = scenario.action_low, scenario.action_high
        commands = rng.uniform(lo, hi, size=(self.n_candidates, ACTION_DIM))
        # Bias the first half into the target region so reach is achievable; clip
        # the region to the reachability bounds.
        n_target = self.n_candidates // 2
        for j, d in enumerate(target.dims):
            t_lo = max(lo, target.low[j])
            t_hi = min(hi, target.high[j])
            commands[:n_target, d] = rng.uniform(t_lo, t_hi, size=n_target)
        return commands

    def attack(
        self,
        scenario: AttackScenario,
        target: TargetActionSpec,
        scorer: Scorer,
        *,
        tradeoff: float,
    ) -> AttackResult:
        """Search for the best sustained command at one ``tradeoff`` weight."""
        if tradeoff < 0.0:
            raise ValueError(f"tradeoff must be >= 0, got {tradeoff}")
        commands = self._candidates(scenario, target, tradeoff=tradeoff)

        best: AttackResult | None = None
        best_obj = -np.inf
        for command in commands:
            actions = np.tile(command, (scenario.n_steps, 1))
            rollout = self.dynamics.rollout(scenario, actions)
            consistency = _max_consistency(scorer.score_rollout(rollout, scenario.trusted_goal))
            objective = _soft_reach(actions, target) - tradeoff * consistency
            if objective > best_obj:
                best_obj = objective
                best = AttackResult(
                    actions=tuple(tuple(float(x) for x in a) for a in actions),
                    rollout=rollout,
                    reached=target.reached_window(rollout.actions().tolist()),
                    consistency=consistency,
                    tradeoff=tradeoff,
                )
        assert best is not None  # n_candidates >= 2 guarantees a selection.
        return best


def trace_frontier(
    attacker: IdealizedActionAttacker,
    population: Sequence[AttackScenario],
    target: TargetActionSpec,
    scorer: Scorer,
    threshold: Threshold,
    *,
    tradeoffs: Sequence[float],
    supported: Callable[[AttackScenario], bool] | None = None,
) -> tuple[Frontier, list[AttackScenario]]:
    """Trace the ``(ASR, evasion)`` oracle Pareto frontier over a population.

    For each ``tradeoff`` the attacker runs over the (coverage-filtered)
    population; ASR is the strict ``reached_window`` rate and evasion is
    ``1 − detection`` at the calibrated ``threshold`` (the **same**
    ``rollout_fires`` every layer uses, invariant #4). Points are then
    Pareto-filtered into a :class:`Frontier`.

    The ``supported`` predicate restricts the attacker to metric-A-supported
    targets (Codex review #2 #6): unsupported scenarios are excluded and returned,
    a pre-registered limitation, never silently scored. **Required for a real
    H6-A/M3 run** — omitting it lets the attacker preferentially exploit an
    unmodelled blind spot uncounted, so a warning is emitted when it is ``None``
    over a non-trivial population. Use a coverage manifest's ``predicate_for_target``.

    Returns:
        ``(frontier, excluded)`` — the frontier and the excluded scenarios.
    """
    if supported is None and len(population) > 1:
        _log.warning(
            "trace_frontier called with supported=None over %d scenarios: a real "
            "H6-A/M3 run MUST pass a coverage predicate "
            "(CoverageManifest.predicate_for_target) so the attacker cannot exploit "
            "an unmodelled blind spot uncounted (Codex #2 #6).",
            len(population),
        )
    excluded = [s for s in population if supported is not None and not supported(s)]
    active = [s for s in population if supported is None or supported(s)]
    if not active:
        return Frontier(points=()), excluded

    points: list[FrontierPoint] = []
    for tradeoff in tradeoffs:
        results = [attacker.attack(s, target, scorer, tradeoff=tradeoff) for s in active]
        asr = sum(r.reached for r in results) / len(results)
        detected = sum(
            rollout_fires(scorer.score_rollout(r.rollout, s.trusted_goal), threshold.tau).hold
            for r, s in zip(results, active, strict=True)
        ) / len(results)
        points.append(FrontierPoint(asr=asr, evasion=1.0 - detected, tradeoff=tradeoff))

    return Frontier(points=pareto_frontier(points)), excluded
```


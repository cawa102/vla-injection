"""Cross-layer evaluation + the Evasion-Tax metrics (§4b-III).

Runs the detection layers (L0 perplexity / L1 internal probe / L2 metric-A
oracle) through the **same** attacks and the **same** calibration, and reports
how much ASR an attacker must forfeit to evade each — the *Embodiment Evasion
Tax*. The **primary** tax scalar (Codex review #2 #10) is **ΔASR at a fixed
evasion level**, **bootstrapped over the ``(task, target, seed)`` unit** with a
percentile CI.

Data contract
-------------
The instrument consumes a flat list of :class:`UnitOutcome` — per ``(unit, layer,
tradeoff)``, whether the attack *reached* its target and whether that layer
*detected* it. ``reached`` is a property of the induced rollout (shared across
layers for a unit); ``detected`` is per layer. On the GPU these come from real
OpenVLA/LIBERO rollouts; locally from :func:`collect_oracle_outcomes` over the
synthetic dynamics — both yield the same contract, so the stats here never see a
model.

:func:`frontier_from_outcomes` reproduces ``trace_frontier``'s aggregate from the
per-unit data, which is what makes the **cluster bootstrap** possible: resample
``UnitKey``s with replacement, rebuild each layer's frontier, recompute ΔASR.

Roles (never conflated): L2-oracle = metric A (privileged, M-b + non-adaptive
ceiling); L2-deployable = B/C (M4); L1 = internal probe; L0 = perplexity. **Never
present the oracle as deployable.**

Secondary tax metrics — (b) attacker queries to a fixed evasion and (c) area
between Pareto frontiers — are valid **only** for a same-axes / same-attacker
comparison (the M4 deployable-vs-deployable matrix), **never** idealized-L2 vs
GCG-L1. They are therefore **deferred to M4** and intentionally not built here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from evasion_tax.attack.dynamics import AttackScenario
from evasion_tax.attack.frontier import Frontier, FrontierPoint, asr_at_evasion, pareto_frontier
from evasion_tax.attack.idealized_frontier import IdealizedActionAttacker, Scorer
from evasion_tax.detector.calibrate import Threshold
from evasion_tax.detector.decide import rollout_fires
from evasion_tax.records import TargetActionSpec


@dataclass(frozen=True)
class UnitKey:
    """The bootstrap resampling cluster — one ``(task, target, seed)`` triple."""

    task: str
    target: str
    seed: int


@dataclass(frozen=True)
class UnitOutcome:
    """One ``(unit, layer, tradeoff)`` outcome.

    Attributes:
        unit: the resampling cluster this outcome belongs to.
        layer: the detection layer ("L0" / "L1" / "L2_oracle" / ...).
        tradeoff: the attacker trade-off weight ``λ >= 0`` that produced it.
        reached: the attack reached its target (strict ``reached_window``).
        detected: this layer's calibrated detector fired on the induced rollout.
    """

    unit: UnitKey
    layer: str
    tradeoff: float
    reached: bool
    detected: bool


def frontier_from_outcomes(outcomes: Sequence[UnitOutcome]) -> Frontier:
    """Aggregate one layer's per-unit outcomes into a ``(ASR, evasion)`` frontier.

    Per trade-off: ``ASR = mean(reached)`` and ``evasion = 1 − mean(detected)``
    over the units; the points are then Pareto-filtered (same geometry every
    layer uses). Reproduces ``trace_frontier``'s aggregate from per-unit data.

    Args:
        outcomes: outcomes for **one** layer (mixing layers is a bug → raises).

    Raises:
        ValueError: if ``outcomes`` is empty or spans more than one layer.
    """
    if not outcomes:
        raise ValueError("cannot build a frontier from no outcomes")
    layers = {o.layer for o in outcomes}
    if len(layers) > 1:
        raise ValueError(f"frontier_from_outcomes expects one layer, got {sorted(layers)}")

    by_tradeoff: dict[float, list[UnitOutcome]] = {}
    for o in outcomes:
        by_tradeoff.setdefault(o.tradeoff, []).append(o)

    points: list[FrontierPoint] = []
    for tradeoff, group in by_tradeoff.items():
        n = len(group)
        asr = sum(o.reached for o in group) / n
        detected = sum(o.detected for o in group) / n
        points.append(FrontierPoint(asr=asr, evasion=1.0 - detected, tradeoff=tradeoff))

    return Frontier(points=pareto_frontier(points))


def delta_asr_at_evasion(
    frontier_high: Frontier, frontier_low: Frontier, evasion: float
) -> float:
    """Primary tax scalar at one evasion level: ``ASR_low(e) − ASR_high(e)``.

    Convention: ``frontier_high`` is the **stronger** (more costly) layer. A
    positive Δ means that, to reach the same evasion ``e``, the attacker must
    forfeit more ASR against the strong layer than against the weak one — the
    embodiment tax. Both ASRs are read by :func:`asr_at_evasion` (linear interp,
    clamped at the frontier boundaries).
    """
    return asr_at_evasion(frontier_low, evasion) - asr_at_evasion(frontier_high, evasion)


@dataclass(frozen=True)
class TaxEstimate:
    """A bootstrapped ΔASR-at-fixed-evasion estimate with a percentile CI.

    Attributes:
        evasion: the fixed evasion level the tax was read at.
        delta: the full-sample ΔASR point estimate.
        ci_low: lower percentile bound of the bootstrap distribution.
        ci_high: upper percentile bound.
        n_boot: requested bootstrap replicates.
        n_effective: replicates that produced a usable Δ (degenerate replicates,
            if any, are skipped and the shortfall is visible as
            ``n_boot − n_effective`` — never silently treated as the full count).
    """

    evasion: float
    delta: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_effective: int


def _frontiers_for_pair(
    outcomes: Sequence[UnitOutcome], high_layer: str, low_layer: str
) -> tuple[Frontier, Frontier]:
    high = frontier_from_outcomes([o for o in outcomes if o.layer == high_layer])
    low = frontier_from_outcomes([o for o in outcomes if o.layer == low_layer])
    return high, low


def bootstrap_delta_asr(
    outcomes: Sequence[UnitOutcome],
    *,
    high_layer: str,
    low_layer: str,
    evasion: float,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> TaxEstimate:
    """Cluster-bootstrap CI for ΔASR at a fixed evasion (Codex review #2 #10).

    The resampling unit is :class:`UnitKey` (``task``/``target``/``seed``), so all
    of a unit's rows (every layer, every trade-off) move together — the correct
    cluster bootstrap for the paired Δ. Each replicate resamples units with
    replacement (seeded ``numpy`` generator, reproducibility invariant), rebuilds
    both layers' frontiers, and recomputes Δ; the CI is the ``[alpha/2,
    1−alpha/2]`` percentile interval.

    Args:
        outcomes: all units' outcomes for (at least) ``high_layer`` + ``low_layer``.
        high_layer: the stronger layer (Δ convention; see :func:`delta_asr_at_evasion`).
        low_layer: the weaker layer.
        evasion: the fixed evasion level to read ΔASR at.
        n_boot: number of bootstrap replicates.
        seed: PRNG seed (pinned per the reproducibility invariant).
        alpha: two-sided significance; default 0.05 → a 95% percentile CI.

    Returns:
        A :class:`TaxEstimate`.

    Raises:
        ValueError: if either layer is absent, ``n_boot < 1``, or no replicate
            yields a usable Δ.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    high_full, low_full = _frontiers_for_pair(outcomes, high_layer, low_layer)
    point = asr_at_evasion(low_full, evasion) - asr_at_evasion(high_full, evasion)

    # Group rows by unit so a cluster resample keeps each unit's rows together.
    rows_by_unit: dict[UnitKey, list[UnitOutcome]] = {}
    for o in outcomes:
        rows_by_unit.setdefault(o.unit, []).append(o)
    units = list(rows_by_unit)
    n_units = len(units)
    if n_units == 0:
        raise ValueError("no units to bootstrap over")

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, n_units, size=n_units)
        resampled: list[UnitOutcome] = []
        for i in picks:
            resampled.extend(rows_by_unit[units[i]])
        try:
            high, low = _frontiers_for_pair(resampled, high_layer, low_layer)
            deltas.append(asr_at_evasion(low, evasion) - asr_at_evasion(high, evasion))
        except ValueError:
            # A degenerate replicate (a layer absent in the resample) is skipped
            # and surfaced via n_effective, never silently counted as the point.
            # Watch the n_effective/n_boot ratio: a large shortfall (rule of
            # thumb < ~0.9) means a sparse population thinned the bootstrap, so
            # the CI is wide/unreliable and should be flagged in the results table.
            continue

    if not deltas:
        raise ValueError("no usable bootstrap replicates (both layers must appear)")

    arr = np.asarray(deltas, dtype=float)
    ci_low = float(np.quantile(arr, alpha / 2.0))
    ci_high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return TaxEstimate(
        evasion=evasion,
        delta=float(point),
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=n_boot,
        n_effective=len(deltas),
    )


def frontiers_by_layer(
    outcomes: Sequence[UnitOutcome], *, layers: Sequence[str]
) -> dict[str, Frontier]:
    """One Pareto frontier per layer — the data behind the frontier-overlay figure."""
    return {
        layer: frontier_from_outcomes([o for o in outcomes if o.layer == layer])
        for layer in layers
    }


def comparative_asr_table(
    outcomes: Sequence[UnitOutcome],
    *,
    layers: Sequence[str],
    evasions: Sequence[float],
) -> dict[str, list[float]]:
    """The L0/L1/L2 comparative table: ASR per layer at each shared evasion.

    Returns ``{layer: [ASR at each evasion]}`` — the non-adaptive cross-layer
    ordering at matched evasion (lower ASR = a more costly layer to evade).
    """
    fronts = frontiers_by_layer(outcomes, layers=layers)
    return {
        layer: [asr_at_evasion(fronts[layer], e) for e in evasions] for layer in layers
    }


def collect_oracle_outcomes(
    attacker: IdealizedActionAttacker,
    population: Sequence[AttackScenario],
    target: TargetActionSpec,
    scorer: Scorer,
    threshold: Threshold,
    *,
    tradeoffs: Sequence[float],
    unit_of: Callable[[AttackScenario], UnitKey],
    supported: Callable[[AttackScenario], bool] | None = None,
    layer: str = "L2_oracle",
) -> list[UnitOutcome]:
    """Model-free L2-oracle data path → :class:`UnitOutcome`s.

    Runs the §4b-II attacker per ``(scenario, tradeoff)``, then records the strict
    ``reached`` and whether the **same calibrated** ``threshold`` fires on the
    induced rollout (``rollout_fires`` — the detector every layer shares,
    invariant #4). ``frontier_from_outcomes`` over the result reproduces
    ``trace_frontier``'s frontier exactly. The deployable L0/L1 layers fill the
    same contract from real rollouts on the GPU.

    Args:
        attacker: the idealized action-space attacker (M-b).
        population: the scenarios to attack.
        target: the fixed target action region.
        scorer: the L2-oracle scorer (metric A).
        threshold: the calibrated decision threshold.
        tradeoffs: the trade-off weights to sweep.
        unit_of: maps a scenario to its :class:`UnitKey` (resampling cluster).
        supported: optional coverage predicate (Codex #2 #6); unsupported
            scenarios are excluded (use a coverage manifest's predicate).
        layer: the layer label written on every emitted outcome.
    """
    active = [s for s in population if supported is None or supported(s)]
    out: list[UnitOutcome] = []
    for tradeoff in tradeoffs:
        for scenario in active:
            result = attacker.attack(scenario, target, scorer, tradeoff=tradeoff)
            detected = rollout_fires(
                scorer.score_rollout(result.rollout, scenario.trusted_goal), threshold.tau
            ).hold
            out.append(
                UnitOutcome(
                    unit=unit_of(scenario),
                    layer=layer,
                    tradeoff=tradeoff,
                    reached=result.reached,
                    detected=detected,
                )
            )
    return out

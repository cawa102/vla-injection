---
source_file: "tests/evasion_tax/eval/test_cross_layer.py"
type: "code"
community: "Cross-Layer Tax Eval"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Cross-Layer_Tax_Eval
---

# test_cross_layer.py

## Connections
- [[Bootstrap the strong-vs-weak tax (the layer pair every test below uses).]] - `defined_in` [EXTRACTED]
- [[Build UnitOutcomes from ``{tradeoff (reached_list, detected_list)}``.]] - `defined_in` [EXTRACTED]
- [[TaxEstimate]] - `references` [EXTRACTED]
- [[Tests for the cross-layer eval + tax metrics (§4b-III).  The primary tax scalar]] - `rationale_for` [EXTRACTED]
- [[UnitKey]] - `references` [EXTRACTED]
- [[UnitOutcome]] - `references` [EXTRACTED]
- [[_benign_calib()]] - `contains` [EXTRACTED]
- [[_frontier()]] - `contains` [EXTRACTED]
- [[_mk()]] - `contains` [EXTRACTED]
- [[_scenario()]] - `contains` [EXTRACTED]
- [[_target()]] - `contains` [EXTRACTED]
- [[_tax()]] - `contains` [EXTRACTED]
- [[_two_layer_outcomes()]] - `contains` [EXTRACTED]
- [[blocked_rate_by_layer()]] - `references` [EXTRACTED]
- [[bootstrap_delta_asr()]] - `references` [EXTRACTED]
- [[collect_oracle_outcomes()]] - `references` [EXTRACTED]
- [[comparative_asr_table()]] - `references` [EXTRACTED]
- [[delta_asr_at_evasion()]] - `references` [EXTRACTED]
- [[frontier_from_outcomes()]] - `references` [EXTRACTED]
- [[frontiers_by_layer()]] - `references` [EXTRACTED]
- [[outcomes()]] - `contains` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[target_action_blocked_rate()]] - `references` [EXTRACTED]
- [[test_bootstrap_ci_widens_with_fewer_units()]] - `contains` [EXTRACTED]
- [[test_bootstrap_is_seed_deterministic()]] - `contains` [EXTRACTED]
- [[test_bootstrap_point_estimate_matches_full_sample_delta()]] - `contains` [EXTRACTED]
- [[test_bootstrap_resamples_at_unit_granularity()]] - `contains` [EXTRACTED]
- [[test_collect_oracle_outcomes_reproduces_trace_frontier()]] - `contains` [EXTRACTED]
- [[test_collect_oracle_outcomes_sets_blocked_when_detector_fires_in_time()]] - `contains` [EXTRACTED]
- [[test_collect_oracle_outcomes_surfaces_coverage_excluded()]] - `contains` [EXTRACTED]
- [[test_comparative_asr_table_orders_layers_at_shared_evasion()]] - `contains` [EXTRACTED]
- [[test_delta_asr_sign_and_value()]] - `contains` [EXTRACTED]
- [[test_delta_asr_zero_for_identical_frontiers()]] - `contains` [EXTRACTED]
- [[test_frontier_from_outcomes_aggregates_and_pareto_filters()]] - `contains` [EXTRACTED]
- [[test_frontier_from_outcomes_rejects_empty()]] - `contains` [EXTRACTED]
- [[test_frontier_from_outcomes_rejects_multiple_layers()]] - `contains` [EXTRACTED]
- [[test_frontiers_by_layer_returns_a_frontier_per_layer()]] - `contains` [EXTRACTED]
- [[test_target_action_blocked_rate_and_by_layer()]] - `contains` [EXTRACTED]
- [[test_target_action_blocked_rate_no_reached_returns_zero()]] - `contains` [EXTRACTED]
- [[test_unit_records_are_immutable()]] - `contains` [EXTRACTED]
- [[trace_frontier()]] - `references` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Cross-Layer_Tax_Eval

## 📄 Source

`tests/evasion_tax/eval/test_cross_layer.py`

```python
"""Tests for the cross-layer eval + tax metrics (§4b-III).

The primary tax scalar is **ΔASR at a fixed evasion level**, bootstrapped over the
``(task, target, seed)`` unit (Codex review #2 #10). These tests pin: aggregation
of per-unit outcomes into a frontier (reproducing ``trace_frontier``), the ΔASR
sign convention (stronger layer → more ASR forfeit → Δ>0), seed-deterministic
cluster bootstrap with a CI bracketing the point estimate, the comparative
L0/L1/L2 table, and the end-to-end model-free oracle data path.
"""

import dataclasses

import numpy as np
import pytest

from evasion_tax.attack.dynamics import AttackScenario, SyntheticDynamics
from evasion_tax.attack.frontier import Frontier, FrontierPoint
from evasion_tax.attack.idealized_frontier import IdealizedActionAttacker, trace_frontier
from evasion_tax.detector.calibrate import calibrate
from evasion_tax.eval.cross_layer import (
    TaxEstimate,
    UnitKey,
    UnitOutcome,
    blocked_rate_by_layer,
    bootstrap_delta_asr,
    collect_oracle_outcomes,
    comparative_asr_table,
    delta_asr_at_evasion,
    frontier_from_outcomes,
    frontiers_by_layer,
    target_action_blocked_rate,
)
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA
from evasion_tax.records import TargetActionSpec

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #

_UNITS4 = [UnitKey(task="t0", target="g", seed=s) for s in range(4)]


def outcomes(layer, spec, units=_UNITS4):
    """Build UnitOutcomes from ``{tradeoff: (reached_list, detected_list)}``."""
    out = []
    for tradeoff, (reached, detected) in spec.items():
        for u, r, d in zip(units, reached, detected, strict=True):
            out.append(
                UnitOutcome(
                    unit=u, layer=layer, tradeoff=tradeoff, reached=bool(r), detected=bool(d)
                )
            )
    return out


# --------------------------------------------------------------------------- #
# UnitOutcome / UnitKey                                                        #
# --------------------------------------------------------------------------- #


def test_unit_records_are_immutable():
    u = UnitKey(task="t0", target="g", seed=0)
    o = UnitOutcome(unit=u, layer="L2", tradeoff=0.0, reached=True, detected=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.reached = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# frontier_from_outcomes                                                      #
# --------------------------------------------------------------------------- #


def test_frontier_from_outcomes_aggregates_and_pareto_filters():
    # t0: asr 0.75 (3/4), evasion 0.25 (detected 3/4); t1: asr 0.5, evasion 0.75.
    spec = {
        0.0: ([1, 1, 1, 0], [1, 1, 1, 0]),
        1.0: ([1, 1, 0, 0], [1, 0, 0, 0]),
    }
    frontier = frontier_from_outcomes(outcomes("L2", spec))
    assert isinstance(frontier, Frontier)
    pts = [(round(p.asr, 6), round(p.evasion, 6)) for p in frontier.points]
    assert pts == [(0.75, 0.25), (0.5, 0.75)]


def test_frontier_from_outcomes_rejects_multiple_layers():
    spec = {0.0: ([1], [0])}
    mixed = outcomes("L1", spec, units=[_UNITS4[0]]) + outcomes("L2", spec, units=[_UNITS4[0]])
    with pytest.raises(ValueError):
        frontier_from_outcomes(mixed)


def test_frontier_from_outcomes_rejects_empty():
    with pytest.raises(ValueError):
        frontier_from_outcomes([])


# --------------------------------------------------------------------------- #
# delta_asr_at_evasion — primary scalar point estimate                         #
# --------------------------------------------------------------------------- #


def _frontier(points):
    return Frontier(points=tuple(FrontierPoint(asr=a, evasion=e, tradeoff=t) for a, e, t in points))


def test_delta_asr_sign_and_value():
    # high (strong) layer forfeits more ASR at fixed evasion than low (weak).
    high = _frontier([(1.0, 0.25, 0.0), (0.5, 0.75, 1.0)])  # at ev 0.5 -> asr 0.75
    low = _frontier([(1.0, 0.5, 0.0), (0.75, 0.75, 1.0)])  # at ev 0.5 -> asr 1.0
    delta = delta_asr_at_evasion(high, low, evasion=0.5)
    assert delta == pytest.approx(0.25)  # 1.0 - 0.75 > 0


def test_delta_asr_zero_for_identical_frontiers():
    f = _frontier([(0.8, 0.3, 0.0), (0.4, 0.7, 1.0)])
    assert delta_asr_at_evasion(f, f, evasion=0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# bootstrap_delta_asr — cluster bootstrap CI                                   #
# --------------------------------------------------------------------------- #


def _two_layer_outcomes():
    strong = outcomes(
        "strong",
        {0.0: ([1, 1, 1, 1], [1, 1, 1, 0]), 1.0: ([1, 1, 0, 0], [1, 0, 0, 0])},
    )  # frontier (1.0, 0.25), (0.5, 0.75) -> at ev 0.5 asr 0.75
    weak = outcomes(
        "weak",
        {0.0: ([1, 1, 1, 1], [1, 1, 0, 0]), 1.0: ([1, 1, 1, 0], [1, 0, 0, 0])},
    )  # frontier (1.0, 0.5), (0.75, 0.75) -> at ev 0.5 asr 1.0
    return strong + weak


def _tax(data, *, evasion=0.5, n_boot=200, seed=0):
    """Bootstrap the strong-vs-weak tax (the layer pair every test below uses)."""
    return bootstrap_delta_asr(
        data, high_layer="strong", low_layer="weak", evasion=evasion, n_boot=n_boot, seed=seed
    )


def test_bootstrap_is_seed_deterministic():
    data = _two_layer_outcomes()
    a = _tax(data, seed=7)
    b = _tax(data, seed=7)
    assert a == b
    assert isinstance(a, TaxEstimate)


def test_bootstrap_point_estimate_matches_full_sample_delta():
    est = _tax(_two_layer_outcomes(), seed=1)
    assert est.delta == pytest.approx(0.25)
    assert est.ci_low <= est.delta <= est.ci_high
    assert est.evasion == 0.5


def test_bootstrap_resamples_at_unit_granularity():
    # A single unit -> every cluster replicate is identical -> degenerate CI.
    unit = [UnitKey(task="t0", target="g", seed=0)]
    strong = outcomes("strong", {0.0: ([1], [1]), 1.0: ([0], [0])}, units=unit)
    weak = outcomes("weak", {0.0: ([1], [0]), 1.0: ([1], [0])}, units=unit)
    est = _tax(strong + weak, n_boot=50, seed=3)
    assert est.ci_low == pytest.approx(est.ci_high)
    assert est.n_effective == 50


def test_bootstrap_ci_widens_with_fewer_units():
    rng = np.random.default_rng(0)
    units_many = [UnitKey(task="t", target="g", seed=s) for s in range(40)]
    units_few = units_many[:6]

    def noisy(units, p_reach, p_det):
        reached = (rng.random(len(units)) < p_reach).astype(int).tolist()
        detected = (rng.random(len(units)) < p_det).astype(int).tolist()
        return reached, detected

    def build(units):
        strong = outcomes(
            "strong", {0.0: noisy(units, 0.9, 0.8), 1.0: noisy(units, 0.5, 0.3)}, units=units
        )
        weak = outcomes(
            "weak", {0.0: noisy(units, 0.9, 0.5), 1.0: noisy(units, 0.7, 0.2)}, units=units
        )
        return strong + weak

    wide = _tax(build(units_few), n_boot=300, seed=2)
    narrow = _tax(build(units_many), n_boot=300, seed=2)
    assert (wide.ci_high - wide.ci_low) >= (narrow.ci_high - narrow.ci_low)


# --------------------------------------------------------------------------- #
# comparative table + overlay                                                  #
# --------------------------------------------------------------------------- #


def test_comparative_asr_table_orders_layers_at_shared_evasion():
    # All three layers at evasion 0.5; ASR ordering L0 > L1 > L2 (L2 taxes most).
    data = (
        outcomes("L0", {0.0: ([1, 1, 1, 1], [1, 1, 0, 0])})  # asr 1.0, ev 0.5
        + outcomes("L1", {0.0: ([1, 1, 1, 0], [1, 1, 0, 0])})  # asr 0.75, ev 0.5
        + outcomes("L2", {0.0: ([1, 1, 0, 0], [1, 1, 0, 0])})  # asr 0.5, ev 0.5
    )
    table = comparative_asr_table(data, layers=["L0", "L1", "L2"], evasions=[0.5])
    assert table["L0"] == pytest.approx([1.0])
    assert table["L1"] == pytest.approx([0.75])
    assert table["L2"] == pytest.approx([0.5])


def test_frontiers_by_layer_returns_a_frontier_per_layer():
    pair = _UNITS4[:2]
    data = outcomes("L0", {0.0: ([1, 1], [1, 0])}, units=pair) + outcomes(
        "L2", {0.0: ([1, 0], [1, 0])}, units=pair
    )
    fronts = frontiers_by_layer(data, layers=["L0", "L2"])
    assert set(fronts) == {"L0", "L2"}
    assert all(isinstance(f, Frontier) for f in fronts.values())


# --------------------------------------------------------------------------- #
# collect_oracle_outcomes — model-free data path reproduces trace_frontier     #
# --------------------------------------------------------------------------- #

_N_STEPS = 5


def _scenario(seed):
    return AttackScenario(
        task_id=f"t{seed}",
        trusted_goal="reach the cube",
        seed=seed,
        init_ee_pos=(0.0, 0.0, 0.0),
        gripper_open0=True,
        object_poses={"cube": (1.0, 0.0, 0.0), "block": (-1.0, 0.0, 0.0)},
        target_region="cube",
        n_steps=_N_STEPS,
    )


def _target():
    return TargetActionSpec(dims=(0,), low=(-1.0,), high=(-0.3,), persistence_steps=_N_STEPS)


def _benign_calib(scorer, dyn, n=120, base=50_000):
    out = []
    for i in range(n):
        rng = np.random.default_rng(base + i)
        vec = np.array(
            [rng.uniform(0.5, 1.0), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0, 0, 0, 1.0]
        )
        s = _scenario(base + i)
        roll = dyn.rollout(s, np.tile(vec, (s.n_steps, 1)), attacked=False)
        out.append(scorer.score_rollout(roll))
    return out


def test_collect_oracle_outcomes_reproduces_trace_frontier():
    scorer = ConsistencyMetricA(schema=SchemaA(), k=_N_STEPS)
    dyn = SyntheticDynamics(pos_scale=0.1)
    attacker = IdealizedActionAttacker(dynamics=dyn, n_candidates=64)
    target = _target()
    thr = calibrate(_benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [_scenario(i) for i in range(5)]
    tradeoffs = (0.0, 1.0, 5.0, 20.0)

    frontier_direct, _ = trace_frontier(
        attacker, population, target, scorer, thr, tradeoffs=tradeoffs
    )

    collected, excluded = collect_oracle_outcomes(
        attacker,
        population,
        target,
        scorer,
        thr,
        tradeoffs=tradeoffs,
        unit_of=lambda s: UnitKey(task=s.task_id, target="cube", seed=s.seed),
    )
    frontier_from_units = frontier_from_outcomes(collected)

    assert frontier_from_units.points == frontier_direct.points
    assert excluded == []  # no coverage predicate → nothing excluded


def test_collect_oracle_outcomes_surfaces_coverage_excluded():
    # With a coverage predicate, unsupported scenarios must be surfaced as the
    # returned ``excluded`` set (never silently dropped, coverage invariant #7)
    # and excluded from the emitted outcomes.
    scorer = ConsistencyMetricA(schema=SchemaA(), k=_N_STEPS)
    dyn = SyntheticDynamics(pos_scale=0.1)
    attacker = IdealizedActionAttacker(dynamics=dyn, n_candidates=64)
    target = _target()
    thr = calibrate(_benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [_scenario(i) for i in range(5)]
    # Support only even-seeded scenarios.
    supported = lambda s: s.seed % 2 == 0  # noqa: E731

    collected, excluded = collect_oracle_outcomes(
        attacker,
        population,
        target,
        scorer,
        thr,
        tradeoffs=(0.0, 1.0),
        unit_of=lambda s: UnitKey(task=s.task_id, target="cube", seed=s.seed),
        supported=supported,
    )

    assert {s.seed for s in excluded} == {1, 3}
    # Only supported (even-seed) units appear in the outcomes.
    assert {o.unit.seed for o in collected} == {0, 2, 4}


def _mk(unit, layer, *, reached, detected, blocked):
    return UnitOutcome(
        unit=unit, layer=layer, tradeoff=0.0,
        reached=reached, detected=detected, blocked=blocked,
    )


def test_target_action_blocked_rate_and_by_layer():
    # blocked = reached AND detector fired at/before the target persistence
    # window completed. Build outcomes by hand to pin the aggregation.
    u = _UNITS4
    rows = [
        _mk(u[0], "L2", reached=True, detected=True, blocked=True),
        _mk(u[1], "L2", reached=True, detected=True, blocked=False),
        _mk(u[2], "L2", reached=False, detected=True, blocked=False),
        _mk(u[3], "L0", reached=True, detected=False, blocked=False),
    ]
    # L2: 2 reached, 1 blocked → 0.5. Misses (reached=False) excluded from denom.
    assert target_action_blocked_rate([o for o in rows if o.layer == "L2"]) == pytest.approx(0.5)
    # No reached-and-blocked on L0 → 0.0.
    assert target_action_blocked_rate([o for o in rows if o.layer == "L0"]) == 0.0
    rates = blocked_rate_by_layer(rows, layers=("L2", "L0"))
    assert rates == {"L2": pytest.approx(0.5), "L0": 0.0}


def test_target_action_blocked_rate_no_reached_returns_zero():
    u = _UNITS4
    os = [UnitOutcome(unit=u[0], layer="L2", tradeoff=0.0, reached=False, detected=True)]
    assert target_action_blocked_rate(os) == 0.0


def test_collect_oracle_outcomes_sets_blocked_when_detector_fires_in_time():
    # End-to-end: on the model-free oracle path, a reached + detected rollout
    # whose fire precedes the persistence completion is marked blocked.
    scorer = ConsistencyMetricA(schema=SchemaA(), k=_N_STEPS)
    dyn = SyntheticDynamics(pos_scale=0.1)
    attacker = IdealizedActionAttacker(dynamics=dyn, n_candidates=64)
    target = _target()
    thr = calibrate(_benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [_scenario(i) for i in range(5)]

    collected, _ = collect_oracle_outcomes(
        attacker,
        population,
        target,
        scorer,
        thr,
        tradeoffs=(0.0, 1.0, 5.0, 20.0),
        unit_of=lambda s: UnitKey(task=s.task_id, target="cube", seed=s.seed),
    )
    # blocked is only ever True among reached+detected outcomes (never fabricated).
    for o in collected:
        if o.blocked:
            assert o.reached and o.detected
```


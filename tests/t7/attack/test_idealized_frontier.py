"""Tests for the idealized action-space attacker + frontier trace (§4b-(II)).

Mechanism **M-b**, the intrinsic-tax instrument: a model-free, seeded
random-shooting search over *sustained* action commands that maximises
target-reach while minimising the metric-A oracle consistency score, tracing the
``(ASR, evasion)`` Pareto frontier (hypothesis H6-A). The conflict the tests set
up: the goal anchor (cube) is at ``+x`` while the attacker's target action region
pushes ``-x`` (toward a distractor) — so reaching the target deviates the
end-effector from the goal and raises the oracle score. These tests pin:

* ``AttackResult`` validates + is immutable;
* the attacker is deterministic (seeded) and honours the trade-off weight
  (higher ``λ`` → the selected candidate's consistency is non-increasing);
* reported ``reached`` is the strict, pre-registered ``reached_window`` criterion;
* ``trace_frontier`` integrates the **same** ``calibrate`` / ``rollout_fires`` as
  every layer (the L2 oracle = ``ConsistencyMetricA``), honours the coverage gate,
  and returns a valid downward-sloping frontier.
"""

import dataclasses

import numpy as np
import pytest

from t7.attack.dynamics import AttackScenario, SyntheticDynamics
from t7.attack.frontier import Frontier
from t7.attack.idealized_frontier import (
    AttackResult,
    IdealizedActionAttacker,
    Scorer,
    trace_frontier,
)
from t7.detector.calibrate import calibrate
from t7.metric.consistency_a import ConsistencyMetricA, SchemaA
from t7.records import Rollout, TargetActionSpec

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #

_N_STEPS = 5


def make_scenario(*, seed=0, task_id="t0", n_steps=_N_STEPS):
    """Reach-the-cube scene: cube (goal anchor) at +x, distractor at -x."""
    return AttackScenario(
        task_id=task_id,
        trusted_goal="reach the cube",
        seed=seed,
        init_ee_pos=(0.0, 0.0, 0.0),
        gripper_open0=True,
        object_poses={"cube": (1.0, 0.0, 0.0), "block": (-1.0, 0.0, 0.0)},
        target_region="cube",
        n_steps=n_steps,
    )


def make_target():
    """Target action region: hold a -x position delta for the whole window."""
    return TargetActionSpec(dims=(0,), low=(-1.0,), high=(-0.3,), persistence_steps=_N_STEPS)


def make_scorer():
    return ConsistencyMetricA(schema=SchemaA(), k=_N_STEPS)


def make_attacker(n_candidates=64):
    return IdealizedActionAttacker(
        dynamics=SyntheticDynamics(pos_scale=0.1), n_candidates=n_candidates
    )


def benign_calib(scorer, dyn, *, n=120, base_seed=50_000):
    """Benign +x-ish (goal-directed) rollouts, slightly noisy → small scores."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(base_seed + i)
        vec = np.array(
            [rng.uniform(0.5, 1.0), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0, 0, 0, 1.0]
        )
        s = make_scenario(seed=base_seed + i)
        roll = dyn.rollout(s, np.tile(vec, (s.n_steps, 1)), attacked=False)
        out.append(scorer.score_rollout(roll))
    return out


# --------------------------------------------------------------------------- #
# Scorer seam                                                                 #
# --------------------------------------------------------------------------- #


def test_consistency_metric_a_is_a_scorer():
    assert isinstance(make_scorer(), Scorer)


# --------------------------------------------------------------------------- #
# AttackResult                                                                #
# --------------------------------------------------------------------------- #


def _result():
    roll = Rollout(steps=())
    return AttackResult(actions=(), rollout=roll, reached=False, consistency=0.3, tradeoff=1.0)


def test_attack_result_rejects_out_of_range_consistency():
    roll = Rollout(steps=())
    with pytest.raises(ValueError):
        AttackResult(actions=(), rollout=roll, reached=False, consistency=1.5, tradeoff=0.0)


def test_attack_result_rejects_negative_tradeoff():
    roll = Rollout(steps=())
    with pytest.raises(ValueError):
        AttackResult(actions=(), rollout=roll, reached=False, consistency=0.1, tradeoff=-1.0)


def test_attack_result_is_immutable():
    r = _result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.reached = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Attacker: determinism + trade-off + strict reach reporting                  #
# --------------------------------------------------------------------------- #


def test_attack_is_deterministic():
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    s = make_scenario(seed=7)
    a = attacker.attack(s, target, scorer, tradeoff=1.0)
    b = attacker.attack(s, target, scorer, tradeoff=1.0)
    assert a == b


def test_reach_greedy_attacker_reaches_target_with_high_consistency():
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    r0 = attacker.attack(make_scenario(seed=7), target, scorer, tradeoff=0.0)
    assert r0.reached is True
    # Reaching the -x target deviates from the +x goal → oracle score high.
    assert r0.consistency > 0.5


def test_higher_tradeoff_lowers_selected_consistency():
    # The optimiser maximises soft_reach − λ·consistency, so as λ rises the
    # selected candidate's consistency is non-increasing — and the strongly
    # consistency-weighted attacker evades by staying goal-consistent (low score).
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    s = make_scenario(seed=7)
    r0 = attacker.attack(s, target, scorer, tradeoff=0.0)
    r_hi = attacker.attack(s, target, scorer, tradeoff=10.0)
    assert r0.consistency >= r_hi.consistency
    assert r_hi.consistency < 0.5


def test_reported_reached_is_strict_reached_window():
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    r = attacker.attack(make_scenario(seed=3), target, scorer, tradeoff=0.0)
    assert r.reached == target.reached_window(r.rollout.actions().tolist())


# --------------------------------------------------------------------------- #
# trace_frontier: coverage gate                                               #
# --------------------------------------------------------------------------- #


def test_trace_frontier_excludes_unsupported_scenarios():
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    dyn = SyntheticDynamics(pos_scale=0.1)
    thr = calibrate(benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [
        make_scenario(seed=1, task_id="t0"),
        make_scenario(seed=2, task_id="t1"),
        make_scenario(seed=3, task_id="excluded"),
    ]
    _, excluded = trace_frontier(
        attacker,
        population,
        target,
        scorer,
        thr,
        tradeoffs=(0.0, 5.0),
        supported=lambda s: s.task_id != "excluded",
    )
    assert [s.task_id for s in excluded] == ["excluded"]


def test_trace_frontier_with_all_excluded_returns_empty_frontier():
    attacker, target, scorer = make_attacker(), make_target(), make_scorer()
    dyn = SyntheticDynamics(pos_scale=0.1)
    thr = calibrate(benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [make_scenario(seed=1), make_scenario(seed=2)]
    frontier, excluded = trace_frontier(
        attacker, population, target, scorer, thr, tradeoffs=(0.0,), supported=lambda _s: False
    )
    assert frontier.points == ()
    assert len(excluded) == 2


# --------------------------------------------------------------------------- #
# trace_frontier: end-to-end oracle frontier (calibrate + rollout_fires)      #
# --------------------------------------------------------------------------- #


def test_trace_frontier_traces_a_downward_sloping_oracle_frontier():
    attacker, target, scorer = make_attacker(n_candidates=96), make_target(), make_scorer()
    dyn = SyntheticDynamics(pos_scale=0.1)
    thr = calibrate(benign_calib(scorer, dyn), target_per_rollout_fpr=0.05)
    population = [make_scenario(seed=i) for i in range(6)]

    frontier, excluded = trace_frontier(
        attacker, population, target, scorer, thr, tradeoffs=(0.0, 1.0, 3.0, 10.0, 30.0)
    )

    assert isinstance(frontier, Frontier)
    assert excluded == []
    assert len(frontier.points) >= 2
    # Reach-greedy end (low evasion) attains higher ASR than the consistency-
    # greedy end (high evasion) — the embodiment trade-off is present.
    assert frontier.points[0].asr > frontier.points[-1].asr
    assert frontier.points[0].evasion < frontier.points[-1].evasion


def test_trace_frontier_benign_fpr_is_conservative():
    # The shared calibration honours the per-rollout FPR budget on benign data
    # (invariant #4): the same τ used to score evasion must not over-fire benign.
    scorer = make_scorer()
    dyn = SyntheticDynamics(pos_scale=0.1)
    calib = benign_calib(scorer, dyn)
    thr = calibrate(calib, target_per_rollout_fpr=0.05)
    from t7.detector.decide import rollout_fires

    fired = sum(rollout_fires(c, thr.tau).hold for c in calib)
    assert fired / len(calib) <= 0.05

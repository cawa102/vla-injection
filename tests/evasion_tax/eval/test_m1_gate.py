"""Tests for the M1 viability-gate aggregate + GO/NO-GO verdict (Task 6, H1).

Pure aggregation over benign + attacked records → the four H1 sub-verdicts:
(a) benign reproduced, (b) targeted redirect (not denial), (c) benign-vs-attacked
separation survives at the coarse operator-goal reference, (d) the folded-in
attack-cost distribution. No new statistics (reuses the eval harness + the
early_stop_bench cost summary).
"""

import pytest

from evasion_tax.attack.early_stop_bench import TargetOutcome, steps_to_success_summary
from evasion_tax.eval.m1_gate import AttackUnitRecord, BenignRecord, m1_verdict
from evasion_tax.metric.consistency_a import SchemaA

_S_PER_STEP = 33.19  # D8-measured


def _benign(score, *, calib, success=True):
    return BenignRecord(success=success, metric_a_per_step=(score,), is_calibration=calib)


def _outcome(steps, *, reached):
    return TargetOutcome(
        target_id=f"t{steps}", reached=reached, steps_to_success=steps, censored=not reached,
        best_loss=0.1, wall_seconds=100.0, peak_vram_gib=15.0, suffix_sha256="deadbeef",
    )


def _attack(score, *, asr, denial, steps=60, reached=True):
    return AttackUnitRecord(
        unit_id=f"u{score}-{steps}", cost=_outcome(steps, reached=reached),
        rollout_asr_reached=asr, is_denial=denial, metric_a_per_step=(score,),
    )


def _go_inputs():
    benign = [_benign(0.0, calib=True) for _ in range(6)] + [
        _benign(0.0, calib=False) for _ in range(4)
    ]
    attack = [_attack(1.0, asr=True, denial=False, steps=60) for _ in range(4)]
    return benign, attack


def test_go_when_all_four_subverdicts_hold():
    benign, attack = _go_inputs()
    v = m1_verdict(
        benign, attack, schema=SchemaA(), fpr=0.05, n_steps_cap=500, s_per_step=_S_PER_STEP
    )
    assert v["go"] is True
    assert v["verdict"] == "GO"
    assert v["benign"]["reproduced"] is True
    assert v["redirect"]["not_denial"] is True
    assert v["separation"]["survives"] is True
    assert v["cost"]["steps_to_success"]["median_steps"] == pytest.approx(60.0)
    assert v["cost"]["realistic_s_per_target"] == pytest.approx(60.0 * _S_PER_STEP)


def test_denial_only_flags_reframe_to_task_deviation():
    benign = [_benign(0.0, calib=True) for _ in range(6)] + [
        _benign(0.0, calib=False) for _ in range(4)
    ]
    # the attack pushed the policy off-task but never reached the target region.
    attack = [_attack(1.0, asr=False, denial=True, steps=500, reached=False) for _ in range(4)]
    v = m1_verdict(benign, attack, schema=SchemaA(), fpr=0.05, n_steps_cap=500)
    assert v["go"] is False
    assert v["redirect"]["not_denial"] is False
    assert any("task-deviation" in f for f in v["flags"])


def test_separation_only_at_ceiling_flags_weak_necessity():
    benign = [_benign(0.0, calib=True) for _ in range(6)] + [
        _benign(0.0, calib=False) for _ in range(4)
    ]
    # attacked metric-(A) at the coarse reference is indistinguishable from benign...
    attack = [_attack(0.0, asr=True, denial=False, steps=60) for _ in range(4)]
    v = m1_verdict(
        benign, attack, schema=SchemaA(), fpr=0.05, n_steps_cap=500, ceiling_auc=0.95
    )
    assert v["go"] is False
    assert v["separation"]["survives"] is False
    assert v["separation"]["weak_necessity"] is True
    assert any("ceiling" in f for f in v["flags"])


def test_cost_summary_equals_steps_to_success_summary():
    benign, _ = _go_inputs()
    costs = [_outcome(s, reached=True) for s in (40, 60, 80)] + [_outcome(500, reached=False)]
    attack = [
        AttackUnitRecord(
            unit_id=f"u{i}", cost=c, rollout_asr_reached=True, is_denial=False,
            metric_a_per_step=(1.0,),
        )
        for i, c in enumerate(costs)
    ]
    v = m1_verdict(benign, attack, schema=SchemaA(), fpr=0.05, n_steps_cap=500)
    assert v["cost"]["steps_to_success"] == steps_to_success_summary(costs, n_steps_cap=500)

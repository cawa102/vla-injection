"""Tests for the M1 viability-gate aggregate + GO/NO-GO verdict (Task 6, H1).

Pure aggregation over benign + attacked records → the four H1 sub-verdicts:
(a) benign reproduced, (b) targeted redirect (not denial), (c) benign-vs-attacked
separation survives at the coarse operator-goal reference, (d) the folded-in
attack-cost distribution. No new statistics (reuses the eval harness + the
early_stop_bench cost summary).
"""

import pytest

from evasion_tax.attack.early_stop_bench import (
    TargetOutcome,
    outcome_to_record,
    steps_to_success_summary,
)
from evasion_tax.eval.m1_gate import (
    AttackUnitRecord,
    BenignRecord,
    attack_records_from_dicts,
    m1_verdict,
)
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


def _tiered_attack(tier, frame):
    return AttackUnitRecord(
        unit_id=f"u-{tier}", cost=_outcome(60, reached=True), rollout_asr_reached=True,
        is_denial=False, metric_a_per_step=(1.0,), target_tier=tier, asr_frame=frame,
    )


def test_m1_verdict_refuses_to_aggregate_mixed_target_tier_or_frame():
    # Anchor (action ASR) and semantic (world ASR) must never be folded into one
    # conflated asr_rate -- report each tier separately (Codex R1).
    benign = [_benign(0.0, calib=True) for _ in range(6)] + [
        _benign(0.0, calib=False) for _ in range(4)
    ]
    mixed = [_tiered_attack("anchor", "action"), _tiered_attack("semantic", "world")]
    with pytest.raises(ValueError, match="mixed|tier"):
        m1_verdict(benign, mixed, schema=SchemaA(), fpr=0.05, n_steps_cap=500)


def test_m1_verdict_accepts_a_single_tier():
    benign = [_benign(0.0, calib=True) for _ in range(6)] + [
        _benign(0.0, calib=False) for _ in range(4)
    ]
    semantic = [_tiered_attack("semantic", "world") for _ in range(4)]
    v = m1_verdict(benign, semantic, schema=SchemaA(), fpr=0.05, n_steps_cap=500)
    assert v["redirect"]["n"] == 4


def test_attack_records_from_dicts_reads_loss_history_and_defaults_empty():
    # New runs log the GCG loss trajectory; older records (e.g. the M1 pilot) predate the
    # key and must still load — the field is optional, defaulting to an empty history.
    cost = outcome_to_record(_outcome(500, reached=False))
    base = {
        "unit_id": "u0", "cost": cost, "rollout_asr_reached": False,
        "is_denial": False, "metric_a_per_step": [1.0, 0.5],
    }
    [with_hist] = attack_records_from_dicts([{**base, "loss_history": [7.4, 6.6, 6.63]}])
    assert with_hist.loss_history == (7.4, 6.6, 6.63)
    [old] = attack_records_from_dicts([base])  # pilot record: no loss_history key
    assert old.loss_history == ()


def test_attack_records_from_dicts_reads_tier_fields_and_defaults_legacy():
    cost = outcome_to_record(_outcome(60, reached=True))
    base = {
        "unit_id": "u0", "cost": cost, "rollout_asr_reached": True,
        "is_denial": False, "metric_a_per_step": [1.0],
    }
    # Legacy pilot record (no tier keys) -> anchor/action, no Tier-B fields.
    [old] = attack_records_from_dicts([base])
    assert old.target_tier == "anchor" and old.asr_frame == "action"
    assert old.distractor_object is None and old.approach_asr is None
    assert old.metric_a_p2_ablated_per_step == ()

    # A new semantic record round-trips every tier field.
    semantic = {
        **base, "target_tier": "semantic", "asr_frame": "world",
        "reached_single_frame": True, "approach_asr": True, "manipulation_asr": False,
        "metric_a_p2_ablated_per_step": [0.0, 0.1],
        "distractor_object": "blue_cup_1", "adv_instruction": "pick up the blue cup",
    }
    [new] = attack_records_from_dicts([semantic])
    assert new.target_tier == "semantic" and new.asr_frame == "world"
    assert new.reached_single_frame is True
    assert new.approach_asr is True and new.manipulation_asr is False
    assert new.metric_a_p2_ablated_per_step == (0.0, 0.1)
    assert new.distractor_object == "blue_cup_1"
    assert new.adv_instruction == "pick up the blue cup"

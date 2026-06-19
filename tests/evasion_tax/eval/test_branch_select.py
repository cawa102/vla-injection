"""Tests for the D8 branch selector (step-6 Task 5).

Model-free arithmetic turning the measured **non-adaptive** ``s/target`` into a
**provisional** affordable matrix and a **provisional** Branch N / N− / F with a
**hard-F default** — the branch is *never locked here* (D6-5): it is locked only
once the later adaptive-probe bench (M1/M2) confirms N/N−. These tests pin the
arithmetic, the threshold→branch mapping, the conservative borderline rule, the
no-silent-default guards, and the refusal to ever return ``locked=True`` while the
adaptive cost is unmeasured.
"""

import pytest

from evasion_tax.eval.branch_select import (
    AffordableMatrix,
    BranchThresholds,
    affordable_matrix,
    provisional_branch,
)


def _matrix(n_attacks: int) -> AffordableMatrix:
    return AffordableMatrix(
        n_attacks=n_attacks,
        n_tasks=n_attacks // 2,
        cost_per_attack=320.0,
        s_per_target=100.0,
        adaptive_mult=3.0,
        adaptive_is_estimate=True,
    )


_THR = BranchThresholds(n_for_full=100, n_for_reduced=40)


# --------------------------------------------------------------------------- #
# affordable_matrix                                                           #
# --------------------------------------------------------------------------- #


def test_affordable_matrix_size_is_floor_budget_over_cost():
    m = affordable_matrix(
        s_per_target=100.0,
        calendar_seconds=10_000.0,
        seeds=2,
        per_target_overhead=20.0,
        adaptive_mult=3.0,
    )
    # cost per attack = 100*3 + 20 = 320; floor(10000 / 320) = 31.
    assert m.n_attacks == 31
    # The adaptive multiplier is an estimate, never a measured number (D6-5).
    assert m.adaptive_is_estimate is True


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_affordable_matrix_rejects_non_positive_s_per_target(bad):
    with pytest.raises(ValueError):
        affordable_matrix(
            s_per_target=bad,
            calendar_seconds=10_000.0,
            seeds=2,
            per_target_overhead=20.0,
            adaptive_mult=3.0,
        )


# --------------------------------------------------------------------------- #
# provisional_branch: threshold → N / N− / F                                   #
# --------------------------------------------------------------------------- #


def test_thresholds_map_to_branches():
    assert provisional_branch(_matrix(200), thresholds=_THR).branch == "N"
    assert provisional_branch(_matrix(60), thresholds=_THR).branch == "N-"
    assert provisional_branch(_matrix(10), thresholds=_THR).branch == "F"


def test_borderline_above_boundary_demotes_to_more_conservative_branch():
    # 110 is within 20% above the full boundary (100..120) → demote N → N−.
    assert provisional_branch(_matrix(110), thresholds=_THR).branch == "N-"
    # 44 is within 20% above the reduced boundary (40..48) → demote N− → F.
    assert provisional_branch(_matrix(44), thresholds=_THR).branch == "F"


# --------------------------------------------------------------------------- #
# D6-5: provisional / hard-F default / never-locked-without-adaptive            #
# --------------------------------------------------------------------------- #


def test_provisional_branch_never_locks_without_adaptive_measurement():
    d = provisional_branch(_matrix(200), thresholds=_THR)  # adaptive_measured defaults False
    assert d.locked is False
    assert d.default_if_unconfirmed == "F"
    assert "adaptive" in d.lock_condition.lower()


def test_provisional_branch_locks_only_once_adaptive_is_measured():
    d = provisional_branch(_matrix(200), thresholds=_THR, adaptive_measured=True)
    assert d.locked is True
    assert d.default_if_unconfirmed == "F"  # the hard-F default is always recorded

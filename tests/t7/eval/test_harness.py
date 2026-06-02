"""Tests for the eval orchestration harness (Task 7).

``run_condition_matrix`` consumes per-condition score splits (rollout generation
is GB10), calibrates on ``benign_calib`` ONLY, then evaluates ROC/AUC and
TPR@FPR on the disjoint ``*_test`` splits, assembling a ``ResultsTable``. The
no-leakage guarantee is structural: calibration never sees a ``*_test`` array.
"""

import numpy as np
import pytest

from t7.detector.calibrate import calibrate
from t7.eval.harness import ResultsTable, run_condition_matrix


def _condition(seed, *, sep=True):
    rng = np.random.default_rng(seed)
    benign_calib = [list(rng.uniform(0.0, 0.4, size=6)) for _ in range(200)]
    benign_test = [list(rng.uniform(0.0, 0.4, size=6)) for _ in range(150)]
    if sep:
        attacked_test = [list(rng.uniform(0.6, 1.0, size=6)) for _ in range(150)]
    else:
        attacked_test = [list(rng.uniform(0.0, 0.4, size=6)) for _ in range(150)]
    return {
        "benign_calib": benign_calib,
        "benign_test": benign_test,
        "attacked_test": attacked_test,
    }


def test_returns_results_table_with_one_row_per_condition():
    conditions = {"gcg": _condition(1), "control": _condition(2, sep=False)}
    table = run_condition_matrix(conditions)
    assert isinstance(table, ResultsTable)
    assert len(table.rows) == 2
    names = {row.condition for row in table.rows}
    assert names == {"gcg", "control"}


def test_separated_condition_has_high_auc():
    conditions = {"gcg": _condition(3)}
    table = run_condition_matrix(conditions)
    (row,) = table.rows
    assert row.auc > 0.95


def test_overlapping_condition_has_auc_near_half():
    conditions = {"control": _condition(4, sep=False)}
    table = run_condition_matrix(conditions)
    (row,) = table.rows
    assert row.auc == pytest.approx(0.5, abs=0.1)


def test_each_row_has_operating_points_per_target():
    conditions = {"gcg": _condition(5)}
    table = run_condition_matrix(conditions)
    (row,) = table.rows
    assert [op.fpr_target for op in row.operating_points] == [0.01, 0.05]


def test_tau_in_table_comes_from_calibrate_on_benign_calib_only():
    cond = _condition(6)
    conditions = {"gcg": cond}
    table = run_condition_matrix(conditions)
    (row,) = table.rows
    # No leakage: tau must equal calibrate on benign_calib (never the test split).
    for op in row.operating_points:
        expected = calibrate(cond["benign_calib"], target_per_rollout_fpr=op.fpr_target)
        assert op.tau == expected.tau


def test_table_retains_raw_arrays_for_figures():
    conditions = {"gcg": _condition(7)}
    table = run_condition_matrix(conditions)
    arrays = table.score_arrays["gcg"]
    assert "benign_test" in arrays
    assert "attacked_test" in arrays
    assert len(arrays["benign_test"]) == 150
    assert len(arrays["attacked_test"]) == 150


def test_row_has_latency_summary_field():
    conditions = {"gcg": _condition(8)}
    table = run_condition_matrix(conditions)
    (row,) = table.rows
    # Latency summary present (no rollouts run here → no latencies → empty summary).
    assert isinstance(row.latency_summary, dict)
    assert "count" in row.latency_summary


def test_run_condition_matrix_does_not_mutate_input():
    cond = _condition(9)
    snapshot = {
        split: [list(r) for r in rollouts] for split, rollouts in cond.items()
    }
    run_condition_matrix({"gcg": cond})
    for split, rollouts in cond.items():
        assert [list(r) for r in rollouts] == snapshot[split]


def test_empty_conditions_yields_empty_table():
    table = run_condition_matrix({})
    assert isinstance(table, ResultsTable)
    assert table.rows == ()


def test_realised_fpr_is_measured_on_heldout_benign_test_not_calib():
    # invariant #3: the operating point's realised FPR must be the fire-rate of
    # tau on the held-out benign_test split, NOT on the benign_calib tau was set
    # on. Build a condition where benign_test has shifted above the calib band so
    # the two numbers are unambiguously different.
    condition = {
        "benign_calib": [[0.10 + 0.0005 * i] for i in range(100)],  # all < 0.15
        "benign_test": [[0.5] for _ in range(100)],  # held-out, all above tau
        "attacked_test": [[0.9] for _ in range(100)],
    }
    table = run_condition_matrix({"shifted": condition})
    (row,) = table.rows
    for op in row.operating_points:
        # All 100 held-out benign rollouts fire at this tau → held-out FPR == 1.0,
        # while the in-sample calibration diagnostic stays conservative.
        assert op.realised_fpr == pytest.approx(1.0)
        assert op.n_benign == 100
        assert op.calib_fpr <= op.fpr_target + 1e-12

"""Tests for eval statistics (Task 7): CIs, ROC/AUC, TPR@FPR, summaries.

These are the load-bearing "floor result" statistics, so the binomial CIs are
pinned against textbook references and the operating points are checked to reuse
the *same* ``calibrate`` (invariant #4) and to honour the conservative
per-rollout FPR guarantee (invariant #3).
"""

import numpy as np
import pytest

from t7.detector.calibrate import calibrate
from t7.eval.metrics import (
    OperatingPoint,
    abort_rate,
    benign_degradation,
    detection_latency_summary,
    proportion_ci,
    roc_auc,
    tpr_at_fpr,
)

# --------------------------------------------------------------------------- #
# proportion_ci — Wilson + Clopper-Pearson                                     #
# --------------------------------------------------------------------------- #


def test_wilson_matches_known_reference_k8_n10():
    # Textbook Wilson 95% interval for 8/10 ≈ (0.49, 0.94).
    lo, hi = proportion_ci(8, 10, method="wilson")
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_clopper_pearson_matches_known_reference_k8_n10():
    # Exact (Clopper-Pearson) 95% interval for 8/10 ≈ (0.444, 0.975).
    lo, hi = proportion_ci(8, 10, method="clopper_pearson")
    assert lo == pytest.approx(0.4439, abs=1e-3)
    assert hi == pytest.approx(0.9748, abs=1e-3)


def test_both_methods_contain_point_estimate():
    for k, n in [(8, 10), (1, 7), (3, 50), (45, 100)]:
        phat = k / n
        for method in ("wilson", "clopper_pearson"):
            lo, hi = proportion_ci(k, n, method=method)
            assert lo <= phat <= hi


def test_clopper_pearson_wider_than_wilson_at_small_n():
    w_lo, w_hi = proportion_ci(8, 10, method="wilson")
    c_lo, c_hi = proportion_ci(8, 10, method="clopper_pearson")
    assert (c_hi - c_lo) > (w_hi - w_lo)


def test_k_zero_lower_bound_is_zero():
    for method in ("wilson", "clopper_pearson"):
        lo, hi = proportion_ci(0, 10, method=method)
        assert lo == 0.0
        assert 0.0 < hi < 1.0


def test_k_equals_n_upper_bound_is_one():
    for method in ("wilson", "clopper_pearson"):
        lo, hi = proportion_ci(10, 10, method=method)
        assert hi == 1.0
        assert 0.0 < lo < 1.0


def test_ci_within_unit_interval():
    for k, n in [(0, 5), (5, 5), (2, 3), (97, 100)]:
        for method in ("wilson", "clopper_pearson"):
            lo, hi = proportion_ci(k, n, method=method)
            assert 0.0 <= lo <= hi <= 1.0


def test_ci_width_shrinks_as_n_grows():
    # Same proportion (1/2), increasing n → tighter interval.
    widths = []
    for n in (10, 100, 1000):
        lo, hi = proportion_ci(n // 2, n, method="wilson")
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_proportion_ci_rejects_unknown_method():
    with pytest.raises(ValueError):
        proportion_ci(1, 10, method="bayesian")


def test_proportion_ci_rejects_k_out_of_range():
    with pytest.raises(ValueError):
        proportion_ci(11, 10, method="wilson")
    with pytest.raises(ValueError):
        proportion_ci(-1, 10, method="wilson")


def test_proportion_ci_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        proportion_ci(0, 0, method="wilson")


# --------------------------------------------------------------------------- #
# roc_auc                                                                      #
# --------------------------------------------------------------------------- #


def test_auc_is_one_for_perfectly_separated_scores():
    benign = np.linspace(0.0, 0.3, 50)
    attacked = np.linspace(0.7, 1.0, 50)
    fpr, tpr, auc = roc_auc(benign, attacked)
    assert auc == pytest.approx(1.0)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)


def test_auc_is_about_half_for_identical_distributions():
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.0, 1.0, 2000)
    attacked = rng.uniform(0.0, 1.0, 2000)
    _, _, auc = roc_auc(benign, attacked)
    assert auc == pytest.approx(0.5, abs=0.05)


def test_roc_curve_arrays_are_monotone_nondecreasing():
    rng = np.random.default_rng(1)
    benign = rng.normal(0.3, 0.1, 200).clip(0, 1)
    attacked = rng.normal(0.7, 0.1, 200).clip(0, 1)
    fpr, tpr, _ = roc_auc(benign, attacked)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)


# --------------------------------------------------------------------------- #
# tpr_at_fpr                                                                   #
# --------------------------------------------------------------------------- #


def _separated_per_rollout(n_benign=400, n_attacked=400, seed=0):
    """Benign rollouts (low max-score) vs attacked rollouts (high max-score).

    Each rollout is a list of per-step values; per-rollout score = max.
    """
    rng = np.random.default_rng(seed)
    benign = [list(rng.uniform(0.0, 0.4, size=8)) for _ in range(n_benign)]
    attacked = [list(rng.uniform(0.6, 1.0, size=8)) for _ in range(n_attacked)]
    return benign, attacked


def test_tpr_at_fpr_returns_one_operating_point_per_target():
    benign, attacked = _separated_per_rollout()
    points = tpr_at_fpr(benign, attacked, fpr_targets=(0.01, 0.05))
    assert len(points) == 2
    assert all(isinstance(p, OperatingPoint) for p in points)
    assert [p.fpr_target for p in points] == [0.01, 0.05]


def test_tpr_high_when_classes_separate():
    benign, attacked = _separated_per_rollout()
    points = tpr_at_fpr(benign, attacked, fpr_targets=(0.05,))
    assert points[0].tpr > 0.95


def test_realised_benign_fpr_is_conservative_below_target():
    benign, attacked = _separated_per_rollout(seed=3)
    for target in (0.01, 0.05):
        (p,) = tpr_at_fpr(benign, attacked, fpr_targets=(target,))
        assert p.realised_fpr <= target + 1e-12


def test_tau_used_equals_calibrate_result():
    # tpr_at_fpr MUST reuse calibrate to pick tau (DRY / invariant #4).
    benign, attacked = _separated_per_rollout(seed=5)
    target = 0.05
    (p,) = tpr_at_fpr(benign, attacked, fpr_targets=(target,))
    expected = calibrate(benign, target_per_rollout_fpr=target)
    assert p.tau == expected.tau


def test_operating_point_cis_within_unit_interval():
    benign, attacked = _separated_per_rollout()
    points = tpr_at_fpr(benign, attacked, fpr_targets=(0.01, 0.05))
    for p in points:
        for lo, hi in (p.tpr_ci, p.realised_fpr_ci):
            assert 0.0 <= lo <= hi <= 1.0


def test_operating_point_records_split_sizes():
    benign, attacked = _separated_per_rollout(n_benign=120, n_attacked=90)
    (p,) = tpr_at_fpr(benign, attacked, fpr_targets=(0.05,))
    assert p.n_benign == 120
    assert p.n_attacked == 90


def test_tpr_at_fpr_accepts_scalar_per_rollout_scores():
    # Each rollout may be given directly as a single per-rollout score value.
    benign = [0.1, 0.2, 0.15, 0.05]
    attacked = [0.8, 0.9, 0.95, 0.7]
    (p,) = tpr_at_fpr(benign, attacked, fpr_targets=(0.05,))
    assert p.tpr == 1.0


def test_tpr_at_fpr_does_not_mutate_input():
    benign, attacked = _separated_per_rollout(seed=9)
    b_snapshot = [list(r) for r in benign]
    a_snapshot = [list(r) for r in attacked]
    tpr_at_fpr(benign, attacked, fpr_targets=(0.05,))
    assert [list(r) for r in benign] == b_snapshot
    assert [list(r) for r in attacked] == a_snapshot


# --------------------------------------------------------------------------- #
# tpr_at_fpr — held-out FPR (invariant #3: set tau on calib, report FPR on a   #
# disjoint held-out split, never on the calibration rollouts)                  #
# --------------------------------------------------------------------------- #


def _calib_low_eval_high():
    """A calib set with low benign scores and a *disjoint* held-out (eval) set
    whose benign scores all sit above any tau calibrated on calib.

    This is the regime invariant #3 exists to expose: tau looks fine on the
    calibration rollouts (~0 false aborts) but the held-out benign distribution
    has shifted, so the *honest* held-out false-abort rate is far higher.
    """
    benign_calib = [0.10 + 0.0005 * i for i in range(100)]  # all < 0.15
    benign_eval = [0.5] * 100  # held-out benign, all above any calib tau
    attacked = [0.9] * 100
    return benign_calib, benign_eval, attacked


def test_realised_fpr_is_measured_on_heldout_benign_eval():
    # With a held-out benign set provided, realised_fpr is its fire-rate at tau
    # (here 1.0), NOT the in-sample calibration fire-rate (here ~0).
    benign_calib, benign_eval, attacked = _calib_low_eval_high()
    (p,) = tpr_at_fpr(
        benign_calib, attacked, benign_eval_scores=benign_eval, fpr_targets=(0.05,)
    )
    assert p.realised_fpr == pytest.approx(1.0)
    assert p.n_benign == 100  # n_benign now describes the held-out eval split


def test_calib_fpr_diagnostic_stays_conservative_below_target():
    # The in-sample calibration FPR is retained as a diagnostic and must remain
    # conservative (<= target), independent of the held-out realised_fpr.
    benign_calib, benign_eval, attacked = _calib_low_eval_high()
    for target in (0.01, 0.05):
        (p,) = tpr_at_fpr(
            benign_calib, attacked, benign_eval_scores=benign_eval, fpr_targets=(target,)
        )
        assert p.calib_fpr <= target + 1e-12
        # The held-out FPR is the honest, much larger number — they must differ.
        assert p.realised_fpr > p.calib_fpr


def test_tau_unchanged_by_benign_eval():
    # benign_eval must not influence tau (tau is calibrated on benign_calib only).
    benign_calib, benign_eval, attacked = _calib_low_eval_high()
    (with_eval,) = tpr_at_fpr(
        benign_calib, attacked, benign_eval_scores=benign_eval, fpr_targets=(0.05,)
    )
    (without_eval,) = tpr_at_fpr(benign_calib, attacked, fpr_targets=(0.05,))
    assert with_eval.tau == without_eval.tau
    # calibrate takes rollouts (sequences of per-step scores); wrap each scalar.
    expected = calibrate([[v] for v in benign_calib], target_per_rollout_fpr=0.05)
    assert with_eval.tau == expected.tau


def test_realised_fpr_falls_back_to_calib_when_no_eval_given():
    # Backward-compatible default: with no held-out set, realised_fpr == calib_fpr.
    benign, attacked = _separated_per_rollout(seed=11)
    (p,) = tpr_at_fpr(benign, attacked, fpr_targets=(0.05,))
    assert p.realised_fpr == p.calib_fpr


def test_heldout_fpr_ci_uses_eval_n():
    # The realised_fpr CI must be computed on the held-out eval count, so it is
    # the CI of the honest held-out rate (the one a 1% claim's power depends on).
    benign_calib, benign_eval, attacked = _calib_low_eval_high()
    (p,) = tpr_at_fpr(
        benign_calib, attacked, benign_eval_scores=benign_eval, fpr_targets=(0.05,)
    )
    expected = proportion_ci(100, 100, method="wilson")  # 100/100 fired held-out
    assert p.realised_fpr_ci == expected


# --------------------------------------------------------------------------- #
# benign_degradation / abort_rate / detection_latency_summary                 #
# --------------------------------------------------------------------------- #


def test_benign_degradation_is_drop_in_success():
    assert benign_degradation(0.80, 0.74) == pytest.approx(0.06)


def test_benign_degradation_zero_when_unchanged():
    assert benign_degradation(0.9, 0.9) == 0.0


def test_abort_rate_basic():
    assert abort_rate(3, 12) == pytest.approx(0.25)


def test_abort_rate_rejects_zero_total():
    with pytest.raises(ValueError):
        abort_rate(0, 0)


def test_detection_latency_summary_filters_none():
    latencies = [0, 2, 5, None, None, 3]
    summary = detection_latency_summary(latencies)
    assert summary["count"] == 4  # four fired
    assert summary["never_fired"] == 2
    assert summary["min"] == 0
    assert summary["max"] == 5
    assert summary["mean"] == pytest.approx((0 + 2 + 5 + 3) / 4)
    assert summary["median"] == pytest.approx(2.5)


def test_detection_latency_summary_all_none():
    summary = detection_latency_summary([None, None, None])
    assert summary["count"] == 0
    assert summary["never_fired"] == 3
    assert summary["mean"] is None
    assert summary["median"] is None
    assert summary["min"] is None
    assert summary["max"] is None

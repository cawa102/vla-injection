---
source_file: "tests/evasion_tax/detector/test_calibrate.py"
type: "code"
community: "Detector Calibration"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Detector_Calibration
---

# test_calibrate.py

## Connections
- [[A synthetic benign calibration set list of rollouts of per-step floats in 0,1]] - `defined_in` [EXTRACTED]
- [[Empirical per-rollout fire-rate a rollout fires iff ANY step value  tau.]] - `defined_in` [EXTRACTED]
- [[Tests for FP-calibration of the detector threshold (Task 6).  Covers invariants]] - `rationale_for` [EXTRACTED]
- [[Threshold_1]] - `references` [EXTRACTED]
- [[benign_rollouts()]] - `contains` [EXTRACTED]
- [[calibrate()]] - `references` [EXTRACTED]
- [[per_rollout_fire_rate()]] - `contains` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[rollout_fires()]] - `references` [EXTRACTED]
- [[test_calibrate_accepts_score_objects()]] - `contains` [EXTRACTED]
- [[test_calibrate_does_not_mutate_input()]] - `contains` [EXTRACTED]
- [[test_calibrate_rejects_empty_calibration_set()]] - `contains` [EXTRACTED]
- [[test_calibrate_rejects_target_above_one()]] - `contains` [EXTRACTED]
- [[test_calibrate_rejects_target_below_zero()]] - `contains` [EXTRACTED]
- [[test_calibrate_rejects_unknown_aggregate()]] - `contains` [EXTRACTED]
- [[test_calibrate_returns_threshold_record()]] - `contains` [EXTRACTED]
- [[test_calibrate_works_on_arbitrary_made_up_scores()]] - `contains` [EXTRACTED]
- [[test_calibrated_per_rollout_fire_rate_at_or_below_target()]] - `contains` [EXTRACTED]
- [[test_per_rollout_and_per_window_give_different_tau()]] - `contains` [EXTRACTED]
- [[test_smaller_target_gives_higher_tau_and_fewer_fires()]] - `contains` [EXTRACTED]
- [[test_target_one_means_almost_everything_fires()]] - `contains` [EXTRACTED]
- [[test_target_zero_means_nothing_fires_on_calibration_set()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Detector_Calibration

## 📄 Source

`tests/evasion_tax/detector/test_calibrate.py`

```python
"""Tests for FP-calibration of the detector threshold (Task 6).

Covers invariants #3 (per-rollout FPR primary) and #4 (the *same* ``calibrate``
serves baselines). The load-bearing guarantee: with strict ``>`` firing, the
realised per-rollout fire-rate on the calibration set is **<= target** and close
to it; a smaller target yields a higher (stricter) tau; per-rollout vs per-window
aggregation give different taus; and the edge cases p=0 / p=1 behave sensibly.
"""

import numpy as np
import pytest

from evasion_tax.detector.calibrate import Threshold, calibrate
from evasion_tax.detector.decide import rollout_fires
from evasion_tax.records import Score


def benign_rollouts(n_rollouts, steps_per, *, seed=0):
    """A synthetic benign calibration set: list of rollouts of per-step floats in [0,1]."""
    rng = np.random.default_rng(seed)
    return [list(rng.uniform(0.0, 0.6, size=steps_per)) for _ in range(n_rollouts)]


def per_rollout_fire_rate(rollouts, tau):
    """Empirical per-rollout fire-rate: a rollout fires iff ANY step value > tau."""
    fired = 0
    for r in rollouts:
        s = [Score(value=v, window_end=i) for i, v in enumerate(r)]
        if rollout_fires(s, threshold=tau).hold:
            fired += 1
    return fired / len(rollouts)


# --------------------------------------------------------------------------- #
# Return type                                                                  #
# --------------------------------------------------------------------------- #


def test_calibrate_returns_threshold_record():
    data = benign_rollouts(50, 10)
    thr = calibrate(data, target_per_rollout_fpr=0.05)
    assert isinstance(thr, Threshold)
    assert thr.aggregate == "per_rollout"
    assert thr.target_fpr == 0.05
    assert 0.0 <= thr.tau


# --------------------------------------------------------------------------- #
# Per-rollout FPR <= target and close to it (invariant #3)                     #
# --------------------------------------------------------------------------- #


def test_calibrated_per_rollout_fire_rate_at_or_below_target():
    data = benign_rollouts(1000, 20, seed=1)
    target = 0.05
    thr = calibrate(data, target_per_rollout_fpr=target)
    realised = per_rollout_fire_rate(data, thr.tau)
    assert realised <= target  # conservative: never exceeds the budget
    assert realised >= target - 0.02  # ... and close to it (not absurdly strict)


def test_smaller_target_gives_higher_tau_and_fewer_fires():
    data = benign_rollouts(1000, 20, seed=2)
    thr5 = calibrate(data, target_per_rollout_fpr=0.05)
    thr1 = calibrate(data, target_per_rollout_fpr=0.01)
    assert thr1.tau >= thr5.tau
    assert per_rollout_fire_rate(data, thr1.tau) <= per_rollout_fire_rate(data, thr5.tau)
    assert per_rollout_fire_rate(data, thr1.tau) <= 0.01


# --------------------------------------------------------------------------- #
# per_rollout vs per_window differ                                            #
# --------------------------------------------------------------------------- #


def test_per_rollout_and_per_window_give_different_tau():
    data = benign_rollouts(500, 30, seed=3)
    thr_roll = calibrate(data, target_per_rollout_fpr=0.05, aggregate="per_rollout")
    thr_win = calibrate(data, target_per_rollout_fpr=0.05, aggregate="per_window")
    assert thr_roll.aggregate == "per_rollout"
    assert thr_win.aggregate == "per_window"
    # Per-window pools every step, so its quantile sits below per-rollout maxima.
    assert thr_win.tau != thr_roll.tau
    assert thr_win.tau < thr_roll.tau


# --------------------------------------------------------------------------- #
# Accepts Score objects as well as raw floats                                 #
# --------------------------------------------------------------------------- #


def test_calibrate_accepts_score_objects():
    floats = benign_rollouts(100, 8, seed=4)
    as_scores = [
        [Score(value=v, window_end=i) for i, v in enumerate(r)] for r in floats
    ]
    thr_f = calibrate(floats, target_per_rollout_fpr=0.05)
    thr_s = calibrate(as_scores, target_per_rollout_fpr=0.05)
    assert thr_f.tau == thr_s.tau


# --------------------------------------------------------------------------- #
# Edge cases: p = 0 and p = 1                                                 #
# --------------------------------------------------------------------------- #


def test_target_zero_means_nothing_fires_on_calibration_set():
    data = benign_rollouts(200, 12, seed=5)
    thr = calibrate(data, target_per_rollout_fpr=0.0)
    assert per_rollout_fire_rate(data, thr.tau) == 0.0


def test_target_one_means_almost_everything_fires():
    data = benign_rollouts(200, 12, seed=6)
    thr = calibrate(data, target_per_rollout_fpr=1.0)
    assert per_rollout_fire_rate(data, thr.tau) >= 0.99


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_calibrate_rejects_target_below_zero():
    with pytest.raises(ValueError):
        calibrate(benign_rollouts(10, 5), target_per_rollout_fpr=-0.01)


def test_calibrate_rejects_target_above_one():
    with pytest.raises(ValueError):
        calibrate(benign_rollouts(10, 5), target_per_rollout_fpr=1.5)


def test_calibrate_rejects_unknown_aggregate():
    with pytest.raises(ValueError):
        calibrate(benign_rollouts(10, 5), target_per_rollout_fpr=0.05, aggregate="per_galaxy")


def test_calibrate_rejects_empty_calibration_set():
    with pytest.raises(ValueError):
        calibrate([], target_per_rollout_fpr=0.05)


def test_calibrate_does_not_mutate_input():
    data = benign_rollouts(20, 6, seed=7)
    snapshot = [list(r) for r in data]
    calibrate(data, target_per_rollout_fpr=0.05)
    assert data == snapshot


# --------------------------------------------------------------------------- #
# Reusable on arbitrary score arrays (invariant #4 — baselines reuse it)       #
# --------------------------------------------------------------------------- #


def test_calibrate_works_on_arbitrary_made_up_scores():
    # Hand-made scores unrelated to the metric — baselines must calibrate the same way.
    data = [
        [0.10, 0.20, 0.30],
        [0.05, 0.15, 0.25],
        [0.40, 0.50, 0.60],
        [0.00, 0.10, 0.20],
    ]
    thr = calibrate(data, target_per_rollout_fpr=0.25)
    # One in four rollouts (the 0.60-max one) may fire at most.
    assert per_rollout_fire_rate(data, thr.tau) <= 0.25
    assert isinstance(thr.tau, float)
```


---
source_file: "tests/evasion_tax/baselines/test_perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/L0_Perplexity_Filter
---

# test_perplexity.py

## Connections
- [[MockPerplexityScorer]] - `references` [EXTRACTED]
- [[PerplexityFilter]] - `references` [EXTRACTED]
- [[RealPerplexityScorer]] - `references` [EXTRACTED]
- [[Tests for the perplexity  text-only filter baseline (Task 8).  The published Ro]] - `rationale_for` [EXTRACTED]
- [[calibrate()]] - `references` [EXTRACTED]
- [[make_rollout()]] - `contains` [EXTRACTED]
- [[records.py]] - `imports_from` [EXTRACTED]
- [[rollout_fires()]] - `references` [EXTRACTED]
- [[test_calibrates_identically_through_shared_calibrate()]] - `contains` [EXTRACTED]
- [[test_empty_rollout_raises()]] - `contains` [EXTRACTED]
- [[test_higher_perplexity_maps_to_higher_score()]] - `contains` [EXTRACTED]
- [[test_mock_heuristic_empty_instruction_is_minimal_perplexity()]] - `contains` [EXTRACTED]
- [[test_mock_heuristic_rates_symbol_heavy_text_more_perplexing_than_english()]] - `contains` [EXTRACTED]
- [[test_perplexity_one_maps_to_zero_and_below_one_clamps()]] - `contains` [EXTRACTED]
- [[test_real_backend_is_gpu_stub()]] - `contains` [EXTRACTED]
- [[test_real_backend_plugs_into_filter_but_errors_when_used()]] - `contains` [EXTRACTED]
- [[test_score_rollout_returns_single_unit_interval_score()]] - `contains` [EXTRACTED]
- [[test_scores_the_operational_instruction_not_the_trusted_goal()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/L0_Perplexity_Filter

## 📄 Source

`tests/evasion_tax/baselines/test_perplexity.py`

```python
"""Tests for the perplexity / text-only filter baseline (Task 8).

The published RoboGCG defences include a text-only perplexity filter; we
reproduce it as a *fair* baseline (plan invariant #4): it goes through the
**same** ``calibrate`` as every other detector. These tests pin the interface
and the calibration path, using a deterministic ``MockPerplexityScorer`` (no LM
on the 8 GB host); the real LM backend is a GPU-only stub.

Design contract:

* the filter scores the **operational instruction** (the channel the attacker
  tampers — the GCG suffix lands there), never ``trusted_goal``;
* perplexity ``ppl >= 1`` maps **monotonically** to ``s = 1 - 1/ppl`` in
  ``[0, 1)`` so calibrating on scores is equivalent to thresholding raw
  perplexity (no handicap);
* one ``Score`` per rollout (a text filter decides up-front), ``window_end = 0``.
"""

import pytest

from evasion_tax.baselines.perplexity import (
    MockPerplexityScorer,
    PerplexityFilter,
    RealPerplexityScorer,
)
from evasion_tax.detector.calibrate import calibrate
from evasion_tax.detector.decide import rollout_fires
from evasion_tax.records import Rollout, RolloutStep, Score


def make_rollout(instruction, *, trusted_goal="reach the cube", n_steps=5):
    steps = tuple(
        RolloutStep(
            run_id="test",
            seed=0,
            git_commit=None,
            suite="libero_spatial",
            task_id="t0",
            step=i,
            observation_ref=f"obs/{i}",
            action=(0.0,) * 7,
            privileged_state={},
            instruction=instruction,
            trusted_goal=trusted_goal,
            attacked=False,
            suffix_ref=None,
        )
        for i in range(n_steps)
    )
    return Rollout(steps=steps)


# --------------------------------------------------------------------------- #
# Interface + range                                                            #
# --------------------------------------------------------------------------- #


def test_score_rollout_returns_single_unit_interval_score():
    flt = PerplexityFilter(MockPerplexityScorer(table={"pick up the bowl": 12.0}))
    scores = flt.score_rollout(make_rollout("pick up the bowl"))
    assert len(scores) == 1
    (score,) = scores
    assert isinstance(score, Score)
    assert 0.0 <= score.value <= 1.0
    assert score.window_end == 0


def test_higher_perplexity_maps_to_higher_score():
    flt = PerplexityFilter(MockPerplexityScorer(table={"low": 2.0, "high": 100.0}))
    (low,) = flt.score_rollout(make_rollout("low"))
    (high,) = flt.score_rollout(make_rollout("high"))
    assert high.value > low.value


def test_perplexity_one_maps_to_zero_and_below_one_clamps():
    flt = PerplexityFilter(MockPerplexityScorer(table={"a": 1.0, "b": 0.5}))
    (a,) = flt.score_rollout(make_rollout("a"))
    (b,) = flt.score_rollout(make_rollout("b"))
    assert a.value == 0.0
    assert b.value == 0.0  # degenerate ppl < 1 clamps, never negative


def test_empty_rollout_raises():
    flt = PerplexityFilter(MockPerplexityScorer())
    with pytest.raises(ValueError):
        flt.score_rollout(Rollout(steps=()))


# --------------------------------------------------------------------------- #
# Reads the operational instruction, not the trusted goal                      #
# --------------------------------------------------------------------------- #


def test_scores_the_operational_instruction_not_the_trusted_goal():
    flt = PerplexityFilter(MockPerplexityScorer(table={"clean": 2.0, "gcg!! }}": 80.0}))
    # Same trusted goal; only the (possibly tampered) instruction differs.
    (clean,) = flt.score_rollout(make_rollout("clean", trusted_goal="reach the cube"))
    (attacked,) = flt.score_rollout(make_rollout("gcg!! }}", trusted_goal="reach the cube"))
    assert attacked.value > clean.value


# --------------------------------------------------------------------------- #
# Calibrates identically through the shared calibrate                          #
# --------------------------------------------------------------------------- #


def test_calibrates_identically_through_shared_calibrate():
    table = {f"benign instruction {i}": 1.5 + 0.01 * i for i in range(50)}
    table["gcg adversarial suffix"] = 500.0
    flt = PerplexityFilter(MockPerplexityScorer(table=table))

    benign_calib = [flt.score_rollout(make_rollout(k)) for k in table if k.startswith("benign")]
    thr = calibrate(benign_calib, target_per_rollout_fpr=0.05)

    attacked = flt.score_rollout(make_rollout("gcg adversarial suffix"))
    assert rollout_fires(attacked, thr.tau).hold

    fired = sum(rollout_fires(c, thr.tau).hold for c in benign_calib)
    assert fired / len(benign_calib) <= 0.05


# --------------------------------------------------------------------------- #
# Mock heuristic default (self-contained separation, no table)                 #
# --------------------------------------------------------------------------- #


def test_mock_heuristic_rates_symbol_heavy_text_more_perplexing_than_english():
    scorer = MockPerplexityScorer()  # no table → crude symbol-density surrogate
    english = scorer.score_perplexity("pick up the red bowl and place it on the plate")
    gibberish = scorer.score_perplexity("}}!@# ]]<< == ;;~~ /\\ *** %%%")
    assert gibberish > english
    assert english >= 1.0  # perplexity is always >= 1


def test_mock_heuristic_empty_instruction_is_minimal_perplexity():
    assert MockPerplexityScorer().score_perplexity("") == 1.0


# --------------------------------------------------------------------------- #
# Real backend is a GPU-only stub                                             #
# --------------------------------------------------------------------------- #


def test_real_backend_is_gpu_stub():
    scorer = RealPerplexityScorer()
    with pytest.raises(NotImplementedError) as exc:
        scorer.score_perplexity("anything")
    assert "GPU" in str(exc.value)


def test_real_backend_plugs_into_filter_but_errors_when_used():
    flt = PerplexityFilter(RealPerplexityScorer())
    with pytest.raises(NotImplementedError):
        flt.score_rollout(make_rollout("pick up the bowl"))
```


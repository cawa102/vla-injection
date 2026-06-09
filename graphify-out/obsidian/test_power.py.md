---
source_file: "tests/evasion_tax/eval/test_power.py"
type: "code"
community: "Eval Harness & Power"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Eval_Harness__Power
---

# test_power.py

## Connections
- [[A minimal OperatingPoint carrying only the fields the rule reads.]] - `defined_in` [EXTRACTED]
- [[DetectorConfig]] - `references` [EXTRACTED]
- [[OperatingPoint]] - `references` [EXTRACTED]
- [[OperatingPoint_2]] - `defined_in` [EXTRACTED]
- [[PowerStatus]] - `references` [EXTRACTED]
- [[RULE_OF_THREE_EVENTS]] - `references` [EXTRACTED]
- [[Tests for the operating-point power  sample-size rule (Codex review 2 3).  Th]] - `rationale_for` [EXTRACTED]
- [[_op()]] - `contains` [EXTRACTED]
- [[annotate_operating_points()]] - `references` [EXTRACTED]
- [[classify_power()]] - `references` [EXTRACTED]
- [[required_benign_n()]] - `references` [EXTRACTED]
- [[test_annotate_operating_points_classifies_each_point()]] - `contains` [EXTRACTED]
- [[test_annotate_uses_held_out_n_benign_not_calib()]] - `contains` [EXTRACTED]
- [[test_detector_config_accepts_primary_fpr_in_targets()]] - `contains` [EXTRACTED]
- [[test_detector_config_defaults_primary_fpr_to_five_percent()]] - `contains` [EXTRACTED]
- [[test_detector_config_rejects_out_of_range_primary_fpr()]] - `contains` [EXTRACTED]
- [[test_detector_config_rejects_primary_fpr_not_in_targets()]] - `contains` [EXTRACTED]
- [[test_power_status_is_immutable()]] - `contains` [EXTRACTED]
- [[test_powered_boundary_is_inclusive_at_required_n()]] - `contains` [EXTRACTED]
- [[test_primary_flag_tracks_primary_fpr()]] - `contains` [EXTRACTED]
- [[test_primary_point_is_powered_at_modest_n()]] - `contains` [EXTRACTED]
- [[test_required_benign_n_honours_min_events_override()]] - `contains` [EXTRACTED]
- [[test_required_benign_n_matches_playbook_floors()]] - `contains` [EXTRACTED]
- [[test_required_benign_n_rejects_out_of_range_fpr()]] - `contains` [EXTRACTED]
- [[test_required_benign_n_rounds_up()]] - `contains` [EXTRACTED]
- [[test_rule_of_three_constant_is_three()]] - `contains` [EXTRACTED]
- [[test_underpowered_tight_point_is_flagged_not_silent()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Eval_Harness__Power

## 📄 Source

`tests/evasion_tax/eval/test_power.py`

```python
"""Tests for the operating-point power / sample-size rule (Codex review #2 #3).

The rule (playbook §5): 5% is the *primary* operating point; a tighter FPR is
only adequately powered if the held-out benign N can estimate it. The "~300 for
1%" floor is the **rule of three** — to bound a target FPR ``p`` away from zero a
``0/N`` benign result must give ``3/N <= p``, i.e. ``N >= 3/p`` (300 at 1%, 60 at
5%). These tests pin the arithmetic, the powered/underpowered boundary, and that
an underpowered tight point can never be silently treated as powered.
"""

import dataclasses
import math

import pytest
from pydantic import ValidationError

from evasion_tax.config.schema import DetectorConfig
from evasion_tax.eval.metrics import OperatingPoint
from evasion_tax.eval.power import (
    RULE_OF_THREE_EVENTS,
    PowerStatus,
    annotate_operating_points,
    classify_power,
    required_benign_n,
)

# --------------------------------------------------------------------------- #
# required_benign_n — the rule-of-three floor                                 #
# --------------------------------------------------------------------------- #


def test_required_benign_n_matches_playbook_floors():
    assert required_benign_n(0.01) == 300  # ceil(3 / 0.01)
    assert required_benign_n(0.05) == 60  # ceil(3 / 0.05)


def test_required_benign_n_rounds_up():
    # 3 / 0.04 = 75 exactly; 3 / 0.03 = 100; 3 / 0.07 = 42.857 -> 43.
    assert required_benign_n(0.04) == 75
    assert required_benign_n(0.07) == 43


def test_required_benign_n_honours_min_events_override():
    assert required_benign_n(0.01, min_events=5.0) == 500


def test_rule_of_three_constant_is_three():
    assert RULE_OF_THREE_EVENTS == 3.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_required_benign_n_rejects_out_of_range_fpr(bad):
    with pytest.raises(ValueError):
        required_benign_n(bad)


# --------------------------------------------------------------------------- #
# classify_power — powered / primary flags                                    #
# --------------------------------------------------------------------------- #


def test_powered_boundary_is_inclusive_at_required_n():
    under = classify_power(0.01, n_benign=299, primary_fpr=0.05)
    exact = classify_power(0.01, n_benign=300, primary_fpr=0.05)
    assert under.is_powered is False
    assert exact.is_powered is True
    assert under.required_n == exact.required_n == 300


def test_primary_point_is_powered_at_modest_n():
    status = classify_power(0.05, n_benign=80, primary_fpr=0.05)
    assert status.is_primary is True
    assert status.is_powered is True  # 80 >= 60


def test_primary_flag_tracks_primary_fpr():
    assert classify_power(0.05, n_benign=100, primary_fpr=0.05).is_primary is True
    assert classify_power(0.01, n_benign=100, primary_fpr=0.05).is_primary is False


def test_underpowered_tight_point_is_flagged_not_silent():
    # The whole point of #3: a 1% claim on a small benign set is exploratory.
    status = classify_power(0.01, n_benign=90, primary_fpr=0.05)
    assert status.is_powered is False
    assert status.is_primary is False
    assert status.fpr_target == 0.01
    assert status.n_benign == 90


def test_power_status_is_immutable():
    status = classify_power(0.05, n_benign=80, primary_fpr=0.05)
    assert isinstance(status, PowerStatus)
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.is_powered = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# annotate_operating_points — bridges eval metrics to the rule                #
# --------------------------------------------------------------------------- #


def _op(fpr_target: float, n_benign: int) -> OperatingPoint:
    """A minimal OperatingPoint carrying only the fields the rule reads."""
    return OperatingPoint(
        fpr_target=fpr_target,
        tau=0.5,
        tpr=0.9,
        tpr_ci=(0.8, 0.95),
        realised_fpr=fpr_target,
        realised_fpr_ci=(0.0, 2 * fpr_target),
        n_benign=n_benign,
        n_attacked=100,
        calib_fpr=fpr_target,
        calib_fpr_ci=(0.0, 2 * fpr_target),
        n_benign_calib=n_benign,
    )


def test_annotate_operating_points_classifies_each_point():
    points = [_op(0.01, n_benign=90), _op(0.05, n_benign=90)]
    statuses = annotate_operating_points(points, primary_fpr=0.05)
    assert [s.fpr_target for s in statuses] == [0.01, 0.05]
    assert statuses[0].is_powered is False  # 90 < 300
    assert statuses[1].is_powered is True  # 90 >= 60
    assert statuses[1].is_primary is True


def test_annotate_uses_held_out_n_benign_not_calib():
    # n_benign is the held-out split (invariant #3) — the N power depends on.
    point = _op(0.01, n_benign=300)
    [status] = annotate_operating_points([point], primary_fpr=0.05)
    assert status.n_benign == 300
    assert status.is_powered is True


# --------------------------------------------------------------------------- #
# DetectorConfig.primary_fpr — pinned + validated                             #
# --------------------------------------------------------------------------- #


def test_detector_config_defaults_primary_fpr_to_five_percent():
    cfg = DetectorConfig(fpr_targets=[0.01, 0.05])
    assert math.isclose(cfg.primary_fpr, 0.05)


def test_detector_config_accepts_primary_fpr_in_targets():
    cfg = DetectorConfig(fpr_targets=[0.01, 0.05], primary_fpr=0.01)
    assert math.isclose(cfg.primary_fpr, 0.01)


def test_detector_config_rejects_primary_fpr_not_in_targets():
    with pytest.raises(ValidationError):
        DetectorConfig(fpr_targets=[0.01, 0.05], primary_fpr=0.02)


def test_detector_config_rejects_out_of_range_primary_fpr():
    with pytest.raises(ValidationError):
        DetectorConfig(fpr_targets=[0.01, 0.05], primary_fpr=1.5)
```


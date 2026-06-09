---
source_file: "src/evasion_tax/eval/power.py"
type: "code"
community: "Eval Harness & Power"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Eval_Harness__Power
---

# power.py

## Connections
- [[Attach a class`PowerStatus` to each operating point, in order.      The report]] - `defined_in` [EXTRACTED]
- [[Classify one operating point as powered  primary against the rule.      Args]] - `defined_in` [EXTRACTED]
- [[Minimum held-out benign N to support a per-rollout FPR claim of ``fpr_target``.]] - `defined_in` [EXTRACTED]
- [[Operating-point power  sample-size rule (Codex review 2 3).  The detector rep]] - `rationale_for` [EXTRACTED]
- [[OperatingPoint_1]] - `defined_in` [EXTRACTED]
- [[Power verdict for one operating point.      Attributes         fpr_target The]] - `defined_in` [EXTRACTED]
- [[PowerStatus]] - `contains` [EXTRACTED]
- [[PowerStatus_1]] - `defined_in` [EXTRACTED]
- [[RULE_OF_THREE_EVENTS]] - `defined_in` [EXTRACTED]
- [[annotate_operating_points]] - `defined_in` [EXTRACTED]
- [[annotate_operating_points()]] - `contains` [EXTRACTED]
- [[classify_power]] - `defined_in` [EXTRACTED]
- [[classify_power()]] - `contains` [EXTRACTED]
- [[required_benign_n (rule of three)]] - `defined_in` [EXTRACTED]
- [[required_benign_n()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Eval_Harness__Power

## 📄 Source

`src/evasion_tax/eval/power.py`

```python
"""Operating-point power / sample-size rule (Codex review #2 #3).

The detector reports per-rollout TPR at fixed benign false-abort budgets. A
*tight* budget (1%) is only estimable if the **held-out** benign split is large
enough — with 30–90 rollouts a 1% quantile is fiction. The floor is the **rule of
three**: a ``0/N`` benign result bounds the FPR at ``3/N``, so to claim a target
``p`` away from zero we need ``3/N <= p``, i.e. ``N >= 3/p`` (300 at 1%, 60 at 5%;
playbook §5). Hence **5% is the primary operating point** and a tighter point is
**exploratory unless its held-out benign N clears the floor**.

This module is pure arithmetic over already-computed
:class:`~evasion_tax.eval.metrics.OperatingPoint`s — it reads no attack output (invariant
#2 untouched). The rule-of-three numerator is a fixed statistical rule (a module
constant + kwarg), not a per-run knob; the *required N* is therefore derived and
logged, which is more defensible than a hand-pinned number. The pinned per-run
choice is only **which** FPR is primary (``DetectorConfig.primary_fpr``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from evasion_tax.eval.metrics import OperatingPoint

# Rule-of-three numerator: a 0/N benign result bounds FPR <= RULE_OF_THREE_EVENTS/N.
RULE_OF_THREE_EVENTS = 3.0


def required_benign_n(
    fpr_target: float, *, min_events: float = RULE_OF_THREE_EVENTS
) -> int:
    """Minimum held-out benign N to support a per-rollout FPR claim of ``fpr_target``.

    The rule-of-three floor ``ceil(min_events / fpr_target)`` — 300 at 1%, 60 at
    5% with the default ``min_events=3``.

    Args:
        fpr_target: The target per-rollout benign false-abort rate, in ``(0, 1)``.
        min_events: Rule-of-three numerator (default 3); the expected benign
            false-aborts the held-out set must be able to resolve.

    Returns:
        The smallest integer ``N`` with ``min_events / N <= fpr_target``.

    Raises:
        ValueError: If ``fpr_target`` is not in the open interval ``(0, 1)``.
    """
    if not (0.0 < fpr_target < 1.0):
        raise ValueError(f"fpr_target must be in (0, 1), got {fpr_target}")
    return math.ceil(min_events / fpr_target)


@dataclass(frozen=True)
class PowerStatus:
    """Power verdict for one operating point.

    Attributes:
        fpr_target: The point's target per-rollout FPR.
        n_benign: The **held-out** benign N the realised FPR was measured on
            (invariant #3) — the N the claim's power depends on.
        required_n: The rule-of-three floor for ``fpr_target``.
        is_powered: ``n_benign >= required_n`` — the point is adequately powered.
        is_primary: This is the pre-registered primary operating point.
    """

    fpr_target: float
    n_benign: int
    required_n: int
    is_powered: bool
    is_primary: bool


def classify_power(
    fpr_target: float,
    n_benign: int,
    *,
    primary_fpr: float,
    min_events: float = RULE_OF_THREE_EVENTS,
) -> PowerStatus:
    """Classify one operating point as powered / primary against the rule.

    Args:
        fpr_target: The point's target per-rollout FPR (in ``(0, 1)``).
        n_benign: Held-out benign N the realised FPR was measured on.
        primary_fpr: The pre-registered primary operating point.
        min_events: Rule-of-three numerator (default 3).

    Returns:
        A :class:`PowerStatus`.
    """
    required = required_benign_n(fpr_target, min_events=min_events)
    return PowerStatus(
        fpr_target=fpr_target,
        n_benign=n_benign,
        required_n=required,
        is_powered=n_benign >= required,
        is_primary=math.isclose(fpr_target, primary_fpr),
    )


def annotate_operating_points(
    points: Sequence[OperatingPoint],
    *,
    primary_fpr: float,
    min_events: float = RULE_OF_THREE_EVENTS,
) -> list[PowerStatus]:
    """Attach a :class:`PowerStatus` to each operating point, in order.

    The reporting gate: an underpowered tight point is flagged
    (``is_powered=False``) so it can never be silently reported as the headline.
    ``n_benign`` is read from the point's **held-out** count, not the calibration
    count (invariant #3).
    """
    return [
        classify_power(
            p.fpr_target, p.n_benign, primary_fpr=primary_fpr, min_events=min_events
        )
        for p in points
    ]
```


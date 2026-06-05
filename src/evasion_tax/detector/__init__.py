"""FP-calibrated detector (Task 6).

Turns per-step consistency scores into a calibrated hold/allow decision at a
chosen benign false-abort budget. The public surface:

* :class:`Threshold` and :func:`calibrate` — set ``tau`` on a benign calibration
  split so the per-rollout false-abort rate is at or below the target (the shared
  fair-comparison primitive every baseline reuses; plan invariant #4).
* :func:`decide`, :func:`rollout_fires`, :func:`detection_latency` — apply the
  threshold causally (first exceedance only, never looking ahead; invariant #1).
"""

from __future__ import annotations

from evasion_tax.detector.calibrate import Threshold, calibrate
from evasion_tax.detector.decide import decide, detection_latency, rollout_fires

__all__ = [
    "Threshold",
    "calibrate",
    "decide",
    "detection_latency",
    "rollout_fires",
]

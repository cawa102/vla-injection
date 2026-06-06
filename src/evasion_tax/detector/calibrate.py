"""FP-calibration of the detector threshold (Task 6).

Chooses ``tau`` on a benign calibration split so the **per-rollout** false-abort
rate is at or below a chosen budget (plan invariant #3: per-rollout FPR is
primary; per-window is auxiliary). This is the shared fair-comparison primitive
every baseline reuses unchanged (invariant #4), so it operates on plain score
*values* and makes no assumption about how they were produced.

Calibration logic
-----------------
The detector fires on strict ``value > tau``. For ``aggregate="per_rollout"`` we
reduce each benign rollout to its **maximum** step score (a rollout fires iff its
max exceeds ``tau``), then set ``tau`` to the empirical quantile of those maxima
at ``1 - target_per_rollout_fpr`` using ``method="higher"``. Because ``"higher"``
snaps ``tau`` up onto an actual benign maximum and firing is strict ``>``, that
benign rollout is *excluded*, so the realised fire-rate is **conservative**
(``<= target``). For ``aggregate="per_window"`` the same rule is applied over the
pooled per-step scores (each step is a unit).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evasion_tax.records import ScoreLike, score_value

_AGGREGATES = ("per_rollout", "per_window")

RolloutScores = Sequence[ScoreLike]


@dataclass(frozen=True)
class Threshold:
    """An immutable calibrated decision threshold.

    Attributes:
        tau: The threshold; the detector fires when a score strictly exceeds it.
        aggregate: Calibration unit — ``"per_rollout"`` (primary) or
            ``"per_window"`` (auxiliary).
        target_fpr: The benign false-abort budget ``tau`` was calibrated to.
    """

    tau: float
    aggregate: str
    target_fpr: float


def _rollout_maxima(benign_scores_calib: Sequence[RolloutScores]) -> np.ndarray:
    """One value per rollout: its maximum per-step score."""
    return np.array(
        [max(score_value(s) for s in rollout) for rollout in benign_scores_calib],
        dtype=float,
    )


def _pooled_steps(benign_scores_calib: Sequence[RolloutScores]) -> np.ndarray:
    """All per-step scores pooled across rollouts (per-window unit)."""
    return np.array(
        [score_value(s) for rollout in benign_scores_calib for s in rollout],
        dtype=float,
    )


def calibrate(
    benign_scores_calib: Sequence[RolloutScores],
    *,
    target_per_rollout_fpr: float,
    aggregate: str = "per_rollout",
) -> Threshold:
    """Calibrate ``tau`` to a benign false-abort budget.

    Args:
        benign_scores_calib: A sequence of rollouts; each rollout is a sequence
            of per-step score values (floats in ``[0, 1]``) or ``Score`` objects.
        target_per_rollout_fpr: The benign false-abort budget ``p`` in
            ``[0, 1]``. The realised per-rollout fire-rate on this set is
            conservative (``<= p``).
        aggregate: ``"per_rollout"`` (default, primary) reduces each rollout to
            its max; ``"per_window"`` (auxiliary) pools every per-step score.

    Returns:
        A frozen :class:`Threshold`.

    Raises:
        ValueError: If ``target_per_rollout_fpr`` is outside ``[0, 1]``,
            ``aggregate`` is unknown, or the calibration set is empty.
    """
    p = target_per_rollout_fpr
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"target_per_rollout_fpr must be in [0, 1], got {p}")
    if aggregate not in _AGGREGATES:
        raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}")
    if len(benign_scores_calib) == 0:
        raise ValueError("benign_scores_calib must contain at least one rollout")

    units = (
        _rollout_maxima(benign_scores_calib)
        if aggregate == "per_rollout"
        else _pooled_steps(benign_scores_calib)
    )

    if p == 0.0:
        # Nothing may fire: tau at the global max → strict `>` excludes everything.
        tau = float(units.max())
    elif p == 1.0:
        # Everything fires: tau just below the global min, so strict `>` fires
        # on every unit (all scores are >= the min).
        tau = float(np.nextafter(units.min(), -np.inf))
    else:
        # Conservative quantile: `higher` snaps tau onto a benign unit, which
        # strict `>` then excludes, keeping the realised fire-rate <= p.
        tau = float(np.quantile(units, 1.0 - p, method="higher"))

    return Threshold(tau=tau, aggregate=aggregate, target_fpr=p)

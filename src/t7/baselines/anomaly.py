"""Goal-agnostic action-anomaly baseline (Task 8).

The **mandatory** M2 comparison detector (playbook §5, plan invariant #4): a
fully model-free out-of-distribution score on the action stream that **never
sees the goal**. Reporting metric (A) against it isolates exactly what
*goal-conditioning* buys over mere OOD anomaly detection — if (A) does not beat
this, the goal-conditioned story is not supported.

Score
-----
Under a diagonal-Gaussian model of the **benign** action distribution
(per-dim mean / std, :class:`BenignActionStats`), each action's squared
Mahalanobis distance over the dimensions that actually vary in benign data is
mapped to ``[0, 1]`` by the chi-square CDF with that many degrees of freedom:

    m² = Σ_d ((a_d − μ_d) / σ_d)²   over active dims (σ_d > floor)
    s  = χ²_cdf(m², df = #active dims)              ∈ [0, 1], higher = more anomalous

``χ²_cdf`` is the fraction of benign actions *closer to the mean* than this one,
so benign actions spread across ``[0, 1]`` (no saturation) while genuinely
out-of-distribution actions push ``s → 1``. The score is **parameter-free** (no
attack-tuned constant) and **causal** — each step is scored from its own action
only, so the score at ``t`` never depends on future steps. It reads neither the
trusted goal nor the privileged state (the property that distinguishes it from
metric (A)).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from t7.records import ACTION_DIM, Rollout, Score

_VAR_FLOOR = 1e-9  # dims with std at or below this are treated as inactive (no benign variation)


@dataclass(frozen=True)
class BenignActionStats:
    """Per-dimension benign action statistics (immutable; plan invariant #6).

    Stored as plain float tuples so the record is hashable and cannot be mutated
    in place; convert to arrays at use via :meth:`as_arrays`.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != ACTION_DIM or len(self.std) != ACTION_DIM:
            raise ValueError(
                f"mean and std must each have length {ACTION_DIM}, "
                f"got mean={len(self.mean)}, std={len(self.std)}"
            )

    @classmethod
    def from_rollouts(cls, rollouts: Sequence[Rollout]) -> BenignActionStats:
        """Estimate per-dim mean/std from the pooled actions of benign rollouts.

        Args:
            rollouts: Benign rollouts; their pooled per-step actions form the
                reference distribution.

        Returns:
            A frozen :class:`BenignActionStats`.

        Raises:
            ValueError: If there are no rollouts or no steps across them.
        """
        if len(rollouts) == 0:
            raise ValueError("need at least one rollout to estimate benign stats")
        stacked = np.concatenate([r.actions() for r in rollouts], axis=0)
        if stacked.shape[0] == 0:
            raise ValueError("benign rollouts contain no steps")
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        return cls(mean=tuple(map(float, mean)), std=tuple(map(float, std)))

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` as float ndarrays."""
        return np.asarray(self.mean, dtype=float), np.asarray(self.std, dtype=float)


def goal_agnostic_anomaly_score(
    rollout: Rollout, *, benign_stats: BenignActionStats
) -> list[Score]:
    """One causal, goal-agnostic anomaly score per step (higher = more anomalous).

    Args:
        rollout: The rollout to score; only its actions are read.
        benign_stats: The benign action distribution to score against.

    Returns:
        A list of :class:`Score` (one per step, ``window_end = step index``);
        ``[]`` for an empty rollout.
    """
    actions = rollout.actions()
    if actions.shape[0] == 0:
        return []

    mean, std = benign_stats.as_arrays()
    active = std > _VAR_FLOOR
    df = int(active.sum())
    if df == 0:
        # No benign variation anywhere → nothing to be anomalous against.
        return [Score(value=0.0, window_end=t) for t in range(actions.shape[0])]

    z = (actions[:, active] - mean[active]) / std[active]
    m_sq = np.sum(z * z, axis=1)
    values = chi2.cdf(m_sq, df=df)
    return [
        Score(value=float(np.clip(v, 0.0, 1.0)), window_end=t)
        for t, v in enumerate(values)
    ]

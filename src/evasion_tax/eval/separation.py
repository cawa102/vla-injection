"""Benign-vs-attacked metric-(A) separation table (Task 3) — pure aggregation.

Scores benign + attacked rollouts with the frozen metric (A) against a
(re-pinned) ``SchemaA`` at the coarse operator-goal reference (``trusted_goal``),
reduces each rollout to one score (max over steps), and reuses the eval harness:
τ is calibrated on the benign **calibration** split only, ``realised_fpr`` is the
**held-out** benign false-abort rate, TPR is on the attacked split (invariant #3).
No new statistics — :func:`run_condition_matrix` does the calibration + CIs.
"""

from __future__ import annotations

from collections.abc import Sequence

from evasion_tax.eval.harness import ResultsTable, run_condition_matrix
from evasion_tax.eval.metrics import _per_rollout_score
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA
from evasion_tax.records import Rollout, ScoreLike

# The M1 gate's reference rung: separation must survive at the coarse operator-goal
# reference, not only the clean-instruction ceiling (playbook §7 M1 / invariant #7).
_CONDITION = "coarse_operator_goal"
_DEFAULT_K = 5  # causal window; schema §5 provisional (swept {3,5,8} elsewhere)


def per_rollout_score(scores: Sequence[ScoreLike]) -> float:
    """Frozen per-rollout reduction: max over the per-step scores (DRY with metrics)."""
    return _per_rollout_score(scores)


def _score_rollouts(
    rollouts: Sequence[Rollout], *, schema: SchemaA, k: int, trusted_goal: str
) -> list[float]:
    metric = ConsistencyMetricA(schema=schema, k=k)
    return [per_rollout_score(metric.score_rollout(r, trusted_goal)) for r in rollouts]


def separation_table(
    benign_calib: Sequence[Rollout],
    benign_eval: Sequence[Rollout],
    attacked: Sequence[Rollout],
    *,
    schema: SchemaA,
    fpr: float,
    trusted_goal: str = "",
    k: int = _DEFAULT_K,
) -> ResultsTable:
    """Score the three splits with metric (A) and evaluate separation at ``fpr``.

    Raises:
        ValueError: on an empty split — an empty attacked set yields no separation
            claim, and a held-out FPR needs a non-empty benign eval split (fail
            fast, never a silent / divide-by-zero result).
    """
    if not attacked:
        raise ValueError(
            "separation_table needs >= 1 attacked rollout (empty attacked = no separation claim)"
        )
    if not benign_calib or not benign_eval:
        raise ValueError("separation_table needs non-empty benign calibration and held-out splits")

    splits = {
        "benign_calib": _score_rollouts(
            benign_calib, schema=schema, k=k, trusted_goal=trusted_goal
        ),
        "benign_test": _score_rollouts(
            benign_eval, schema=schema, k=k, trusted_goal=trusted_goal
        ),
        "attacked_test": _score_rollouts(
            attacked, schema=schema, k=k, trusted_goal=trusted_goal
        ),
    }
    return run_condition_matrix(
        {_CONDITION: splits}, fpr_targets=(fpr,), primary_fpr=fpr
    )

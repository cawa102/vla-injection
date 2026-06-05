"""Eval orchestration over a condition matrix (Task 7).

Consumes per-condition score splits (rollout *generation* is on the GPU node) and produces
a :class:`ResultsTable`: one row per condition with its ROC/AUC, TPR@FPR
operating points and latency summary, plus the raw test-score arrays that
``make_figures`` (Task 9) regenerates figures from.

No-leakage guarantee (plan invariant #3): ``tau`` is calibrated on
``benign_calib`` only; AUC and operating points are evaluated on the disjoint
``benign_test`` / ``attacked_test`` splits. Calibration never sees a test array.
In particular the operating points' reported ``realised_fpr`` is the **held-out**
benign false-abort rate on ``benign_test`` (the calibration-split fire-rate is
kept only as the ``calib_fpr`` diagnostic).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evasion_tax.detector.calibrate import ScoreLike
from evasion_tax.eval.metrics import (
    OperatingPoint,
    _per_rollout_scores,
    detection_latency_summary,
    roc_auc,
    tpr_at_fpr,
)

# A condition supplies three splits, each a sequence of per-rollout scores
# (a scalar/Score value, or a sequence of per-step values reduced by max).
RolloutScore = ScoreLike | Sequence[ScoreLike]
ConditionSplits = Mapping[str, Sequence[RolloutScore]]

_DEFAULT_FPR_TARGETS = (0.01, 0.05)


@dataclass(frozen=True)
class ConditionRow:
    """One condition's evaluated result."""

    condition: str
    auc: float
    operating_points: tuple[OperatingPoint, ...]
    latency_summary: dict


@dataclass(frozen=True)
class ResultsTable:
    """Per-condition results plus the raw arrays ``make_figures`` consumes.

    Attributes:
        rows: One :class:`ConditionRow` per condition.
        score_arrays: ``condition -> {"benign_test": [...], "attacked_test":
            [...]}`` per-rollout score arrays (Python lists) for figures.
    """

    rows: tuple[ConditionRow, ...]
    score_arrays: dict


def run_condition_matrix(
    conditions: Mapping[str, ConditionSplits],
    *,
    ci_method: str = "wilson",
) -> ResultsTable:
    """Evaluate every condition: calibrate on calib, score on disjoint test.

    Args:
        conditions: Maps ``condition_name`` to a mapping with keys
            ``"benign_calib"``, ``"benign_test"`` and ``"attacked_test"``, each
            a sequence of per-rollout scores.
        ci_method: CI method passed to :func:`tpr_at_fpr`.

    Returns:
        A :class:`ResultsTable`. ``tau`` is calibrated on ``benign_calib`` only
        (via :func:`tpr_at_fpr`, which reuses ``calibrate``), so no test data
        leaks into calibration.
    """
    rows: list[ConditionRow] = []
    score_arrays: dict = {}

    for name, splits in conditions.items():
        benign_calib = splits["benign_calib"]
        benign_test = splits["benign_test"]
        attacked_test = splits["attacked_test"]

        benign_test_scores = _per_rollout_scores(benign_test)
        attacked_test_scores = _per_rollout_scores(attacked_test)

        _, _, auc = roc_auc(benign_test_scores, attacked_test_scores)

        # Calibrate tau on benign_calib ONLY; report the held-out benign FPR on
        # the disjoint benign_test split (invariant #3) and TPR on attacked_test.
        operating_points = tpr_at_fpr(
            benign_calib,
            attacked_test,
            benign_eval_scores=benign_test,
            fpr_targets=_DEFAULT_FPR_TARGETS,
            ci_method=ci_method,
        )

        rows.append(
            ConditionRow(
                condition=name,
                auc=auc,
                operating_points=tuple(operating_points),
                # No rollouts run here, so no detection latencies to summarise.
                latency_summary=detection_latency_summary([]),
            )
        )
        score_arrays[name] = {
            "benign_test": benign_test_scores.tolist(),
            "attacked_test": attacked_test_scores.tolist(),
        }

    return ResultsTable(rows=tuple(rows), score_arrays=score_arrays)

"""Evaluation harness + statistics (Task 7).

The model-free scientific core that produces the floor result from logged
per-rollout score arrays:

* :func:`proportion_ci`, :func:`roc_auc`, :func:`tpr_at_fpr` and the scalar
  summaries in :mod:`t7.eval.metrics` — binomial CIs, ROC/AUC, and calibrated
  TPR@FPR operating points (tau via the shared ``calibrate``; invariant #4).
* :func:`assert_disjoint` in :mod:`t7.eval.splits` — guards calibration<->test
  leakage (invariant #3).
* :func:`run_condition_matrix` / :class:`ResultsTable` in
  :mod:`t7.eval.harness` — orchestrate per-condition evaluation and retain the
  raw arrays ``make_figures`` consumes.
"""

from __future__ import annotations

from t7.eval.harness import ConditionRow, ResultsTable, run_condition_matrix
from t7.eval.metrics import (
    OperatingPoint,
    abort_rate,
    benign_degradation,
    detection_latency_summary,
    proportion_ci,
    roc_auc,
    tpr_at_fpr,
)
from t7.eval.splits import assert_disjoint

__all__ = [
    "ConditionRow",
    "OperatingPoint",
    "ResultsTable",
    "abort_rate",
    "assert_disjoint",
    "benign_degradation",
    "detection_latency_summary",
    "proportion_ci",
    "roc_auc",
    "run_condition_matrix",
    "tpr_at_fpr",
]

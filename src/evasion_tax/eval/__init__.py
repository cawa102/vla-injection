"""Evaluation harness + statistics (Task 7).

The model-free scientific core that produces the floor result from logged
per-rollout score arrays:

* :func:`proportion_ci`, :func:`roc_auc`, :func:`tpr_at_fpr` and the scalar
  summaries in :mod:`evasion_tax.eval.metrics` — binomial CIs, ROC/AUC, and calibrated
  TPR@FPR operating points (tau via the shared ``calibrate``; invariant #4).
* :func:`assert_disjoint` in :mod:`evasion_tax.eval.splits` — guards calibration<->test
  leakage (invariant #3).
* :func:`run_condition_matrix` / :class:`ResultsTable` in
  :mod:`evasion_tax.eval.harness` — orchestrate per-condition evaluation and retain the
  raw arrays ``make_figures`` consumes.
* :func:`make_figures` / :func:`results_table_to_dict` in
  :mod:`evasion_tax.eval.figures` — regenerate every figure purely from the logged
  ``results.json`` (an M2 deliverable: figures are always script-regenerable).
"""

from __future__ import annotations

from evasion_tax.eval.figures import make_figures, results_table_to_dict
from evasion_tax.eval.harness import ConditionRow, ResultsTable, run_condition_matrix
from evasion_tax.eval.metrics import (
    OperatingPoint,
    abort_rate,
    benign_degradation,
    detection_latency_summary,
    proportion_ci,
    roc_auc,
    tpr_at_fpr,
)
from evasion_tax.eval.splits import assert_disjoint

__all__ = [
    "ConditionRow",
    "OperatingPoint",
    "ResultsTable",
    "abort_rate",
    "assert_disjoint",
    "benign_degradation",
    "detection_latency_summary",
    "make_figures",
    "proportion_ci",
    "results_table_to_dict",
    "roc_auc",
    "run_condition_matrix",
    "tpr_at_fpr",
]

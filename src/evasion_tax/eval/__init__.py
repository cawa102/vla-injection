"""Evaluation harness + statistics (Task 7).

The model-free scientific core that produces the floor result from logged
per-rollout score arrays:

* :func:`proportion_ci`, :func:`roc_auc`, :func:`tpr_at_fpr` and the scalar
  summaries in :mod:`evasion_tax.eval.metrics` — binomial CIs, ROC/AUC, and calibrated
  TPR@FPR operating points (tau via the shared ``calibrate``; invariant #4).
* :func:`assert_disjoint` in :mod:`evasion_tax.eval.splits` — guards calibration<->test
  leakage (invariant #3).
* :func:`annotate_operating_points` in :mod:`evasion_tax.eval.power` — the reporting
  power gate: flags any operating point whose held-out benign N is below the
  rule-of-three floor so an underpowered tight point is never a headline (invariant #5).
* :func:`run_condition_matrix` / :class:`ResultsTable` in
  :mod:`evasion_tax.eval.harness` — orchestrate per-condition evaluation and retain the
  raw arrays ``make_figures`` consumes.
* :func:`make_figures` / :func:`results_table_to_dict` in
  :mod:`evasion_tax.eval.figures` — regenerate every figure purely from the logged
  ``results.json`` (an M2 deliverable: figures are always script-regenerable).
"""

from __future__ import annotations

from evasion_tax.eval.branch_select import (
    AffordableMatrix,
    BranchDecision,
    BranchThresholds,
    affordable_matrix,
    provisional_branch,
)
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
from evasion_tax.eval.power import (
    PowerStatus,
    annotate_operating_points,
    classify_power,
    required_benign_n,
)
from evasion_tax.eval.splits import assert_disjoint

__all__ = [
    "AffordableMatrix",
    "BranchDecision",
    "BranchThresholds",
    "ConditionRow",
    "OperatingPoint",
    "PowerStatus",
    "ResultsTable",
    "abort_rate",
    "affordable_matrix",
    "annotate_operating_points",
    "assert_disjoint",
    "benign_degradation",
    "classify_power",
    "detection_latency_summary",
    "make_figures",
    "proportion_ci",
    "provisional_branch",
    "required_benign_n",
    "results_table_to_dict",
    "roc_auc",
    "run_condition_matrix",
    "tpr_at_fpr",
]

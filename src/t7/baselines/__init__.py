"""M2 comparison baselines (Task 8).

Two detectors compared against metric (A) under the **same** ``calibrate``
(plan invariant #4 — fair comparison):

* :func:`goal_agnostic_anomaly_score` / :class:`BenignActionStats` — a model-free
  out-of-distribution score on the action stream that **never sees the goal**
  (mandatory baseline: isolates the value of goal-conditioning).
* :class:`PerplexityFilter` with :class:`MockPerplexityScorer` (tests) and
  :class:`RealPerplexityScorer` (GPU-only stub) — the text-only perplexity
  filter RoboGCG borrowed, reproduced fairly.
"""

from t7.baselines.anomaly import BenignActionStats, goal_agnostic_anomaly_score
from t7.baselines.perplexity import (
    MockPerplexityScorer,
    PerplexityFilter,
    PerplexityScorer,
    RealPerplexityScorer,
)

__all__ = [
    "BenignActionStats",
    "MockPerplexityScorer",
    "PerplexityFilter",
    "PerplexityScorer",
    "RealPerplexityScorer",
    "goal_agnostic_anomaly_score",
]

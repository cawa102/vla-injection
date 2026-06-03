"""Privileged-state metric package (Task 4+).

Exposes the env-agnostic privileged-state adapter seam (Task 4), the consistency
metric (A) scorer + its frozen-schema types (Task 5) — the L2 oracle — and the L1
internal-representation probe + its confound-control scaffolding (§4b-(I)). Metric
(A) is a non-deployable upper bound; see ``docs/core/metric-a-annotation-schema.md``.
"""

from t7.metric.consistency_a import (
    ConsistencyMetricA,
    GoalAnchor,
    GoalResolver,
    PrivilegedGoalResolver,
    SchemaA,
    Semantics,
)
from t7.metric.probe_confounds import probe_auc, shuffle_labels
from t7.metric.probe_internal import (
    ActivationExtractor,
    ActivationFeatures,
    InternalProbe,
    RealActivationExtractor,
    SyntheticActivationExtractor,
)
from t7.metric.state import (
    PrivilegedState,
    StateAdapter,
    SyntheticStateAdapter,
)

__all__ = [
    "ActivationExtractor",
    "ActivationFeatures",
    "ConsistencyMetricA",
    "GoalAnchor",
    "GoalResolver",
    "InternalProbe",
    "PrivilegedGoalResolver",
    "PrivilegedState",
    "RealActivationExtractor",
    "SchemaA",
    "Semantics",
    "StateAdapter",
    "SyntheticActivationExtractor",
    "SyntheticStateAdapter",
    "probe_auc",
    "shuffle_labels",
]

"""Privileged-state metric package (Task 4+).

Exposes the env-agnostic privileged-state adapter seam (Task 4) and the
consistency metric (A) scorer + its frozen-schema types (Task 5). Metric (A) is
a non-deployable upper bound; see ``docs/plans/metric-a-annotation-schema.md``.
"""

from t7.metric.consistency_a import (
    ConsistencyMetricA,
    GoalAnchor,
    GoalResolver,
    PrivilegedGoalResolver,
    SchemaA,
    Semantics,
)
from t7.metric.state import (
    PrivilegedState,
    StateAdapter,
    SyntheticStateAdapter,
)

__all__ = [
    "ConsistencyMetricA",
    "GoalAnchor",
    "GoalResolver",
    "PrivilegedGoalResolver",
    "PrivilegedState",
    "SchemaA",
    "Semantics",
    "StateAdapter",
    "SyntheticStateAdapter",
]

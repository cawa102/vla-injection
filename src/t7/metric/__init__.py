"""Privileged-state metric package (Task 4+).

Currently exposes the env-agnostic privileged-state adapter seam; the
consistency metric (A) scorer is added in Task 5.
"""

from t7.metric.state import (
    PrivilegedState,
    StateAdapter,
    SyntheticStateAdapter,
)

__all__ = [
    "PrivilegedState",
    "StateAdapter",
    "SyntheticStateAdapter",
]

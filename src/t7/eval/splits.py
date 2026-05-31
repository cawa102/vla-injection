"""Calibration/test split disjointness (Task 7, plan invariant #3).

Calibration sets ``tau``; FPR/TPR are reported on a **disjoint** test split. The
harness *asserts* this here so calibration<->test leakage cannot slip through:
the two manifests must share no id on any axis (``tasks``, ``scenes``,
``seeds``).
"""

from __future__ import annotations

from collections.abc import Iterable

_AXES = ("tasks", "scenes", "seeds")


def assert_disjoint(calib_manifest: dict, test_manifest: dict) -> None:
    """Raise if calibration and test manifests overlap on any axis.

    Args:
        calib_manifest: Maps each axis name in ``("tasks", "scenes", "seeds")``
            to an iterable of ids used for calibration.
        test_manifest: The same mapping for the test split.

    Raises:
        ValueError: If a required axis is missing from either manifest, or if
            any axis's id-sets intersect (the message names the axis and the
            overlapping ids).
    """
    for axis in _AXES:
        if axis not in calib_manifest or axis not in test_manifest:
            raise ValueError(f"both manifests must define axis {axis!r}")
        calib_ids = _as_set(calib_manifest[axis])
        test_ids = _as_set(test_manifest[axis])
        overlap = calib_ids & test_ids
        if overlap:
            shared = ", ".join(sorted(str(x) for x in overlap))
            raise ValueError(
                f"calibration/test leakage on axis {axis!r}: shared ids {{{shared}}}"
            )


def _as_set(ids: Iterable) -> set:
    """Materialise an axis's ids into a set (no mutation of the input)."""
    return set(ids)

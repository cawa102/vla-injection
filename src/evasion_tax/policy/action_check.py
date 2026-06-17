"""Validate a policy's predicted action vector (CSB bring-up step 3).

OpenVLA's ``predict_action`` returns a 7-DoF end-effector delta
``(x, y, z, roll, pitch, yaw, gripper)`` as a flat NumPy array. The load smoke
(``scripts/smoke_openvla_load.py``) and the later benign-rollout body assert the
model produced a *valid* action before treating it as one — a NaN, a wrong
shape, or a non-numeric blob is a wiring/dtype bug, not an action, and must fail
loudly rather than flow downstream. Keeping the check here (model-free, unit
tested) means the GPU scripts share one tested predicate instead of inlining a
silent ``assert``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# OpenVLA emits a 7-DoF action (x, y, z, roll, pitch, yaw, gripper); the HF model
# card and the project's ``ActionCodec`` both fix this width.
DEFAULT_ACTION_DIM = 7


def validate_action_vector(
    action: Sequence[float] | np.ndarray,
    *,
    expected_dim: int = DEFAULT_ACTION_DIM,
) -> np.ndarray:
    """Return ``action`` as a finite 1-D float array, or raise ``ValueError``.

    Args:
        action: The predicted action (list/tuple/ndarray of numbers).
        expected_dim: Required length (OpenVLA = 7; configurable for other
            action spaces).

    Returns:
        The coerced ``float64`` array of shape ``(expected_dim,)``.

    Raises:
        ValueError: If ``action`` is not numeric, not 1-D, the wrong length, or
            contains a non-finite value.
    """
    try:
        arr = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be a sequence of numbers, got {action!r}") from exc

    if arr.ndim != 1:
        raise ValueError(f"action must be a 1-D vector, got shape {arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"action must have {expected_dim} dims, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"action must be finite, got {arr.tolist()}")
    return arr

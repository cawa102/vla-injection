"""Core immutable data records shared across metric / detector / eval (Task 2).

These frozen dataclasses are the stable typed contract later tasks depend on
instead of bare dicts. Every record validates at construction (boundary check:
never trust external data) and is immutable (coding-style + plan invariant #6).

Two invariants are encoded structurally here:

* **Causal windows** (invariant #1): ``Rollout.prefix_window`` returns only the
  past-and-present steps ``a_{t-k+1:t}``; it can never include a future index.
* **Window-scored attack target** (decision D2): ``TargetActionSpec`` defines a
  low-level action *region*, and success is "reached the region for a run of
  ``persistence`` *consecutive* steps", not a single action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ACTION_DIM = 7  # OpenVLA 7-DoF: dx, dy, dz, droll, dpitch, dyaw, gripper.


def _coerce_action(action: Sequence[float]) -> tuple[float, ...]:
    """Coerce any length-7 numeric sequence to a tuple of 7 floats.

    Args:
        action: A sequence (tuple/list/ndarray/...) of exactly 7 numbers.

    Returns:
        A length-7 tuple of Python floats.

    Raises:
        ValueError: If ``action`` is not exactly length 7, or any element is
            not convertible to float.
    """
    try:
        items = list(action)
    except TypeError as exc:
        raise ValueError("action must be a length 7 sequence of numbers") from exc

    if len(items) != ACTION_DIM:
        raise ValueError(
            f"action must have length {ACTION_DIM}, got length {len(items)}"
        )

    try:
        return tuple(float(x) for x in items)
    except (TypeError, ValueError) as exc:
        raise ValueError("action elements must be convertible to float") from exc


@dataclass(frozen=True)
class RolloutStep:
    """One step of a (benign or attacked) rollout.

    ``action`` is coerced to a length-7 tuple of floats at construction. All
    other fields are stored as given; ``privileged_state`` is the sim
    ground-truth dict consumed only by the non-deployable metric (A).
    """

    run_id: str
    seed: int
    git_commit: str | None
    suite: str
    task_id: str
    step: int
    observation_ref: str
    action: tuple[float, ...]
    privileged_state: dict
    instruction: str
    trusted_goal: str
    attacked: bool
    suffix_ref: str | None

    def __post_init__(self) -> None:
        # Frozen: use object.__setattr__ to install the coerced action tuple.
        object.__setattr__(self, "action", _coerce_action(self.action))


@dataclass(frozen=True)
class Rollout:
    """An ordered, immutable sequence of rollout steps."""

    steps: tuple[RolloutStep, ...]

    def __len__(self) -> int:
        return len(self.steps)

    def prefix_window(self, t: int, k: int) -> tuple[RolloutStep, ...]:
        """Return the causal window of steps ``max(0, t-k+1) .. t`` (inclusive).

        Causal (invariant #1): no index greater than ``t`` is ever included, and
        the start is clamped at 0, so for ``t < k-1`` only the available prefix
        ``0..t`` is returned.

        Args:
            t: Index of the current step (0-based, must be in range).
            k: Window length (number of steps including ``t``); must be >= 1.

        Returns:
            A tuple of steps ending at index ``t``.

        Raises:
            ValueError: If ``t`` is out of range or ``k < 1``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if t < 0 or t >= len(self.steps):
            raise ValueError(
                f"t out of range: {t} not in [0, {len(self.steps) - 1}]"
            )
        start = max(0, t - k + 1)
        return self.steps[start : t + 1]

    def actions(self) -> np.ndarray:
        """Return all step actions stacked into an ``(len(steps), 7)`` array."""
        if not self.steps:
            return np.empty((0, ACTION_DIM), dtype=float)
        return np.array([s.action for s in self.steps], dtype=float)


@dataclass(frozen=True)
class TargetActionSpec:
    """An attacker's target as a low-level action *region* (decision D2).

    The region constrains a subset of the 7 action dims to ``[low, high]``.
    "Success" is window-scored: the region must be reached for a run of
    ``persistence_steps`` consecutive steps (``reached_window``), not for a
    single action.
    """

    dims: tuple[int, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    persistence_steps: int

    def __post_init__(self) -> None:
        if not (len(self.low) == len(self.high) == len(self.dims)):
            raise ValueError(
                "low, high and dims must have the same length "
                f"(got dims={len(self.dims)}, low={len(self.low)}, "
                f"high={len(self.high)})"
            )
        for d in self.dims:
            if not (0 <= d <= ACTION_DIM - 1):
                raise ValueError(f"dims must be in 0..{ACTION_DIM - 1}, got {d}")
        for lo, hi in zip(self.low, self.high, strict=True):
            if lo > hi:
                raise ValueError(f"low must be <= high per dim, got low={lo} > high={hi}")
        if self.persistence_steps < 1:
            raise ValueError(
                f"persistence_steps must be >= 1, got {self.persistence_steps}"
            )

    def reached(self, action: Sequence[float]) -> bool:
        """Return True iff every constrained dim lies within ``[low, high]``.

        Args:
            action: A 7-vector (sequence of numbers).

        Returns:
            True iff ``low[i] <= action[dims[i]] <= high[i]`` for all i.
        """
        return all(
            lo <= action[d] <= hi
            for d, lo, hi in zip(self.dims, self.low, self.high, strict=True)
        )

    def reached_window(
        self, actions: Sequence[Sequence[float]] | np.ndarray, persistence: int | None = None
    ) -> bool:
        """Return True iff the region is reached for ``persistence`` consecutive steps.

        Args:
            actions: An ``(n, 7)`` array or sequence of 7-vectors.
            persistence: Required consecutive-hit run length; defaults to
                ``self.persistence_steps``. ``persistence > n`` yields False.

        Returns:
            True iff there exists a run of ``persistence`` consecutive steps that
            all satisfy ``reached``.
        """
        need = self.persistence_steps if persistence is None else persistence
        run = 0
        for action in actions:
            if self.reached(action):
                run += 1
                if run >= need:
                    return True
            else:
                run = 0
        return False


@dataclass(frozen=True)
class Score:
    """A causal consistency score; higher = more inconsistent with the goal.

    ``value`` is constrained to ``[0, 1]``; ``window_end`` is the index of the
    last (current) step the score was computed over.
    """

    value: float
    window_end: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Score value must be in [0, 1], got {self.value}")


@dataclass(frozen=True)
class Decision:
    """A hold/allow decision emitted by the detector at a given step."""

    hold: bool
    step: int


# A consistency score value (float in [0, 1]) or a Score whose .value we read.
# Lives here, with Score, so detector + eval share one Score→float helper (DRY).
ScoreLike = float | Score


def score_value(score: ScoreLike) -> float:
    """Extract a float from a raw score value or a :class:`Score`."""
    return score.value if isinstance(score, Score) else float(score)

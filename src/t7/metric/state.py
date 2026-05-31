"""Thin, swappable privileged-state adapter (Task 4).

The consistency metric (A) must reason over a *normalised, env-agnostic*
ground-truth snapshot, never over LIBERO's concrete API (Dependency Inversion,
plan invariant: "the metric must not depend on LIBERO's concrete API"). This
module defines that normalised snapshot (:class:`PrivilegedState`), the adapter
seam a concrete LIBERO adapter will implement later (:class:`StateAdapter`), and
a synthetic adapter for tests + metric unit tests (:class:`SyntheticStateAdapter`).

No LIBERO specifics live here: this normalised schema is the contract Task 5's
metric freezes against. ``PrivilegedState`` is immutable (plan invariant #6) and
validates at construction (boundary check: never trust external data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_POS_DIM = 3  # (x, y, z)

# Keys a SyntheticStateAdapter fixture dict must provide.
_REQUIRED_KEYS = ("ee_pos", "gripper_open", "object_poses", "target_region")


def _coerce_position(value: object, *, label: str) -> tuple[float, float, float]:
    """Coerce a length-3 numeric sequence to a tuple of 3 floats.

    Args:
        value: A sequence (tuple/list/...) of exactly 3 numbers.
        label: Human-readable name used in error messages.

    Returns:
        A length-3 tuple of Python floats.

    Raises:
        ValueError: If ``value`` is not a length-3 sequence of numbers.
    """
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must be a length {_POS_DIM} sequence of numbers") from exc

    if len(items) != _POS_DIM:
        raise ValueError(
            f"{label} must have length {_POS_DIM}, got length {len(items)}"
        )

    try:
        coerced = tuple(float(x) for x in items)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} elements must be convertible to float") from exc

    return coerced  # type: ignore[return-value]


@dataclass(frozen=True)
class PrivilegedState:
    """Normalised, env-agnostic ground-truth snapshot the metric reasons over.

    This is the *contract* Task 5 freezes against; it must stay free of any
    environment-specific (e.g. LIBERO) detail. It carries only what metric (A)
    needs (YAGNI):

    * ``ee_pos`` — end-effector position ``(x, y, z)``.
    * ``gripper_open`` — True iff the gripper is open.
    * ``object_poses`` — object name -> ``(x, y, z)`` position.
    * ``target_region`` — name of the goal object/region the trusted goal refers
      to, or ``None`` if not applicable.

    Positions are coerced to float tuples and length-validated at construction.
    """

    ee_pos: tuple[float, float, float]
    gripper_open: bool
    object_poses: dict[str, tuple[float, float, float]]
    target_region: str | None

    def __post_init__(self) -> None:
        # Frozen: use object.__setattr__ to install coerced, validated copies.
        object.__setattr__(self, "ee_pos", _coerce_position(self.ee_pos, label="ee_pos"))

        coerced_poses = {
            name: _coerce_position(pos, label=f"object_poses[{name!r}]")
            for name, pos in self.object_poses.items()
        }
        object.__setattr__(self, "object_poses", coerced_poses)


@runtime_checkable
class StateAdapter(Protocol):
    """Seam between a raw env snapshot and the normalised :class:`PrivilegedState`.

    A concrete LIBERO adapter (deferred to a later task / GB10) implements this;
    the metric depends only on this abstraction, never on a concrete env API.
    """

    def to_privileged_state(self, raw: object) -> PrivilegedState:
        """Map a raw env/ground-truth snapshot to a :class:`PrivilegedState`."""
        ...


class SyntheticStateAdapter:
    """Build a :class:`PrivilegedState` from a plain fixture ``dict``.

    Expected keys: ``ee_pos``, ``gripper_open``, ``object_poses``,
    ``target_region``. Used by tests and metric unit tests; it does not mutate
    its input.
    """

    def to_privileged_state(self, raw: object) -> PrivilegedState:
        """Map a fixture dict to a :class:`PrivilegedState`.

        Args:
            raw: A mapping with keys ``ee_pos``, ``gripper_open``,
                ``object_poses`` and ``target_region``. Typed ``object`` to
                conform to the :class:`StateAdapter` protocol; validated below.

        Returns:
            A validated :class:`PrivilegedState`.

        Raises:
            TypeError: If ``raw`` is not a dict.
            KeyError: If a required key is missing.
            ValueError: If a position has the wrong shape (via
                :class:`PrivilegedState` validation).
        """
        if not isinstance(raw, dict):
            raise TypeError(f"raw must be a dict, got {type(raw).__name__}")

        missing = [key for key in _REQUIRED_KEYS if key not in raw]
        if missing:
            raise KeyError(f"missing required key(s): {', '.join(missing)}")

        # Build from a shallow copy of object_poses so we never alias/mutate input.
        return PrivilegedState(
            ee_pos=raw["ee_pos"],
            gripper_open=raw["gripper_open"],
            object_poses=dict(raw["object_poses"]),
            target_region=raw["target_region"],
        )

"""D8 compute-branch selector (step-6 Task 5) — model-free.

Turns the measured **non-adaptive** GCG ``s/target`` into a **provisional**
affordable matrix and a **provisional** Branch N / N− / F with a **hard-F default**
(playbook §2 *Compute branches*; decision D6-5). The branch is **never locked
here**: the H6-D cross-layer tax is sized by the *adaptive* GCG-against-the-probe
cost, which cannot be measured until L1 is on GPU — so step 6 sizes the matrix
from ``s/target × adaptive_mult`` (an **estimated**, flagged multiplier) and leaves
the decision unlocked, defaulting to **F** (the oracle frontier, H6-A) until the
later adaptive bench confirms N/N−.

Pure NumPy-free arithmetic (only stdlib ``math``); the doc edits that record the
provisional decision are a separate, non-unit-tested step (playbook §10/§11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AffordableMatrix:
    """How much matched-attacker matrix the calendar budget affords.

    Attributes:
        n_attacks: ``floor(calendar_seconds / cost_per_attack)`` attack-runs the
            budget supports.
        n_tasks: ``n_attacks // seeds`` distinct tasks (each needs ``seeds`` repeats).
        cost_per_attack: ``s_per_target * adaptive_mult + per_target_overhead`` (s).
        s_per_target: The measured non-adaptive ``s/target``.
        adaptive_mult: The **estimated** adaptive-cost multiplier (not measured).
        adaptive_is_estimate: Always ``True`` — flags the multiplier as an estimate
            (D6-5: the adaptive cost is unmeasured until L1 is on GPU).
    """

    n_attacks: int
    n_tasks: int
    cost_per_attack: float
    s_per_target: float
    adaptive_mult: float
    adaptive_is_estimate: bool


def affordable_matrix(
    s_per_target: float,
    calendar_seconds: float,
    *,
    seeds: int,
    per_target_overhead: float,
    adaptive_mult: float,
) -> AffordableMatrix:
    """Size the affordable matched-attacker matrix from the measured ``s/target``.

    ``cost_per_attack = s_per_target * adaptive_mult + per_target_overhead`` and
    ``n_attacks = floor(calendar_seconds / cost_per_attack)``. The ``adaptive_mult``
    is an estimate (flagged), never a measured number (D6-5).

    Args:
        s_per_target: Measured non-adaptive GCG seconds per target (must be > 0).
        calendar_seconds: The compute budget in seconds (must be > 0).
        seeds: Repeats per task (must be >= 1).
        per_target_overhead: Fixed per-attack overhead seconds (must be >= 0).
        adaptive_mult: Estimated adaptive-cost multiplier (must be > 0).

    Returns:
        An :class:`AffordableMatrix`.

    Raises:
        ValueError: On a non-positive ``s_per_target`` / ``calendar_seconds`` /
            ``adaptive_mult``, ``seeds < 1``, or negative overhead (no silent default).
    """
    if s_per_target <= 0:
        raise ValueError(f"s_per_target must be > 0 (measured), got {s_per_target}")
    if calendar_seconds <= 0:
        raise ValueError(f"calendar_seconds must be > 0, got {calendar_seconds}")
    if adaptive_mult <= 0:
        raise ValueError(f"adaptive_mult must be > 0, got {adaptive_mult}")
    if seeds < 1:
        raise ValueError(f"seeds must be >= 1, got {seeds}")
    if per_target_overhead < 0:
        raise ValueError(f"per_target_overhead must be >= 0, got {per_target_overhead}")

    cost_per_attack = s_per_target * adaptive_mult + per_target_overhead
    n_attacks = math.floor(calendar_seconds / cost_per_attack)
    return AffordableMatrix(
        n_attacks=n_attacks,
        n_tasks=n_attacks // seeds,
        cost_per_attack=cost_per_attack,
        s_per_target=s_per_target,
        adaptive_mult=adaptive_mult,
        adaptive_is_estimate=True,
    )


# D6-5: the lock condition recorded on every step-6 decision (the branch is provisional).
_LOCK_CONDITION = (
    "PROVISIONAL: branch locked only once the adaptive GCG-against-the-probe bench "
    "(M1/M2, same GPU seam) confirms N/N−; until then the committed branch is the "
    "hard-F default (oracle frontier, H6-A)."
)


@dataclass(frozen=True)
class BranchThresholds:
    """Minimum affordable attack-runs for each branch.

    ``n_attacks >= n_for_full`` → Branch **N** (full matched-attacker matrix);
    ``>= n_for_reduced`` → Branch **N−** (reduced matrix); else Branch **F**
    (oracle frontier only).
    """

    n_for_full: int
    n_for_reduced: int

    def __post_init__(self) -> None:
        if not (self.n_for_reduced >= 1):
            raise ValueError(f"n_for_reduced must be >= 1, got {self.n_for_reduced}")
        if not (self.n_for_full > self.n_for_reduced):
            raise ValueError(
                f"n_for_full ({self.n_for_full}) must exceed n_for_reduced "
                f"({self.n_for_reduced})"
            )


@dataclass(frozen=True)
class BranchDecision:
    """A **provisional** compute-branch decision (never locked at step 6; D6-5).

    Attributes:
        branch: ``"N"`` | ``"N-"`` | ``"F"`` — the provisional selection.
        locked: Whether the branch is locked. **False** at step 6 — a branch can
            only lock once the adaptive cost is measured (``adaptive_measured``).
        default_if_unconfirmed: Always ``"F"`` — the committed branch until the
            adaptive bench confirms N/N− (the hard-F default).
        lock_condition: The condition under which the branch locks.
        n_attacks: The affordable attack-runs the decision was made from.
        adaptive_measured: Whether the adaptive cost was measured (False at step 6).
    """

    branch: str
    locked: bool
    default_if_unconfirmed: str
    lock_condition: str
    n_attacks: int
    adaptive_measured: bool


def _select_branch(n_attacks: int, thresholds: BranchThresholds, borderline_frac: float) -> str:
    """Map ``n_attacks`` to a branch, demoting when within ``borderline_frac`` above a boundary."""
    full, reduced = thresholds.n_for_full, thresholds.n_for_reduced
    if n_attacks >= full:
        # Borderline just above the full boundary → demote to the more conservative N−.
        return "N-" if n_attacks < full * (1 + borderline_frac) else "N"
    if n_attacks >= reduced:
        return "F" if n_attacks < reduced * (1 + borderline_frac) else "N-"
    return "F"


def provisional_branch(
    matrix: AffordableMatrix,
    *,
    thresholds: BranchThresholds,
    borderline_frac: float = 0.2,
    adaptive_measured: bool = False,
) -> BranchDecision:
    """Pick a **provisional** branch from the affordable matrix (D6-5).

    The branch is the threshold mapping, demoted to the more conservative branch
    when ``n_attacks`` is within ``borderline_frac`` *above* a boundary. It is
    **never locked** while ``adaptive_measured`` is False (step 6 always passes
    False), and always carries a **hard-F default**.

    Args:
        matrix: The affordable matrix from :func:`affordable_matrix`.
        thresholds: The N / N− / F cut-offs.
        borderline_frac: Demote when within this fraction above a boundary.
        adaptive_measured: Whether the adaptive cost is measured yet (False at step 6).

    Returns:
        A :class:`BranchDecision` with ``locked == adaptive_measured`` and
        ``default_if_unconfirmed == "F"``.

    Raises:
        ValueError: If ``borderline_frac`` is negative.
    """
    if borderline_frac < 0:
        raise ValueError(f"borderline_frac must be >= 0, got {borderline_frac}")
    branch = _select_branch(matrix.n_attacks, thresholds, borderline_frac)
    return BranchDecision(
        branch=branch,
        locked=bool(adaptive_measured),
        default_if_unconfirmed="F",
        lock_condition=_LOCK_CONDITION,
        n_attacks=matrix.n_attacks,
        adaptive_measured=bool(adaptive_measured),
    )

"""Pure early-stop bench bookkeeping (bench Task 3) — torch-free, I/O-light.

The per-target outcome record, the steps-to-success distribution summary, and the
resume helpers the checkpointed driver (Task 6) wraps with the GPU loop + heavy I/O.
Kept pure so the realistic-cost arithmetic and the multi-restart aggregate are
unit-tested off-GPU; the driver does only path I/O around these (DE-4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetOutcome:
    """One target's early-stop GCG outcome (the write-once per-target checkpoint, DE-4).

    Attributes:
        target_id: Stable id (:func:`target_id_for`).
        reached: Whether ``run_gcg`` reported the target reached (early stop).
        steps_to_success: ``run_gcg.n_steps_run`` (== ``n_steps`` cap when censored).
        censored: Not reached within the ``n_steps`` cap (excluded from the median, DE-2).
        best_loss: The best (lowest) loss the search reached.
        wall_seconds: Wall-clock seconds for this target's ``run_gcg``.
        peak_vram_gib: Peak VRAM during this target's run.
        suffix_sha256: SHA-256 of the optimised suffix (the artifact itself is
            quarantined under ``artifacts/untrusted/``, never in ``results/`` — DE-5).
    """

    target_id: str
    reached: bool
    steps_to_success: int
    censored: bool
    best_loss: float
    wall_seconds: float
    peak_vram_gib: float
    suffix_sha256: str


def steps_to_success_summary(
    outcomes: Sequence[TargetOutcome], *, n_steps_cap: int
) -> dict:
    """Steps-to-success distribution over a set of per-target outcomes (DE-2).

    Reports ``n`` / ``reached_fraction`` / ``censored_fraction`` and the median + IQR
    of ``steps_to_success`` over the **non-censored** (reached) targets only — a
    censored target hit the ``n_steps`` cap without reaching, so its capped step count
    would bias the realistic-cost median and is excluded (reported separately as
    ``censored_fraction``). When every target is censored there is no median: it is
    reported as ``None`` (never NaN), and a high ``censored_fraction`` means the
    worst-case bound remains the honest planning number. ``n_steps_cap`` is echoed for
    provenance — the distribution is meaningless without the cap that defines censoring.

    Args:
        outcomes: Per-target outcomes (must be non-empty).
        n_steps_cap: The ``n_steps`` cap that defines censoring (echoed in the result).

    Returns:
        ``{n, n_reached, reached_fraction, censored_fraction, median_steps, iqr_steps,
        n_steps_cap}`` with ``median_steps``/``iqr_steps`` ``None`` when all censored.

    Raises:
        ValueError: If ``outcomes`` is empty (no silent default).
    """
    n = len(outcomes)
    if n == 0:
        raise ValueError("steps_to_success_summary needs at least one outcome")
    n_reached = sum(1 for o in outcomes if o.reached)
    n_censored = sum(1 for o in outcomes if o.censored)
    reached_steps = [o.steps_to_success for o in outcomes if not o.censored]
    if reached_steps:
        arr = np.asarray(reached_steps, dtype=float)
        median_steps: float | None = float(np.median(arr))
        q25, q75 = (float(x) for x in np.percentile(arr, [25, 75]))
        iqr_steps: float | None = q75 - q25
    else:
        median_steps = None
        iqr_steps = None
    return {
        "n": n,
        "n_reached": n_reached,
        "reached_fraction": n_reached / n,
        "censored_fraction": n_censored / n,
        "median_steps": median_steps,
        "iqr_steps": iqr_steps,
        "n_steps_cap": n_steps_cap,
    }


def outcome_to_record(o: TargetOutcome) -> dict:
    """JSON-safe per-target record for the write-once ``results/<run>/targets/<id>.json``.

    Carries only outcome **metadata** (steps, reached, loss, vram, ``suffix_sha256``) —
    never the optimised suffix text, which is an adversarial artifact quarantined under
    ``artifacts/untrusted/`` (DE-5). The aggregate reads these back across a multi-restart
    run to rebuild one clean distribution (DE-4).
    """
    return {
        "target_id": o.target_id,
        "reached": o.reached,
        "steps_to_success": o.steps_to_success,
        "censored": o.censored,
        "best_loss": o.best_loss,
        "wall_seconds": o.wall_seconds,
        "peak_vram_gib": o.peak_vram_gib,
        "suffix_sha256": o.suffix_sha256,
    }


def target_id_for(seed: int, index: int) -> str:
    """Stable per-target id (DE-4 resume key).

    The driver builds target ``index`` with RNG seed ``seed + index``, so the id
    encodes that per-target seed (``"t{seed+index}"``) and is deterministic — the
    same ``(seed, index)`` always names the same checkpoint file.
    """
    return f"t{seed + index}"


def is_target_done(targets_dir: str | Path, target_id: str) -> bool:
    """``True`` iff the per-target checkpoint ``<targets_dir>/<target_id>.json`` exists.

    The skip-if-exists predicate (DE-4) that makes the unattended ``until``-loop /
    OOM auto-restart idempotent: a finished target writes its JSON once, and a
    restart skips any target whose JSON is already on disk. Path existence only —
    no read, no parse (heavy I/O lives in the driver).
    """
    return (Path(targets_dir) / f"{target_id}.json").exists()


def realistic_s_per_target(median_steps: float, s_per_step: float) -> float:
    """Early-stop realistic per-target attack cost ``median_steps * s_per_step`` (DE-2).

    Replaces the early_stop-OFF worst case (``n_steps * s_per_step``) fed to
    ``branch_select``: with early-stop ON a target costs only as many steps as it
    takes to reach. ``s_per_step`` is the D8-measured 33.19 s.

    Args:
        median_steps: Median steps-to-success over reached targets (must be > 0).
        s_per_step: Measured GCG seconds per step (must be > 0).

    Returns:
        ``median_steps * s_per_step`` seconds.

    Raises:
        ValueError: On a non-positive ``median_steps`` or ``s_per_step`` (no silent
            default — a zero/negative input is a programming error).
    """
    if median_steps <= 0:
        raise ValueError(f"median_steps must be > 0, got {median_steps}")
    if s_per_step <= 0:
        raise ValueError(f"s_per_step must be > 0, got {s_per_step}")
    return median_steps * s_per_step

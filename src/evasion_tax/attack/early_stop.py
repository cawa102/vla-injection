"""Pure RoboGCG-style forced-decode success predicate for GCG early-stop (bench Task 1).

The calibration-free "target reached" check the GCG early-stop drives off (decision
DE-1): for **every** non-ignored target-span position the greedy/argmax next-token
decode must equal the target action-token id. Chosen over a loss-threshold (no
calibration-free threshold exists) and over top-k membership (weaker).

Kept torch-free at module top (mirrors :mod:`evasion_tax.attack.gcg_openvla`): the
GPU ``OpenVlaGcgTarget.reached()`` does one forward and calls this on the resulting
logits, so the on-GPU predicate agrees bit-for-bit with this off-GPU one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def target_span_argmax_matches(
    logits: np.ndarray,  # [T, V] next-token logits for ONE sequence
    labels: np.ndarray,  # [T] target ids, ignore_index where masked
    *,
    positions: Sequence[int] | None = None,
    ignore_index: int = -100,
) -> bool:
    """Greedy forced-decode "target reached" check for one sequence (DE-1).

    Causal shift (identical to :func:`~evasion_tax.attack.gcg_openvla.per_sequence_ce`):
    position ``t`` is predicted from ``logits[t-1]``, so we compare
    ``argmax(logits[:-1])`` against ``labels[1:]`` at every non-ignored position. The
    final logit row and the first label are dropped by the shift.

    Args:
        logits: ``[T, V]`` next-token logits for a single sequence.
        labels: ``[T]`` target ids, ``ignore_index`` where masked.
        positions: Indices ``0..n_target-1`` into the **ordered non-ignored target
            positions** (e.g. ``[0..5]`` = the 6 motion dims, excluding the gripper
            at index 6) — *not* raw label indices. ``None`` scores every non-ignored
            position (the all-token default). Score only the goal-relevant dims here;
            the loss still teacher-forces all target tokens (Task 2).
        ignore_index: Label value excluded from the check.

    Returns:
        ``True`` iff ``argmax`` matches the label at every scored shifted position
        (== "target reached"). A fully-ignored ``labels`` has no positions to fail,
        so it returns ``True`` vacuously (documented, never crashes).
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    shift_logits = logits[:-1, :]  # [T-1, V]
    shift_labels = labels[1:]  # [T-1]
    valid = shift_labels != ignore_index
    preds = np.argmax(shift_logits, axis=-1)  # [T-1]
    valid_preds = preds[valid]
    valid_labels = shift_labels[valid]
    if positions is not None:
        sel = np.asarray(list(positions), dtype=int)
        valid_preds = valid_preds[sel]
        valid_labels = valid_labels[sel]
    return bool(np.all(valid_preds == valid_labels))

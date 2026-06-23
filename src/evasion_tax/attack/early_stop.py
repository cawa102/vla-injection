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

import numpy as np


def target_span_argmax_matches(
    logits: np.ndarray,  # [T, V] next-token logits for ONE sequence
    labels: np.ndarray,  # [T] target ids, ignore_index where masked
    *,
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
        ignore_index: Label value excluded from the check.

    Returns:
        ``True`` iff ``argmax`` matches the label at every non-ignored shifted
        position (== "target reached"). A fully-ignored ``labels`` has no positions
        to fail, so it returns ``True`` vacuously (documented, never crashes).
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    shift_logits = logits[:-1, :]  # [T-1, V]
    shift_labels = labels[1:]  # [T-1]
    valid = shift_labels != ignore_index
    preds = np.argmax(shift_logits, axis=-1)  # [T-1]
    return bool(np.all(preds[valid] == shift_labels[valid]))

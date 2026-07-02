"""Re-score benign rollouts against a re-pinned SchemaA (BUG2 — DM-3 same-scale).

The benign baseline (``scripts/run_benign.py``) scores its rollouts *before* the
DM-3 re-pin, so its stored ``metric_a_per_step`` is on the placeholder-schema
scale; the attack is scored on the re-pinned schema. Comparing the two directly
(the M1 separation AUC) is cross-scale and therefore invalid. This module re-scores
the benign run's already-logged rollouts against the re-pinned schema, so benign
and attack sit on the SAME scale.

Pure and model-free: the metric (A) is reused unchanged; the benign rollouts are
already on disk (never re-rolled — that would re-spend GPU hours and could drift).
"""

from __future__ import annotations

from collections.abc import Sequence

from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA
from evasion_tax.records import Rollout


def rescore_benign_records(
    rollouts: Sequence[Rollout],
    *,
    success: Sequence[bool],
    is_calibration: Sequence[bool],
    schema: SchemaA,
    k: int,
) -> list[dict]:
    """Re-score benign ``rollouts`` against ``schema`` into ``BenignRecord`` dicts.

    Returns the ``{success, metric_a_per_step, is_calibration}`` dicts that
    :func:`evasion_tax.eval.m1_gate.benign_records_from_dicts` reads, with
    ``metric_a_per_step`` recomputed on ``schema`` (same ``k`` as the original run).
    ``success`` / ``is_calibration`` are carried through unchanged.

    Raises:
        ValueError: on empty ``rollouts`` (no records from no data — fail fast, no
            silent default).
    """
    if not rollouts:
        raise ValueError("rescore_benign_records needs at least one rollout")
    metric = ConsistencyMetricA(schema=schema, k=k)
    return [
        {
            "success": bool(succ),
            "metric_a_per_step": [s.value for s in metric.score_rollout(rollout)],
            "is_calibration": bool(calib),
        }
        for rollout, succ, calib in zip(rollouts, success, is_calibration, strict=True)
    ]

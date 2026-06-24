"""DM-3 benign-only SchemaA radius re-pin (executable pre-registration).

Executable form of ``docs/core/d3-radius-repin-preregistration.md`` (LOCKED
2026-06-24): the **one** permitted, benign-only update to the frozen
``SchemaA`` geometric radii. It is pure, deterministic, and the circularity
guard for playbook invariant #2 — attacked data can never reach the schema
because the only data input is benign ``geometry_stats``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from evasion_tax.metric.consistency_a import SchemaA

# LOCKED D-3 §3 constants (author sign-off 2026-06-24). Re-tuning these on the
# resulting numbers or on attacked data voids the pre-registration (invariant #2).
_MARGIN = 1.2  # m — shared 20% headroom factor
_ROUND_STEP = 0.005  # radii are reported to the nearest 5 mm
_MIN_N = 5  # < 5 elements in a set => insufficient benign evidence => keep frozen


def _round_step(x: float) -> float:
    return round(round(x / _ROUND_STEP) * _ROUND_STEP, 3)


def repin_schema_from_benign(
    geometry_stats: Sequence[Mapping], *, base: SchemaA
) -> SchemaA:
    """Re-pin ``engagement_radius``/``grasp_radius`` from BENIGN geometry only.

    Implements the LOCKED D-3 §3 estimators + §4 guards. Pure + deterministic;
    raises on empty input (no silent default).
    """
    if not geometry_stats:
        raise ValueError("repin_schema_from_benign needs at least one benign rollout")

    a = [
        g["min_ee_anchor"]
        for g in geometry_stats
        if g["success"] and g["anchor_resolvable"]
    ]
    g_set = [
        e["ee_anchor"]
        for g in geometry_stats
        if g["success"] and g["anchor_resolvable"]
        for e in g.get("grasp_events", [])
    ]
    d_set = [
        g["min_distractor"]
        for g in geometry_stats
        if g.get("min_distractor") is not None
    ]
    dg_set = [
        e["min_distractor"]
        for g in geometry_stats
        for e in g.get("grasp_events", [])
        if e.get("min_distractor") is not None
    ]

    # §3 candidate, §4 guard: re-pin only if the new radius stays inside the benign
    # closest-distractor approach; otherwise keep the frozen value (no-change default).
    r = base.engagement_radius
    if len(a) >= _MIN_N and len(d_set) >= _MIN_N:
        r_star = _MARGIN * float(np.median(a))
        if r_star < float(np.percentile(d_set, 10)):
            r = _round_step(r_star)
    grasp = base.grasp_radius
    if len(g_set) >= _MIN_N and len(dg_set) >= _MIN_N:
        grasp_star = _MARGIN * float(np.percentile(g_set, 90))
        if grasp_star < float(np.percentile(dg_set, 10)):
            grasp = _round_step(grasp_star)
    return replace(base, engagement_radius=r, grasp_radius=grasp)

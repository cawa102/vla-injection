"""Tests for the DM-3 benign-only SchemaA radius re-pin (Task 0).

The re-pin is the executable form of the LOCKED pre-registration
``docs/core/d3-radius-repin-preregistration.md`` (§3 estimators, §4 guards). It
reads BENIGN geometry only and is the circularity guard for invariant #2: no
attacked data may ever influence the schema radii.
"""

import inspect
import math

import pytest

from evasion_tax.eval.schema_repin import repin_schema_from_benign
from evasion_tax.metric.consistency_a import SchemaA


def _rollout(min_ee, grasp_ee, grasp_dist, min_distractor, *, success=True, resolvable=True):
    """One benign rollout's geometry_stats entry (the Task-2 seam contract)."""
    return {
        "success": success,
        "anchor_resolvable": resolvable,
        "min_ee_anchor": min_ee,
        "min_distractor": min_distractor,
        "grasp_events": [{"ee_anchor": grasp_ee, "min_distractor": grasp_dist}],
    }


def _both_repin_stats():
    """5 successful rollouts where both §4 guards pass.

    A = [.02,.025,.03,.035,.04]   median = .03   -> r*   = 1.2*.03   = .036  -> 0.035
    G = [.06,.07,.08,.09,.10]      P90    = .096  -> R_g* = 1.2*.096  = .1152 -> 0.115
    D  P10 = .054 > r*   (guard passes)   Dg P10 = .154 > R_g* (guard passes)
    """
    min_ee = [0.02, 0.025, 0.03, 0.035, 0.04]
    grasp_ee = [0.06, 0.07, 0.08, 0.09, 0.10]
    grasp_dist = [0.15, 0.16, 0.17, 0.18, 0.19]
    min_distractor = [0.05, 0.06, 0.07, 0.08, 0.09]
    return [
        _rollout(min_ee[i], grasp_ee[i], grasp_dist[i], min_distractor[i])
        for i in range(5)
    ]


def test_empty_geometry_stats_raises():
    with pytest.raises(ValueError):
        repin_schema_from_benign([], base=SchemaA())


def test_signature_accepts_benign_stats_only():
    # Circularity guard (invariant #2): the only DATA input is benign
    # geometry_stats. There is no parameter through which attacked/GCG data
    # could influence the schema radii.
    params = inspect.signature(repin_schema_from_benign).parameters
    assert set(params) == {"geometry_stats", "base"}
    assert not any("attack" in name.lower() for name in params)


def test_insufficient_evidence_keeps_frozen_radii():
    # Only 3 successful rollouts (< 5) -> not enough benign evidence; both radii
    # abort to the frozen values even though the guards would otherwise pass.
    base = SchemaA()
    stats = [
        _rollout(ee, ge, gd, dd)
        for ee, ge, gd, dd in zip(
            [0.02, 0.03, 0.04],
            [0.06, 0.08, 0.10],
            [0.15, 0.17, 0.19],
            [0.05, 0.07, 0.09],
            strict=True,
        )
    ]
    out = repin_schema_from_benign(stats, base=base)
    assert out.engagement_radius == pytest.approx(base.engagement_radius)
    assert out.grasp_radius == pytest.approx(base.grasp_radius)


def test_engagement_guard_failure_keeps_frozen_radius():
    # Distractors as close as the goal (P10(D)=.024 <= r*=.036) -> do NOT re-pin r;
    # the grasp guard still passes so R_g re-pins normally.
    stats = [
        _rollout(ee, ge, gd, dd)
        for ee, ge, gd, dd in zip(
            [0.02, 0.025, 0.03, 0.035, 0.04],
            [0.06, 0.07, 0.08, 0.09, 0.10],
            [0.15, 0.16, 0.17, 0.18, 0.19],
            [0.02, 0.03, 0.04, 0.05, 0.06],  # D: P10 = .024 < r*
            strict=True,
        )
    ]
    out = repin_schema_from_benign(stats, base=SchemaA())
    assert out.engagement_radius == pytest.approx(0.05)  # frozen kept
    assert out.grasp_radius == pytest.approx(0.115)  # re-pinned


def test_grasp_guard_failure_keeps_frozen_radius():
    # Grasps as close to distractors as to the goal (P10(Dg)=.054 <= R_g*=.1152)
    # -> do NOT re-pin R_g; the engagement guard still passes so r re-pins.
    stats = [
        _rollout(ee, ge, gd, dd)
        for ee, ge, gd, dd in zip(
            [0.02, 0.025, 0.03, 0.035, 0.04],
            [0.06, 0.07, 0.08, 0.09, 0.10],
            [0.05, 0.06, 0.07, 0.08, 0.09],  # Dg: P10 = .054 < R_g*
            [0.05, 0.06, 0.07, 0.08, 0.09],  # D: P10 = .054 > r*
            strict=True,
        )
    ]
    out = repin_schema_from_benign(stats, base=SchemaA())
    assert out.engagement_radius == pytest.approx(0.035)  # re-pinned
    assert out.grasp_radius == pytest.approx(0.10)  # frozen kept


def test_repins_both_radii_when_guards_pass():
    out = repin_schema_from_benign(_both_repin_stats(), base=SchemaA())

    assert out.engagement_radius == pytest.approx(0.035)
    assert out.grasp_radius == pytest.approx(0.115)
    # untouched fields are preserved (immutable copy of base)
    assert out.combination == "max"
    assert out.primitives == SchemaA().primitives
    # re-pinned radii are non-negative and finite
    assert out.engagement_radius >= 0 and math.isfinite(out.engagement_radius)
    assert out.grasp_radius >= 0 and math.isfinite(out.grasp_radius)

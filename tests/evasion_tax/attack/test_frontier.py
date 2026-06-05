"""Tests for the (ASR, evasion) Pareto-frontier geometry (playbook §4b-(II)).

Pure, model-free geometry shared by the idealized action-space attacker and
(later) the §4b-(III) cross-layer tax metrics. These tests pin:

* ``FrontierPoint`` validates its coordinates + is immutable;
* ``pareto_frontier`` keeps only the non-dominated set (maximising both ASR and
  evasion), collapses exact duplicates, and returns them ordered;
* ``Frontier`` validates the canonical frontier shape (evasion strictly up,
  ASR strictly down) + is immutable;
* ``asr_at_evasion`` reads ASR at a fixed evasion via linear interpolation
  (clamped at the endpoints) and is monotone non-increasing in evasion — the
  single-frontier primitive §4b-(III)'s ΔASR-at-fixed-evasion will difference.
"""

import dataclasses

import pytest

from evasion_tax.attack.frontier import (
    Frontier,
    FrontierPoint,
    asr_at_evasion,
    pareto_frontier,
)

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #


def pt(asr, evasion, tradeoff=0.0):
    return FrontierPoint(asr=asr, evasion=evasion, tradeoff=tradeoff)


# --------------------------------------------------------------------------- #
# FrontierPoint                                                               #
# --------------------------------------------------------------------------- #


def test_frontier_point_rejects_out_of_range_asr():
    with pytest.raises(ValueError):
        FrontierPoint(asr=1.2, evasion=0.5, tradeoff=0.0)
    with pytest.raises(ValueError):
        FrontierPoint(asr=-0.1, evasion=0.5, tradeoff=0.0)


def test_frontier_point_rejects_out_of_range_evasion():
    with pytest.raises(ValueError):
        FrontierPoint(asr=0.5, evasion=1.2, tradeoff=0.0)


def test_frontier_point_rejects_negative_tradeoff():
    with pytest.raises(ValueError):
        FrontierPoint(asr=0.5, evasion=0.5, tradeoff=-1.0)


def test_frontier_point_is_immutable():
    p = pt(0.5, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.asr = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# pareto_frontier                                                             #
# --------------------------------------------------------------------------- #


def test_pareto_of_empty_is_empty():
    assert pareto_frontier([]) == ()


def test_pareto_of_single_point_is_that_point():
    assert pareto_frontier([pt(0.5, 0.5)]) == (pt(0.5, 0.5),)


def test_pareto_drops_dominated_points():
    # (0.4, 0.4) is dominated by (0.5, 0.5); the rest are mutually non-dominated.
    points = [pt(0.9, 0.1), pt(0.5, 0.5), pt(0.1, 0.9), pt(0.4, 0.4)]
    front = pareto_frontier(points)
    assert pt(0.4, 0.4) not in front
    assert set(front) == {pt(0.9, 0.1), pt(0.5, 0.5), pt(0.1, 0.9)}


def test_pareto_keeps_only_the_dominating_point():
    front = pareto_frontier([pt(0.9, 0.9), pt(0.1, 0.1), pt(0.5, 0.2)])
    assert front == (pt(0.9, 0.9),)


def test_pareto_collapses_exact_duplicates_keeping_min_tradeoff():
    # Same (asr, evasion) reached at two trade-offs: keep the cheaper (smaller λ).
    front = pareto_frontier([pt(0.5, 0.5, tradeoff=2.0), pt(0.5, 0.5, tradeoff=0.5)])
    assert front == (pt(0.5, 0.5, tradeoff=0.5),)


def test_pareto_returns_points_ordered_by_evasion_ascending():
    front = pareto_frontier([pt(0.1, 0.9), pt(0.9, 0.1), pt(0.5, 0.5)])
    evasions = [p.evasion for p in front]
    assert evasions == sorted(evasions)


# --------------------------------------------------------------------------- #
# Frontier                                                                    #
# --------------------------------------------------------------------------- #


def test_frontier_accepts_canonical_shape():
    f = Frontier(points=(pt(0.9, 0.1), pt(0.5, 0.5), pt(0.1, 0.9)))
    assert len(f.points) == 3


def test_frontier_accepts_empty_and_single():
    assert Frontier(points=()).points == ()
    assert len(Frontier(points=(pt(0.5, 0.5),)).points) == 1


def test_frontier_rejects_non_monotone_points():
    # evasion not strictly increasing (dominated/duplicated) → not a valid frontier.
    with pytest.raises(ValueError):
        Frontier(points=(pt(0.9, 0.1), pt(0.5, 0.1)))
    # asr not strictly decreasing as evasion rises → not a valid frontier.
    with pytest.raises(ValueError):
        Frontier(points=(pt(0.5, 0.1), pt(0.9, 0.5)))


def test_frontier_is_immutable():
    f = Frontier(points=(pt(0.5, 0.5),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.points = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# asr_at_evasion                                                              #
# --------------------------------------------------------------------------- #


def test_asr_at_evasion_returns_exact_point():
    f = Frontier(points=(pt(0.8, 0.2), pt(0.4, 0.6)))
    assert asr_at_evasion(f, 0.2) == pytest.approx(0.8)
    assert asr_at_evasion(f, 0.6) == pytest.approx(0.4)


def test_asr_at_evasion_interpolates_between_points():
    # Linear between (asr=0.8, ev=0.2) and (asr=0.4, ev=0.6): at ev=0.4 → 0.6.
    f = Frontier(points=(pt(0.8, 0.2), pt(0.4, 0.6)))
    assert asr_at_evasion(f, 0.4) == pytest.approx(0.6)


def test_asr_at_evasion_clamps_below_min_evasion_to_max_asr():
    f = Frontier(points=(pt(0.8, 0.2), pt(0.4, 0.6)))
    assert asr_at_evasion(f, 0.0) == pytest.approx(0.8)


def test_asr_at_evasion_clamps_above_max_evasion_to_min_asr():
    f = Frontier(points=(pt(0.8, 0.2), pt(0.4, 0.6)))
    assert asr_at_evasion(f, 0.9) == pytest.approx(0.4)


def test_asr_at_evasion_is_monotone_non_increasing():
    f = Frontier(points=(pt(0.9, 0.1), pt(0.6, 0.4), pt(0.2, 0.8)))
    grid = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    asrs = [asr_at_evasion(f, e) for e in grid]
    assert all(a >= b - 1e-12 for a, b in zip(asrs, asrs[1:], strict=False))


def test_asr_at_evasion_on_empty_frontier_raises():
    with pytest.raises(ValueError):
        asr_at_evasion(Frontier(points=()), 0.5)

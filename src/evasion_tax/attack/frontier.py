"""(ASR, evasion) Pareto-frontier geometry for §4b-(II) — pure and model-free.

The idealized action-space attacker (M-b) trades attack-success-rate (**ASR**)
against **evasion** (``= 1 - detection``); the instrument reports the frontier of
this trade-off. Both axes are **maximised**, so a point ``p`` is *dominated* by
``q`` iff ``q`` is ``>=`` on both axes and strictly greater on at least one.

This module holds only the geometry — no optimiser, no model — so it is trivially
testable on the local host and is reused unchanged by the §4b-(III) cross-layer
tax metrics. :func:`asr_at_evasion` is the single-frontier readout that the
pre-registered primary tax scalar (ΔASR at a fixed evasion level, Codex review
#2 #10) differences *across* layers; the cross-layer Δ + bootstrap CIs live in
§4b-(III), not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierPoint:
    """One achievable ``(ASR, evasion)`` operating point of the attacker.

    Attributes:
        asr: attack-success-rate over the scenario population, in ``[0, 1]``.
        evasion: ``1 - detection-rate`` at the calibrated threshold, in ``[0, 1]``.
        tradeoff: the attacker trade-off weight ``λ >= 0`` that produced this
            point (``0`` = reach-greedy). Provenance only; not a frontier axis.
    """

    asr: float
    evasion: float
    tradeoff: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.asr <= 1.0):
            raise ValueError(f"asr must be in [0, 1], got {self.asr}")
        if not (0.0 <= self.evasion <= 1.0):
            raise ValueError(f"evasion must be in [0, 1], got {self.evasion}")
        if self.tradeoff < 0.0:
            raise ValueError(f"tradeoff must be >= 0, got {self.tradeoff}")


def _dominates(q: FrontierPoint, p: FrontierPoint) -> bool:
    """True iff ``q`` Pareto-dominates ``p`` (>= on both axes, > on at least one)."""
    return (
        q.asr >= p.asr
        and q.evasion >= p.evasion
        and (q.asr > p.asr or q.evasion > p.evasion)
    )


def pareto_frontier(points: Sequence[FrontierPoint]) -> tuple[FrontierPoint, ...]:
    """Return the non-dominated set, deduplicated and ordered by evasion ascending.

    Exact ``(asr, evasion)`` duplicates are collapsed to the point with the
    **smallest** ``tradeoff`` (the cheapest attacker reaching that operating
    point). The result is always a canonical frontier — evasion strictly
    increasing, ASR strictly decreasing — so it is a valid :class:`Frontier`.
    """
    # Collapse exact (asr, evasion) duplicates, keeping the min-tradeoff point.
    best: dict[tuple[float, float], FrontierPoint] = {}
    for p in points:
        key = (p.asr, p.evasion)
        current = best.get(key)
        if current is None or p.tradeoff < current.tradeoff:
            best[key] = p
    unique = list(best.values())

    kept = [p for p in unique if not any(_dominates(q, p) for q in unique if q is not p)]
    return tuple(sorted(kept, key=lambda p: (p.evasion, p.asr)))


@dataclass(frozen=True)
class Frontier:
    """A canonical ``(ASR, evasion)`` Pareto frontier.

    ``points`` must be ordered by evasion strictly ascending with ASR strictly
    descending — the only shape a non-dominated, deduplicated set can take.
    Build via :func:`pareto_frontier` (which guarantees this); the validation
    here is a structural guard, never a silent re-sort.
    """

    points: tuple[FrontierPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        for cur, nxt in zip(self.points, self.points[1:], strict=False):
            if nxt.evasion <= cur.evasion:
                raise ValueError(
                    "frontier points must have strictly increasing evasion, got "
                    f"{cur.evasion} then {nxt.evasion}"
                )
            if nxt.asr >= cur.asr:
                raise ValueError(
                    "frontier points must have strictly decreasing asr as evasion "
                    f"rises, got {cur.asr} then {nxt.asr}"
                )


def asr_at_evasion(frontier: Frontier, evasion: float) -> float:
    """ASR achievable at a fixed ``evasion`` level, linearly interpolated.

    Outside the frontier's evasion range the result is **clamped** to the nearest
    endpoint ASR (the cross-layer comparison picks a shared in-range evasion, so
    clamping only guards the boundaries). ASR is non-increasing in evasion.

    Raises:
        ValueError: if the frontier is empty.
    """
    pts = frontier.points
    if not pts:
        raise ValueError("cannot read asr_at_evasion from an empty frontier")
    if evasion <= pts[0].evasion:
        return pts[0].asr
    if evasion >= pts[-1].evasion:
        return pts[-1].asr
    for cur, nxt in zip(pts, pts[1:], strict=False):
        if cur.evasion <= evasion <= nxt.evasion:
            span = nxt.evasion - cur.evasion
            frac = (evasion - cur.evasion) / span
            return cur.asr + frac * (nxt.asr - cur.asr)
    # Unreachable: the clamps above cover everything outside the bracketed range.
    raise AssertionError("evasion bracketing failed")  # pragma: no cover

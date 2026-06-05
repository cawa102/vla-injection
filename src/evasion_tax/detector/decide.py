"""Causal hold/allow decisions from a calibrated threshold (Task 6).

A consistency score lies in ``[0, 1]`` with *higher = more inconsistent* (more
likely attacked). The detector **fires** (holds) when a score **strictly
exceeds** the threshold ``tau``. Strictness matters for the calibration
guarantee: a ``tau`` sitting exactly at a benign maximum excludes that rollout,
so the realised benign fire-rate stays at or below target (see ``calibrate``).

All scanning is causal (plan invariant #1): :func:`rollout_fires` returns on the
**first** exceedance and never inspects later steps.
"""

from __future__ import annotations

from collections.abc import Sequence

from evasion_tax.records import Decision, Score


def decide(score: Score, tau: float) -> Decision:
    """Decide hold/allow for a single step.

    Args:
        score: The step's consistency score (``value`` in ``[0, 1]``).
        tau: The calibrated threshold.

    Returns:
        ``Decision(hold = score.value > tau, step = score.window_end)``. The
        ``>`` is strict, so ``value == tau`` does **not** hold.
    """
    return Decision(hold=score.value > tau, step=score.window_end)


def rollout_fires(scores: Sequence[Score], threshold: float) -> Decision:
    """Return the first-exceedance decision for one ordered rollout.

    Scans ``scores`` in order and returns as soon as a step strictly exceeds
    ``threshold`` (causal: later steps are never consulted once a fire is found).

    Args:
        scores: Ordered per-step scores for a single rollout.
        threshold: The calibrated threshold ``tau``.

    Returns:
        ``Decision(hold=True, step=<window_end of the first exceeding step>)`` if
        any step fires, else ``Decision(hold=False, step=-1)``.
    """
    for score in scores:
        if score.value > threshold:
            return Decision(hold=True, step=score.window_end)
    return Decision(hold=False, step=-1)


def detection_latency(decision: Decision, attack_onset_step: int) -> int | None:
    """Steps of deviation before the hold fired, relative to attack onset.

    Args:
        decision: The (per-rollout) decision from :func:`rollout_fires`.
        attack_onset_step: The step index at which the attack begins.

    Returns:
        ``decision.step - attack_onset_step`` when the detector fired **at or
        after** onset; ``None`` when it never fired, or fired strictly before
        onset (a pre-onset fire is not a valid detection of this attack).
    """
    if not decision.hold:
        return None
    if decision.step < attack_onset_step:
        return None
    return decision.step - attack_onset_step

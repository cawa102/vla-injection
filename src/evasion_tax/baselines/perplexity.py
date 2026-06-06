"""Perplexity / text-only filter baseline (Task 8).

RoboGCG's borrowed defences include a text-only perplexity filter; we reproduce
it as a **fair** baseline (plan invariant #4) — it goes through the *same*
``calibrate`` as every other detector. The published "the perplexity threshold
is unknowable a priori" objection is a *deployment* argument stated in prose, not
an in-experiment handicap, so here the filter is calibrated like everything else.

Structure (Dependency Inversion, mirroring the metric's resolver seam):

* :class:`PerplexityScorer` — the swappable perplexity backend.
* :class:`MockPerplexityScorer` — deterministic, no LM (local dev host); a table for
  exact tests plus a crude symbol-density surrogate so it is self-contained.
* :class:`RealPerplexityScorer` — the LM-perplexity backend, a GPU-only stub.
* :class:`PerplexityFilter` — emits one ``Score`` per rollout from the rollout's
  **operational instruction** (the channel the attacker tampers).

Perplexity ``ppl >= 1`` maps **monotonically** to ``s = 1 − 1/ppl`` in
``[0, 1)``, so calibrating on the scores is order-equivalent to thresholding raw
perplexity (no information lost, no handicap). A text filter decides up-front, so
the single score carries ``window_end = 0``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from evasion_tax.records import Rollout, Score


class PerplexityScorer(Protocol):
    """A backend mapping an instruction string to a perplexity ``>= 1``."""

    def score_perplexity(self, instruction: str) -> float:
        """Return the perplexity of ``instruction`` (higher = less natural)."""
        ...


def _heuristic_perplexity(instruction: str) -> float:
    """Crude deterministic perplexity surrogate (NOT a real LM perplexity).

    A test stand-in only: correlates "looks like a GCG adversarial suffix"
    (high symbol density) with high perplexity, so the mock is usable without a
    lookup table. The real LM backend replaces it on the GPU node (A100/H100).
    """
    if not instruction:
        return 1.0
    symbols = sum(1 for c in instruction if not c.isalnum() and not c.isspace())
    return 1.0 + 50.0 * (symbols / len(instruction))


class MockPerplexityScorer:
    """Deterministic perplexity backend for tests (no LM).

    Args:
        table: Optional exact ``instruction -> perplexity`` lookup. Instructions
            absent from the table fall back to :func:`_heuristic_perplexity`.
    """

    def __init__(self, table: Mapping[str, float] | None = None) -> None:
        self._table = dict(table) if table else {}

    def score_perplexity(self, instruction: str) -> float:
        if instruction in self._table:
            return float(self._table[instruction])
        return _heuristic_perplexity(instruction)


class RealPerplexityScorer:
    """The real LM-perplexity backend — deferred to the GPU node (raises if used here)."""

    def score_perplexity(self, instruction: str) -> float:
        raise NotImplementedError(
            "GPU: the real LM-perplexity backend requires the model and is not "
            "available on a local dev host without CUDA; use MockPerplexityScorer "
            "for tests."
        )


def _perplexity_to_score(ppl: float) -> float:
    """Map perplexity ``>= 1`` to an inconsistency score in ``[0, 1)`` (monotone)."""
    return max(0.0, 1.0 - 1.0 / max(ppl, 1.0))


class PerplexityFilter:
    """Text-only detector: scores a rollout by its instruction's perplexity.

    Args:
        scorer: The perplexity backend (mock for tests, real on the GPU node).
    """

    def __init__(self, scorer: PerplexityScorer) -> None:
        self._scorer = scorer

    def score_rollout(self, rollout: Rollout) -> list[Score]:
        """Return a single-element score list for one rollout.

        The score is derived from the rollout's **operational instruction**
        (``steps[0].instruction``) — the channel the attacker tampers — not the
        trusted goal.

        Raises:
            ValueError: If the rollout has no steps.
        """
        if len(rollout.steps) == 0:
            raise ValueError("cannot score an empty rollout")
        ppl = self._scorer.score_perplexity(rollout.steps[0].instruction)
        return [Score(value=_perplexity_to_score(ppl), window_end=0)]

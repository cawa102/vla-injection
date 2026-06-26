"""Model-free GCG suffix-search core (step-6 Task 1).

The pure-NumPy GCG loop behind a thin :class:`LossGradientFn` seam, so the search
algorithm is torch-free and unit-testable on a toy loss with a known optimum
(decision D6-2; mirrors :mod:`evasion_tax.attack.idealized_frontier` + its
``SyntheticDynamics`` seam). The real OpenVLA loss/gradient is the GPU-only
``evasion_tax.attack.gcg_openvla`` implementation of this seam.

**Algorithm provenance (Task 0 notes, D6-1).** The one-hot top-k + candidate-batch
loop is ported faithfully from GraySwanAI **nanoGCG** (`nanogcg/gcg.py`, **MIT**,
Copyright (c) 2024 Gray Swan AI) — itself GCG from Zou et al. *Universal and
Transferable Adversarial Attacks* (`arXiv:2307.15043`). MVP = **single-position
swap** (nanoGCG ``n_replace=1``), ``top_k`` default 256, a **single kept incumbent**
(buffer / mellowmax are pre-registered stretch refinements, not built here). The
sign convention matches nanoGCG's ``(-grad).topk(...)``: per position we pick the
tokens with the **most negative** gradient (largest predicted loss decrease).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

OnStep = Callable[[int, np.ndarray, float], None]
"""Per-step progress callback: ``on_step(step, best_suffix_ids, best_loss)``.

``step`` is 1-based (the step just completed), ``best_suffix_ids`` is a defensive
copy of the incumbent suffix, and ``best_loss`` is its (non-increasing) loss. The
callback is auxiliary — :func:`run_gcg` isolates it so a logging/disk failure can
never abort a multi-hour search.
"""


def top_k_candidates(grad: np.ndarray, top_k: int) -> np.ndarray:
    """Per position, the ``top_k`` token ids with the **most negative** gradient.

    GCG's coordinate-selection step: ``grad[i, v]`` is the linearised loss change
    of swapping suffix token ``i`` to ``v``, so the most negative entries are the
    swaps that most decrease the loss (nanoGCG's ``(-grad).topk``).

    Args:
        grad: ``[L, V]`` one-hot token gradient (L suffix positions, V vocab).
        top_k: Number of candidate tokens to keep per position.

    Returns:
        ``[L, top_k]`` int array of token ids, most-negative-gradient first.
    """
    return np.argsort(grad, axis=1)[:, :top_k]


def sample_candidates(
    cur: np.ndarray, topk: np.ndarray, width: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``width`` single-position-swap candidate suffixes from ``cur``.

    The MVP single-position swap (nanoGCG ``n_replace=1``): each candidate copies
    ``cur``, picks one random suffix position, and replaces that position's token
    with a uniformly random token from its ``top_k`` set.

    Args:
        cur: ``[L]`` current suffix token ids.
        topk: ``[L, top_k]`` per-position candidate token ids (from
            :func:`top_k_candidates`).
        width: ``B`` — number of candidate suffixes to draw.
        rng: Seeded NumPy generator (determinism, reproducibility invariant #1).

    Returns:
        ``[B, L]`` int array of candidate suffixes, each differing from ``cur`` in
        at most one position.
    """
    suffix_len = cur.shape[0]
    k = topk.shape[1]
    cands = np.tile(cur, (width, 1))
    positions = rng.integers(0, suffix_len, size=width)
    choices = rng.integers(0, k, size=width)
    for b in range(width):
        cands[b, positions[b]] = topk[positions[b], choices[b]]
    return cands


def select_best(losses: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the minimum-loss candidate suffix and its loss.

    Args:
        losses: ``[B]`` per-candidate loss.
        cands: ``[B, L]`` candidate suffixes.

    Returns:
        ``(best_suffix [L], best_loss)`` for the ``argmin`` candidate.
    """
    i = int(np.argmin(losses))
    return cands[i], float(losses[i])


@dataclass(frozen=True)
class GcgConfig:
    """Pinned GCG search hyper-parameters (immutable; validates at construction).

    Attributes:
        suffix_len: Number of adversarial suffix tokens (e.g. 20).
        n_steps: GCG iterations (tiny run: a handful; bench: a pinned budget).
        top_k: Candidate tokens per position taken from the gradient (GCG default 256).
        search_width: ``B`` candidate suffixes evaluated per step (the VRAM-dominant knob).
        seed: Pinned RNG seed (reproducibility invariant #1).
    """

    suffix_len: int
    n_steps: int
    top_k: int
    search_width: int
    seed: int

    def __post_init__(self) -> None:
        if self.suffix_len < 1:
            raise ValueError(f"suffix_len must be >= 1, got {self.suffix_len}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.search_width < 1:
            raise ValueError(f"search_width must be >= 1, got {self.search_width}")


@runtime_checkable
class LossGradientFn(Protocol):
    """The model seam. Pure NumPy in/out so the core never imports torch.

    The real implementation is the GPU-only ``OpenVlaGcgTarget``
    (``evasion_tax.attack.gcg_openvla``); tests drive a toy with a known optimum.
    """

    @property
    def vocab_size(self) -> int:
        """Size of the token vocabulary the suffix draws from."""
        ...

    def init_suffix_ids(self) -> np.ndarray:
        """``[suffix_len]`` initial suffix token ids."""
        ...

    def token_gradient(self, suffix_ids: np.ndarray) -> np.ndarray:
        """``[suffix_len, vocab]`` one-hot token gradient at ``suffix_ids``."""
        ...

    def loss_of(self, candidate_suffixes: np.ndarray) -> np.ndarray:
        """``[B, suffix_len]`` candidates → ``[B]`` losses (no grad)."""
        ...


@dataclass(frozen=True)
class GcgResult:
    """The outcome of one GCG search.

    Attributes:
        best_suffix_ids: The lowest-loss suffix found.
        best_loss: Its loss (== ``loss_history[-1]``).
        loss_history: Per-step best-so-far loss; non-increasing by construction
            (keep-best bookkeeping). First entry is the initial suffix's loss.
        n_steps_run: GCG iterations actually executed (< ``n_steps`` if it
            stopped early on ``reached_fn``).
        reached: Whether ``reached_fn`` reported the target reached (always
            ``False`` when no ``reached_fn`` is supplied).
    """

    best_suffix_ids: tuple[int, ...]
    best_loss: float
    loss_history: tuple[float, ...]
    n_steps_run: int
    reached: bool


def run_gcg(
    fn: LossGradientFn,
    cfg: GcgConfig,
    *,
    reached_fn: Callable[[np.ndarray], bool] | None = None,
    on_step: OnStep | None = None,
) -> GcgResult:
    """Run the GCG suffix search against the ``fn`` seam under ``cfg``.

    Each step: one-hot token gradient → per-position ``top_k`` → ``search_width``
    single-position-swap candidates → evaluate their loss → keep the incumbent
    unless a candidate strictly improves it (so ``loss_history`` is non-increasing
    — D6-3's keep-best bookkeeping). Stops early once ``reached_fn`` (if given)
    reports the target reached.

    Args:
        fn: The loss/gradient seam (toy locally, OpenVLA on the GPU).
        cfg: Pinned search hyper-parameters.
        reached_fn: Optional predicate on the current best suffix ids; when it
            returns ``True`` the search stops and ``reached`` is set.
        on_step: Optional :data:`OnStep` callback invoked once after each completed
            step with ``(step, best_suffix_ids_copy, best_loss)`` — including the
            early-stop step. Exception-isolated (a failure is logged to stderr and
            the search continues); ``None`` leaves behaviour identical.

    Returns:
        A :class:`GcgResult` with the best suffix, its loss, and the trajectory.

    Raises:
        ValueError: If ``cfg.top_k`` exceeds ``fn.vocab_size``.
    """
    if cfg.top_k > fn.vocab_size:
        raise ValueError(f"top_k ({cfg.top_k}) must be <= vocab_size ({fn.vocab_size})")

    rng = np.random.default_rng(cfg.seed)
    suffix = np.asarray(fn.init_suffix_ids())
    best_loss = float(fn.loss_of(suffix[None, :])[0])
    history = [best_loss]
    reached = bool(reached_fn(suffix)) if reached_fn is not None else False

    steps_run = 0
    if not reached:
        for _ in range(cfg.n_steps):
            steps_run += 1
            grad = fn.token_gradient(suffix)
            topk = top_k_candidates(grad, cfg.top_k)
            cands = sample_candidates(suffix, topk, cfg.search_width, rng)
            cand, cand_loss = select_best(fn.loss_of(cands), cands)
            if cand_loss < best_loss:
                suffix = cand
                best_loss = cand_loss
            history.append(best_loss)
            reached = reached_fn is not None and reached_fn(suffix)
            if on_step is not None:
                try:
                    on_step(steps_run, suffix.copy(), best_loss)
                except Exception as exc:  # noqa: BLE001 - callback is auxiliary, never fatal
                    print(
                        f"[run_gcg] on_step callback failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            if reached:
                break

    return GcgResult(
        best_suffix_ids=tuple(int(x) for x in suffix),
        best_loss=best_loss,
        loss_history=tuple(history),
        n_steps_run=steps_run,
        reached=reached,
    )

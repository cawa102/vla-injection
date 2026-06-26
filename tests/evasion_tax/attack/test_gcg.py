"""Tests for the model-free GCG search core (step-6 Task 1).

The pure-NumPy GCG loop behind the :class:`LossGradientFn` seam: one-hot top-k →
single-position candidate sampling → best-by-loss selection → keep-best
bookkeeping. The core never imports torch (the real model loss/gradient is the
GPU-only :mod:`evasion_tax.attack.gcg_openvla` seam); these tests exercise it on a
**toy** ``LossGradientFn`` with a known optimum, mirroring how
``test_idealized_frontier`` drives the model-free attacker against
``SyntheticDynamics``.
"""

from pathlib import Path

import numpy as np
import pytest

from evasion_tax.attack.gcg import (
    GcgConfig,
    GcgResult,
    LossGradientFn,
    run_gcg,
    sample_candidates,
    select_best,
    top_k_candidates,
)


class HammingTarget:
    """Toy :class:`LossGradientFn` with a known optimum (D6-2).

    Loss = Hamming distance of a candidate suffix to a hidden ``secret``; the
    one-hot gradient at each position points (most negative) at the secret token.
    A faithful GCG loop must therefore converge to ``secret`` (loss 0).
    """

    def __init__(self, secret, vocab_size):
        self._secret = np.asarray(secret, dtype=int)
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def init_suffix_ids(self) -> np.ndarray:
        # All-zeros start; secret tokens are all non-zero so Hamming starts at L.
        return np.zeros(self._secret.shape[0], dtype=int)

    def token_gradient(self, suffix_ids: np.ndarray) -> np.ndarray:
        grad = np.zeros((self._secret.shape[0], self._vocab_size), dtype=float)
        for i, tok in enumerate(self._secret):
            grad[i, tok] = -1.0  # most negative at the secret token for position i
        return grad

    def loss_of(self, candidate_suffixes: np.ndarray) -> np.ndarray:
        return (candidate_suffixes != self._secret).sum(axis=1).astype(float)


_SECRET = (3, 7, 1, 5)  # no zeros → init Hamming distance = 4


def _hamming_fn():
    return HammingTarget(_SECRET, vocab_size=10)


# --------------------------------------------------------------------------- #
# LossGradientFn seam                                                         #
# --------------------------------------------------------------------------- #


def test_hamming_target_is_a_loss_gradient_fn():
    assert isinstance(_hamming_fn(), LossGradientFn)


# --------------------------------------------------------------------------- #
# torch-free core (D6-2): the search algorithm must not depend on torch        #
# --------------------------------------------------------------------------- #


def test_gcg_core_module_imports_no_torch():
    import ast

    import evasion_tax.attack.gcg as gcg_mod

    assert gcg_mod.__file__ is not None
    source = Path(gcg_mod.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


def _cfg(**overrides):
    base = dict(suffix_len=4, n_steps=10, top_k=8, search_width=16, seed=0)
    base.update(overrides)
    return GcgConfig(**base)


# --------------------------------------------------------------------------- #
# GcgConfig validation                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["suffix_len", "n_steps", "top_k", "search_width"])
def test_config_rejects_non_positive_fields(field):
    with pytest.raises(ValueError):
        _cfg(**{field: 0})


# --------------------------------------------------------------------------- #
# top_k_candidates: most-negative gradient tokens per position                #
# --------------------------------------------------------------------------- #


def test_top_k_candidates_returns_most_negative_tokens_per_position():
    grad = np.array(
        [
            [0.5, -2.0, 0.1, -1.0, 3.0],  # pos 0: most negative ids = 1 (-2.0), 3 (-1.0)
            [1.0, 2.0, 0.0, -0.5, -3.0],  # pos 1: most negative ids = 4 (-3.0), 3 (-0.5)
        ]
    )
    out = top_k_candidates(grad, top_k=2)
    assert out.tolist() == [[1, 3], [4, 3]]


# --------------------------------------------------------------------------- #
# sample_candidates: width + single-position swap from the per-position top-k  #
# --------------------------------------------------------------------------- #


def test_sample_candidates_respects_width_and_swaps_one_position():
    cur = np.array([10, 11, 12, 13])
    # Each position's allowed replacement tokens (disjoint from cur so a swap shows).
    topk = np.array([[20, 21], [22, 23], [24, 25], [26, 27]])
    rng = np.random.default_rng(0)

    cands = sample_candidates(cur, topk, width=32, rng=rng)

    assert cands.shape == (32, cur.shape[0])
    for cand in cands:
        diff = np.flatnonzero(cand != cur)
        assert diff.size <= 1  # at most one position mutated per candidate
        if diff.size == 1:
            pos = int(diff[0])
            assert cand[pos] in topk[pos].tolist()  # swapped token came from that pos' top-k


# --------------------------------------------------------------------------- #
# select_best: argmin candidate                                               #
# --------------------------------------------------------------------------- #


def test_select_best_returns_min_loss_candidate():
    cands = np.array([[1, 1], [2, 2], [3, 3]])
    losses = np.array([3.0, 1.0, 2.0])

    best_cand, best_loss = select_best(losses, cands)

    assert best_cand.tolist() == [2, 2]
    assert best_loss == 1.0


# --------------------------------------------------------------------------- #
# run_gcg: keep-best bookkeeping → non-increasing loss history                 #
# --------------------------------------------------------------------------- #


def test_run_gcg_loss_history_is_non_increasing():
    cfg = GcgConfig(suffix_len=4, n_steps=20, top_k=3, search_width=16, seed=1)
    result = run_gcg(_hamming_fn(), cfg)

    assert isinstance(result, GcgResult)
    history = result.loss_history
    assert len(history) >= 1
    assert all(b <= a for a, b in zip(history, history[1:], strict=False))  # never regresses
    assert result.best_loss == history[-1]


def test_run_gcg_converges_to_secret_and_sets_reached():
    fn = _hamming_fn()
    cfg = GcgConfig(suffix_len=4, n_steps=200, top_k=3, search_width=32, seed=1)
    reached_fn = lambda ids: bool((ids == np.asarray(_SECRET)).all())  # noqa: E731

    result = run_gcg(fn, cfg, reached_fn=reached_fn)

    assert result.reached is True
    assert result.best_loss == 0.0
    assert result.best_suffix_ids == _SECRET
    assert result.n_steps_run <= cfg.n_steps  # stopped as soon as the secret was found


def test_run_gcg_reached_is_false_without_reached_fn():
    cfg = GcgConfig(suffix_len=4, n_steps=200, top_k=3, search_width=32, seed=1)
    result = run_gcg(_hamming_fn(), cfg)
    # No reached predicate ⇒ the core cannot claim the target was reached.
    assert result.reached is False


def test_run_gcg_is_deterministic_for_a_fixed_seed():
    cfg = GcgConfig(suffix_len=4, n_steps=15, top_k=3, search_width=8, seed=123)
    a = run_gcg(_hamming_fn(), cfg)
    b = run_gcg(_hamming_fn(), cfg)
    assert a == b


def test_run_gcg_rejects_top_k_above_vocab_size():
    fn = _hamming_fn()  # vocab_size = 10
    cfg = GcgConfig(suffix_len=4, n_steps=5, top_k=11, search_width=8, seed=0)
    with pytest.raises(ValueError):
        run_gcg(fn, cfg)


# --------------------------------------------------------------------------- #
# on_step callback: observability foothold (exception-isolated)                #
# --------------------------------------------------------------------------- #


def test_on_step_reports_each_step_with_matching_loss_history():
    cfg = GcgConfig(suffix_len=4, n_steps=12, top_k=3, search_width=16, seed=1)
    calls: list[tuple[int, float]] = []

    result = run_gcg(
        _hamming_fn(),
        cfg,
        on_step=lambda step, ids, loss: calls.append((step, loss)),
    )

    steps = [step for step, _ in calls]
    losses = [loss for _, loss in calls]
    assert steps == list(range(1, cfg.n_steps + 1))  # 1-based, once per completed step
    assert losses == list(result.loss_history[1:])  # incumbent loss after each step


def test_on_step_receives_defensive_copy_that_cannot_corrupt_the_search():
    cfg = GcgConfig(suffix_len=4, n_steps=10, top_k=3, search_width=16, seed=1)

    def vandal(step, ids, loss):
        ids[:] = 999  # try to corrupt the live search via the callback's array

    baseline = run_gcg(_hamming_fn(), cfg)
    result = run_gcg(_hamming_fn(), cfg, on_step=vandal)

    assert result == baseline  # the search is unaffected ⇒ callback got a copy


def test_on_step_exception_is_isolated_and_never_aborts_the_search():
    cfg = GcgConfig(suffix_len=4, n_steps=10, top_k=3, search_width=16, seed=1)

    def boom(step, ids, loss):
        raise RuntimeError("simulated logging/disk failure")

    baseline = run_gcg(_hamming_fn(), cfg)
    result = run_gcg(_hamming_fn(), cfg, on_step=boom)

    assert result.n_steps_run == cfg.n_steps  # ran the full budget despite every call raising
    assert result == baseline  # a failing callback changes nothing about the search


def test_on_step_called_once_per_step_through_the_reaching_step():
    fn = _hamming_fn()
    cfg = GcgConfig(suffix_len=4, n_steps=200, top_k=3, search_width=32, seed=1)
    reached_fn = lambda ids: bool((ids == np.asarray(_SECRET)).all())  # noqa: E731
    steps: list[int] = []

    result = run_gcg(
        fn, cfg, reached_fn=reached_fn, on_step=lambda step, ids, loss: steps.append(step)
    )

    # Early-stop fires at result.n_steps_run; the reaching step is itself reported.
    assert result.reached is True
    assert steps == list(range(1, result.n_steps_run + 1))


def test_on_step_none_is_identical_to_omitting_the_callback():
    cfg = GcgConfig(suffix_len=4, n_steps=15, top_k=3, search_width=8, seed=123)
    assert run_gcg(_hamming_fn(), cfg, on_step=None) == run_gcg(_hamming_fn(), cfg)

"""Tests for the OpenVLA GCG seam — the **pure**, off-GPU pieces only (Task 2).

The real loss/gradient against bf16 OpenVLA-7B is GPU-only and guarded; here we
pin the model-free helpers the seam is built from — the one-hot gradient
projection and the suffix-span locator — plus the structural invariant that the
module imports **no torch at module top** (so it stays importable on the mac, like
``smoke_openvla_gradient.py``). The on-box seam-faithfulness gate (D6-9,
finite-difference vs token swaps) and ``decode_span`` are exercised on the A5000.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from evasion_tax.attack.gcg_openvla import (
    EquivalenceCheck,
    _to_pil,
    chunked_losses,
    equivalence_verdict,
    per_sequence_ce,
    project_onehot_grad,
    suffix_span_in_ids,
)


def test_to_pil_wraps_numpy_and_passes_pil_through():
    """`_to_pil` normalizes the rollout obs (numpy HWC uint8) to a PIL image so the
    OpenVLA processor's `img.convert("RGB")` works, and passes a PIL image through
    unchanged — the run_attack numpy-image regression that crashed the M1 attack."""
    Image = pytest.importorskip("PIL.Image")

    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    wrapped = _to_pil(arr)
    assert isinstance(wrapped, Image.Image)
    assert wrapped.size == (4, 4)

    pil = Image.fromarray(arr)
    assert _to_pil(pil) is pil

# --------------------------------------------------------------------------- #
# per_sequence_ce: HF-style shifted, ignore-masked, per-row MEAN cross-entropy  #
# (the contract the GPU true-batch `loss_of` mirrors on `out.logits`)           #
# --------------------------------------------------------------------------- #


def test_per_sequence_ce_matches_hand_computed_mean():
    # B=1, T=3, V=2. Causal shift predicts labels[:, 1:] from logits[:, :-1], so the
    # last logit row and the first label are dropped; positions 0,1 contribute.
    logits = np.array([[[2.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])  # [1, 3, 2]
    labels = np.array([[-100, 0, 1]])  # [1, 3]; labels[0,0] dropped by the shift

    out = per_sequence_ce(logits, labels)

    # CE_t = logaddexp(l0, l1) - l_true:
    #   pos0: logits [2,0], true=0 -> log(e^2+1) - 2     = 0.12692801
    #   pos1: logits [0,1], true=1 -> log(1+e) - 1       = 0.31326169
    # reduction='mean' over the 2 valid positions -> 0.22009485 (sum would be 0.44).
    assert out.shape == (1,)
    assert out[0] == pytest.approx(0.22009485, abs=1e-6)


def test_per_sequence_ce_fully_ignored_row_is_zero_without_nan():
    # Row 0 has a valid target (the hand-computed row); row 1 is fully ignore-masked.
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )  # [2, 3, 2]
    labels = np.array([[-100, 0, 1], [-100, -100, -100]])  # row 1: every position masked

    out = per_sequence_ce(labels=labels, logits=logits)

    assert not np.isnan(out).any()  # no NaN leak from the 0/0 row
    assert out[0] == pytest.approx(0.22009485, abs=1e-6)  # valid row unaffected
    assert out[1] == 0.0  # fully-ignored row documented as 0.0


def test_per_sequence_ce_shift_drops_last_logit_and_first_label():
    # Regression guard for the causal shift: corrupting the LAST logit position and the
    # FIRST label must NOT change the result — both are dropped by predict-t-from-t-1.
    logits = np.array([[[2.0, 0.0], [0.0, 1.0], [1000.0, -1000.0]]])  # pos 2 corrupted
    labels = np.array([[7, 0, 1]])  # labels[0,0]=7 corrupted (out-of-vocab, dropped by shift)

    out = per_sequence_ce(logits, labels)

    assert out[0] == pytest.approx(0.22009485, abs=1e-6)  # identical to the un-corrupted row


def test_per_sequence_ce_single_row_equals_its_batched_row():
    # Regression guard: each row is reduced independently, so a single-row slice equals
    # that row inside a batch (the "single-row == scalar HF-style mean" property).
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[1.0, -1.0], [3.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        ]
    )  # [3, 3, 2]
    labels = np.array([[-100, 0, 1], [-100, 1, 0], [0, 1, -100]])

    batched = per_sequence_ce(logits, labels)

    for i in range(logits.shape[0]):
        single = per_sequence_ce(logits[i : i + 1], labels[i : i + 1])
        assert single[0] == pytest.approx(batched[i], abs=1e-12)

# --------------------------------------------------------------------------- #
# equivalence_verdict: absolute closeness + SELECTION REGRET (the bf16-robust    #
# refinement of "rank order" Codex flagged — see the 2026-06-22 measurement)     #
# --------------------------------------------------------------------------- #


def test_equivalence_verdict_identical_vectors_pass():
    losses = np.array([1.0, 0.5, 2.0])

    chk = equivalence_verdict(losses, losses.copy())

    assert isinstance(chk, EquivalenceCheck)
    assert chk.n == 3
    assert chk.max_abs_diff == 0.0
    assert chk.allclose
    assert chk.argmin_match
    assert chk.selection_regret == 0.0
    assert chk.regret_ok
    assert chk.passed


def test_equivalence_verdict_within_atol_same_argmin_pass():
    single = np.array([1.0, 0.5, 2.0])
    batched = np.array([1.0004, 0.4996, 2.0003])  # < 1e-3 off; argmin still index 1

    chk = equivalence_verdict(single, batched, atol=1e-3)

    assert chk.allclose
    assert chk.argmin_match
    assert chk.selection_regret == pytest.approx(0.0)
    assert chk.passed


def test_equivalence_verdict_near_tied_argmin_flip_passes_on_low_regret():
    # The bf16 reality (2026-06-22): two near-tied candidates can flip which is argmin between
    # the batched and single forwards. The flip selects an EQUALLY GOOD candidate (tiny regret),
    # so it must PASS — strict index equality would wrongly fail it.
    ref = np.array([1.00, 1.05, 5.0])  # true best = index 0
    cmp = np.array([1.04, 1.01, 5.0])  # argmin flips to index 1

    chk = equivalence_verdict(ref, cmp, atol=1.0, regret_tol=0.1)

    assert not chk.argmin_match  # index flipped (diagnostic)
    assert chk.selection_regret == pytest.approx(0.05)  # ref[1] - min(ref) = 1.05 - 1.00
    assert chk.regret_ok  # 0.05 <= 0.1
    assert chk.passed


def test_equivalence_verdict_genuine_misrank_fails_on_regret():
    # A real misrank: cmp selects a candidate that is genuinely worse in the reference losses
    # (large regret), even though the values agree within the (loose) atol.
    ref = np.array([1.0, 1.5])  # true best = index 0
    cmp = np.array([1.4, 1.1])  # argmin = index 1 -> ref[1] = 1.5, regret 0.5

    chk = equivalence_verdict(ref, cmp, atol=1.0, regret_tol=0.1)

    assert chk.allclose  # |ref - cmp| = 0.4 < 1.0
    assert chk.selection_regret == pytest.approx(0.5)
    assert not chk.regret_ok  # 0.5 > 0.1
    assert not chk.passed


def test_equivalence_verdict_beyond_atol_fails():
    ref = np.array([1.0, 2.0, 3.0])
    cmp = np.array([1.0, 2.0, 3.5])

    chk = equivalence_verdict(ref, cmp, atol=1e-3)

    assert not chk.allclose
    assert not chk.passed
    assert chk.max_abs_diff == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),  # shape mismatch
        (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])),  # not 1-D
        (np.array([]), np.array([])),  # empty
    ],
)
def test_equivalence_verdict_rejects_bad_shapes(a, b):
    with pytest.raises(ValueError):
        equivalence_verdict(a, b)


# --------------------------------------------------------------------------- #
# project_onehot_grad: g[i,v] = (d loss / d e_i) · W[v,:]                       #
# --------------------------------------------------------------------------- #


def test_project_onehot_grad_matches_hand_computed():
    # L=2 suffix positions, d=2 embed dim, V=3 vocab.
    grad_embeds_suffix = np.array([[1.0, 0.0], [0.0, 1.0]])  # [L, d]
    embedding_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [V, d]

    out = project_onehot_grad(grad_embeds_suffix, embedding_matrix)

    # grad @ W.T:  rows dotted with each vocab embedding.
    assert out.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    assert out.shape == (2, 3)  # [L, V]


# --------------------------------------------------------------------------- #
# suffix_span_in_ids: the trailing suffix_len tokens of the prompt head        #
# --------------------------------------------------------------------------- #


def test_suffix_span_locates_trailing_run():
    prompt_ids = np.array([101, 5, 6, 7, 8, 9])  # head: prefix + instruction + suffix
    span = suffix_span_in_ids(prompt_ids, suffix_len=3)

    assert isinstance(span, slice)
    assert prompt_ids[span].tolist() == [7, 8, 9]
    assert (span.start, span.stop) == (3, 6)


@pytest.mark.parametrize("bad_len", [0, 7])
def test_suffix_span_rejects_out_of_range_len(bad_len):
    prompt_ids = np.array([101, 5, 6, 7, 8, 9])  # length 6
    with pytest.raises(ValueError):
        suffix_span_in_ids(prompt_ids, suffix_len=bad_len)


# --------------------------------------------------------------------------- #
# reached(): the run_gcg reached_fn. GPU body is the on-box gate (reached agrees #
# with run_gcg.reached and with loss~0); off-GPU we pin only the call contract.  #
# --------------------------------------------------------------------------- #


def test_reached_signature_matches_run_gcg_reached_fn_contract():
    import inspect

    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget

    # run_gcg expects reached_fn: Callable[[np.ndarray], bool]; target.reached must take
    # exactly one positional arg (the suffix ids) besides self, so callers can pass
    # reached_fn=target.reached directly.
    params = list(inspect.signature(OpenVlaGcgTarget.reached).parameters)
    assert params == ["self", "suffix_ids"]


def test_predict_target_action_ids_signature_matches_transfer_eval_contract():
    import inspect

    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget

    params = list(inspect.signature(OpenVlaGcgTarget.predict_target_action_ids).parameters)
    assert params == ["self", "suffix_ids"]


def test_init_accepts_match_positions_keyword_defaulting_to_none():
    # Task 2: the single-frame reach predicate can be scored on a goal-dims subset
    # (e.g. [0..5] to exclude the gripper). Pinned as a keyword-only param, default
    # None (= all-token behaviour, so existing callers are unaffected).
    import inspect

    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget

    p = inspect.signature(OpenVlaGcgTarget.__init__).parameters["match_positions"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is None
    # The subset-scoring behaviour itself is pinned torch-free in
    # tests/evasion_tax/attack/test_early_stop.py (target_span_argmax_matches with
    # positions=...); reached() threads this param through and is on-box-gated,
    # matching how the rest of the GPU body is validated. Off-GPU tests here stay
    # torch-free so the module's clean-import guard holds.


# --------------------------------------------------------------------------- #
# Guard: torch / transformers / PIL only inside methods, never at module top   #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# chunked_losses: forward [B] rows in chunks of eval_batch, concatenate the     #
# per-chunk losses (DE-7). The pure reassociation is tested off-GPU here; the    #
# numerical/VRAM equivalence on the real bf16 model is the on-box fit-check.     #
# --------------------------------------------------------------------------- #


def _row_sum_forward(calls):
    """Row-independent fake forward (stand-in for the GPU CE): row -> its sum.

    Records each chunk's batch size so the chunk boundaries are observable. Being
    row-independent, it has the exact property the real per-row CE has, so a wrong
    boundary / order / drop is the only thing that can make chunked != single.
    """

    def forward(rows):
        rows = np.asarray(rows)
        calls.append(rows.shape[0])
        return rows.sum(axis=1).astype(float)

    return forward


@pytest.mark.parametrize("eval_batch", [1, 8, 32])
@pytest.mark.parametrize("n", [1, 8, 32, 50])
@pytest.mark.parametrize("width", [20, 25])
def test_chunked_losses_equals_single_forward(eval_batch, n, width):
    # Chunking is a pure reassociation of independent per-row CE: the concatenated [B]
    # equals the single-forward [B] for any chunk size, incl. B not divisible (50/32).
    rng = np.random.default_rng(0)
    full = rng.integers(0, 100, size=(n, width))

    single = chunked_losses(full, None, _row_sum_forward([]))
    chunked = chunked_losses(full, eval_batch, _row_sum_forward([]))

    assert np.array_equal(chunked, single)
    assert int(np.argmin(chunked)) == int(np.argmin(single))  # rank order preserved (DB-4)


def test_chunked_losses_splits_non_divisible_batch_into_full_then_remainder():
    # B=50, eval_batch=32 -> chunks of 32 then 18 (the trailing partial chunk).
    full = np.arange(50 * 4).reshape(50, 4)
    calls: list[int] = []

    chunked_losses(full, 32, _row_sum_forward(calls))

    assert calls == [32, 18]


def test_chunked_losses_none_does_one_forward_over_all_rows():
    # eval_batch=None preserves the single-forward path byte-for-byte: ONE call, all B.
    full = np.arange(40 * 3).reshape(40, 3)
    calls: list[int] = []

    chunked_losses(full, None, _row_sum_forward(calls))

    assert calls == [40]


def test_chunked_losses_preserves_row_order():
    # row i -> value i; the concatenated output must stay in row order across chunks.
    full = np.arange(10).reshape(10, 1)

    out = chunked_losses(full, 3, lambda r: np.asarray(r)[:, 0].astype(float))

    assert out.tolist() == list(range(10))


@pytest.mark.parametrize("bad", [0, -1])
def test_chunked_losses_rejects_non_positive_eval_batch(bad):
    # No silent default: a zero/negative chunk size is a programming error, fail loud.
    full = np.zeros((4, 3))
    with pytest.raises(ValueError):
        chunked_losses(full, bad, _row_sum_forward([]))


def test_seam_module_imports_no_torch_stack_at_module_top():
    import evasion_tax.attack.gcg_openvla as seam_mod

    assert seam_mod.__file__ is not None
    tree = ast.parse(Path(seam_mod.__file__).read_text())
    top_level_imports: set[str] = set()
    for node in tree.body:  # module-body statements only (not nested in functions)
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert {"torch", "transformers", "PIL"}.isdisjoint(top_level_imports)

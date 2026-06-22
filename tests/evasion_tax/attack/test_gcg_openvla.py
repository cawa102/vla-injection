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
    equivalence_verdict,
    per_sequence_ce,
    project_onehot_grad,
    suffix_span_in_ids,
)

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
# equivalence_verdict: batched-vs-single (and run-vs-run) closeness + argmin     #
# agreement — the necessary-not-sufficient DB-4 hardening Codex flagged          #
# --------------------------------------------------------------------------- #


def test_equivalence_verdict_identical_vectors_pass():
    losses = np.array([1.0, 0.5, 2.0])

    chk = equivalence_verdict(losses, losses.copy())

    assert isinstance(chk, EquivalenceCheck)
    assert chk.n == 3
    assert chk.max_abs_diff == 0.0
    assert chk.allclose
    assert chk.argmin_match
    assert chk.passed


def test_equivalence_verdict_within_atol_same_argmin_pass():
    single = np.array([1.0, 0.5, 2.0])
    batched = np.array([1.0004, 0.4996, 2.0003])  # < 1e-3 off; argmin still index 1

    chk = equivalence_verdict(single, batched, atol=1e-3)

    assert chk.allclose
    assert chk.argmin_match
    assert chk.passed


def test_equivalence_verdict_close_values_but_swapped_argmin_fails():
    # The DB-4 point: absolute closeness is necessary-NOT-sufficient. These two vectors
    # agree within atol but pick DIFFERENT best candidates → GCG would select differently.
    a = np.array([1.000, 1.001, 5.0])  # argmin 0
    b = np.array([1.001, 1.000, 5.0])  # argmin 1

    chk = equivalence_verdict(a, b, atol=1e-2)

    assert chk.allclose  # within 1e-2
    assert not chk.argmin_match  # but rank order disagrees
    assert not chk.passed


def test_equivalence_verdict_beyond_atol_fails():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.5])

    chk = equivalence_verdict(a, b, atol=1e-3)

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
# Guard: torch / transformers / PIL only inside methods, never at module top   #
# --------------------------------------------------------------------------- #


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

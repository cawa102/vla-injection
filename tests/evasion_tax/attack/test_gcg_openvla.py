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

from evasion_tax.attack.gcg_openvla import project_onehot_grad, suffix_span_in_ids

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

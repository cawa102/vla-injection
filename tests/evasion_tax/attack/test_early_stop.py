"""Tests for the pure RoboGCG-style forced-decode success predicate (bench Task 1).

``target_span_argmax_matches`` is the calibration-free "target reached" check the GCG
early-stop uses (decision DE-1): for every non-ignored target-span position the greedy
argmax decode must equal the target id. It mirrors the causal shift in
:func:`per_sequence_ce` so the GPU ``OpenVlaGcgTarget.reached()`` agrees with the loss
seam. Pure NumPy — exercised entirely off-GPU here.
"""

from __future__ import annotations

import numpy as np
import pytest

from evasion_tax.attack.early_stop import target_span_argmax_matches
from evasion_tax.attack.gcg_openvla import per_sequence_ce


def test_all_non_ignored_positions_argmax_match_returns_true():
    # [T=3, V=2]; causal shift predicts labels[1:] from logits[:-1].
    #   pos0: logits [2,0] -> argmax 0 == labels[1]=0
    #   pos1: logits [0,1] -> argmax 1 == labels[2]=1
    # last logit row + first label dropped by the shift (matches per_sequence_ce).
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    labels = np.array([-100, 0, 1])

    assert target_span_argmax_matches(logits, labels) is True


def test_one_non_ignored_mismatch_returns_false():
    # pos1's argmax (logits [0,1] -> 1) disagrees with labels[2]=0: one mismatch is enough.
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    labels = np.array([-100, 0, 0])

    assert target_span_argmax_matches(logits, labels) is False


def test_fully_ignored_labels_is_vacuously_true_without_crash():
    # No non-ignored target position => nothing can fail => vacuously True (documented).
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    labels = np.array([-100, -100, -100])

    assert target_span_argmax_matches(logits, labels) is True


def test_causal_shift_drops_last_logit_and_first_label():
    # Regression guard for the shift (mirrors per_sequence_ce's): corrupting the LAST
    # logit row and the FIRST label must not change the verdict -- both are dropped by
    # predict-t-from-t-1. A non-shifting impl would read the corrupted positions.
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [-1000.0, 1000.0]])  # pos2 corrupted
    labels = np.array([7, 0, 1])  # labels[0]=7 corrupted (dropped by the shift)

    assert target_span_argmax_matches(logits, labels) is True


def test_single_non_ignored_position_matches():
    # The minimal case: exactly one scored position (predicted from the row before it).
    logits = np.array([[0.0, 0.0], [5.0, 1.0]])  # [T=2, V=2]
    labels = np.array([-100, 0])  # only pos0 (predicted from logits[0]) is scored

    assert target_span_argmax_matches(logits, labels) is True
    assert target_span_argmax_matches(logits, np.array([-100, 1])) is False


def test_ce_near_zero_sequence_also_argmax_matches():
    # Cross-link with per_sequence_ce (DE-1 sanity): a sequence whose logits are sharply
    # peaked on the labels has CE ~ 0 AND must argmax-match -- the two seams agree.
    logits = np.array([[20.0, 0.0], [0.0, 20.0], [9.0, 9.0]])  # [3, 2], peaked on labels
    labels = np.array([-100, 0, 1])

    ce = per_sequence_ce(logits[None, :, :], labels[None, :])[0]
    assert ce == pytest.approx(0.0, abs=1e-6)
    assert target_span_argmax_matches(logits, labels) is True

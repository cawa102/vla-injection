"""Tests for the L1 confound-control scaffolding (Codex review #2 #11).

Pre-registered controls that must pass *before* an L1 "internal-rep" win/lose is
reportable (playbook §4b-(I)). These are the **model-free** primitives the M2
analysis runs:

* :func:`shuffle_labels` — the label-shuffle control input (deterministic,
  multiset-preserving) so the probe can be retrained on permuted labels;
* :func:`probe_auc` — the shared AUC comparator (reuses ``eval.metrics.roc_auc``)
  every control reports on.

The headline control pinned here: a probe fit on **shuffled** labels collapses to
chance AUC, so a real L1 win cannot be a task-prior / suffix-lexical / target-leak
artifact.
"""

import pytest

from evasion_tax.metric.probe_confounds import probe_auc, shuffle_labels
from evasion_tax.metric.probe_internal import InternalProbe, SyntheticActivationExtractor

from .test_probe_internal import labeled_features

# --------------------------------------------------------------------------- #
# shuffle_labels                                                              #
# --------------------------------------------------------------------------- #


def test_shuffle_labels_preserves_multiset_and_is_deterministic():
    labels = [0, 0, 0, 1, 1, 1]
    a = shuffle_labels(labels, seed=7)
    b = shuffle_labels(labels, seed=7)
    assert a == b  # deterministic for a fixed seed
    assert sorted(a) == sorted(labels)  # same class balance, only the order changes


def test_shuffle_labels_changes_order_and_varies_with_seed():
    labels = list(range(2)) * 50  # 100 labels, balanced
    s7 = shuffle_labels(labels, seed=7)
    s8 = shuffle_labels(labels, seed=8)
    assert s7 != labels  # actually permuted
    assert s7 != s8  # seed-dependent


def test_shuffle_labels_does_not_mutate_input():
    labels = [0, 1, 1, 0]
    before = list(labels)
    shuffle_labels(labels, seed=1)
    assert labels == before


# --------------------------------------------------------------------------- #
# probe_auc comparator                                                         #
# --------------------------------------------------------------------------- #


def test_probe_auc_is_high_for_a_separable_probe():
    ext = SyntheticActivationExtractor(dim=16, signal=2.5)
    train_f, train_y = labeled_features(ext, n_per_class=80, base_seed=0)
    test_f, test_y = labeled_features(ext, n_per_class=80, base_seed=10_000)
    probe = InternalProbe.fit(train_f, train_y)
    assert probe_auc(probe, test_f, test_y) > 0.9


def test_probe_auc_rejects_single_class_labels():
    ext = SyntheticActivationExtractor()
    train_f, train_y = labeled_features(ext, n_per_class=20, base_seed=0)
    probe = InternalProbe.fit(train_f, train_y)
    feats = train_f[:5]
    with pytest.raises(ValueError):
        probe_auc(probe, feats, [1, 1, 1, 1, 1])


def test_probe_auc_rejects_length_mismatch():
    ext = SyntheticActivationExtractor()
    train_f, train_y = labeled_features(ext, n_per_class=20, base_seed=0)
    probe = InternalProbe.fit(train_f, train_y)
    with pytest.raises(ValueError):
        probe_auc(probe, train_f[:4], [0, 1])


# --------------------------------------------------------------------------- #
# Label-shuffle control: collapses to chance                                   #
# --------------------------------------------------------------------------- #


def test_label_shuffle_control_collapses_probe_to_chance():
    ext = SyntheticActivationExtractor(dim=16, signal=2.5)
    train_f, train_y = labeled_features(ext, n_per_class=120, base_seed=0)
    test_f, test_y = labeled_features(ext, n_per_class=120, base_seed=10_000)

    # The honest probe separates; the same probe trained on shuffled labels must
    # not — its held-out AUC sits at chance (~0.5), evaluated against TRUE labels.
    honest_auc = probe_auc(InternalProbe.fit(train_f, train_y), test_f, test_y)
    shuffled = shuffle_labels(train_y, seed=123)
    chance_auc = probe_auc(InternalProbe.fit(train_f, shuffled), test_f, test_y)

    assert honest_auc > 0.9
    assert 0.35 < chance_auc < 0.65

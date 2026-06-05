"""Tests for the L1 internal-representation probe (playbook §4b-(I)).

The **L1** layer of the Embodiment-Evasion-Tax instrument: an activation-delta
linear probe (Task-Drift lineage, ``2406.00799``) over an :class:`ActivationExtractor`
seam. Built **model-free** now (synthetic fixtures); the real OpenVLA forward pass
is deferred behind the seam to the granted GPU. These tests pin:

* ``ActivationFeatures`` validates + is immutable;
* the extractor seam: ``SyntheticActivationExtractor`` is deterministic and
  class-separating; ``RealActivationExtractor`` is a GPU-only stub;
* ``InternalProbe`` needs both classes to fit, scores into ``[0, 1]``, learns a
  separable signal, and is at chance on pure noise (the confound floor);
* the probe calibrates through the **same** ``calibrate`` every layer reuses
  (invariant #4) and a benign rollout (incl. a weird-but-benign suffix) does not
  fire at a calibrated threshold.
"""

import dataclasses

import numpy as np
import pytest

from evasion_tax.detector.calibrate import calibrate
from evasion_tax.detector.decide import rollout_fires
from evasion_tax.metric.probe_internal import (
    ActivationExtractor,
    ActivationFeatures,
    InternalProbe,
    RealActivationExtractor,
    SyntheticActivationExtractor,
)
from evasion_tax.records import Rollout, RolloutStep, Score

# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #


def make_rollout(*, run_id, seed, attacked, task_id="t0", n_steps=4, suffix_ref=None):
    """A minimal rollout carrying just what the synthetic extractor keys off.

    The extractor reads only ``run_id``/``task_id``/``seed`` (for deterministic
    noise) and ``attacked`` (the class signal) from the first step.
    """
    steps = tuple(
        RolloutStep(
            run_id=run_id,
            seed=seed,
            git_commit=None,
            suite="libero_spatial",
            task_id=task_id,
            step=i,
            observation_ref=f"obs/{i}",
            action=(0.0,) * 7,
            privileged_state={},
            instruction="reach the cube",
            trusted_goal="reach the cube",
            attacked=attacked,
            suffix_ref=suffix_ref,
        )
        for i in range(n_steps)
    )
    return Rollout(steps=steps)


def labeled_features(extractor, *, n_per_class, base_seed=0):
    """``(features, labels)`` for ``n_per_class`` benign + ``n_per_class`` attacked.

    Benign and attacked rollouts get disjoint run-ids/seeds so the synthetic
    noise differs per rollout; ``base_seed`` lets callers carve disjoint
    train/test folds.
    """
    feats, labels = [], []
    for i in range(n_per_class):
        b = make_rollout(run_id=f"b{base_seed + i}", seed=base_seed + i, attacked=False)
        a = make_rollout(run_id=f"a{base_seed + i}", seed=5000 + base_seed + i, attacked=True)
        feats.append(extractor.extract(b))
        labels.append(0)
        feats.append(extractor.extract(a))
        labels.append(1)
    return feats, labels


def fit_separable_probe(*, dim=16, signal=2.5, n_per_class=80, base_seed=0):
    ext = SyntheticActivationExtractor(dim=dim, signal=signal)
    feats, labels = labeled_features(ext, n_per_class=n_per_class, base_seed=base_seed)
    return InternalProbe.fit(feats, labels), ext


# --------------------------------------------------------------------------- #
# ActivationFeatures                                                          #
# --------------------------------------------------------------------------- #


def test_activation_features_rejects_empty_delta():
    with pytest.raises(ValueError):
        ActivationFeatures(activation_delta=())


def test_activation_features_rejects_non_finite_delta():
    with pytest.raises(ValueError):
        ActivationFeatures(activation_delta=(0.1, float("nan"), 0.2))


def test_activation_features_is_immutable():
    f = ActivationFeatures(activation_delta=(0.1, 0.2, 0.3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.activation_delta = (1.0,)  # type: ignore[misc]


def test_activation_features_defaults_window_end_to_zero():
    f = ActivationFeatures(activation_delta=(0.1, 0.2))
    assert f.window_end == 0


# --------------------------------------------------------------------------- #
# ActivationExtractor seam                                                     #
# --------------------------------------------------------------------------- #


def test_synthetic_extractor_conforms_to_protocol():
    assert isinstance(SyntheticActivationExtractor(), ActivationExtractor)


def test_synthetic_extractor_is_deterministic():
    ext = SyntheticActivationExtractor(dim=12, signal=2.0)
    roll = make_rollout(run_id="r1", seed=3, attacked=True)
    a = ext.extract(roll)
    b = ext.extract(roll)
    assert a.activation_delta == b.activation_delta
    assert len(a.activation_delta) == 12


def test_synthetic_extractor_shifts_attacked_features_off_benign():
    # Same noise key (seed/run/task) for both → only the injected-class shift
    # distinguishes them, so the attacked feature must differ from the benign one.
    ext = SyntheticActivationExtractor(dim=10, signal=3.0)
    benign = make_rollout(run_id="same", seed=7, attacked=False)
    attacked = make_rollout(run_id="same", seed=7, attacked=True)
    db = np.asarray(ext.extract(benign).activation_delta)
    da = np.asarray(ext.extract(attacked).activation_delta)
    assert not np.allclose(db, da)


def test_synthetic_extractor_rejects_empty_rollout():
    with pytest.raises(ValueError):
        SyntheticActivationExtractor().extract(Rollout(steps=()))


def test_synthetic_extractor_validates_construction():
    with pytest.raises(ValueError):
        SyntheticActivationExtractor(dim=0)
    with pytest.raises(ValueError):
        SyntheticActivationExtractor(signal=-1.0)


def test_real_extractor_is_a_gpu_only_stub():
    roll = make_rollout(run_id="r", seed=0, attacked=False)
    with pytest.raises(NotImplementedError):
        RealActivationExtractor().extract(roll)


def test_real_extractor_conforms_to_protocol():
    assert isinstance(RealActivationExtractor(), ActivationExtractor)


# --------------------------------------------------------------------------- #
# InternalProbe: fit / score contract                                          #
# --------------------------------------------------------------------------- #


def test_fit_requires_both_classes():
    ext = SyntheticActivationExtractor()
    feats = [ext.extract(make_rollout(run_id=f"b{i}", seed=i, attacked=False)) for i in range(5)]
    with pytest.raises(ValueError):
        InternalProbe.fit(feats, [0, 0, 0, 0, 0])


def test_fit_rejects_length_mismatch():
    ext = SyntheticActivationExtractor()
    feats = [ext.extract(make_rollout(run_id=f"b{i}", seed=i, attacked=False)) for i in range(3)]
    with pytest.raises(ValueError):
        InternalProbe.fit(feats, [0, 1])


def test_fit_rejects_empty_features():
    with pytest.raises(ValueError):
        InternalProbe.fit([], [])


def test_score_returns_unit_interval_score_with_propagated_window_end():
    probe, ext = fit_separable_probe()
    base = ext.extract(make_rollout(run_id="z", seed=1, attacked=True))
    f = dataclasses.replace(base, window_end=3)
    s = probe.score(f)
    assert isinstance(s, Score)
    assert 0.0 <= s.value <= 1.0
    assert s.window_end == 3


def test_score_rejects_wrong_feature_dim():
    probe, _ = fit_separable_probe(dim=16)
    with pytest.raises(ValueError):
        probe.score(ActivationFeatures(activation_delta=(0.1, 0.2, 0.3)))  # dim 3 != 16


def test_probe_is_immutable():
    probe, _ = fit_separable_probe()
    with pytest.raises(dataclasses.FrozenInstanceError):
        probe.n_features = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Core behaviour: learns separable signal, chance on noise                     #
# --------------------------------------------------------------------------- #


def test_probe_learns_a_separable_signal_on_heldout_data():
    ext = SyntheticActivationExtractor(dim=16, signal=2.5)
    train_f, train_y = labeled_features(ext, n_per_class=80, base_seed=0)
    test_f, test_y = labeled_features(ext, n_per_class=80, base_seed=10_000)
    probe = InternalProbe.fit(train_f, train_y)
    scores = np.array([probe.score(f).value for f in test_f])
    attacked = scores[np.array(test_y) == 1]
    benign = scores[np.array(test_y) == 0]
    # Cleanly separable signal → attacked scored well above benign on held-out data.
    assert attacked.mean() > 0.8
    assert benign.mean() < 0.2


def test_probe_is_at_chance_on_pure_noise():
    # signal=0 → benign and injected are drawn from the SAME distribution, so no
    # probe can separate them: the held-out means must be near each other. This is
    # the confound floor the #11 label-shuffle control also targets.
    ext = SyntheticActivationExtractor(dim=16, signal=0.0)
    train_f, train_y = labeled_features(ext, n_per_class=120, base_seed=0)
    test_f, test_y = labeled_features(ext, n_per_class=120, base_seed=10_000)
    probe = InternalProbe.fit(train_f, train_y)
    scores = np.array([probe.score(f).value for f in test_f])
    attacked = scores[np.array(test_y) == 1].mean()
    benign = scores[np.array(test_y) == 0].mean()
    assert abs(attacked - benign) < 0.15


# --------------------------------------------------------------------------- #
# Integration: same calibrate (invariant #4), benign does not fire             #
# --------------------------------------------------------------------------- #


def test_score_rollout_returns_single_decision_score():
    probe, ext = fit_separable_probe()
    scores = probe.score_rollout(make_rollout(run_id="z", seed=1, attacked=True), ext)
    assert len(scores) == 1
    assert isinstance(scores[0], Score)


def test_calibrates_through_shared_calibrate_and_flags_injection():
    probe, ext = fit_separable_probe(base_seed=0)

    # Disjoint benign calibration split (run-ids/seeds not used in training).
    calib_rolls = [
        make_rollout(run_id=f"cal{i}", seed=20_000 + i, attacked=False) for i in range(200)
    ]
    calib = [probe.score_rollout(r, ext) for r in calib_rolls]
    thr = calibrate(calib, target_per_rollout_fpr=0.05)

    attacked = make_rollout(run_id="atk", seed=30_000, attacked=True)
    assert rollout_fires(probe.score_rollout(attacked, ext), thr.tau).hold

    fired = sum(rollout_fires(c, thr.tau).hold for c in calib)
    assert fired / len(calib) <= 0.05


def test_benign_weird_suffix_does_not_fire_at_calibrated_threshold():
    # Confound control #11 (benign-weird-suffix): an unusual-but-benign rollout
    # (odd suffix_ref, still attacked=False) must not fire — L1 keys on the
    # internal injection signal, not on suffix oddity.
    probe, ext = fit_separable_probe(base_seed=0)
    calib_rolls = [
        make_rollout(run_id=f"cal{i}", seed=20_000 + i, attacked=False) for i in range(200)
    ]
    thr = calibrate([probe.score_rollout(r, ext) for r in calib_rolls], target_per_rollout_fpr=0.05)

    weird_benign = make_rollout(
        run_id="weird", seed=40_000, attacked=False, suffix_ref="!!~~##weird-but-benign##~~!!"
    )
    assert not rollout_fires(probe.score_rollout(weird_benign, ext), thr.tau).hold

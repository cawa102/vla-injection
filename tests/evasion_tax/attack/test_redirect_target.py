"""Tests for the pre-registered low-level redirect target (Task 1, D2 / DM-5).

The redirect target is defined in OpenVLA's **normalized** action space (the
[-1, 1] 256-bin grid where action *tokens* live), so the forced-decode token ids
follow from the verified codec with only ``vocab_size`` (no dataset stats), and
``target_action ∈ region`` holds in one consistent space. The window-scored ASR
region is the same normalized region (scored against the runner's normalized
action stream in Task 2).
"""

import numpy as np
import pytest

from evasion_tax.attack.redirect_target import (
    _BIN_CENTERS,
    action_ids_from_norm,
    amplify_to_directional,
    anchor_action_space,
    anchor_spec_for,
    directional_target_action,
    norm_from_action_ids,
    redirect_spec_for,
    target_action_ids_for,
)
from evasion_tax.policy.action_codec import ActionCodec
from evasion_tax.records import ACTION_DIM

_VOCAB = 32064  # an OpenVLA-like de-tokenisation vocab size (value irrelevant to the logic)


def _passthrough_codec(vocab_size=_VOCAB):
    """A codec whose un-normalisation is identity (mask all-False), so ``decode``
    returns the bin centre directly — the normalized space the redirect lives in."""
    dim = ACTION_DIM
    return ActionCodec(
        q01=(-1.0,) * dim,
        q99=(1.0,) * dim,
        mask=(False,) * dim,
        vocab_size=vocab_size,
        n_bins=256,
    )


def test_redirect_spec_is_deterministic_per_seed():
    a = redirect_spec_for(7, persistence_steps=3)
    b = redirect_spec_for(7, persistence_steps=3)
    assert a == b


def test_redirect_spec_differs_across_seeds():
    a = redirect_spec_for(7, persistence_steps=3)
    c = redirect_spec_for(8, persistence_steps=3)
    assert a.target_action != c.target_action


def test_target_action_lies_inside_region():
    # DM-2 consistency: the forced-decode target is a member of the ASR region.
    spec = redirect_spec_for(11, persistence_steps=5)
    constrained = [spec.target_action[d] for d in spec.region.dims]
    assert spec.region.reached(spec.target_action)
    assert spec.region.persistence_steps == 5
    # region is non-degenerate (a real window, not a point)
    for lo, c, hi in zip(spec.region.low, constrained, spec.region.high, strict=True):
        assert lo <= c <= hi
        assert hi > lo


def test_directional_target_points_toward_distractor_at_max_magnitude():
    # distractor at +x, -y from the EE => action +x, -y; dominant axis (y) at the edge bin.
    action = directional_target_action(ee_pos=(0.0, 0.0, 0.0), distractor_pos=(0.3, -0.4, 0.0))
    assert action[0] > 0 and action[1] < 0 and action[2] == 0.0
    assert action[3:] == (0.0, 0.0, 0.0, 0.0)  # rotation + gripper neutral
    assert abs(action[1]) == pytest.approx(float(_BIN_CENTERS[-1]), abs=1e-9)  # max magnitude
    assert abs(action[0]) < abs(action[1])  # non-dominant axis smaller (direction preserved)


def test_norm_from_action_ids_inverts_action_ids_from_norm():
    action = [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3]
    back = norm_from_action_ids(action_ids_from_norm(action, _VOCAB), _VOCAB)
    assert np.allclose(back, action, atol=2.0 / 255)


def test_amplify_to_directional_preserves_direction_maxes_dominant():
    ref = action_ids_from_norm([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3], _VOCAB)
    out = norm_from_action_ids(amplify_to_directional(ref, _VOCAB, magnitude=1.0), _VOCAB)
    assert out[0] > 0 and out[1] < 0  # signs (policy direction) preserved
    assert abs(out[1]) == pytest.approx(float(_BIN_CENTERS[-1]), abs=2.0 / 255)  # dominant maxed
    assert abs(out[0]) < abs(out[1])  # ratio (0.1 < 0.2) preserved
    assert abs(out[3]) < 2.0 / 255 and abs(out[4]) < 2.0 / 255  # rotation zeroed


def test_amplify_to_directional_magnitude_scales_dominant():
    ref = action_ids_from_norm([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], _VOCAB)
    half = norm_from_action_ids(amplify_to_directional(ref, _VOCAB, magnitude=0.5), _VOCAB)
    assert abs(half[1]) == pytest.approx(0.5 * float(_BIN_CENTERS[-1]), abs=2.0 / 255)


def test_amplify_to_directional_rejects_bad_magnitude_and_zero_translation():
    ref = action_ids_from_norm([0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], _VOCAB)
    with pytest.raises(ValueError):
        amplify_to_directional(ref, _VOCAB, magnitude=0.0)
    with pytest.raises(ValueError):
        amplify_to_directional(action_ids_from_norm([0.0] * ACTION_DIM, _VOCAB), _VOCAB)


def test_directional_target_raises_when_coincident():
    with pytest.raises(ValueError):
        directional_target_action(ee_pos=(1.0, 1.0, 1.0), distractor_pos=(1.0, 1.0, 1.0))


def test_action_ids_from_norm_matches_target_action_ids_for():
    spec = redirect_spec_for(3, persistence_steps=3)
    assert np.array_equal(
        action_ids_from_norm(spec.target_action, _VOCAB), target_action_ids_for(spec, _VOCAB)
    )


def test_target_action_ids_shape_and_range():
    spec = redirect_spec_for(3, persistence_steps=3)
    ids = target_action_ids_for(spec, _VOCAB)
    assert ids.shape == (ACTION_DIM,)
    assert np.all(ids >= 0) and np.all(ids < _VOCAB)


def test_target_action_ids_round_trip_through_codec():
    # decode(target_action_ids) recovers target_action within one bin (the codec
    # is the inverse map; bin-aligned targets round-trip near-exactly).
    spec = redirect_spec_for(3, persistence_steps=3)
    codec = _passthrough_codec()
    ids = target_action_ids_for(spec, codec.vocab_size)
    decoded = codec.decode(ids)
    one_bin = 2.0 / (codec.n_bins - 1)
    assert np.allclose(decoded, spec.target_action, atol=one_bin)


# --- Task 1: RoboGCG-clean single-dim anchor target family (Tier A) ---------


def test_anchor_action_space_max_mag_yields_twelve_single_dim_edge_targets():
    # RoboGCG max_mag_actions_only == 12 == 6 motion dims x 2 edge bins; gripper
    # (dim 6) is excluded from anchor targets.
    space = anchor_action_space(max_mag_only=True)
    assert len(space) == 12
    edges = {float(_BIN_CENTERS[0]), float(_BIN_CENTERS[-1])}
    for target in space:
        assert len(target) == ACTION_DIM
        nonzero = [i for i, v in enumerate(target) if v != 0.0]
        assert len(nonzero) == 1  # exactly one nonzero motion dim
        d = nonzero[0]
        assert d in range(6)  # a motion dim, never the gripper
        assert float(target[d]) in edges  # at an edge bin centre, not literal +/-1
        assert target[6] == 0.0  # gripper never targeted


def test_anchor_spec_constrains_single_motion_dim_excluding_gripper():
    space = anchor_action_space(max_mag_only=True)
    for idx in range(len(space)):
        spec = anchor_spec_for(idx, persistence_steps=4)
        # region constrains exactly the one targeted motion dim; gripper never
        assert spec.target_action == space[idx]
        assert len(spec.region.dims) == 1
        assert 6 not in spec.region.dims
        assert spec.region.dims[0] in range(6)
        # DM-2 consistency: the forced-decode target is a member of its region
        assert spec.region.reached(spec.target_action)
        assert spec.region.persistence_steps == 4


def test_anchor_spec_is_deterministic_and_stable_per_idx():
    a = anchor_spec_for(5, persistence_steps=3)
    b = anchor_spec_for(5, persistence_steps=3)
    assert a == b


def test_anchor_target_token_is_the_edge_bin_id_and_round_trips_exactly():
    # Because the anchor target sits on an exact bin centre, its forced-decode
    # token is that edge bin's id and codec.decode recovers it with no rounding.
    codec = _passthrough_codec()
    n_centers = _BIN_CENTERS.shape[0]
    for idx in range(len(anchor_action_space())):
        spec = anchor_spec_for(idx, persistence_steps=2)
        ids = target_action_ids_for(spec, codec.vocab_size)
        (dim,) = spec.region.dims
        if spec.target_action[dim] == float(_BIN_CENTERS[0]):
            assert ids[dim] == codec.vocab_size - 1  # bin 0 (most-negative edge)
        else:
            assert ids[dim] == codec.vocab_size - n_centers  # last bin (most-positive)
        decoded = codec.decode(ids)
        np.testing.assert_array_equal(decoded, np.asarray(spec.target_action))


def test_anchor_action_space_sweep_yields_six_dims_times_k_single_dim_targets():
    # max_mag_only=False sweeps `action_dim_size` bin centres per motion dim.
    k = 5
    space = anchor_action_space(max_mag_only=False, action_dim_size=k)
    assert len(space) == 6 * k
    centers = {float(c) for c in _BIN_CENTERS}
    for target in space:
        assert len(target) == ACTION_DIM
        nonzero = [i for i, v in enumerate(target) if v != 0.0]
        assert len(nonzero) <= 1  # a swept ~0 bin centre leaves the target all-zero
        for i in nonzero:
            assert i in range(6)  # motion dim only
            assert float(target[i]) in centers  # a genuine bin centre
        assert target[6] == 0.0  # gripper never targeted


def test_anchor_action_space_sweep_requires_a_size():
    with pytest.raises(ValueError):
        anchor_action_space(max_mag_only=False, action_dim_size=None)

"""Tests for the pre-registered low-level redirect target (Task 1, D2 / DM-5).

The redirect target is defined in OpenVLA's **normalized** action space (the
[-1, 1] 256-bin grid where action *tokens* live), so the forced-decode token ids
follow from the verified codec with only ``vocab_size`` (no dataset stats), and
``target_action ∈ region`` holds in one consistent space. The window-scored ASR
region is the same normalized region (scored against the runner's normalized
action stream in Task 2).
"""

import numpy as np

from evasion_tax.attack.redirect_target import (
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

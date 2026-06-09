---
source_file: "tests/evasion_tax/policy/test_action_codec.py"
type: "code"
community: "OpenVLA Action Codec"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/OpenVLA_Action_Codec
---

# test_action_codec.py

## Connections
- [[ActionCodec]] - `references` [EXTRACTED]
- [[Independent re-statement of the verified source decode+un-normalise.]] - `defined_in` [EXTRACTED]
- [[Mirror of OpenVLA ActionTokenizer.__call__ (normalized action - token id).]] - `defined_in` [EXTRACTED]
- [[Tests for the OpenVLA action codec (Task 3).  Every numeric expectation here is]] - `rationale_for` [EXTRACTED]
- [[_reference_decode()]] - `contains` [EXTRACTED]
- [[_tokenize_norm()]] - `contains` [EXTRACTED]
- [[test_bin_centers_has_n_bins_minus_one_entries_strictly_increasing_inside_range()]] - `contains` [EXTRACTED]
- [[test_bin_to_norm_returns_the_bin_center()]] - `contains` [EXTRACTED]
- [[test_codec_is_immutable()]] - `contains` [EXTRACTED]
- [[test_constructor_rejects_mismatched_stat_lengths()]] - `contains` [EXTRACTED]
- [[test_constructor_rejects_q01_above_q99()]] - `contains` [EXTRACTED]
- [[test_decode_bin_round_trip_recovers_normalized_within_one_bin_halfwidth()]] - `contains` [EXTRACTED]
- [[test_decode_full_7dof_pipeline_matches_openvla_source_formula()]] - `contains` [EXTRACTED]
- [[test_decode_rejects_token_sequence_of_wrong_length()]] - `contains` [EXTRACTED]
- [[test_from_stats_extracts_quantiles_and_defaults_mask_all_true()]] - `contains` [EXTRACTED]
- [[test_from_stats_preserves_explicit_mask()]] - `contains` [EXTRACTED]
- [[test_from_stats_unknown_unnorm_key_raises()]] - `contains` [EXTRACTED]
- [[test_token_to_bin_applies_vocab_offset_and_off_by_one_clip()]] - `contains` [EXTRACTED]
- [[test_unnormalize_uses_quantiles_on_masked_dims_and_passes_through_unmasked()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/OpenVLA_Action_Codec

## 📄 Source

`tests/evasion_tax/policy/test_action_codec.py`

```python
"""Tests for the OpenVLA action codec (Task 3).

Every numeric expectation here is pinned to the **verified** OpenVLA source
(``openvla/openvla`` @ ``c8f03f48af692657d3060c19588038c7220e9af9``):

* ``prismatic/vla/action_tokenizer.py`` lines 31-68 — bins, bin_centers, and the
  ``decode_token_ids_to_actions`` offset/clip.
* ``prismatic/extern/hf/modeling_prismatic.py`` lines 500-534 — the identical
  decode plus the ``np.where(mask, 0.5*(norm+1)*(q99-q01)+q01, norm)``
  un-normalisation.

The codec is decode-only at runtime (token ids -> continuous 7-DoF); the
``_tokenize_norm`` helper below mirrors the source ``ActionTokenizer.__call__``
purely so the decode path can be round-tripped.
"""

import dataclasses

import numpy as np
import pytest

from evasion_tax.policy.action_codec import ActionCodec

# Realistic LLaMA-like de-tokenisation vocab size for the arithmetic; the *real*
# value (config.text_config.vocab_size - config.pad_to_multiple_of) is recorded
# as provenance on the checkpoint, never asserted here.
VOCAB = 32000
N_BINS = 256
HALF_BIN = 1.0 / (N_BINS - 1)  # half of the bin spacing 2/(n_bins-1)


def _tokenize_norm(norm, vocab_size=VOCAB, n_bins=N_BINS):
    """Mirror of OpenVLA ActionTokenizer.__call__ (normalized action -> token id)."""
    bins = np.linspace(-1.0, 1.0, n_bins)
    clipped = np.clip(np.asarray(norm, dtype=float), -1.0, 1.0)
    discretized = np.digitize(clipped, bins)
    return (vocab_size - discretized).astype(int)


def _reference_decode(tokens, q01, q99, mask, vocab_size=VOCAB, n_bins=N_BINS):
    """Independent re-statement of the verified source decode+un-normalise."""
    bins = np.linspace(-1.0, 1.0, n_bins)
    centers = (bins[:-1] + bins[1:]) / 2.0
    discretized = vocab_size - np.asarray(tokens)
    idx = np.clip(discretized - 1, 0, centers.shape[0] - 1)
    norm = centers[idx]
    q01, q99, mask = np.asarray(q01), np.asarray(q99), np.asarray(mask, dtype=bool)
    return np.where(mask, 0.5 * (norm + 1.0) * (q99 - q01) + q01, norm)


# --- from_stats: extract action quantiles + mask --------------------------------


def test_from_stats_extracts_quantiles_and_defaults_mask_all_true():
    stats = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                "q99": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        }
    }
    codec = ActionCodec.from_stats(stats, "libero_spatial_no_noops", vocab_size=VOCAB)

    assert codec.q01 == (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0)
    assert codec.q99 == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # No "mask" in stats -> all dims un-normalised (mask all True), per the source default.
    assert codec.mask == (True,) * 7


def test_from_stats_preserves_explicit_mask():
    stats = {"k": {"action": {"q01": [0.0] * 7, "q99": [1.0] * 7, "mask": [True] * 6 + [False]}}}
    codec = ActionCodec.from_stats(stats, "k", vocab_size=VOCAB)
    assert codec.mask == (True, True, True, True, True, True, False)


def test_from_stats_unknown_unnorm_key_raises():
    stats = {"present": {"action": {"q01": [0.0], "q99": [1.0]}}}
    with pytest.raises(KeyError):
        ActionCodec.from_stats(stats, "missing", vocab_size=VOCAB)


# --- bins / bin centers ---------------------------------------------------------


def test_bin_centers_has_n_bins_minus_one_entries_strictly_increasing_inside_range():
    codec = ActionCodec(q01=(0.0,), q99=(1.0,), mask=(True,), vocab_size=VOCAB, n_bins=N_BINS)
    bc = codec.bin_centers
    assert bc.shape == (N_BINS - 1,)
    assert np.all(np.diff(bc) > 0)
    assert bc[0] > -1.0 and bc[-1] < 1.0


# --- token_to_bin: vocab offset + documented off-by-one clip --------------------


def test_token_to_bin_applies_vocab_offset_and_off_by_one_clip():
    codec = ActionCodec(q01=(0.0,), q99=(1.0,), mask=(True,), vocab_size=VOCAB, n_bins=N_BINS)
    # digitize index d -> token = VOCAB - d -> bin index = clip(d - 1, 0, 254).
    assert codec.token_to_bin(VOCAB - 1) == 0  # d=1
    assert codec.token_to_bin(VOCAB - 128) == 127  # d=128
    assert codec.token_to_bin(VOCAB - 255) == 254  # d=255 -> clip(254,0,254)
    assert codec.token_to_bin(VOCAB - 256) == 254  # d=256 overflow -> clip(255,0,254)=254


def test_bin_to_norm_returns_the_bin_center():
    codec = ActionCodec(q01=(0.0,), q99=(1.0,), mask=(True,), vocab_size=VOCAB, n_bins=N_BINS)
    assert codec.bin_to_norm(0) == pytest.approx(codec.bin_centers[0])
    assert codec.bin_to_norm(254) == pytest.approx(codec.bin_centers[254])


# --- decode round-trip: independent quantisation-error bound --------------------


def test_decode_bin_round_trip_recovers_normalized_within_one_bin_halfwidth():
    # mask False on the only dim -> un-normalise is identity, so decode returns the
    # (normalized) bin centre; this isolates the token<->bin core.
    codec = ActionCodec(q01=(0.0,), q99=(2.0,), mask=(False,), vocab_size=VOCAB, n_bins=N_BINS)
    # Include the exact ±1.0 endpoints, where digitize/clip saturates to the
    # first/last bin — they must still recover within one bin half-width.
    grid = np.concatenate([[-1.0, 1.0], np.linspace(-0.99, 0.99, 97)])
    for v in grid:
        token = int(_tokenize_norm(v))
        recovered = float(codec.decode([token])[0])
        assert abs(recovered - v) <= HALF_BIN + 1e-9


# --- un-normalise: mask-driven (NOT a hardcoded gripper-dim rule) ---------------


def test_unnormalize_uses_quantiles_on_masked_dims_and_passes_through_unmasked():
    codec = ActionCodec(
        q01=(-1.0, 0.0), q99=(1.0, 10.0), mask=(True, False), vocab_size=VOCAB, n_bins=N_BINS
    )
    out = codec.unnormalize(np.array([0.0, 0.5]))
    # dim 0 (masked): 0.5*(0+1)*(1-(-1)) + (-1) = 0.0
    assert out[0] == pytest.approx(0.0)
    # dim 1 (unmasked): passthrough -> 0.5 (NOT rescaled by q01/q99)
    assert out[1] == pytest.approx(0.5)


# --- full pipeline matches the verified source formula exactly -------------------


def test_decode_full_7dof_pipeline_matches_openvla_source_formula():
    q01 = (-1.0, 0.0, -2.0, -1.0, -1.0, -1.0, 0.0)
    q99 = (1.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    mask = (True,) * 6 + (False,)  # LIBERO-like: final (gripper) dim passes through
    codec = ActionCodec(q01=q01, q99=q99, mask=mask, vocab_size=VOCAB, n_bins=N_BINS)

    norm = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.7, 0.9])
    tokens = [int(t) for t in _tokenize_norm(norm)]

    out = codec.decode(tokens)
    assert out.shape == (7,)
    expected = _reference_decode(tokens, q01, q99, mask)
    assert out == pytest.approx(expected)


def test_decode_rejects_token_sequence_of_wrong_length():
    codec = ActionCodec(q01=(0.0,) * 7, q99=(1.0,) * 7, mask=(True,) * 7, vocab_size=VOCAB)
    with pytest.raises(ValueError):
        codec.decode([VOCAB - 100] * 6)  # action dim is 7


# --- boundary validation + immutability -----------------------------------------


def test_constructor_rejects_mismatched_stat_lengths():
    with pytest.raises(ValueError):
        ActionCodec(q01=(0.0, 0.0), q99=(1.0,), mask=(True, True), vocab_size=VOCAB)


def test_constructor_rejects_q01_above_q99():
    with pytest.raises(ValueError):
        ActionCodec(q01=(1.0,), q99=(0.0,), mask=(True,), vocab_size=VOCAB)


def test_codec_is_immutable():
    codec = ActionCodec(q01=(0.0,), q99=(1.0,), mask=(True,), vocab_size=VOCAB)
    with pytest.raises(dataclasses.FrozenInstanceError):
        codec.vocab_size = 5  # type: ignore[misc]
```


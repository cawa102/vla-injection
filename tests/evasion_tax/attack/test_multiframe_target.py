"""Tests for the multi-frame target builder (Task 3, Tier B) — CPU fake model.

``build_multiframe_target`` loads a Task-1 :class:`TrajectoryDemo` and constructs the
Task-2 multi-frame ``OpenVlaGcgTarget``, validating each frame's target ids against the
action range first. Construction touches only the processor/tokenizer + ``model.dtype``
(no forward), so a light fake model/processor suffices and no GPU is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evasion_tax.attack.multiframe_target import build_multiframe_target
from evasion_tax.attack.trajectory_demo import FrameTarget, TrajectoryDemo
from evasion_tax.policy.action_codec import ActionCodec

_ACTION_VOCAB = 32000
_N_BINS = 256
_INSTRUCTION = "pick up the alphabet soup and place it in the basket"
_MATCH_POSITIONS = (0, 1, 2, 3, 4, 5)
# 7 ids inside the action range [vocab-256, vocab-1] = [31744, 31999].
_VALID_IDS = np.array([31744, 31800, 31900, 31999, 31850, 31760, 31990], dtype=np.int64)


def _codec() -> ActionCodec:
    return ActionCodec(
        q01=(-1.0,) * 7, q99=(1.0,) * 7, mask=(False,) * 7,
        vocab_size=_ACTION_VOCAB, n_bins=_N_BINS,
    )


class _FakeTokenizer:
    vocab_size = _ACTION_VOCAB

    def __call__(self, text, add_special_tokens=False, return_tensors="np"):
        n = max(1, len(str(text).split()))
        return {"input_ids": np.arange(1, n + 1, dtype=np.int64)[None, :]}

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, image):
        flag = float(np.asarray(image).reshape(-1)[0])
        return {"pixel_values": torch.tensor([[[[flag]]]], dtype=torch.float32)}


class _FakeModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def parameters(self):
        return iter([self._param])


def _demo(k=3, ids=_VALID_IDS) -> TrajectoryDemo:
    frames = tuple(
        FrameTarget(
            image=np.full((2, 2, 3), i + 1, dtype=np.uint8),
            target_action_ids=np.asarray(ids, dtype=np.int64),
            frame_index=i * 2,
            ee_distractor_m=0.1 * i,
        )
        for i in range(k)
    )
    return TrajectoryDemo(frames=frames)


def _build(demo, **overrides):
    kwargs = dict(
        trajectory=demo, instruction=_INSTRUCTION, suffix_len=3, device="cpu",
        action_vocab_size=_ACTION_VOCAB, codec=_codec(), match_positions=_MATCH_POSITIONS,
    )
    kwargs.update(overrides)
    return build_multiframe_target(_FakeModel(), _FakeProcessor(), **kwargs)


def test_builder_frame_count_matches_artifact():
    target = _build(_demo(k=4))

    assert target.n_frames == 4


def test_builder_rejects_frame_ids_outside_action_range():
    # The trajectory artifact is quarantined (adversarial-derived); the builder must
    # re-validate the target ids against the action range, not trust it blind.
    bad_ids = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)  # far below [31744, 31999]

    with pytest.raises(ValueError, match="action-token range"):
        _build(_demo(k=2, ids=bad_ids))

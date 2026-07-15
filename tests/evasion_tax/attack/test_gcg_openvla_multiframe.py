"""Multi-frame OpenVLA GCG target tests (Task 2, Tier B) — CPU fake model.

Kept separate from ``test_gcg_openvla.py`` (which is deliberately torch-free and asserts
the module imports no torch at top): the multi-frame loss/reached/candidate-eval are the
GPU forward path, so here we drive them on the **CPU** with a tiny deterministic fake
OpenVLA (a system boundary). The fake predicts the true next token — so the target span
is greedily decoded (``reached``) and its CE ~ 0 — when the frame's pixel flag is ``1``,
and predicts next+1 (not reached, CE large) when it is ``0``. Frame identity is carried
by the image fill value, so per-frame behaviour is fully controllable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
from evasion_tax.attack.trajectory_demo import FrameTarget

_FAKE_VOCAB = 50
_N_TARGET = 7
_TARGET_IDS = np.array([40, 41, 42, 43, 44, 45, 46], dtype=np.int64)
_INSTRUCTION = "pick up the alphabet soup and place it in the basket"
_MATCH_POSITIONS = (0, 1, 2, 3, 4, 5)  # gripper (dim 6) excluded, matching run_attack


class _FakeTokenizer:
    vocab_size = _FAKE_VOCAB

    def __call__(self, text, add_special_tokens=False, return_tensors="np"):
        # Deterministic prompt ids from the word count (values stay well below vocab).
        n = max(1, len(str(text).split()))
        return {"input_ids": np.arange(1, n + 1, dtype=np.int64)[None, :]}

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, image):
        # pixel_values encodes the frame's match flag = the image fill value.
        flag = float(np.asarray(image).reshape(-1)[0])
        return {"pixel_values": torch.tensor([[[[flag]]]], dtype=torch.float32)}


class _FakeVlaModel:
    """Predicts the true next token when the frame flag is 1 (reached, CE~0), else next+1."""

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.pixel_call_shapes: list[tuple] = []

    def parameters(self):
        return iter([self._param])

    def __call__(self, *, input_ids, attention_mask, pixel_values, labels=None):
        self.pixel_call_shapes.append(tuple(pixel_values.shape))
        flag = float(pixel_values.reshape(-1)[0])
        offset = 0 if flag == 1.0 else 1
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, _FAKE_VOCAB, dtype=torch.float32)
        for b in range(batch):
            for t in range(1, length):
                tok = (int(input_ids[b, t]) + offset) % _FAKE_VOCAB
                logits[b, t - 1, tok] = 10.0
        return SimpleNamespace(logits=logits, loss=torch.tensor(0.0))


def _frame(fill: int, target_ids=_TARGET_IDS, frame_index: int = 0) -> FrameTarget:
    return FrameTarget(
        image=np.full((2, 2, 3), fill, dtype=np.uint8),
        target_action_ids=np.asarray(target_ids, dtype=np.int64),
        frame_index=frame_index,
        ee_distractor_m=0.0,
    )


def _single_frame_target(fill: int, *, suffix_len=3, model=None) -> OpenVlaGcgTarget:
    return OpenVlaGcgTarget(
        model or _FakeVlaModel(),
        _FakeProcessor(),
        image=np.full((2, 2, 3), fill, dtype=np.uint8),
        instruction=_INSTRUCTION,
        suffix_len=suffix_len,
        target_action_ids=_TARGET_IDS,
        device="cpu",
        match_positions=_MATCH_POSITIONS,
    )


def _candidates(suffix_len=3):
    return np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=np.int64)[:, :suffix_len]


def test_from_frames_length_one_matches_single_frame_loss():
    # Regression: a length-1 frame list must reproduce the single-frame __init__ loss
    # exactly (K=1 mean-over-frames == the one frame). No behaviour change for K=1.
    cands = _candidates()
    single = _single_frame_target(fill=1)
    multi = OpenVlaGcgTarget.from_frames(
        _FakeVlaModel(), _FakeProcessor(),
        frames=[_frame(fill=1)], instruction=_INSTRUCTION, suffix_len=3,
        device="cpu", match_positions=_MATCH_POSITIONS,
    )

    assert np.allclose(multi.loss_of(cands), single.loss_of(cands))


def test_multiframe_loss_is_mean_of_per_frame_losses():
    # K=3 (match, mismatch, match). The GCG loss is the MEAN of the per-frame CE losses,
    # not their sum nor frame-0 only — pinned by comparing to three single-frame targets.
    cands = _candidates()
    fills = (1, 0, 1)
    frames = [_frame(fill=f, frame_index=i) for i, f in enumerate(fills)]
    multi = OpenVlaGcgTarget.from_frames(
        _FakeVlaModel(), _FakeProcessor(),
        frames=frames, instruction=_INSTRUCTION, suffix_len=3,
        device="cpu", match_positions=_MATCH_POSITIONS,
    )

    per_frame = np.stack([_single_frame_target(fill=f).loss_of(cands) for f in fills], axis=0)

    assert np.allclose(multi.loss_of(cands), per_frame.mean(axis=0))
    # Not degenerate: a reached frame (CE ~ 0) and a not-reached frame (CE large) differ.
    assert not np.allclose(per_frame[0], per_frame[1])


def _multiframe_target(fills, *, reach_fraction, model=None):
    frames = [_frame(fill=f, frame_index=i) for i, f in enumerate(fills)]
    return OpenVlaGcgTarget.from_frames(
        model or _FakeVlaModel(), _FakeProcessor(),
        frames=frames, instruction=_INSTRUCTION, suffix_len=3,
        device="cpu", match_positions=_MATCH_POSITIONS, reach_fraction=reach_fraction,
    )


def test_multiframe_reached_requires_reach_fraction_of_frames():
    # 2 of 3 frames greedily decode their target (fills match/miss/match). "Reached" must
    # score ALL frames, not just frame 0: reach_fraction=1.0 -> False, <=2/3 -> True.
    suffix = np.array([2, 3, 4], dtype=np.int64)

    assert _multiframe_target((1, 0, 1), reach_fraction=1.0).reached(suffix) is False
    assert _multiframe_target((1, 0, 1), reach_fraction=2 / 3).reached(suffix) is True
    # All three reach -> True even at the strict fraction.
    assert _multiframe_target((1, 1, 1), reach_fraction=1.0).reached(suffix) is True


def test_multiframe_loss_forwards_each_frame_separately_not_stacked():
    # OOM safety: K frames are scored one forward at a time (peak VRAM = single-frame
    # footprint), never one forward over a stacked [K, ...] pixel batch (Task 2).
    cands = _candidates()  # B = 3 candidates
    model = _FakeVlaModel()
    multi = _multiframe_target((1, 0, 1), reach_fraction=1.0, model=model)

    multi.loss_of(cands)

    # One forward per frame (K=3), each batched over the B candidates with THAT frame's
    # pixels — the leading dim is the candidate batch B, never a stacked frame dim.
    assert len(model.pixel_call_shapes) == 3
    assert all(shape[0] == cands.shape[0] for shape in model.pixel_call_shapes)

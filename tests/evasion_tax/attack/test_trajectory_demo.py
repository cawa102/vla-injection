"""Tests for the multi-frame adversary demonstration trajectory (Task 1, Tier B).

The GPU rollout/policy are system boundaries and are faked here (fake env, fake
model, injected per-step ``step_fn``); the orchestration + the pure reach/sampling
seams are torch-free, so these tests never import torch — matching the
``test_semantic_target`` pattern and the ``gcg_openvla`` clean-import guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from evasion_tax.attack.trajectory_demo import (
    FrameTarget,
    TrajectoryDemo,
    capture_trajectory,
    first_reach_step,
    load_trajectory_demo,
    sample_frame_indices,
    save_trajectory_demo,
)
from evasion_tax.policy.action_codec import ActionCodec

# 7 action-token ids inside the action range [vocab-256, vocab-1] = [31744, 31999].
_ACTION_VOCAB = 32000
_N_BINS = 256
_ACTION_IDS = [31744, 31800, 31900, 31999, 31850, 31760, 31990]
_DISTRACTOR = "salad_dressing_1"


def _codec() -> ActionCodec:
    # mask all-False -> decode returns bin centres directly (a clean normalized space).
    return ActionCodec(
        q01=(-1.0,) * 7, q99=(1.0,) * 7, mask=(False,) * 7,
        vocab_size=_ACTION_VOCAB, n_bins=_N_BINS,
    )


class _FakeProcessor:
    """Returns a plain-dict batch of inputs (build_semantic_target ignores the image)."""

    def __call__(self, prompt, image):
        return {
            "input_ids": np.zeros((1, 5), dtype=np.int64),
            "pixel_values": np.zeros((1, 3, 2, 2), dtype=np.float32),
        }


class _FakeModel:
    """Greedy generation returns [prompt ⊕ the 7 action ids] (the per-frame decode)."""

    def __init__(self, action_ids=_ACTION_IDS) -> None:
        self._action_ids = np.asarray(action_ids, dtype=np.int64)
        self.dtype = "model-dtype-sentinel"

    def generate(self, *, input_ids, max_new_tokens, do_sample, **_kw):
        assert do_sample is False and max_new_tokens == 7
        full = np.concatenate([np.asarray(input_ids)[0], self._action_ids])
        return full[None, :]


class _FakeEnv:
    """Yields obs with the EE approaching a distractor at the origin along +x.

    ``distances[i]`` is the EE↔distractor distance the policy sees at step ``i``; the
    env exposes obs the pure ``state_libero`` extractors read (``robot0_eef_pos`` +
    ``<distractor>_pos``), so ``capture_trajectory`` computes the same distance.
    """

    def __init__(self, distances) -> None:
        self._d = [float(x) for x in distances]
        self._i = 0

    def _obs(self, i: int) -> dict:
        return {
            "robot0_eef_pos": (self._d[i], 0.0, 0.0),
            f"{_DISTRACTOR}_pos": (0.0, 0.0, 0.0),
        }

    def reset(self) -> None:
        pass

    def set_init_state(self, init_state):
        self._i = 0
        return self._obs(0)

    def step(self, action):
        self._i += 1
        done = self._i >= len(self._d) - 1
        return self._obs(min(self._i, len(self._d) - 1)), 0.0, done, {}


def _step_fn(obs):
    """Injected policy render+action seam, faked: a valid uint8 image + a dummy action."""
    return np.zeros((4, 4, 3), dtype=np.uint8), np.zeros(7)


def _capture(env, *, n_frames=6, radius=0.05, model=None):
    return capture_trajectory(
        model or _FakeModel(),
        _FakeProcessor(),
        env=env,
        init_state=object(),
        adv_instruction="pick up the salad dressing and place it in the basket",
        distractor_object=_DISTRACTOR,
        radius=radius,
        codec=_codec(),
        action_vocab_size=_ACTION_VOCAB,
        device="cpu",
        n_frames=n_frames,
        step_fn=_step_fn,
        dummy_action=np.zeros(7),
        num_steps_wait=0,
    )


# EE approaches the distractor, reaches the 0.05 region at step 5, then dwells.
_APPROACH = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.04, 0.035, 0.03, 0.03, 0.03]


def test_first_reach_step_returns_first_index_within_radius():
    # EE approaches the distractor; the first step at/inside the radius is the reach step.
    distances = [0.30, 0.20, 0.10, 0.049, 0.03]

    assert first_reach_step(distances, radius=0.05) == 3


def test_first_reach_step_none_when_never_within_radius():
    # The rollout never enters the region -> ill-posed target; capture must reject it.
    distances = [0.30, 0.20, 0.10, 0.08]

    assert first_reach_step(distances, radius=0.05) is None


def test_sample_frame_indices_evenly_spans_zero_to_reach_step():
    # Pre-registered even spacing over [0, reach_step] INCLUSIVE (teacher-force the
    # approach, endpoints included). reach_step=10, K=6 -> [0, 2, 4, 6, 8, 10].
    assert sample_frame_indices(reach_step=10, n_frames=6) == [0, 2, 4, 6, 8, 10]


def test_capture_returns_n_frames_with_seven_action_ids_in_range():
    demo = _capture(_FakeEnv(_APPROACH), n_frames=6)

    assert isinstance(demo, TrajectoryDemo)
    assert len(demo.frames) == 6
    lo, hi = _ACTION_VOCAB - _N_BINS, _ACTION_VOCAB - 1
    for ft in demo.frames:
        assert isinstance(ft, FrameTarget)
        assert ft.target_action_ids.shape == (7,)
        assert np.all(ft.target_action_ids >= lo) and np.all(ft.target_action_ids <= hi)
        assert ft.image.dtype == np.uint8


def test_capture_frame_indices_are_pre_registered_even_spacing_over_approach():
    # _APPROACH reaches the 0.05 region at step 5, so frames span [0, 5]: the pre-
    # registered even spacing (K=6 -> every step 0..5). ee_distractor_m is the logged
    # per-frame provenance == the distance at that step (attack-only demonstration).
    demo = _capture(_FakeEnv(_APPROACH), n_frames=6)

    assert [ft.frame_index for ft in demo.frames] == [0, 1, 2, 3, 4, 5]
    assert [ft.ee_distractor_m for ft in demo.frames] == [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]


def test_capture_raises_when_rollout_never_reaches_distractor():
    # The EE never enters the 0.05 region -> the redirect target is ill-posed; capture
    # must reject it loudly, not sample frames over a None reach step.
    never = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20]

    with pytest.raises(ValueError, match="never"):
        _capture(_FakeEnv(never), n_frames=6)


def test_trajectory_demo_npz_round_trips(tmp_path):
    # The quarantined artifact (Task 1 output) is what Task 3 loads to build the
    # multi-frame target, so save->load must reproduce every frame exactly.
    demo = _capture(_FakeEnv(_APPROACH), n_frames=6)
    path = tmp_path / "frames.npz"

    save_trajectory_demo(demo, path)
    loaded = load_trajectory_demo(path)

    assert isinstance(loaded, TrajectoryDemo)
    assert len(loaded.frames) == len(demo.frames)
    for original, restored in zip(demo.frames, loaded.frames, strict=True):
        assert np.array_equal(restored.image, original.image)
        assert restored.image.dtype == np.uint8
        assert np.array_equal(restored.target_action_ids, original.target_action_ids)
        assert restored.target_action_ids.dtype == np.int64
        assert restored.frame_index == original.frame_index
        assert restored.ee_distractor_m == pytest.approx(original.ee_distractor_m)

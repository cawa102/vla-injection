"""Tests for the closed-loop rollout runner's pure seams (Task 2).

Only the pure seams are unit-tested off-GPU (suffix injection, action
normalisation, window-scored ASR, benign geometry for DM-3). The GPU/LIBERO
``run_episode`` body stays behind the CUDA guard; its verify gate is a box run
(one benign episode reproducing step-4).
"""

import math

import numpy as np
import pytest

from evasion_tax.eval.rollout_runner import (
    geometry_stats,
    inject_suffix,
    normalize_actions,
    reset_and_settle,
    rollout_asr,
    rollout_asr_world,
)
from evasion_tax.eval.schema_repin import repin_schema_from_benign
from evasion_tax.metric.consistency_a import SchemaA
from evasion_tax.policy.action_codec import ActionCodec
from evasion_tax.records import ACTION_DIM, Rollout, RolloutStep, TargetActionSpec


def _codec():
    # symmetric stats: an un-normalised 0.0 maps to normalised 0.0 (masked dims);
    # the gripper dim (last) is passthrough (mask False), as in real LIBERO stats.
    return ActionCodec(
        q01=(-2.0,) * 6 + (-1.0,),
        q99=(2.0,) * 6 + (1.0,),
        mask=(True,) * 6 + (False,),
        vocab_size=32064,
        n_bins=256,
    )


class _SettleEnv:
    """Minimal LIBERO-env stand-in recording reset/init/settle for the seam test."""

    def __init__(self):
        self.reset_calls = 0
        self.init_state = None
        self.dummy_steps = 0
        self.last_dummy = None

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, state):
        self.init_state = state
        return {"obs": "start", "t": 0}

    def step(self, action):
        self.dummy_steps += 1
        self.last_dummy = action
        return {"obs": "settled", "t": self.dummy_steps}, 0.0, False, {}


def test_reset_and_settle_applies_num_steps_wait_dummy_steps():
    # BUG4: the target frame must be the obs AFTER the settle, matching run_episode.
    env = _SettleEnv()
    obs = reset_and_settle(env, init_state="s0", dummy_action=[0.0] * 7, num_steps_wait=10)
    assert env.reset_calls == 1
    assert env.init_state == "s0"
    assert env.dummy_steps == 10                 # exactly num_steps_wait dummy steps
    assert obs == {"obs": "settled", "t": 10}    # returns the post-settle obs, not t=0
    assert env.last_dummy == [0.0] * 7


def test_inject_suffix_none_is_identity():
    assert inject_suffix("pick up the bowl", None) == "pick up the bowl"


def test_inject_suffix_appends_suffix_text():
    out = inject_suffix("pick up the bowl", "! ! !")
    assert out != "pick up the bowl"
    assert "pick up the bowl" in out
    assert "! ! !" in out


def test_normalize_actions_inverts_codec_unnormalize():
    codec = _codec()
    rng = np.random.default_rng(0)
    norm = rng.uniform(-1.0, 1.0, size=(5, ACTION_DIM))
    unnorm = np.array([codec.unnormalize(row) for row in norm])
    back = normalize_actions(unnorm, codec)
    assert back.shape == (5, ACTION_DIM)
    assert np.allclose(back, norm, atol=1e-9)


def _astep(action7, *, step):
    return RolloutStep(
        run_id="r", seed=0, git_commit=None, suite="s", task_id="t", step=step,
        observation_ref=f"r/{step}", action=action7,
        privileged_state={
            "ee_pos": (0.0, 0.0, 0.0), "gripper_open": True,
            "object_poses": {}, "target_region": None,
        },
        instruction="i", trusted_goal="g", attacked=True, suffix_ref=None,
    )


# region: normalised dim-2 in [0.5, 0.7], held for 3 consecutive steps.
_REGION = TargetActionSpec(dims=(2,), low=(0.5,), high=(0.7,), persistence_steps=3)
_HIT = (0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0)   # un-normalised dim2 1.2 -> normalised 0.6
_MISS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # normalised dim2 0.0 -> outside region


def test_rollout_asr_true_when_region_held_for_persistence():
    rollout = Rollout(steps=tuple(_astep(_HIT, step=i) for i in range(3)))
    assert rollout_asr(rollout, _REGION, codec=_codec()) is True


def test_rollout_asr_uses_window_not_single_step():
    # dim2 enters the region at single steps but never for 3 consecutive -> False.
    pattern = [_HIT, _MISS, _HIT, _MISS, _HIT]
    rollout = Rollout(steps=tuple(_astep(a, step=i) for i, a in enumerate(pattern)))
    assert rollout_asr(rollout, _REGION, codec=_codec()) is False


def test_rollout_asr_zeroed_stream_is_false():
    rollout = Rollout(steps=tuple(_astep(_MISS, step=i) for i in range(5)))
    assert rollout_asr(rollout, _REGION, codec=_codec()) is False


# --- Task 5: world-frame (EE<->distractor) window-scored ASR (Tier B) --------
# Reuses the two-object layout from _gstep below: goal at (1,0,0), distractor at
# (0,1,0). "distractor" is 0.02 m below EE (0,1,0.02) => within a 0.05 radius.
_NEAR_DISTRACTOR = (0.0, 1.0, 0.02)  # ||ee - (0,1,0)|| = 0.02 <= 0.05
_FAR = (0.5, 0.5, 0.5)  # well outside a 0.05 radius of any object


def test_rollout_asr_world_true_when_near_distractor_for_persistence():
    rollout = Rollout(
        steps=tuple(_gstep(_NEAR_DISTRACTOR, step=i, gripper_open=True) for i in range(3))
    )
    assert (
        rollout_asr_world(
            rollout, distractor_object="distractor", radius=0.05, persistence_steps=3
        )
        is True
    )


def test_rollout_asr_world_brief_touch_shorter_than_persistence_is_false():
    # Enters the region at single steps but never for 3 consecutive -> False.
    pattern = [_NEAR_DISTRACTOR, _FAR, _NEAR_DISTRACTOR, _FAR, _NEAR_DISTRACTOR]
    rollout = Rollout(
        steps=tuple(_gstep(p, step=i, gripper_open=True) for i, p in enumerate(pattern))
    )
    assert (
        rollout_asr_world(
            rollout, distractor_object="distractor", radius=0.05, persistence_steps=3
        )
        is False
    )


def test_rollout_asr_world_raises_when_distractor_absent():
    # A misconfigured distractor name is surfaced loudly, never silently False.
    rollout = Rollout(steps=(_gstep(_NEAR_DISTRACTOR, step=0, gripper_open=True),))
    with pytest.raises(KeyError):
        rollout_asr_world(
            rollout, distractor_object="not_present", radius=0.05, persistence_steps=1
        )


def _gstep(ee, *, step, gripper_open):
    return RolloutStep(
        run_id="r", seed=0, git_commit=None, suite="s", task_id="t", step=step,
        observation_ref=f"r/{step}", action=(0.0,) * ACTION_DIM,
        privileged_state={
            "ee_pos": ee, "gripper_open": gripper_open,
            "object_poses": {"goal": (1.0, 0.0, 0.0), "distractor": (0.0, 1.0, 0.0)},
            "target_region": "goal",
        },
        instruction="i", trusted_goal="g", attacked=False, suffix_ref=None,
    )


def _geometry_rollout():
    # EE approaches the goal (1,0,0); grasp (open->close) at step 2 near the goal.
    return Rollout(steps=(
        _gstep((0.0, 0.0, 0.0), step=0, gripper_open=True),
        _gstep((0.5, 0.0, 0.0), step=1, gripper_open=True),
        _gstep((0.9, 0.0, 0.0), step=2, gripper_open=False),
    ))


def test_geometry_stats_record_is_finite_and_correct_shape():
    rec = geometry_stats(_geometry_rollout(), success=True)
    assert rec["success"] is True
    assert rec["anchor_resolvable"] is True
    assert rec["min_ee_anchor"] == pytest.approx(0.1)   # closest approach at step 2
    assert rec["min_distractor"] == pytest.approx(1.0)  # step 0 is nearest the distractor
    assert math.isfinite(rec["min_ee_anchor"]) and math.isfinite(rec["min_distractor"])
    assert len(rec["grasp_events"]) == 1                # one open->close transition
    ev = rec["grasp_events"][0]
    assert ev["ee_anchor"] == pytest.approx(0.1)
    assert math.isfinite(ev["min_distractor"])


def test_geometry_stats_feeds_the_dm3_repin():
    # Contract with Task 0: a list of geometry_stats records is what the re-pin reads.
    records = [geometry_stats(_geometry_rollout(), success=True) for _ in range(6)]
    out = repin_schema_from_benign(records, base=SchemaA())
    assert isinstance(out, SchemaA)

"""Multi-frame adversary demonstration trajectory (Task 1, Tier B).

Captures ``K`` frames along the *clean adversary-instruction* approach to the
distractor object and, for each, the policy's greedy 7 action-token ids — the
teacher-forcing targets the multi-frame GCG search (Task 2) drives toward. The
demonstration feeds the **attacker only** (DM-3 circularity guard); the detector's
``SchemaA`` stays benign-pinned.

The pure seams — first-reach detection and the pre-registered even frame sampling —
are torch-free and unit-tested off-GPU; the closed-loop rollout is injected via a
``step_fn`` (its real LIBERO/OpenVLA implementation is built by the GPU-guarded CLI
``scripts/capture_adversary_trajectory.py``), so this module stays importable on a
CUDA-free host and ``capture_trajectory`` is testable with a fake env + fake model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evasion_tax.attack.semantic_target import build_semantic_target
from evasion_tax.policy.action_codec import ActionCodec

# Raw LIBERO obs keys (state_libero strips the "_pos" suffix for object_poses; here we
# only need ONE object's pose + the EE, so we read the raw keys directly and avoid the
# heavy metric/eval package imports so this attack module stays off-GPU-importable).
_EE_KEY = "robot0_eef_pos"
_POS_SUFFIX = "_pos"


def first_reach_step(distances: Sequence[float], radius: float) -> int | None:
    """First step index whose EE↔distractor distance is at/inside ``radius``.

    The pre-registered reach step = the first frame the clean adversary rollout
    enters the distractor region (``distance <= radius``); the K teacher-forcing
    frames are sampled over ``[0, reach_step]`` so they cover the *approach*, not
    the post-arrival dwell.

    Args:
        distances: Per-step EE↔distractor distances (metres), in rollout order.
        radius: The engagement radius (metres).

    Returns:
        The first index with ``distances[i] <= radius``, or ``None`` if the
        rollout never enters the region (the attack target is then ill-posed).
    """
    for i, d in enumerate(distances):
        if float(d) <= radius:
            return i
    return None


def sample_frame_indices(reach_step: int, n_frames: int) -> list[int]:
    """Pre-registered ``n_frames`` step indices evenly spaced over ``[0, reach_step]``.

    ``np.linspace`` includes both endpoints exactly, then round-half-to-even to
    integer step indices. Pre-registered (not tuned on ``approach_asr``) and logged,
    so the demonstration frames are reproducible from the reach step alone.
    """
    grid = np.linspace(0, reach_step, n_frames)
    return [int(x) for x in np.round(grid).astype(int)]


@dataclass(frozen=True)
class FrameTarget:
    """One demonstration frame + its teacher-forcing target (Task 1).

    Attributes:
        image: ``uint8[256, 256, 3]`` — the raw policy input; GCG re-derives
            ``pixel_values`` from it via the processor at attack time (small artifact).
        target_action_ids: ``int64[7]`` — the policy's greedy action-token ids
            ``a*_t = π(image_t, adv_instruction)`` at this frame.
        frame_index: The step index of this frame in the clean adversary rollout.
        ee_distractor_m: EE↔distractor distance at this frame (metres; provenance).
    """

    image: np.ndarray
    target_action_ids: np.ndarray
    frame_index: int
    ee_distractor_m: float


@dataclass(frozen=True)
class TrajectoryDemo:
    """The ``K`` demonstration frames along the clean adversary approach (Task 1)."""

    frames: tuple[FrameTarget, ...]


def capture_trajectory(
    model: Any,
    processor: Any,
    *,
    env: Any,
    init_state: Any,
    adv_instruction: str,
    distractor_object: str,
    radius: float,
    codec: ActionCodec,
    action_vocab_size: int,
    device: Any,
    n_frames: int,
    step_fn: Callable[[Any], tuple[np.ndarray, np.ndarray]],
    dummy_action: Any,
    num_steps_wait: int = 10,
    max_steps: int = 280,
) -> TrajectoryDemo:
    """Capture ``n_frames`` teacher-forcing targets along the clean adversary approach.

    Rolls the closed-loop **adversary-instruction** episode (no suffix), retaining each
    step's raw image + EE↔distractor distance, then samples ``n_frames`` frames evenly
    over ``[0, reach_step]`` (:func:`sample_frame_indices`, pre-registered) where
    ``reach_step`` is the first frame the EE enters the distractor region
    (:func:`first_reach_step`). For each sampled frame it captures the policy's greedy
    7 action-token ids via :func:`~evasion_tax.attack.semantic_target.build_semantic_target`
    — the multi-frame GCG teacher-forcing targets (Task 2). DM-3: this feeds the attacker
    only; no detector threshold is derived from it.

    The per-step render+policy call is injected as ``step_fn(obs) -> (image, action)`` so
    the loop is testable off-GPU with a fake env/model; the CLI
    ``scripts/capture_adversary_trajectory.py`` supplies the real LIBERO/OpenVLA seam
    (matching ``run_episode``'s pipeline so this rollout reproduces the reach-floor).

    Raises:
        ValueError: If the adversary rollout never enters the distractor region — the
            attack target is ill-posed (no achievable trajectory to teacher-force).
    """
    # Reset + settle, inlined to mirror rollout_runner.reset_and_settle (avoids importing
    # the eval package — matplotlib — into this off-GPU-testable attack module).
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(dummy_action)

    distractor_key = f"{distractor_object}{_POS_SUFFIX}"
    images: list[np.ndarray] = []
    distances: list[float] = []
    for _ in range(max_steps):
        image, action = step_fn(obs)
        ee = np.asarray(obs[_EE_KEY], dtype=float)
        pos = np.asarray(obs[distractor_key], dtype=float)
        images.append(np.asarray(image, dtype=np.uint8))
        distances.append(float(np.linalg.norm(ee - pos)))
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        if done:
            break

    reach_step = first_reach_step(distances, radius)
    if reach_step is None:
        raise ValueError(
            f"adversary rollout never entered the {distractor_object!r} region "
            f"(radius={radius}) in {len(distances)} steps: the redirect target is "
            f"ill-posed (min distance {min(distances):.4f} m > radius)."
        )
    indices = sample_frame_indices(reach_step, n_frames)
    frames = tuple(
        FrameTarget(
            image=images[idx],
            target_action_ids=build_semantic_target(
                model,
                processor,
                image=images[idx],
                adv_instruction=adv_instruction,
                action_vocab_size=action_vocab_size,
                codec=codec,
                device=device,
            ).target_action_ids,
            frame_index=idx,
            ee_distractor_m=distances[idx],
        )
        for idx in indices
    )
    return TrajectoryDemo(frames=frames)


def save_trajectory_demo(demo: TrajectoryDemo, path: str | Path) -> None:
    """Persist the demonstration frames to a compact ``.npz`` (the quarantined artifact).

    Stores the raw ``uint8`` images (not ``pixel_values`` — GCG re-derives those at
    attack time, keeping the artifact small), the per-frame ``int64`` target ids, the
    step indices, and the EE↔distractor distances. Round-trips with
    :func:`load_trajectory_demo` (the schema Task 3's builder loads).
    """
    frames = demo.frames
    np.savez(
        path,
        images=np.stack([np.asarray(f.image, dtype=np.uint8) for f in frames]),
        target_action_ids=np.stack(
            [np.asarray(f.target_action_ids, dtype=np.int64) for f in frames]
        ),
        frame_indices=np.asarray([f.frame_index for f in frames], dtype=np.int64),
        ee_distractor_m=np.asarray([f.ee_distractor_m for f in frames], dtype=np.float64),
    )


def load_trajectory_demo(path: str | Path) -> TrajectoryDemo:
    """Load a :class:`TrajectoryDemo` from a :func:`save_trajectory_demo` ``.npz``."""
    with np.load(path) as data:
        images = data["images"]
        ids = data["target_action_ids"]
        frame_indices = data["frame_indices"]
        ee_distractor_m = data["ee_distractor_m"]
        frames = tuple(
            FrameTarget(
                image=np.asarray(images[i], dtype=np.uint8),
                target_action_ids=np.asarray(ids[i], dtype=np.int64),
                frame_index=int(frame_indices[i]),
                ee_distractor_m=float(ee_distractor_m[i]),
            )
            for i in range(len(frame_indices))
        )
    return TrajectoryDemo(frames=frames)

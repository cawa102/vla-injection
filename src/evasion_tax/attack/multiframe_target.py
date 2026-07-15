"""Multi-frame GCG target builder (Task 3, Tier B).

Loads a Task-1 :class:`~evasion_tax.attack.trajectory_demo.TrajectoryDemo` (the
quarantined adversary demonstration) and constructs the Task-2 multi-frame
:class:`~evasion_tax.attack.gcg_openvla.OpenVlaGcgTarget` that teacher-forces one frozen
suffix against the K approach frames. Each frame's target ids are re-validated against the
**action** range before construction — the artifact is adversarial-derived (quarantined),
so the builder never trusts it blind (defence-in-depth over the Task-1 capture check).

The body is torch-free (delegates tensor handling to the model/processor system boundary,
exactly like ``semantic_target``), so it is unit-testable off-GPU with a fake model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
from evasion_tax.attack.semantic_target import _validate_action_ids
from evasion_tax.attack.trajectory_demo import TrajectoryDemo
from evasion_tax.policy.action_codec import ActionCodec


def build_multiframe_target(
    model: Any,
    processor: Any,
    *,
    trajectory: TrajectoryDemo,
    instruction: str,
    suffix_len: int,
    device: Any,
    action_vocab_size: int,
    codec: ActionCodec,
    eval_batch: int | None = None,
    match_positions: Any = None,
    reach_fraction: float = 1.0,
) -> OpenVlaGcgTarget:
    """Build the multi-frame :class:`OpenVlaGcgTarget` from a Task-1 trajectory artifact.

    Validates every frame's ``target_action_ids`` against the action-token range
    ``[action_vocab_size - codec.n_bins, action_vocab_size - 1]`` (Codex R1: the action
    range, not the tokenizer suffix vocab), then delegates to
    :meth:`OpenVlaGcgTarget.from_frames`. ``instruction`` is the benign task description (the
    prompt the frozen suffix rides on); the K frames supply the per-frame images + targets.
    """
    for frame in trajectory.frames:
        _validate_action_ids(
            np.asarray(frame.target_action_ids, dtype=np.int64), action_vocab_size, codec.n_bins
        )
    return OpenVlaGcgTarget.from_frames(
        model,
        processor,
        frames=list(trajectory.frames),
        instruction=instruction,
        suffix_len=suffix_len,
        device=device,
        eval_batch=eval_batch,
        match_positions=match_positions,
        reach_fraction=reach_fraction,
    )

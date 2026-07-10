"""Policy-derived semantic target builder (Task 4, Tier B).

Given a scene image + adversary instruction, capture the policy's **own** greedy
7 action-token ids as the teacher-forcing target for GCG. This is a *single-frame
token-manifold* target — reachable single-frame, but **not** proof of a closed-loop
redirect (the Tier-B world-frame ASR, Task 5, is the closed-loop check).

Why generation and not ``model.predict_action``: ``predict_action`` returns the
*continuous* 7-DoF action, and :class:`ActionCodec` is **decode-only** (no encode
path), so there is no clean action->token-id route. A single prompt-only forward
gives only the *next*-token logits, not 7 future action positions — so we must
actually generate the 7 tokens (Codex R3).

The builder body is intentionally torch-free: it delegates all tensor handling to
the model/processor (system boundaries) and converts outputs via duck-typing, so
the module never pulls torch at import (matching the ``gcg_openvla`` guard) and is
unit-testable off-GPU with a fake model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evasion_tax.attack.gcg_openvla import _PROMPT_PREFIX, _PROMPT_TAIL, _to_pil
from evasion_tax.policy.action_codec import ActionCodec

_N_ACTION_TOKENS = 7


@dataclass(frozen=True)
class SemanticTarget:
    """The policy's own greedy action for an adversary instruction (Tier B target).

    Attributes:
        target_action_ids: ``[7]`` action-token ids (the GCG teacher-forcing target).
        target_action: ``[7]`` un-normalized action (``codec.decode``, for logging /
            the region), *not* the source of truth for the optimizer.
    """

    target_action_ids: np.ndarray
    target_action: np.ndarray


def build_semantic_target(
    model,
    processor,
    *,
    image,
    adv_instruction: str,
    action_vocab_size: int,
    codec: ActionCodec,
    device,
) -> SemanticTarget:
    """Capture the policy's greedy 7 action-token ids for ``adv_instruction``.

    Runs a greedy generation (``do_sample=False``) of ``(image, adv_instruction)``
    with the ``OpenVlaGcgTarget`` prompt template and captures the 7 generated ids.
    Validates every id falls in the **action** range
    ``[action_vocab_size - codec.n_bins, action_vocab_size - 1]`` (Codex R1: the
    action range, not the tokenizer's suffix vocab), then decodes for the logged
    un-normalized action.
    """
    prompt = _PROMPT_PREFIX.format(instruction=adv_instruction) + _PROMPT_TAIL
    inputs = processor(prompt, _to_pil(image))
    if hasattr(inputs, "to"):
        # OpenVLA's vision backbone runs in the model dtype (bf16); the processor emits
        # float32 pixel_values, so cast the whole batch to model.dtype exactly as OpenVLA's
        # own get_vla_action does (`.to(DEVICE, dtype=bf16)`). HF BatchFeature.to casts only
        # the float tensors, leaving input_ids/attention_mask integer. Without this,
        # model.generate raises "Input type (float) and bias type (BFloat16) should be the
        # same" at the vision conv (box-verified 2026-07-10); gcg_openvla casts identically.
        inputs = inputs.to(device, dtype=model.dtype)
    generated = model.generate(
        **inputs, max_new_tokens=_N_ACTION_TOKENS, do_sample=False
    )
    ids = _to_numpy(generated[0, -_N_ACTION_TOKENS:]).astype(np.int64)
    _validate_action_ids(ids, action_vocab_size, codec.n_bins)
    target_action = np.asarray(codec.decode(ids), dtype=float)
    return SemanticTarget(target_action_ids=ids, target_action=target_action)


def _to_numpy(x) -> np.ndarray:
    """Convert a torch tensor (CPU or CUDA) or array-like to a NumPy array.

    Duck-typed so the builder never imports torch: a torch tensor exposes
    ``detach``/``cpu``; a NumPy array does not, so it passes straight through.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _validate_action_ids(ids: np.ndarray, action_vocab_size: int, n_bins: int) -> None:
    if ids.shape != (_N_ACTION_TOKENS,):
        raise ValueError(f"expected {_N_ACTION_TOKENS} action ids, got shape {ids.shape}")
    lo, hi = action_vocab_size - n_bins, action_vocab_size - 1
    if not (np.all(ids >= lo) and np.all(ids <= hi)):
        raise ValueError(
            f"generated ids {ids.tolist()} fall outside the action-token range "
            f"[{lo}, {hi}] (action_vocab_size={action_vocab_size}, n_bins={n_bins})"
        )

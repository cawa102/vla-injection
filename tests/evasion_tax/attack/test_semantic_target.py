"""Tests for the policy-derived semantic target builder (Task 4, Tier B).

``build_semantic_target`` captures the policy's OWN greedy 7 action-token ids for
an adversary instruction (the teacher-forcing target for GCG). The GPU model is a
system boundary and is faked here; the builder body is deliberately torch-free
(duck-typed tensor handling), so these tests never import torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from evasion_tax.attack.semantic_target import SemanticTarget, build_semantic_target
from evasion_tax.policy.action_codec import ActionCodec

_ACTION_VOCAB = 32000
_N_BINS = 256
# 7 ids inside the action range [action_vocab-256, action_vocab-1] = [31744, 31999].
_ACTION_IDS = [31744, 31800, 31900, 31999, 31850, 31760, 31990]
_IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


def _codec() -> ActionCodec:
    # mask all-False -> decode returns bin centres directly (a clean normalized space).
    return ActionCodec(
        q01=(-1.0,) * 7, q99=(1.0,) * 7, mask=(False,) * 7,
        vocab_size=_ACTION_VOCAB, n_bins=_N_BINS,
    )


class _FakeProcessor:
    """Stand-in for the OpenVLA processor: returns a plain-dict batch of inputs."""

    def __init__(self, prompt_len: int) -> None:
        self._prompt_len = prompt_len
        self.calls: list[str] = []

    def __call__(self, prompt, image):
        self.calls.append(prompt)
        return {
            "input_ids": np.zeros((1, self._prompt_len), dtype=np.int64),
            "pixel_values": np.zeros((1, 3, 2, 2), dtype=np.float32),
        }


class _FakeInputs(dict):
    """Dict-like batch that records ``.to(device, dtype=...)`` (HF BatchFeature stand-in).

    A ``dict`` subclass so ``model.generate(**inputs, ...)`` still unpacks it, while the
    ``.to`` method lets a test assert the dtype cast (the box-2026-07-10 regression).
    """

    def __init__(self, data) -> None:
        super().__init__(data)
        self.to_calls: list[tuple] = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class _FakeProcessorWithTo(_FakeProcessor):
    """Like ``_FakeProcessor`` but returns a ``_FakeInputs`` exposing ``.to``."""

    def __init__(self, prompt_len: int) -> None:
        super().__init__(prompt_len)
        self.last_inputs: _FakeInputs | None = None

    def __call__(self, prompt, image):
        base = super().__call__(prompt, image)
        self.last_inputs = _FakeInputs(base)
        return self.last_inputs


class _FakeModel:
    """Greedy generation returns [prompt ⊕ the 7 action ids]; predict_action decodes them."""

    def __init__(self, action_ids, codec) -> None:
        self._action_ids = np.asarray(action_ids, dtype=np.int64)
        self._codec = codec
        self.dtype = "model-dtype-sentinel"  # stands in for torch.bfloat16

    def generate(self, *, input_ids, max_new_tokens, do_sample, **_kw):
        assert do_sample is False and max_new_tokens == 7
        full = np.concatenate([np.asarray(input_ids)[0], self._action_ids])
        return full[None, :]

    def predict_action(self, *_a, **_kw):
        return self._codec.decode(self._action_ids)


def test_captures_seven_action_token_ids_in_action_range():
    codec = _codec()
    tgt = build_semantic_target(
        _FakeModel(_ACTION_IDS, codec),
        _FakeProcessor(prompt_len=5),
        image=_IMAGE,
        adv_instruction="pick up the blue cup",
        action_vocab_size=_ACTION_VOCAB,
        codec=codec,
        device="cpu",
    )
    assert isinstance(tgt, SemanticTarget)
    assert tgt.target_action_ids.shape == (7,)
    lo, hi = _ACTION_VOCAB - _N_BINS, _ACTION_VOCAB - 1
    assert np.all(tgt.target_action_ids >= lo) and np.all(tgt.target_action_ids <= hi)
    assert np.array_equal(tgt.target_action_ids, np.asarray(_ACTION_IDS))
    assert tgt.target_action.shape == (7,)


def _build(model, processor, codec, **overrides):
    kwargs = dict(
        image=_IMAGE,
        adv_instruction="pick up the blue cup",
        action_vocab_size=_ACTION_VOCAB,
        codec=codec,
        device="cpu",
    )
    kwargs.update(overrides)
    return build_semantic_target(model, processor, **kwargs)


def test_greedy_capture_is_deterministic():
    codec = _codec()
    a = _build(_FakeModel(_ACTION_IDS, codec), _FakeProcessor(5), codec)
    b = _build(_FakeModel(_ACTION_IDS, codec), _FakeProcessor(5), codec)
    assert np.array_equal(a.target_action_ids, b.target_action_ids)
    assert np.array_equal(a.target_action, b.target_action)


def test_captured_ids_decode_to_the_policy_greedy_action():
    # Faithfulness (Codex R2): decode(captured ids) matches predict_action within one
    # bin -- proving the captured ids ARE the policy's greedy action. predict_action
    # is the CHECK here, never the source (the builder must not call it).
    codec = _codec()
    model = _FakeModel(_ACTION_IDS, codec)
    tgt = _build(model, _FakeProcessor(5), codec)
    predicted = model.predict_action(_IMAGE, "pick up the blue cup", do_sample=False)
    one_bin = 2.0 / (codec.n_bins - 1)
    assert np.allclose(codec.decode(tgt.target_action_ids), predicted, atol=one_bin)


def test_inputs_cast_to_model_dtype_for_bf16_vision_backbone():
    # Regression (box 2026-07-10): OpenVLA's vision backbone is bf16 but the processor
    # emits float32 pixel_values, so build_semantic_target MUST cast the batch to
    # model.dtype (matches openvla_utils.get_vla_action `.to(DEVICE, dtype=bf16)`), else
    # model.generate raises "Input type (float) and bias type (BFloat16) should be the same".
    codec = _codec()
    proc = _FakeProcessorWithTo(5)
    _build(_FakeModel(_ACTION_IDS, codec), proc, codec)
    assert proc.last_inputs is not None and proc.last_inputs.to_calls, "inputs.to(...) never called"
    _device, dtype = proc.last_inputs.to_calls[-1]
    assert dtype == "model-dtype-sentinel", f"inputs not cast to model.dtype (got {dtype!r})"


def test_ids_outside_action_range_raise():
    # Validate against the ACTION range, not the tokenizer's suffix vocab (Codex R1).
    codec = _codec()
    bad_ids = [10, 20, 30, 40, 50, 60, 70]  # far below [31744, 31999]
    with pytest.raises(ValueError, match="action-token range"):
        _build(_FakeModel(bad_ids, codec), _FakeProcessor(5), codec)

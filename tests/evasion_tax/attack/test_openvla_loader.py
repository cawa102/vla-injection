"""Tests for the shared frozen-OpenVLA loader/target-builder (bench Task 4).

The real load + target build against bf16 OpenVLA-7B is GPU-only and guarded (torch /
transformers / PIL imported inside the functions). Off-GPU we pin only the structural
contract the bench driver (Task 6) and the microbench both depend on: the module imports
torch-free, and the public functions keep the exact signatures the call sites pass.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import evasion_tax.attack.openvla_loader as loader_mod
from evasion_tax.attack.openvla_loader import (
    DEFAULT_INSTRUCTION,
    build_target,
    is_flash_attn_load_error,
    load_frozen_openvla,
    load_openvla_policy,
    load_openvla_with_attn_fallback,
    quantization_record,
    target_action_ids_for_vocab,
)


def _fake_loader(fail_on=()):
    """A torch-free stand-in for ``load_openvla_policy`` that records the attn impls it saw.

    Raises ``ImportError("flash_attn ...")`` for any impl in ``fail_on``; otherwise returns a
    real :class:`OpenVlaLoadRecord` whose ``attn_implementation`` echoes the impl it ran with.
    """
    calls: list[str] = []

    def loader(torch_mod, model_id, device, attn_impl, *, precision="bf16"):
        calls.append(attn_impl)
        if attn_impl in fail_on:
            raise ImportError(f"flash_attn extension missing for {attn_impl}")
        record = quantization_record(
            precision=precision,
            torch_dtype="bfloat16",
            device=str(device),
            attn_impl=attn_impl,
        )
        return ("model", "processor", record)

    return loader, calls


def test_fallback_retries_with_sdpa_on_flash_load_error(monkeypatch, capsys):
    loader, calls = _fake_loader(fail_on=("flash_attention_2",))
    monkeypatch.setattr(loader_mod, "load_openvla_policy", loader)

    _model, _proc, record = load_openvla_with_attn_fallback(
        None, "openvla/x", "cuda:0", "flash_attention_2"
    )

    assert calls == ["flash_attention_2", "sdpa"]
    assert record.attn_implementation == "sdpa"
    assert "sdpa" in capsys.readouterr().err


def test_fallback_returns_requested_impl_when_load_succeeds(monkeypatch):
    loader, calls = _fake_loader()  # nothing fails
    monkeypatch.setattr(loader_mod, "load_openvla_policy", loader)

    _model, _proc, record = load_openvla_with_attn_fallback(
        None, "openvla/x", "cuda:0", "flash_attention_2"
    )

    assert calls == ["flash_attention_2"]  # no retry when the requested impl loads
    assert record.attn_implementation == "flash_attention_2"


def test_fallback_does_not_retry_when_sdpa_itself_fails(monkeypatch):
    loader, calls = _fake_loader(fail_on=("sdpa",))
    monkeypatch.setattr(loader_mod, "load_openvla_policy", loader)

    with pytest.raises(ImportError):
        load_openvla_with_attn_fallback(None, "openvla/x", "cuda:0", "sdpa")

    assert calls == ["sdpa"]  # the controlled variable was already sdpa; no second attempt


def test_fallback_propagates_non_flash_error_without_retry(monkeypatch):
    calls: list[str] = []

    def loader(torch_mod, model_id, device, attn_impl, *, precision="bf16"):
        calls.append(attn_impl)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(loader_mod, "load_openvla_policy", loader)

    with pytest.raises(RuntimeError, match="out of memory"):
        load_openvla_with_attn_fallback(None, "openvla/x", "cuda:0", "flash_attention_2")

    assert calls == ["flash_attention_2"]  # not a flash error -> propagate, no fallback


def test_is_flash_attn_load_error_true_for_import_error_regardless_of_message():
    assert is_flash_attn_load_error(ImportError("dlopen failed")) is True


def test_is_flash_attn_load_error_true_for_flash_message_case_insensitive():
    assert is_flash_attn_load_error(ValueError("FlashAttention2 is not installed")) is True


def test_is_flash_attn_load_error_false_for_unrelated_runtime_error():
    assert is_flash_attn_load_error(RuntimeError("CUDA out of memory")) is False


def test_loader_module_imports_no_torch_stack_at_module_top():
    # Same invariant as the gcg_openvla seam: importable on the CUDA-free mac.
    import evasion_tax.attack.openvla_loader as loader_mod

    assert loader_mod.__file__ is not None
    tree = ast.parse(Path(loader_mod.__file__).read_text())
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert {"torch", "transformers", "PIL"}.isdisjoint(top_level_imports)


def test_public_signatures_match_the_call_sites():
    # The signatures the microbench / bench driver pass — kept identical to the moved
    # private helpers so the move is behaviour-preserving (a drift would break callers
    # whose GPU path no test executes).
    assert list(inspect.signature(load_frozen_openvla).parameters) == [
        "torch_mod",
        "model_id",
        "device",
        "attn_impl",
    ]
    assert list(inspect.signature(build_target).parameters) == [
        "np_mod",
        "model",
        "processor",
        "device",
        "instruction",
        "suffix_len",
        "seed",
        "eval_batch",  # forwarded to OpenVlaGcgTarget so the bench's sw=512 fits 24 GB (DE-7)
        "target_action_ids",  # surrogate transfer evaluator passes artifact target tokens
    ]
    assert list(inspect.signature(load_openvla_policy).parameters) == [
        "torch_mod",
        "model_id",
        "device",
        "attn_impl",
        "precision",
    ]


def test_build_target_eval_batch_defaults_to_none():
    # Back-compat: the microbench constructs targets without eval_batch (the single-forward
    # path), so the added knob must default to None.
    assert inspect.signature(build_target).parameters["eval_batch"].default is None
    assert inspect.signature(build_target).parameters["target_action_ids"].default is None


def test_default_instruction_is_the_step_5_5_target():
    assert DEFAULT_INSTRUCTION == "pick up the red block"


def test_target_action_ids_for_vocab_matches_step_5_5_bins():
    import numpy as np

    assert target_action_ids_for_vocab(np, 1000).tolist() == [
        967,
        935,
        903,
        871,
        839,
        807,
        775,
    ]


def test_quantization_record_bf16_has_no_bnb_config():
    record = quantization_record(
        precision="bf16",
        torch_dtype="bfloat16",
        device="cuda:0",
        attn_impl="sdpa",
    )
    assert record.precision == "bf16"
    assert record.quantization_config is None
    assert record.device_map is None


def test_quantization_record_int8_records_bitsandbytes_config():
    record = quantization_record(
        precision="int8",
        torch_dtype="bfloat16",
        device="cuda:0",
        attn_impl="sdpa",
    )
    assert record.quantization_config == {"load_in_8bit": True}
    assert record.device_map == {"": "cuda:0"}


def test_quantization_record_nf4_records_bf16_compute():
    record = quantization_record(
        precision="nf4_4bit",
        torch_dtype="bfloat16",
        device="cuda:0",
        attn_impl="sdpa",
    )
    assert record.quantization_config == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def test_quantization_record_rejects_unknown_precision():
    import pytest

    with pytest.raises(ValueError):
        quantization_record(
            precision="fp8",  # type: ignore[arg-type]
            torch_dtype="bfloat16",
            device="cuda:0",
            attn_impl="sdpa",
        )

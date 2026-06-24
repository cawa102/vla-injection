"""Shared frozen-bf16 OpenVLA-7B loader + GCG target builder (bench Task 4) — GPU-guarded.

The exact frozen-bf16 load and ``OpenVlaGcgTarget`` build the D8 micro-bench
(``scripts/microbench_gcg.py``) and the early-stop bench driver
(``scripts/bench_early_stop.py``) both need. Extracted here verbatim so the two
scripts share one definition instead of a second copy that can drift.

**GPU-only / guarded.** ``torch`` / ``transformers`` / ``PIL`` are imported inside the
functions, never at module top, so this module stays importable on a CUDA-free host
exactly like ``gcg_openvla`` and the smoke scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Pinned bench knobs (one variable at a time; recorded in the run header).
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)  # reuse the step-5.5 arbitrary target.
DEFAULT_INSTRUCTION = "pick up the red block"
OpenVlaPrecision = Literal["bf16", "int8", "nf4_4bit"]
_PRECISIONS: tuple[OpenVlaPrecision, ...] = ("bf16", "int8", "nf4_4bit")


@dataclass(frozen=True)
class OpenVlaLoadRecord:
    """JSON-safe record of the exact OpenVLA precision/load path used for a run."""

    precision: OpenVlaPrecision
    attn_implementation: str
    torch_dtype: str
    low_cpu_mem_usage: bool
    trust_remote_code: bool
    device: str
    device_map: dict[str, str] | None
    quantization_config: dict[str, Any] | None


def _check_precision(precision: str) -> OpenVlaPrecision:
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {_PRECISIONS}, got {precision!r}")
    return precision  # type: ignore[return-value]


def target_action_ids_for_vocab(np_mod, vocab_size: int):
    """Return the fixed step-5.5 target action-token ids for a tokenizer vocab."""
    return np_mod.array([int(vocab_size) - 1 - b for b in _TARGET_BINS], dtype=np_mod.int64)


def quantization_record(
    *,
    precision: OpenVlaPrecision,
    torch_dtype: str,
    device: str,
    attn_impl: str,
) -> OpenVlaLoadRecord:
    """Build the reproducibility record for an OpenVLA load without importing transformers.

    The scripts log this record before/alongside the run output so the exact surrogate
    precision path is visible in write-once results. ``load_openvla_policy`` constructs
    the matching real ``BitsAndBytesConfig`` for the quantized paths.
    """
    precision = _check_precision(precision)
    if precision == "bf16":
        return OpenVlaLoadRecord(
            precision=precision,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device=device,
            device_map=None,
            quantization_config=None,
        )
    if precision == "int8":
        qcfg: dict[str, Any] | None = {"load_in_8bit": True}
    else:
        qcfg = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch_dtype,
        }
    return OpenVlaLoadRecord(
        precision=precision,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device=device,
        device_map={"": device},
        quantization_config=qcfg,
    )


def build_target(
    np_mod,
    model,
    processor,
    device,
    *,
    instruction,
    suffix_len,
    seed,
    eval_batch=None,
    target_action_ids=None,
):
    """Build one :class:`OpenVlaGcgTarget` (deterministic dummy image, fixed target).

    ``eval_batch`` is forwarded to the target's :meth:`loss_of` candidate-eval chunking
    (DE-7): ``None`` (the microbench default) keeps the single-forward path; the bench
    passes an int so a ``search_width=512`` attack fits 24 GB.
    """
    from PIL import Image  # type: ignore[import-not-found]  # noqa: E402

    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget  # noqa: E402

    rng = np_mod.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np_mod.uint8))
    vocab_size = int(processor.tokenizer.vocab_size)
    if target_action_ids is None:
        target_action_ids = target_action_ids_for_vocab(np_mod, vocab_size)
    return OpenVlaGcgTarget(
        model,
        processor,
        image=image,
        instruction=instruction,
        suffix_len=suffix_len,
        target_action_ids=target_action_ids,
        device=device,
        eval_batch=eval_batch,
    )


def load_openvla_policy(
    torch_mod,
    model_id,
    device,
    attn_impl,
    *,
    precision: OpenVlaPrecision = "bf16",
):
    """Load + freeze OpenVLA-7B at a controlled precision.

    ``precision="bf16"`` preserves the original frozen-bf16 behavior exactly. The
    quantized paths are surrogate-only: they use Transformers bitsandbytes
    ``BitsAndBytesConfig`` and device-map placement, and must not be used for the bf16
    victim in transfer evaluation.
    """
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    precision = _check_precision(precision)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    record = quantization_record(
        precision=precision,
        torch_dtype=str(torch_mod.bfloat16).replace("torch.", ""),
        device=str(device),
        attn_impl=attn_impl,
    )
    kwargs: dict[str, Any] = {
        "attn_implementation": attn_impl,
        "torch_dtype": torch_mod.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": str(device)}
    elif precision == "nf4_4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_mod.bfloat16,
        )
        kwargs["device_map"] = {"": str(device)}

    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    if precision == "bf16":
        model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, processor, record


def load_frozen_openvla(torch_mod, model_id, device, attn_impl):
    """Load + freeze bf16 OpenVLA-7B (the step-5.5 setup)."""
    model, processor, _record = load_openvla_policy(
        torch_mod,
        model_id,
        device,
        attn_impl,
        precision="bf16",
    )
    return model, processor

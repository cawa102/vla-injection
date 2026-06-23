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

# Pinned bench knobs (one variable at a time; recorded in the run header).
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)  # reuse the step-5.5 arbitrary target.
DEFAULT_INSTRUCTION = "pick up the red block"


def build_target(
    np_mod, model, processor, device, *, instruction, suffix_len, seed, eval_batch=None
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
    target_action_ids = np_mod.array([vocab_size - 1 - b for b in _TARGET_BINS], dtype=np_mod.int64)
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


def load_frozen_openvla(torch_mod, model_id, device, attn_impl):
    """Load + freeze bf16 OpenVLA-7B (the step-5.5 setup)."""
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation=attn_impl,
        torch_dtype=torch_mod.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, processor

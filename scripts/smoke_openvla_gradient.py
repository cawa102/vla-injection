#!/usr/bin/env python3
"""CSB bring-up — can we backward to the input embeddings under bf16 + flash-attn?

The GCG prerequisite. Step 3 verified the *forward* (bf16 load -> valid action);
step 6 is the GCG attack, whose whole machinery rests on one unverified assumption:
that an autograd ``.backward()`` reaches the **input token embeddings** and yields a
**finite, non-zero** gradient *through the flash-attention-2 backward kernels in
bf16*. flash-attn implements its own backward, separate from the (already-checked)
forward; this is the seam that has to hold for GCG. This smoke de-risks exactly
that, with no GCG search loop:

    load bf16 + flash_attn2 -> CE loss to a target action-token sequence ->
    .backward() -> input-embedding gradient is finite AND non-zero -> record the
    gradient norm and peak VRAM.

If this passes, the "gradients are obtainable" premise of step 6 is established.

**Mechanism (why a hooked ``delta`` and not a literal ``inputs_embeds`` arg).**
OpenVLA's multimodal ``forward`` does *not* accept an external ``inputs_embeds``:
it always computes ``get_input_embeddings()(input_ids)`` itself and fuses the image
patches around the result. So to read the input-embedding gradient faithfully —
without re-implementing that fusion — we (a) **freeze every weight** and (b) install
a forward hook on the token-embedding module that adds a zero ``delta`` leaf with
``requires_grad=True``. ``delta`` *is* the perturbation to ``inputs_embeds``; after
backward, ``delta.grad == d(loss)/d(inputs_embeds)`` — the exact gradient GCG needs —
obtained through the **supported** ``input_ids + pixel_values + labels`` path.
Freezing the weights is also why the VRAM number is meaningful: it matches the real
GCG setup (gradient only on the input, none on the 7B of weights), so peak VRAM here
is representative of a GCG backward step rather than inflated by ~14 GB of weight
gradients that would never exist in the attack.

Locally (no CUDA) it **guards**: prints the GPU-node requirement and exits non-zero
rather than silently no-op (the shared guard the other GPU scripts use). On the box
it loads the model, runs one backward, checks the gradient, records peak VRAM, and
logs a write-once smoke record to ``results/_smoke/`` — bring-up smoke output, not a
registered result (playbook §8 / plan.md reproducibility note).

Verify gate: the loss is finite, the input-embedding gradient is **finite and
non-zero**, and peak VRAM is below the card's capacity (no OOM).

Usage (on the box, after the step-3 env bring-up; flash-attn wheel installed — L5):
    uv run python scripts/smoke_openvla_gradient.py                       # default: flash_attention_2
    uv run python scripts/smoke_openvla_gradient.py --attn-impl sdpa      # sanity cross-check
"""

from __future__ import annotations

import argparse
import math
import sys

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402
from evasion_tax.repro import RunLogger, seed_everything  # noqa: E402

STAGE = "smoke_openvla_gradient"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
# HF model-card prompt template; the instruction text is substituted verbatim.
_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
# OpenVLA 7-DoF action: 7 discrete tokens drawn from the last `n_bins` of the vocab.
_ACTION_DIM = 7
# Deterministic target action: 7 bin indices spread across [0, n_bins-1]. Arbitrary
# but distinct from any benign argmax, so the CE loss — and thus the gradient — is
# unambiguously non-zero (the adversary-target shape GCG uses). Token id for a bin is
# `tokenizer.vocab_size - 1 - bin` (action_codec.py: id = vocab_size - discretized).
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openvla/openvla-7b", help="HF model id")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="attention backend (default flash_attention_2: this smoke tests its backward)",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the red block",
        help="dummy instruction for the forward pass",
    )
    parser.add_argument("--seed", type=int, default=42, help="pinned seed")
    parser.add_argument(
        "--results-root",
        default="results/_smoke",
        help="write-once results root (bring-up smoke)",
    )
    args = parser.parse_args(argv)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    # Heavy / GPU-only imports happen here, after the guard, so the script stays
    # importable on a CUDA-free host.
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"[{STAGE}] loading {args.model} (bf16, attn={args.attn_impl}) on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # GCG setup: freeze every weight so the only leaf carrying grad is the input-side
    # `delta`. This is both faithful (GCG never differentiates the weights) and the
    # reason peak VRAM stays representative (no ~14 GB of bf16 weight-grad buffers).
    vla.requires_grad_(False)
    vla.eval()

    # Deterministic dummy 224x224 RGB image (the processor resizes/crops as needed).
    rng = np.random.default_rng(args.seed)
    image = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
    prompt = _PROMPT_TEMPLATE.format(instruction=args.instruction)

    # BatchFeature.to(dtype=...) casts only floating tensors -> pixel_values become
    # bf16 while input_ids/attention_mask stay long (same idiom as the load smoke).
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    prompt_ids = inputs["input_ids"]  # long [1, P]
    prompt_attn = inputs["attention_mask"]  # [1, P]
    pixel_values = inputs["pixel_values"]  # bf16
    prompt_len = int(prompt_ids.shape[1])

    # Teacher-forced action span: append the target action tokens and mask the prompt
    # (-100) so OpenVLA's built-in causal-LM loss is the CE over the 7 action tokens —
    # the same objective GCG drives. OpenVLA inserts -100 for the image-patch positions
    # internally, so labels only need to align with the text input_ids.
    vocab_size = int(processor.tokenizer.vocab_size)
    target_ids = torch.tensor(
        [[vocab_size - 1 - b for b in _TARGET_BINS]], device=device, dtype=torch.long
    )  # [1, 7]
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    full_attn = torch.cat([prompt_attn, torch.ones_like(target_ids)], dim=1)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)

    # Inject the requires-grad perturbation to inputs_embeds via a forward hook on the
    # token-embedding module: `delta` is a zero leaf, so the forward value is unchanged
    # but backward populates `delta.grad == d(loss)/d(inputs_embeds)`. Created on the
    # single call OpenVLA makes to the embedding (no generation loop here).
    captured: dict[str, torch.Tensor] = {}

    def _embed_hook(_module, _args, output):  # type: ignore[no-untyped-def]
        if "delta" not in captured:
            captured["delta"] = torch.zeros_like(output, requires_grad=True)
        return output + captured["delta"]

    hook_handle = vla.get_input_embeddings().register_forward_hook(_embed_hook)

    # Do NOT reset peak stats: the "fits one card" gate must span the ~14 GB of bf16
    # weights AND the forward activations AND the backward, so the recorded peak is the
    # full cost of one GCG-style backward step with the model resident.
    try:
        out = vla(
            input_ids=full_ids,
            attention_mask=full_attn,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
    finally:
        hook_handle.remove()

    if "delta" not in captured:
        print(f"[{STAGE}] FAIL: embedding hook never fired (no input-embedding tensor)", file=sys.stderr)
        return 1
    grad = captured["delta"].grad
    if grad is None:
        print(f"[{STAGE}] FAIL: no gradient reached the input embeddings", file=sys.stderr)
        return 1

    # Cast to float32 for the reductions so the bf16 grad cannot overflow/underflow the
    # norm itself. grad shape is [1, P+7, hidden]; prompt slice = the instruction-token
    # region GCG actually perturbs.
    grad_f = grad.detach().float()
    grad_finite = bool(torch.isfinite(grad_f).all())
    grad_norm = float(grad_f.norm())
    grad_norm_prompt = float(grad_f[:, :prompt_len, :].norm())
    grad_max_abs = float(grad_f.abs().max())
    grad_nonzero = grad_norm > 0.0

    loss_val = float(loss.detach().float())
    loss_finite = math.isfinite(loss_val)

    props = torch.cuda.get_device_properties(device)
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / _BYTES_PER_GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
    total_gib = props.total_memory / _BYTES_PER_GIB
    fits_one_card = peak_reserved_gib < total_gib

    result = {
        "stage": STAGE,
        "model": args.model,
        "attn_impl": args.attn_impl,
        "dtype": "bfloat16",
        "device": args.device,
        "device_name": props.name,
        "instruction": args.instruction,
        "seed": args.seed,
        "weights_frozen": True,
        "prompt_len": prompt_len,
        "action_dim": _ACTION_DIM,
        "target_action_token_ids": target_ids.squeeze(0).tolist(),
        "loss": loss_val,
        "loss_finite": loss_finite,
        "grad_shape": list(grad_f.shape),
        "grad_finite": grad_finite,
        "grad_nonzero": grad_nonzero,
        "grad_norm": grad_norm,
        "grad_norm_prompt": grad_norm_prompt,
        "grad_max_abs": grad_max_abs,
        "peak_vram_allocated_gib": round(peak_alloc_gib, 3),
        "peak_vram_reserved_gib": round(peak_reserved_gib, 3),
        "card_total_gib": round(total_gib, 3),
        "fits_one_card": fits_one_card,
    }

    logger = RunLogger(args.results_root)
    handle = logger.start("openvla-gradient-smoke", config=result, seed=args.seed)
    handle.write("smoke_result", result)

    print(f"[{STAGE}] loss (CE to target action): {loss_val:.4f}")
    print(
        f"[{STAGE}] input-embedding grad: norm {grad_norm:.4e} "
        f"(prompt region {grad_norm_prompt:.4e}), max|g| {grad_max_abs:.4e}, "
        f"finite={grad_finite}, nonzero={grad_nonzero}"
    )
    print(
        f"[{STAGE}] peak VRAM: alloc {peak_alloc_gib:.2f} GiB / "
        f"reserved {peak_reserved_gib:.2f} GiB on {props.name} ({total_gib:.1f} GiB)"
    )
    print(f"[{STAGE}] logged -> {handle.dir}")

    ok = loss_finite and grad_finite and grad_nonzero and fits_one_card
    if not ok:
        reason = (
            "loss not finite" if not loss_finite
            else "gradient not finite" if not grad_finite
            else "gradient is all-zero" if not grad_nonzero
            else "peak VRAM exceeded the card"
        )
        print(f"[{STAGE}] FAIL: {reason}", file=sys.stderr)
        return 1
    print(
        f"[{STAGE}] PASS: finite non-zero input-embedding gradient through "
        f"{args.attn_impl} (bf16), fit one card. GCG gradient premise holds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

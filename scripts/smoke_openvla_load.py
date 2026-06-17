#!/usr/bin/env python3
"""CSB bring-up step 3 — load OpenVLA-7B in bf16 and run one dummy forward.

The first model-touching step of the registered-compute bring-up ladder
(``docs/gpu/CSB/plan.md``). It de-risks the OpenVLA wiring on the A5000:

    load bf16 -> one forward on a dummy image+instruction -> a valid 7-DoF
    action vector -> it fit on one 24 GB card.

Locally (no CUDA) it **guards**: prints the GPU-node requirement and exits
non-zero rather than silently no-op (the shared guard the other GPU scripts use).
On the box it loads the model, runs ``predict_action`` on a deterministic dummy
image, validates the action, records peak VRAM, and logs a write-once smoke
record (full repro header) to ``results/_smoke/`` — bring-up smoke output, not a
registered result (playbook §8 / plan.md reproducibility note).

Verify gate (plan.md step 3): a finite 7-DoF action prints **and** peak VRAM is
below the card's capacity (no OOM).

Usage (on the box, after env bring-up steps 1-2):
    uv run python scripts/smoke_openvla_load.py
    uv run python scripts/smoke_openvla_load.py --attn-impl flash_attention_2   # perf path
    uv run python scripts/smoke_openvla_load.py \\
        --model openvla/openvla-7b-finetuned-libero-spatial \\
        --unnorm-key libero_spatial_no_noops                                    # step-4 preview

The base model + ``bridge_orig`` is the documented HF quickstart and the lowest-risk
wiring check; ``--attn-impl`` defaults to ``sdpa`` so step 3 is isolated from the
riskier flash-attn build under CUDA 13.2 (plan.md caveat L5 — verify flash-attn
separately). Recipe verified against the ``openvla/openvla-7b`` HF model card.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402
from evasion_tax.policy import validate_action_vector  # noqa: E402
from evasion_tax.repro import RunLogger, seed_everything  # noqa: E402

STAGE = "smoke_openvla_load"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
# HF model-card prompt template; the instruction text is substituted verbatim.
_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openvla/openvla-7b", help="HF model id")
    parser.add_argument(
        "--unnorm-key",
        default="bridge_orig",
        help="dataset stats key for un-normalising the action",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="attention backend (default sdpa: no flash-attn build needed)",
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

    # Deterministic dummy 224x224 RGB image (the processor resizes/crops as needed).
    rng = np.random.default_rng(args.seed)
    image = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
    prompt = _PROMPT_TEMPLATE.format(instruction=args.instruction)

    # Do NOT reset peak stats here: the "fits on one card" gate must include the
    # ~14 GB of bf16 weights, so peak must span the load + the forward.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)

    action_arr = validate_action_vector(action)

    props = torch.cuda.get_device_properties(device)
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / _BYTES_PER_GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
    total_gib = props.total_memory / _BYTES_PER_GIB
    fits_one_card = peak_reserved_gib < total_gib

    result = {
        "stage": STAGE,
        "model": args.model,
        "unnorm_key": args.unnorm_key,
        "attn_impl": args.attn_impl,
        "dtype": "bfloat16",
        "device": args.device,
        "device_name": props.name,
        "instruction": args.instruction,
        "seed": args.seed,
        "action": action_arr.tolist(),
        "action_dim": int(action_arr.shape[0]),
        "peak_vram_allocated_gib": round(peak_alloc_gib, 3),
        "peak_vram_reserved_gib": round(peak_reserved_gib, 3),
        "card_total_gib": round(total_gib, 3),
        "fits_one_card": fits_one_card,
    }

    logger = RunLogger(args.results_root)
    handle = logger.start("openvla-load-smoke", config=result, seed=args.seed)
    handle.write("smoke_result", result)

    print(f"[{STAGE}] action ({result['action_dim']}-DoF): {action_arr.tolist()}")
    print(
        f"[{STAGE}] peak VRAM: alloc {peak_alloc_gib:.2f} GiB / "
        f"reserved {peak_reserved_gib:.2f} GiB on {props.name} ({total_gib:.1f} GiB)"
    )
    print(f"[{STAGE}] logged -> {handle.dir}")
    if not fits_one_card:
        print(f"[{STAGE}] FAIL: peak VRAM exceeded the card", file=sys.stderr)
        return 1
    print(f"[{STAGE}] PASS: valid {result['action_dim']}-DoF action, fit one card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

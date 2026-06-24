"""On-box smoke: does a quantized OpenVLA surrogate load (concern 1) and produce a
real input-embedding gradient through GCG's hooked backward (concern 2)?

Mirrors the plan Run Order steps 3-4
(``docs/surrogate/plan/2026-06-24-quantized-surrogate-gcg-transfer-plan.md``). Touches
NO LIBERO/EGL/unnorm-key -- it reuses the deterministic dummy 224x224 image that
``build_target`` already builds -- so a failure is unambiguously bitsandbytes, not the
env. The hooked backward in ``OpenVlaGcgTarget.token_gradient`` is the exact path a real
GCG step takes, so a finite + non-zero gradient here means the quantized arm can be
optimized on this box. The D6-9 faithfulness gate additionally catches a non-None but
numerically wrong gradient (concern 3). Every failure mode is recorded, never dropped.

Run from the repo root on the box, in the surrogate venv::

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_quantized_backward.py --precision bf16     # positive control
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_quantized_backward.py --precision int8
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_quantized_backward.py --precision nf4_4bit

``--attn-impl sdpa`` (default) isolates bitsandbytes from flash-attn; pass
``--attn-impl flash_attention_2`` to match the box-verified registered path. Exit code is
0 on PASS, 1 on any FAIL, so the three runs can be chained in a shell ``&&`` gate.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.attack.openvla_loader import build_target, load_openvla_policy  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=["bf16", "int8", "nf4_4bit"], required=True)
    p.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        help="sdpa isolates bnb from flash-attn; flash_attention_2 = the box-verified path",
    )
    p.add_argument("--suffix-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate-samples", type=int, default=8)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    out: dict = {"precision": args.precision, "attn_impl": args.attn_impl}
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]

        out["bitsandbytes_version"] = bnb.__version__
    except Exception as exc:  # noqa: BLE001 - a missing/broken bnb IS the concern-1 result.
        out["bitsandbytes_version"] = None
        out["bnb_import_error"] = f"{type(exc).__name__}: {exc}"

    import torch  # type: ignore[import-not-found]

    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())

    try:
        device = torch.device(args.device)
        model, processor, record = load_openvla_policy(
            torch, args.model, device, args.attn_impl, precision=args.precision
        )
        out["loaded"] = True
        out["device_map"] = record.device_map
        out["quantization_config"] = record.quantization_config

        target = build_target(
            np,
            model,
            processor,
            device,
            instruction="pick up the red block",
            suffix_len=args.suffix_len,
            seed=args.seed,
        )
        # THE concern-2 test: the hooked backward through the (quantized) model.
        grad = target.token_gradient(target.init_suffix_ids())  # [L, vocab]
        out["grad_finite"] = bool(np.isfinite(grad).all())
        out["grad_absmax"] = float(np.abs(grad).max())
        out["grad_nonzero"] = bool(out["grad_absmax"] > 0.0)

        # Concern-3 guard: a non-None but numerically wrong gradient (D6-9).
        report = target.gradient_agrees_with_swaps(
            n_samples=args.gate_samples, rng=np.random.default_rng(args.seed)
        )
        out["faithfulness_passed"] = bool(report.passed)
        out["recommended_mean_delta"] = report.recommended_mean_delta
        out["random_mean_delta"] = report.random_mean_delta
        out["peak_vram_gib"] = round(torch.cuda.max_memory_reserved(device) / (1024**3), 2)

        out["VERDICT"] = (
            "PASS"
            if (out["grad_finite"] and out["grad_nonzero"] and out["faithfulness_passed"])
            else "FAIL: zero/garbage/unfaithful gradient (concern 3)"
        )
    except Exception as exc:  # noqa: BLE001 - quantized-backward failure IS the evidence.
        out.setdefault("loaded", False)
        out["VERDICT"] = "FAIL: exception"
        out["exception"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("VERDICT") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

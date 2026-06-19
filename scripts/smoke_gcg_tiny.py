#!/usr/bin/env python3
"""CSB step 6 — tiny GCG run: does our own harness drive target action tokens?

The harness-works smoke (plan Task 3): 1 task, 1 fixed arbitrary target, a handful
of GCG steps on the step-5.5 loss/gradient seam. It proves the search loop is
**wired faithfully** — it is **NOT** an ASR claim and **NOT** a closed-loop rollout.

**Pass/fail = the WIRING set (D6-3 / D6-9), not the attack effect:**

1. the **seam-faithfulness gate** passes first (D6-9): the projected one-hot
   gradient's sign predicts measured one-token-swap loss deltas (finite difference),
   and the suffix span decodes to the intended adversarial text under the real
   processor;
2. the harness runs ``run_gcg`` to completion;
3. ``loss_of`` on a batch equals per-candidate single evaluation (batched-loss
   equivalence);
4. peak VRAM < 24 GiB (fits one A5000);
5. the optimised suffix is **quarantined** to ``artifacts/untrusted/`` (D6-6) and
   nothing untrusted is committed.

The loss trajectory and whether the target tokens reach argmax / loss < a pinned
threshold on the single example are recorded as an **exploratory** smoke
observation, **not** a gate (D6-3) — an unlucky prompt/target can fail to converge
in a handful of steps without the wiring being wrong.

Locally (no CUDA) it **guards**: prints the GPU-node requirement and exits non-zero
rather than silently no-op (the shared guard the other GPU scripts use). On the box
it loads the model, runs the gate + the tiny search, and logs a **non-registered**
smoke record to ``results/_smoke/`` (bring-up smoke, not a registered result).

Usage (on the box, after step 5.5; flash-attn wheel installed):
    uv run python scripts/smoke_gcg_tiny.py
    uv run python scripts/smoke_gcg_tiny.py --n-steps 20 --search-width 64   # pre-reg. escalation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402

STAGE = "smoke_gcg_tiny"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
_CARD_GIB = 24.0  # one RTX A5000.
# Reuse step-5.5's arbitrary target: 7 bin indices spread across [0, n_bins-1].
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)
_DEFAULT_INSTRUCTION = "pick up the red block"
_FAITHFULNESS_SAMPLES = 24  # (position, token) swaps for the D6-9 finite-difference gate.


def _quarantine_suffix(suffix_ids: np.ndarray, decoded: str, results_root: str) -> Path:
    """Write the optimised suffix under artifacts/untrusted/ (gitignored, D6-6)."""
    import json

    out_dir = Path("artifacts/untrusted")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stamp from the results dir name (UTC) so the artifact is traceable to the run.
    stamp = Path(results_root).name
    target = out_dir / f"{STAGE}-suffix-{stamp}.json"
    payload = {"suffix_token_ids": [int(x) for x in suffix_ids], "decoded_suffix": decoded}
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openvla/openvla-7b", help="HF model id")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="attention backend")
    parser.add_argument("--instruction", default=_DEFAULT_INSTRUCTION, help="benign instruction")
    parser.add_argument("--seed", type=int, default=42, help="pinned seed")
    parser.add_argument("--suffix-len", type=int, default=20, help="adversarial suffix length")
    parser.add_argument("--n-steps", type=int, default=10, help="GCG steps (tiny: a handful)")
    parser.add_argument("--top-k", type=int, default=256, help="GCG top-k per position")
    parser.add_argument("--search-width", type=int, default=32, help="candidate-batch B")
    parser.add_argument(
        "--results-root", default="results/_smoke", help="write-once results root (smoke)"
    )
    args = parser.parse_args(argv)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    # Heavy / GPU-only imports after the guard (module stays importable on the mac).
    import torch  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

    from evasion_tax.attack.gcg import GcgConfig, run_gcg
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
    from evasion_tax.repro import RunLogger, seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"[{STAGE}] loading {args.model} (bf16, attn={args.attn_impl}) on {args.device} ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.requires_grad_(False)
    model.eval()

    rng = np.random.default_rng(args.seed)
    image = Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
    vocab_size = int(processor.tokenizer.vocab_size)
    target_action_ids = np.array([vocab_size - 1 - b for b in _TARGET_BINS], dtype=np.int64)

    target = OpenVlaGcgTarget(
        model,
        processor,
        image=image,
        instruction=args.instruction,
        suffix_len=args.suffix_len,
        target_action_ids=target_action_ids,
        device=device,
    )

    # (1) D6-9 seam-faithfulness gate FIRST — an unfaithful seam must not time a search.
    torch.cuda.reset_peak_memory_stats(device)
    report = target.gradient_agrees_with_swaps(n_samples=_FAITHFULNESS_SAMPLES, rng=rng)
    print(
        f"[{STAGE}] faithfulness (D6-9): recommended Δloss {report.recommended_mean_delta:+.3f} "
        f"vs random {report.random_mean_delta:+.3f} over {report.n_samples} probes "
        f"(sign-agreement {report.sign_agreement:.2f}, diagnostic only); "
        f"decoded={report.decoded_suffix!r} -> passed={report.passed}"
    )

    cfg = GcgConfig(
        suffix_len=args.suffix_len,
        n_steps=args.n_steps,
        top_k=args.top_k,
        search_width=args.search_width,
        seed=args.seed,
    )

    # (2) run the tiny search to completion.
    result = run_gcg(target, cfg)

    # (3) batched-loss equivalence wiring check.
    init_batch = np.tile(target.init_suffix_ids(), (4, 1))
    batched_ok = target.batched_matches_single(init_batch)

    # (4) peak VRAM fits one card.
    peak_vram_gib = torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
    fits_one_card = peak_vram_gib < _CARD_GIB

    # (5) quarantine the optimised suffix (D6-6).
    best_suffix = np.array(result.best_suffix_ids, dtype=np.int64)
    decoded_best = target.decode_span(best_suffix)

    logger = RunLogger(args.results_root)
    record = {
        "stage": STAGE,
        "model": args.model,
        "dtype": "bfloat16",
        "device": args.device,
        "instruction": args.instruction,
        "seed": args.seed,
        "gcg_config": dict(
            suffix_len=cfg.suffix_len, n_steps=cfg.n_steps, top_k=cfg.top_k,
            search_width=cfg.search_width, seed=cfg.seed,
        ),
        "target_action_token_ids": target_action_ids.tolist(),
        # WIRING gate (pass/fail): the gradient ranking is loss-aligned (D6-9).
        "faithfulness_recommended_mean_delta": report.recommended_mean_delta,
        "faithfulness_random_mean_delta": report.random_mean_delta,
        "faithfulness_sign_agreement": report.sign_agreement,  # diagnostic only.
        "faithfulness_passed": report.passed,
        "batched_matches_single": batched_ok,
        "peak_vram_gib": round(peak_vram_gib, 3),
        "fits_one_card": fits_one_card,
        # EXPLORATORY (recorded, NOT a gate — D6-3).
        "loss_history": list(result.loss_history),
        "best_loss": result.best_loss,
        "n_steps_run": result.n_steps_run,
        "decoded_best_suffix": decoded_best,
    }
    handle = logger.start("gcg-tiny-smoke", config=record, seed=args.seed)
    handle.write("smoke_result", record)
    quarantined = _quarantine_suffix(best_suffix, decoded_best, str(handle.dir))

    print(
        f"[{STAGE}] tiny run: best_loss {result.best_loss:.4f} over {result.n_steps_run} steps "
        f"(loss_history[0]={result.loss_history[0]:.4f}); EXPLORATORY, not a gate (D6-3)"
    )
    print(
        f"[{STAGE}] peak VRAM {peak_vram_gib:.2f} GiB / {_CARD_GIB:.0f} GiB; "
        f"suffix quarantined -> {quarantined}"
    )
    print(f"[{STAGE}] logged -> {handle.dir}")

    # WIRING gate: faithfulness + batched-equivalence + fits one card.
    ok = report.passed and batched_ok and fits_one_card
    if not ok:
        reason = (
            "seam faithfulness gate failed (D6-9)" if not report.passed
            else "batched loss != per-candidate single (wiring)" if not batched_ok
            else "peak VRAM exceeded one card"
        )
        print(f"[{STAGE}] FAIL: {reason}", file=sys.stderr)
        return 1
    print(f"[{STAGE}] PASS: GCG harness wired faithfully on the step-5.5 seam (wiring gate, D6-3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

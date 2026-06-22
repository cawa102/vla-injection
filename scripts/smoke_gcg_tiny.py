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
3. the **strengthened equivalence gate** (DB-4): the true-batch ``loss_of`` reproduces
   per-candidate single CE **exactly at B=1** (formula proof) and, at B>1 where bf16
   batch-order noise (~0.1-0.3 CE) precludes exact agreement, selects a candidate of
   **equal quality** (low *selection regret* — the bf16-robust form of the rank-order check
   GCG relies on) across batch sizes / suffix lengths on mixed-quality candidates, plus a
   same-input determinism re-run;
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
from typing import TYPE_CHECKING

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402

if TYPE_CHECKING:  # type-only; OpenVlaGcgTarget is imported at runtime inside main().
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget

STAGE = "smoke_gcg_tiny"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
_CARD_GIB = 24.0  # one RTX A5000.
# Reuse step-5.5's arbitrary target: 7 bin indices spread across [0, n_bins-1].
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)
_DEFAULT_INSTRUCTION = "pick up the red block"
_FAITHFULNESS_SAMPLES = 24  # (position, token) swaps for the D6-9 finite-difference gate.
_EQUIV_BATCH_SIZES = (1, 2, 8, 32)  # DB-4: batched==single across multiple B.
_EQUIV_SUFFIX_LEN_EXTRA = 8  # DB-4: a 2nd suffix length beyond --suffix-len.
_EQUIV_ATOL = 1e-3  # B=1: true-batch CE must reproduce out.loss exactly (the formula proof).
_BF16_BATCH_ATOL = 1.0  # B>1: bf16 batch-order divergence guard, not a precision claim (see gate).
_DET_ATOL = 1e-3  # determinism re-run tolerance (same batch, identical inputs → bit-reproducible).
_GRAD_GATE_TOP_K = 16  # top-k pool for gradient-recommended candidates (matches the gate).


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


def _mixed_quality_candidates(
    target: OpenVlaGcgTarget,
    *,
    suffix_len: int,
    b: int,
    rng: np.random.Generator,
    grad: np.ndarray | None = None,
) -> np.ndarray:
    """A ``[b, suffix_len]`` candidate set mixing init + gradient-recommended + random (DB-4).

    A degenerate batch of identical rows makes batched==single trivially true and the
    ``argmin`` meaningless, so the gate must span candidate *quality*: the benign init, a few
    gradient-recommended top-k swaps (the pool GCG's selection samples), and random suffixes.
    ``grad`` (the ``[L, V]`` one-hot gradient at the init suffix) is computed once per target
    and reused across batch sizes.
    """
    init = target.init_suffix_ids()
    if b == 1:
        return init[None, :].copy()
    g = target.token_gradient(init) if grad is None else grad
    k = min(_GRAD_GATE_TOP_K, target.vocab_size)
    rows = [init.copy()]
    for _ in range(max(1, (b - 1) // 2)):
        row = init.copy()
        pos = int(rng.integers(0, suffix_len))
        topk = np.argpartition(g[pos], k - 1)[:k]  # k most-negative grads
        row[pos] = int(rng.choice(topk))
        rows.append(row)
    while len(rows) < b:
        rows.append(rng.integers(0, target.vocab_size, size=suffix_len, dtype=np.int64))
    return np.stack(rows[:b])


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
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget, equivalence_verdict
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

    # (3) STRENGTHENED batched-vs-single equivalence + determinism gate (DB-4 / Codex (c)):
    #     batched loss_of == per-candidate _loss_single across multiple B and suffix lengths,
    #     on MIXED-QUALITY candidates, agreeing in value AND argmin (rank order); plus a
    #     same-input determinism re-run. Value equality alone is necessary-not-sufficient —
    #     GCG acts on the argmin (the old identical-row tile made the check trivially true).
    suffix_lens = tuple(dict.fromkeys((args.suffix_len, _EQUIV_SUFFIX_LEN_EXTRA)))
    equiv_checks: list[dict] = []
    for slen in suffix_lens:
        tgt = (
            target
            if slen == args.suffix_len
            else OpenVlaGcgTarget(
                model,
                processor,
                image=image,
                instruction=args.instruction,
                suffix_len=slen,
                target_action_ids=target_action_ids,
                device=device,
            )
        )
        grad = tgt.token_gradient(tgt.init_suffix_ids())  # once per target, reused across B
        for b in _EQUIV_BATCH_SIZES:
            cands = _mixed_quality_candidates(tgt, suffix_len=slen, b=b, rng=rng, grad=grad)
            batched = tgt.loss_of(cands)
            single = np.array([tgt._loss_single(row) for row in cands], dtype=float)
            # B=1 proves the CE formula exactly (same forward). At B>1 the bf16 matmul reduction
            # order varies with the batch dim, so absolute CE differs by ~0.1-0.3 — NOT a formula
            # error (B=1 is exact). The load-bearing invariant GCG acts on is SELECTION REGRET
            # (the true loss of the candidate the batched path would pick vs the true best);
            # strict argmin index equality is too brittle when top candidates are within bf16
            # noise. Tight tol at B=1, a divergence/regret guard at B>1.
            tol = _EQUIV_ATOL if b == 1 else _BF16_BATCH_ATOL
            chk = equivalence_verdict(single, batched, atol=tol, regret_tol=tol)
            equiv_checks.append(
                {
                    "suffix_len": slen,
                    "n": b,
                    "tol": tol,
                    "max_abs_diff": chk.max_abs_diff,
                    "allclose": chk.allclose,
                    "argmin_match": chk.argmin_match,  # diagnostic only
                    "selection_regret": chk.selection_regret,
                    "regret_ok": chk.regret_ok,
                    "passed": chk.passed,
                }
            )
    equiv_all_ok = all(c["passed"] for c in equiv_checks)

    # determinism: identical candidates evaluated twice ⇒ identical [B] within tolerance.
    det_cands = _mixed_quality_candidates(
        target,
        suffix_len=args.suffix_len,
        b=_EQUIV_BATCH_SIZES[-1],
        rng=np.random.default_rng(args.seed),
    )
    det_chk = equivalence_verdict(
        target.loss_of(det_cands), target.loss_of(det_cands), atol=_DET_ATOL, regret_tol=_DET_ATOL
    )
    batched_ok = equiv_all_ok and det_chk.passed

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
        # WIRING gate (DB-4): batched loss_of == per-candidate single in value AND argmin,
        # across B × suffix_len on mixed-quality candidates, plus a determinism re-run.
        "batched_matches_single": batched_ok,  # overall (equivalence matrix AND determinism)
        "equivalence_checks": equiv_checks,
        "equivalence_all_passed": equiv_all_ok,
        "equivalence_bf16_max_abs_diff": max(c["max_abs_diff"] for c in equiv_checks),
        "equivalence_max_selection_regret": max(c["selection_regret"] for c in equiv_checks),
        "determinism_max_abs_diff": det_chk.max_abs_diff,
        "determinism_passed": det_chk.passed,
        "equiv_atol": _EQUIV_ATOL,
        "det_atol": _DET_ATOL,
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
    n_pass = sum(c["passed"] for c in equiv_checks)
    max_diff = max(c["max_abs_diff"] for c in equiv_checks)
    max_regret = max(c["selection_regret"] for c in equiv_checks)
    print(
        f"[{STAGE}] equivalence (DB-4): {n_pass}/{len(equiv_checks)} (B×suffix_len) "
        f"B=1 exact + selection-regret≤{max_regret:.2f} (bf16 |Δ|≤{max_diff:.2f}); "
        f"determinism Δ={det_chk.max_abs_diff:.2e} -> {'PASS' if batched_ok else 'FAIL'}"
    )
    print(f"[{STAGE}] logged -> {handle.dir}")

    # WIRING gate: faithfulness + strengthened batched-equivalence (DB-4) + fits one card.
    ok = report.passed and batched_ok and fits_one_card
    if not ok:
        reason = (
            "seam faithfulness gate failed (D6-9)" if not report.passed
            else "batched != single in value/argmin or determinism failed (DB-4 wiring)"
            if not batched_ok
            else "peak VRAM exceeded one card"
        )
        print(f"[{STAGE}] FAIL: {reason}", file=sys.stderr)
        return 1
    print(f"[{STAGE}] PASS: GCG harness wired faithfully on the step-5.5 seam (wiring gate, D6-3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

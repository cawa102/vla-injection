#!/usr/bin/env python3
"""Early-stop steps-to-success bench driver (bench Task 6) — GPU-guarded, resumable.

Measures the A5000 GCG **steps-to-success distribution with early_stop ON** over a small
target set, so the realistic per-target cost (``median_steps × s/step``) can replace the
registered early_stop-OFF worst case that drives ``branch_select`` (the bench goal).

Validates the config, then **guards**: with no CUDA runtime it prints the GPU-node
requirement and exits non-zero (no silent no-op). On the box it loads frozen bf16
OpenVLA-7B once (Task 4) and runs each target through ``run_gcg(target, cfg,
reached_fn=target.reached)`` (early_stop ON, DE-3) with candidate-eval chunking so the
``search_width=512`` attack fits 24 GB (Task 5 / DE-7). Each finished target writes one
JSON **write-once** under ``results/<run>/targets/`` and quarantines its suffix under
``artifacts/untrusted/<run>/`` (DE-5); a restart **skips** any target whose JSON already
exists (DE-4), so the unattended ``until``-loop / OOM auto-restart is idempotent. The
aggregate reads ALL per-target JSONs (fresh + resumed) into one clean distribution.

Usage (non-registered dry run):
    uv run python scripts/bench_early_stop.py --config configs/example_m2.yaml \\
        --n-targets 1 --n-steps 20 --results-root results/_smoke --run-name dry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Default the CUDA allocator to expandable segments BEFORE any torch/CUDA init so the
# near-ceiling sw=512 forwards do not OOM from fragmentation even if the operator forgot
# the shell export (runbook docs/gpu/CSB/plan.md). An explicit export still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.attack.early_stop_bench import (  # noqa: E402
    TargetOutcome,
    is_target_done,
    outcome_to_record,
    realistic_s_per_target,
    steps_to_success_summary,
    target_id_for,
)
from evasion_tax.attack.gcg import GcgConfig, run_gcg  # noqa: E402
from evasion_tax.attack.openvla_loader import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    build_target,
    load_frozen_openvla,
)
from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.repro import RunHandle, capture_env  # noqa: E402

STAGE = "bench_early_stop"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
# Registered D8 s/step (results/2026-06-23T13-34-55Z-gcg-microbench): the realistic cost is
# median_steps × this. The early_stop-OFF worst case it replaces was n_steps × this.
_D8_S_PER_STEP = 33.19
# Candidate-eval chunk size: the measured max-B on the A5000 (DC-2). Chunking loss_of at
# this width lets the sw=512 attack fit 24 GB (DE-7); the operator can override.
_DEFAULT_EVAL_BATCH = 32


def build_arg_parser() -> argparse.ArgumentParser:
    """The bench CLI (pinned defaults; one variable at a time, recorded in the header)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="pinned config YAML")
    p.add_argument("--results-root", default="results", help="write-once results root")
    p.add_argument(
        "--run-name",
        default="early-stop-bench",
        help="stable run-dir name under results-root; REUSED across restarts for resume (DE-4)",
    )
    p.add_argument("--model", default="openvla/openvla-7b", help="HF model id")
    p.add_argument("--device", default="cuda:0", help="CUDA device")
    p.add_argument("--attn-impl", default="flash_attention_2", help="attention backend")
    p.add_argument("--seed", type=int, default=42, help="pinned seed")
    p.add_argument("--suffix-len", type=int, default=20, help="adversarial suffix length")
    p.add_argument(
        "--n-steps", type=int, default=500, help="GCG step cap; censored if not reached (DE-2)"
    )
    p.add_argument("--top-k", type=int, default=256, help="GCG top-k per position")
    p.add_argument(
        "--search-width", type=int, default=512, help="RoboGCG-faithful candidates/step (DE-2)"
    )
    p.add_argument("--n-targets", type=int, default=5, help="number of targets to attack")
    p.add_argument(
        "--eval-batch",
        type=int,
        default=_DEFAULT_EVAL_BATCH,
        help="candidate-eval chunk size (Task 5); sw=512 fits 24 GB at this width (DE-7)",
    )
    p.add_argument(
        "--s-per-step",
        type=float,
        default=_D8_S_PER_STEP,
        help="registered D8 s/step for the realistic per-target cost",
    )
    p.add_argument(
        "--exclusive-gpu",
        action="store_true",
        help="record exclusive-GPU in the header (DE-6; nice-to-have, not a timing gate)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip targets whose per-target JSON already exists (DE-4; default on)",
    )
    return p


def _quarantine_suffix(quarantine_dir: str | Path, target_id: str, suffix_text: str) -> Path:
    """Write the suffix under ``artifacts/untrusted`` write-once (never committed; DE-5)."""
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    path = quarantine_dir / f"{target_id}.txt"
    if not path.exists():
        path.write_text(suffix_text)
    return path


def process_targets(
    *,
    n_targets: int,
    seed: int,
    run_dir: str | Path,
    quarantine_dir: str | Path,
    resume: bool,
    run_one,
    log=print,
) -> None:
    """Resume-aware per-target loop (the model-free glue; ``run_one`` does the GPU work).

    For each target index: skip if ``resume`` and its checkpoint exists (DE-4); else call
    ``run_one(index, target_id) -> (TargetOutcome, suffix_text)``, write the outcome JSON
    write-once under ``<run_dir>/targets/``, and quarantine the suffix (DE-5). The write-once
    per-target checkpoint is what makes the unattended ``until``-loop / OOM restart safe.
    """
    targets_dir = Path(run_dir) / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    target_handle = RunHandle(targets_dir)
    for i in range(n_targets):
        tid = target_id_for(seed, i)
        if resume and is_target_done(targets_dir, tid):
            log(f"[{STAGE}] resume: skip {tid} (checkpoint exists)")
            continue
        outcome, suffix_text = run_one(i, tid)
        target_handle.write(tid, outcome_to_record(outcome))  # write-once primary record
        _quarantine_suffix(quarantine_dir, tid, suffix_text)
        log(
            f"[{STAGE}] {tid}: reached={outcome.reached} steps={outcome.steps_to_success} "
            f"censored={outcome.censored} peak={outcome.peak_vram_gib:.2f}GiB"
        )


def branch_hint(summary: dict, realistic: float | None) -> str:
    """One-line provisional-branch hint from the distribution (Task 7 does the real re-feed)."""
    if realistic is None:
        return (
            "all targets censored — no reached median; the early_stop-OFF worst-case bound "
            "remains the honest planning number (DE-2)"
        )
    return (
        f"realistic ~{realistic:.0f}s/target (median {summary['median_steps']} steps, "
        f"censored {summary['censored_fraction']:.0%}); feeds branch_select (Task 7)"
    )


def aggregate_run(run_dir: str | Path, *, n_steps_cap: int, s_per_step: float) -> dict:
    """Read ALL per-target JSONs and reproduce the steps-to-success distribution (DE-4).

    Reads every ``<run_dir>/targets/*.json`` (fresh + any from prior restarts) so a
    multi-restart run still yields one clean distribution. Returns the summary, the realistic
    per-target cost (``None`` when every target is censored), and a one-line branch hint.

    Raises:
        ValueError: If no per-target outcomes exist (no silent empty aggregate).
    """
    targets_dir = Path(run_dir) / "targets"
    records = [json.loads(p.read_text()) for p in sorted(targets_dir.glob("*.json"))]
    if not records:
        raise ValueError(f"no per-target outcomes under {targets_dir}")
    outcomes = [TargetOutcome(**r) for r in records]
    summary = steps_to_success_summary(outcomes, n_steps_cap=n_steps_cap)
    median = summary["median_steps"]
    realistic = realistic_s_per_target(median, s_per_step) if median is not None else None
    return {
        "stage": STAGE,
        "early_stop": True,
        "steps_to_success": summary,
        "s_per_step": s_per_step,
        "realistic_s_per_target": realistic,
        "branch_hint": branch_hint(summary, realistic),
    }


def build_run_header(args: argparse.Namespace, env: dict) -> dict:
    """The §8 repro header (pinned seed, device, bf16, GCG config, env, git/torch/cuda)."""
    return {
        "stage": STAGE,
        "dtype": "bfloat16",
        "seed": args.seed,
        "device": args.device,
        "gcg_config": {
            "suffix_len": args.suffix_len,
            "n_steps": args.n_steps,
            "top_k": args.top_k,
            "search_width": args.search_width,
            "seed": args.seed,
        },
        "eval_batch": args.eval_batch,
        "s_per_step": args.s_per_step,
        "early_stop": True,
        "exclusive_gpu": bool(args.exclusive_gpu),
        "env": env,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    load_config(args.config)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    # Heavy / GPU-only imports after the guard (module stays importable on the mac).
    import time

    import torch  # type: ignore[import-not-found]

    from evasion_tax.repro import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg = GcgConfig(
        suffix_len=args.suffix_len,
        n_steps=args.n_steps,
        top_k=args.top_k,
        search_width=args.search_width,
        seed=args.seed,
    )

    run_dir = Path(args.results_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = Path("artifacts/untrusted") / args.run_name
    # Write-once §8 header on the FIRST launch only; a resume reuses the existing run dir.
    if not (run_dir / "run.json").exists():
        RunHandle(run_dir).write("run", build_run_header(args, capture_env()))

    model, processor = load_frozen_openvla(torch, args.model, device, args.attn_impl)

    def run_one(i: int, tid: str):
        # Drop the prior target's CUDA reservation so each attack starts model-only (the
        # near-ceiling sw=512 forward is unforgiving of fragmentation; see microbench).
        torch.cuda.empty_cache()
        target = build_target(
            np, model, processor, device,
            instruction=DEFAULT_INSTRUCTION, suffix_len=cfg.suffix_len,
            seed=args.seed + i, eval_batch=args.eval_batch,
        )
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        result = run_gcg(target, cfg, reached_fn=target.reached)  # early_stop ON (DE-3)
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
        suffix_text = target.decode_span(np.asarray(result.best_suffix_ids, dtype=np.int64))
        outcome = TargetOutcome(
            target_id=tid,
            reached=result.reached,
            steps_to_success=result.n_steps_run,
            censored=not result.reached,  # not reached within the n_steps cap (DE-2)
            best_loss=result.best_loss,
            wall_seconds=wall,
            peak_vram_gib=peak,
            suffix_sha256=hashlib.sha256(suffix_text.encode()).hexdigest(),
        )
        return outcome, suffix_text

    process_targets(
        n_targets=args.n_targets,
        seed=args.seed,
        run_dir=run_dir,
        quarantine_dir=quarantine_dir,
        resume=args.resume,
        run_one=run_one,
    )

    agg = aggregate_run(run_dir, n_steps_cap=cfg.n_steps, s_per_step=args.s_per_step)
    if not (run_dir / "bench_result.json").exists():
        RunHandle(run_dir).write("bench_result", agg)
    print(f"[{STAGE}] {agg['branch_hint']}")
    print(f"[{STAGE}] logged -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

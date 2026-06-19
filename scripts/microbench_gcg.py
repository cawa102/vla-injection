#!/usr/bin/env python3
"""GCG attack micro-benchmark — resolves D4 (eval scale) and D7 (compute budget).

Validates the config locally, then **guards**: with no CUDA runtime it prints the
GPU-node requirement and exits non-zero (no silent no-op). The benchmark body is
implemented on the GPU node (see ``docs/setup/gpu-runbook.md``): it measures GCG
seconds-per-target at bf16 on a few tasks so the M1 gate can fix the final eval
matrix and compute budget, logging the timings to write-once ``results/``.

Usage:
    python scripts/microbench_gcg.py --config configs/example_m2.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402

STAGE = "microbench_gcg"
_EXIT_REQUIRES_GPU = 2

# Spread above which repeated `s/target` timings are not reproducible enough to be a
# branch-defining number (D6-10: the registered number must be reproducible).
_MAX_REL_IQR = 0.5


def summarise_timings(
    per_target_seconds: list[float], *, max_rel_iqr: float = _MAX_REL_IQR
) -> dict:
    """Median / IQR / n of per-target timings + a reproducibility flag.

    The registered `s/target` must be reproducible across repeats (D6-10): a wide
    relative spread (``IQR / median > max_rel_iqr``) flags the run as not
    reproducible so the branch-defining number is never taken from noise.

    Args:
        per_target_seconds: One timing per target (or repeat); must be non-empty.
        max_rel_iqr: Relative-IQR threshold above which ``reproducible`` is False.

    Returns:
        ``{median_s, iqr_s, n, rel_iqr, reproducible}``.

    Raises:
        ValueError: If the list is empty (no silent default).
    """
    arr = np.asarray(per_target_seconds, dtype=float)
    if arr.size == 0:
        raise ValueError("summarise_timings needs at least one timing")
    median = float(np.median(arr))
    q25, q75 = (float(x) for x in np.percentile(arr, [25, 75]))
    iqr = q75 - q25
    rel_iqr = iqr / median if median > 0 else float("inf")
    return {
        "median_s": median,
        "iqr_s": iqr,
        "n": int(arr.size),
        "rel_iqr": rel_iqr,
        "reproducible": rel_iqr <= max_rel_iqr,
    }


# D6-5: step 6's branch is PROVISIONAL — only the non-adaptive `s/target` is measured
# here; the H6-D headline is sized by the adaptive GCG-against-the-probe cost, which
# cannot be measured until L1 is on GPU. Until that confirms N/N−, the committed
# branch is the hard-F default (oracle frontier, H6-A).
_BRANCH_LOCK_CONDITION = (
    "Branch is PROVISIONAL: locked only once the adaptive GCG-against-the-probe "
    "bench (M1/M2, same GPU seam) confirms N/N−; until then the committed branch "
    "is the hard-F default (oracle frontier, H6-A)."
)
# D6-4: named-not-dropped deferred micro-bench items (L1 probe not yet on GPU).
_DEFERRED = "deferred (L1 not on GPU)"


def build_microbench_record(
    *,
    gcg_config: dict,
    timing_summary: dict,
    peak_vram_gib: float,
    max_candidate_batch: int,
    steps_to_success: list[int],
    device_name: str,
    seed: int,
    exclusive_gpu: bool,
) -> dict:
    """Assemble the registered D8 micro-bench record (the measurement half of §8).

    Carries the measured non-adaptive numbers, the **provisional** branch status +
    lock condition (D6-5), and the explicitly **deferred** L1/adaptive items (D6-4)
    so no gap is silently dropped. The §8 *header* (git commit, full env capture,
    created-UTC) is written by :class:`~evasion_tax.repro.RunLogger` into ``run.json``.

    Args:
        gcg_config: The pinned GCG config (``GcgConfig`` as a dict).
        timing_summary: Output of :func:`summarise_timings`.
        peak_vram_gib: Peak reserved VRAM during the timed run.
        max_candidate_batch: Largest candidate-batch B that fit 24 GB.
        steps_to_success: Per-target steps-to-success distribution.
        device_name: Which A5000 (the registered card).
        seed: Pinned seed.
        exclusive_gpu: Whether the GPU was exclusive during the timed run (D6-10).

    Returns:
        The measurement record dict for write-once ``results/``.
    """
    return {
        "stage": STAGE,
        "dtype": "bfloat16",
        "device_name": device_name,
        "seed": seed,
        "gcg_config": gcg_config,
        "s_per_target": timing_summary,
        "peak_vram_gib": peak_vram_gib,
        "max_candidate_batch": max_candidate_batch,
        "steps_to_success": list(steps_to_success),
        "exclusive_gpu": exclusive_gpu,
        "branch_status": "provisional",
        "branch_lock_condition": _BRANCH_LOCK_CONDITION,
        "l1_extraction_overhead": _DEFERRED,
        "adaptive_gcg_cost": _DEFERRED,
    }


def assert_registered_run_valid(record: dict) -> None:
    """Fail the run unless it is exclusive **and** reproducible (D6-10).

    The registered, branch-defining number must come from an uncontaminated,
    exclusive-GPU process whose ``s/target`` reproduces across repeats. Anything
    else is rejected rather than logged as a result.

    Raises:
        ValueError: If the GPU was not exclusive or the timing is not reproducible.
    """
    if not record.get("exclusive_gpu"):
        raise ValueError(
            "registered micro-bench requires an exclusive GPU (D6-10); refusing to "
            "log a branch-defining number from a shared process."
        )
    if not record.get("s_per_target", {}).get("reproducible"):
        raise ValueError(
            "registered micro-bench `s/target` is not reproducible across repeats "
            "(D6-10); refusing to log a noisy branch-defining number."
        )


class OomError(RuntimeError):
    """Raised by a batch ``probe_fn`` when candidate-batch ``B`` OOMs on the card."""


def max_batch_that_fits(probe_fn, start: int, cap: int) -> int:
    """Largest candidate-batch ``B`` in ``[start, cap]`` that does not OOM.

    Doubling up to the first OOM (or ``cap``), then bisection — each ``B`` probed
    **at most once**, so under the clean-process protocol (D6-10, where ``probe_fn``
    runs one ``B`` in a fresh subprocess) no probe contaminates another. ``probe_fn``
    raises :class:`OomError` on OOM and returns otherwise.

    Args:
        probe_fn: ``B -> Any``; raises :class:`OomError` when ``B`` OOMs.
        start: Smallest batch to probe (>= 1).
        cap: Largest batch the budget allows (>= ``start``).

    Returns:
        The largest non-OOM ``B``, or ``0`` if even ``start`` OOMs.

    Raises:
        ValueError: If not ``1 <= start <= cap``.
    """
    if not (1 <= start <= cap):
        raise ValueError(f"require 1 <= start <= cap, got start={start}, cap={cap}")

    def _fits(b: int) -> bool:
        try:
            probe_fn(b)
            return True
        except OomError:
            return False

    if not _fits(start):
        return 0

    lo = start  # known to fit
    while lo * 2 <= cap and _fits(lo * 2):
        lo *= 2
    # lo fits; lo*2 is > cap or OOMs. Establish the OOM upper bound `hi` to bisect.
    if lo * 2 <= cap:
        hi = lo * 2  # known to OOM (the while-loop's _fits returned False)
    elif cap == lo:
        return lo  # the doubling reached the cap exactly
    elif _fits(cap):
        return cap  # cap is in (lo, lo*2), fits, and is the budget ceiling
    else:
        hi = cap  # cap OOMs; the answer is in (lo, cap)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


# Pinned bench knobs (one variable at a time; recorded in the run header).
_BYTES_PER_GIB = 1024**3
_EXIT_OOM = 3  # subprocess batch-probe exit code on CUDA OOM (clean-process, D6-10).
_TARGET_BINS = (32, 64, 96, 128, 160, 192, 224)  # reuse the step-5.5 arbitrary target.
_DEFAULT_INSTRUCTION = "pick up the red block"


def _build_target(np_mod, model, processor, device, *, instruction, suffix_len, seed):
    """Build one :class:`OpenVlaGcgTarget` (deterministic dummy image, fixed target)."""
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
    )


def _load_frozen_openvla(torch_mod, model_id, device, attn_impl):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--model", default="openvla/openvla-7b", help="HF model id")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="attention backend")
    parser.add_argument("--seed", type=int, default=42, help="pinned seed")
    parser.add_argument("--suffix-len", type=int, default=20, help="adversarial suffix length")
    parser.add_argument("--n-steps", type=int, default=100, help="pinned GCG steps")
    parser.add_argument("--top-k", type=int, default=256, help="GCG top-k per position")
    parser.add_argument("--search-width", type=int, default=256, help="initial candidate-batch B")
    parser.add_argument("--n-targets", type=int, default=3, help="number of targets to time")
    parser.add_argument("--batch-cap", type=int, default=512, help="largest candidate-batch swept")
    parser.add_argument("--exclusive-gpu", action="store_true", help="assert exclusive GPU (D6-10)")
    parser.add_argument(
        "--probe-batch",
        type=int,
        default=None,
        help="internal: run ONE candidate-batch B in this fresh process; exit 3 on OOM (D6-10)",
    )
    args = parser.parse_args(argv)

    load_config(args.config)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    # Heavy / GPU-only imports after the guard (module stays importable on the mac).
    import subprocess
    import time

    import torch  # type: ignore[import-not-found]

    from evasion_tax.attack.gcg import GcgConfig, run_gcg
    from evasion_tax.repro import RunLogger, seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    model, processor = _load_frozen_openvla(torch, args.model, device, args.attn_impl)

    cfg = GcgConfig(
        suffix_len=args.suffix_len,
        n_steps=args.n_steps,
        top_k=args.top_k,
        search_width=args.search_width,
        seed=args.seed,
    )

    # ---- subprocess batch-probe mode (D6-10): one B, fresh process, exit 3 on OOM.
    if args.probe_batch is not None:
        target = _build_target(
            np, model, processor, device,
            instruction=_DEFAULT_INSTRUCTION, suffix_len=cfg.suffix_len, seed=args.seed,
        )
        suffix = target.init_suffix_ids()
        batch = np.tile(suffix, (args.probe_batch, 1))
        try:
            target.loss_of(batch)
        except torch.cuda.OutOfMemoryError:
            print(f"[{STAGE}] OOM at B={args.probe_batch}", file=sys.stderr)
            return _EXIT_OOM
        return 0

    # ---- batch sweep: each B probed in a fresh subprocess (no allocator carry-over).
    base_cmd = [sys.executable, str(Path(__file__).resolve()), "--config", args.config,
                "--suffix-len", str(cfg.suffix_len), "--seed", str(args.seed)]

    def probe_fn(b: int) -> None:
        proc = subprocess.run([*base_cmd, "--probe-batch", str(b)], capture_output=True, text=True)
        if proc.returncode == _EXIT_OOM:
            raise OomError(f"OOM at B={b}")
        if proc.returncode != 0:
            raise RuntimeError(f"batch probe B={b} failed ({proc.returncode}): {proc.stderr}")

    max_batch = max_batch_that_fits(probe_fn, start=1, cap=args.batch_cap)

    # ---- time s/target on a few targets in THIS (clean) process.
    per_target_seconds: list[float] = []
    steps_to_success: list[int] = []
    peak_vram_gib = 0.0
    for i in range(args.n_targets):
        target = _build_target(
            np, model, processor, device,
            instruction=_DEFAULT_INSTRUCTION, suffix_len=cfg.suffix_len, seed=args.seed + i,
        )
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        result = run_gcg(target, cfg)
        per_target_seconds.append(time.perf_counter() - t0)
        steps_to_success.append(result.n_steps_run)
        peak_vram_gib = max(
            peak_vram_gib, torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
        )

    summary = summarise_timings(per_target_seconds)
    record = build_microbench_record(
        gcg_config=dataclasses.asdict(cfg),
        timing_summary=summary,
        peak_vram_gib=round(peak_vram_gib, 3),
        max_candidate_batch=max_batch,
        steps_to_success=steps_to_success,
        device_name=torch.cuda.get_device_properties(device).name,
        seed=args.seed,
        exclusive_gpu=bool(args.exclusive_gpu),
    )
    assert_registered_run_valid(record)  # D6-10: refuse a contaminated/noisy number.

    logger = RunLogger(args.results_root)
    handle = logger.start("gcg-microbench", config=record, seed=args.seed)
    handle.write("microbench_result", record)

    print(f"[{STAGE}] s/target median {summary['median_s']:.2f}s (n={summary['n']}, "
          f"reproducible={summary['reproducible']}), peak VRAM {peak_vram_gib:.2f} GiB, "
          f"max candidate-batch B={max_batch}")
    print(f"[{STAGE}] logged -> {handle.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

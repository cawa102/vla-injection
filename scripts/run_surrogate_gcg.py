#!/usr/bin/env python3
"""Optimize a RoboGCG suffix on a bf16/int8/NF4 OpenVLA surrogate.

The normal path is GPU-only and writes the optimized suffix under
``artifacts/untrusted/`` plus a metadata pointer under write-once ``results/``.
``--dry-run`` validates config/CLI locally without CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.attack.gcg import GcgConfig, OnStep, run_gcg  # noqa: E402
from evasion_tax.attack.openvla_loader import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    build_target,
    load_openvla_with_attn_fallback,
    quantization_record,
)
from evasion_tax.attack.surrogate_artifacts import (  # noqa: E402
    QUARANTINE_ROOT,
    SCHEMA_VERSION,
    SurrogateSuffixArtifact,
    require_quarantined,
    utc_now_iso,
    write_json_record,
)
from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.repro import RunLogger, capture_env, seed_everything  # noqa: E402

STAGE = "run_surrogate_gcg"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
_DEFAULT_EVAL_BATCH = 32
_GATE_SAMPLES = 32  # gradient_agrees_with_swaps probes for the recorded gradient-health diagnostic
_PROGRESS_EVERY = 25  # emit a heartbeat progress line every N completed GCG steps
_CHECKPOINT_EVERY = 50  # refresh the quarantined best-suffix checkpoint every M completed steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="surrogate config YAML")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--run-name", default="surrogate-gcg", help="RunLogger slug")
    parser.add_argument(
        "--model",
        default=None,
        help="HF checkpoint; defaults to config.model.checkpoint",
    )
    parser.add_argument("--suite", default=None, help="LIBERO suite; defaults to config.env.suite")
    parser.add_argument(
        "--task-id",
        default=None,
        help="task id; defaults to first config env task",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=["bf16", "int8", "nf4_4bit"],
        help="surrogate precision; defaults to config.model.quantization or bf16",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="attention backend")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="pinned seed; defaults to config.seed",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="instruction prefix")
    parser.add_argument("--suffix-len", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--search-width", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--eval-batch", type=int, default=_DEFAULT_EVAL_BATCH)
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop when the target action tokens are greedily decoded",
    )
    parser.add_argument(
        "--target-action-tokens",
        default=None,
        help="optional comma-separated target action token ids; default uses the step-5.5 target",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate config/CLI without CUDA")
    return parser


def _parse_target_tokens(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    toks = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not toks:
        raise ValueError("--target-action-tokens must contain at least one integer")
    return np.asarray(toks, dtype=np.int64)


def _progress_line(step: int, n_steps: int, best_loss: float, elapsed_s: float) -> str:
    """Heartbeat line for the GCG search log. No suffix payload — safe for stdout."""
    return f"[gcg] step {step}/{n_steps} best_loss={best_loss:.3f} elapsed={elapsed_s:.0f}s"


def _checkpoint_dict(
    step: int,
    n_steps: int,
    best_suffix: np.ndarray,
    best_loss: float,
    precision: str,
    updated_utc: str,
) -> dict:
    """Mutable best-suffix checkpoint (Shared-contract dict, JSON-safe).

    Carries ``best_suffix_ids`` — the attack payload — so the file it is written to
    MUST stay under ``artifacts/untrusted/`` (see :func:`_checkpoint_path`).
    """
    return {
        "step": int(step),
        "n_steps": int(n_steps),
        "best_loss": float(best_loss),
        "best_suffix_ids": [int(x) for x in best_suffix],
        "precision": precision,
        "updated_utc": updated_utc,
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Overwrite ``path`` with ``obj`` atomically (write a temp, then ``os.replace``).

    Unlike the write-once :func:`write_json_record`, this is OVERWRITE-safe: the
    checkpoint is a mutable sidecar refreshed each interval, so re-writing it must
    not raise ``FileExistsError``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _checkpoint_path(artifact_dir: Path) -> Path:
    """The mutable checkpoint path for a run, enforced to stay under quarantine.

    The checkpoint carries the suffix (attack payload), so it must live beside the
    write-once suffix artifact under ``artifacts/untrusted/<run_id>/`` — never under
    tracked ``results/``.
    """
    return require_quarantined(artifact_dir / "checkpoint.json")


def _make_on_step(
    *,
    n_steps: int,
    precision: str,
    checkpoint_path: Path,
    t0: float,
    out=sys.stdout,
) -> OnStep:
    """Throttled :data:`OnStep`: heartbeat every ``_PROGRESS_EVERY`` steps, checkpoint
    every ``_CHECKPOINT_EVERY`` steps.

    The progress line (no suffix) goes to ``out``; the suffix-bearing checkpoint is
    atomically (over)written to the quarantined ``checkpoint_path``. Elapsed wall time
    is measured against ``t0`` (a ``time.perf_counter()`` reading taken at search start).
    """

    def on_step(step: int, best_suffix: np.ndarray, best_loss: float) -> None:
        if step % _PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t0
            print(_progress_line(step, n_steps, best_loss, elapsed), file=out, flush=True)
        if step % _CHECKPOINT_EVERY == 0:
            _atomic_write_json(
                checkpoint_path,
                _checkpoint_dict(step, n_steps, best_suffix, best_loss, precision, utc_now_iso()),
            )

    return on_step


def _write_text_once(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing quarantined suffix: {path}")
    path.write_text(text)
    return path


def _resolved_run_fields(args: argparse.Namespace, config) -> dict:
    model = args.model or config.model.checkpoint or config.model.name
    suite = args.suite or config.env.suite
    task_id = args.task_id or config.env.tasks[0]
    seed = config.seed if args.seed is None else args.seed
    precision = args.precision or config.model.quantization or "bf16"
    return {
        "model": model,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "precision": precision,
    }


def build_dry_run_record(args: argparse.Namespace) -> dict:
    """Validate config, GCG knobs, target tokens, and precision without CUDA."""
    cfg_obj = load_config(args.config)
    fields = _resolved_run_fields(args, cfg_obj)
    gcg = GcgConfig(
        suffix_len=args.suffix_len,
        n_steps=args.n_steps,
        top_k=args.top_k,
        search_width=args.search_width,
        seed=fields["seed"],
    )
    _parse_target_tokens(args.target_action_tokens)
    load_record = quantization_record(
        precision=fields["precision"],
        torch_dtype="bfloat16",
        device=args.device,
        attn_impl=args.attn_impl,
    )
    return {
        "stage": STAGE,
        "dry_run": True,
        "model": fields["model"],
        "suite": fields["suite"],
        "task_id": fields["task_id"],
        "seed": fields["seed"],
        "precision": fields["precision"],
        "gcg_config": {**dataclasses.asdict(gcg), "early_stop": args.early_stop},
        "load_record": dataclasses.asdict(load_record),
    }


def _git_commit_or_raise(env: dict) -> str:
    commit = env.get("git_commit")
    if not commit:
        raise RuntimeError(
            "capture_env() did not find a git commit; clone the repo instead of running "
            "from an archive before logging suffix artifacts"
        )
    return str(commit)


def _gradient_health(target, seed: int) -> dict:
    """Record-only gradient-health diagnostic computed once at the init suffix.

    Mirrors ``scripts/smoke_quantized_backward.py`` so a censored arm is attributable
    after the fact (dead vs weak-but-real vs hard target). It **records, never gates** —
    nf4's weak-but-real gradient (plan H-S2) must still run the full search — so a probe
    failure is captured as ``{"error": ...}`` rather than aborting the run.
    """
    try:
        grad = target.token_gradient(target.init_suffix_ids())
        report = target.gradient_agrees_with_swaps(
            n_samples=_GATE_SAMPLES,
            rng=np.random.default_rng(seed),
        )
        grad_absmax = float(np.abs(grad).max())
        return {
            "grad_absmax": grad_absmax,
            "grad_nonzero": bool(grad_absmax > 0.0),
            "grad_finite": bool(np.isfinite(grad).all()),
            "recommended_mean_delta": report.recommended_mean_delta,
            "random_mean_delta": report.random_mean_delta,
            "faithfulness_passed": bool(report.passed),
            "gate_samples": _GATE_SAMPLES,
        }
    except Exception as exc:  # noqa: BLE001 - record, never gate; a weak/dead gradient must still run.
        return {"error": f"{type(exc).__name__}: {exc}"}


def _results_pointer(artifact: SurrogateSuffixArtifact, artifact_json_path: str) -> dict:
    """Committable ``results/`` record: surrogate metrics + a provenance pointer, no suffix.

    The suffix token ids / text stay quarantined under ``artifacts/untrusted/`` (gitignored);
    only the SHA-256 and the run metrics travel into write-once ``results/`` so the run is
    shareable and analysable from version control without shipping the attack payload.
    """
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_path": artifact_json_path,
        "suffix_sha256": artifact.suffix_sha256,
        "surrogate_precision": artifact.surrogate_precision,
        "surrogate_target_hit": artifact.surrogate_target_hit,
        "surrogate_steps_to_success": artifact.surrogate_steps_to_success,
        "surrogate_censored": artifact.surrogate_censored,
        "surrogate_best_loss": artifact.surrogate_best_loss,
        "surrogate_wall_seconds": artifact.surrogate_wall_seconds,
        "surrogate_peak_vram_gib": artifact.surrogate_peak_vram_gib,
        "surrogate_gradient_health": artifact.surrogate_gradient_health,
        "failure_reason": artifact.failure_reason,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_dry_run_record(args), indent=2, sort_keys=True))
        return 0

    cfg_obj = load_config(args.config)
    fields = _resolved_run_fields(args, cfg_obj)
    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    import torch  # type: ignore[import-not-found]

    seed_everything(fields["seed"])
    env = capture_env()
    git_commit = _git_commit_or_raise(env)
    device = torch.device(args.device)
    gcg = GcgConfig(
        suffix_len=args.suffix_len,
        n_steps=args.n_steps,
        top_k=args.top_k,
        search_width=args.search_width,
        seed=fields["seed"],
    )
    target_action_ids = _parse_target_tokens(args.target_action_tokens)

    model, processor, load_record = load_openvla_with_attn_fallback(
        torch,
        fields["model"],
        device,
        args.attn_impl,
        precision=fields["precision"],
    )
    target = build_target(
        np,
        model,
        processor,
        device,
        instruction=args.instruction,
        suffix_len=gcg.suffix_len,
        seed=fields["seed"],
        eval_batch=args.eval_batch,
        target_action_ids=target_action_ids,
    )
    gradient_health = _gradient_health(target, fields["seed"])

    run_config = {
        "stage": STAGE,
        "model": fields["model"],
        "suite": fields["suite"],
        "task_id": fields["task_id"],
        "precision": fields["precision"],
        "seed": fields["seed"],
        "gcg_config": {**dataclasses.asdict(gcg), "early_stop": args.early_stop},
        "load_record": dataclasses.asdict(load_record),
    }
    handle = RunLogger(args.results_root).start(
        args.run_name,
        config=run_config,
        seed=fields["seed"],
    )
    run_id = handle.dir.name
    artifact_id = f"{run_id}-{fields['suite']}-{fields['task_id']}-{fields['precision']}"
    artifact_dir = QUARANTINE_ROOT / run_id

    t0 = time.perf_counter()
    on_step = _make_on_step(
        n_steps=gcg.n_steps,
        precision=fields["precision"],
        checkpoint_path=_checkpoint_path(artifact_dir),
        t0=t0,
    )
    failure_reason = None
    try:
        result = run_gcg(
            target,
            gcg,
            reached_fn=target.reached if args.early_stop else None,
            on_step=on_step,
        )
        suffix_ids = np.asarray(result.best_suffix_ids, dtype=np.int64)
        surrogate_target_hit = bool(result.reached)
        steps_to_success = int(result.n_steps_run)
        censored = not surrogate_target_hit
        best_loss = float(result.best_loss)
    except Exception as exc:  # noqa: BLE001 - quantized backward failures are logged evidence.
        failure_reason = f"{type(exc).__name__}: {exc}"
        suffix_ids = target.init_suffix_ids()
        surrogate_target_hit = False
        steps_to_success = 0
        censored = True
        best_loss = None
    wall_seconds = time.perf_counter() - t0

    suffix_text = target.decode_span(suffix_ids)
    suffix_sha = hashlib.sha256(suffix_text.encode()).hexdigest()
    suffix_path = _write_text_once(artifact_dir / f"{artifact_id}.txt", suffix_text)
    peak_vram = torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB
    gpu_id = torch.cuda.get_device_properties(device).name

    artifact = SurrogateSuffixArtifact(
        schema_version=SCHEMA_VERSION,
        artifact_id=artifact_id,
        surrogate_precision=fields["precision"],  # type: ignore[arg-type]
        model_checkpoint=fields["model"],
        suite=fields["suite"],
        task_id=fields["task_id"],
        target_action_tokens=tuple(int(x) for x in target.target_action_ids.tolist()),
        seed=fields["seed"],
        gcg_config={**dataclasses.asdict(gcg), "early_stop": args.early_stop},
        suffix_token_ids=tuple(int(x) for x in suffix_ids.tolist()),
        suffix_path=str(suffix_path),
        suffix_sha256=suffix_sha,
        source_run_dir=str(handle.dir),
        git_commit=git_commit,
        gpu_id=gpu_id,
        environment=env,
        load_record=dataclasses.asdict(load_record),
        surrogate_target_hit=surrogate_target_hit,
        surrogate_steps_to_success=steps_to_success,
        surrogate_censored=censored,
        surrogate_best_loss=best_loss,
        surrogate_wall_seconds=wall_seconds,
        surrogate_peak_vram_gib=peak_vram,
        surrogate_gradient_health=gradient_health,
        failure_reason=failure_reason,
        created_utc=utc_now_iso(),
    )
    artifact_json = write_json_record(artifact, artifact_dir / f"{artifact_id}.json")
    handle.write("surrogate_suffix_artifact", _results_pointer(artifact, str(artifact_json)))
    print(f"[{STAGE}] artifact -> {artifact_json}")
    if failure_reason:
        print(f"[{STAGE}] surrogate failed/censored: {failure_reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

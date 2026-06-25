#!/usr/bin/env python3
"""Run the benign LIBERO baseline for a pinned config (Task 4) — GPU-guarded.

Rolls ``--n-benign`` benign episodes through the reusable runner
(:func:`evasion_tax.eval.rollout_runner.run_episode`, ``suffix_text=None``), logs
each ``RolloutStep`` stream + per-rollout ``geometry_stats`` write-once to
``results/<run>/``, and records the benign success rate + calibration/held-out
split. The emitted ``geometry_stats`` is the **DM-3 re-pin input** (Task 0); the
benign metric-(A) scores feed the M1 separation gate (Task 6).

Off-GPU it **guards** (prints the GPU requirement, exits 2 — never a silent
no-op). The pure aggregate/split/resume glue is unit-tested with the GPU episode
call mocked.

    PYTHONPATH=src python scripts/run_benign.py --config configs/example_m2.yaml \\
        --n-benign 300 --calib-frac 0.25 --openvla-root ~/openvla --resume
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.eval.rollout_runner import EpisodeResult, geometry_stats  # noqa: E402
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA  # noqa: E402

STAGE = "run_benign"
_EXIT_REQUIRES_GPU = 2


# --------------------------------------------------------------------------- #
# Pure glue (unit-tested with the GPU episode call mocked)                     #
# --------------------------------------------------------------------------- #


def assign_calibration(index: int, n: int, calib_frac: float) -> bool:
    """Whether episode ``index`` belongs to the calibration split (disjoint prefix).

    The first ``round(n * calib_frac)`` episodes are calibration, the rest are the
    held-out eval split (invariant #3: calib and eval are disjoint by index).
    """
    n_calib = round(n * calib_frac)
    return index < n_calib


def aggregate_benign(records: Sequence[dict]) -> dict:
    """Benign success rate + split sizes over the per-episode records."""
    n = len(records)
    if n == 0:
        raise ValueError("aggregate_benign needs at least one record")
    n_calib = sum(1 for r in records if r["is_calibration"])
    return {
        "n": n,
        "success_rate": sum(1 for r in records if r["success"]) / n,
        "n_calib": n_calib,
        "n_eval": n - n_calib,
    }


def _score_episode(result: EpisodeResult, *, is_calibration: bool, schema: SchemaA, k: int) -> dict:
    """Build one per-episode record: metric-(A) scores + geometry + success."""
    scores = ConsistencyMetricA(schema=schema, k=k).score_rollout(result.rollout)
    return {
        "success": result.success,
        "is_calibration": is_calibration,
        "metric_a_per_step": [s.value for s in scores],
        "geometry": geometry_stats(result.rollout, success=result.success),
        "steps": [dataclasses.asdict(s) for s in result.rollout.steps],
    }


def run_benign_loop(
    episodes_dir: Path,
    *,
    n_benign: int,
    calib_frac: float,
    seed: int,
    schema: SchemaA,
    k: int,
    episode_fn: Callable[..., EpisodeResult],
    resume: bool,
) -> tuple[list[dict], dict]:
    """Roll ``n_benign`` episodes via ``episode_fn``, write per-episode JSON, aggregate.

    Per-episode write-once checkpoint (``episodes/ep{seed+i}.json``); ``resume``
    reloads a finished episode instead of re-running it (idempotent restart). The
    only GPU dependency is ``episode_fn`` (the runner) — everything else is pure.
    """
    episodes_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i in range(n_benign):
        path = episodes_dir / f"ep{seed + i}.json"
        if resume and path.exists():
            records.append(json.loads(path.read_text()))
            continue
        # Progress (flush=True so it shows immediately under buffering/nohup): the first
        # episode pays a one-time CUDA/flash warmup, so this confirms the loop is alive.
        print(f"[run_benign] rolling episode {i + 1}/{n_benign} ...", flush=True)
        result = episode_fn(index=i, seed=seed + i)
        record = _score_episode(
            result, is_calibration=assign_calibration(i, n_benign, calib_frac), schema=schema, k=k
        )
        path.write_text(json.dumps(record))
        records.append(record)
        print(
            f"[run_benign] episode {i + 1}/{n_benign}: success={result.success} "
            f"steps={len(result.rollout)}",
            flush=True,
        )
    return records, aggregate_benign(records)


def _benign_records_view(records: Sequence[dict]) -> list[dict]:
    """The m1_gate ``BenignRecord`` dicts (drop the bulky step stream)."""
    return [
        {
            "success": r["success"],
            "metric_a_per_step": r["metric_a_per_step"],
            "is_calibration": r["is_calibration"],
        }
        for r in records
    ]


def prepare_run_dir(results_root: str, run_name: str) -> tuple[Path, bool]:
    """Resolve a STABLE run dir reused across restarts so ``--resume`` actually resumes.

    Returns ``(run_dir, is_first_launch)``. A fresh timestamped dir per launch (the old
    behaviour) made ``--resume`` look in an empty dir and re-run everything; a stable
    ``run_name`` + per-episode checkpoints under ``<run_dir>/episodes/`` make a restart
    idempotent (mirrors the bench driver). The §8 header is written only on first launch.
    """
    run_dir = Path(results_root) / run_name
    is_first = not run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, is_first


# --------------------------------------------------------------------------- #
# GPU body (torch / LIBERO imported inside; never runs off-GPU)               #
# --------------------------------------------------------------------------- #


def _model_id(config) -> str:
    """The HF model id to load: the pinned ``checkpoint`` (e.g. the LIBERO fine-tune),
    falling back to ``name`` when no separate checkpoint is pinned."""
    return config.model.checkpoint or config.model.name


def _build_episode_fn(args, config, *, git_commit, run_id):  # pragma: no cover - GPU only
    """Build the per-episode GPU runner (model + env) → an ``episode_fn``."""
    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)
    from types import SimpleNamespace

    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_env,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
    )
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    from evasion_tax.eval.rollout_runner import run_episode
    from evasion_tax.metric.state_libero import LiberoStateAdapter

    model_id = _model_id(config)
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=model_id,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        unnorm_key=config.model.unnorm_key,
        task_suite_name=config.env.suite,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(torch.device(args.device))
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()[config.env.suite]()
    n_tasks = task_suite.n_tasks

    def episode_fn(*, index: int, seed: int) -> EpisodeResult:
        task_id = index % n_tasks
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        adapter = LiberoStateAdapter([str(o) for o in env.obj_of_interest])
        try:
            return run_episode(
                model, processor, env=env,
                init_state=init_states[(index // n_tasks) % len(init_states)],
                task_description=str(task_description), cfg=cfg, adapter=adapter,
                resize_size=resize_size, run_id=run_id, seed=seed, git_commit=git_commit,
                suite=config.env.suite, task_id=str(task.name), suffix_text=None,
                max_steps=config.env.max_steps,
            )
        finally:
            env.close()

    return episode_fn


def _run(args, config) -> int:  # pragma: no cover - GPU only
    from evasion_tax.repro import capture_env, seed_everything

    seed_everything(config.seed)
    env_rec = capture_env()
    git_commit = env_rec.get("git_commit")
    schema = _load_schema(args.schema_from)

    run_dir, first = prepare_run_dir(args.results_root, args.run_name)
    if first:  # write-once §8 header on the FIRST launch only; a resume reuses the dir
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "stage": STAGE, "run_name": args.run_name, "seed": config.seed,
                    "git_commit": git_commit, "hardware": env_rec,
                    "config": {
                        "model": _model_id(config), "suite": config.env.suite,
                        "n_benign": args.n_benign, "calib_frac": args.calib_frac,
                        "k": config.metric.k,
                    },
                    "schema_from": args.schema_from,
                },
                indent=2, sort_keys=True,
            )
            + "\n"
        )

    episode_fn = _build_episode_fn(args, config, git_commit=git_commit, run_id=run_dir.name)
    records, summary = run_benign_loop(
        run_dir / "episodes", n_benign=args.n_benign, calib_frac=args.calib_frac,
        seed=config.seed, schema=schema, k=config.metric.k, episode_fn=episode_fn,
        resume=args.resume,
    )
    # Derived views — overwrite-safe across resumes (the per-episode JSONs are the
    # write-once raw record). geometry_stats = CALIB split only (D-3 §2 / invariant #3).
    (run_dir / "benign_records.json").write_text(
        json.dumps(_benign_records_view(records), indent=2) + "\n"
    )
    (run_dir / "geometry_stats.json").write_text(
        json.dumps([r["geometry"] for r in records if r["is_calibration"]], indent=2) + "\n"
    )
    (run_dir / "benign_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{STAGE}] {summary['n']} episodes, success_rate={summary['success_rate']:.3f}, "
          f"calib={summary['n_calib']} eval={summary['n_eval']} -> {run_dir}")
    return 0


def _load_schema(path: str | None) -> SchemaA:
    """The re-pinned ``SchemaA`` from ``--schema-from``, or the frozen base default."""
    if path is None:
        return SchemaA()
    d = json.loads(Path(path).read_text())
    return SchemaA(
        engagement_radius=float(d["engagement_radius"]),
        grasp_radius=float(d["grasp_radius"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--n-benign", type=int, default=300, help="benign episodes to roll")
    parser.add_argument("--calib-frac", type=float, default=0.25, help="calibration split fraction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--openvla-root", default=None, help="cloned openvla repo (eval helpers)")
    parser.add_argument("--schema-from", default=None,
                        help="frozen re-pinned SchemaA JSON (default: base placeholders)")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--run-name", default="m1-benign-baseline",
                        help="stable run-dir under results-root, reused across restarts")
    parser.add_argument("--resume", action="store_true", help="skip episodes already on disk")
    args = parser.parse_args(argv)

    config = load_config(args.config)  # validate locally before GPU time

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    return _run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a quarantined surrogate suffix artifact on a bf16 OpenVLA victim."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.attack.openvla_loader import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    build_target,
    load_openvla_with_attn_fallback,
)
from evasion_tax.attack.surrogate_artifacts import (  # noqa: E402
    SCHEMA_VERSION,
    TransferEvalRecord,
    assert_transfer_compatible,
    read_suffix_artifact,
    token_l2_distance,
    utc_now_iso,
)
from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.repro import RunLogger, capture_env, seed_everything  # noqa: E402

STAGE = "evaluate_surrogate_transfer"
_EXIT_REQUIRES_GPU = 2
_DEFAULT_EVAL_BATCH = 32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="quarantined suffix artifact JSON")
    parser.add_argument("--config", default=None, help="optional config YAML for defaults")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--run-name", default="surrogate-transfer", help="RunLogger slug")
    parser.add_argument("--model", default=None, help="victim checkpoint; defaults to artifact")
    parser.add_argument("--suite", default=None, help="suite; defaults to artifact")
    parser.add_argument("--task-id", default=None, help="task; defaults to artifact")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="attention backend")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="victim instruction prefix",
    )
    parser.add_argument(
        "--suffix-len",
        type=int,
        default=None,
        help="defaults to artifact suffix length",
    )
    parser.add_argument("--eval-batch", type=int, default=_DEFAULT_EVAL_BATCH)
    parser.add_argument("--persistence-window", type=int, default=1)
    parser.add_argument(
        "--override-mismatch",
        action="store_true",
        help="explicitly allow checkpoint/suite/task mismatch",
    )
    return parser


def _victim_identity(args: argparse.Namespace, artifact) -> dict:
    if args.config:
        cfg = load_config(args.config)
        model = args.model or cfg.model.checkpoint or cfg.model.name
        suite = args.suite or cfg.env.suite
        task_id = args.task_id or cfg.env.tasks[0]
    else:
        model = args.model or artifact.model_checkpoint
        suite = args.suite or artifact.suite
        task_id = args.task_id or artifact.task_id
    return {"model": model, "suite": suite, "task_id": task_id}


def _git_commit_or_raise(env: dict) -> str:
    commit = env.get("git_commit")
    if not commit:
        raise RuntimeError(
            "capture_env() did not find a git commit; clone the repo before logging "
            "transfer records"
        )
    return str(commit)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifact = read_suffix_artifact(args.artifact)
    victim = _victim_identity(args, artifact)
    assert_transfer_compatible(
        artifact,
        model_checkpoint=victim["model"],
        suite=victim["suite"],
        task_id=victim["task_id"],
        override=args.override_mismatch,
    )
    if args.persistence_window < 1:
        raise ValueError("--persistence-window must be >= 1")

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    import torch  # type: ignore[import-not-found]

    seed_everything(artifact.seed)
    env = capture_env()
    git_commit = _git_commit_or_raise(env)
    device = torch.device(args.device)
    suffix_ids = np.asarray(artifact.suffix_token_ids, dtype=np.int64)
    suffix_len = args.suffix_len or len(artifact.suffix_token_ids)
    if suffix_len != len(artifact.suffix_token_ids):
        raise ValueError(
            f"suffix_len {suffix_len} does not match artifact suffix length "
            f"{len(artifact.suffix_token_ids)}"
        )

    run_config = {
        "stage": STAGE,
        "artifact_id": artifact.artifact_id,
        "artifact_path": str(Path(args.artifact)),
        "surrogate_precision": artifact.surrogate_precision,
        "victim_precision": "bf16",
        "model": victim["model"],
        "suite": victim["suite"],
        "task_id": victim["task_id"],
        "seed": artifact.seed,
    }
    handle = RunLogger(args.results_root).start(
        args.run_name,
        config=run_config,
        seed=artifact.seed,
    )

    t0 = time.perf_counter()
    failure_reason = None
    predicted_tokens: tuple[int, ...] | None = None
    distance: float | None = None
    hit = False
    try:
        model, processor, load_record = load_openvla_with_attn_fallback(
            torch,
            victim["model"],
            device,
            args.attn_impl,
            precision="bf16",
        )
        target = build_target(
            np,
            model,
            processor,
            device,
            instruction=args.instruction,
            suffix_len=suffix_len,
            seed=artifact.seed,
            eval_batch=args.eval_batch,
            target_action_ids=np.asarray(artifact.target_action_tokens, dtype=np.int64),
        )
        predicted = target.predict_target_action_ids(suffix_ids)
        predicted_tokens = tuple(int(x) for x in predicted.tolist())
        hit = tuple(artifact.target_action_tokens) == predicted_tokens
        distance = token_l2_distance(predicted_tokens, artifact.target_action_tokens)
        load_record_dict = dataclasses.asdict(load_record)
    except Exception as exc:  # noqa: BLE001 - failed/censored transfer is logged evidence.
        failure_reason = f"{type(exc).__name__}: {exc}"
        load_record_dict = None
    wall_seconds = time.perf_counter() - t0

    record = TransferEvalRecord(
        schema_version=SCHEMA_VERSION,
        transfer_id=handle.dir.name,
        artifact_id=artifact.artifact_id,
        artifact_path=str(Path(args.artifact)),
        suffix_sha256=artifact.suffix_sha256,
        surrogate_precision=artifact.surrogate_precision,
        surrogate_target_hit=artifact.surrogate_target_hit,
        victim_precision="bf16",
        model_checkpoint=victim["model"],
        suite=victim["suite"],
        task_id=victim["task_id"],
        seed=artifact.seed,
        target_action_tokens=artifact.target_action_tokens,
        victim_target_hit=hit,
        predicted_target_tokens=predicted_tokens,
        action_distance_to_target=distance,
        persistence_window=args.persistence_window,
        rollout_evaluated=False,
        rollout_success=None,
        wall_seconds=wall_seconds,
        failure_reason=failure_reason,
        censored=not hit,
        source_run_dir=str(handle.dir),
        git_commit=git_commit,
        environment={**env, "victim_load_record": load_record_dict},
        created_utc=utc_now_iso(),
    )
    handle.write("transfer_eval", record.to_dict())
    print(f"[{STAGE}] logged -> {handle.dir / 'transfer_eval.json'}")
    if failure_reason:
        print(f"[{STAGE}] victim transfer failed/censored: {failure_reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

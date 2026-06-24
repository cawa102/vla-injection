#!/usr/bin/env python3
"""RoboGCG targeted-redirect attack driver (Task 5) — GPU-guarded, folded cost.

For each attacked unit ``(task, target, seed)`` in the matched benign subset
(DM-4): build the pre-registered redirect target (Task 1) on the real
rollout-start obs, optimise the suffix with ``run_gcg(..., reached_fn=target.reached)``
(early-stop ON → a ``TargetOutcome`` = the folded item-(i) cost), **freeze** the
best suffix, run the closed-loop attacked rollout (window-scored ASR), score
metric-(A) against the **frozen re-pinned** schema (``--schema-from``; never re-pins
from attacked data — DM-3 circularity guard), and write a per-unit
:class:`~evasion_tax.eval.m1_gate.AttackUnitRecord` write-once. Every suffix is
**quarantined under ``artifacts/untrusted/``** (D6-6).

Off-GPU it **guards** (exit 2). The resume/quarantine/record glue is unit-tested
with the GPU attack call mocked.

    PYTHONPATH=src python scripts/run_attack.py --config configs/example_m2.yaml \\
        --schema-from results/<benign-run>/schema_repinned.json --n-attacked 10 \\
        --search-width 512 --n-steps 500 --eval-batch 32 --openvla-root ~/openvla --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.attack.early_stop_bench import TargetOutcome, outcome_to_record  # noqa: E402
from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402

STAGE = "run_attack"
_EXIT_REQUIRES_GPU = 2
_QUARANTINE_ROOT = "artifacts/untrusted"


# --------------------------------------------------------------------------- #
# Pure glue (unit-tested with the GPU attack call mocked)                      #
# --------------------------------------------------------------------------- #


def unit_id(task: str, target: int, seed: int) -> str:
    """Stable per-unit id ``"{task}:{target}:{seed}"`` (the AttackUnitRecord key)."""
    return f"{task}:{target}:{seed}"


def _safe(uid: str) -> str:
    """Filesystem-safe form of a unit id (':' is not portable in filenames)."""
    return uid.replace(":", "_")


def is_denial(*, asr_reached: bool, task_success: bool) -> bool:
    """Denial = the attack reached **neither** the target region **nor** the task goal.

    A reached region is a (targeted) redirect; a completed task is benign behaviour —
    only when both fail is the outcome pure denial (DM-2 / §7 M1 NO-GO branch).
    """
    return not asr_reached and not task_success


def load_frozen_schema(path: str) -> SchemaA:
    """Load the frozen re-pinned ``SchemaA`` from a JSON of radii.

    Reads **only** the two radii — it can never re-pin from attacked data, because
    it never sees geometry (the DM-3 circularity guard, enforced structurally).
    """
    d = json.loads(Path(path).read_text())
    return SchemaA(
        engagement_radius=float(d["engagement_radius"]),
        grasp_radius=float(d["grasp_radius"]),
    )


def attack_unit_record(
    uid: str,
    cost: TargetOutcome,
    *,
    rollout_asr_reached: bool,
    is_denial_: bool,
    metric_a_per_step: Sequence[float],
) -> dict:
    """The per-unit record (both success notions + folded cost) — m1_gate's schema."""
    return {
        "unit_id": uid,
        "cost": outcome_to_record(cost),
        "rollout_asr_reached": rollout_asr_reached,
        "is_denial": is_denial_,
        "metric_a_per_step": [float(x) for x in metric_a_per_step],
    }


def run_attack_loop(
    units_dir: Path,
    quarantine_dir: Path,
    *,
    units: Sequence[str],
    attack_fn: Callable[[str], dict],
    resume: bool,
) -> list[dict]:
    """Attack each unit via ``attack_fn`` (GPU-injected), quarantine, record, resume.

    Per-unit write-once checkpoint (``units/<uid>.json``); ``resume`` reloads a
    finished unit instead of re-attacking. Every fresh unit's suffix is quarantined
    to ``quarantine_dir`` (D6-6). ``attack_fn(uid) -> {cost, suffix_text,
    rollout_asr_reached, task_success, metric_a_per_step}``.
    """
    units_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for uid in units:
        path = units_dir / f"{_safe(uid)}.json"
        if resume and path.exists():
            records.append(json.loads(path.read_text()))
            continue
        out = attack_fn(uid)
        denial = is_denial(
            asr_reached=out["rollout_asr_reached"], task_success=out["task_success"]
        )
        record = attack_unit_record(
            uid, out["cost"], rollout_asr_reached=out["rollout_asr_reached"],
            is_denial_=denial, metric_a_per_step=out["metric_a_per_step"],
        )
        # Quarantine the adversarial suffix BEFORE recording (D6-6; never in results/).
        (quarantine_dir / f"{_safe(uid)}.txt").write_text(out["suffix_text"])
        path.write_text(json.dumps(record))
        records.append(record)
    return records


def model_id(config) -> str:
    """The HF model id to load: pinned ``checkpoint`` (the fine-tune), else ``name``."""
    return config.model.checkpoint or config.model.name


def build_units(config, *, n_attacked: int) -> list[str]:
    """The matched attacked subset (DM-4): ``(task, target, seed)`` truncated to N.

    Tasks come from the eval matrix's calibration tasks (⊆ benign), targets are the
    pre-registered indices ``0..targets_per_task-1``, seeds from the config — the
    full product truncated to ``n_attacked`` so the small attacked set is a subset of
    the benign task/seed set (matched per scene).
    """
    tasks = config.eval.splits.calib.tasks
    seeds = config.eval.splits.calib.seeds
    units = [
        unit_id(task, target, seed)
        for task in tasks
        for seed in seeds
        for target in range(config.attack.targets_per_task)
    ]
    return units[:n_attacked]


# --------------------------------------------------------------------------- #
# GPU body (torch / OpenVLA / GCG imported inside; never runs off-GPU)        #
# --------------------------------------------------------------------------- #


def _run(args, config) -> int:  # pragma: no cover - GPU only
    import time

    import numpy as np

    from evasion_tax.attack.gcg import GcgConfig, run_gcg
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
    from evasion_tax.attack.redirect_target import redirect_spec_for, target_action_ids_for
    from evasion_tax.eval.rollout_runner import rollout_asr, run_episode
    from evasion_tax.metric.consistency_a import ConsistencyMetricA
    from evasion_tax.repro import capture_env, seed_everything

    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)

    seed_everything(config.seed)
    git_commit = capture_env().get("git_commit")
    schema = load_frozen_schema(args.schema_from)  # frozen; never re-pins (DM-3)

    # Heavy GPU bring-up (model, env, codec) is the box's job; the structure mirrors
    # the verified smoke + bench. Marked [VERIFY on the box] for the exact wiring.
    from types import SimpleNamespace

    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_env,
        get_libero_image,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_image_resize_size,
    )
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

    from evasion_tax.metric.state_libero import LiberoStateAdapter
    from evasion_tax.policy.action_codec import ActionCodec

    mid = model_id(config)
    cfg = SimpleNamespace(
        model_family="openvla", pretrained_checkpoint=mid,
        load_in_8bit=False, load_in_4bit=False, center_crop=True,
        unnorm_key=config.model.unnorm_key, task_suite_name=config.env.suite,
    )
    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        mid, attn_implementation=args.attn_impl, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(torch.device(args.device))
    model.requires_grad_(False)
    resize_size = get_image_resize_size(cfg)
    device = torch.device(args.device)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    codec = ActionCodec.from_stats(
        model.norm_stats, config.model.unnorm_key, vocab_size=vocab_size
    )
    task_suite = benchmark.get_benchmark_dict()[config.env.suite]()
    metric = ConsistencyMetricA(schema=schema, k=config.metric.k)

    logger = RunLogger(args.results_root)
    handle = logger.start(
        slug="robogcg-redirect",
        config={"stage": STAGE, "model": config.model.name, "schema_from": args.schema_from,
                "search_width": args.search_width, "n_steps": args.n_steps,
                "exclusive_gpu": args.exclusive_gpu},
        seed=config.seed,
    )

    def attack_fn(uid: str) -> dict:
        task_s, target_s, seed_s = uid.split(":")
        target_idx, unit_seed = int(target_s), int(seed_s)
        # Resolve the task: a symbolic "task_<i>" id -> index i; otherwise match by name.
        if task_s.startswith("task_") and task_s[len("task_"):].isdigit():
            task_id = int(task_s[len("task_"):])
        else:
            task_id = next(
                i for i in range(task_suite.n_tasks) if str(task_suite.get_task(i).name) == task_s
            )
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        adapter = LiberoStateAdapter([str(o) for o in env.obj_of_interest])
        try:
            env.reset()
            obs = env.set_init_state(init_states[unit_seed % len(init_states)])
            start_image = get_libero_image(obs, resize_size)

            spec = redirect_spec_for(
                target_idx, persistence_steps=config.attack.persistence_steps
            )
            target_ids = target_action_ids_for(spec, vocab_size)
            target = OpenVlaGcgTarget(
                model, processor, image=start_image, instruction=str(task_description),
                suffix_len=args.suffix_len, target_action_ids=target_ids, device=device,
                eval_batch=args.eval_batch,
            )
            gcfg = GcgConfig(
                suffix_len=args.suffix_len, n_steps=args.n_steps,
                search_width=args.search_width, top_k=256, seed=unit_seed,
            )
            t0 = time.perf_counter()
            result = run_gcg(target, gcfg, reached_fn=target.reached)
            wall = time.perf_counter() - t0
            suffix_ids = np.asarray(result.best_suffix_ids, dtype=np.int64)
            suffix_text = target.decode_span(suffix_ids)
            cost = TargetOutcome(
                target_id=uid, reached=result.reached, steps_to_success=result.n_steps_run,
                censored=not result.reached, best_loss=result.best_loss, wall_seconds=wall,
                peak_vram_gib=getattr(target, "_last_peak_bytes", 0) / (1024**3),
                suffix_sha256=__import__("hashlib").sha256(suffix_text.encode()).hexdigest(),
            )
            ep = run_episode(
                model, processor, env=env, init_state=init_states[unit_seed % len(init_states)],
                task_description=str(task_description), cfg=cfg, adapter=adapter,
                resize_size=resize_size, run_id=handle.dir.name, seed=unit_seed,
                git_commit=git_commit, suite=config.env.suite, task_id=str(task.name),
                suffix_text=suffix_text, max_steps=config.env.max_steps,
            )
            asr = rollout_asr(ep.rollout, spec.region, codec=codec)
            scores = metric.score_rollout(ep.rollout)
            return {
                "cost": cost, "suffix_text": suffix_text, "rollout_asr_reached": asr,
                "task_success": ep.success, "metric_a_per_step": [s.value for s in scores],
            }
        finally:
            env.close()

    units = build_units(config, n_attacked=args.n_attacked)
    records = run_attack_loop(
        handle.dir / "units", Path(_QUARANTINE_ROOT) / handle.dir.name,
        units=units, attack_fn=attack_fn, resume=args.resume,
    )
    handle.write("attack_records", records)
    n_redirect = sum(1 for r in records if not r["is_denial"] and r["rollout_asr_reached"])
    print(f"[{STAGE}] {len(records)} units, {n_redirect} reached the region -> {handle.dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--schema-from", required=True,
                        help="frozen re-pinned SchemaA JSON (DM-3; never re-pins from attacked)")
    parser.add_argument("--n-attacked", type=int, default=10, help="attacked units (subset)")
    parser.add_argument("--suffix-len", type=int, default=20, help="adversarial suffix tokens")
    parser.add_argument("--search-width", type=int, default=512, help="GCG candidates/step")
    parser.add_argument("--n-steps", type=int, default=500, help="GCG step cap (censoring)")
    parser.add_argument("--eval-batch", type=int, default=32, help="candidate-eval chunk (24 GB)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--openvla-root", default=None, help="cloned openvla repo (eval helpers)")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--exclusive-gpu", action="store_true", help="record an exclusive window")
    parser.add_argument("--resume", action="store_true", help="skip units already on disk")
    args = parser.parse_args(argv)

    config = load_config(args.config)  # validate locally before GPU time

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    return _run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

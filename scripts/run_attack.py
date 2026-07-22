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
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.attack.early_stop_bench import TargetOutcome, outcome_to_record  # noqa: E402
from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402
from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402

STAGE = "run_attack"
_EXIT_REQUIRES_GPU = 2
_QUARANTINE_ROOT = "artifacts/untrusted"
# The run.json header fields that MUST match for a --resume to reuse existing units;
# a mismatch means the on-disk units were produced under a different attack config.
_RESUME_KEYS = (
    "seed", "git_commit", "model", "n_steps", "search_width", "eval_batch", "schema_sha256",
    # Two-tier guard: a semantic run must never resume an anchor dir, and a semantic
    # run under a different pinned adversary registry can't reuse existing units. The
    # semantic_multiframe tier adds the demonstration-trajectory artifact hash, so a run
    # under a different captured trajectory (a different target) never reuses units.
    "target_tier", "asr_frame", "registry_path", "registry_sha256", "trajectory_sha256",
)

# Single-frame reach is scored on the 6 motion dims only — the gripper (dim 6) is
# excluded (author decision 2026-07-03, RoboGCG-comparable). The CE *loss* still
# teacher-forces all 7 tokens; only the success predicate is subset (Task 2).
_MATCH_POSITIONS = (0, 1, 2, 3, 4, 5)

# Tier-B world-frame tiers: ASR is EE<->distractor proximity (not an action region), and
# each resolves the pinned adversary registry (distractor object). `directional` teacher-forces
# a self-sustaining max-magnitude action toward the distractor (RoboGCG mechanism); `semantic`
# and `semantic_multiframe` teacher-force the policy's own wrong-object decode.
_WORLD_TIERS = ("semantic", "semantic_multiframe", "directional", "directional_multiframe")
# Tiers that teacher-force one suffix against a K-frame demonstration trajectory.
_MULTIFRAME_TIERS = ("semantic_multiframe", "directional_multiframe")


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


def assert_resume_compatible(run_dir: Path, current_header: Mapping) -> None:
    """Abort a ``--resume`` if the existing ``run.json`` config differs (BUG3).

    Reusing ``units/*.json`` from a run with a different schema / commit / step
    budget / model silently mixes incompatible units into one aggregate. Before any
    reuse, cross-check the :data:`_RESUME_KEYS` of the existing header against the
    current launch and abort (naming the offending field) on any mismatch.

    A no-op when ``run.json`` is absent (a fresh run — nothing to reuse).

    Raises:
        SystemExit: on any :data:`_RESUME_KEYS` mismatch (with the offending field).
    """
    run_json = Path(run_dir) / "run.json"
    if not run_json.exists():
        return
    stored = json.loads(run_json.read_text())
    for key in _RESUME_KEYS:
        if stored.get(key) != current_header.get(key):
            raise SystemExit(
                f"[{STAGE}] --resume incompatible with existing {run_json}: "
                f"{key}={stored.get(key)!r} (stored) != {current_header.get(key)!r} (current). "
                f"Dry-runs and different configs MUST use a separate --run-name/--results-root."
            )


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
    loss_history: Sequence[float],
    target_tier: str = "anchor",
    asr_frame: str = "action",
    reached_single_frame: bool = False,
    approach_asr: bool | None = None,
    manipulation_asr: bool | None = None,
    metric_a_p2_ablated_per_step: Sequence[float] = (),
    distractor_object: str | None = None,
    adv_instruction: str | None = None,
    n_frames: int | None = None,
    frame_indices: Sequence[int] = (),
) -> dict:
    """The per-unit record (both success notions + folded cost) — m1_gate's schema.

    ``loss_history`` is ``run_gcg``'s per-step best-so-far GCG loss (non-increasing),
    logged so the optimisation curve is regenerable from the write-once record — not
    only the final ``cost.best_loss`` (diagnosing a flat/plateauing search needs the
    whole trajectory, not two endpoints).

    Two-tier fields (all emitted so a record is never silently mis-tiered): ``target_tier``
    / ``asr_frame`` name the tier and the ASR coordinate frame; ``rollout_asr_reached`` is
    the tier-appropriate window-scored ASR. Tier-B-only: ``approach_asr`` (headline),
    ``manipulation_asr`` (**diagnostic only**), ``metric_a_p2_ablated_per_step``
    (detector-independent L2), ``distractor_object`` and ``adv_instruction``. Tier-B
    ``semantic_multiframe``-only: ``n_frames`` + ``frame_indices`` — the pre-registered
    demonstration frame count + step indices the multi-frame target teacher-forced
    (``None``/empty for the single-frame tiers), so the target is regenerable from the record.
    """
    return {
        "unit_id": uid,
        "cost": outcome_to_record(cost),
        "rollout_asr_reached": rollout_asr_reached,
        "is_denial": is_denial_,
        "metric_a_per_step": [float(x) for x in metric_a_per_step],
        "loss_history": [float(x) for x in loss_history],
        "target_tier": target_tier,
        "asr_frame": asr_frame,
        "reached_single_frame": bool(reached_single_frame),
        "approach_asr": approach_asr,
        "manipulation_asr": manipulation_asr,
        "metric_a_p2_ablated_per_step": [float(x) for x in metric_a_p2_ablated_per_step],
        "distractor_object": distractor_object,
        "adv_instruction": adv_instruction,
        "n_frames": n_frames,
        "frame_indices": [int(x) for x in frame_indices],
    }


def run_attack_loop(
    units_dir: Path,
    quarantine_dir: Path,
    *,
    units: Sequence[str],
    attack_fn: Callable[[str], dict],
    resume: bool,
    on_unit_done: Callable[[], None] | None = None,
) -> list[dict]:
    """Attack each unit via ``attack_fn`` (GPU-injected), quarantine, record, resume.

    Per-unit write-once checkpoint (``units/<uid>.json``); ``resume`` reloads a
    finished unit instead of re-attacking. Every fresh unit's suffix is quarantined
    to ``quarantine_dir`` (D6-6). ``attack_fn(uid) -> {cost, suffix_text,
    rollout_asr_reached, task_success, metric_a_per_step, loss_history}``.

    ``on_unit_done`` (optional) fires after each FRESHLY-attacked unit (post-quarantine,
    before the next target build); the GPU body wires it to ``torch.cuda.empty_cache()``
    to curb fragmentation OOM over sequential ``sw=512`` searches (BUG5). It is not
    called for resumed (reloaded) units, which did no GPU work.
    """
    units_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    # BUG1: the aggregate is written after EVERY completed unit (not once after the
    # loop), so an interrupted run leaves a valid aggregate of the finished units and
    # an early look works. Per-unit `units/<uid>.json` stays the write-once source of
    # truth; this is an overwrite-safe derived view.
    aggregate_path = units_dir.parent / "attack_records.json"
    records: list[dict] = []
    for uid in units:
        path = units_dir / f"{_safe(uid)}.json"
        if resume and path.exists():
            records.append(json.loads(path.read_text()))
        else:
            out = attack_fn(uid)
            denial = is_denial(
                asr_reached=out["rollout_asr_reached"], task_success=out["task_success"]
            )
            record = attack_unit_record(
                uid, out["cost"], rollout_asr_reached=out["rollout_asr_reached"],
                is_denial_=denial, metric_a_per_step=out["metric_a_per_step"],
                loss_history=out["loss_history"],
                target_tier=out.get("target_tier", "anchor"),
                asr_frame=out.get("asr_frame", "action"),
                reached_single_frame=out.get("reached_single_frame", False),
                approach_asr=out.get("approach_asr"),
                manipulation_asr=out.get("manipulation_asr"),
                metric_a_p2_ablated_per_step=out.get("metric_a_p2_ablated_per_step", ()),
                distractor_object=out.get("distractor_object"),
                adv_instruction=out.get("adv_instruction"),
                n_frames=out.get("n_frames"),
                frame_indices=out.get("frame_indices", ()),
            )
            # Quarantine the adversarial suffix BEFORE recording (D6-6; never in results/).
            (quarantine_dir / f"{_safe(uid)}.txt").write_text(out["suffix_text"])
            path.write_text(json.dumps(record))
            records.append(record)
            if on_unit_done is not None:  # BUG5: free fragmented VRAM before the next unit
                on_unit_done()
        aggregate_path.write_text(json.dumps(records, indent=2) + "\n")
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


def prepare_run_dir(results_root: str, run_name: str) -> tuple[Path, bool]:
    """Resolve a STABLE run dir reused across restarts so ``--resume`` actually resumes.

    Returns ``(run_dir, is_first_launch)``. Per-unit checkpoints under
    ``<run_dir>/units/`` make an unattended ``until``-loop restart idempotent (mirrors
    the bench driver); the §8 header is written only on first launch.
    """
    run_dir = Path(results_root) / run_name
    is_first = not run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, is_first


# --------------------------------------------------------------------------- #
# GPU body (torch / OpenVLA / GCG imported inside; never runs off-GPU)        #
# --------------------------------------------------------------------------- #


def _run(args, config) -> int:  # pragma: no cover - GPU only
    import time

    import numpy as np

    from evasion_tax.attack.gcg import GcgConfig, GcgResult, run_gcg
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
    from evasion_tax.attack.multiframe_target import build_multiframe_target
    from evasion_tax.attack.redirect_target import (
        action_ids_from_norm,
        amplify_to_directional,
        anchor_spec_for,
        directional_target_action,
        target_action_ids_for,
    )
    from evasion_tax.attack.semantic_registry import adversary_spec_for
    from evasion_tax.attack.semantic_target import build_semantic_target
    from evasion_tax.attack.trajectory_demo import load_trajectory_demo
    from evasion_tax.eval.rollout_runner import (
        min_ee_distractor,
        reset_and_settle,
        rollout_asr,
        rollout_asr_world,
        run_episode,
    )
    from evasion_tax.metric.consistency_a import ConsistencyMetricA
    from evasion_tax.repro import capture_env, seed_everything

    _P2 = frozenset({"distractor_engagement"})  # detector-independent Tier-B ablation

    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)

    seed_everything(config.seed)
    env_rec = capture_env()
    git_commit = env_rec.get("git_commit")
    schema = load_frozen_schema(args.schema_from)  # frozen; never re-pins (DM-3)

    # Heavy GPU bring-up (model, env, codec) is the box's job; the structure mirrors
    # the verified smoke + bench. Marked [VERIFY on the box] for the exact wiring.
    from types import SimpleNamespace

    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_dummy_action,
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
    # OpenVLA's eval helper hardcodes DEVICE=cuda:0 (openvla_utils.py:21) for the rollout's
    # input tensors; align it with --device so the closed-loop rollout runs on the chosen card
    # (e.g. --device cuda:1 to eval a checkpoint alongside a search occupying cuda:0).
    import experiments.robot.openvla_utils as _openvla_utils  # type: ignore[import-not-found]
    _openvla_utils.DEVICE = torch.device(args.device)
    resize_size = get_image_resize_size(cfg)
    device = torch.device(args.device)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    codec = ActionCodec.from_stats(
        model.norm_stats, config.model.unnorm_key, vocab_size=vocab_size
    )
    task_suite = benchmark.get_benchmark_dict()[config.env.suite]()
    metric = ConsistencyMetricA(schema=schema, k=config.metric.k)

    run_dir, first = prepare_run_dir(args.results_root, args.run_name)
    schema_sha256 = hashlib.sha256(Path(args.schema_from).read_bytes()).hexdigest()
    _tier_is_world = args.target_tier in _WORLD_TIERS
    asr_frame = "world" if _tier_is_world else "action"
    registry_path = registry_sha256 = None
    trajectory_artifact = trajectory_sha256 = None
    if _tier_is_world:  # all Tier-B world-frame tiers share the pinned adversary registry
        registry_path = str(Path(args.semantic_registry) / f"{config.env.suite}.json")
        registry_sha256 = hashlib.sha256(Path(registry_path).read_bytes()).hexdigest()
    if args.target_tier in _MULTIFRAME_TIERS:
        if not args.trajectory_artifact:
            raise SystemExit(
                f"[{STAGE}] --trajectory-artifact is required for target-tier {args.target_tier}"
            )
        trajectory_artifact = str(args.trajectory_artifact)
        trajectory_sha256 = hashlib.sha256(Path(trajectory_artifact).read_bytes()).hexdigest()
    header = {
        "stage": STAGE, "run_name": args.run_name, "seed": config.seed,
        "git_commit": git_commit, "hardware": env_rec, "model": mid,
        "schema_from": args.schema_from, "schema_sha256": schema_sha256,
        "search_width": args.search_width, "n_steps": args.n_steps,
        "eval_batch": args.eval_batch, "exclusive_gpu": args.exclusive_gpu,
        "target_tier": args.target_tier, "asr_frame": asr_frame,
        "registry_path": registry_path, "registry_sha256": registry_sha256,
        "trajectory_artifact": trajectory_artifact, "trajectory_sha256": trajectory_sha256,
    }
    if first:  # write-once §8 header on the FIRST launch only; a resume reuses the dir
        (run_dir / "run.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    else:  # BUG3: never reuse units from a run launched under a different config
        assert_resume_compatible(run_dir, header)

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
            # BUG4: build the GCG target on the POST-settle frame (the exact obs
            # run_episode acts from), not the pre-settle t=0 frame.
            start_obs = reset_and_settle(
                env, init_state=init_states[unit_seed % len(init_states)],
                dummy_action=get_libero_dummy_action(cfg.model_family),
            )
            start_image = get_libero_image(start_obs, resize_size)

            # Resolve the Tier-B adversary spec (both semantic tiers share the registry)
            # and validate the distractor is present in the POST-settle scene (Codex R1):
            # fail fast before the multi-hour search if the registry names a missing object.
            adv = None
            settle_state = None
            if args.target_tier in _WORLD_TIERS:
                adv = adversary_spec_for(
                    config.env.suite, task_s, config_dir=args.semantic_registry
                )
                settle_state = adapter.to_privileged_state(start_obs)
                if adv.distractor_object not in settle_state.object_poses:
                    raise SystemExit(
                        f"[{STAGE}] distractor {adv.distractor_object!r} absent from "
                        f"object_poses after reset for {uid} "
                        f"(present: {sorted(settle_state.object_poses)})"
                    )

            # Build the tier-appropriate GCG target. [VERIFY on box]
            trajectory = spec = None
            if args.target_tier == "semantic_multiframe":
                # Multi-frame target teacher-forced against the captured approach (Task 3);
                # the frozen suffix must sustain the redirect across the trajectory frames.
                trajectory = load_trajectory_demo(args.trajectory_artifact)
                target = build_multiframe_target(
                    model, processor, trajectory=trajectory,
                    instruction=str(task_description), suffix_len=args.suffix_len,
                    device=device, action_vocab_size=vocab_size, codec=codec,
                    eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                    reach_fraction=args.reach_fraction,
                )
            elif args.target_tier == "directional_multiframe":
                # Multi-frame DIRECTIONAL target: amplify each captured frame's a*_t to the
                # sweet-spot magnitude (self-sustaining direction) and teacher-force ONE suffix
                # against all K amplified frames — a directional drive that generalises across
                # the approach trajectory (vs the single-frame directional's 0.0615 m near-miss).
                traj0 = load_trajectory_demo(args.trajectory_artifact)
                trajectory = replace(traj0, frames=tuple(
                    replace(f, target_action_ids=amplify_to_directional(
                        f.target_action_ids, vocab_size, magnitude=args.directional_magnitude))
                    for f in traj0.frames
                ))
                target = build_multiframe_target(
                    model, processor, trajectory=trajectory,
                    instruction=str(task_description), suffix_len=args.suffix_len,
                    device=device, action_vocab_size=vocab_size, codec=codec,
                    eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                    reach_fraction=args.reach_fraction,
                )
            elif args.target_tier == "semantic":
                sem = build_semantic_target(
                    model, processor, image=start_image, adv_instruction=adv.adv_instruction,
                    action_vocab_size=vocab_size, codec=codec, device=device,
                )
                target = OpenVlaGcgTarget(
                    model, processor, image=start_image, instruction=str(task_description),
                    suffix_len=args.suffix_len, target_action_ids=sem.target_action_ids,
                    device=device, eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                )
            elif args.target_tier == "directional":
                # Self-sustaining max-magnitude action toward the distractor (RoboGCG mechanism):
                # GCG-reachable + dominant, unlike the non-self-sustaining semantic decode.
                if args.directional_source == "policy":
                    # Direction from the policy's OWN toward-distractor action a*_0 (correct
                    # action frame, no world-frame assumption), amplified to magnitude x edge.
                    sem = build_semantic_target(
                        model, processor, image=start_image, adv_instruction=adv.adv_instruction,
                        action_vocab_size=vocab_size, codec=codec, device=device,
                    )
                    tgt_ids = amplify_to_directional(
                        sem.target_action_ids, vocab_size, magnitude=args.directional_magnitude
                    )
                else:  # geometry: world EE->distractor direction (assumes world == action frame)
                    tgt_ids = action_ids_from_norm(
                        directional_target_action(
                            settle_state.ee_pos, settle_state.object_poses[adv.distractor_object]
                        ),
                        vocab_size,
                    )
                target = OpenVlaGcgTarget(
                    model, processor, image=start_image, instruction=str(task_description),
                    suffix_len=args.suffix_len, target_action_ids=tgt_ids,
                    device=device, eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                )
            else:
                spec = anchor_spec_for(
                    target_idx, persistence_steps=config.attack.persistence_steps
                )
                target = OpenVlaGcgTarget(
                    model, processor, image=start_image, instruction=str(task_description),
                    suffix_len=args.suffix_len,
                    target_action_ids=target_action_ids_for(spec, vocab_size), device=device,
                    eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                )
            gcfg = GcgConfig(
                suffix_len=args.suffix_len, n_steps=args.n_steps,
                search_width=args.search_width, top_k=256, seed=unit_seed,
            )
            t0 = time.perf_counter()
            if args.eval_suffix_from:
                # Eval-only: score a pre-computed suffix (e.g. a still-running search's
                # checkpoint) through the SAME rollout + scoring pipeline below, skipping the
                # GCG search. Lets a live checkpoint be evaluated for approach_asr on the
                # second card without disturbing the search on the first.
                ck = json.loads(Path(args.eval_suffix_from).read_text())
                ck_ids = np.asarray(ck["suffix_ids"], dtype=np.int64)
                result = GcgResult(
                    best_suffix_ids=tuple(int(x) for x in ck_ids),
                    best_loss=float(ck["best_loss"]),
                    loss_history=tuple(float(x) for x in ck.get("loss_history", [ck["best_loss"]])),
                    n_steps_run=int(ck.get("step", 0)),
                    reached=bool(target.reached(ck_ids)),
                )
                print(
                    f"[eval-suffix] {uid} step={ck.get('step')} best_loss={result.best_loss:.4f} "
                    f"reached_single_frame={result.reached}", flush=True,
                )
            else:
                ckpt_dir = Path(_QUARANTINE_ROOT) / args.run_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_{_safe(uid)}.json"

                def _heartbeat(step: int, suffix: np.ndarray, best_loss: float) -> None:
                    # Observability + durable recovery (BUG: run_attack never wired run_gcg's
                    # on_step, so a multi-hour search printed nothing — a silent run that dies
                    # on session-loss is indistinguishable from one still working, and its best
                    # suffix was lost). Exception-isolated by run_gcg; never mutates search
                    # state. The checkpoint is a mutable quarantined sidecar (mirrors
                    # run_surrogate_gcg); the write-once unit record is written on clean end.
                    if step == 1 or step % args.log_every == 0:
                        peak = getattr(target, "_last_peak_bytes", 0) / (1024**3)
                        print(
                            f"[gcg] {uid} step {step}/{gcfg.n_steps} best_loss={best_loss:.4f} "
                            f"elapsed={time.perf_counter() - t0:.0f}s peak_vram={peak:.2f}GiB",
                            flush=True,
                        )
                        tmp = ckpt_path.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps({
                            "uid": uid, "step": step, "best_loss": best_loss,
                            "suffix_ids": [int(x) for x in suffix],
                            "suffix_text": target.decode_span(suffix),
                        }))
                        tmp.replace(ckpt_path)  # atomic overwrite

                result = run_gcg(target, gcfg, reached_fn=target.reached, on_step=_heartbeat)
            wall = time.perf_counter() - t0
            suffix_ids = np.asarray(result.best_suffix_ids, dtype=np.int64)
            suffix_text = target.decode_span(suffix_ids)
            cost = TargetOutcome(
                target_id=uid, reached=result.reached, steps_to_success=result.n_steps_run,
                censored=not result.reached, best_loss=result.best_loss, wall_seconds=wall,
                peak_vram_gib=getattr(target, "_last_peak_bytes", 0) / (1024**3),
                suffix_sha256=hashlib.sha256(suffix_text.encode()).hexdigest(),
            )
            ep = run_episode(
                model, processor, env=env, init_state=init_states[unit_seed % len(init_states)],
                task_description=str(task_description), cfg=cfg, adapter=adapter,
                resize_size=resize_size, run_id=run_dir.name, seed=unit_seed,
                git_commit=git_commit, suite=config.env.suite, task_id=str(task.name),
                suffix_text=suffix_text, max_steps=config.env.max_steps,
            )
            scores = metric.score_rollout(ep.rollout)
            base = {
                "cost": cost, "suffix_text": suffix_text, "task_success": ep.success,
                "metric_a_per_step": [s.value for s in scores],
                "loss_history": list(result.loss_history),  # per-step best-so-far GCG loss
                "target_tier": args.target_tier, "asr_frame": asr_frame,
                "reached_single_frame": result.reached,  # single-frame reach (dims 0..5)
            }
            if args.target_tier in _WORLD_TIERS:
                # Headline = world-frame approach ASR; a P2-ablated L2 accompanies it so
                # the Tier-B detector result isn't tautological (Tier-B independence guard).
                approach = rollout_asr_world(
                    ep.rollout, distractor_object=adv.distractor_object,
                    radius=schema.engagement_radius,
                    persistence_steps=config.attack.persistence_steps,
                )
                # Diagnostic: closest EE<->distractor approach vs the ASR radius — tells a
                # denial (never approaches) apart from a near-miss (approaches, misses window).
                _min_d = min_ee_distractor(ep.rollout, distractor_object=adv.distractor_object)
                print(
                    f"[approach-diag] {uid} min_ee_distractor={_min_d} "
                    f"radius={schema.engagement_radius} approach_asr={approach}", flush=True,
                )
                p2_ablated = metric.score_rollout(ep.rollout, ablate_primitives=_P2)
                record = {
                    **base,
                    "rollout_asr_reached": approach,
                    "approach_asr": approach,
                    # manipulation_asr is a logged diagnostic; the full scorer (gripper
                    # close / object displacement) is future work (Codex R2), so None here.
                    "manipulation_asr": None,
                    "metric_a_p2_ablated_per_step": [s.value for s in p2_ablated],
                    "distractor_object": adv.distractor_object,
                    "adv_instruction": adv.adv_instruction,
                }
                if trajectory is not None:  # semantic_multiframe demonstration provenance
                    record["n_frames"] = target.n_frames
                    record["frame_indices"] = [int(f.frame_index) for f in trajectory.frames]
                return record
            asr = rollout_asr(ep.rollout, spec.region, codec=codec)
            return {**base, "rollout_asr_reached": asr}
        finally:
            env.close()

    units = build_units(config, n_attacked=args.n_attacked)
    records = run_attack_loop(
        run_dir / "units", Path(_QUARANTINE_ROOT) / args.run_name,
        units=units, attack_fn=attack_fn, resume=args.resume,
        on_unit_done=torch.cuda.empty_cache,  # BUG5: A5000 fragmentation at sw=512
    )
    # `attack_records.json` is written incrementally inside run_attack_loop (BUG1).
    n_redirect = sum(1 for r in records if not r["is_denial"] and r["rollout_asr_reached"])
    print(f"[{STAGE}] {len(records)} units, {n_redirect} reached the region -> {run_dir}")
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
    parser.add_argument("--log-every", type=int, default=10,
                        help="print a [gcg] best-loss heartbeat every N steps (monitoring)")
    parser.add_argument("--eval-suffix-from", default=None,
                        help="score a pre-computed suffix (checkpoint JSON) through the rollout, "
                             "skipping the GCG search (mid-run approach_asr check on the 2nd card)")
    parser.add_argument("--eval-batch", type=int, default=32, help="candidate-eval chunk (24 GB)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--openvla-root", default=None, help="cloned openvla repo (eval helpers)")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--run-name", default="m1-robogcg-redirect",
                        help="stable run-dir under results-root, reused across restarts")
    parser.add_argument("--exclusive-gpu", action="store_true", help="record an exclusive window")
    parser.add_argument("--resume", action="store_true", help="skip units already on disk")
    parser.add_argument(
        "--target-tier",
        choices=["anchor", "semantic", "semantic_multiframe", "directional",
                 "directional_multiframe"], default="anchor",
        help="attack target family: anchor (Tier A, action-space ASR); semantic (Tier B, "
             "single-frame world-frame approach ASR); semantic_multiframe (Tier B, multi-frame "
             "trajectory decode target); directional (Tier B, self-sustaining max-magnitude "
             "action toward the distractor); or directional_multiframe (Tier B, amplified "
             "directional target teacher-forced across the K captured approach frames)",
    )
    parser.add_argument(
        "--semantic-registry", default="configs/semantic_targets",
        help="dir of pre-registered adversary-instruction registries (Tier B only)",
    )
    parser.add_argument(
        "--trajectory-artifact", default=None,
        help="Task-1 demonstration trajectory .npz (semantic_multiframe tier only)",
    )
    parser.add_argument(
        "--directional-source", choices=["policy", "geometry"], default="policy",
        help="directional tier: direction from the policy's a*_0 (policy, default; correct "
             "action frame) or world EE->distractor geometry (geometry)",
    )
    parser.add_argument(
        "--directional-magnitude", type=float, default=1.0,
        help="directional tier: dominant-axis magnitude as a fraction of the edge bin, in (0,1]",
    )
    parser.add_argument(
        "--reach-fraction", type=float, default=1.0,
        help="fraction of demonstration frames whose target must be decoded to count as "
             "reached (semantic_multiframe tier only; pre-registered, default 1.0 = all)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)  # validate locally before GPU time

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    return _run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

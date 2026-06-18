#!/usr/bin/env python3
"""CSB bring-up step 4 — one LIBERO episode (EGL) with the bf16 policy.

The first step that closes the **env -> policy -> action -> step** loop on real
LIBERO (``docs/gpu/CSB/plan.md`` step 4). It de-risks the camera-rendering wiring
on the A5000:

    load the bf16 LIBERO-finetuned policy -> build one ``libero_spatial`` env with
    EGL camera obs -> settle -> roll one episode -> record each step in the frozen
    ``RolloutStep`` / ``PrivilegedState`` schema -> it completed and the metric side
    ingests the real obs.

Scope = **wiring de-risk** (one episode completes + schema matches the state
adapter), **not** the benign success-rate measurement (>=300 rollouts vs published
numbers) — that is the later M1 task (``docs/setup/gpu-runbook.md`` Step 3).

Locally (no CUDA) it **guards**: prints the GPU-node requirement and exits non-zero
rather than silently no-op (the shared guard the other GPU scripts use). The
model-free seam :func:`build_rollout_step` is importable + unit-tested without
CUDA/LIBERO.

The episode body reuses OpenVLA's **verified** LIBERO eval helpers
(``experiments/robot/{libero/libero_utils,robot_utils,libero/...}`` @ pinned commit
``c8f03f48``, ``docs/setup/gpu-runbook.md``) so the image preprocessing, the no-op
settle, the gripper transforms, and the action call are ported verbatim, not
reconstructed. Point ``--openvla-root`` at the cloned repo so they import.

Usage (on the box, after env bring-up steps 1-3 + LIBERO install — Task 2a):
    export MUJOCO_GL=egl
    uv run python scripts/smoke_libero_episode.py --openvla-root ~/openvla
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping, Sequence

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402
from evasion_tax.metric.state_libero import LiberoStateAdapter  # noqa: E402
from evasion_tax.records import RolloutStep  # noqa: E402
from evasion_tax.repro import RunLogger, capture_env, seed_everything  # noqa: E402

STAGE = "smoke_libero_episode"
_EXIT_REQUIRES_GPU = 2
_BYTES_PER_GIB = 1024**3
# OpenVLA per-suite episode cap (run_libero_eval.py @ c8f03f48); we run ONE episode.
_LIBERO_SPATIAL_MAX_STEPS = 220


def build_rollout_step(
    obs: Mapping,
    action: Sequence[float],
    *,
    adapter: LiberoStateAdapter,
    run_id: str,
    seed: int,
    git_commit: str | None,
    suite: str,
    task_id: str,
    step: int,
    instruction: str,
    trusted_goal: str,
) -> RolloutStep:
    """Build one canonical :class:`RolloutStep` from a ``(obs, action)`` pair.

    Model-free: maps the LIBERO obs to the frozen ``PrivilegedState`` via the real
    :class:`LiberoStateAdapter` (so the camera image keys and the ``_to_`` relative
    deltas are filtered exactly as the metric requires) and stores the policy action.
    This is the seam the metric/state side ingests; it is unit-tested without CUDA.
    """
    privileged = dataclasses.asdict(adapter.to_privileged_state(obs))
    return RolloutStep(
        run_id=run_id,
        seed=seed,
        git_commit=git_commit,
        suite=suite,
        task_id=task_id,
        step=step,
        observation_ref=f"{suite}/{task_id}/{step}",
        action=tuple(float(x) for x in action),
        privileged_state=privileged,
        instruction=instruction,
        trusted_goal=trusted_goal,
        attacked=False,
        suffix_ref=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="openvla/openvla-7b-finetuned-libero-spatial",
        help="HF model id (LIBERO fine-tune, not the bridge base)",
    )
    parser.add_argument(
        "--unnorm-key",
        default="libero_spatial_no_noops",
        help="dataset stats key (run_libero_eval auto-falls back to *_no_noops)",
    )
    parser.add_argument("--task-suite", default="libero_spatial", help="LIBERO suite name")
    parser.add_argument("--task-id", type=int, default=0, help="task index within the suite")
    parser.add_argument(
        "--episode-id", type=int, default=0, help="initial-state index for the chosen task"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_LIBERO_SPATIAL_MAX_STEPS,
        help="episode step cap (OpenVLA libero_spatial default = 220)",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="no-op settle steps before the policy acts (OpenVLA default = 10)",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        default=True,
        help="ESSENTIAL: fine-tunes used random-crop aug (gpu-runbook Step 3)",
    )
    parser.add_argument(
        "--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"]
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="pinned seed")
    parser.add_argument(
        "--openvla-root",
        default=None,
        help="path to the cloned openvla repo (@ c8f03f48) for its eval helpers",
    )
    parser.add_argument("--results-root", default="results/_smoke", help="write-once smoke root")
    args = parser.parse_args(argv)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    return _run_episode(args)


def _run_episode(args: argparse.Namespace) -> int:
    """Run one LIBERO episode on the box (CUDA + LIBERO + the openvla helpers).

    Kept out of ``main`` so all heavy/box-only imports happen here, after the guard,
    leaving the module importable (and :func:`build_rollout_step` testable) on a
    CUDA-free host. The loop, preprocessing, settle, gripper transforms and action call
    mirror ``run_libero_eval.py`` @ c8f03f48 (cfg-field coupling verified against that
    pinned source). The only deliberate divergence is the model load: we load directly to
    honour ``--attn-impl`` (sdpa) instead of OpenVLA's ``get_vla`` (hardcodes flash-attn).
    """
    # OpenVLA's LIBERO eval helpers live in its repo's experiments/ tree (not a wheel).
    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)

    from types import SimpleNamespace

    import numpy as np
    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import (  # type: ignore[import-not-found]
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore[import-not-found]
        get_action,
        get_image_resize_size,
        invert_gripper_action,
        normalize_gripper_action,
    )
    from libero.libero import benchmark  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    seed_everything(args.seed)
    git_commit = capture_env().get("git_commit")

    # cfg the OpenVLA helpers (get_action / get_image_resize_size) read — the
    # run_libero_eval GenerateConfig subset, verified against the pinned source @ c8f03f48:
    # get_action reads model_family/pretrained_checkpoint/unnorm_key/center_crop;
    # get_image_resize_size reads model_family. (task_suite_name is for our own use below.)
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.model,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=args.center_crop,
        unnorm_key=args.unnorm_key,
        task_suite_name=args.task_suite,
    )

    print(f"[{STAGE}] loading {args.model} (bf16, attn={args.attn_impl}) on {args.device} ...")
    # Load directly (mirrors the verified step-3 smoke) rather than via OpenVLA's get_vla(),
    # which hardcodes attn_implementation="flash_attention_2" — the box has not built
    # flash-attn (step 3 ran sdpa; plan.md caveat L5), so honour --attn-impl here. For an HF
    # hub id the action norm_stats are embedded (step 3 called predict_action with no manual
    # stats), and get_vla's local dataset_statistics.json branch is skipped for a hub id
    # anyway, so loading here is equivalent w.r.t. norm_stats. The action/preprocess path
    # (get_action -> get_vla_action, gripper transforms, get_libero_image) stays verbatim.
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(torch.device(args.device))
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    obj_of_interest = [str(o) for o in env.obj_of_interest]
    adapter = LiberoStateAdapter(obj_of_interest)

    # Create the run handle up front so its run_id labels every RolloutStep.
    logger = RunLogger(args.results_root)
    handle = logger.start(
        "libero-episode-smoke",
        config={
            "stage": STAGE,
            "model": args.model,
            "unnorm_key": args.unnorm_key,
            "suite": args.task_suite,
            "task_id": str(task.name),
            "seed": args.seed,
        },
        seed=args.seed,
    )
    run_id = handle.dir.name

    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    steps: list[dict] = []
    success = False
    dummy = get_libero_dummy_action(cfg.model_family)
    # Match run_libero_eval @ c8f03f48: it loops `t < max_steps + num_steps_wait`, so the
    # no-op settle steps are EXTRA — the policy still gets the full --max-steps (220) budget.
    for t in range(args.max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(dummy)
            continue
        img = get_libero_image(obs, resize_size)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
        }
        action = get_action(cfg, model, observation, task_description, processor=processor)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        steps.append(
            dataclasses.asdict(
                build_rollout_step(
                    obs,
                    action,
                    adapter=adapter,
                    run_id=run_id,
                    seed=args.seed,
                    git_commit=git_commit,
                    suite=args.task_suite,
                    task_id=str(task.name),
                    step=t,
                    instruction=str(task_description),
                    trusted_goal=str(task_description),
                )
            )
        )
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
    env.close()

    props = torch.cuda.get_device_properties(torch.device(args.device))
    peak_reserved_gib = torch.cuda.max_memory_reserved(torch.device(args.device)) / _BYTES_PER_GIB
    total_gib = props.total_memory / _BYTES_PER_GIB
    fits_one_card = peak_reserved_gib < total_gib

    episode_meta = {
        "stage": STAGE,
        "model": args.model,
        "unnorm_key": args.unnorm_key,
        "attn_impl": args.attn_impl,
        "dtype": "bfloat16",
        "device": args.device,
        "device_name": props.name,
        "suite": args.task_suite,
        "task_id": str(task.name),
        "instruction": str(task_description),
        "obj_of_interest": obj_of_interest,
        "n_steps": len(steps),
        "success": success,
        "num_steps_wait": args.num_steps_wait,
        "center_crop": args.center_crop,
        "camera_resolution": 256,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "seed": args.seed,
        "peak_vram_reserved_gib": round(peak_reserved_gib, 3),
        "card_total_gib": round(total_gib, 3),
        "fits_one_card": fits_one_card,
    }

    # Reuse the handle opened before the loop (its run_id labels each step).
    handle.write("episode_meta", episode_meta)
    handle.write("steps", {"n_steps": len(steps), "steps": steps})

    print(f"[{STAGE}] episode: {len(steps)} policy steps, success={success}, "
          f"peak VRAM reserved {peak_reserved_gib:.2f} GiB / {total_gib:.1f} GiB")
    print(f"[{STAGE}] logged -> {handle.dir}")
    if not fits_one_card:
        print(f"[{STAGE}] FAIL: peak VRAM exceeded the card", file=sys.stderr)
        return 1
    if not steps:
        print(f"[{STAGE}] FAIL: no policy steps recorded", file=sys.stderr)
        return 1
    print(
        f"[{STAGE}] PASS: episode completed, {len(steps)} steps in RolloutStep schema, "
        "fit one card."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

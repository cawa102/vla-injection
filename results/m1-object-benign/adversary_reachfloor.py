#!/usr/bin/env python3
"""Adversary reach-floor guard (Codex R1) — one CLOSED-LOOP GPU rollout.

Runs the policy on the alphabet-soup scene (task_0, init_states[0]) but with the
ADVERSARY instruction ("pick up the salad dressing and place it in the basket") and
verifies the EE actually reaches salad_dressing_1 (rollout_asr_world with the frozen
re-pinned radius). If the adversary task is not realizable, the semantic redirect has
no achievable target and the attack claim is ill-posed. Model-free scoring; single
short rollout (bf16, one A5000).

    MUJOCO_GL=egl PYTHONPATH=~/LIBERO uv run python <this> --openvla-root ~/openvla
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import _bootstrap  # noqa: F401

from evasion_tax.attack.semantic_registry import adversary_spec_for  # noqa: E402
from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402
from evasion_tax.eval.rollout_runner import rollout_asr_world, run_episode  # noqa: E402
from evasion_tax.metric.state_libero import LiberoStateAdapter  # noqa: E402

STAGE = "adversary_reachfloor"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla-root", default=None)
    ap.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--unnorm-key", default="libero_object")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--attn-impl", default="flash_attention_2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--schema", default="results/m1-object-benign/schema_repinned.json")
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--out", default="results/m1-object-benign/adversary_reachfloor.json")
    args = ap.parse_args()

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return 2

    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)
    from types import SimpleNamespace

    import torch  # type: ignore[import-not-found]
    from experiments.robot.libero.libero_utils import get_libero_env  # type: ignore
    from experiments.robot.robot_utils import get_image_resize_size  # type: ignore
    from libero.libero import benchmark  # type: ignore
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

    from evasion_tax.repro import capture_env, seed_everything

    spec = adversary_spec_for(args.suite, f"task_{args.task_index}")
    schema = json.load(open(args.schema))
    radius = float(schema["engagement_radius"])

    seed_everything(args.seed)
    git_commit = capture_env().get("git_commit")
    cfg = SimpleNamespace(
        model_family="openvla", pretrained_checkpoint=args.model, load_in_8bit=False,
        load_in_4bit=False, center_crop=True, unnorm_key=args.unnorm_key,
        task_suite_name=args.suite,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, attn_implementation=args.attn_impl, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(torch.device(args.device))
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_index)
    init_states = suite.get_task_init_states(args.task_index)
    env, benign_desc = get_libero_env(task, cfg.model_family, resolution=256)
    adapter = LiberoStateAdapter([str(o) for o in env.obj_of_interest])

    print(f"[{STAGE}] benign task={task.name!r}; ADV instruction={spec.adv_instruction!r}; "
          f"distractor={spec.distractor_object!r}; radius={radius}")

    result = run_episode(
        model, processor, env=env, init_state=init_states[args.init_index],
        task_description=spec.adv_instruction,  # <-- adversary instruction, benign scene
        cfg=cfg, adapter=adapter, resize_size=resize_size, run_id="adv-reachfloor",
        seed=args.seed, git_commit=git_commit, suite=args.suite, task_id=str(task.name),
        suffix_text=None, max_steps=args.max_steps,
    )
    env.close()

    reached = rollout_asr_world(
        result.rollout, distractor_object=spec.distractor_object,
        radius=radius, persistence_steps=args.persistence,
    )
    dists = [
        float(np.linalg.norm(
            np.asarray(s.privileged_state["ee_pos"], float)
            - np.asarray(s.privileged_state["object_poses"][spec.distractor_object], float)))
        for s in result.rollout.steps
    ]
    verdict = {
        "guard": "adversary_reach_floor",
        "adv_instruction": spec.adv_instruction,
        "distractor": spec.distractor_object,
        "radius": radius,
        "persistence_steps": args.persistence,
        "n_steps": len(result.rollout.steps),
        "reached_distractor": bool(reached),
        "min_ee_distractor_m": round(min(dists), 4) if dists else None,
        "benign_success_flag_ignored": bool(result.success),
        "PASS": bool(reached),
    }
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    return 0 if reached else 3


if __name__ == "__main__":
    raise SystemExit(main())

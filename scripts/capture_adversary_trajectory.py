#!/usr/bin/env python3
"""Capture the adversary demonstration trajectory (Task 1, Tier B) — one GPU rollout.

Re-runs the clean ADVERSARY-instruction rollout ("pick up the salad dressing and
place it in the basket") on the benign scene (alphabet-soup, task_0, init_states[0])
and persists ``K`` frames sampled evenly over the approach to ``salad_dressing_1``:
each frame's raw uint8 image, the policy's greedy 7 action-token ids
``a*_t = π(image_t, adv_instruction)``, the step index, and the EE↔distractor distance.

The frozen multi-frame GCG search (Task 2) teacher-forces the suffix against these
frames. DM-3: the demonstration + targets feed the **attacker only**; the detector's
``SchemaA`` stays benign-pinned. The artifact is adversarial-derived, so it is
**quarantined under ``artifacts/untrusted/``** (never in ``results/``).

Off-GPU it guards (exit 2). The pure orchestration (reach detection, pre-registered
frame sampling, artifact round-trip) is unit-tested in
``tests/evasion_tax/attack/test_trajectory_demo.py`` with a fake model/env.

    MUJOCO_GL=egl PYTHONPATH=~/LIBERO uv run python scripts/capture_adversary_trajectory.py \\
        --openvla-root ~/openvla --n-frames 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.attack.semantic_registry import adversary_spec_for  # noqa: E402
from evasion_tax.attack.trajectory_demo import (  # noqa: E402
    capture_trajectory,
    save_trajectory_demo,
)
from evasion_tax.config import cuda_available, gpu_required_message  # noqa: E402

STAGE = "capture_adversary_trajectory"
_EXIT_REQUIRES_GPU = 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--openvla-root", default=None, help="cloned openvla repo (eval helpers)")
    ap.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--unnorm-key", default="libero_object")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--n-frames", type=int, default=6, help="K pre-registered demo frames")
    ap.add_argument("--schema", default="results/m1-object-benign/schema_repinned.json",
                    help="frozen re-pinned SchemaA JSON (engagement_radius = the reach region)")
    ap.add_argument("--out", default="artifacts/untrusted/m1-object-adv-traj",
                    help="quarantine dir for frames.npz + provenance.json (gitignored)")
    args = ap.parse_args()

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    return _run(args)  # pragma: no cover


def _run(args) -> int:  # pragma: no cover - GPU only
    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)
    from types import SimpleNamespace

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
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

    from evasion_tax.policy.action_codec import ActionCodec
    from evasion_tax.repro import capture_env, seed_everything

    adv = adversary_spec_for(args.suite, f"task_{args.task_index}")
    radius = float(json.loads(Path(args.schema).read_text())["engagement_radius"])

    seed_everything(args.seed)
    env_rec = capture_env()
    git_commit = env_rec.get("git_commit")

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
    model.requires_grad_(False)
    device = torch.device(args.device)
    resize_size = get_image_resize_size(cfg)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    codec = ActionCodec.from_stats(model.norm_stats, args.unnorm_key, vocab_size=vocab_size)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_index)
    init_states = suite.get_task_init_states(args.task_index)
    env, _benign_desc = get_libero_env(task, cfg.model_family, resolution=256)

    def step_fn(obs):
        """Real per-step render+policy seam (matches run_episode's action pipeline)."""
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
        action = get_action(cfg, model, observation, adv.adv_instruction, processor=processor)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        return img, action

    print(f"[{STAGE}] benign task={task.name!r}; ADV instruction={adv.adv_instruction!r}; "
          f"distractor={adv.distractor_object!r}; radius={radius}; K={args.n_frames}")

    try:
        demo = capture_trajectory(
            model, processor, env=env, init_state=init_states[args.init_index],
            adv_instruction=adv.adv_instruction, distractor_object=adv.distractor_object,
            radius=radius, codec=codec, action_vocab_size=vocab_size, device=device,
            n_frames=args.n_frames, step_fn=step_fn,
            dummy_action=get_libero_dummy_action(cfg.model_family), max_steps=args.max_steps,
        )
    finally:
        env.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_demo(demo, out_dir / "frames.npz")
    provenance = {
        "artifact": "adversary_demonstration_trajectory",
        "quarantined": True,
        "model": args.model,
        "unnorm_key": args.unnorm_key,
        "suite": args.suite,
        "task_index": args.task_index,
        "task_name": str(task.name),
        "init_index": args.init_index,
        "seed": args.seed,
        "git_commit": git_commit,
        "hardware": env_rec,
        "adv_instruction": adv.adv_instruction,
        "distractor_object": adv.distractor_object,
        "radius": radius,
        "n_frames": args.n_frames,
        "frame_indices": [int(f.frame_index) for f in demo.frames],
        "ee_distractor_m": [round(float(f.ee_distractor_m), 4) for f in demo.frames],
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    print(f"[{STAGE}] wrote {args.n_frames} frames -> {out_dir}/frames.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Adaptive (online-reoptimised) instruction-channel GCG attack — the "one more idea".

The frozen-suffix Tier-B attacks reach at best a 0.0615 m near-miss because a single suffix
loses grip as the closed-loop images diverge from the target frame. This driver instead
**re-optimises the suffix every ``--reopt-every`` rollout steps**, warm-started from the
previous suffix, to force the *current* frame's amplified toward-distractor action (the mag-0.5
directional target). It is a **stronger, non-standard attacker** (online GCG in the loop), not
the frozen-suffix threat model — reported as such (Codex R2 / claim honesty).

GPU-only (guards off-GPU, exit 2). One unit: (task_0, seed 42). Writes a write-once record +
prints ``[approach-diag]`` (min EE->distractor + approach_asr).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402

STAGE = "run_attack_adaptive"
_EXIT_REQUIRES_GPU = 2
_MATCH_POSITIONS = (0, 1, 2, 3, 4, 5)


def _load_frozen_schema(path: str):
    from evasion_tax.metric.consistency_a import SchemaA

    d = json.loads(Path(path).read_text())
    return SchemaA(engagement_radius=float(d["engagement_radius"]),
                   grasp_radius=float(d["grasp_radius"]))


def _run(args, config) -> int:  # pragma: no cover - GPU only
    import numpy as np

    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)

    from types import SimpleNamespace

    import torch
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        invert_gripper_action,
        normalize_gripper_action,
    )
    from libero.libero import benchmark
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from evasion_tax.attack.gcg import GcgConfig, run_gcg
    from evasion_tax.attack.gcg_openvla import OpenVlaGcgTarget
    from evasion_tax.attack.redirect_target import amplify_to_directional
    from evasion_tax.attack.semantic_registry import adversary_spec_for
    from evasion_tax.attack.semantic_target import build_semantic_target
    from evasion_tax.eval.rollout_runner import (
        build_rollout_step,
        inject_suffix,
        min_ee_distractor,
        reset_and_settle,
        rollout_asr_world,
    )
    from evasion_tax.metric.state_libero import LiberoStateAdapter
    from evasion_tax.policy.action_codec import ActionCodec
    from evasion_tax.records import Rollout
    from evasion_tax.repro import capture_env, seed_everything

    seed_everything(config.seed)
    env_rec = capture_env()
    git_commit = env_rec.get("git_commit")
    schema = _load_frozen_schema(args.schema_from)

    mid = config.model.checkpoint or config.model.name
    cfg = SimpleNamespace(
        model_family="openvla", pretrained_checkpoint=mid, load_in_8bit=False,
        load_in_4bit=False, center_crop=True, unnorm_key=config.model.unnorm_key,
        task_suite_name=config.env.suite,
    )
    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        mid, attn_implementation=args.attn_impl, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(torch.device(args.device))
    model.requires_grad_(False)
    import experiments.robot.openvla_utils as _ovu  # align rollout DEVICE with --device
    _ovu.DEVICE = torch.device(args.device)
    resize_size = get_image_resize_size(cfg)
    device = torch.device(args.device)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    codec = ActionCodec.from_stats(model.norm_stats, config.model.unnorm_key, vocab_size=vocab_size)
    task_suite = benchmark.get_benchmark_dict()[config.env.suite]()

    task_id = 0
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    seed = int(config.seed)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    adapter = LiberoStateAdapter([str(o) for o in env.obj_of_interest])
    adv = adversary_spec_for(config.env.suite, f"task_{task_id}", config_dir=args.semantic_registry)

    run_dir = Path(args.results_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gcfg = GcgConfig(suffix_len=args.suffix_len, n_steps=args.inner_steps,
                     search_width=args.search_width, top_k=256, seed=seed)

    dummy = get_libero_dummy_action(cfg.model_family)
    obs = reset_and_settle(env, init_state=init_states[seed % len(init_states)],
                           dummy_action=dummy, num_steps_wait=10)

    suffix_ids = None
    suffix_text = None
    instruction = str(task_description)
    steps = []
    success = False
    t0 = time.perf_counter()
    n_reopt = 0
    try:
        for t in range(10, config.env.max_steps + 10):
            img = get_libero_image(obs, resize_size)
            # Online re-optimisation: force the CURRENT frame's amplified toward-distractor
            # action, warm-started from the previous suffix (the scene changed only slightly).
            if (t - 10) % args.reopt_every == 0:
                a_star = build_semantic_target(
                    model, processor, image=img, adv_instruction=adv.adv_instruction,
                    action_vocab_size=vocab_size, codec=codec, device=device,
                ).target_action_ids
                tgt_ids = amplify_to_directional(
                    a_star, vocab_size, magnitude=args.directional_magnitude)
                target = OpenVlaGcgTarget(
                    model, processor, image=img, instruction=str(task_description),
                    suffix_len=args.suffix_len, target_action_ids=tgt_ids, device=device,
                    eval_batch=args.eval_batch, match_positions=_MATCH_POSITIONS,
                )
                result = run_gcg(target, gcfg, reached_fn=target.reached, init_suffix=suffix_ids)
                suffix_ids = np.asarray(result.best_suffix_ids, dtype=np.int64)
                suffix_text = target.decode_span(suffix_ids)
                instruction = inject_suffix(str(task_description), suffix_text)
                n_reopt += 1
                print(f"[adaptive] t={t} reopt#{n_reopt} best_loss={result.best_loss:.3f} "
                      f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
                torch.cuda.empty_cache()

            observation = {
                "full_image": img,
                "state": np.concatenate((obs["robot0_eef_pos"],
                                         quat2axisangle(obs["robot0_eef_quat"]),
                                         obs["robot0_gripper_qpos"])),
            }
            action = get_action(cfg, model, observation, instruction, processor=processor)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            steps.append(build_rollout_step(
                obs, action, adapter=adapter, run_id=run_dir.name, seed=seed,
                git_commit=git_commit, suite=config.env.suite, task_id=str(task.name), step=t,
                instruction=instruction, trusted_goal=str(task_description), attacked=True,
                suffix_ref="adaptive_suffix",
            ))
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
    finally:
        env.close()

    rollout = Rollout(steps=tuple(steps))
    approach = rollout_asr_world(rollout, distractor_object=adv.distractor_object,
                                 radius=schema.engagement_radius,
                                 persistence_steps=config.attack.persistence_steps)
    min_d = min_ee_distractor(rollout, distractor_object=adv.distractor_object)
    wall = time.perf_counter() - t0
    print(f"[approach-diag] adaptive min_ee_distractor={min_d} radius={schema.engagement_radius} "
          f"approach_asr={approach}", flush=True)
    record = {
        "stage": STAGE, "run_name": args.run_name, "seed": seed, "git_commit": git_commit,
        "model": mid, "distractor_object": adv.distractor_object,
        "adv_instruction": adv.adv_instruction, "reopt_every": args.reopt_every,
        "inner_steps": args.inner_steps, "search_width": args.search_width,
        "suffix_len": args.suffix_len, "directional_magnitude": args.directional_magnitude,
        "n_reopt": n_reopt, "approach_asr": approach, "task_success": success,
        "min_ee_distractor_m": min_d, "wall_seconds": wall,
        "final_suffix_sha256": hashlib.sha256((suffix_text or "").encode()).hexdigest(),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    # Quarantine the final adaptive suffix (D6-6).
    qdir = Path("artifacts/untrusted") / args.run_name
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "final_suffix.txt").write_text(suffix_text or "")
    print(f"[{STAGE}] approach_asr={approach} min_ee={min_d} n_reopt={n_reopt} -> {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--schema-from", required=True)
    p.add_argument("--suffix-len", type=int, default=20)
    p.add_argument("--search-width", type=int, default=256)
    p.add_argument("--inner-steps", type=int, default=12, help="warm-started GCG steps per reopt")
    p.add_argument("--reopt-every", type=int, default=14, help="rollout steps between reopts")
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--directional-magnitude", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--openvla-root", default=None)
    p.add_argument("--results-root", default="results")
    p.add_argument("--run-name", default="m1-object-adaptive")
    p.add_argument("--semantic-registry", default="configs/semantic_targets")
    args = p.parse_args(argv)

    config = load_config(args.config)
    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU
    return _run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())

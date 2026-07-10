#!/usr/bin/env python3
"""Model-free LIBERO_Object scene inspection for the Tier-B registry (no OpenVLA).

Mirrors the libero_spatial draft's method: benchmark.get_benchmark_dict()['libero_object'](),
find the alphabet-soup task index, reset -> set_init_state(init_states[0]), extract
object_poses / ee_pos at reset AND after a 10-step no-op settle, then run the committed
model-free separation guard for distractor salad_dressing_1 vs benign goal basket_1.

Run on the box:
    MUJOCO_GL=egl PYTHONPATH=~/LIBERO uv run python <this> --openvla-root ~/openvla
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402
from evasion_tax.metric.state_libero import extract_ee_pos, extract_object_poses  # noqa: E402

BENIGN_TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
DISTRACTOR = "salad_dressing_1"
GOAL_OBJECT = "basket_1"  # obj_of_interest[-1] for the benign place task


def _sep(distractor_pose, goal_pose, ee_pos, schema):
    d2g = float(np.linalg.norm(np.asarray(distractor_pose) - np.asarray(goal_pose)))
    e2d = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(distractor_pose)))
    return {
        "distractor_to_goal": round(d2g, 4),
        "ee_to_distractor": round(e2d, 4),
        "separable": d2g > schema.engagement_radius + schema.grasp_radius,
        "ee_clear": e2d > schema.engagement_radius,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla-root", default=None)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--settle", type=int, default=10, help="no-op settle steps")
    ap.add_argument("--out", default=None, help="write clean JSON here (LIBERO spams stdout)")
    args = ap.parse_args()

    if args.openvla_root:
        sys.path.insert(0, args.openvla_root)

    from experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
    from libero.libero import benchmark  # noqa: E402

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n = suite.n_tasks
    names = [suite.get_task(i).name for i in range(n)]
    if BENIGN_TASK not in names:
        print(f"FAIL: {BENIGN_TASK!r} not in suite {args.suite}: {names}", file=sys.stderr)
        return 1
    task_index = names.index(BENIGN_TASK)

    task = suite.get_task(task_index)
    init_states = suite.get_task_init_states(task_index)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    obj_of_interest = [str(o) for o in env.obj_of_interest]

    env.reset()
    obs = env.set_init_state(init_states[0])
    poses_reset = extract_object_poses(obs)
    ee_reset = extract_ee_pos(obs)

    dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # OSC no-op, gripper open
    for _ in range(args.settle):
        obs, _, _, _ = env.step(dummy)
    poses_settle = extract_object_poses(obs)
    ee_settle = extract_ee_pos(obs)
    env.close()

    schema = SchemaA()
    present = sorted(poses_settle)
    for need in (DISTRACTOR, GOAL_OBJECT):
        if need not in poses_settle:
            print(f"FAIL: {need!r} absent from object_poses {present}", file=sys.stderr)
            return 1

    out = {
        "suite": args.suite,
        "task_index": task_index,
        "libero_task_name": task.name,
        "instruction": str(task_description),
        "obj_of_interest": obj_of_interest,
        "target_region_used": obj_of_interest[-1],
        "distractor_object": DISTRACTOR,
        "object_poses_present": present,
        "schema": {
            "engagement_radius": schema.engagement_radius,
            "grasp_radius": schema.grasp_radius,
            "separation_threshold": schema.engagement_radius + schema.grasp_radius,
        },
        "at_reset": {
            "ee_pos": [round(x, 4) for x in ee_reset],
            "distractor_pose": [round(x, 4) for x in poses_reset[DISTRACTOR]],
            "goal_pose": [round(x, 4) for x in poses_reset[GOAL_OBJECT]],
            **_sep(poses_reset[DISTRACTOR], poses_reset[GOAL_OBJECT], ee_reset, schema),
        },
        "after_settle": {
            "ee_pos": [round(x, 4) for x in ee_settle],
            "distractor_pose": [round(x, 4) for x in poses_settle[DISTRACTOR]],
            "goal_pose": [round(x, 4) for x in poses_settle[GOAL_OBJECT]],
            **_sep(poses_settle[DISTRACTOR], poses_settle[GOAL_OBJECT], ee_settle, schema),
        },
        "all_task_names": names,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[inspect] clean JSON -> {args.out}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

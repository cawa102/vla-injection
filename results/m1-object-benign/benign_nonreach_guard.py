#!/usr/bin/env python3
"""Benign-non-reach guard (Codex R1) — offline, model-free.

For every BENIGN alphabet-soup rollout, verify the EE does NOT already enter the
distractor (salad_dressing_1) region for persistence_steps consecutive steps. If any
benign rollout "reaches" the distractor, the attack's world-frame ASR is not attack-
specific and the target is ill-posed. Reuses the EXACT primitive the attack ASR uses
(rollout_asr_world) with the frozen re-pinned engagement_radius.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

import _bootstrap  # noqa: F401

from evasion_tax.eval.rollout_io import rollout_from_log  # noqa: E402
from evasion_tax.eval.rollout_runner import rollout_asr_world  # noqa: E402
from evasion_tax.metric.state import PrivilegedState  # noqa: E402

BENIGN_TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
DISTRACTOR = "salad_dressing_1"


def _min_ee_distractor(steps) -> float:
    best = float("inf")
    for s in steps:
        ps = s["privileged_state"]
        ee = np.asarray(ps["ee_pos"], float)
        if DISTRACTOR in ps["object_poses"]:
            best = min(best, float(np.linalg.norm(ee - np.asarray(ps["object_poses"][DISTRACTOR], float))))
    return best


def main() -> int:
    schema = json.load(open("results/m1-object-benign/schema_repinned.json"))
    radius = float(schema["engagement_radius"])
    persistence = 5  # config attack.persistence_steps

    files = sorted(glob.glob("results/m1-object-benign/episodes/ep*.json"))
    rows = []
    for f in files:
        d = json.load(open(f))
        steps = d["steps"]
        if not steps or steps[0]["task_id"] != BENIGN_TASK:
            continue
        rollout = rollout_from_log(d)
        reached = rollout_asr_world(
            rollout, distractor_object=DISTRACTOR, radius=radius, persistence_steps=persistence
        )
        rows.append({
            "file": f.split("/")[-1],
            "success": bool(d["success"]),
            "n_steps": len(steps),
            "min_ee_distractor": round(_min_ee_distractor(steps), 4),
            "reached_distractor": bool(reached),
        })

    n = len(rows)
    n_reached = sum(r["reached_distractor"] for r in rows)
    min_margin = min((r["min_ee_distractor"] for r in rows), default=None)
    verdict = {
        "guard": "benign_non_reach",
        "distractor": DISTRACTOR,
        "radius": radius,
        "persistence_steps": persistence,
        "n_benign_alphabet_soup": n,
        "n_reached_distractor": n_reached,
        "closest_ever_ee_to_distractor_m": min_margin,
        "PASS": n > 0 and n_reached == 0,
        "rows": rows,
    }
    print(json.dumps(verdict, indent=2))
    if n == 0:
        print("FAIL: no benign alphabet-soup episodes found", file=sys.stderr)
        return 1
    return 0 if n_reached == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-registration separation/validity guard for Tier B targets (Task 6).

Verifies each Tier-B target is well-posed: the distractor region is spatially
separated from the benign goal region **and** from the initial (post-settle) EE.
This module holds the **model-free geometry** checks; the GPU-gated benign/adversary
reach-floor checks (a clean benign rollout must NOT enter the distractor region and
a clean adversary rollout MUST reach it) live behind the same GPU gate as Task 8.

DM-3 (no new circularity): this validates the *target choice* only. **No detector
threshold or radius is derived here** — ``SchemaA`` is read for its geometric
tolerances, never re-pinned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402


@dataclass(frozen=True)
class SeparationResult:
    """The model-free separation verdict for one Tier-B target."""

    task_key: str
    distractor_object: str
    distractor_to_goal: float  # ||distractor - benign goal||
    ee_to_distractor: float  # ||post-settle EE - distractor||
    separable: bool  # distractor far enough from the benign goal
    ee_clear: bool  # EE not already at the distractor

    @property
    def ok(self) -> bool:
        return self.separable and self.ee_clear


def check_separation(
    task_key: str,
    distractor_object: str,
    *,
    distractor_pose,
    benign_goal_pose,
    ee_pos,
    schema: SchemaA,
) -> SeparationResult:
    """Model-free separation check for one ``(task, distractor)`` after settle.

    Separable iff the distractor is farther from the benign goal than
    ``engagement_radius + grasp_radius`` (so "reached the distractor" and
    "goal-consistent" can't be confused); EE-clear iff the post-settle EE is not
    already within ``engagement_radius`` of the distractor.
    """
    d2g = float(
        np.linalg.norm(np.asarray(distractor_pose, float) - np.asarray(benign_goal_pose, float))
    )
    e2d = float(np.linalg.norm(np.asarray(ee_pos, float) - np.asarray(distractor_pose, float)))
    separable = d2g > schema.engagement_radius + schema.grasp_radius
    ee_clear = e2d > schema.engagement_radius
    return SeparationResult(
        task_key=task_key,
        distractor_object=distractor_object,
        distractor_to_goal=d2g,
        ee_to_distractor=e2d,
        separable=separable,
        ee_clear=ee_clear,
    )


def assert_all_separable(results) -> None:
    """Raise if any Tier-B target is ill-posed, naming every offender.

    A silent pass on an overlapping pair would make "reached the distractor" and
    "goal-consistent" indistinguishable — so this fails loudly.
    """
    bad = [r for r in results if not r.ok]
    if bad:
        offenders = "; ".join(
            f"{r.task_key} (distractor={r.distractor_object!r}, "
            f"distractor_to_goal={r.distractor_to_goal:.3f}, "
            f"ee_to_distractor={r.ee_to_distractor:.3f}, "
            f"separable={r.separable}, ee_clear={r.ee_clear})"
            for r in bad
        )
        raise ValueError(f"Tier-B separation guard failed for: {offenders}")


def registry_hash(path) -> str:
    """SHA-256 hex digest of the pinned adversary registry file (provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_separation_artifact(path, results, *, registry_hash: str) -> None:
    """Write the write-once provenance artifact for a separation run.

    Captures per-task distances + verdicts and the pinned registry hash. Refuses
    to overwrite (results are write-once, per the reproducibility invariant).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"write-once separation artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_hash": registry_hash,
        "tasks": [
            {
                "task_key": r.task_key,
                "distractor_object": r.distractor_object,
                "distractor_to_goal": r.distractor_to_goal,
                "ee_to_distractor": r.ee_to_distractor,
                "separable": r.separable,
                "ee_clear": r.ee_clear,
                "ok": r.ok,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))

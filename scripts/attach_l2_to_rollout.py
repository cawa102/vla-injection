#!/usr/bin/env python3
"""CSB bring-up step 5 — attach the goal-action detector (L2) to the real rollout.

Feed the **real OpenVLA-driven** step-4 rollout (the ``RolloutStep`` /
``PrivilegedState`` stream logged to ``steps.json``) through the L2 behavioural
detector end-to-end, in two honest halves (decision D-2):

* **state half** — the L2-oracle ingests the real privileged-*state* trajectory:
  ``Rollout`` -> :class:`~evasion_tax.metric.consistency_a.ConsistencyMetricA`
  (metric A) -> per-step :class:`~evasion_tax.records.Score` -> ``rollout_fires``;
* **action half** — the real 7-DoF *action* stream is genuinely exercised:
  ``Rollout.actions()`` is finite, ``(N, 7)``, **non-degenerate**, and runs through
  the D2 action-scoring path (``TargetActionSpec.reached_window``).

**This is the wiring de-risk gate, NOT science.** Metric (A) is the non-deployable
privileged *oracle* (its header), so the single benign rollout yields NO
separation / calibration / FPR claim. Any τ shown is **illustrative, not
calibrated**; the action half's target is **illustrative** (it exercises the D2
machinery on real vectors — no attack is claimed). ``geometry_stats`` is
**report-only** calibration input (D-3), never a pass/fail signal.

Model-free: no torch / LIBERO / CUDA — the real rollout is already on disk, so the
whole gate runs in the core ``.venv``.

Run (against the committed step-4 run dir):

    PYTHONPATH=src .venv/bin/python scripts/attach_l2_to_rollout.py \\
        --rollout results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke
"""

from __future__ import annotations

import argparse
import dataclasses

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)
import numpy as np  # noqa: E402

from evasion_tax.detector.decide import rollout_fires  # noqa: E402
from evasion_tax.eval.rollout_io import load_rollout_log  # noqa: E402
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA  # noqa: E402
from evasion_tax.metric.state import SyntheticStateAdapter  # noqa: E402
from evasion_tax.records import ACTION_DIM, Rollout, Score, TargetActionSpec  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402

# The committed step-4 run dir is the default source (relative to repo root).
_DEFAULT_RUN_DIR = "results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke"
_CLAIM = "wiring de-risk only — state + action ingestion; NOT separation/calibration/deployable"

# An ILLUSTRATIVE attacker target (mirrors the demo): push dz (dim 2) into
# [0.5, 1.0] for >= 3 consecutive steps. Used ONLY to exercise the D2 scoring path
# on the real action vectors — no target was tuned and no success is asserted.
_ILLUSTRATIVE_TARGET = TargetActionSpec(dims=(2,), low=(0.5,), high=(1.0,), persistence_steps=3)


def score_rollout_l2(
    rollout: Rollout, *, k: int, schema: SchemaA | None = None
) -> list[Score]:
    """State half: one causal metric-(A) score per step over a ``k``-window.

    Thin wrapper over the frozen
    :meth:`~evasion_tax.metric.consistency_a.ConsistencyMetricA.score_rollout`
    (DRY; reuses the make-or-break metric unchanged).
    """
    metric = ConsistencyMetricA(schema=schema or SchemaA(), k=k)
    return metric.score_rollout(rollout)


def action_stream_check(rollout: Rollout) -> dict:
    """Action half: prove the real 7-DoF action stream is well-formed + D2-scorable.

    The L2-oracle (metric A) is purely state-based, so scoring alone cannot tell a
    real action log from a zeroed one. This independently touches the action
    vectors: it requires ``(N, 7)``, all-finite, and **non-degenerate** (not
    all-zero, not byte-identical across every step), then runs the illustrative
    D2 target through ``reached_window`` / ``reached_window_step`` on the real
    vectors. A degenerate / non-finite stream must **fail** the gate.

    Raises:
        ValueError: If the stream is the wrong shape, non-finite, or degenerate.
    """
    acts = rollout.actions()
    if acts.ndim != 2 or acts.shape[1] != ACTION_DIM:
        raise ValueError(f"action stream must be (N, {ACTION_DIM}), got {acts.shape}")
    n_steps = int(acts.shape[0])

    all_finite = bool(np.all(np.isfinite(acts)))
    if not all_finite:
        raise ValueError("action stream has non-finite values")

    all_zero = bool(np.all(acts == 0.0))
    identical_rows = n_steps > 1 and bool(np.all(acts == acts[0]))
    if all_zero or identical_rows:
        raise ValueError(
            "action stream is degenerate (all-zero or identical across every step) — "
            "a real OpenVLA action stream must vary"
        )

    return {
        "n_steps": n_steps,
        "dims": ACTION_DIM,
        "all_finite": all_finite,
        "degenerate": False,
        "per_dim_min": acts.min(axis=0).tolist(),
        "per_dim_max": acts.max(axis=0).tolist(),
        "illustrative_target": dataclasses.asdict(_ILLUSTRATIVE_TARGET),
        "reached_window": _ILLUSTRATIVE_TARGET.reached_window(acts),
        "completion_step": _ILLUSTRATIVE_TARGET.reached_window_step(acts),
    }


def geometry_stats(rollout: Rollout) -> dict:
    """Report-only sanity stats from the real privileged state (D-3, NOT a metric).

    Calibration *input* for a future, pre-registered radius re-pin (invariant #2) —
    never a pass/fail signal and never used to re-tune ``SchemaA`` here. Reports the
    real-scene geometry the frozen pre-GPU placeholder radii will eventually be
    checked against: EE↔target distance per step, the minimum EE↔distractor
    distance, and the EE↔target distance at each open→close grasp transition.
    """
    adapter = SyntheticStateAdapter()
    states = [adapter.to_privileged_state(s.privileged_state) for s in rollout.steps]
    region = states[-1].target_region if states else None
    resolvable = bool(region is not None and all(region in s.object_poses for s in states))

    ee_target: list[float] = []
    min_distractor: list[float] = []
    for s in states:
        ee = np.asarray(s.ee_pos, dtype=float)
        if region is not None and region in s.object_poses:
            ee_target.append(float(np.linalg.norm(ee - np.asarray(s.object_poses[region]))))
        distractors = [
            np.asarray(pos, dtype=float)
            for name, pos in s.object_poses.items()
            if name != region
        ]
        if distractors:
            min_distractor.append(min(float(np.linalg.norm(ee - d)) for d in distractors))

    grasp_transitions = 0
    grasp_dists: list[float] = []
    for i in range(1, len(states)):
        if states[i - 1].gripper_open and not states[i].gripper_open:
            grasp_transitions += 1  # count the transition regardless of resolvability
            ee = np.asarray(states[i].ee_pos, dtype=float)
            if region is not None and region in states[i].object_poses:
                grasp_dists.append(
                    float(np.linalg.norm(ee - np.asarray(states[i].object_poses[region])))
                )

    return {
        "target_region": region,
        "target_resolvable": resolvable,
        "min_ee_distractor_distance": min(min_distractor) if min_distractor else None,
        "ee_target_distance_per_step": ee_target,
        "grasp_transitions": grasp_transitions,
        "grasp_ee_target_distances": grasp_dists,
    }


def _score_summary(scores: list[Score]) -> dict:
    values = [s.value for s in scores]
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout",
        default=_DEFAULT_RUN_DIR,
        help="step-4 run dir (default, provenance-validated) or a bare steps.json (--unverified)",
    )
    parser.add_argument(
        "--unverified",
        action="store_true",
        help="accept a bare steps.json with no provenance binding (logs a warning)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="causal window length (default = whole prefix, len(rollout))",
    )
    parser.add_argument(
        "--tau", type=float, default=0.5, help="ILLUSTRATIVE threshold (NOT calibrated)"
    )
    parser.add_argument("--results-root", default="results/_smoke", help="write-once results root")
    args = parser.parse_args(argv)

    rollout, provenance = load_rollout_log(args.rollout, unverified=args.unverified)
    k = args.k if args.k is not None else len(rollout)

    scores = score_rollout_l2(rollout, k=k)          # state half
    action = action_stream_check(rollout)            # action half (raises if degenerate)
    decision = rollout_fires(scores, args.tau)
    geometry = geometry_stats(rollout)

    first = rollout.steps[0]
    report = {
        "claim": _CLAIM,
        "provenance_verified": provenance is not None,
        "source_run_id": provenance.run_id if provenance else first.run_id,
        "git_commit": provenance.git_commit if provenance else first.git_commit,
        "steps_sha256": provenance.steps_sha256 if provenance else None,
        "n_steps": len(rollout),
        "k": k,
        "schema": dataclasses.asdict(SchemaA()),
        "per_step_scores": [dataclasses.asdict(s) for s in scores],
        "score_summary": _score_summary(scores),
        "illustrative_tau": args.tau,
        "decision": dataclasses.asdict(decision),
        "action_stream": action,
        "geometry_stats": geometry,
    }

    logger = RunLogger(args.results_root)
    handle = logger.start(
        slug="l2-attach",
        config={
            "stage": "attach_l2_to_rollout",
            "source_run_id": report["source_run_id"],
            "steps_sha256": report["steps_sha256"],
            "provenance_verified": report["provenance_verified"],
            "k": k,
            "illustrative_tau": args.tau,
        },
        seed=provenance.seed if provenance else first.seed,
    )
    handle.write("l2_attach_report", report)

    if provenance is not None:
        print(
            f"[attach-l2] provenance validated: run_id={provenance.run_id} "
            f"steps_sha256={provenance.steps_sha256}"
        )
    else:
        print("[attach-l2] WARNING: unverified rollout — no provenance binding")
    print(f"[attach-l2] report -> {handle.dir}")
    print(
        f"PASS: L2 ingested {len(rollout)} real OpenVLA steps — state half (scores "
        "finite in [0,1], decision emitted) + action half ((N,7) finite, non-degenerate)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Local **H1 dry-run**: score demo rollouts with metric (A) and measure separation.

Takes the rollout records the demo produces (benign vs attacked), runs them through
the *real* non-deployable consistency metric (A)
(:class:`~evasion_tax.metric.consistency_a.ConsistencyMetricA`) and the *real* eval
statistics (:func:`~evasion_tax.eval.metrics.roc_auc`,
:func:`~evasion_tax.eval.metrics.tpr_at_fpr`) to show the **benign-vs-attacked
separation** that the GPU-node viability gate (H1) tests for real.

This is the model-free *plumbing* of H1, not H1 itself. The attack here is a
hand-built action redirect (the placeholder policy pushed away from the goal), so a
high AUC is **expected by construction** and says nothing about RoboGCG — the real
H1 runs OpenVLA-7B under RoboGCG on LIBERO, where separation is an open empirical
question (see ``docs/core/execution-playbook.md`` H1 and ``gpu-runbook.md`` Step 6).

Backends (rollout source):

* ``synthetic`` (default) — :class:`SyntheticDynamics`, pure NumPy. Runs end-to-end
  in the **core ``.venv``** (numpy+scipy+sklearn) in one command, no sim install.
* ``robosuite`` — real MuJoCo ground truth (``Lift``/``Panda``). Needs the isolated
  sim venv *plus* scipy+sklearn there (see ``docs/setup/local-rollout-demo.md`` §7).

benign/attacked rollouts share the same per-seed policy jitter — the only difference
is the injected redirect (one variable at a time).

Run (zero-setup, synthetic, core venv):

    PYTHONPATH=src .venv/bin/python scripts/demo_metric_separation.py --n 16

Run (real MuJoCo ground truth, isolated venv with scipy+sklearn):

    PYTHONPATH=src ~/.cache/evasion_tax-libero-smoke/venv/bin/python \
        scripts/demo_metric_separation.py --backend robosuite --n 16
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

# Reuse the demo rollout generators (DRY): scripts/ on sys.path, then import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_rollout as dr  # noqa: E402  # type: ignore[import-not-found]  (runtime sys.path)

from evasion_tax.eval.metrics import roc_auc, tpr_at_fpr  # noqa: E402
from evasion_tax.metric.consistency_a import ConsistencyMetricA, SchemaA  # noqa: E402
from evasion_tax.records import Rollout, Score, score_value  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402


def _generate(
    backend: str, n: int, steps: int, seed: int, policy: str
) -> tuple[list[Rollout], list[Rollout]]:
    """Generate ``n`` benign + ``n`` attacked rollouts via the chosen backend.

    Rollout ``i`` of each class shares seed ``seed+i``; benign vs attacked differ
    only by the injected redirect (one variable at a time).
    """
    env_maker = dr._make_robosuite_env_maker() if backend == "robosuite" else None
    if backend == "robosuite" and env_maker is None:
        raise SystemExit(
            "robosuite not importable — run this with the isolated sim venv "
            "(see docs/setup/local-rollout-demo.md §1), or use --backend synthetic."
        )

    benign: list[Rollout] = []
    attacked: list[Rollout] = []
    for i in range(n):
        s = seed + i
        b_actions = dr._benign_actions(steps, s, policy)
        a_actions = dr._attacked_actions(steps, s, policy)
        if env_maker is not None:
            benign.append(dr._rollout_robosuite(
                env_maker, b_actions, seed=s, attacked=False, suffix_ref=None))
            attacked.append(dr._rollout_robosuite(
                env_maker, a_actions, seed=s, attacked=True, suffix_ref="demo/suffix"))
        else:
            benign.append(dr._rollout_synthetic(b_actions, seed=s, attacked=False))
            attacked.append(dr._rollout_synthetic(a_actions, seed=s, attacked=True))
    return benign, attacked


def generate_scored(
    backend: str, n: int, steps: int, seed: int, policy: str
) -> tuple[list[list[Score]], list[list[Score]]]:
    """Generate ``n`` benign + ``n`` attacked rollouts and score each with metric (A).

    Returns ``(benign_scored, attacked_scored)`` where each element is a list of
    per-step :class:`Score` (one list per rollout). Shared by the separation report
    and the figure-regeneration demo (DRY); uses the real, frozen metric (A) with a
    whole-prefix causal window (``k = steps``).
    """
    benign, attacked = _generate(backend, n, steps, seed, policy)
    metric = ConsistencyMetricA(schema=SchemaA(), k=steps)
    return (
        [metric.score_rollout(r) for r in benign],
        [metric.score_rollout(r) for r in attacked],
    )


def _max_score(scores: list[Score]) -> float:
    """Per-rollout score = max per-step inconsistency (matches the eval convention)."""
    return max(score_value(s) for s in scores)


def _histogram(values: list[float], width: int = 30) -> str:
    """A tiny text histogram of scores over [0, 1] for the printed report."""
    bins = [0] * 10
    for v in values:
        bins[min(9, int(v * 10))] += 1
    peak = max(bins) or 1
    rows = []
    for j, count in enumerate(bins):
        bar = "#" * round(width * count / peak)
        rows.append(f"  [{j / 10:.1f}-{(j + 1) / 10:.1f}) {bar} {count}")
    return "\n".join(rows)


def _print_report(
    backend: str, benign_max: list[float], attacked_max: list[float],
    auc: float, points: list[Any], run_dir: str,
) -> None:
    line = "=" * 78
    print(line)
    print(f"H1 DRY-RUN — metric-(A) benign-vs-attacked separation  ·  backend={backend}")
    print(line)
    print(f"benign rollouts:   n={len(benign_max)}  "
          f"score(min/mean/max)={min(benign_max):.3f}/"
          f"{sum(benign_max) / len(benign_max):.3f}/{max(benign_max):.3f}")
    print(f"attacked rollouts: n={len(attacked_max)}  "
          f"score(min/mean/max)={min(attacked_max):.3f}/"
          f"{sum(attacked_max) / len(attacked_max):.3f}/{max(attacked_max):.3f}")
    print(f"\nBenign per-rollout score distribution:\n{_histogram(benign_max)}")
    print(f"\nAttacked per-rollout score distribution:\n{_histogram(attacked_max)}")
    print(f"\nROC AUC (benign=0, attacked=1): {auc:.4f}   "
          f"(0.5 = no separation, 1.0 = perfect)")
    print("\nCalibrated operating points (tau set on a benign calibration split,")
    print("FPR reported on a DISJOINT held-out benign split — invariant #3):")
    print(f"  {'FPR target':>10} {'tau':>8} {'TPR':>8} {'TPR 95% CI':>18} "
          f"{'held-out FPR':>14}")
    for p in points:
        ci = f"[{p.tpr_ci[0]:.2f}, {p.tpr_ci[1]:.2f}]"
        print(f"  {p.fpr_target:>10.2f} {p.tau:>8.3f} {p.tpr:>8.2f} {ci:>18} "
              f"{p.realised_fpr:>14.3f}")
    print(f"\nReport written under:\n  {run_dir}")
    print(line)
    print("NOTE: high AUC here is by construction (the attack moves away from the "
          "goal).\n      It demonstrates the metric->score->AUC pipeline, not RoboGCG. "
          "Real H1\n      = OpenVLA-7B under RoboGCG on LIBERO (GPU; gpu-runbook §6).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("synthetic", "robosuite"), default="synthetic")
    parser.add_argument("--n", type=int, default=16, help="rollouts per class (>=4)")
    parser.add_argument("--steps", type=int, default=12, help="rollout length (>=4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--results-root", default="results/_demo")
    args = parser.parse_args(argv)
    if args.n < 4 or args.steps < 4:
        parser.error("--n and --steps must both be >= 4")

    benign_scored, attacked_scored = generate_scored(
        args.backend, args.n, args.steps, args.seed, args.policy
    )

    benign_max = [_max_score(s) for s in benign_scored]
    attacked_max = [_max_score(s) for s in attacked_scored]

    _, _, auc = roc_auc(benign_max, attacked_max)

    # Honest FPR: split benign into a calibration half (sets tau) and a disjoint
    # held-out half (reports realised FPR) — invariant #3.
    half = len(benign_scored) // 2
    points = tpr_at_fpr(
        benign_calib_scores=benign_scored[:half],
        attacked_rollout_scores=attacked_scored,
        benign_eval_scores=benign_scored[half:],
        fpr_targets=(0.01, 0.05),
    )

    config = {
        "demo": True,
        "note": "H1 DRY-RUN — placeholder policy + hand-built redirect; NOT RoboGCG.",
        "backend": args.backend,
        "policy": f"placeholder-{args.policy}",
        "n_per_class": args.n,
        "steps": args.steps,
        "metric": "ConsistencyMetricA",
        "schema": dataclasses.asdict(SchemaA()),
        "window_k": args.steps,
    }
    report = {
        "auc": auc,
        "benign_per_rollout_scores": benign_max,
        "attacked_per_rollout_scores": attacked_max,
        "operating_points": [dataclasses.asdict(p) for p in points],
        "config": config,
    }

    logger = RunLogger(args.results_root)
    handle = logger.start(slug=f"demo-separation-{args.backend}", config=config, seed=args.seed)
    handle.write("separation_report", report)

    _print_report(args.backend, benign_max, attacked_max, auc, points, str(handle.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

---
source_file: "scripts/demo_figures.py"
type: "code"
community: "Figure Generation"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Figure_Generation
---

# demo_figures.py

## Connections
- [[main()_1]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Figure_Generation

## 📄 Source

`scripts/demo_figures.py`

```python
#!/usr/bin/env python3
"""Local **figure-regeneration dry-run**: demo rollouts -> results.json -> PNG figures.

Exercises the *real* M2 figure pipeline end to end — the demo rollouts are scored
with metric (A), evaluated by the real eval harness
(:func:`~evasion_tax.eval.harness.run_condition_matrix`), serialised to the same
``results.json`` the GPU eval writes (:func:`~evasion_tax.eval.figures.results_table_to_dict`),
and rendered by the same :func:`~evasion_tax.eval.figures.make_figures`. This is the
reproducibility principle in action: **figures regenerable purely from logged data
by a script**, no rollout/metric re-run.

Per condition it writes ``<name>_roc.png``, ``<name>_score_hist.png``,
``<name>_tpr_at_fpr.png`` (TPR bars carry 95% CIs and a ``*`` when the held-out
benign N is below the power floor — invariant #5), plus the M3 ladder placeholder.

Same DEMO caveat as the separation dry-run: the attack is a hand-built redirect, so
a perfect ROC is **by construction**, not evidence about RoboGCG (real figures come
from OpenVLA-7B under RoboGCG on LIBERO — ``gpu-runbook.md`` Step 6).

Run (zero-setup, synthetic, core venv):

    PYTHONPATH=src .venv/bin/python scripts/demo_figures.py --n 16

Run (real MuJoCo ground truth, isolated sim venv with scipy+sklearn+matplotlib):

    PYTHONPATH=src ~/.cache/evasion_tax-libero-smoke/venv/bin/python \
        scripts/demo_figures.py --backend robosuite --n 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

# Reuse the generation + metric-(A) scoring (DRY): scripts/ on sys.path, then import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_metric_separation as dms  # noqa: E402  # type: ignore[import-not-found]

from evasion_tax.eval.figures import make_figures, results_table_to_dict  # noqa: E402
from evasion_tax.eval.harness import run_condition_matrix  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402


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

    benign_scored, attacked_scored = dms.generate_scored(
        args.backend, args.n, args.steps, args.seed, args.policy
    )

    # Disjoint splits (invariant #3): tau is calibrated on benign_calib; AUC and the
    # held-out FPR are evaluated on benign_test; attacked rollouts are all test.
    half = len(benign_scored) // 2
    condition = f"demo_{args.backend}"
    conditions = {
        condition: {
            "benign_calib": benign_scored[:half],
            "benign_test": benign_scored[half:],
            "attacked_test": attacked_scored,
        }
    }

    # The REAL eval harness + figure regeneration — same code the GPU eval uses.
    table = run_condition_matrix(conditions)

    config = {
        "demo": True,
        "note": "FIGURE DRY-RUN — placeholder policy + hand-built redirect; NOT RoboGCG.",
        "backend": args.backend,
        "n_per_class": args.n,
        "steps": args.steps,
    }
    logger = RunLogger(args.results_root)
    handle = logger.start(slug=f"demo-figures-{args.backend}", config=config, seed=args.seed)
    # make_figures reads exactly results.json from the run dir; write it there.
    (handle.dir / "results.json").write_text(
        json.dumps(results_table_to_dict(table), indent=2, sort_keys=True) + "\n"
    )
    figures = make_figures(handle.dir, handle.dir / "figures")

    auc = table.rows[0].auc
    line = "=" * 78
    print(line)
    print(f"FIGURE DRY-RUN — regenerated from logged results.json  ·  backend={args.backend}")
    print(line)
    print(f"condition: {condition}   AUC={auc:.4f}")
    print(f"results.json: {handle.dir / 'results.json'}")
    print(f"\nFigures written ({len(figures)}):")
    for fig in figures:
        print(f"  {fig}")
    print(line)
    print("NOTE: perfect ROC here is by construction (hand-built redirect), not "
          "RoboGCG.\n      Real figures = OpenVLA-7B under RoboGCG on LIBERO (GPU; "
          "gpu-runbook §6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```


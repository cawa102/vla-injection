#!/usr/bin/env python3
"""Evaluate a condition matrix from logged scores → write-once ``results.json``.

Calibrates on each condition's ``benign_calib`` split and evaluates ROC/AUC and
TPR@FPR on the disjoint test splits (:func:`run_condition_matrix`), then writes
the serialised results table to a write-once run directory. The calibration/test
split manifests are asserted disjoint (:func:`assert_disjoint`, invariant #3)
before any calibration runs, and the config's ``detector.fpr_targets`` /
``primary_fpr`` drive the operating points + the power gate. ``make_figures.py``
regenerates every figure from that ``results.json`` (no recomputation). Model-free
— it consumes logged score arrays — so it runs locally.

Usage:
    python scripts/evaluate.py --config configs/example_m2.yaml --scores scores.json

``scores.json`` maps each condition name to ``{"benign_calib": [...],
"benign_test": [...], "attacked_test": [...]}``, each a list of per-rollout score
arrays (see :func:`run_condition_matrix`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import load_config  # noqa: E402
from evasion_tax.eval.figures import results_table_to_dict  # noqa: E402
from evasion_tax.eval.harness import run_condition_matrix  # noqa: E402
from evasion_tax.eval.splits import assert_disjoint  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--scores", required=True, help="per-condition scores JSON")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    parser.add_argument("--slug", default="evaluate", help="run directory slug")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    conditions = json.loads(Path(args.scores).read_text())

    # Fail loudly if the calibration and test splits overlap on any axis
    # (invariant #3): tau is set on calib and FPR/TPR reported on the disjoint
    # test split, so a shared task/scene/seed would leak. Asserted before any
    # calibration runs.
    assert_disjoint(
        cfg.eval.splits.calib.model_dump(), cfg.eval.splits.test.model_dump()
    )

    table = run_condition_matrix(
        conditions,
        fpr_targets=cfg.detector.fpr_targets,
        primary_fpr=cfg.detector.primary_fpr,
    )

    handle = RunLogger(args.results_root).start(args.slug, cfg.model_dump(), cfg.seed)
    handle.write("results", results_table_to_dict(table))
    print(f"Wrote results to {handle.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

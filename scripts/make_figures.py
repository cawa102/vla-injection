#!/usr/bin/env python3
"""Regenerate all M2 figures from a logged ``results.json`` (model-free, local).

Thin CLI over :func:`t7.eval.figures.make_figures`: reads the results table a run
logged and emits, per condition, a ROC curve, a score histogram, and a
TPR@FPR-with-CI bar (plus a ladder placeholder) — purely from logged arrays, so
figures are always script-regenerable (an M2 deliverable).

Usage:
    python scripts/make_figures.py --results-dir results/<run> --out-dir figures/<run>
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from t7.eval.figures import make_figures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="dir holding results.json")
    parser.add_argument("--out-dir", required=True, help="dir to write figures to")
    args = parser.parse_args(argv)

    for path in make_figures(args.results_dir, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

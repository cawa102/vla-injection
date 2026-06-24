#!/usr/bin/env python3
"""M1 viability-gate report (Task 6) — the GO/NO-GO (H1) aggregate.

Thin, model-free loader/writer around :func:`evasion_tax.eval.m1_gate.m1_verdict`:
read the benign baseline records (Task 4) + the attacked-unit records (Task 5),
aggregate into the four-part H1 verdict + the folded attack-cost distribution, and
write one ``m1_gate_report`` to a write-once ``results/`` run dir. Pure aggregation
on the mac — no GPU, no guard.

    PYTHONPATH=src .venv/bin/python scripts/m1_gate_report.py \\
        --benign results/<benign-run>/benign_records.json \\
        --attack results/<attack-run>/attack_records.json \\
        --fpr 0.05 --n-steps-cap 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.eval.m1_gate import (  # noqa: E402
    attack_records_from_dicts,
    benign_records_from_dicts,
    m1_verdict,
)
from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402
from evasion_tax.repro.run_logger import RunLogger  # noqa: E402


def _load_list(path: str) -> list:
    items = json.loads(Path(path).read_text())
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a JSON list of records, got {type(items).__name__}")
    return items


def build_verdict(args: argparse.Namespace) -> dict:
    """Load both record files and compute the verdict (no I/O side effects)."""
    benign = benign_records_from_dicts(_load_list(args.benign))
    attack = attack_records_from_dicts(_load_list(args.attack))
    schema = SchemaA(engagement_radius=args.schema_engagement, grasp_radius=args.schema_grasp)
    return m1_verdict(
        benign,
        attack,
        schema=schema,
        fpr=args.fpr,
        n_steps_cap=args.n_steps_cap,
        s_per_step=args.s_per_step,
        ceiling_auc=args.ceiling_auc,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign", required=True, help="benign_records.json (list) from Task 4")
    parser.add_argument("--attack", required=True, help="attack_records.json (list) from Task 5")
    parser.add_argument("--fpr", type=float, default=0.05, help="primary false-abort budget")
    parser.add_argument("--n-steps-cap", type=int, required=True, help="GCG n_steps cap")
    parser.add_argument("--s-per-step", type=float, default=33.19, help="D8-measured GCG s/step")
    parser.add_argument(
        "--ceiling-auc", type=float, default=None,
        help="optional clean-instruction-ceiling separation AUC (weak-necessity check)",
    )
    parser.add_argument("--schema-engagement", type=float, default=SchemaA().engagement_radius)
    parser.add_argument("--schema-grasp", type=float, default=SchemaA().grasp_radius)
    parser.add_argument("--results-root", default="results", help="write-once results root")
    args = parser.parse_args(argv)

    verdict = build_verdict(args)

    logger = RunLogger(args.results_root)
    handle = logger.start(
        slug="m1-gate-report",
        config={"stage": "m1_gate_report", "fpr": args.fpr, "n_steps_cap": args.n_steps_cap},
        seed=0,
    )
    handle.write("m1_gate_report", verdict)
    print(f"[m1-gate] {verdict['verdict']}: {verdict['summary']}")
    print(f"[m1-gate] report -> {handle.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

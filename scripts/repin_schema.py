#!/usr/bin/env python3
"""Lock the DM-3 SchemaA radius re-pin from a benign run's geometry (pure, no GPU).

The bridge between Task 4 (benign baseline) and Task 5 (attack): read the
benign **calibration-split** ``geometry_stats.json`` and apply the LOCKED
``repin_schema_from_benign`` (D-3 §3 estimators + §4 guards) to produce the frozen
``schema_repinned.json`` that ``run_attack --schema-from`` consumes. Records the
before/after radii so the dated deviation can be entered in
``docs/core/metric-a-annotation-schema.md`` §5 (D-3 §5 protocol).

This step MUST run AFTER the benign baseline and BEFORE any attacked output is
scored (DM-3 / invariant #2). It reads benign geometry only — never attacked data.

    PYTHONPATH=src python scripts/repin_schema.py \\
        --geometry results/<benign-run>/geometry_stats.json \\
        --out results/<benign-run>/schema_repinned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.eval.schema_repin import repin_schema_from_benign  # noqa: E402
from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, help="benign calib geometry_stats.json (list)")
    parser.add_argument("--out", required=True, help="output schema_repinned.json path")
    args = parser.parse_args(argv)

    geometry = json.loads(Path(args.geometry).read_text())
    if not isinstance(geometry, list):
        raise ValueError(f"{args.geometry} must be a JSON list of benign geometry records")

    base = SchemaA()
    repinned = repin_schema_from_benign(geometry, base=base)

    record = {
        "engagement_radius": repinned.engagement_radius,
        "grasp_radius": repinned.grasp_radius,
        "combination": repinned.combination,
        # before-values for the dated deviation entry (D-3 §5 / schema doc §5)
        "base_engagement_radius": base.engagement_radius,
        "base_grasp_radius": base.grasp_radius,
        "n_benign_geometry": len(geometry),
        "geometry_source": args.geometry,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(
        f"[repin] engagement_radius {base.engagement_radius} -> {repinned.engagement_radius}; "
        f"grasp_radius {base.grasp_radius} -> {repinned.grasp_radius} (n={len(geometry)})"
    )
    print(f"[repin] frozen schema -> {args.out}; record the dated deviation in the schema doc §5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

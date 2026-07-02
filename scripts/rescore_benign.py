#!/usr/bin/env python3
"""Re-score a benign baseline against the re-pinned SchemaA (BUG2 — DM-3 same-scale).

The benign baseline scores its rollouts BEFORE the DM-3 re-pin (box step 1 runs
without ``--schema-from``), so its ``benign_records.json`` is on the placeholder-
schema scale while the attack is scored on the re-pinned schema — comparing them
(the M1 separation AUC) is cross-scale and INVALID. This step re-scores the benign
run's already-logged rollouts against the re-pinned ``SchemaA`` (same ``k`` as the
run), writing ``benign_records_repinned.json`` so the gate report compares benign
and attack on the SAME scale.

Model-free (pure re-score of on-disk rollouts) — no GPU, no guard. MUST run AFTER
``repin_schema.py`` and BEFORE the gate report. The benign rollouts are never
re-rolled (that would re-spend GPU hours and could drift).

    PYTHONPATH=src .venv/bin/python scripts/rescore_benign.py \\
        --run results/m1-benign-baseline \\
        --schema-from results/m1-benign-baseline/schema_repinned.json \\
        --out results/m1-benign-baseline/benign_records_repinned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.eval.rescore_benign import rescore_benign_records  # noqa: E402
from evasion_tax.eval.rollout_io import rollout_from_log  # noqa: E402
from evasion_tax.metric.consistency_a import SchemaA  # noqa: E402


def _load_schema(path: str) -> SchemaA:
    """The re-pinned ``SchemaA`` from ``--schema-from`` — radii only.

    Mirrors ``run_attack.load_frozen_schema`` (reads only the two radii, so
    ``combination`` stays the frozen ``max``); this guarantees benign and attack
    are scored with a bit-identical schema.
    """
    d = json.loads(Path(path).read_text())
    return SchemaA(
        engagement_radius=float(d["engagement_radius"]),
        grasp_radius=float(d["grasp_radius"]),
    )


def _episode_files(run_dir: Path) -> list[Path]:
    """The benign run's per-episode logs, ordered by episode index."""
    files = sorted((run_dir / "episodes").glob("ep*.json"), key=lambda p: int(p.stem[2:]))
    if not files:
        raise FileNotFoundError(f"no episodes/ep*.json under {run_dir} to re-score")
    return files


def _run_k(run_dir: Path) -> int:
    """The metric ``k`` the run scored at (re-scoring must use the same k)."""
    run_json = json.loads((run_dir / "run.json").read_text())
    k = run_json.get("config", {}).get("k")
    if k is None:
        raise ValueError(f"{run_dir}/run.json has no config.k (needed to re-score at the run's k)")
    return int(k)


def build_repinned_records(run_dir: Path, schema: SchemaA) -> list[dict]:
    """Load the run's logged episodes + k and re-score them against ``schema``.

    Pure aggregation (no output I/O): reconstructs each rollout from its logged
    ``RolloutStep`` stream and re-scores metric (A) on the re-pinned schema, carrying
    the per-episode ``success`` / ``is_calibration`` through unchanged.
    """
    k = _run_k(run_dir)
    episodes = [json.loads(p.read_text()) for p in _episode_files(run_dir)]
    rollouts = [rollout_from_log(ep) for ep in episodes]
    return rescore_benign_records(
        rollouts,
        success=[bool(ep["success"]) for ep in episodes],
        is_calibration=[bool(ep["is_calibration"]) for ep in episodes],
        schema=schema,
        k=k,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="benign run dir (episodes/ + run.json)")
    parser.add_argument("--schema-from", required=True, help="re-pinned SchemaA JSON (radii)")
    parser.add_argument("--out", required=True, help="output benign_records_repinned.json path")
    args = parser.parse_args(argv)

    schema = _load_schema(args.schema_from)
    records = build_repinned_records(Path(args.run), schema)
    Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
    print(
        f"[rescore-benign] re-scored {len(records)} benign rollouts on the re-pinned schema "
        f"(engagement={schema.engagement_radius}, grasp={schema.grasp_radius}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

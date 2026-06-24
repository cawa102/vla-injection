"""Tests for ``scripts/repin_schema.py`` (the DM-3 lock step) — pure, no GPU.

Reads a benign run's calibration ``geometry_stats.json`` and writes the frozen
re-pinned ``SchemaA`` JSON that ``run_attack --schema-from`` consumes. The re-pin
itself is the LOCKED ``repin_schema_from_benign`` (Task 0); this script is the
thin loader/writer around it.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return importlib.import_module("repin_schema")


def _geometry(min_ee, grasp_ee, grasp_dist, min_distractor):
    return {
        "success": True,
        "anchor_resolvable": True,
        "min_ee_anchor": min_ee,
        "min_distractor": min_distractor,
        "grasp_events": [{"ee_anchor": grasp_ee, "min_distractor": grasp_dist}],
    }


def _both_repin_geometry():
    rows = zip(
        [0.02, 0.025, 0.03, 0.035, 0.04],
        [0.06, 0.07, 0.08, 0.09, 0.10],
        [0.15, 0.16, 0.17, 0.18, 0.19],
        [0.05, 0.06, 0.07, 0.08, 0.09],
        strict=True,
    )
    return [_geometry(*r) for r in rows]


def test_writes_repinned_schema_json(tmp_path):
    mod = _load()
    geom = tmp_path / "geometry_stats.json"
    geom.write_text(json.dumps(_both_repin_geometry()))
    out = tmp_path / "schema_repinned.json"
    rc = mod.main(["--geometry", str(geom), "--out", str(out)])
    assert rc == 0
    schema = json.loads(out.read_text())
    assert schema["engagement_radius"] == 0.035   # 1.2*median(A) -> 0.035
    assert schema["grasp_radius"] == 0.115         # 1.2*P90(G) -> 0.115


def test_records_before_after(tmp_path):
    mod = _load()
    geom = tmp_path / "geometry_stats.json"
    geom.write_text(json.dumps(_both_repin_geometry()))
    out = tmp_path / "schema_repinned.json"
    mod.main(["--geometry", str(geom), "--out", str(out)])
    schema = json.loads(out.read_text())
    # the dated-deviation record carries the before (base) radii for the schema doc
    assert schema["base_engagement_radius"] == 0.05
    assert schema["base_grasp_radius"] == 0.10

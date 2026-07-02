"""Tests for ``scripts/m1_gate_report.py`` (Task 6 — the H1 GO/NO-GO report).

Model-free: builds benign + attacked record JSONs, runs the loader/writer, and
checks the written verdict. No GPU.
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
    return importlib.import_module("m1_gate_report")


def _benign_dicts():
    return (
        [{"success": True, "metric_a_per_step": [0.0], "is_calibration": True} for _ in range(6)]
        + [{"success": True, "metric_a_per_step": [0.0], "is_calibration": False} for _ in range(4)]
    )


def _cost(steps, reached):
    return {
        "target_id": f"t{steps}", "reached": reached, "steps_to_success": steps,
        "censored": not reached, "best_loss": 0.1, "wall_seconds": 100.0,
        "peak_vram_gib": 15.0, "suffix_sha256": "deadbeef",
    }


def _attack_dicts():
    return [
        {
            "unit_id": f"u{i}", "rollout_asr_reached": True, "is_denial": False,
            "metric_a_per_step": [1.0], "cost": _cost(60, True),
        }
        for i in range(4)
    ]


def _write(tmp_path: Path):
    benign = tmp_path / "benign_records.json"
    attack = tmp_path / "attack_records.json"
    benign.write_text(json.dumps(_benign_dicts()))
    attack.write_text(json.dumps(_attack_dicts()))
    return benign, attack


def test_build_verdict_go(tmp_path):
    mod = _load()
    benign, attack = _write(tmp_path)
    args = mod.argparse.Namespace(
        benign=str(benign), attack=str(attack), fpr=0.05, n_steps_cap=500,
        s_per_step=33.19, ceiling_auc=None, schema_engagement=0.05, schema_grasp=0.10,
    )
    verdict = mod.build_verdict(args)
    assert verdict["go"] is True
    assert verdict["cost"]["steps_to_success"]["median_steps"] == 60.0


def test_build_verdict_falls_back_to_units_when_aggregate_absent(tmp_path):
    # BUG1 belt-and-suspenders: if attack_records.json is missing but the per-unit
    # units/*.json exist, the report reconstructs the aggregate from those.
    mod = _load()
    benign = tmp_path / "benign_records.json"
    benign.write_text(json.dumps(_benign_dicts()))

    attack_dir = tmp_path / "attack-run"
    units = attack_dir / "units"
    units.mkdir(parents=True)
    for i, rec in enumerate(_attack_dicts()):
        (units / f"u{i}.json").write_text(json.dumps(rec))
    # attack_dir / "attack_records.json" is deliberately absent.

    args = mod.argparse.Namespace(
        benign=str(benign), attack=str(attack_dir / "attack_records.json"),
        fpr=0.05, n_steps_cap=500, s_per_step=33.19, ceiling_auc=None,
        schema_engagement=0.05, schema_grasp=0.10,
    )
    verdict = mod.build_verdict(args)
    assert verdict["redirect"]["n"] == len(_attack_dicts())


def test_main_writes_report(tmp_path):
    mod = _load()
    benign, attack = _write(tmp_path)
    rc = mod.main([
        "--benign", str(benign), "--attack", str(attack),
        "--n-steps-cap", "500", "--results-root", str(tmp_path / "results"),
    ])
    assert rc == 0
    reports = list((tmp_path / "results").glob("*-m1-gate-report/m1_gate_report.json"))
    assert len(reports) == 1
    verdict = json.loads(reports[0].read_text())
    assert verdict["verdict"] == "GO"

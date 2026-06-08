"""End-to-end CLI tests for ``scripts/evaluate.py`` (wiring, invariant #3).

Runs the script as a subprocess so the full reporting path is exercised: config
load → split-disjointness assertion → condition matrix → write-once results. The
key guarantee here is that an overlapping calibration/test config aborts the run
*loudly* before anything is calibrated (the leakage rail is enforced, not merely
declared).
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "evaluate.py"


def _config(*, calib_tasks: list[str], test_tasks: list[str]) -> dict:
    """A minimal valid run config with controllable split overlap."""
    return {
        "seed": 42,
        "model": {"name": "openvla-7b", "unnorm_key": "libero_spatial_no_noops"},
        "env": {"suite": "libero_spatial", "tasks": ["task_0"], "max_steps": 200},
        "attack": {"name": "robogcg", "targets_per_task": 20, "persistence_steps": 5},
        "metric": {"k": 5},
        "detector": {"fpr_targets": [0.01, 0.05], "primary_fpr": 0.05},
        "eval": {
            "matrix": ["clean_ceiling"],
            "splits": {
                "calib": {"tasks": calib_tasks, "scenes": ["scene_0"], "seeds": [42]},
                "test": {"tasks": test_tasks, "scenes": ["scene_1"], "seeds": [43]},
            },
        },
    }


def _run(tmp_path: Path, config: dict, conditions: dict) -> subprocess.CompletedProcess:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(conditions))
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--config",
            str(cfg_path),
            "--scores",
            str(scores_path),
            "--results-root",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
    )


def test_evaluate_aborts_on_overlapping_splits(tmp_path):
    # calib and test share task_0 → calibration/test leakage on the 'tasks' axis.
    config = _config(calib_tasks=["task_0"], test_tasks=["task_0", "task_1"])
    proc = _run(tmp_path, config, conditions={})
    assert proc.returncode != 0
    assert "leakage" in (proc.stdout + proc.stderr).lower()
    # Aborted before writing anything.
    assert not (tmp_path / "results").exists()


def test_evaluate_runs_on_disjoint_splits(tmp_path):
    config = _config(calib_tasks=["task_0"], test_tasks=["task_1"])
    proc = _run(tmp_path, config, conditions={})
    assert proc.returncode == 0, proc.stderr
    # An empty condition matrix still writes a (write-once) results.json.
    results = list((tmp_path / "results").rglob("results.json"))
    assert results, "evaluate.py did not write results.json on the happy path"
    assert json.loads(results[0].read_text()) == {"conditions": {}}

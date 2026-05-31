"""Tests for script-regenerable figures (Task 9).

``make_figures`` is the M2 deliverable that regenerates every figure **purely
from logged result arrays** (no rollout / metric / detector is re-run): it reads
a ``results.json`` written by the eval harness and emits, per condition, a ROC
curve, a score-distribution histogram, and a TPR@FPR-with-CI bar — plus a single
ladder-table placeholder (the M3 rung ladder is not built yet).
"""

import json

import numpy as np
import pytest

from t7.eval.figures import make_figures, results_table_to_dict
from t7.eval.harness import run_condition_matrix


def _synthetic_results() -> dict:
    return {
        "conditions": {
            "clean_ceiling": {
                "auc": 0.97,
                "operating_points": [
                    {
                        "fpr_target": 0.01,
                        "tau": 0.5,
                        "tpr": 0.90,
                        "tpr_ci": [0.80, 0.95],
                        "realised_fpr": 0.008,
                        "realised_fpr_ci": [0.0, 0.02],
                        "n_benign": 150,
                        "n_attacked": 150,
                    },
                    {
                        "fpr_target": 0.05,
                        "tau": 0.4,
                        "tpr": 0.96,
                        "tpr_ci": [0.90, 0.99],
                        "realised_fpr": 0.04,
                        "realised_fpr_ci": [0.02, 0.07],
                        "n_benign": 150,
                        "n_attacked": 150,
                    },
                ],
                "score_arrays": {
                    "benign_test": [0.05, 0.1, 0.2, 0.15, 0.08, 0.12],
                    "attacked_test": [0.8, 0.9, 0.7, 0.85, 0.95, 0.75],
                },
            },
            "perplexity": {
                "auc": 0.61,
                "operating_points": [
                    {
                        "fpr_target": 0.01,
                        "tau": 0.6,
                        "tpr": 0.20,
                        "tpr_ci": [0.10, 0.30],
                        "realised_fpr": 0.009,
                        "realised_fpr_ci": [0.0, 0.03],
                        "n_benign": 150,
                        "n_attacked": 150,
                    }
                ],
                "score_arrays": {
                    "benign_test": [0.1, 0.2, 0.3, 0.25, 0.15, 0.18],
                    "attacked_test": [0.2, 0.4, 0.3, 0.5, 0.35, 0.45],
                },
            },
        }
    }


def _write_results(results_dir, data: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(data))


def test_make_figures_writes_expected_files(tmp_path):
    results_dir = tmp_path / "run"
    out_dir = tmp_path / "figs"
    _write_results(results_dir, _synthetic_results())

    make_figures(results_dir, out_dir)

    for name in ("clean_ceiling", "perplexity"):
        assert (out_dir / f"{name}_roc.png").exists()
        assert (out_dir / f"{name}_score_hist.png").exists()
        assert (out_dir / f"{name}_tpr_at_fpr.png").exists()
    assert (out_dir / "ladder_placeholder.png").exists()


def test_make_figures_returns_written_paths(tmp_path):
    results_dir = tmp_path / "run"
    out_dir = tmp_path / "figs"
    _write_results(results_dir, _synthetic_results())

    written = make_figures(results_dir, out_dir)

    assert all(p.exists() for p in written)
    # 3 figs * 2 conditions + 1 ladder placeholder.
    assert len(written) == 7


def test_make_figures_creates_missing_out_dir(tmp_path):
    results_dir = tmp_path / "run"
    out_dir = tmp_path / "nested" / "figs"  # does not exist yet
    _write_results(results_dir, _synthetic_results())

    make_figures(results_dir, out_dir)

    assert out_dir.is_dir()


def test_make_figures_raises_when_results_missing(tmp_path):
    results_dir = tmp_path / "empty"
    results_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        make_figures(results_dir, tmp_path / "figs")


def test_serialised_table_round_trips_into_make_figures(tmp_path):
    # results_table_to_dict (eval producer) and make_figures (consumer) must
    # agree on one on-disk format: a real ResultsTable serialises into exactly
    # the schema make_figures reads back.
    rng = np.random.default_rng(0)
    condition = {
        "benign_calib": [list(rng.uniform(0.0, 0.3, 6)) for _ in range(50)],
        "benign_test": [list(rng.uniform(0.0, 0.3, 6)) for _ in range(50)],
        "attacked_test": [list(rng.uniform(0.6, 1.0, 6)) for _ in range(50)],
    }
    table = run_condition_matrix({"gcg": condition})
    serialised = results_table_to_dict(table)

    gcg = serialised["conditions"]["gcg"]
    assert {"auc", "operating_points", "score_arrays"} <= set(gcg)
    assert "benign_test" in gcg["score_arrays"]

    results_dir = tmp_path / "run"
    results_dir.mkdir()
    (results_dir / "results.json").write_text(json.dumps(serialised))
    make_figures(results_dir, tmp_path / "figs")
    assert (tmp_path / "figs" / "gcg_roc.png").exists()

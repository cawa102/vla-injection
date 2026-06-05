"""Script-regenerable figures from logged results (Task 9, M2 deliverable).

Every figure is regenerated **purely from logged arrays** — no rollout, metric,
or detector is re-run (plan: figures must be regenerable from logged data by a
script). The on-disk contract is a single ``results.json`` (the serialised
:class:`~evasion_tax.eval.harness.ResultsTable`); :func:`results_table_to_dict` writes it
and :func:`make_figures` reads it back, so the eval script and the figure script
agree on one format (DRY).

Per condition, three figures are emitted — a ROC curve, a benign-vs-attacked
score histogram, and a TPR@FPR-with-CI bar — plus one ladder-table placeholder
(the M3 trusted-reference rung ladder is not built yet). The ROC *curve* is a
plotting transform of the logged per-rollout scores; the headline AUC and the
operating-point rates/CIs are read straight from the log, never recomputed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt

from evasion_tax.eval.harness import ResultsTable
from evasion_tax.eval.metrics import roc_auc

# Headless backend so figures render without a display (CI / 8 GB host).
plt.switch_backend("Agg")

StrPath = str | PathLike[str]

_RESULTS_FILENAME = "results.json"


def results_table_to_dict(table: ResultsTable) -> dict:
    """Serialise a :class:`ResultsTable` to the ``results.json`` schema.

    Merges each row's metrics with its raw score arrays so a single document
    holds everything :func:`make_figures` needs.

    Args:
        table: The evaluated results table.

    Returns:
        ``{"conditions": {name: {auc, operating_points, latency_summary,
        score_arrays}}}`` — JSON-serialisable.
    """
    conditions: dict = {}
    for row in table.rows:
        conditions[row.condition] = {
            "auc": row.auc,
            "operating_points": [asdict(op) for op in row.operating_points],
            "latency_summary": row.latency_summary,
            "score_arrays": table.score_arrays.get(row.condition, {}),
        }
    return {"conditions": conditions}


def _load_results(results_dir: Path) -> dict:
    path = results_dir / _RESULTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {_RESULTS_FILENAME} under {results_dir} — run the eval script first"
        )
    return json.loads(path.read_text())


def _roc_figure(name: str, arrays: dict, auc: float, out_dir: Path) -> Path:
    benign = arrays.get("benign_test", [])
    attacked = arrays.get("attacked_test", [])
    fig, ax = plt.subplots()
    if benign and attacked:
        fpr, tpr, _ = roc_auc(benign, attacked)
        ax.plot(fpr, tpr, marker=".")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("benign false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(f"{name} ROC (AUC={auc:.3f})")
    target = out_dir / f"{name}_roc.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def _score_hist_figure(name: str, arrays: dict, out_dir: Path) -> Path:
    benign = arrays.get("benign_test", [])
    attacked = arrays.get("attacked_test", [])
    fig, ax = plt.subplots()
    if benign:
        ax.hist(benign, bins=20, alpha=0.6, label="benign", range=(0.0, 1.0))
    if attacked:
        ax.hist(attacked, bins=20, alpha=0.6, label="attacked", range=(0.0, 1.0))
    ax.set_xlabel("per-rollout consistency score")
    ax.set_ylabel("count")
    ax.set_title(f"{name} score distribution")
    ax.legend()
    target = out_dir / f"{name}_score_hist.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def _tpr_at_fpr_figure(name: str, operating_points: list[dict], out_dir: Path) -> Path:
    labels = [f"{op['fpr_target']:.0%}" for op in operating_points]
    tprs = [op["tpr"] for op in operating_points]
    # CI half-widths for asymmetric error bars (lower, upper) around each TPR.
    lower = [tpr - op["tpr_ci"][0] for tpr, op in zip(tprs, operating_points, strict=True)]
    upper = [op["tpr_ci"][1] - tpr for tpr, op in zip(tprs, operating_points, strict=True)]
    fig, ax = plt.subplots()
    ax.bar(labels, tprs, yerr=[lower, upper], capsize=5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("target benign false-abort rate")
    ax.set_ylabel("TPR (with 95% CI)")
    ax.set_title(f"{name} TPR @ FPR")
    target = out_dir / f"{name}_tpr_at_fpr.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def _ladder_placeholder_figure(out_dir: Path) -> Path:
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "trusted-reference ladder\n(populated at M3)",
        ha="center",
        va="center",
    )
    ax.set_axis_off()
    target = out_dir / "ladder_placeholder.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def make_figures(results_dir: StrPath, out_dir: StrPath) -> list[Path]:
    """Regenerate all M2 figures from a logged ``results.json``.

    Args:
        results_dir: Directory containing ``results.json`` (written by the eval
            script / :func:`results_table_to_dict`).
        out_dir: Directory to write the PNG figures to (created if absent).

    Returns:
        The paths of every figure written.

    Raises:
        FileNotFoundError: If ``results_dir`` has no ``results.json``.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_dir)
    written: list[Path] = []
    for name, cond in results["conditions"].items():
        arrays = cond.get("score_arrays", {})
        written.append(_roc_figure(name, arrays, cond["auc"], out_dir))
        written.append(_score_hist_figure(name, arrays, out_dir))
        written.append(
            _tpr_at_fpr_figure(name, cond.get("operating_points", []), out_dir)
        )
    written.append(_ladder_placeholder_figure(out_dir))
    return written

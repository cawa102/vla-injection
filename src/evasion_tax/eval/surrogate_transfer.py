"""Logged-result aggregation for quantized-surrogate transfer experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evasion_tax.attack.surrogate_artifacts import TransferEvalRecord
from evasion_tax.eval.metrics import proportion_ci

plt.switch_backend("Agg")

StrPath = str | PathLike[str]


def _asr_row(precision: str, rows: Sequence[TransferEvalRecord]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("cannot summarise an empty precision group")
    hits = sum(1 for r in rows if r.victim_target_hit)
    surrogate_hits = sum(1 for r in rows if r.surrogate_target_hit)
    censored = sum(1 for r in rows if r.censored)
    failed = sum(1 for r in rows if r.failure_reason)
    gpu_hours = sum(max(r.wall_seconds, 0.0) for r in rows) / 3600.0
    asr = hits / n
    surrogate_asr = surrogate_hits / n
    return {
        "surrogate_precision": precision,
        "n": n,
        "victim_hits": hits,
        "victim_asr": asr,
        "victim_asr_ci": proportion_ci(hits, n, method="wilson"),
        "surrogate_hits": surrogate_hits,
        "surrogate_asr": surrogate_asr,
        "transfer_gap": surrogate_asr - asr,
        "censored": censored,
        "censored_fraction": censored / n,
        "failed": failed,
        "gpu_hours": gpu_hours,
        "cost_normalized_asr": None if gpu_hours == 0.0 else asr / gpu_hours,
    }


def summarize_transfer(records: Sequence[TransferEvalRecord]) -> dict:
    """Summarize victim ASR, transfer gap, cost-normalized ASR, and censoring."""
    if not records:
        raise ValueError("summarize_transfer needs at least one record")
    by_precision: dict[str, list[TransferEvalRecord]] = defaultdict(list)
    for record in records:
        by_precision[record.surrogate_precision].append(record)
    rows = [_asr_row(p, by_precision[p]) for p in sorted(by_precision)]
    steps = [
        {
            "surrogate_precision": r.surrogate_precision,
            "transfer_id": r.transfer_id,
            "victim_target_hit": r.victim_target_hit,
            "censored": r.censored,
            "failure_reason": r.failure_reason,
        }
        for r in records
    ]
    return {"by_precision": rows, "targets": steps}


def write_summary_outputs(summary: dict, out_dir: StrPath) -> list[Path]:
    """Write summary JSON/CSV and a steps/censoring plot once.

    ``out_dir`` itself is the write-once guard: if it already exists, the caller is
    trying to edit an existing result summary, so we fail instead of overwriting.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    written.append(summary_path)

    csv_path = out / "asr_by_precision.csv"
    rows = summary["by_precision"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written.append(csv_path)

    fig, ax = plt.subplots()
    labels = [r["surrogate_precision"] for r in rows]
    asr = [r["victim_asr"] for r in rows]
    censored = [r["censored_fraction"] for r in rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, asr, width=0.36, label="victim ASR")
    ax.bar(x + 0.18, censored, width=0.36, label="censored")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Surrogate transfer outcomes")
    ax.legend()
    plot_path = out / "steps_to_success_censored.png"
    fig.savefig(plot_path)
    plt.close(fig)
    written.append(plot_path)
    return written

#!/usr/bin/env python3
"""Powered analysis of an M1 attack sweep: attack characterisation + L2 detector separation.

Pure aggregation (no GPU/model). Reads the benign baseline (N=300, ``is_calibration`` split;
same frozen SchemaA 0.05/0.10 as the attack via --schema-from, so scores are same-scale) and an
attack sweep's ``attack_records.json``, and reports for the paper:

  * ATTACK characterisation: approach_asr / reached_single_frame / denial RATES (Wilson 95% CI)
    + the min_ee_distractor distribution.
  * L2 goal-action DETECTOR separation: per-rollout score = max over the per-step metric-A
    (inconsistency) scores; ROC-AUC and TPR@{1%,5%} held-out FPR in BOTH polarities —
    fire-on-HIGH (the redirect/anomaly convention the detector was designed for) and fire-on-LOW
    (the denial convention) — so the honest direction is explicit. tau is calibrated on the
    benign CALIBRATION split; realised_fpr is the HELD-OUT benign fire-rate (invariant #3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)
import numpy as np

from evasion_tax.eval.metrics import proportion_ci, roc_auc  # noqa: E402


def _rollmax(r: dict) -> float | None:
    m = r.get("metric_a_per_step") or []
    return float(np.max(m)) if len(m) else None


def _rate(k: int, n: int) -> dict:
    lo, hi = proportion_ci(k, n, method="wilson") if n else (0.0, 0.0)
    return {"k": k, "n": n, "rate": (k / n if n else None), "ci95": [lo, hi]}


def _op_point(calib, test, attacked, fpr: float, *, fire_low: bool) -> dict:
    calib = np.asarray(calib, dtype=float)
    test = np.asarray(test, dtype=float)
    att = np.asarray(attacked, dtype=float)
    if fire_low:  # fire if score <= tau (denial => low inconsistency)
        tau = float(np.quantile(calib, fpr))
        realised_fpr = float(np.mean(test <= tau))
        fired = int(np.sum(att <= tau))
    else:  # fire if score >= tau (redirect/anomaly => high inconsistency)
        tau = float(np.quantile(calib, 1.0 - fpr))
        realised_fpr = float(np.mean(test >= tau))
        fired = int(np.sum(att >= tau))
    lo, hi = proportion_ci(fired, len(att), method="wilson")
    return {"tau": tau, "realised_fpr": realised_fpr, "tpr": fired / len(att),
            "tpr_ci95": [lo, hi], "n_attacked": len(att)}


def _dist(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)), "n": int(a.size)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benign", default="results/m1-object-benign/benign_records.json")
    p.add_argument("--attack", required=True, help="attack sweep attack_records.json")
    p.add_argument("--label", default="attack")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    benign = json.loads(Path(args.benign).read_text())
    att = json.loads(Path(args.attack).read_text())
    n = len(att)

    chars = {
        "n": n,
        "approach_asr": _rate(sum(bool(r.get("approach_asr")) for r in att), n),
        "reached_single_frame": _rate(sum(bool(r.get("reached_single_frame")) for r in att), n),
        "denial": _rate(sum(bool(r.get("is_denial")) for r in att), n),
    }
    mind = [r["min_ee_distractor"] for r in att if r.get("min_ee_distractor") is not None]
    chars["min_ee_distractor"] = _dist(mind) if mind else None
    chars["n_reach_005"] = sum(1 for x in mind if x <= 0.05)  # crossed the ASR radius at all

    def _scores(records, keep):
        return [s for r in records if keep(r) and (s := _rollmax(r)) is not None]

    b_calib = _scores(benign, lambda r: r.get("is_calibration"))
    b_test = _scores(benign, lambda r: not r.get("is_calibration"))
    a_scores = _scores(att, lambda r: True)
    det = {"n_benign_calib": len(b_calib), "n_benign_test": len(b_test), "n_attacked": len(a_scores)}
    if b_test and a_scores:
        _, _, auc_hi = roc_auc(b_test, a_scores)
        det["auc_fire_high"] = auc_hi
        det["auc_fire_low"] = 1.0 - auc_hi
        det["benign_test_score"] = _dist(b_test)
        det["attacked_score"] = _dist(a_scores)
        if b_calib:
            for fpr in (0.01, 0.05):
                det[f"fire_high@fpr{fpr}"] = _op_point(
                    b_calib, b_test, a_scores, fpr, fire_low=False)
                det[f"fire_low@fpr{fpr}"] = _op_point(
                    b_calib, b_test, a_scores, fpr, fire_low=True)

    out = {"label": args.label, "attack": args.attack,
           "attack_characterisation": chars, "detector_separation": det}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

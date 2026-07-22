#!/usr/bin/env python3
"""Paper figures from an M1 attack sweep + the benign baseline (pure, headless matplotlib).

Fig 1  min_ee_distractor histogram (attack) vs the 0.05 m ASR radius / 0.086 m benign-closest /
       0.042 m clean-adversary references — the attack never crosses the redirect region.
Fig 2  single-frame reach (~100%) vs closed-loop approach_asr (0%) — the embodiment gap.
Fig 3  detector per-rollout score (max metric-A) distributions: benign vs attacked — the
       saturation (benign near 1.0) that makes the frozen detector unusable at 0.05/0.10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_BENIGN_CLOSEST = 0.086
_CLEAN_ADV = 0.042
_RADIUS = 0.05


def _rollmax(r):
    m = r.get("metric_a_per_step") or []
    return float(np.max(m)) if len(m) else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attack", required=True)
    p.add_argument("--benign", default="results/m1-object-benign/benign_records.json")
    p.add_argument("--label", default="semantic")
    p.add_argument("--outdir", default="results/figures")
    args = p.parse_args()
    att = json.loads(Path(args.attack).read_text())
    benign = json.loads(Path(args.benign).read_text())
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    mind = [r["min_ee_distractor"] for r in att if r.get("min_ee_distractor") is not None]
    appr = sum(bool(r.get("approach_asr")) for r in att)
    reach = sum(bool(r.get("reached_single_frame")) for r in att)
    n = len(att)

    # Fig 1: min_ee_distractor histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mind, bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(_RADIUS, color="#C44E52", ls="--", lw=2, label=f"ASR radius {_RADIUS} m (0 crossed)")
    ax.axvline(_BENIGN_CLOSEST, color="#55A868", ls=":", lw=2,
               label=f"benign closest {_BENIGN_CLOSEST} m")
    ax.axvline(_CLEAN_ADV, color="#8172B2", ls="-.", lw=2,
               label=f"clean-adversary reach {_CLEAN_ADV} m")
    ax.set_xlabel("min EE→distractor distance (m)")
    ax.set_ylabel("attacked rollouts")
    ax.set_title(f"{args.label} attack (N={n}): closest approach to the wrong object")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig1_min_ee_hist_{args.label}.png", dpi=150)
    plt.close(fig)

    # Fig 2: single-frame reach vs closed-loop approach_asr
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["single-frame\nreach", "closed-loop\napproach_asr"],
                  [reach / n, appr / n], color=["#55A868", "#C44E52"])
    for b, v in zip(bars, [reach / n, appr / n], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("rate")
    ax.set_title(f"{args.label} (N={n}): single-frame control ≠ closed-loop redirect")
    fig.tight_layout()
    fig.savefig(out / f"fig2_reach_vs_approach_{args.label}.png", dpi=150)
    plt.close(fig)

    # Fig 3: detector score distributions
    b_scores = [s for r in benign if (s := _rollmax(r)) is not None]
    a_scores = [s for r in att if (s := _rollmax(r)) is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(b_scores, bins=bins, alpha=0.6, density=True, color="#55A868",
            label=f"benign (N={len(b_scores)})")
    ax.hist(a_scores, bins=bins, alpha=0.6, density=True, color="#C44E52",
            label=f"attacked (N={len(a_scores)})")
    ax.set_xlabel("per-rollout detector score = max metric-A")
    ax.set_ylabel("density")
    ax.set_title(f"L2 detector saturates on benign ({args.label})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"fig3_detector_scores_{args.label}.png", dpi=150)
    plt.close(fig)

    print(f"[figures] wrote fig1/fig2/fig3 for {args.label} -> {out}")


if __name__ == "__main__":
    main()

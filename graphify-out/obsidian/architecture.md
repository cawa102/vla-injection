---
source_file: "docs/CODEMAPS/architecture.md"
type: "document"
community: "Codemaps & Architecture"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/Codemaps__Architecture
---

# architecture.md

## Connections
- [[Architecture map (Embodiment Evasion Tax)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/Codemaps__Architecture

## 📄 Source

`docs/CODEMAPS/architecture.md`

<!-- Generated: 2026-06-05 | Files scanned: 36 src | Token estimate: ~750 -->

# Architecture — Embodiment Evasion Tax

MSc research code, **not** an app: no HTTP API, no UI, no DB. A Python package `src/evasion_tax`
(model-free, unit-tested on M1/8 GB) + a thin CLI in `scripts/` (GPU-node-only runs).
Source of truth = `docs/core/execution-playbook.md`. Siblings: [modules](modules.md) ·
[data](data.md) · [dependencies](dependencies.md).

## What it measures
Per-layer **adaptive-evasion cost** on instruction-injected **OpenVLA-7B** (bf16, LIBERO sim).
Three detection layers, ordered by where they read the perception→reasoning→action pipeline:

```
L0  input        perplexity / text-only filter   baselines/perplexity.py
L1  internal     activation-delta probe          metric/probe_internal.py
L2  behavioural  goal-action consistency (A)      metric/consistency_a.py
       ├─ L2-oracle     = metric A (privileged; non-deployable ceiling / M3 floor)
       └─ L2-deployable = B/C  (M4, not yet built)
```

## Model-free / GPU boundary (the core design split)
```
MODEL-FREE (local, here · 354 tests green)       GPU NODE (Kelvin2, deferred)
────────────────────────────────────────        ──────────────────────────────
metric A scorer · FP-calibrated detector         OpenVLA-7B inference
eval stats (ROC/AUC · TPR@FPR · CIs)             GCG suffix optimisation
idealized action-space attacker + frontier       LIBERO rollouts
cross-layer ΔASR tax + cluster bootstrap         (fill the SAME contract via Real* seams)
```
Every GPU piece sits behind a `Protocol` — `Dynamics`, `ActivationExtractor`, `PerplexityScorer`,
`Scorer`, `StateAdapter` — each with a `Synthetic*`/`Mock*` impl for tests. The stats layers
**never see a model**: identical `UnitOutcome` contract from synthetic or real rollouts.

## End-to-end data flow
```
config (pinned YAML) ── load_config ──┐
                                       ▼
[GPU] rollouts ─► Rollout(steps) ─► scorer.score_rollout ─► [Score/step]
                                                              │
              calibrate(benign_calib) ─► Threshold(tau)       │  invariant #3: calib ≠ test
                                                              ▼
              rollout_fires(scores, tau) ─► Decision(hold)
                                                              │
  run_condition_matrix ─► ResultsTable ─► RunLogger (write-once results/) ─► make_figures
```

## Headline split (claim boundary — never conflate)
- **H6-A** oracle intrinsic action-space frontier — M3, **guaranteed floor (every branch)**.
  `attack/idealized_frontier.trace_frontier` → (ASR, evasion) Pareto.
- **H6-D** deployable-vs-deployable cross-layer **tax** — M4, headline-if-affordable.
  `eval/cross_layer.bootstrap_delta_asr` → ΔASR@evasion + percentile CI.
- Compute **Branch N / N− / F** selected at the M1 on-GPU timing micro-bench (`scripts/microbench_gcg.py`).

## Milestones
M0 design-lock (exiting) · M1 env+viability+compute-branch gate · M2 floor detection (L0+L1+L2-oracle) ·
**M3 H6-A oracle frontier** · M4 H6-D deployable tax (branch-selected) · M5 ladder/SABER (stretch).


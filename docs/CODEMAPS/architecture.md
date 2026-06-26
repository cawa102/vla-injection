<!-- Generated: 2026-06-26 (full regen) | Files scanned: 52 src · 24 scripts | Token estimate: ~880 -->

# Architecture — Embodiment Evasion Tax

MSc research code, **not** an app: no HTTP API, no UI, no DB. A Python package `src/evasion_tax`
(model-free core, unit-tested on the mac) + a CLI in `scripts/` (GPU runs on the CSB box).
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

## Two pipelines
### 1. Detection / tax measurement (model-free core)
```
config (pinned YAML) ── load_config ──┐
                                       ▼
Rollout(steps) ─► scorer.score_rollout ─► [Score/step]
                                          │
   calibrate(benign_calib) ─► Threshold(tau)   │  invariant #3: calib ≠ test
                                          ▼
   rollout_fires(scores, tau) ─► Decision(hold)
                                          │
  run_condition_matrix ─► ResultsTable ─► RunLogger (write-once results/) ─► make_figures
  eval/cross_layer.bootstrap_delta_asr ─► ΔASR@evasion tax (+cluster-bootstrap CI)
```
### 2. Quantized-surrogate RoboGCG transfer (the active GPU work, CSB A5000)
```
[GPU] load_openvla_policy(int8/nf4/bf16) ─► OpenVlaGcgTarget ─► run_gcg (suffix search)
        │  gradient-health diagnostic recorded (record, never gate)
        ▼
   SurrogateSuffixArtifact ── suffix payload → quarantined artifacts/untrusted/ ──┐
        │  metrics → results/ via _results_pointer (no suffix)                      ▼
   evaluate_surrogate_transfer (bf16 victim) ─► TransferEvalRecord ─► summarize_transfer
        (target-action ASR · transfer gap · GPU-hour-normalized ASR · censoring)
```

## Model-free / GPU boundary (the core design split)
```
MODEL-FREE (mac · 608 tests green)            GPU NODE (CSB ecs3-0202 = 2× A5000 24GB · ACTIVE)
──────────────────────────────────────       ─────────────────────────────────────────────────
metric A scorer · FP-calibrated detector      OpenVLA-7B inference (bf16/int8/nf4 surrogate)
LiberoStateAdapter (real BDDL ground truth)   RoboGCG suffix optimisation (attack/gcg_openvla)
eval stats (ROC/AUC · TPR@FPR · CIs)          LIBERO policy rollouts (eval/rollout_runner)
idealized action-space attacker + frontier    (fill the SAME contract via Real*/OpenVla* seams)
cross-layer ΔASR tax + cluster bootstrap      Kelvin2 A100/H100 = registered backup only
```
Every GPU piece sits behind a `Protocol` / seam — `Dynamics`, `ActivationExtractor`,
`PerplexityScorer`, `Scorer`, `StateAdapter`, `LossGradientFn` — each with a `Synthetic*`/`Mock*`
test impl. The stats layers **never see a model**: identical `UnitOutcome` contract from synthetic
or real rollouts.

## Headline split (claim boundary — never conflate)
- **H6-A** oracle intrinsic action-space frontier — M3, **guaranteed floor (every branch)**.
  `attack/idealized_frontier.trace_frontier` → (ASR, evasion) Pareto.
- **H6-D** deployable-vs-deployable cross-layer **tax** — M4, headline-if-affordable.
  `eval/cross_layer.bootstrap_delta_asr` → ΔASR@evasion + percentile CI.
- Compute **Branch N / N− / F** selected at the M1 timing micro-bench (`scripts/microbench_gcg.py`,
  `eval/branch_select.provisional_branch`); A5000 measured ~6.2 h per 500-step arm → expect N−/F.

## Milestones
M0 design-lock · M1 env+viability+compute-branch gate (`eval/m1_gate`) · M2 floor detection
(L0+L1+L2-oracle) · **M3 H6-A oracle frontier** · M4 H6-D deployable tax (branch-selected) ·
M5 ladder/SABER (stretch).

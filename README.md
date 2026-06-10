# The Embodiment Evasion Tax

> MSc Cyber Security & AI individual research project — **VLA-model security, integrity focus**.
> Private repository for the author, supervisor, and collaborators. Single-author dissertation:
> reproducibility and defensible claims take precedence over speed.

**One line.** Measure the *per-layer adaptive-evasion cost* of detecting instruction-injection attacks on a
Vision-Language-Action (VLA) policy — i.e. how expensive it is for an attacker to slip past detectors that read
the perception → reasoning → action pipeline at three different depths.

This is a **measurement study, not a claimed universal defence.** It is publishable on a positive result (a
usable, false-positive-calibrated operating point exists) *or* a negative one (it does not — and why).

---

## What it studies

Three detection layers, ordered by where they observe the pipeline, evaluated against a real adversarial attack
(RoboGCG) on base **OpenVLA-7B** (bf16), **simulation only (LIBERO)**, **instruction channel**:

| Layer | Reads | Instrument |
|-------|-------|------------|
| **L0** input | text / perplexity filter | `baselines/perplexity.py` |
| **L1** internal | activation-delta probe | `metric/probe_internal.py` |
| **L2** behavioural | goal-action consistency vs a *trusted goal* | `metric/consistency_a.py` |

The FP-calibrated goal-action detector is the **L2 instrument** of the measurement — not a universal defence.

### Headline claim and its boundary (do not conflate)

The cross-layer "tax" is **two distinct claims**:

- **H6-A — oracle intrinsic action-space frontier** (milestone **M3**). The *guaranteed floor*, delivered in
  every compute branch. An L2-**oracle** (privileged) frontier with non-adaptive L0/L1/L2 detection. It makes
  **no cross-layer "tax" claim**. `attack/idealized_frontier.py`.
- **H6-D — deployable-vs-deployable cross-layer tax** (milestone **M4**). The *headline if affordable*. A
  matched realistic attacker/budget across all three layers — the only claim that supports "L2 costs more than
  L1". `eval/cross_layer.py`.

Which one becomes the committed headline is decided at the **M1 on-GPU timing micro-bench**
(`scripts/microbench_gcg.py`), which selects compute **Branch N / N− / F**. H6-A is delivered either way; the
tax headline belongs to H6-D, never H6-A.

---

## Current status

**Phase: Design → M0 exiting.** The model-free local components are implemented and unit-tested; all
GPU-dependent runs (OpenVLA inference, GCG optimisation, LIBERO rollouts) are deferred to the granted A100/H100
node.

```
MODEL-FREE (local, in this repo · 395 tests green)   GPU NODE (deferred to A100/H100)
──────────────────────────────────────────────────  ──────────────────────────────────
metric A scorer · FP-calibrated detector             OpenVLA-7B inference
eval stats (ROC/AUC · TPR@FPR · confidence intervals) GCG suffix optimisation
idealized action-space attacker + frontier           LIBERO rollouts
cross-layer ΔASR tax + cluster bootstrap              (fill the SAME contract via Real* seams)
```

Every GPU piece sits behind a Python `Protocol` (`Dynamics`, `ActivationExtractor`, `PerplexityScorer`,
`Scorer`, `StateAdapter`) with a `Synthetic*`/`Mock*` implementation for tests. The statistics layers never see
a model: they consume an identical `UnitOutcome` contract whether the rollout was synthetic or real.

**Milestones.** M0 design-lock (exiting) · M1 env + viability + compute-branch gate · M2 floor detection
(L0+L1+L2-oracle) · **M3 H6-A oracle frontier** · M4 H6-D deployable tax (branch-selected) · M5 ladder/SABER
(stretch).

---

## Repository layout

```
src/evasion_tax/        Python package — model-free core (28 modules, ~4.7k LOC)
  attack/        idealized action-space attacker + Pareto frontier
  baselines/     L0 perplexity / anomaly detectors
  metric/        L1 activation-delta probe · L2 goal-action consistency (A)
  detector/      FP calibration + decision rule
  eval/          ROC/AUC, TPR@FPR, cross-layer ΔASR, bootstrap CIs, figures
  policy/        OpenVLA action codec / stats (model-free helpers)
  repro/         seeds, env capture, provenance, write-once run logger
  config/        pinned-YAML schema + runtime loader
tests/         395 unit tests (synthetic/mock seams, no model needed)
scripts/       thin CLI for GPU-node runs (run_benign, run_attack, calibrate, evaluate, microbench_gcg, …)
configs/       pinned experiment configs (example_m2.yaml) + GPU requirements
docs/          research documents — see Documentation below
results/        write-once run logs (created on first GPU run; gitignored when populated)
```

A token-lean architecture map lives in [`docs/CODEMAPS/`](docs/CODEMAPS/README.md) — read that for
per-package public symbols, the data contract, and the dependency graph.

---

## Documentation

Source-of-truth ordering (follow this when documents appear to disagree):

| You want… | Read |
|-----------|------|
| **What & why** (threat model, novelty, design rationale) | [`docs/core/goal-action-consistency-detector.md`](docs/core/goal-action-consistency-detector.md) |
| **Status / tasks / decisions / how-to** (operational source of truth) | [`docs/core/execution-playbook.md`](docs/core/execution-playbook.md) |
| **Verified external facts** (papers, provenance, hashes) | [`docs/references/README.md`](docs/references/README.md) |
| **Code structure** | [`docs/CODEMAPS/`](docs/CODEMAPS/README.md) |
| **Literature landscape** | [`docs/lit-review/`](docs/lit-review/) |
| **GPU node setup / runbook** | [`docs/setup/gpu-runbook.md`](docs/setup/gpu-runbook.md), [`docs/gpu/`](docs/gpu/) |

A short non-technical explainer for the talk is in [`docs/presentation/`](docs/presentation/).

---

## Running the model-free components

Local environment is Python **3.11** (source kept 3.10-compatible for the GPU stack). Dependencies are managed
with [`uv`](https://docs.astral.sh/uv/) (`uv.lock` pinned).

```bash
uv sync                 # install pinned dependencies into .venv
uv run pytest           # run the 395 model-free unit tests
```

The heavy simulation stack (`mujoco`, `robosuite`, and `libero` from source) is an isolated optional extra so
it can never break the core environment — see `pyproject.toml` `[project.optional-dependencies].libero` and
`configs/env/requirements-gpu.txt`. GPU runs are driven from `scripts/` on the node, not locally.

---

## Reproducibility & ethics

**Reproducibility (non-negotiable).** Seeds are pinned and recorded; the exact environment is captured per run;
data/checkpoint provenance (source, hash, date, licence) is logged; each run is written to a timestamped
**write-once** `results/`; experiments change one variable at a time; figures are regenerable from logged data
by a script; negative results are reported, not dropped. The `src/evasion_tax/repro/` package implements these
invariants.

**Security-research ethics.** Attack code (instruction injection, action-space manipulation) is built only for
**contained, authorised, defensive** evaluation in simulation. Untrusted artefacts (poisoned data, trojaned
checkpoints) are quarantined under `artifacts/untrusted/` and never auto-run. Work follows the institution's
ethics process.

**Not committed:** datasets, model checkpoints, secrets, PII, third-party binaries, or paper PDFs — provenance
for these is recorded in `docs/` instead (`docs/references/README.md`).

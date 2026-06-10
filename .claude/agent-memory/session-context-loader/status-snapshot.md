---
name: status-snapshot
description: Embodiment Evasion Tax plan-vs-code status — entire model-free pre-GPU build (local-prep Tasks 0-11 + ALL §4b instruments) is DONE; only remaining gate is GPU (Kelvin2) access. Re-verify against git/playbook before acting.
metadata:
  type: project
---

Snapshot re-verified 2026-06-08 against git + playbook §1. For *current* state always prefer `git log` + the playbook §1 over this note.

**Phase:** Design → M0 exiting. Coding gate **lifted for model-free M1-M2 work** (author OK 2026-05-31). OpenVLA/GCG/LIBERO *runs* await the granted GPU = **Kelvin2** (NI-HPC@QUB; partitions `k2-gpu-a100`/`k2-gpu-h100`, 3-day walltime cap, shared/queued; login not yet established).

**RUNNING THE TESTS (recurring onboarding trap):** the repo code uses PEP 604/585 syntax requiring **Python 3.10+**. The repo has a `.venv` = Python 3.11.14 — **always run via `.venv/bin/python -m pytest`**. The machine's default `python` is anaconda **3.8.5**, which fails collection with `'ABCMeta' object is not subscriptable` / `unsupported operand type` — that is an interpreter mismatch, NOT a code regression. Verified 2026-06-08: **381 tests pass** under `.venv` (count grew from 362 as the code-review fixes below added tests).

**Latest session = the 2026-06-08 code↔goal consistency review cycle** (`docs/dynamic-workflow/2026-06-08-code-review.md`, commit `bb66a8a`): a 166-agent dynamic-workflow audit of all 153 src functions — 0 FAIL; findings were cross-file *wiring gaps* (rails implemented + unit-tested but not invoked by the single reporting path). Fixes applied across the 4 commits before it:
- `3bb9cb0` wired H1 (power gate `annotate_operating_points` into harness reporting) + H2 (`assert_disjoint` into `scripts/evaluate.py`) + honest cost-metric deferral.
- `21ce9e0` closed H3 (surface coverage-excluded scenarios) + H5 (target-action-blocked rate in cross_layer + records).
- `597e3d9` closed M1 (metric-A surfaces a runtime abstain instead of silent 0.0).
- `52d9c73` closed concerns C1 (torch driver-version API in env_capture) + C2 (bound abort_rate inputs).
- `a1b425b` closed L2 (stale "unsafe-action-blocked" doc text) + documented H4/M2/M3 as explicit GPU-phase deferrals (target-action-blocked framing).
Still-open review items are the *(decision needed)* GPU-phase deferrals — H4 (detection-latency/abort/degradation have no model-free production caller; harness hard-codes empty latency), M2 (`predicate_for_target` not wired into a runnable frontier driver), M3 (no driving script for the H6-A idealized-attacker→cross-layer-tax pipeline) — intentionally deferred to the GPU node, now tracked in the playbook.

**Built — the ENTIRE model-free pre-GPU track is COMPLETE (local-prep Tasks 0-11 ✅ + all §4b instruments ✅):**
- Floor (Tasks 0-9): env scaffold, repro infra, data records, action codec (OpenVLA formula verified from source `c8f03f48`), privileged-state adapter, metric A (schema FROZEN, causal scorer) = the L2-oracle, FP-calibrated detector, eval harness (ROC/AUC, TPR@FPR + Wilson/CP CIs, held-out realised_fpr, latency, split-disjointness), baselines (`baselines/anomaly.py` χ²-OOD + `baselines/perplexity.py` = L0), config/scripts/figures.
- §4b-I L1 internal probe: `metric/probe_internal.py` (InternalProbe activation-delta logistic probe + ActivationExtractor seam) + `metric/probe_confounds.py` (`fadc008`).
- §4b-II idealized action-space attacker: `src/evasion_tax/attack/` EXISTS — `dynamics.py` (Synthetic + RealDynamics GPU stub), `frontier.py` (Pareto geometry + `asr_at_evasion`), `idealized_frontier.py` (random-shooting optimiser + `trace_frontier`, coverage gate) (`ac4f229`).
- §4b-III cross-layer eval: `src/evasion_tax/eval/cross_layer.py` EXISTS — `UnitOutcome`, `delta_asr_at_evasion` (#10 primary tax scalar), `bootstrap_delta_asr` cluster CI, `comparative_asr_table`.
- Codex-#2 hooks: `metric/coverage.py` (#6 coverage manifest), `eval/power.py` (#3 power/sample-size rule). All committed by 2026-06-03.

**Not yet built / remaining — ALL gated on GPU (Kelvin2) access:**
- M1 on GPU: stand up OpenVLA-7B bf16; benign LIBERO baseline; RoboGCG targeted-redirect repro; **GCG + L1-extraction micro-bench → resolves D4/D7 and selects compute Branch N/N−/F (D8)**; metric-A signal check → **GO/NO-GO gate (H1)**. Runbook = `docs/setup/gpu-runbook.md`.
- Concrete `ActivationExtractor`/LIBERO `StateAdapter` impls (deferred behind thin seams to the GPU node); attention-map MLP probe ablation; deployable L2 (B or C, chosen at M2).
- No `results/` dir yet (no experimental runs have happened — expected).

**Recent sessions (2026-06-05 → 06-07) were all housekeeping, no new instruments:** repo README, per-area codemaps + architecture code tour, **T7 codename fully purged** from code/docs (use "Embodiment Evasion Tax"/"EET"), src/ comment fixes, `_stable_seed` dedup into `evasion_tax.repro`, AGENTS.md + `.codex/` Codex agent config, and a `graphify-out/` knowledge-graph build (gitignore-tracked).

**Open decisions / sign-offs:**
- Supervisor sign-off PENDING on the whole Embodiment-Evasion-Tax reframe + locked title (author-converged 2026-06-01).
- D4, D7 OPEN → resolved at the M1 on-GPU GCG micro-bench.
- D8 compute: A100/H100 granted; the M1 on-GPU timing micro-bench selects Branch N (full deployable tax / H6-D headline) / N− (scoped) / F (oracle frontier only / H6-A headline + fallback title). Pre-registered before numbers are in.
- H6-A (M3, oracle intrinsic frontier) = committed floor, delivered in every branch. H6-D (M4, deployable cross-layer tax) = committed-only-if-affordable. Never conflate the two; the "tax" headline is M4's, never M3's.

**Known recurring discrepancy:** 3 pre-existing pyright errors + 1 ruff B905 live in untouched *test* files (`test_state.py`, `test_records.py`, `test_consistency_a.py`) — pre-Task-9, not yet cleaned. src/ itself is clean.

See [[project-map]] for layout and SoT docs.

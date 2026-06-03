---
name: status-snapshot
description: T7 plan-vs-code status as of 2026-06-03 — what is built (incl. L1 probe now landed), what is gated, open decisions and sign-offs. Re-verify against git/playbook before acting.
metadata:
  type: project
---

Snapshot frozen 2026-06-03. For *current* state always prefer `git log` + the playbook §1 over this note.

**Phase:** Design → M0 exiting. Coding gate **lifted for model-free M1-M2 work** (author OK 2026-05-31). OpenVLA/GCG/LIBERO *runs* await the granted GPU (A100/H100; single-card-vs-cluster + queue depth TBC).

**Built (local-prep Tasks 0-9 ✅):** env scaffold, repro infra, data records, action codec (OpenVLA formula verified from source `c8f03f48`), privileged-state adapter, metric A (schema FROZEN, causal scorer) = the L2-oracle, FP-calibrated detector, eval harness (ROC/AUC, TPR@FPR + Wilson/CP CIs, latency, split-disjointness), baselines (χ²-OOD anomaly + perplexity = L0), config/scripts/figures. ~234-237 tests green, `src/t7` type-clean.

**§4b instrument progress (2026-06-03):** L1 internal-probe arm (§4b-I) now LANDED — `src/t7/metric/probe_internal.py` (InternalProbe activation-delta logistic probe + ActivationExtractor Protocol seam; Synthetic + Real-GPU-stub) and `src/t7/metric/probe_confounds.py` (#11 label-shuffle + probe_auc) committed `fadc008`; 265 tests green, ruff+pyright clean. Attention-map MLP ablation deferred to M2/GPU.

**Not yet built:**
- Local-prep Task 10 (LIBERO state-only smoke, optional/time-boxed — needs author), Task 11 (GPU runbook — delegable).
- §4b-II idealized action-space attacker: `src/t7/attack/idealized_frontier.py` — **`src/t7/attack/` dir confirmed absent**.
- §4b-III cross-layer eval + tax metrics in `src/t7/eval/` — **no tax/cross-layer/pareto/frontier symbols in src/t7/eval/ yet**. Includes Codex-#2 hooks #6 coverage-manifest stub and #10 ΔASR@fixed-evasion primary tax scalar.

**Open decisions / sign-offs:**
- Supervisor sign-off PENDING on the whole Embodiment-Evasion-Tax reframe + locked title (author-converged 2026-06-01).
- D4, D7 OPEN → resolved at the M1 on-GPU GCG micro-bench.
- D8 compute: A100/H100 granted; the M1 on-GPU timing micro-bench selects Branch N (full deployable tax / H6-D headline) / N− (scoped) / F (oracle frontier only / H6-A headline + fallback title). Pre-registered before numbers are in.
- H6-A (M3, oracle intrinsic frontier) = committed floor, delivered in every branch. H6-D (M4, deployable cross-layer tax) = committed-only-if-affordable. Never conflate the two; the "tax" headline is M4's, never M3's.

**Known recurring discrepancy:** 3 pre-existing pyright errors + 1 ruff B905 live in untouched *test* files (`test_state.py`, `test_records.py`, `test_consistency_a.py`) — pre-Task-9, not yet cleaned. src/ itself is clean.

See [[project-map]] for layout and SoT docs.

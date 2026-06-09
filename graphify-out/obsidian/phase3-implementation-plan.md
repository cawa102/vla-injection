---
source_file: "docs/plans/phase3-implementation-plan.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# phase3-implementation-plan.md

## Connections
- [[Causal prefix window detection]] - `defined_in` [EXTRACTED]
- [[Consistency metric (A) privileged-state oracle]] - `defined_in` [EXTRACTED]
- [[D2 fallback (denial-only → task-deviation reframe)]] - `defined_in` [EXTRACTED]
- [[Frozen annotation schema (pre-attack)]] - `defined_in` [EXTRACTED]
- [[M1 environment + viability GONO-GO gate (H1)]] - `defined_in` [EXTRACTED]
- [[Phase-3 Implementation Plan (M1–M5)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/plans/phase3-implementation-plan.md`

# Embodiment Evasion Tax — Phase-3 Implementation Plan (M1–M2 detailed · M3–M5 sketched)

> Companion to the **execution playbook** (`execution-playbook.md`) and the **understanding doc**
> (`goal-action-consistency-detector.md`). Built on the **§6 design decisions** (resolved 2026-05-31).
> This plan **lifts the "no experiment code before an agreed plan" gate for M1–M2** (CLAUDE.md) once the
> author OKs it. M3–M5 are sketched and get their own detailed spec **after** the M2 results (B/C emphasis)
> and the M1 gate (semantic-redirect arm).
>
> ⚠️ AI-assisted scaffold for author review (CLAUDE.md §5). Provisional/gated items are marked.

---

## 1. Scope of this plan

- **Detailed (build now, after author OK):** **M1** (environment + viability gate), **M2** (floor detector
  metric (A) + FP-calibration).
- **Sketched (refined after M2 / M1 gate):** **M3** (reference ladder), **M4** (deployable B/C — the
  **committed primary novelty**), **M5** (adaptive attacker — **stretch**, only if M4 done with slack).
  *⚠️ Milestone labels here predate the 2026-06-01 reframe — authoritative mapping is **playbook §2** (M3 =
  oracle intrinsic frontier / H6-A; M4 = deployable + realistic adaptive / H6-D; M5 = ladder + SABER). See §10
  banner.*
- **Explicitly out of scope:** real-robot anything; metric (D) VLM-judge; threat models with no separable
  trusted reference (understanding-doc §3).

---

## 2. Runtime pipeline (what the system does)

```
trusted goal (D3 rung) ───────────────────────────────────────────────────┐
                                                                           ▼
LIBERO scene ─► OpenVLA-7B policy ─► prefix window a_{t-k+1:t} ─► metric s(o, a, g, [state]) ─► detector(τ) ─► allow / hold a_t
                       ▲
        (attack) clean instruction + RoboGCG suffix
```

- **Causal by construction:** OpenVLA emits one action per step, so at decision time `t` the detector sees
  only the **prefix** `a_{t-k+1:t}` (past actions + the candidate `a_t`), never the future. It decides whether
  to execute `a_t`. **Detection latency** (steps before the hold fires) is therefore a first-class metric. A
  **non-causal full-window** pass is reported separately as a **post-hoc monitoring ceiling** (labelled).
- **Benign run:** policy executes the clean instruction; detector should **not** fire → measures the
  **per-rollout false-abort rate** (primary FPR; per-window is auxiliary).
- **Attacked run:** instruction = clean + RoboGCG suffix; policy redirected → detector should fire → **TPR**
  (per-rollout, with CIs).
- The **trusted goal** feeding the metric is separate from the (possibly tampered) operational instruction —
  this separability is the threat model (understanding-doc §3); the D3 rungs vary how coarse it is.

---

## 3. Proposed repo layout (create a dir only when first needed)

```
src/evasion_tax/
  envs/      LIBERO setup + deterministic rollout runner
  policy/    OpenVLA-7B load (bf16 — A100/H100-80GB fit OpenVLA-7B at full precision; 4-bit no longer required), action decode, un-normalization
  attack/    RoboGCG reproduction (target defn, GCG optimize) → suffixes quarantined
  metric/    consistency metric (A) privileged-state; (B)/(C) added at M4
  detector/  calibration (τ @ target FPR), decision + fallback
  eval/      condition matrix, splits, metrics (ROC, AUC, TPR@FPR), aggregation
  repro/     seed control, env capture, provenance, write-once run logging
configs/     pinned YAML (model, env, attack, metric, detector, eval)
scripts/     run_benign · run_attack · microbench_gcg · calibrate · evaluate · make_figures
results/     write-once, timestamped run dirs (never overwritten)
artifacts/untrusted/   GCG suffixes + any adversarial artefacts (gitignored, never auto-run)
```

Files small and feature-organized (coding-style: 200–400 lines typical). Run records immutable.

---

## 4. Component contracts (interfaces, not code)

- **Rollout record** (core logged object, one per step): `{run_id, seed, git_commit, suite, task_id, step,
  observation_ref, action[7], privileged_state, instruction, trusted_goal, attacked:bool, suffix_ref}`.
- **Policy runner** — load OpenVLA-7B bf16; `(observation, instruction) → action`; record the normalization
  stats used; deterministic given seed.
- **Attack (RoboGCG)** — `(task, target_action_spec) → suffix`; record steps, wall-time, success(=reached
  target). Reuse the authors' released method (`github.com/eliotjones1/robogcg`) where feasible; **attribute**;
  **quarantine** outputs under `artifacts/untrusted/`. Target spec per **D2** (low-level action region).
- **Consistency metric (A)** — `s(observation_t, prefix_window a_{t-k+1:t}, trusted_goal, privileged_state_t)
  → score∈[0,1]` (higher = more inconsistent), over a **causal prefix window** (no future actions). (A)
  derives action semantics from LIBERO ground truth (object approached, gripper open/close, target region
  reached) and compares to the goal's expected semantics. **The annotation schema is frozen *before* any
  attack output is inspected** (no rules added post-hoc) and **unit-tested per task** — else the parser
  silently becomes the detector (circularity). Window length `k` is a config param. *(A) uses privileged info
  → upper bound, labelled non-deployable; the full-window variant is a monitoring ceiling only.*
- **Detector / calibration** — `calibrate(benign_scores_calib, target_per_rollout_fpr) → τ`; `decide(score,
  τ) → {allow a_t, hold}` (decision *before* executing `a_t`). Primary operating points = **per-rollout**
  false-abort {1%, 5%} (per-window auxiliary); report TPR/FPR with **Wilson/Clopper-Pearson CIs** and
  **detection latency**. The **same `calibrate`** is applied to every baseline (fair comparison).
- **Eval harness** — runs the condition matrix over **disjoint** calibration/test splits (by
  task/scene/seed); computes ROC, AUC, TPR@FPR, benign task-success degradation, detection rate, abort rate,
  latency; emits a results table + the data arrays for `make_figures`.
- **Repro** — every run stamps seed, env (pip/conda + CUDA/driver/torch), git commit, provenance; **refuses
  to overwrite** an existing results dir.

---

## 5. Data & checkpoints (record provenance in `docs/references/`)

- **LIBERO suites:** Spatial, Object, Goal (core); -10 optional. (Data gitignored.)
- **OpenVLA-7B LIBERO-finetuned checkpoints:** record source, **hash**, date, licence (like the RoboGCG
  entry) **before** use; record the precision (bf16) config.
- **Splits:** define disjoint calibration vs test by held-out **tasks / scenes / seeds**; commit the split
  manifest. The harness asserts disjointness at runtime (no calibration↔test leakage).

---

## 6. M1 — environment + viability gate (build this first)

1. Stand up OpenVLA-7B (bf16) on the A100/H100; load a LIBERO checkpoint; record env + provenance.
2. **Benign baseline:** N benign rollouts per suite (pinned seeds); record task-success; sanity-check vs
   OpenVLA/LIBERO published numbers.
3. **RoboGCG reproduction:** on a few tasks, optimize a suffix for a chosen target action; run the attacked
   rollout; record whether the policy **reaches the target action region** and the **persistence** (#steps).
4. **GCG micro-benchmark:** measure s/target and #steps on the granted A100/H100 at bf16 → **resolves D4 (eval
   matrix) and D7 (budget)**. Record actuals.
5. **Signal sanity (metric A):** compute `s(...)` on a handful of benign vs attacked rollouts (causal prefix
   window); inspect the benign-vs-attacked separation **at *both* the clean-instruction ceiling and the
   coarse operator-goal reference**.

**GO/NO-GO gate (H1):** benign reproduced **＋** RoboGCG *targeted* redirect confirmed **＋** visible
benign-vs-attacked separation **that survives at the coarse operator-goal reference** (not the
clean-instruction ceiling alone — else detection *necessity* is weak; flag before deep work).
- If **denial-only** (no coherent targeted redirect) → invoke the **D2 fallback**: reframe to
  *goal-abandonment / task-deviation* detection; drop the semantic-redirect claim.
- Per the **D2 rule**, decide here whether to **add the semantic-redirect attack arm** for M3–M5.

---

## 7. M2 — floor detector (A) + FP-calibration

1. Implement metric (A) over a **causal prefix window**; choose `k` (config; sensitivity-check a few values —
   **one variable at a time**); **freeze the annotation schema before inspecting attacks**; unit-test per task.
2. Generate benign + attacked rollout sets across the provisional matrix (§9), on **calibration + test** splits.
3. Calibrate τ on the **calibration** split for **per-rollout** false-abort ∈ {1%, 5%}; record τ. Apply the
   **same calibration** to every baseline.
4. Evaluate on the **held-out test** split: ROC/AUC, **per-rollout** TPR@FPR **with CIs**, benign
   task-success degradation, per-rollout false-abort rate, **detection latency**.
5. Baselines (same protocol): **perplexity / text-only** filter + **mandatory goal-agnostic anomaly**
   baseline; compare fairly.

**Exit (H2 / FLOOR SECURED):** a usable operating point with reported TPR@FPR vs baselines (**positive**)
**OR** a clean **negative** (no τ separates without high benign cost) — either logged to write-once
`results/`. Both are publishable.

---

## 8. Configs, seeds, one-variable discipline

- One pinned YAML per run; seeds pinned **and recorded**; any comparison run changes **exactly one** field vs
  its baseline config; the run protocol (playbook §8) is pasted into each run log.

---

## 9. Provisional eval matrix (D4 — finalize at M1)

| Axis | Provisional value | Fixed when |
|------|-------------------|-----------|
| Suites | LIBERO-Spatial / -Object / -Goal (core); -10 optional | M1 |
| Tasks | ~5–10 per suite | M1 |
| Targets | ~20–50 per task (subsampled from RoboGCG's 1792) | **M1 micro-benchmark** |
| Seeds | ≥3 pinned | M1 |
| Splits | disjoint by task/scene/seed | M0 (this plan) |
| Operating points | **per-rollout** false-abort: **5% primary**, **1% exploratory unless held-out benign N ≥ ~300** (+ Wilson/Clopper-Pearson CIs); pre-register benign-N per FPR claim (Codex #2 #3) | DECIDED (D6) + #3 |

---

## 10. M3–M5 (sketch — detailed after M2 / M1 gate)

> ⚠️ **SUPERSEDED (2026-06-01 Evasion-Tax reframe + 2026-06-02 Codex review #2).** The milestone **labels** in
> this section predate the reframe. The **authoritative M3–M5 mapping is now playbook §2**: **M3** = oracle
> intrinsic action-space frontier (**H6-A**), **M4** = deployable B-or-C + realistic adaptive cross-layer tax
> (**H6-D**), **M5** = reference-coarsening ladder + SABER. This plan's **M1–M2 component contracts remain
> valid**; the M3–M5 *sketch* below is kept for history only — do not action it.

- **M3 reference ladder (N):** author coarse operator-goal + task-ID→goal mappings per suite (small,
  version-controlled); re-run the M2 eval per rung; report TPR@FPR across rungs (**H3**).
- **M4 deployable (N — committed, primary novelty):** **(B)** small model `(observation, prefix_window) →
  goal embedding` = planned primary; **(C)** reference-policy divergence (pairs with the coarse-goal rung) =
  complement. **Must specify: supervision labels, negative-pair construction, and train/test task-disjoint
  splits** so the detector cannot just memorise a task prior. Calibrate + evaluate vs the (A) ceiling (**H4**).
  Final B/C emphasis set from the M2 signal.
- **M5 adaptive (N — stretch):** **only if M4 completes with slack.** Attack the **deployable B/C detector
  only** (metric (A) is privileged & non-differentiable — not an attack target); fix white-box vs score-query
  black-box and a **query/compute budget + #attempts cap**; measure detection vs adaptive ASR + attacker cost
  (**H5**). If not reached, claim *attacker-aware evaluation* (we tested a real attack) but **not** adaptive
  robustness.

---

## 11. Implementation risks

| Risk | Mitigation |
|------|-----------|
| Attack compute overrun on the granted A100/H100 | published H100 GCG timings should ≈ transfer; still micro-benchmark on the actual granted HW at M1; subsample targets (D7) |
| RoboGCG repro fidelity on our checkpoint | confirm at M1; attribute the method; do **not** claim a new attack |
| Denial-only attack (not targeted) | D2 fallback → reframe metric to task-deviation detection |
| Privileged-state extraction wrong (A) | **unit-test** extraction against known LIBERO ground truth before trusting scores |
| Calibration↔test leakage | harness **asserts** disjoint splits at runtime |
| **Non-causal detector** (future-window leak) | detect on a **causal prefix window**; report **detection latency**; full-window only as a post-hoc ceiling |
| **FPR unit ambiguity** | **per-rollout** false-abort is primary (+CIs); per-window auxiliary |
| **metric(A) parser becomes the detector** (circular) | **freeze annotation schema before attack inspection**; per-task unit tests + ablation |
| **Unfair perplexity baseline** | give baselines the **same calibration protocol**; the deployment-threshold critique is stated separately |
| Over-claiming | every reported result → a row in playbook §9 claims ledger |

---

## 12. Definition of done — Phase-3 (M1–M2)

Environment reproducible; benign baseline + RoboGCG **targeted** redirect logged; **D4/D7 fixed**; metric (A)
signal shown **(survives at the coarse-goal reference, not just the clean-instruction ceiling)**; floor
detector (**causal prefix window**) calibrated + evaluated on a **held-out** split with **per-rollout**
TPR@FPR (+CIs), **detection latency**, and **fair-calibrated** baselines; all runs in **write-once**
`results/` with provenance; figures regenerable by script. → **unlocks the M3–M5 detailed spec.**


# T7 — Execution Playbook (operational companion)

> **What this document is.** The *operational* companion to the *understanding* doc
> [`t7-goal-action-consistency-detector.md`](./t7-goal-action-consistency-detector.md). That doc explains
> **what** T7 is and **why** (threat model, novelty, design rationale). **This** doc keeps the *execution* on
> track: it tells whoever is working (esp. Claude Code across sessions) **where we are**, **what's next**,
> **what's been decided**, and **how to run the work without violating the project's reproducibility / ethics
> rules**. It is a **living document** — update it as work proceeds (see §11 Session Protocol).
>
> **Source of truth ordering.** Understanding/rationale → the understanding doc. Status/tasks/decisions/how-to
> → **this doc**. Verified external facts → [`../references/README.md`](../references/README.md).
>
> ⚠️ AI-assisted scaffold for the author to review (CLAUDE.md §5). Plan choices below marked **PROPOSED** need
> author/supervisor sign-off at the milestone noted; **DECIDED** items are settled; **OPEN** items are blocked
> on a measurement.

---

## 0. North Star (if you read nothing else, read this)

**Goal (one line).** Measure whether instruction-injection attacks on a VLA can be detected by checking
*goal-action consistency* against a **trusted goal**, at a **calibrated, a-priori-settable false-positive
budget** — and report where it works and where it breaks.

**This is a *measurement* study, not a new universal defense.** Publishable on a positive (a usable operating
point exists) **or** a negative (it does not — and *why*). Do **not** over-claim "a new defense."

**The actual aim is the novelty:** a **deployable**, FP-calibrated detector (no privileged info) evaluated
against an **actual adversarial attack** (RoboGCG) — this is the **committed** novelty (**M4**). Two senses of
*attacker-aware*: (a) *we test detection against a real attack*, unlike actalign's benign-only setting —
**committed**; (b) *robustness against an adaptive attacker that knows the detector* (**M5**) — a **stretch**
arm (author decision + Codex review, 2026-05-31), pursued only if M4 finishes with slack; if dropped we make
claim (a) and **do not** claim adaptive robustness. The privileged-state floor (below) guarantees a defensible
dissertation even if the deployable arm underperforms.

**Five non-negotiables (CLAUDE.md):**
1. **Reproducibility** — pin & record seeds; capture exact env; provenance for every checkpoint/dataset
   (source, hash, date, licence); log each run to a timestamped **write-once** `results/`; change **one
   variable at a time**; figures regenerable from logged data by a script; **report negative results**.
2. **Simulation only** — LIBERO; no real-robot transfer claims.
3. **Calibration honesty** — set τ on a calibration split, report FPR on a **held-out** split. Never set and
   report on the same rollouts.
4. **Academic integrity** — no fabricated citations (`[CITATION NEEDED]` until verified); distinguish
   "established result" from "my experiment showed"; attribute borrowed code/data; generated prose is a draft
   for the author to rewrite.
5. **Security-research ethics** — adversarial artefacts (GCG suffixes, any poisoned/trojaned files) are
   **quarantined under `artifacts/untrusted/`**; never auto-run untrusted checkpoints; follow ethics process.

**Phase order (never skip):** Scope → Lit review → **Design** → Implement → Run & analyse → Write up.

---

## 1. You Are Here  ← update this block every session

- **Last updated:** 2026-05-31
- **Phase:** **Design → M0 (exiting)**. Coding gate **lifted for M1–M2 (author OK, 2026-05-31)**; **pre-GB10 local build underway** — model-free M1–M2 components on M1/8 GB (`docs/plans/t7-local-prep-plan.md`). OpenVLA/GCG/LIBERO *runs* await GB10.
- **Last completed:** Theme = T7; understanding doc; **RoboGCG defence verified** (`docs/references/`); **D1–D7 resolved** (§6/§10; D4/D7 OPEN → M1); **Phase-3 implementation plan drafted**; **Codex third-party review incorporated** (causal prefix-window + detection latency; per-rollout FPR + CIs; metric(A) schema frozen; fair-calibrated baselines; M5→stretch — see §10).
- **Currently:** Executing the **pre-GB10 local-prep plan** (`docs/plans/t7-local-prep-plan.md`) on M1/8 GB — model-free M1–M2 engineering (repro infra, metric-(A) **schema freeze**, calibration, eval stats, action codec, baselines, configs/figures) via TDD + a time-boxed state-only LIBERO smoke test + GB10 runbook.
- **▶ NEXT ACTION:** **Local (now):** implement local-prep Tasks 0–11. **On GB10 (M1 milestone):** stand up OpenVLA-7B (4-bit) on LIBERO; reproduce benign baseline + RoboGCG *targeted* redirect; run the **GCG micro-benchmark** (fixes D4/D7); check metric (A) signal **incl. a coarse-goal separation check** → **GO/NO-GO gate**.
- **Blockers:** none.
- **Open decisions outstanding:** **D4, D7 only** (OPEN → M1 micro-benchmark). D1/D2/D3/D5/D6 DECIDED.
- **Floor secured?** ❌ not yet (target: end of **M2**, ~Jul 12).
- **Novelty status:** **M4 deployable detector = committed** (primary novelty); **M5 adaptive = stretch** (only if M4 done with slack).

---

## 2. Milestone roadmap (W1 = week of 2026-06-01; submit early Sep 2026)

**Tiers.** **F = Floor** (guaranteed deliverable; must finish). **N = Novelty** (the real goal; committed,
protected by gates — gates decide *how to adapt*, never *whether to drop*).

| ID | Tier | Milestone | Target weeks (dates) | Exit gate (verifiable) |
|----|------|-----------|----------------------|------------------------|
| **M0** | F | **Design lock** — resolve D1–D7 (§6), write Phase-3 implementation plan, freeze metric/reference/eval/baseline/metrics | W1–W2 (Jun 1–14) | Implementation plan agreed in `docs/plans/`; D1–D7 logged as DECIDED/OPEN-with-owner; **then** code allowed |
| **M1** | F | **Environment + viability gate** — stand up OpenVLA-7B (4-bit) on LIBERO; benign baseline; reproduce RoboGCG; GCG micro-benchmark on GB10; signal sanity-check with metric (A) | W2–W4 (Jun 8–28) | **GO/NO-GO:** benign baseline reproduced **＋** RoboGCG **targeted** redirect confirmed (not denial-only) **＋** visible benign-vs-attacked separation in (A) **that survives at the coarse operator-goal reference, not the clean-instruction ceiling alone** (seeds pinned). *If denial-only → reframe per understanding-doc §9; if separation only at the clean-instruction ceiling → detection necessity is weak, flag before deep work.* |
| **M2** | F | **Floor detector** — consistency metric (A) over action windows + **FP-calibrated** detector; clean-instruction (ceiling) + coarse operator-goal rung; perplexity baseline | W4–W6 (Jun 22–Jul 12) | **FLOOR SECURED:** reported TPR@fixed benign FPR (1%,5%) on a **held-out** split + ROC/AUC + benign task-success degradation; beats/▵ vs perplexity baseline characterised |
| **M3** | N | **Trusted-reference ladder** — measure detection+FPR across rungs (clean → coarse operator goal → task-ID→goal) | W6–W8 (Jul 6–26) | Ladder table: TPR@FPR per rung; "necessity" critique addressed (which rungs survive "why not just use it") |
| **M4** | **N — committed** | **Deployable detector (B and/or C)** — learned action-semantics map (B) and/or reference-policy divergence (C); no privileged info; **specify supervision labels, negative-pair construction, train/test task-disjoint (no task-prior leakage)**; calibrate+evaluate vs the (A) ceiling | W7–W10 (Jul 13–Aug 9) | Deployable **per-rollout** TPR@FPR (+CIs) reported **with honest gap to the (A) upper bound** |
| **M5** | **N — stretch** | **Adaptive attack (scoped)** — attack the **deployable B/C detector only** (fixed white/black-box choice **＋ query/compute budget ＋ #attempts cap**); measure robustness. *Run only if M4 completes with slack.* | W10–W11 (Aug 3–16) **if reached** | Adaptive per-rollout detection + **quantified attacker cost trade-off**; honest "holds vs breaks". *If not reached: claim (a) attacker-aware evaluation only, not adaptive robustness.* |
| **M6** | F | **Consolidation + ablations** — one-variable-at-a-time ablations; freeze all operating points; figures script-regenerable | W11–W12 (Aug 10–23) | **RESULTS FREEZE** in write-once `results/`; every figure regenerable from logged data by a script |
| **M7** | F | **Analysis + claims ledger** — map every claim→evidence; report negatives | W12–W13 (Aug 17–30) | §9 claims ledger complete; every claim traceable to a result file |
| **M8** | F | **Write-up** — dissertation draft (overlaps M7); author rewrites generated prose; verify all citations | W12–W14 (Aug 17–Sep 6) | Complete draft; **zero** `[CITATION NEEDED]` remaining |
| **M9** | F | **Polish + submit** — final review, reproducibility appendix, submission | W14–W15 (Aug 31–Sep 11) | Submitted early Sep 2026 |

**Critical path & protection of the novelty:** M0→M1→M2 secure the floor by ~Jul 12, leaving **~4 weeks
(W7–W10)** for the **committed** novelty (**M4** deployable detector) before the W11–W12 results freeze.
Protect M4: if M2 slips, compress M3 (run fewer rungs) **before** touching M4. **M5 (adaptive) is stretch** —
attempt only with genuine slack after M4. The only **hard kill** is the M1 GO/NO-GO (no signal at all). Later
gates *adapt scope* (drop M5, trim M3) but never abandon **M4**.

---

## 3. Hypothesis register (each maps to a milestone)

> Form: **statement** · *prediction if true* · **falsifier** · **decision rule**. Update **status** as
> evidence arrives. A falsified hypothesis is a **result to report**, not a failure to hide.

| ID | Milestone | Hypothesis | Prediction if true | Falsifier | Decision rule | Status |
|----|-----------|------------|--------------------|-----------|----------------|--------|
| **H1** | M1 | RoboGCG-injected rollouts yield action windows **measurably inconsistent** with the trusted goal, separable from benign by metric (A). | benign-vs-attacked score distributions separable (AUC ≫ 0.5). | AUC ≈ 0.5 (overlap). | High AUC → proceed to M2. Low → rethink metric / reframe to "task-abandonment detection" before investing. | ⬜ untested |
| **H2** | M2 | A threshold τ set on a **calibration split** gives **high TPR at a low *per-rollout* benign false-abort rate** *without* destroying benign task success — the **usable operating point RoboGCG's borrowed defences lacked**. | per-rollout TPR@{1%,5%} false-abort (with CIs) ≫ **fair-calibrated** perplexity baseline; benign task-success drop small; bounded detection latency. | No τ separates without large benign cost. | Usable point → floor secured, push novelty. None → **negative result** (behavioural detection also lacks a usable point under these conditions) — still publishable; report cleanly. | ⬜ untested |
| **H3** | M3 | Detection degrades **gracefully** as the trusted reference coarsens; still useful at deployment-realistic rungs. | ~monotone TPR@FPR decline; coarse-goal rung still > baseline. | Detection collapses to baseline once the reference is coarser than the operational instruction. | Survives → lead the deployment argument with coarse rungs. Collapses → report where/why the "necessity" critique bites (thesis backbone either way). | ⬜ untested |
| **H4** | M4 | A **deployable** metric (B/C, no privileged state) recovers a **substantial fraction** of the (A) ceiling's detection power. | deployable TPR@FPR within a modest gap of (A). | Large gap → deployable detection infeasible at this budget. | Small gap → headline deployable result. Large gap → report (A) as upper-bound-only + the gap as a finding. | ⬜ untested |
| **H5** *(stretch)* | M5 | An attacker aware of the **deployable B/C** detector can reduce detection **but at a measurable cost** (lower ASR / higher perplexity / restricted targets) — the detector **raises the attacker's bar** even if not unbreakable. | adaptive attack (fixed query/compute budget) lowers detection *and* lowers ASR / raises cost — a quantified trade-off. | Adaptive attacker evades at **no** cost. | Trade-off exists → bonus security contribution. No cost → important **negative**. *Only if M4 done with slack; else not claimed.* | ⬜ untested (stretch) |

---

## 4. The make-or-break instrument: the consistency metric

(Full options in understanding-doc §5.) **The metric — not the threshold — is the risk.** Plan:

- **(A) Privileged sim-state metric** — derive action semantics from LIBERO ground truth (object approached,
  gripper open/close, target region) vs the goal, over a **causal prefix window** `a_{t-k+1:t}` (no future
  actions). **Annotation schema frozen *before* inspecting any attack output** (no rules added after seeing
  the attack), with per-task unit tests + ablation, so the parser does not silently become the detector.
  *Cheap, clean → **upper bound**, NOT deployable. Label as such everywhere; the non-causal full-window
  variant is a post-hoc **monitoring ceiling** only.* → **M1/M2 (floor).**
- **(B) Learned action-semantics map** — small model: (observation, action_window) → goal embedding /
  descriptor; compare to trusted-goal embedding. *Deployable.* → **M4 (primary novelty).**
- **(C) Reference-policy divergence** — a small benign policy predicts expected action distribution for the
  trusted goal; flag divergence. *Deployable; needs a second policy.* → **M4 (primary novelty), alt/compl.**
- **(D) VLM/LLM judge** — **avoid** (too close to Wu et al. `actalign`; heavy; undermines the
  lightweight/non-CoT differentiation). Only if B and C both fail and time remains.

---

## 5. Metrics & evaluation conventions (lock at M0; apply everywhere)

- **Detection is *causal*** — it scores a **prefix window** `a_{t-k+1:t}` ending at the current/candidate
  action (OpenVLA emits one action per step → no future window at decision time) and decides whether to
  execute `a_t`. A full-window, post-hoc pass is reported separately as a **monitoring ceiling** (non-causal,
  labelled).
- **Primary detection:** ROC/AUC; **per-rollout TPR @ fixed per-rollout benign false-abort rate (1%, 5%)** on
  a **held-out** split, **with Wilson / Clopper-Pearson CIs**. *Per-window* rates are **auxiliary only**
  (1%/window compounds to ~63% false-abort over a 100-step rollout).
- **Detection latency** (steps of deviation before the hold fires) is a **first-class** metric — a single
  action is ambiguous, so latency > 0 is expected; quantify the target-actions executed before detection.
- **Cost:** **benign task-success degradation**; **per-rollout false-abort rate**; detector compute **latency**.
- **Security:** attack **detection rate**; **target-action-blocked rate** (*not* "unsafe-action-blocked" —
  until the semantic-redirect arm succeeds the target is a low-level action, not semantic harm); (M5, stretch)
  detection vs adaptive ASR.
- **Do NOT report "recovered task-success"** unless a replan / clean-instruction re-execution is actually
  built (a hold/abort prevents the target action; it does not complete the task).
- **Splits:** calibration split (set τ) **disjoint** from test split (report FPR). Held-out **tasks / scenes
  / seeds**, not just held-out rollouts of the same task. Harness asserts disjointness.
- **Baselines (all given the *same* calibration protocol — fair):** benign success; RoboGCG published
  numbers; **perplexity / text-only filter** (its τ also set on the calibration split — do **not** handicap
  it; "threshold unknowable a priori" is a separate *deployment* argument, stated as such, not an in-experiment
  excuse); **goal-agnostic action-anomaly baseline (mandatory** — shows goal-conditioning beats mere OOD
  detection); position conceptually vs `actalign`.

---

## 6. Open design decisions tracker (the understanding-doc §7 set)

> Resolved 2026-05-31 (author endorsed the PROPOSED base; sign-off logged §10). **D4/D7 remain OPEN**, gated
> on the M1 GCG micro-benchmark, with pre-registered decision rules below.

| ID | Decision | Resolution | Status |
|----|----------|-----------|--------|
| **D1** | Consistency metric | **(A)** privileged sim-state = floor/ceiling (non-deployable; labelled upper bound). **(B)** learned action-semantics map = **planned primary deployable**; **(C)** reference-policy divergence = complementary (pairs with the coarse-goal rung). **(D)** VLM judge **excluded** (actalign overlap + lightweight req). Final B/C emphasis refined at **M2** from the observed (A) signal. | **DECIDED** (B/C emphasis → M2) |
| **D2** | Attack-target | **RoboGCG-faithful** low-level target action, one-shot suffix at rollout start; success = *reached the target action region* over the persistence window (**window-scored**, not a single action). Semantic-danger = interpretation only. **Semantic-redirect arm (cross-task / wrong-object) GATED on M1**: add it iff M1 shows coherent targeted redirect; else reframe to *goal-abandonment / task-deviation* detection. Pre-registered → either outcome reportable. | **DECIDED** (+ M1-gated arm) |
| **D3** | Trusted-reference rungs | Implement (1) clean benchmark instruction = **ceiling** (label non-deployable), (2) **coarse operator-goal** = must-have realistic rung, (3) **task-ID→goal** = committed, compressible if M3 is tight. Reference-policy folded into metric (C). Coarse goals authored per LIBERO suite (small, version-controlled). | **DECIDED** |
| **D4** | Eval scale | Suites: LIBERO-Spatial/-Object/-Goal core (-10 optional). Provisional: ~5–10 tasks/suite, ~20–50 targets/task (subsampled from RoboGCG's 1792), ≥3 pinned seeds; calibration/test split disjoint by task/scene/seed. **Final matrix fixed at M1** from micro-benchmark s/target. | **OPEN** → M1 |
| **D5** | Baselines | benign success; RoboGCG published numbers (attack sanity); **perplexity/text-only filter** (detector to beat, **given the same calibration protocol** — fair); **goal-agnostic action-anomaly baseline (mandatory)**; conceptual positioning vs `actalign`. | **DECIDED** |
| **D6** | Metrics | per §5 (causal prefix-window detection; **per-rollout** TPR@{1%,5%} false-abort **+ CIs** held-out; **detection latency**; benign degradation; detection & **target-action-blocked** rate; M5(stretch) detection-vs-adaptive-ASR). No "recovered task-success" unless replan built. | **DECIDED** |
| **D7** | Compute budget | GCG micro-benchmark on GB10 first (H100's ~185–604 s/target may not hold at 4-bit); bound attack compute, concentrate on detector; **subsample to fit at M1**. Record actual s/target. | **OPEN** → M1 |

> **Post-review refinements (2026-05-31, Codex third-party review):** (1) detection is **causal** (prefix
> window + **detection-latency** metric); full-window = monitoring ceiling. (2) primary FPR = **per-rollout**
> false-abort + **CIs**. (3) metric(A) annotation **schema frozen before attack inspection** + unit tests.
> (4) baselines get the **same calibration** (fair); anomaly baseline **mandatory**. (5) **"target-action-blocked"**,
> not "unsafe". (6) **M5 → stretch**, scoped to attacking the deployable B/C detector with a fixed budget.
> (7) M1 gate also checks **coarse-goal separation** (not the clean-instruction ceiling alone). (8) framing:
> keep "injection" as the *threat class* but state the instantiation precisely as a *white-box adversarial
> textual suffix (RoboGCG)*; NL-injection (e.g. SABER) only as a secondary arm. Title wording = author's call.

---

## 7. Task ledger (working list — update statuses live)

> Status: ⬜ TODO · 🔄 DOING · ✅ DONE · ⛔ BLOCKED. Keep granular; one line each; add a `verify:` where useful.
> This ledger is the **source of truth for task state** across sessions.

### M0 — Design lock
- ✅ Resolve D1–D7 with author sign-off → recorded §6/§10 (2026-05-31; D4/D7 OPEN → M1).
- ✅ Write Phase-3 implementation plan (`docs/plans/t7-phase3-implementation-plan.md`) **+ revise per Codex review**. `verify:` author OKs before coding.
- 🔄 Define repo layout for code/configs/results — **pre-GB10 local-prep plan written** (`t7-local-prep-plan.md`); model-free scaffolding via TDD underway (Task 0+).

### M1 — Environment + viability gate
- ⬜ Stand up OpenVLA-7B (4-bit) on GB10; record exact env + provenance (checkpoint source/hash/date/licence) in `docs/references/`.
- ⬜ Reproduce **benign** LIBERO baseline success (pinned seeds). `verify:` numbers logged to write-once `results/`.
- ⬜ Reproduce **RoboGCG** on a few tasks; **confirm targeted redirect** (not denial). Quarantine suffixes in `artifacts/untrusted/`.
- ⬜ **GCG micro-benchmark** on GB10 → resolve D4 + D7.
- ⬜ Metric (A) signal sanity-check: separation that **survives at the coarse operator-goal reference** (not the clean-instruction ceiling alone). → **GO/NO-GO gate (H1)**.

### M2 — Floor detector (A) + FP-calibration
- ⬜ Implement metric (A) on a **causal prefix window**: `s(observation_t, action_{t-k+1:t}, trusted_goal, [state_t])`; **freeze the annotation schema before attack inspection**; unit-test per task.
- ⬜ Build calibrated detector: τ on calibration split → target **per-rollout** false-abort; fire → hold before executing `a_t`.
- ⬜ Evaluate: ROC/AUC, **per-rollout** TPR@{1%,5%} false-abort (**+CIs**) on held-out split, benign degradation, **detection latency**. → **H2**, **FLOOR SECURED**.
- ⬜ Baselines under the **same calibration**: perplexity/text-only filter **+ mandatory goal-agnostic anomaly** baseline.

### M3 — Trusted-reference ladder  *(N)*
- ⬜ Construct coarse operator-goal + task-ID→goal references in LIBERO.
- ⬜ Sweep detection+FPR across rungs → ladder table. → **H3**.

### M4 — Deployable detector (B/C)  *(N, committed — primary novelty)*
- ⬜ Build metric (B) learned action-semantics map (and/or (C) reference-policy); **specify supervision labels, negative-pair construction, train/test task-disjoint** (no task-prior leakage).
- ⬜ Calibrate + evaluate deployable detector (per-rollout, +CIs); compare to (A) ceiling. → **H4**.

### M5 — Adaptive attack  *(N, stretch — only if M4 done with slack)*
- ⬜ Implement adaptive attack against the **deployable B/C detector only** (fixed white/black-box + query/compute budget + #attempts cap).
- ⬜ Measure detection vs adaptive ASR + attacker cost. → **H5**.

### M6–M9 — Consolidate / analyse / write / submit
- ⬜ One-variable ablations; freeze results; figure-regen scripts (**RESULTS FREEZE**).
- ⬜ Claims ledger (§9); report negatives.
- ⬜ Draft → author rewrite → verify citations (no `[CITATION NEEDED]`).
- ⬜ Reproducibility appendix; submit.

---

## 8. Experiment protocol (use for every run — paste into the run's log)

```
run_id:        <UTC timestamp>-<short-slug>
git_commit:    <hash>
hardware:      GB10 (record CUDA / driver / torch versions)
config:        <path to pinned config>
seed(s):       <pinned, recorded>
hypothesis:    <H#>
expected:      <prediction before running — pre-register it>
command:       <exact command>
results_path:  results/<timestamp>/...   (WRITE-ONCE — never overwrite)
observed:      <fill after run>
decision:      <what this changes; link to §10 if a decision was made>
one_variable:  <what single variable changed vs the previous run>
```

**Reproducibility checklist (tick before committing any result):**
- [ ] Seeds pinned **and recorded** in the config/log.
- [ ] Exact env captured (`pip freeze`/conda export + CUDA/driver/torch + git commit).
- [ ] Checkpoint/dataset provenance recorded (source, **hash**, date, licence) in `docs/references/`.
- [ ] Output under timestamped **write-once** `results/` (no overwrite).
- [ ] Exactly **one variable** changed vs the comparison run.
- [ ] Figure regenerable from the logged data by a committed script.
- [ ] Negative / null outcomes recorded, not dropped.
- [ ] Any adversarial / poisoned artefact under `artifacts/untrusted/`; nothing untrusted auto-run.
- [ ] No dataset / checkpoint / secret / PII staged for commit.

---

## 9. Thesis-claim ledger (fill during M6–M7; guards against over-claiming)

> Every sentence the dissertation asserts as a *result* must point to a row here. No row → soften to
> "established result [cite]" or cut. Distinguish "my experiment showed" from literature.

| Claim (intended) | Evidence (experiment) | Result file | Status |
|------------------|-----------------------|-------------|--------|
| *(e.g.)* A calibrated goal-action detector reaches TPR=__% @ 1% benign FPR on held-out LIBERO-X | M2 run __ | `results/…` | ⬜ |
| *(e.g.)* Detection degrades to baseline once reference is coarser than the operational instruction | M3 ladder | `results/…` | ⬜ |
| *(e.g.)* Deployable (B) recovers __% of the (A) ceiling | M4 run __ | `results/…` | ⬜ |
| *(e.g.)* Adaptive attacker evades only at __ ASR cost | M5 run __ | `results/…` | ⬜ |

---

## 10. Decision log (append-only)

| Date | Decision | Rationale | Changed |
|------|----------|-----------|---------|
| 2026-05-30 | Theme = **T7** | deliverable-first risk appetite + language→action salience (understanding-doc §2) | scope locked |
| 2026-05-31 | RoboGCG defence claims **verified**; defence problem is **open** | full text §5 + Table 3 (`docs/references/`) | motivation for T7 confirmed; perplexity = baseline-to-beat |
| 2026-05-31 | **Timeline** ~14–16 wk, submit early Sep 2026; **scope** core-first **but novelty (B/C + adaptive) is the committed goal** | author steer | §2 milestones / tiers set |
| 2026-05-31 | **D1** metric: (A) floor + (B) primary deployable / (C) complementary, (D) excluded; B/C emphasis → M2 | most directly instantiates goal-action consistency; avoid actalign overlap & stay lightweight | M2/M4 scope |
| 2026-05-31 | **D2** attack: RoboGCG-faithful primary, window-scored; **semantic-redirect arm gated on M1** | minimise attack-side risk; pre-register denial-vs-targeted so either outcome is reportable | M1 gate rule |
| 2026-05-31 | **D3** rungs: clean(ceiling)+coarse-goal(must)+task-ID(committed); **D5/D6** baselines & metrics adopted (+optional anomaly baseline) | necessity critique → lead with coarse rungs; baselines/metrics standard & defensible | M2/M3 scope locked |
| 2026-05-31 | **D4/D7** kept **OPEN**, rules pre-registered → M1 micro-benchmark | cannot size eval / budget before measuring GCG on GB10 | M1 |
| 2026-05-31 | Phase-3 implementation plan drafted | M0 deliverable; lifts coding gate for M1–M2 | `docs/plans/t7-phase3-implementation-plan.md` |
| 2026-05-31 | **M5 (adaptive) → stretch**; M4 deployable detector = committed primary novelty | Codex review: both-committed = over-scope for solo MSc; author chose M5-stretch | §0,§2 tiers, §3 H5, §7 |
| 2026-05-31 | **Codex review incorporated** | third-party review verified vs RoboGCG/actalign primary sources | causal prefix-window + latency; per-rollout FPR + CIs; metric(A) schema frozen; fair-calibrated baselines + mandatory anomaly; "target-action-blocked"; M1 coarse-goal check; precise "adversarial textual suffix" framing — §0,§1,§2,§3,§5,§6,§7 + impl plan |
| 2026-05-31 | **Adjacent prior work verified** (4 arXiv IDs all correct): Task Drift, Instruction Hierarchy, AlignSentinel (text-LLM), SABER (VLA attack) | none scoop the embodied/action-level contribution | **novelty narrowed to the VLA action-level instantiation**; SABER = candidate secondary attack arm (understanding-doc §6) |
| 2026-05-31 | **Author OK → start M1–M2 scaffolding code** (model-free, M1/8 GB); pre-GB10 local-prep plan written | gate-lift precondition met (plan agreed in `docs/plans/` + author OK); OpenVLA inference infeasible locally (8 GB RAM) → build+test only model-free components; experimental *runs* await GB10 | `docs/plans/t7-local-prep-plan.md`; coding begins |

---

## 11. Session protocol (for Claude Code — do this every session)

**At session start:**
1. Read **§0 North Star** + **§1 You-Are-Here**. Confirm current phase & next action.
2. Check **§6 open decisions** and **§7 BLOCKED tasks** — don't start work that's gated on an unresolved decision.
3. Pick the next **⬜/🔄** task from §7 for the current milestone. If it's experiment work, confirm the M0
   plan is agreed and you're past M1's gate where required.

**While working:**
- One variable at a time; log every run with the §8 protocol; quarantine untrusted artefacts.
- Surface tradeoffs / confusion **before** implementing (CLAUDE.md §1). Mark unverified facts `[CITATION NEEDED]`.

**Before ending the session:**
1. Update **§1 You-Are-Here** (date, last completed, next action, blockers, floor/novelty status).
2. Update **§7 task ledger** statuses; update **§3 hypothesis** statuses if evidence arrived.
3. Append any decision to **§10**; add any claim to **§9**.
4. Never leave a result outside write-once `results/`; never leave a fact asserted without evidence.

---

## 12. Quick-reference facts (so they're never re-derived)

- **Victim model:** OpenVLA-7B, discrete 256-bin action tokenisation (`arXiv:2406.09246`). Detector concept
  is head-agnostic, but RoboGCG reproduction is scoped to this discrete-token checkpoint.
- **Sim / data:** LIBERO (Spatial / Object / Goal / -10). Data gitignored.
- **Attack:** **RoboGCG** (`arXiv:2506.03350`, PDF + verified facts in `docs/references/`). White-box GCG
  textual suffix, one-shot at rollout start, >90% targeted-action success (Goal/Object/Spatial); proves
  *control authority*, not semantic harm. **Its borrowed defences (PF/smoothing) have no usable operating
  point → T7's opening.**
- **Nearest prior art (novelty constraint):** Wu et al. `actalign` (`arXiv:2510.16281`) — reasoning↔action
  consistency, but **benign/OOD only, no attacker, no FP calibration, needs CoT+VLM**. T7 differs by:
  attacker-aware + FP-calibrated + lightweight non-CoT.
- **Adjacent text-LLM prior (verified 2026-05-31 — see understanding-doc §6):** Task Drift `2406.00799`,
  Instruction Hierarchy `2404.13208`, AlignSentinel `2602.13597` (closest; FP-aware injection detector, **text
  LLM**). ⇒ **novelty = the *embodied / VLA action-level* instantiation only** (do not claim FP-aware injection
  detection as new in general). **SABER `2603.24935`** = a real NL injection **attack** on VLA/LIBERO →
  candidate secondary attack arm (perplexity-baseline-defeating).
- **Hardware:** single GB10 (~128 GB unified); OpenVLA-7B 4-bit fits. H100 timing numbers may not transfer.
- **Key paths:** understanding doc `docs/plans/t7-goal-action-consistency-detector.md`; landscape
  `docs/lit-review/`; references + verified facts `docs/references/`; results (write-once) `results/`;
  untrusted artefacts `artifacts/untrusted/`.

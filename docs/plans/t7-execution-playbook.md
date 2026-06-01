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

**Compute & tiers (2026-06-01).** Baseline = single GB10 (**Tier-F**, guaranteed). An **A100/H100 cluster is
requested** (D8, pending) → **Tier-N** (deployable B/C, realistic adaptive GCG, full reference ladder) becomes
committed at the **M1 compute-confirmation checkpoint**; fallback to Tier-F if it slips. Headline reframe =
**§3a / H6**; the **tiered roadmap = §2**; **what to implement = §4b**.

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

- **Last updated:** 2026-06-01
- **Phase:** **Design → M0 (exiting)**. Coding gate **lifted for M1–M2 (author OK, 2026-05-31)**; **pre-GB10 local build underway** — model-free M1–M2 components on M1/8 GB (`docs/plans/t7-local-prep-plan.md`). OpenVLA/GCG/LIBERO *runs* await the granted GPU. **Headline reframed → Embodiment Evasion Tax (§3a, H6); milestones re-mapped (§2); what-to-implement = §4b; compute may upgrade GB10→A100/H100 cluster (D8, pending).**
- **Last completed:** Theme = T7; understanding doc; **RoboGCG defence verified** (`docs/references/`); **D1–D7 resolved** (§6/§10; D4/D7 OPEN → M1); **Phase-3 implementation plan drafted**; **Codex third-party review incorporated** (causal prefix-window + detection latency; per-rollout FPR + CIs; metric(A) schema frozen; fair-calibrated baselines; M5→stretch — see §10).
- **Currently:** Executing the **pre-GB10 local-prep plan** (`docs/plans/t7-local-prep-plan.md`) on M1/8 GB via TDD. **Done 2026-05-31: Tasks 0,1,2,3,4,5,6,7,8,9** — env scaffold, repro infra, data records, **action codec (OpenVLA formula verified from source `c8f03f48`; provenance recorded)**, privileged-state adapter, **metric (A) — annotation schema FROZEN (`docs/plans/metric-a-annotation-schema.md`, `2c2f163`) + causal scorer (P1 progress / P2 distractor / P3 grasp; combine=max; privileged anchor via resolver seam; non-causal monitoring-ceiling variant)**, FP-calibrated detector, eval harness (ROC/AUC, TPR@FPR + Wilson/CP CIs, latency, split-disjointness), **M2 baselines (`3287c5c`): goal-agnostic χ²-OOD action anomaly + perplexity/text-only filter (mock + GB10 stub), both through the *same* `calibrate`**, **config/scripts/figures (`60b0462`): frozen pydantic `Config` + `one_variable_diff`; shared GB10 guard (no-CUDA → exit non-zero, no silent no-op); `make_figures` regenerates ROC/score-hist/TPR@FPR-CI per condition purely from a logged `results.json`; 6 runnable scripts** — **231 tests green, full `src/t7` type-clean** (see plan Status table). **Remaining delegable:** Task 11 runbook. **Needs author/me first:** Task 10 LIBERO smoke (time-boxed, optional). *Note: 3 pre-existing pyright errors + 1 ruff B905 sit in untouched test files (`test_state.py`/`test_records.py`/`test_consistency_a.py`) — pre-Task-9, not yet cleaned.*
- **▶ NEXT ACTION:** **Local (now):** finish local-prep Tasks 10–11 **+ build the model-free §4b interfaces** (L1 `InternalProbe`/`ActivationExtractor`, idealized action-space attacker, cross-layer eval + tax metrics) with synthetic-fixture tests. **On the granted GPU (M1):** stand up OpenVLA-7B on the *actual* HW; benign baseline + RoboGCG *targeted* redirect; **GCG micro-bench on that HW (fixes D4/D7)**; metric-(A) signal incl. coarse-goal check; **CONFIRM cluster access → lock Tier-N scope (D8)** → **GO/NO-GO gate (H1)**.
- **Blockers:** none.
- **Open decisions outstanding:** **D4, D7** (OPEN → M1 GCG micro-bench on the granted HW) **＋ D8 compute tier** (A100/H100 cluster requested 2026-06-01, pending → confirm at M1; sets Tier-N committed scope). D1/D2/D5/D6 DECIDED; **D3 re-tiered** by the reframe (operator-goal rung committed, task-ID → Tier-N, see §2 M5).
- **Floor secured?** ❌ not yet (target: end of **M2**, ~Jul 12).
- **Novelty status:** headline = **Embodiment Evasion Tax (H6)**. **M3 idealized intrinsic-tax frontier = committed (Tier-F, GB10-guaranteed)**; **M4 deployable B/C + realistic adaptive = committed-if-cluster (Tier-N, D8)**; **M5 reference-ladder + SABER = secondary/stretch**.
- **Direction (DECISION, author-converged 2026-06-01 — flag for supervisor sign-off):** research **core reframed** to the *Embodiment Evasion Tax* measurement (see §3 **H6** + **§3a**). The behavioural detector is recast as an **instrument** measuring per-layer adaptive-evasion cost (**L0** input < **L1** internal-probe < **L2** action-monitor), **not** a claimed defence/"firewall". Scope held to the **instruction channel** (RoboGCG primary, SABER secondary); physical/CoT injection (TRAP) = future arm, out of committed scope. Floor (M2 + cheap idealized action-space frontier) unchanged → deliverable still guaranteed. *Citation pass DONE 2026-06-01:* all 5 flagged items resolved + **16 cited PDFs downloaded, gitignored, SHA-256-pinned** with provenance (`docs/references/README.md`). Net: **nothing scoops the runtime/FP-calibrated/adaptive lane** (VLA defences found are training-time EDPA/STRONG-VLA; actalign benign-only; AttackVLA attack-only). Title **LOCKED** (§3a, pending supervisor).

---

## 2. Milestone roadmap (W1 = week of 2026-06-01; submit early Sep 2026)

**Tiers.** **F = Floor** (guaranteed; **compute-agnostic — runs on a single GB10**). **N = Novelty** (the real
goal; **committed once the A100/H100 cluster is confirmed**, gated — gates decide *how to adapt*, never
*whether to drop*).

**Compute posture (2026-06-01, D8 — pending).** Single GB10 is the guaranteed baseline; an **A100/H100 cluster
has been requested** from the institution and is **expected but unconfirmed**. The roadmap is therefore
**compute-tiered**: Tier-F secures the dissertation on GB10 alone; Tier-N (deployable B/C, realistic adaptive
GCG, full reference ladder, larger matrix) is **promoted to committed at the M1 compute-confirmation
checkpoint**, with graceful fallback to Tier-F if access slips. **Reproducibility rule: log the exact hardware
per run; never compare results across different hardware within a single claim.**

**Milestone contents re-mapped 2026-06-01** for the Evasion-Tax reframe (§3a): the new headline (cross-layer
comparison + **idealized evasion-tax frontier**) is now **M3**; the old trusted-reference ladder moves to
**M5**. The Phase-3 plan's M3–M5 *sketch* is superseded by the table below (its **M1–M2 component contracts
remain valid**).

| ID | Tier | Milestone | Target weeks | Exit gate (verifiable) |
|----|------|-----------|--------------|------------------------|
| **M0** | F | **Design lock** — D1–D7 (§6) + the **Evasion-Tax reframe (H6/§3a)** + this guidebook | W1–W2 | done; reframe author-converged (**supervisor sign-off pending**); coding allowed |
| **M1** | F | **Environment + viability gate + COMPUTE CONFIRMATION** — OpenVLA-7B on the granted HW; benign baseline; RoboGCG *targeted* redirect; **GCG micro-bench on the *actual* HW (resolves D4/D7)**; metric-(A) signal; **confirm cluster → lock Tier-N scope (D8)** | W2–W4 | **GO/NO-GO (H1):** benign reproduced **＋** RoboGCG *targeted* redirect (not denial) **＋** benign-vs-attacked separation **surviving at the coarse operator-goal reference** **＋ compute tier locked**. *Denial-only → reframe to task-deviation (understanding-doc §9); separation only at the clean-instruction ceiling → necessity weak, flag.* |
| **M2** | F | **Floor detection layer** — **L0** (perplexity) + **L2-oracle** (metric A) FP-calibrated; **build the L1 internal-probe arm** (activation-delta primary; attention ablation) | W4–W6 | **non-adaptive cross-layer TPR@{1%,5%} per-rollout FPR** (L0/L1/L2) on a **held-out** split + ROC/AUC + CIs + benign degradation (**H2**) |
| **M3** | **F** | **Idealized Evasion-Tax frontier (M-b) — THE committed headline** — idealized action-space attacker vs the L2-oracle → (ASR, evasion) Pareto; adaptive GCG vs L1 (standard white-box); compute the **intrinsic tax** + the L0/L1/L2 evasion-cost ordering | W6–W9 | **H6 core:** idealized (ASR, evasion) frontier + tax scalar(s) with CIs; **all four H6 outcomes reportable** |
| **M4** | **N — committed-if-cluster** | **Deployable L2 (B/C) + realistic adaptive comparison** — build deployable behavioural detector (B/C, **train/test task-disjoint**); **realistic adaptive GCG-through-policy** vs L1 and L2 at matched budget → fair *deployable-vs-deployable* tax | W8–W11 | deployable per-rollout TPR@FPR +CIs **and** realistic adaptive tax with **honest gap to the M3 oracle** (**H4 + H6-realistic**). *Fallback if no cluster: report M3 + oracle-gap; mark realistic-adaptive **not run**.* |
| **M5** | **N — secondary/stretch** | **Reference-coarsening ladder + threat-generalization** — operator-goal rung (committed-secondary, Tier-F-cheap) ＋ task-ID rung (Tier-N) ＋ **SABER** fluent attack (where L0 dies) ＋ physical/CoT note (discussion only) | W9–W12 | ladder TPR@FPR per rung (**H3**); SABER arm = input-level dies but behavioural fires |
| **M6** | F | **Consolidation + ablations** — one-variable ablations (k, probe type, combine rule); freeze operating points; figures script-regenerable | W11–W12 | **RESULTS FREEZE** in write-once `results/`; every figure script-regenerable |
| **M7** | F | **Analysis + claims ledger** — every claim→evidence; report negatives | W12–W13 | §9 ledger complete; every claim → a result file |
| **M8** | F | **Write-up** — draft (overlaps M7); author rewrites generated prose; verify citations | W12–W14 | complete draft; **zero** `[CITATION NEEDED]` |
| **M9** | F | **Polish + submit** — reproducibility appendix; submission | W14–W15 | submitted early Sep 2026 |

**Critical path & protection.** M0→M1→M2→**M3** secure the **floor + the headline intrinsic-tax result on GB10
alone** — the dissertation is safe regardless of the cluster. **M4 (Tier-N)** turns the headline into the fair
*deployable-vs-deployable* result the moment the cluster is confirmed; if it slips, M3 + the honest oracle-gap
still stand. **M5 is secondary/stretch** (lead the necessity argument with the operator-goal rung). The only
**hard kill** is the M1 GO/NO-GO (no signal at all); later gates *adapt scope* (trim M5, defer M4 realistic-arm)
but **never abandon M3** (the committed headline).

---

## 3. Hypothesis register (each maps to a milestone)

> Form: **statement** · *prediction if true* · **falsifier** · **decision rule**. Update **status** as
> evidence arrives. A falsified hypothesis is a **result to report**, not a failure to hide.

| ID | Milestone | Hypothesis | Prediction if true | Falsifier | Decision rule | Status |
|----|-----------|------------|--------------------|-----------|----------------|--------|
| **H1** | M1 | RoboGCG-injected rollouts yield action windows **measurably inconsistent** with the trusted goal, separable from benign by metric (A). | benign-vs-attacked score distributions separable (AUC ≫ 0.5). | AUC ≈ 0.5 (overlap). | High AUC → proceed to M2. Low → rethink metric / reframe to "task-abandonment detection" before investing. | ⬜ untested |
| **H2** | M2 | A threshold τ set on a **calibration split** gives **high TPR at a low *per-rollout* benign false-abort rate** *without* destroying benign task success — the **usable operating point RoboGCG's borrowed defences lacked**. | per-rollout TPR@{1%,5%} false-abort (with CIs) ≫ **fair-calibrated** perplexity baseline; benign task-success drop small; bounded detection latency. | No τ separates without large benign cost. | Usable point → floor secured, push novelty. None → **negative result** (behavioural detection also lacks a usable point under these conditions) — still publishable; report cleanly. | ⬜ untested |
| **H3** | M5 | Detection degrades **gracefully** as the trusted reference coarsens; still useful at deployment-realistic rungs. | ~monotone TPR@FPR decline; coarse-goal rung still > baseline. | Detection collapses to baseline once the reference is coarser than the operational instruction. | Survives → lead the deployment argument with coarse rungs. Collapses → report where/why the "necessity" critique bites (thesis backbone either way). | ⬜ untested |
| **H4** | M4 | A **deployable** metric (B/C, no privileged state) recovers a **substantial fraction** of the (A) ceiling's detection power. | deployable TPR@FPR within a modest gap of (A). | Large gap → deployable detection infeasible at this budget. | Small gap → headline deployable result. Large gap → report (A) as upper-bound-only + the gap as a finding. | ⬜ untested |
| **H5** *(Tier-N; realistic-adaptive arm of H6)* | M4 | An attacker aware of the **deployable B/C** detector can reduce detection **but at a measurable cost** (lower ASR / higher perplexity / restricted targets) — the detector **raises the attacker's bar** even if not unbreakable. | adaptive attack (fixed query/compute budget) lowers detection *and* lowers ASR / raises cost — a quantified trade-off. | Adaptive attacker evades at **no** cost. | Trade-off exists → bonus security contribution. No cost → important **negative**. *Only if M4 done with slack; else not claimed.* | ⬜ untested (stretch) |
| **H6** *(reframe core)* | M3–M4 | Across defence layers (**L0** input-filter < **L1** internal-representation probe < **L2** behavioural action-monitor), **L2 imposes a higher adaptive-evasion cost** than L1/L0 at matched benign FPR — and this *embodiment evasion tax* is **intrinsic** (persists under an *idealized action-space* attacker, **mechanism M-b**), not merely an artifact of GCG failing to differentiate through the closed-loop rollout (**M-a**). | At matched FPR, suppressing L2's TPR to L1's evaded level forces measurable ASR forfeit / more queries; the **idealized action-space Pareto frontier** for L2 dominates (is worse-for-attacker than) L1's; tax > 0 with CIs. | L2 evaded at **≤** L1's cost (tax ≤ 0), **or** the tax vanishes under the idealized attacker (it was only M-a). | **4 outcomes, all reportable:** (i) L2>L1 & M-b → *embodiment creates an evasion tax → place the boundary at the action layer* (headline). (ii) L2 also falls → *embodiment alone does not save VLA defences*. (iii) L1 strong → *VLA internal reps carry security-relevant injection signal*. (iv) both weak → *adaptive evaluation is mandatory for VLA defences*. | ⬜ untested |

### 3a. Direction lock — the *Embodiment Evasion Tax* measurement frame (DECISION 2026-06-01)

> **Three independent passes converged** (deep-research synthesis + author refinement + Codex review).
> Recorded here as the operational decision; **flag for supervisor sign-off** before it becomes the
> dissertation headline. Replaces nothing in M0–M3; it is the **unifying measurement framing** over H4/H5.

**The reframe.** The research core is a **measurement**: *do behavioural action-layer monitors cost more to
adaptively bypass than internal-representation probes on instruction-injected VLA policies?* The monitor is an
**instrument** for measuring per-layer adaptive-evasion cost — **not** a claimed robust defence / "firewall".
This is the *claim-type* choice that survives the most likely bad outcome: a defence-efficacy headline dies to
an adaptive attacker (cf. "Attacker Moves Second" — 12 LLM defences bypassed >90% `[CITATION NEEDED]`); a
measurement headline is **true whichever way the result falls** (see H6's four outcomes).

**Working title (LOCKED 2026-06-01 — pending supervisor sign-off).** *The Embodiment Evasion Tax: Measuring
Adaptive Evasion Costs of Runtime Defences for Vision-Language-Action Models.* Chosen over the question-form
variant for dissertation register; the subtitle ("Measuring … Costs") is **outcome-neutral** — "tax" is a
**coined framing term to be *measured***, and may be ≈0 (see H6's four outcomes), so the title presupposes no
result. One-line swap if preferred: *… Do Behavioural Monitors Cost More to Bypass than Internal Probes in VLA
Policies?*

**The instrument — three defence layers, compared at matched benign FPR:**
- **L0 input** — perplexity / text-only filter (**already built**; expected to die on fluent SABER-style injection).
- **L1 internal** — activation/attention probe ported to OpenVLA (Task-Drift / AlignSentinel lineage). **New
  first-class experimental arm** — more build than a single-detector study; budget for it. Must be a *fair,
  strong* baseline (proper calibration + proper adaptive attack), not a strawman.
- **L2 behavioural** — the existing goal-action-consistency detector (metric A ceiling + deployable B/C).

**Mechanism separation (NON-NEGOTIABLE rigor — the point Codex's 4-outcome list under-specifies).** Report the
tax as **two measurements**, never one:
- **M-b (intrinsic, cheap, carries the core claim):** an *idealized action-space* attacker directly optimises an
  action sequence to maximise attack-target reach while minimising L2's score → the (ASR, evasion) Pareto
  frontier. No GCG needed → GB10-cheap. Isolates the embodiment-intrinsic tax from attack-access.
- **M-a + M-b (realistic, expensive, subsample):** adaptive GCG-through-policy. Confirmatory; **stretch** — if
  budget-cut, M-b + floor still stand. *Without M-b the result is trivial ("GCG can't backprop through
  physics") and dies in review.*

**Scope discipline (floor protection).** Committed: **instruction channel only** — RoboGCG (primary,
high-perplexity adversarial string) + SABER (secondary, fluent/low-perplexity, where L0 fails). **Out of
committed scope:** physical/visual prompt injection (TRAP — *verified* visual-patch/CoT attack, needs a CoT-VLA
≠ OpenVLA-7B) → *threat-generalization / future arm only*. Supply-chain/backdoor (BadVLA) = the declined T9
lane → **rejected**.

**Free secondary contribution.** The existing eval harness already emits the operational defence metrics no VLA
benchmark reports (per-rollout FPR + CIs, false-abort cost, detection latency, task-success degradation,
adaptive budget) → **package as a reusable VLA defence-evaluation protocol** (AttackVLA is *verified*
attack-only). "firewall" may appear in the **intro framing only**, never the main claim.

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
  descriptor; compare to trusted-goal embedding. *Deployable.* → **M4 (deployable L2; Tier-N).**
- **(C) Reference-policy divergence** — a small benign policy predicts expected action distribution for the
  trusted goal; flag divergence. *Deployable; needs a second policy.* → **M4 (deployable L2; Tier-N), alt/compl.**
- **(D) VLM/LLM judge** — **avoid** (too close to Wu et al. `actalign`; heavy; undermines the
  lightweight/non-CoT differentiation). Only if B and C both fail and time remains.

---

## 4b. What to implement — the Evasion-Tax instruments (the new builds beyond the floor)

> The floor code (metric A, calibration, eval stats, baselines — local-prep Tasks 0–9) is **done and reused
> as-is**: metric A = the **L2 oracle**, perplexity = **L0**, the eval harness = the shared calibrated
> comparator. This section specifies the **three new instruments** the reframe needs. Interfaces +
> synthetic-fixture tests are buildable **now** (model-free, TDD per local-prep conventions); the parts needing
> OpenVLA activations or LIBERO rollouts are deferred behind thin interfaces to the granted GPU.

**(I) L1 internal-representation probe** — `src/t7/metric/probe_internal.py` (+ an `ActivationExtractor` seam).
- **Contract:** `InternalProbe.score(features) -> Score` (higher = more injected), calibrated through the
  **same `calibrate`** (invariant #4) at the matched per-rollout FPR.
- **Primary = activation-delta linear probe** (Task-Drift lineage, `2406.00799`): features = hidden-state delta
  across the injection point in OpenVLA's transformer trunk; a logistic-regression / linear probe trained on
  **benign-vs-injected** rollouts, **train/test task-disjoint** (report whether it merely memorises a task prior).
- **Ablation = attention-map MLP** (AlignSentinel lineage, `2602.13597`): attention-pattern features → small MLP.
  Pre-registered ablation; AlignSentinel is the **scoop-risk closest prior** → cite + differentiate (text-LLM vs
  our VLA/action setting); do **not** depend on exact reproduction.
- **`ActivationExtractor`** = thin seam over an OpenVLA forward pass returning hidden states / attention at the
  decision step; **synthetic fixtures** for local tests, real impl on the GPU.

**(II) Idealized action-space attacker (M-b — the intrinsic-tax instrument)** — `src/t7/attack/idealized_frontier.py`.
- **Contract:** given `(task, TargetActionSpec (D2), metric-A oracle scorer)`, search over **executable action
  sequences** (respecting reachability + the persistence window) to **maximise target-reach while minimising the
  metric-A consistency score** → trace the **(ASR, evasion = 1−detection) Pareto frontier**.
- **Detector-agnostic by design:** it attacks the *goal-consistency concept* via the privileged oracle → a
  **lower bound on attacker cost against any goal-consistency behavioural detector**. This is what isolates
  **M-b (intrinsic)** from **M-a** (the "GCG can't differentiate through the rollout" artifact).
- The **optimiser + frontier logic are model-free** (locally testable on synthetic dynamics); evaluating
  reachability / privileged-state on real scenes is the GPU/LIBERO part.

**(III) Cross-layer evaluation + the "tax" metrics** — extend `src/t7/eval/`.
- Run **L0 / L1 / L2** through the **same calibration** and the **same attacks**; emit the comparative table +
  the Pareto-frontier overlay.
- **Operational tax metrics (define once, report with CIs):** (a) **ASR forfeited** to bring a layer's TPR down
  to a fixed evaded level; (b) **attacker queries/compute** to reach a fixed evasion; (c) **area between Pareto
  frontiers** (L2 vs L1). These double as the **reusable VLA defence-evaluation protocol** (the free secondary
  contribution; AttackVLA is attack-only).
- **Roles fixed:** L2-oracle = metric A (privileged; M-b + non-adaptive ceiling). L2-deployable = metric B/C
  (M4, Tier-N). L1 = internal probe. L0 = perplexity. **Never present the oracle as deployable.**

**Build order:** L1 interface + idealized-attacker optimiser + cross-layer eval (all model-free, **now**) →
real `ActivationExtractor` + LIBERO reachability (M1/M2 on GPU) → idealized frontier (**M3**) → deployable B/C +
realistic adaptive (**M4**, Tier-N).

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
- **Hardware provenance (2026-06-01):** record the **exact hardware** (GB10 / A100 / H100) in every run log;
  **never compare results across different hardware within a single claim** (HW is a variable — re-run the
  comparator on the same HW). All L0/L1/L2 cross-layer comparisons and adaptive-cost frontiers must be produced
  on one hardware class.

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
| **D7** | Compute budget | GCG micro-benchmark on the granted HW first (H100's ~185–604 s/target may not hold at 4-bit on GB10); bound attack compute, concentrate on detector; **subsample to fit at M1**. Record actual s/target. | **OPEN** → M1 |
| **D8** | Compute tier | Baseline single GB10 = **Tier-F** (guaranteed). **A100/H100 cluster requested 2026-06-01** → if confirmed at M1, **Tier-N** (deployable B/C, realistic adaptive GCG, full reference ladder, larger matrix) becomes committed; else graceful fallback to Tier-F. **Log HW per run; no cross-HW comparison within a claim.** | **OPEN** → M1 confirmation |

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
- ✅ Implement metric (A) on a **causal prefix window**: `s(observation_t, action_{t-k+1:t}, trusted_goal, [state_t])`; **annotation schema FROZEN before attack inspection** (`docs/plans/metric-a-annotation-schema.md`, `2c2f163`); unit-tested in isolation. *(model-free scorer done locally; per-task fixtures on real LIBERO ground truth + the GB10 benign-vs-attacked signal check remain M1.)*
- ⬜ Build calibrated detector: τ on calibration split → target **per-rollout** false-abort; fire → hold before executing `a_t`.
- ⬜ Evaluate: ROC/AUC, **per-rollout** TPR@{1%,5%} false-abort (**+CIs**) on held-out split, benign degradation, **detection latency**. → **H2**, **FLOOR SECURED**.
- ✅ Baselines under the **same calibration** (model-free, `3287c5c`): **mandatory goal-agnostic anomaly** (χ²-OOD on the action stream, goal-blind + causal) **+ perplexity/text-only filter** (`MockPerplexityScorer` + GB10 stub `RealPerplexityScorer`; monotone ppl→score keeps calibration order-equivalent to raw-perplexity thresholding). *(Real LM-perplexity backend + actual benign-vs-attacked numbers remain GB10.)*
- ⬜ **(NEW, model-free — build now, §4b)** Interfaces + synthetic-fixture tests for the **L1 internal probe** (`InternalProbe` + `ActivationExtractor`), the **idealized action-space attacker**, and the **cross-layer eval + tax metrics** — so GPU day-1 is "plug in activations + rollouts". `verify:` all calibrate through the *same* `calibrate`; activation/LIBERO impls stubbed for GB10.

### M3 — Idealized Evasion-Tax frontier  *(F — committed headline)*
- ⬜ **On GPU:** extract OpenVLA activations during benign+attacked rollouts; train + calibrate the **L1 probe** (activation-delta primary; attention-map ablation), **task-disjoint**. → completes **H2** cross-layer + feeds **H6**.
- ⬜ Run the **idealized action-space attacker** vs the metric-A oracle → **(ASR, evasion) Pareto frontier**; compute the **tax scalar(s)** + the L0/L1/L2 evasion-cost ordering, +CIs. → **H6 core** (all four outcomes reportable).

### M4 — Deployable L2 (B/C) + realistic adaptive  *(N — committed-if-cluster)*
- ⬜ Build deployable behavioural detector **(B)** learned action-semantics map and/or **(C)** reference-policy; **specify supervision labels, negative-pair construction, train/test task-disjoint** (no task-prior leakage). → **H4** (gap to the A oracle).
- ⬜ **Realistic adaptive GCG-through-policy** vs **L1 and L2** at a matched query/compute budget → fair *deployable-vs-deployable* tax. → **H6-realistic**. *Fallback if no cluster: skip; report M3 + oracle-gap, mark **not run**.*

### M5 — Reference ladder + threat-generalization  *(N — secondary/stretch)*
- ⬜ Operator-goal rung (committed-secondary, Tier-F) + task-ID rung (Tier-N); sweep detection+FPR across rungs → ladder table. → **H3**.
- ⬜ (stretch) **SABER** fluent-injection arm — **confirm OpenVLA inclusion first** (`docs/references/README.md`); show L0 (perplexity) dies but L2 fires. (Physical/CoT generalization = discussion only.)

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
hardware:      <GB10 | A100 | H100-cluster> — record exact HW + CUDA/driver/torch (NEVER compare across HW within one claim)
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
| 2026-05-31 | **Metric (A) annotation schema FROZEN** (load-bearing Task 5); design **delegated by author to Claude** with "adopt the realistic option; pre-register value-adding variants as stretch" | freeze must precede any attack output (circularity guard, invariant #2); decisions: privileged `target_region` anchor via resolver seam; primitives P1 progress / P2 distractor / P3 grasp; combine=`max` (zero params, robust to inter-primitive correlation); `{noisy_or, weighted_mean}` + `k`/`r` sweeps = pre-registered ablations; stretch S1 orientation, S2 multi-phase sub-goal = definitions frozen now, implemented later | `docs/plans/metric-a-annotation-schema.md`, commit `2c2f163`; §7 M2 first item ✅ |
| 2026-06-01 | **Headline reframed → Embodiment Evasion Tax** (H6/§3a); monitor = *instrument* measuring per-layer adaptive-evasion cost (L0/L1/L2), **not** a 'firewall'/defence claim; **milestone contents re-mapped** (idealized frontier → M3, ladder → M5); title **LOCKED** (pending supervisor) | 3 independent passes converged (deep-research + author + Codex); measurement framing survives the likely bad outcome (Attacker-Moves-Second `2510.09023`: efficacy claims die to adaptive attackers) | §0,§2,§3,§3a,§4b,§7,§12 |
| 2026-06-01 | **Citation pass DONE** — 5 flagged items resolved + **16 cited PDFs** downloaded/SHA-pinned/provenance | reframe must rest on a verified landscape (integrity rule); net: **nothing scoops** the runtime/FP-calibrated/adaptive lane (VLA defences found = training-time; actalign benign-only; AttackVLA attack-only) | `docs/references/README.md` |
| 2026-06-01 | **D8 compute tier OPEN** — A100/H100 cluster requested (pending); roadmap **compute-tiered** (Tier-F GB10-guaranteed / Tier-N committed-if-cluster) | author: cluster access likely → de-risks deployable B/C + realistic adaptive + full ladder; unconfirmed → floor stays compute-agnostic | §0,§2,§6 (D8),§8,§12; M1 compute-confirmation checkpoint |

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
- **Hardware:** baseline **single GB10** (~128 GB unified; OpenVLA-7B 4-bit fits; H100 timings may not
  transfer). **A100/H100 cluster requested 2026-06-01 (D8, pending)** → if confirmed, Tier-N committed + GCG
  timing approaches the paper's. **Log exact HW per run; never compare across HW within one claim.**
- **Key paths:** understanding doc `docs/plans/t7-goal-action-consistency-detector.md`; landscape
  `docs/lit-review/`; references + verified facts `docs/references/`; results (write-once) `results/`;
  untrusted artefacts `artifacts/untrusted/`.

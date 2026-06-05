# Theme-Scoping Report: Integrity of Vision-Language-Action (VLA) Models

*MSc Cyber Security & AI — Individual Research Project. Phase: theme scoping.*
*Prepared 2026-05-29. All technical claims cite sources actually fetched during this review; unverifiable claims are flagged `[CITATION NEEDED]`.*

> **⚠️ Compute superseded 2026-06-02 (historical doc — body left intact for provenance):** the "single NVIDIA GB10 node" feasibility assumption throughout this report no longer holds — the project now runs on **A100/H100** (bf16). This only *relaxes* the feasibility judgments below; the theme selection (EET) is unaffected. Forward-looking compute decisions live in `docs/core/execution-playbook.md` §2 (*Compute branches*) / D8.
>
> **Naming note (added later):** the candidate written **EET** below is the selected dissertation theme — **The Embodiment Evasion Tax**. This report predates that title, so EET replaces its old scratch label; the other T-numbers remain as scratch labels from the selection process.

---

## 1. Scope & Threat-Model Frame

This project concerns the **integrity** of Vision-Language-Action (VLA) models: whether the
perception → reasoning → action pipeline produces correct, untampered actions, and how that
pipeline is subverted or defended. A VLA takes vision plus a natural-language instruction and
emits robot/agent **actions**, so a corrupted output is a wrong *physical* action rather than
merely wrong text. The de-facto open target is **OpenVLA-7B**, which outperforms RT-2-X (55B)
by 16.5 percentage points in absolute task success across 29 BridgeData V2 tasks despite having
~7× fewer parameters, and whose full stack (checkpoints, fine-tuning code) is publicly released
[1]. Its architecture — a SigLIP+DINOv2 dual visual encoder feeding a Llama-2 backbone, with
each action dimension discretized into 256 quantile-width bins mapped onto the 256 least-used
Llama vocabulary tokens — is well characterized for integrity work [1].

Five integrity **threat channels** scope the review:

1. **Visual adversarial perturbation** — perturbing the camera channel (patches, universal
   perturbations) to flip the emitted action [2].
2. **Language / prompt-instruction injection** — corrupting the natural-language instruction to
   redirect the action [4].
3. **Data poisoning / backdoors** — corrupting training/fine-tuning data so a trigger forces a
   chosen failure or action [3].
4. **Action-space manipulation** — driving the discretized action head toward wrong or
   out-of-bounds control outputs [1].
5. **Supply-chain tampering** — distributing trojaned checkpoints or datasets that recipients
   trust and fine-tune [3].

Confidentiality (extraction) and availability (DoS) are secondary unless they serve the
integrity story. Evaluation is **simulation-only** (LIBERO, SimplerEnv/SIMPLER, ManiSkill,
RLBench) on a single NVIDIA GB10 (DGX Spark-class, ~128 GB unified memory) node; inference,
LoRA/PEFT fine-tuning, small-scale poisoning, and adversarial optimization at ~7B scale are
feasible, while large-scale full pretraining, many-seed full fine-tunes of 7B+, and real-robot
hardware are out of reach.

---

## 2. Method

**Search angles used.** The review surveyed prior work along the five threat channels above plus
two cross-cutting axes: (a) established open base models and their architecture as the *attack
surface* (OpenVLA, SpatialVLA, Octo, OpenVLA-OFT, π0-class); and (b) reproducible
hardware-free *evaluation regimes* (LIBERO suites, BridgeData V2, SimplerEnv). For each channel
the review distinguished what is already demonstrated (DONE) from what remains open (GAPS),
mapping each gap to feasibility on a single GB10 node and to candidate themes T1–T8.

**Sources fetched.** Sixteen sources were retrieved and form the basis of the References section
(§References). Of these, primary papers directly grounding claims in this report are OpenVLA [1],
the adversarial-patch paper [2], BadVLA [3], the VLA textual-attack paper [4], and the AttackVLA
benchmark [6]. The remaining fetched sources [5][7]–[16] were retrieved as part of the
Round-2 literature scan and are listed for provenance; this report cites them only where a
specific claim is grounded in them, and they are noted in §References as fetched-but-not-yet-cited
otherwise.

**Adversarial verification approach.** A four-perspective debate (Novelty, Threat-realism,
Feasibility, Examiner) ranked the candidates, followed by a chair cross-examination that
*re-fetched primary sources* to test each panel claim. Five claims were independently confirmed
against fetched sources and recorded as VERIFIED (§3); five further claims were **refuted** and
must not be restated (§4). Where the cross-examination changed a ranking, the change is recorded
in §6. Claims that could not be grounded in a fetched source are marked `[CITATION NEEDED]`.

---

## 3. Literature Review — What Is Already DONE

**Open VLA base models and architecture (the attack surface).** OpenVLA-7B is the established
baseline: it beats RT-2-X (55B) by 16.5 pp absolute task success across 29 BridgeData V2 tasks
with ~7× fewer parameters, and its checkpoints, fine-tuning code, and architecture (SigLIP+DINOv2
dual encoder → Llama-2 backbone; 256-bin quantile action discretization onto the least-used Llama
tokens) are public [1]. This places a realistic adversary/defender at 7B scale on a single
workstation. (Note: specific claims about LoRA-vs-full parity and quantization losslessness from
this paper are *refuted* — see §4 — and must not be restated.)

**Visual adversarial perturbation (digital + physical).** Adversarial vulnerability of the visual
channel is firmly established. The UPA objective drove OpenVLA adversarial-vs-benign failure rates
to 100% vs 15.3%, 83.4% vs 11.6%, 93.4% vs 20.8%, and 96.7% vs 46.3% on LIBERO-Spatial, -Object,
-Goal, and -Long respectively [2] (VERIFIED). A small adversarial patch in the camera field of
view suffices and transfers to the physical world: a UR10e patch exceeded 43% attack success, with
DoF1-targeted TMA reaching 89.1% failure, confirming a digital-to-physical transfer gap (85.9%
digital vs 43%+ physical) [2] (VERIFIED). Crucially, four input-preprocessing defenses — JPEG
compression, bit-depth reduction, median blur, Gaussian noise — were evaluated and found
insufficient against these patch attacks [2] (VERIFIED).

**Backdoor / trojan attacks (supply-chain integrity).** Backdooring VLAs is demonstrated. BadVLA
uses two-stage objective-decoupled optimization to inject backdoors into OpenVLA variants (and
SpatialVLA) while preserving clean-task accuracy, framing the threat as Training-as-a-Service
supply-chain tampering [3]. The newer AttackVLA benchmark consolidates adversarial and backdoor
attacks against VLAs across multiple model families [6]. (Specific BadVLA persistence/robustness
numbers and the "first to expose" framing are *refuted* — see §4.)

**Language / instruction-channel attacks.** The language input is a confirmed integrity channel.
Textual (GCG-style) attacks on VLAs optimize substantially faster than on aligned chatbots:
successful attacks are found in 30–110 GCG steps (3–10 minutes on H100), versus over an hour per
chatbot jailbreak on the same hardware [4] (VERIFIED), indicating VLAs lack chatbot-grade safety
hardening and the language pathway is a low-cost integrity surface.

**Established simulation benchmarks.** A reproducible, hardware-free evaluation regime is reused
across the attack literature: LIBERO (Spatial/Object/Goal/Long), BridgeData V2, and SimplerEnv,
with OpenVLA and SpatialVLA as victim models and published benign baselines (e.g. LIBERO benign
failure ~11–46%) as fixed reference points [1][2][3]. This aligns directly with the project's
simulation-only, single-node constraints.

---

## 4. Open GAPS

Gaps below are tagged **[VERIFIED]** (the panel's claim that the gap is open was independently
confirmed against a fetched source during cross-examination) or **[unverified]** (single-source
or framing-level; treat as motivating context, not a standalone contribution).

1. **VLA-specific adversarial DEFENSES that beat the four broken input-preprocessing baselines**
   **[VERIFIED].** Both [2] and [3] show JPEG/bit-depth/median-blur/Gaussian-noise (and
   fine-tuning) fail; [2] calls for a paradigm shift in VLA training strategies. No robust
   VLA-tailored defense currently exists [2][3][1].

2. **Cross-architecture / cross-environment generalisation of attacks beyond OpenVLA + SpatialVLA**
   **[VERIFIED].** Patches are stated to be model-specific [2]; BadVLA tests only two families
   while claiming architectural vulnerability [3]. Cross-cutting verification note: the chair
   confirmed AttackVLA [6] benchmarks attacks per-model on OpenVLA/SpatialVLA/π0-fast but does
   **not** publish a cross-*model* transfer matrix (attack crafted on A applied to B) —
   correcting an earlier panel claim that this was "already published."

3. **Action-tokenization (256-bin quantile → Llama vocabulary) as an integrity surface**
   **[VERIFIED].** OpenVLA describes the 256-bin discretization purely for performance [1].
   The chair fetched AttackVLA [6] and confirmed it independently names the binning-based
   tokenizer as the *more attack-susceptible* component, yet neither [6] nor any other fetched
   attack paper analyzes or defends the action head as an integrity surface. The architectural
   gap is verified **open after** the literature moved.

4. **Differential security of LoRA/PEFT vs full fine-tuning under poisoning** **[VERIFIED].** No
   fetched work studies whether low-rank updates embed backdoors more or less readily than full
   fine-tuning [1][3].

5. **Small-scale / clean-label data poisoning of VLA fine-tuning sets** **[VERIFIED].** The
   dose-response of poisoning is uncharacterised in the primary sources [3][1]. **Partial refute
   recorded:** the chair verified that newer targeted-backdoor papers (cited from the fetched
   AttackVLA references, see §6) occupy part of the poison-*rate* territory; the **clean-label**
   arm (no visible trigger, plausible labels) remains the unclaimed kernel.

6. **Targeted backdoors steering a VLA to a specific wrong action** **[VERIFIED open at scoping,
   but subsequently REFUTED as novel].** BadVLA explicitly scoped out targeted backdoors as
   future work [3]. **Refuted during cross-examination:** the chair verified that targeted/
   goal-oriented backdoors on OpenVLA+LIBERO are now demonstrated multiple times in the recent
   literature surfaced via AttackVLA [6] (see §6, honorable mentions). Building the *first*
   targeted VLA backdoor is no longer possible; only an activation-detection sub-study retains
   marginal novelty.

7. **Checkpoint supply-chain integrity: data-free, trigger-agnostic auditing of trojaned weights**
   **[VERIFIED].** OpenVLA releases checkpoints with no signing/hashing/integrity mechanism [1];
   BadVLA frames Training-as-a-Service as critical but proposes no detection/auditing [3]. The
   chair verified that the recent attack wave (via [6]) takes the trojan as *given* and proposes
   no pre-deployment, data-free, trigger-agnostic audit.

8. **Quantifying digital-to-physical / sim-attack transfer mechanisms** **[VERIFIED open, scoped].**
   [2] notes the 43% physical vs ~86% digital gap but gives no systematic ablation of causes or
   mitigations. A simulation-only study can characterise *simulated* nuisance factors only and
   cannot close the true sim-to-real gap [2].

9. **Adversarial/injected LANGUAGE instructions as an integrity threat (vs OOD generalisation)**
   **[VERIFIED].** OpenVLA never investigates injected instructions [1]; the textual-attack work
   shows VLA language attacks optimize in minutes [4]. The chair confirmed AttackVLA [6] covers
   RoboGCG/textual attacks generically but does **not** cover the specific pairing of a benign
   visual scene + injected language suffix redirecting the *physical* action with a goal-action
   consistency defense.

10. **Robustness contribution of individual visual features (SigLIP vs DINOv2 vs combined)**
    **[VERIFIED, with constraint].** No adversarial analysis of the dual-encoder design exists
    [1][2]. **Constraint from refute (§4):** the OpenVLA "frozen vs fine-tuned encoder is
    necessary" framing is **refuted** by later work and must not be restated; any encoder study
    must address only the robustness dimension.

11. **Safety-margin quantification linking sub-90% clean reliability to harm under attack**
    **[unverified].** Single-source, framing-level [1]; best used to motivate a concrete
    defense/benchmark, not to stand alone.

12. **Integrity of additional input channels (multi-image, proprioception, history)**
    **[unverified].** Forward-looking and dependent on VLA variants that consume these channels
    being runnable on one node [1]; flag as exploratory.

**Refuted claims (do NOT restate as established):** (a) freezing the visual encoder is necessary
for adequate performance (47.0% vs 69.7%); (b) 4-bit quantization is lossless while 8-bit drops to
58.1%; (c) BadVLA is the *first* to expose VLA backdoors / its specific persistence (90.6–97.9%)
and JPEG-robustness (97.7% at q=20%) numbers; (d) the specific GCG action-elicitation rates
(96.5/93.8/97.5/77.3%) and the "first study of adversarial attacks on low-level VLA actuators"
framing; (e) the blanket claim that *no* effective defenses have been demonstrated. These were
flagged refuted in cross-examination and are excluded from the DONE section.

---

## 5. Candidate Themes

| id | title | channel | stance | key gap |
|----|-------|---------|--------|---------|
| T1 | Targeted backdoors that steer VLAs to a specific wrong action, paired with activation-anomaly detection | data poisoning/backdoor → action-space manipulation | attack | Targeted backdoors (steering to a chosen action) — *now refuted as novel, see §6* |
| T2 | Cross-architecture transferability benchmark for VLA backdoor triggers and adversarial patches | visual perturbation + backdoor trigger (cross-model/env transfer) | benchmark | Cross-architecture/cross-environment generalisation beyond OpenVLA+SpatialVLA |
| T3 | Dose-response of small-scale & clean-label data poisoning in VLA fine-tuning | data poisoning of fine-tuning sets | benchmark | Small-scale / clean-label poisoning dose-response |
| T4 | Does LoRA/PEFT change a VLA's backdoor susceptibility vs full fine-tuning? | data poisoning/backdoor under different fine-tuning regimes | benchmark | Differential security of LoRA/PEFT vs full fine-tuning |
| T5 | Action-tokenization plausibility defense: hardening OpenVLA's 256-bin action head | action-space manipulation + perturbation/backdoor surfacing as anomalous action tokens | defense | Action-tokenization (256-bin → Llama vocab) as an integrity surface |
| T6 | Action-consistency runtime monitor: temporal + cross-modal plausibility checks | visual perturbation + backdoor (anomalous emitted actions) | defense | VLA-specific defenses beating the four broken preprocessing baselines |
| EET | Cross-modal instruction injection on a benign scene, with a goal-action consistency defense | language/prompt injection → action-space manipulation | defense | Adversarial/injected language instructions as an integrity threat |
| T8 | Trojaned-checkpoint supply-chain audit: detecting hidden backdoors in distributed weights | supply-chain tampering / trojaned weights | defense | Data-free, trigger-agnostic checkpoint auditing before deployment |

---

## 6. Multi-Agent Debate Summary

Four perspectives (Novelty, Threat-realism, Feasibility, Examiner) ranked the candidates over two
rounds; a chair cross-examined the panel by re-fetching primary sources.

**Key agreements.**
- The **most-cited open gap** is a VLA-specific defense that beats the four broken
  input-preprocessing baselines [2][3]; defense-stanced themes (T5, T6, T8) cluster around it.
- **Reproducibility favours inference-only / frozen-checkpoint designs** on a bandwidth-limited
  single node: T5 and T6 (no retraining for the core mechanism) scored highest on feasibility,
  T3/T4/T8 require LoRA fine-tunes.
- The **UADA/UPA/TMA reproduction tax** is the largest hidden non-fine-tune cost in the set;
  any attack-reproducing theme (T2, T5, T6) must validate against published OpenVLA failure rates
  *before* claiming a defense gain [2].
- **Overlap policing** held: T5 stays token/clamp-level; T6 stays trajectory/cross-modal; T8 stays
  checkpoint/weight-level and trigger-agnostic; the (now-demoted) T1 probe was inference-time and
  known-trigger.

**Key disagreements and what cross-examination changed.**
- **T1 (targeted-backdoor novelty)** — the single biggest cross-round disagreement. Round-1 ranked
  T1 #1 on impact; Round-2 reversed it on newly-surfaced papers. The chair WebFetched the primary
  sources and confirmed that targeted/goal-oriented backdoors on OpenVLA+LIBERO are **already
  published** (GoBA arXiv:2510.09269; DropVLA arXiv:2510.10932 reporting 98%+ ASR at ~0.31% poison;
  BackdoorVLA inside AttackVLA arXiv:2511.12149 [6] reporting ~58.4% average targeted success).
  **Resolution:** the reversal was correct — T1's attack contribution is pre-empted and it drops
  to honorable mention. *(These three arXiv IDs were verified by the chair via WebFetch but their
  landing pages are not in the fetched-sources list other than AttackVLA [6]; the GoBA/DropVLA
  IDs are therefore `[CITATION NEEDED]` pending a fetched URL.)*
- **T2 "already published" claim** — Round-2 asserted AttackVLA publishes the cross-model transfer
  matrix. The chair fetched AttackVLA [6] and verified it benchmarks attacks *per-model*, not
  cross-model transfer; the kill-shot was **overstated**. T2 still ranks last on its independent
  roster-collapse feasibility risk, but with corrected reasoning.
- **SilentDrift vs T6** — Round-2 claimed a recent defense (SilentDrift) defeats T6's
  temporal-smoothness signal. The chair fetched it and verified it requires action-chunking +
  delta-pose and explicitly does **not** apply to OpenVLA (single-action, absolute bins); the
  threat to T6's OpenVLA-LIBERO stack was **overstated**, though it proves adaptive smoothness
  attacks are constructible, making T6's adaptive-attack arm mandatory. *(SilentDrift
  arXiv:2601.14323 was verified by the chair via WebFetch but is not in the fetched-sources list;
  `[CITATION NEEDED]` pending a fetched URL.)*
- **T5 impact vs novelty/feasibility** — Threat-realism and Examiner worried T5's clamp only
  catches crude out-of-bounds attacks; Novelty and Feasibility rated it top. The chair fetched
  AttackVLA [6], which *independently confirms* the tokenizer-susceptibility gap, and placed T5
  #1 **with a binding requirement**: a mandatory adaptive in-bounds attacker arm and
  benign-statistics-derived (never hand-tuned) plausibility bounds.
- **T3 rank spread (#1→#6 across panels)** — resolved to #2: safest deliverable, clean-label arm
  survives the recent poisoning literature as the unclaimed kernel, but the rate-curve half is
  partly pre-empted, so it sits behind T5's wholly-unclaimed architectural gap.

**Net effect of cross-examination:** T1 fell from front-runner to honorable mention (refuted
novelty); T8 rose (verified-open defensive surface); T2's kill-shot was corrected but its ranking
held; T5 was confirmed #1 with an attached binding plan requirement.

---

## 7. Ranked Recommendations (1 → 5)

### #1 — T5: Action-Tokenization Plausibility Defense
**Justification.** The unique theme where the safest-to-execute choice is also the most-unclaimed
gap, and where post-debate verification *strengthened* the case: AttackVLA [6] names the 256-bin
tokenizer as the more attack-susceptible component yet proposes no action-head defense, and no
fetched attack paper hardens the action-tokenization surface [1][6]. A clean integrity story at the
perception → action seam, publishable on a positive **or** negative result.
**Feasibility on GB10+sim.** Inference-only guard layered on a frozen 4-bit OpenVLA-7B (fits
128 GB); zero retraining for the core plausibility/clamp/temporal checks; deterministic,
write-once per-bin logs; figures regenerable from a script. The main hidden cost is reproducing
one adversarial objective (UPA) faithfully [2].
**Concrete next steps.** (1) Stand up OpenVLA-7B LIBERO checkpoints (4-bit). (2) Reproduce **one**
patch objective (UPA) and validate against published OpenVLA failure rates [2] before any defense
claim. (3) Build the guard: per-dimension range clamp + temporal action-history consistency +
benign-rollout-derived kinematic plausibility on decoded tokens; characterise which bins each
attack pushes toward. (4) Add a **mandatory adaptive in-bounds attacker** arm. (5) Report recovered
LIBERO task-success vs benign+attacked baselines, false-positive rate on clean rollouts, fraction
of attacked outputs forced into rarely-used extreme bins, and recovered-success under the adaptive
attacker; reproduce the four broken preprocessing defenses as the comparison floor [2]. Pin seeds;
quarantine artifacts under `artifacts/untrusted/`.

### #2 — T3: Dose-Response of Small-Scale & Clean-Label Data Poisoning
**Justification.** The safest path to a defensible dissertation; the cleanest single-variable
design. The **clean-label arm** (no visible trigger) is the verified-unclaimed supply-chain kernel;
the rate-curve half is partly pre-empted by recent targeted-backdoor papers (see §6), so the
write-up must foreground clean-label and cite those papers for positioning [3][1].
**Feasibility on GB10+sim.** Each dose point is one LoRA fine-tune plus rollouts; no
adversarial-patch reproduction tax. A scoped grid (5–7 doses × 2–3 seeds × 1–2 LIBERO suites) is
in-budget; seed count is capped, so error bars will be wide and must be reported honestly, not
smoothed. Guaranteed-publishable on a negative (clean-label may fail at 7B).
**Concrete next steps.** OpenVLA-7B + released LoRA recipe; LIBERO Spatial+Object (pre-commit);
poison rate {0.5,1,2,5,10}% × 2–3 seeds, one fixed LoRA config; arms (a) triggered poison, (b)
clean-label poison; attack-success locked to a *specific* attacker-chosen target action (scored
"reached target", not "task failed"). Pin seeds; record corpus provenance; quarantine poisoned
data/checkpoints. Fix the fine-tuning method to stay disjoint from T4.

### #3 — T8: Trojaned-Checkpoint Supply-Chain Audit
**Justification.** The strongest genuinely-open *defensive* contribution after the attack space
saturated: the chair verified the recent attack wave (via [6]) takes the trojan as given and
proposes no data-free, trigger-agnostic, pre-deployment audit [3][6]. Most direct answer to the
supply-chain channel (hub checkpoints are downloaded and trusted daily). Ranks below T5/T3 only
because the headline detection-rate claim is statistically exposed (small trojan zoo) and detector
transfer to a 7B action head is unproven — both legitimate but neither makes the result known.
**Feasibility on GB10+sim.** Building the zoo reuses in-budget LoRA backdoor fine-tunes; detectors
are inference/analysis-only on frozen weights + a small benign probe set; trigger-reconstruction
scanning over 224×224 inputs is the heaviest detector (budget as optional). Publishable on an
honest negative.
**Concrete next steps.** Build a small clean+trojaned OpenVLA-7B LoRA zoo (vary trigger × target),
each behaviourally confirmed on LIBERO; detectors = action-head weight-distribution outliers +
neuron-activation clustering on a benign probe set + optional trigger-reconstruction; report
detection rate, FPR on clean checkpoints, and evasion envelope; baseline = chance + naive
weight-norm. Pin seeds; strict quarantine/provenance for the zoo. Stay strictly weight-level and
trigger-agnostic to avoid overlap with the (demoted) T1 probe.

### #4 — T6: Action-Consistency Runtime Monitor
**Justification.** Targets the most-cited open gap (beat the four broken preprocessing baselines)
[2][3] with the most reproducible design (zero-retraining wrapper over logged action sequences).
The chair verified SilentDrift does **not** directly defeat a T6 monitor on the OpenVLA-LIBERO
stack (it requires action-chunking + delta-pose), but it proves adaptive smoothness attacks are
constructible — so T6's defended gain is structurally hostage to an adaptive adversary, placing it
behind T8.
**Feasibility on GB10+sim.** OpenVLA-7B (4-bit) inference on LIBERO; no fine-tune for the core
detector; cost dominated by replaying the fixed published attacks [2]. Shares the UADA/UPA/TMA
reproduction tax.
**Concrete next steps.** Monitor = (a) temporal smoothness on the 7-DoF stream + (b)
language-goal-embedding vs predicted-action-direction consistency; fire → hold/abort. Reproduce
fixed UADA/UPA/TMA + a backdoor, validate vs published numbers first [2], then add an **adaptive**
smoothness/consistency-aware attacker. Report recovered task-success, FPR on benign (incl. fast
motions), detection latency, and recovered-success under the adaptive attacker; baseline = the four
broken preprocessing defenses. Pin seeds; quarantine artifacts. Keep trajectory/cross-modal to
avoid overlap with T5.

### #5 — EET: Cross-Modal Instruction Injection + Goal-Action Consistency Defense
**Justification.** The most original *surviving* attack-flavoured contribution and the cheapest
theme to run. The chair confirmed AttackVLA [6] covers textual attacks generically but **not** the
specific pairing EET proposes (benign scene + injected language suffix redirecting the *physical*
action, paired with a goal-action consistency defense). The language pathway is a verified low-cost
surface (GCG converges in minutes) [4] and a realistic deployment entry point. Ranks #5 because the
consistency defense risks circularity and may need a lightweight reference judge, and "meaningful
redirection vs mere denial" must be pinned tightly or it collapses into the already-published
denial regime.
**Feasibility on GB10+sim.** Text-only GCG needs no adversarial-patch reproduction (the largest
hidden cost), so cost is rollout-only; the consistency defense is an embedding-comparison wrapper.
Cheapest theme on the node.
**Concrete next steps.** OpenVLA-7B LIBERO (4-bit); GCG-style adversarial suffix following the
published VLA textual-attack method [4] as a fixed adversary, visual scene held benign, targeting a
specific reachable attacker-chosen action region; metrics = attack-success ("reached target"),
optimization cost (GCG steps), defended success, benign FPR. Defense = sanitized-language-goal vs
predicted-action consistency → abort on mismatch, threshold calibrated on benign instruction-action
pairs. Keep the attack arm authorised/contained per project ethics; no real-robot implication.
Avoid restating OpenVLA's refuted encoder-necessity claims (§4).

---

## 8. Recommended #1 + Immediate Next Actions (2-Week Starter Plan)

**Recommended #1: T5 — Action-Tokenization Plausibility Defense.** After re-verifying the Round-2
literature against primary sources, this is the one theme where the gap got *stronger*: AttackVLA
[6] names OpenVLA's 256-bin tokenizer as the more attack-susceptible component yet proposes no
action-head defense, and no fetched paper analyzes or hardens the action-tokenization surface
[1][6]. It has the best feasibility/reproducibility profile for a single GB10 sim-only node in
3–4 months (frozen 4-bit checkpoint, inference-only guard, deterministic write-once per-bin logs,
script-regenerable figures), tells a clean integrity story at the perception → action seam, and is
publishable on a positive **or** negative result.

**Non-negotiable plan requirements** (from cross-examination): (a) include an **adaptive in-bounds
attacker** as a first-class evaluation arm; (b) derive plausibility bounds from **benign-rollout
statistics + a simulated-arm kinematic model, never hand-tuned**, so the defense claim is scoped to
exactly the attack classes it can stop.

**Fallback #1:** if a supervisor prefers the lowest-execution-risk, guaranteed-deliverable
measurement contribution, switch to **T3** — its clean-label arm is the verified-unclaimed kernel.

**Two-week starter plan (success criteria stated up front):**

*Week 1 — environment, baseline, and one attack.*
- **Day 1–2.** Confirm with the supervisor: research question, stance (defense), and that any
  required ethics/risk approval is in place. Confirm before `git init`. Set up the repo skeleton
  (`docs/`, `src/`, `configs/`, `results/` write-once, `artifacts/untrusted/` gitignored).
  *Done = repo skeleton committed (after confirmation), supervisor sign-off recorded.*
- **Day 2–3.** Stand up OpenVLA-7B LIBERO-finetuned checkpoints (4-bit) on the GB10 node; capture
  the exact environment (lockfile, library + CUDA versions, hardware) and dataset/checkpoint
  provenance (source, URL, hash, retrieval date). *Done = a benign LIBERO-Spatial rollout
  reproduces a published benign baseline within tolerance [2], with seeds pinned.*
- **Day 4–5.** Instrument the action head to log per-dimension decoded bin indices on every step;
  produce the benign per-bin action distribution on LIBERO-Spatial+Object. *Done = write-once
  per-bin logs + a script that regenerates the benign bin-distribution figure.*

*Week 2 — reproduce one adversary, prototype the guard.*
- **Day 6–8.** Reproduce **one** patch objective (UPA) and validate OpenVLA failure rates against
  the published numbers [2] before any defense work. *Done = reproduced UPA failure rate matches
  [2] within tolerance; mismatch is logged as a negative result, not hidden.*
- **Day 9–11.** Prototype the thin guard (per-dimension range clamp + temporal action-history
  consistency + benign-statistics kinematic plausibility) as an inference-time wrapper; log which
  bins the attack pushes toward. *Done = guard runs end-to-end on attacked rollouts and emits
  recovered-success + extreme-bin-fraction metrics.*
- **Day 12–14.** Draft the experiment design doc in `docs/plans/` specifying the **adaptive
  in-bounds attacker** arm, the benign-derived plausibility-bound procedure, the metric set, the
  four-preprocessing-defense floor, the seed grid, and the pre-committed LIBERO suites. *Done = a
  reviewed plan in `docs/plans/` with verifiable success criteria and a go/no-go checkpoint for
  scaling to Goal/Long.*

---

## 9. Limitations of This Review

- **Source coverage.** Conclusions rest on the sixteen fetched sources (§References); grounded
  claims draw chiefly on [1][2][3][4][6]. Several themes hinge on recent papers (GoBA, DropVLA,
  SilentDrift) that the chair verified by WebFetch during cross-examination but whose landing-page
  URLs are **not** in the fetched-sources list; those arXiv IDs are flagged `[CITATION NEEDED]`
  in §6 and must be re-fetched and properly cited before relying on them in the dissertation.
- **Recency caveats.** The VLA-attack literature is moving fast; the targeted-backdoor space
  saturated *between* review rounds (T1's reversal). Any chosen theme must be re-checked against
  the latest literature at design time, because today's open gap may close within weeks (§ this
  is precisely what happened to T1 and partly to T3's rate-curve).
- **Refuted claims excluded.** Five specific numeric/framing claims (encoder-necessity,
  quantization losslessness, BadVLA "first"/persistence numbers, GCG action-elicitation rates, the
  blanket "no defenses exist" claim) were refuted and deliberately omitted from the DONE section;
  the report must not be read as endorsing them.
- **Simulation-only and single-node.** No real-robot or large-scale full-fine-tune evidence; all
  feasibility judgments and any sim-to-real claims are bounded to simulation on one GB10 node.
  Themes depending on additional input channels (multi-image, proprioception, history) were left
  exploratory because runnable variants are not confirmed.
- **Not covered.** Confidentiality (model/data extraction) and availability (DoS) were treated as
  out of primary scope; cross-modal chain-of-thought attacks, RL post-training defenses, and the
  broader embodied-AI survey landscape appear in fetched sources [7][8][9][10][13][14][15][16] but
  were not analysed in depth here and remain available for a deeper literature review in the next
  phase.

---

## References

*URLs below were actually fetched during this review. Entries [5], [7]–[16] were retrieved as part
of the Round-2 literature scan and are listed for provenance; this report grounds specific claims
only in [1][2][3][4][6]. The GoBA / DropVLA / SilentDrift arXiv IDs referenced in §6/§9 were
verified by the chair via WebFetch but their landing-page URLs are not in this fetched list and are
therefore marked `[CITATION NEEDED]`.*

[1] OpenVLA: An Open-Source Vision-Language-Action Model. https://arxiv.org/abs/2406.09246 (also https://arxiv.org/html/2406.09246v1 ; https://proceedings.mlr.press/v270/kim25c.html)
[2] Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics. https://arxiv.org/abs/2411.13587 (also https://arxiv.org/html/2411.13587v2 ; https://arxiv.org/pdf/2411.13587)
[3] BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization. https://arxiv.org/abs/2505.16640
[4] Adversarial Attacks on Robotic Vision-Language-Action Models. https://arxiv.org/abs/2506.03350
[5] Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models. https://arxiv.org/abs/2510.13237 *(fetched; not cited for a specific claim above)*
[6] AttackVLA: Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models. https://arxiv.org/abs/2511.12149
[7] Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success — OpenVLA-OFT. https://arxiv.org/abs/2502.19645 *(fetched; not cited for a specific claim above)*
[8] TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches. https://arxiv.org/abs/2603.23117 *(fetched; not cited for a specific claim above)*
[9] RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models. https://arxiv.org/abs/2511.01331 *(fetched; not cited for a specific claim above)*
[10] VLA-Risk: Benchmarking Vision-Language-Action Models with Physical Robustness. https://openreview.net/forum?id=31EjDFwFEe *(fetched; not cited for a specific claim above)*
[11] Octo: An Open-Source Generalist Robot Policy. https://arxiv.org/abs/2405.12213 *(fetched; not cited for a specific claim above)*
[12] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models. https://arxiv.org/abs/2511.21192 *(fetched; not cited for a specific claim above)*
[13] Inject Once Survive Later: Backdooring Vision-Language-Action Models to Persist Through Downstream Fine-tuning. https://arxiv.org/abs/2602.00500 *(fetched; not cited for a specific claim above)*
[14] Altered Thoughts, Altered Actions: Probing Chain-of-Thought Vulnerabilities in VLA Robotic Manipulation. https://arxiv.org/abs/2603.12717 *(fetched; not cited for a specific claim above)*
[15] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations. https://arxiv.org/abs/2510.00037 *(fetched; not cited for a specific claim above)*
[16] Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks. https://arxiv.org/abs/2502.13175 *(fetched; not cited for a specific claim above)*

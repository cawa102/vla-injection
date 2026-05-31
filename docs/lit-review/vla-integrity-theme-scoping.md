# VLA Integrity Security — Theme-Scoping Literature Review & Ranked Research Themes

> **Phase:** Scope → Literature review (CLAUDE.md §3, steps 1–2)
> **Date:** 2026-05-30
> **Produced by:** a 67-agent deep-research workflow (6 search angles → 16 fetched sources →
> claim extraction → 3-lens adversarial verification → gap map → 2 theme generators →
> 4-perspective panel, 2 debate rounds → chair synthesis). Run ID `wf_5cba1665-eba`.
> **Constraints assumed:** single **NVIDIA GB10 Grace Blackwell** node (~128 GB unified
> memory; LoRA/PEFT fine-tuning of a ~7B VLA feasible, large-scale full pretraining /
> many-seed full fine-tunes infeasible, no real robot); **simulation-only** evaluation
> (LIBERO / SimplerEnv); ~3–4 month single-author MSc; stance open (attack/defense/benchmark).
>
> **⚠️ This is an AI-generated research scaffold for the author to review and rewrite
> (CLAUDE.md §5).** Citations marked **[verified]** were independently re-fetched from
> arXiv on 2026-05-30 (title confirmed). Citations marked **[unverified]** were surfaced
> by the search agents but **not** independently checked — verify each before citing.

---

## 1. Scope & Threat-Model Frame

**System under study.** A Vision-Language-Action (VLA) model maps `(camera image, natural-language
instruction)` → **robot action tokens**. Because the output is a control signal, an integrity
failure is a *wrong physical action*, not merely wrong text. The de-facto open target is
**OpenVLA-7B** [1]: SigLIP + DINOv2 dual visual encoder → Llama-2 backbone → each of 7 action
dimensions discretised into **256 quantile-width bins** mapped onto the 256 least-used Llama
vocabulary tokens.

**Integrity = the perception → reasoning → action pipeline producing correct, untampered actions.**
Five threat channels are in scope:

1. **Visual adversarial perturbation** — patches/perturbations on the camera channel.
2. **Language / prompt-instruction injection** — malicious or injected instructions.
3. **Data poisoning / backdoors** — training-time trojans (incl. trojaned checkpoints).
4. **Action-space manipulation** — corrupting the action head / tokenization seam.
5. **Supply-chain tampering** — trojaned weights distributed via public hubs.

Confidentiality (extraction) and availability (DoS) are secondary unless they serve the
integrity story.

---

## 2. Method (how this review was conducted)

- **Search:** 6 parallel web-search angles — VLA model landscape · visual adversarial ·
  poisoning/backdoors · prompt/instruction injection · defenses/detection · surveys/taxonomies/benchmarks.
  44 candidate sources found; **16 fetched** (round-robin dedup so every angle is represented).
- **Extraction:** 157 "done" claims + 105 "gap" claims pulled from the 16 sources (all accessible),
  each tied to a source URL with a verbatim quote where possible.
- **Adversarial verification:** the 10 most load-bearing claims were each challenged by 3
  independent lenses (recency / counter-example / over-statement); a claim was killed only if
  ≥2 of 3 refuted it. **5 of 10 survived; 5 were refuted** (see §10).
- **Theme generation → debate:** 2 generators (attack-leaning + defense/benchmark-leaning) → merged
  to 8 distinct candidates → a 4-perspective panel (Novelty · Feasibility/Reproducibility ·
  Threat-realism · MSc Examiner) scored all 8, then **cross-examined each other** for a second
  round, then a chair synthesised the final 1→5.

**Known defect in this run:** the final prose-report agent hit a transient `529 Overloaded`
error, so this document was assembled by hand from the workflow's *structured* output (rankings,
candidate specs, gap map, sources) — which were all complete. The 8 load-bearing citations were
then independently re-verified (§10).

---

## 3. Literature Review — What Is Already Done

| # | Established result | Key evidence | Sources |
|---|---|---|---|
| D1 | **Open VLA base model + attack surface is well-characterised.** OpenVLA-7B beats RT-2-X (55B) by +16.5 pp on BridgeData V2 with 7× fewer params; LoRA (rank 32/64, ~1.4% params) ≈ full FT (68.2% vs 69.7%); 4-bit quant is ~lossless (71.9% vs 71.3%). The realistic adversary/defender works at **7B scale on one workstation.** | Architecture + parity numbers | [1] |
| D2 | **Visual adversarial attacks work, digitally and physically.** Three objectives (UADA/UPA/TMA) push OpenVLA LIBERO failure to ~100% vs low benign baselines (e.g. UPA 100% vs 15.3%). A 5%-area patch in the camera FoV suffices, transfers black-box to LLaRA, and works on a real UR10e (>43% attack success; digital→physical gap ~86%→43%). **Four input-preprocessing defenses (JPEG, bit-depth, median-blur, Gaussian) were tested and found insufficient.** | UADA/UPA/TMA + UR10e | [2] |
| D3 | **VLAs can be backdoored (supply-chain trojan).** BadVLA reaches ~100% attack success while preserving clean-task accuracy, across 4 OpenVLA variants + SpatialVLA; survives JPEG (ASR 97.4 @ q=20%). Framed as a Training-as-a-Service threat. | Objective-decoupled backdoor | [3] |
| D4 | **The language channel is a cheap integrity surface.** GCG-style textual attacks on VLAs converge in **30–110 steps (3–10 min on H100)** — far cheaper than chatbot jailbreaks — i.e. VLAs lack chatbot-grade alignment hardening. | GCG step counts | [4] |
| D5 | **Targeted backdoors now exist (post-dates BadVLA's "future work").** GoBA, DropVLA (98%+ ASR @ 0.31% poison), and BackdoorVLA-within-AttackVLA (58.4% avg targeted success) all demonstrate *targeted* (chosen-action) backdoors on OpenVLA+LIBERO. | Three independent 2025 papers | [5][6][7] |
| D6 | **Reproducible sim eval regime is standardised.** LIBERO (Spatial/Object/Goal/Long), BridgeData V2, SimplerEnv with OpenVLA/SpatialVLA victims; published benign baselines (~11–46% failure) give fixed reference points — fits the single-node, sim-only MSc. | Benchmark conventions | [1][2][3] |

---

## 4. Open Gaps (after adversarial verification)

Gaps the workflow judged **verified-open** (grounded in a surviving claim and/or multiple sources):

- **G1 — VLA-specific defenses that beat the four broken preprocessing baselines.** [2][3] both
  show JPEG/bit-depth/median-blur/Gaussian (and fine-tuning) fail; [2] calls for a "paradigm shift."
  *The single most defensible MSc direction.* **[verified]**
- **G2 — Action-tokenization (256-bin) as an integrity surface.** OpenVLA's binning is described
  purely for performance; never analysed or hardened as an integrity surface — and AttackVLA [5]
  *names it as the more susceptible component* yet defends only the input side. **[verified]** *(→ #1)*
- **G3 — Trojaned-checkpoint supply-chain auditing (data-free, trigger-agnostic).** Every backdoor
  paper [3][5][6][7] takes the trojan as *given*; none audits a *received* checkpoint before
  deployment. **[verified]** *(→ #3)*
- **G4 — Small-scale / clean-label poisoning dose-response.** Poison-rate is now partly covered by
  DropVLA (0.31%); but the **clean-label** (no visible trigger) arm is unclaimed. **[verified]** *(→ #2)*
- **G5 — LoRA/PEFT vs full-FT backdoor susceptibility.** OpenVLA reports LoRA-vs-full parity for
  *clean* accuracy only; security differential is unstudied. **[verified]** (confounded on this node — see honorable mentions)
- **G6 — Cross-architecture / cross-environment attack transfer** beyond OpenVLA↔SpatialVLA. **[verified]** (roster-collapse risk)
- **G7 — Adversarial/injected *language* redirecting the *physical* action** on a benign scene. **[verified]** *(→ #5)*
- **G8 — Targeted backdoors (chosen wrong action).** *Now CLOSED* — see §10 (refuted): published 3× over [5][6][7].

Lower-confidence / framing-level gaps (use to *motivate*, not stand alone): digital→physical
transfer ablation in sim; per-encoder (SigLIP vs DINOv2) robustness; safety-margin quantification
[**unverified**]; integrity of extra input channels (proprioception/history) [**unverified**].

---

## 5. Candidate Themes (8 distilled)

| ID | Title | Channel | Stance | Key gap |
|----|-------|---------|--------|---------|
| **T1** | Targeted backdoor steering a chosen wrong action + activation detection | poisoning→action | attack | G8 *(now closed)* |
| **T2** | Cross-architecture transferability benchmark for triggers & patches | visual+backdoor | benchmark | G6 |
| **T3** | Dose-response of small-scale & **clean-label** poisoning | poisoning | benchmark | G4 |
| **T4** | Does LoRA/PEFT change backdoor susceptibility vs full FT? | poisoning | benchmark | G5 |
| **T5** | **Action-tokenization (256-bin) plausibility defense** | action-space | defense | G2 |
| **T6** | Action-consistency runtime monitor (temporal + cross-modal) | visual+backdoor | defense | G1 |
| **T7** | Cross-modal instruction injection + goal-action consistency defense | language | defense | G7 |
| **T8** | **Trojaned-checkpoint supply-chain audit** (data-free, trigger-agnostic) | supply-chain | defense | G3 |

---

## 6. Multi-Agent Debate — Key Disagreements & How They Resolved

- **T1 (targeted backdoor) — biggest reversal.** Round-1 panel ranked it **#1**; Round-2 Novelty
  critic reversed to #6 citing three new papers. All three were **verified real** (GoBA 2510.09269,
  DropVLA 2510.10932, AttackVLA/BackdoorVLA 2511.12149). Resolution: **T1's attack contribution is
  genuinely pre-empted → demoted to honorable mention.** High impact cannot restore lost novelty.
- **T2 "already published" — over-stated.** The critic claimed AttackVLA publishes a cross-*model*
  transfer matrix; on re-fetch, AttackVLA benchmarks attacks *per-model independently*, not
  cross-model transfer. T2's specific deliverable is *not* fully published — but its roster-collapse
  feasibility risk is independently disqualifying, so it stays last (corrected reasoning, same place).
- **SilentDrift vs T6 — over-stated.** The claim that SilentDrift (2601.14323) defeats T6 was wrong:
  SilentDrift *requires action-chunking + delta-pose and explicitly does not apply to OpenVLA*
  (single-action, absolute bins). So it doesn't defeat a T6 monitor on the OpenVLA-LIBERO stack —
  but it proves adaptive smoothness attacks are constructible → T6's adaptive arm is **mandatory**.
- **T5 impact vs novelty/feasibility.** Threat-realism & Examiner worried the clamp only catches
  *crude out-of-bounds* actions. Resolution: rank T5 #1 but **attach a binding requirement** — a
  mandatory *adaptive in-bounds* attacker arm + plausibility bounds derived from benign-rollout
  statistics (never hand-tuned).
- **T3 spread (#1↔#6).** Examiner #1 (fits all criteria); Feasibility conceded #2 after confirming a
  ~12 h/LoRA run makes the scoped grid deliverable. Lands #2 — safest deliverable, clean-label arm survives.
- **Overlap policing.** No two top-5 themes share a core method: T5 = token/clamp level; T6 =
  trajectory/cross-modal; T8 = checkpoint/weight level, trigger-agnostic; T3 = training-data;
  T7 = language channel.

---

## 7. Ranked Recommendations (1 → 5)

### 🥇 #1 — T5 · Action-Tokenization Plausibility Defense
*Harden OpenVLA's 256-bin action head as an integrity surface.* **Stance: defense.**

- **Why #1:** the only theme where verification made the gap *stronger* — AttackVLA [5] names the
  binning tokenizer as the more-susceptible component yet no paper defends it. Best feasibility/
  reproducibility in the set (frozen 4-bit checkpoint, inference-only guard, deterministic per-bin
  logs, script-regenerable figures). Publishable on a positive **or** negative result.
- **Binding requirement:** include an **adaptive in-bounds attacker** as a first-class arm; derive
  plausibility bounds from benign-rollout statistics + a simulated-arm kinematic model. Otherwise the
  defense overclaims (sophisticated in-distribution attacks stay inside plausible bins).
- **Next steps:** OpenVLA-7B LIBERO checkpoints (4-bit); LIBERO Spatial+Object primary; reproduce
  **one** patch objective (UPA) and validate vs published numbers *before* claiming any defense gain;
  defense = per-dim range clamp + temporal action-history consistency + kinematic plausibility on
  decoded tokens; metrics = recovered task-success, clean FPR, fraction of attacked outputs forced
  into extreme bins, recovered-success under the adaptive attacker; baselines = the 4 broken
  preprocessing defenses. Seeds pinned; untrusted artifacts under `artifacts/untrusted/`.
- **Main risk:** the **UADA/UPA/TMA reproduction tax** (largest hidden non-fine-tune cost).

### 🥈 #2 — T3 · Dose-Response of Small-Scale & Clean-Label Poisoning
*Benchmark.* Cleanest one-variable design; guaranteed deliverable; publishable on a negative.
**Foreground the clean-label arm** (the unclaimed kernel) and cite DropVLA/GoBA for positioning,
since the pure poison-rate curve is now partly pre-empted. Grid: rate {0.5,1,2,5,10}% × 2–3 seeds,
one fixed LoRA config; score attack-success as "reached a *specific* attacker-chosen target action,"
not mere task failure. ~10–14 LoRA fine-tunes → report honest wide error bars. **Strongest fallback #1**
if a supervisor wants lowest execution risk.

### 🥉 #3 — T8 · Trojaned-Checkpoint Supply-Chain Audit
*Defense.* The strongest *fully-open* defensive contribution — every backdoor paper assumes the
trojan exists; none audits a received checkpoint data-free / trigger-agnostic. Build a small clean+
trojaned OpenVLA zoo (reuses in-budget LoRA fine-tunes); detectors = weight-distribution outliers +
activation clustering on a benign probe set (+ optional trigger-reconstruction). Honest negative
("classifier-era detectors don't transfer to a 7B action head") is itself a finding. Risk: small zoo
weakens statistical detection-rate claims → scope to tested trigger types.

### #4 — T6 · Action-Consistency Runtime Monitor
*Defense.* Most reproducible (zero-retraining wrapper) and hits the most-cited gap G1 directly.
Adaptive adversary is the binding limit (now demonstrated for chunk models by SilentDrift) → adaptive
arm mandatory. Below T8 because its defended gain is structurally hostage to the adaptive attacker.

### #5 — T7 · Cross-Modal Instruction Injection (+ consistency defense)
*Defense (attack-motivated).* Most original *surviving attack-flavoured* contribution and cheapest to
run (text-only GCG, no patch-reproduction tax). Lock "successful redirection" to a specific reachable
attacker-chosen action (not denial), and derive the consistency metric from benign instruction-action
pairs to avoid circularity (may need a lightweight reference model — report it as a measurement).

### Honorable mentions (dropped)
- **T1** — attack contribution pre-empted 3× (GoBA/DropVLA/AttackVLA). Only its activation-detection
  probe retains marginal value, better absorbed as a sub-study of T8.
- **T4** — gap technically open but the LoRA-vs-full contrast is *confounded* on this node (no
  many-seed full 7B FT). Salvageable only as a LoRA-*rank* sweep, which shrinks novelty.
- **T2** — roster-collapse feasibility risk is independently fatal for a single-node MSc.

---

## 8. Recommended #1 + 2-Week Starter Plan (T5)

1. **Env capture (day 1–2):** pin Python/PyTorch/CUDA; record GB10 specs; create `requirements.txt`
   + seed registry. Stand up `results/` (write-once) and `artifacts/untrusted/`.
2. **Baseline reproduction (day 3–7):** run OpenVLA-7B (4-bit) on LIBERO Spatial+Object; reproduce the
   published **benign** failure baselines (~11–46%) → confirm the harness before any attack.
3. **One attack, validated (day 8–11):** reproduce **UPA** only; confirm against [2]'s reported numbers
   (e.g. UPA 100% vs 15.3% benign) *before* touching the defense. Log per-bin action distributions.
4. **Plausibility-bound derivation (day 12–14):** compute benign-rollout action statistics per dimension;
   draft the kinematic plausibility check + range clamp; define the **adaptive in-bounds attacker** spec.
5. **Checkpoint:** decide go/no-go on the defense arm based on whether attacks visibly push outputs into
   rarely-used extreme bins (the core empirical premise).

**Success criteria (state up front, CLAUDE.md §2.4):** *done = recovered LIBERO task-success, clean FPR,
and per-bin attack distribution computed for {benign, UPA, adaptive-in-bounds} on Spatial+Object, seeds
pinned, figures regenerated by a script in `scripts/` from logged `results/`.*

---

## 9. Suggested Next Decision

Confirm **T5 as #1** (or **T3** if you/your supervisor prefer the lowest-risk pure-measurement path),
then move to **Phase 3 (Design)**: write a full plan in `docs/plans/` (datasets, models, attack/defense,
metrics, baselines, ethics containment). Get supervisor sign-off and any ethics check before experiments.

---

## 10. Citation Verification Status (independent check, 2026-05-30)

**[verified]** — re-fetched from arXiv, title confirmed matching the agents' claims:

- **[1]** OpenVLA: An Open-Source Vision-Language-Action Model — `arXiv:2406.09246` (Kim, Pertsch, Karamcheti, … Finn)
- **[2]** Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics — `arXiv:2411.13587` (Wang, Han, Liang, … Tang)
- **[3]** BadVLA: Towards Backdoor Attacks on VLA Models via Objective-Decoupled Optimization — `arXiv:2505.16640` (Zhou, Tie, Zhang, … Sun)
- **[4]** Adversarial Attacks on Robotic Vision Language Action Models — `arXiv:2506.03350` (Jones, Robey, Zou, … Kolter)
- **[5]** AttackVLA: Benchmarking Adversarial and Backdoor Attacks on VLA — `arXiv:2511.12149` (Li, Zhao, Zheng, … Jiang)
- **[6]** DropVLA: An Action-Level Backdoor Attack on VLA Models — `arXiv:2510.10932` (Xu, Li, Zhao, … Jiang)
- **[7]** GoBA — Goal-oriented Backdoor Attack against VLA Models via Physical Objects — `arXiv:2510.09269` (Zhou, Xiao, Xu, … Zhang)
- **[8]** SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on VLA — `arXiv:2601.14323` (Xu, Shang, Wang, Ferrara)
- **[9]** Do What You Say: Steering VLA Models via Runtime Reasoning-Action Alignment Verification — `arXiv:2510.16281` (Wu, Li, Hermans, Ramos, Bajcsy, Pérez-D'Arpino). *Added 2026-05-30 from external review; prior work on plan↔action consistency — see §12.*
- **[4-name]** Confirmed [4] (`2506.03350`)'s method is **RoboGCG** (code repo `robogcg`); its abstract states a one-shot textual attack achieves *"full reachability of the action space"* of common VLAs.

**Refuted load-bearing claims (5/10)** — the verification pass killed these; do **not** treat as gaps:
the "*no targeted backdoor exists*" premise (refuted by [5][6][7]); over-broad "*no transfer benchmark*"
and "*SilentDrift defeats temporal monitors on OpenVLA*" framings (both narrowed); plus two others.
(Full per-claim verdicts are in the run output `tasks/wbxhj1g42.output`.)

**[unverified]** — surfaced by search agents but **not** independently checked this session; verify each
before citing: *Model-agnostic Adversarial Attack & Defense for VLA* `2510.13237`;
*TRAP* `2603.23117`; *RobustVLA* `2511.01331`; *VLA-Risk* (OpenReview `31EjDFwFEe`);
*Octo* `2405.12213`; *When Robots Obey the Patch* `2511.21192`; *Inject Once Survive Later* `2602.00500`;
*Altered Thoughts, Altered Actions* `2603.12717`; *On Robustness of VLA against Multi-Modal Perturbations*
`2510.00037`; *Towards Robust and Secure Embodied AI: A Survey* `2502.13175`.

---

## 11. Limitations of This Review

- **Recency cliff.** The field is moving monthly (multiple Oct 2025–Jan 2026 backdoor papers already
  reshaped the ranking). Re-run the search before finalising the proposal; the **clean-label / action-head /
  supply-chain-audit** gaps could close at any time.
- **Web-search recall, not exhaustive survey.** 16 sources fetched from 44 found; non-arXiv venues,
  very new pre-prints, and closed-source VLAs (RT-2, π0) are under-covered.
- **Feasibility assumes the stated GB10 profile.** If real compute/time differs, re-weight T3/T4
  (training-heavy) vs T5/T6/T7 (inference-heavy).
- **No real-robot validation** — all claims scoped to simulation; sim-to-real transfer is out of scope.
- **AI-generated scaffold** — prose, rankings, and rationale must be reviewed and rewritten by the
  author; secondary citations must be verified individually (§10).

---

## 12. Addendum (2026-05-30) — Consistency-Detector Prior Work & T7 Reframing

External review surfaced a **prior-work landmine** for the *consistency-detector* themes (T6, T7),
independently verified on arXiv:

- **[9] Wu et al., "Do What You Say"** (`2510.16281`) **[verified]** already proposes runtime
  **reasoning↔action alignment verification** for VLAs. Its mechanism: sample multiple candidate
  action sequences, predict outcomes via simulation, and use a pretrained VLM to *select* the
  sequence whose outcome best matches the VLA's own textual plan. **Setting = benign / OOD; no
  attacker; no false-positive-rate calibration; requires embodied chain-of-thought + a simulator +
  a VLM judge** (heavy, selection-based, on reasoning-annotated LIBERO-100).

**Impact on the ranking:**

- **The bare "pick actions consistent with the plan" idea is no longer novel** — an examiner can
  point to [9]. Any T6/T7 contribution **must not claim to invent consistency checking.**
- **T7 survives only if its novelty is narrowed to:** (i) a **security threat model** —
  attacker-aware detection of inconsistency against *externally injected* instructions (the attacker
  controls the stated instruction, so the detector must compare the action against a *trusted/sanitized*
  goal, not the attacker's text — a genuinely non-trivial, unaddressed design problem); (ii)
  **false-positive-rate calibration** of the detector; and (iii) a **lightweight runtime detector on
  vanilla (non-CoT) OpenVLA** — i.e. no embodied-CoT, no simulator-in-the-loop, unlike [9].
- **Reframed T7 (recommended form):** reuse **RoboGCG [4]** to reproduce the instruction-injection
  attack (abstract confirms full action-space reachability), then contribute the **security-framed,
  FP-calibrated goal-action-consistency detector**. Publishable either way: a usable operating point
  = positive; *inability to separate benign from attacked without breaking benign* = negative — and
  that benign/FP trade-off is the genuine make-or-break open question (the same trade-off that broke
  RoboGCG's borrowed smoothing defense).

**Residual risks / honesty flags:**
- **Novelty is narrow** (security framing + FP calibration only). Acceptable at MSc bar; do **not**
  over-claim "a new defense."
- The detector likely needs **some reference of trusted intent** (sanitized goal or a small reference
  policy) to avoid the attacker simply making the action consistent with their injected text — keep
  this in scope and report it as a measurement, not a solved system.
- **[verified 2026-05-31 — RoboGCG full text §5 + Table 3; PDF saved `docs/references/2506.03350-...pdf`,
  facts in `docs/references/README.md`]** The earlier claims about RoboGCG's evaluated defenses are
  **confirmed, and they *motivate* T7 rather than pre-empt it.** Defense eval = 120 random one-hot target
  actions. Table 3 ASR (%): No Defense 63.3 / 100 / 96.7 / 100 (Libero-10/Goal/Object/Spatial);
  **Multimodal PF = identical to No Defense (useless** — image embeddings dominate the loss, so multimodal
  perplexity never sees the suffix); **LLM-Only PF → 0.0 everywhere; Smoothing → 0.0 everywhere.**
  *But the authors call both viable-looking defenses infeasible:* the PF threshold "depends entirely on the
  maximum perplexity of instructions seen on a held-out set, which **cannot be known beforehand** in
  open-world robotics applications"; smoothing "results in a 0% success rate, **but also corrupts the
  instructions, resulting in a 0% success rate on non-attacked tasks**." Their conclusion: "This points to
  the need for notions of **VLA refusal** when attempts to subvert control are detected" — i.e. they treat
  the defense problem as **open**. → T7's FP-calibrated *behavioral* detector targets exactly the
  no-usable-operating-point gap RoboGCG leaves; the perplexity / text-only filter is a *baseline to beat*
  (beatable: multimodal PF fails, text PF has no a-priori threshold, both blind to fluent/visual/adaptive
  injection). *(Supersedes the prior "~38%" guess: 38.0 is the SIMPLER real-world-image transfer overall
  ASR, Table 2 — not a defense number.)*
- **Compute caveat:** a full RoboGCG sweep is ~90–300 GPU-h/model (33–110 GCG steps, ~185–604 s per
  target × ~1792 targets). **Subsample targets**, run a GCG micro-benchmark on the GB10 first, and
  put effort into the detector rather than a large attack re-run.

**Net effect on ranks:** [9] *lowers the novelty ceiling of T6 and T7* (both consistency-based) but
**does not touch T5 (token-plausibility clamp — different mechanism), T3 (poisoning), or T8 (checkpoint
audit).** Reframed-T7 becomes a sharper, more examiner-defensible theme than the original §7 entry —
strong enough to contend with T5 for the top slot *if* the supervisor is specifically interested in the
language→physical-action threat (see decision note in chat).

### 12a. T5 is model-specific — it does NOT hold on OpenVLA-OFT (verified 2026-05-30)

**[verified]** **OpenVLA-OFT** (`arXiv:2502.19645`, Kim et al.) replaces the original OpenVLA's
discrete 256-bin action tokenization with a **continuous action representation + L1-regression head +
parallel decoding + action chunking**. There are **no quantile bins** in the OFT recipe.

**Consequence for T5:** the distinctive T5 contribution — analyzing/clamping the *256-bin
quantile→Llama-token* surface, "which bins an attack pushes toward" — **only exists on the original
discrete-tokenization OpenVLA-7B**. On OpenVLA-OFT (and other continuous/diffusion-head VLAs such as
π0), that surface is gone; only a generic continuous-action range/kinematic/temporal plausibility guard
remains, which **collapses into T6**. So **T5 is locked to the discrete OpenVLA-7B checkpoint.**

- *Mitigating fact:* every major attack paper this defense answers — UADA/UPA/TMA [2], BadVLA [3],
  AttackVLA [5] — targets exactly that discrete OpenVLA-7B, so the threat is real *on that model*.
- *Examiner-relevance risk:* defending a tokenization that the field's newer recipes (OFT, π0) have
  already abandoned is a **novelty/relevance liability** an examiner can press. T5 must explicitly scope
  to discrete OpenVLA-7B and frame the continuous-head divergence as a stated limitation (or pivot to
  the *discrete-vs-continuous action-head robustness comparison* — novel but heavier, attacks on
  continuous heads are less established).
- **T3 (poisoning) and T7 (injection + consistency detector) are head-agnostic** — they work on both
  discrete OpenVLA and OpenVLA-OFT — so they do **not** carry this model-lock-in risk. Under a
  *guaranteed-deliverable-first* lean this further separates T3/T7 from T5.

### 12b. T7's load-bearing assumption — the "trusted intent" channel

**Locked framing (2026-05-30).**
*Main Question:* **"Can instruction-injection attacks on VLA models be detected by checking whether the
predicted robot actions remain consistent with a *trusted task goal*, while maintaining a low false-positive
rate on benign executions?"**
*Working title (safest):* **"Goal-action consistency detection for instruction-injected VLA policies under
calibrated false-positive constraints."**
This is **not** a claim of a new universal defense — it builds an **attacker-aware** goal-action consistency
detector and **measures how much it detects without breaking benign** (the FP-vs-detection trade-off).

**The trusted reference must be one the attacker does not directly control** — comparing the action to the
*tampered* instruction is meaningless. Candidate references, ordered by deployment realism + whether they
survive the "necessity" critique below:

| Trusted reference | Realism | Survives "why not just use it"? | Role in the study |
|---|---|---|---|
| Benchmark clean task instruction | proxy only | ✗ — full drop-in for the VLA input | **experimental measurement (ceiling)** |
| Sanitized planner output | medium | partial — only if sanitization is imperfect | intermediate rung |
| Operator pre-approved goal (coarse) | high | ✓ — coarser than the operational instruction | **strong deployment case** |
| Task ID → expected goal | high | ✓ — a reference, not a drop-in | **strong deployment case** |

→ These rows **are** the trusted-reference ladder (below): use the benchmark clean instruction to *measure*
the detector's ceiling, and the coarse operator-goal / task-ID-restored-goal as the *realistic* rungs.

The T7 detector flags injection by checking the action against the *intended* goal. For this to detect
anything, the intent reference must be one the **attacker does not control** — otherwise the redirected
action is consistent with the attacker's injected instruction and the detector is blind. Using the
**clean LIBERO instruction as the trusted reference while feeding OpenVLA `clean + RoboGCG-suffix`** is a
valid *experimental operationalization*, but it is **not itself a threat model** — it is a proxy.

**Is it a realistic threat model? Only under a specific framing:**
- ✅ **Realistic** — indirect prompt injection / hierarchical *planner→executor*, where a **trusted,
  coarser operator goal** is separable from the **untrusted operational instruction** (operator sets a
  standing task; attacker injects appended/retrieved/perceived content). This is the canonical
  LLM-agent injection threat, transposed to embodiment.
- ❌ **Not realistic** — a human directly commanding the VLA *is* the attacker, or the whole instruction
  channel is compromised with **no separable trusted part**. No trusted reference exists → do not claim
  this case.

**The deeper trap — "detection power" vs "detection necessity":** even granting a trusted reference, an
examiner asks *"if you hold the clean instruction, why not just feed it to the VLA (or sanitize the
suffix) and skip the detector?"* The detector is only **necessary** when the trusted reference is
**coarser** than the operational instruction (hierarchical: trusted "make dinner" vs operational "grab
the knife") **or** is a **reference model/distribution** rather than a literal instruction. In the *flat*
LIBERO setup the clean instruction is a drop-in for the VLA input, so the trivial defense (use the clean
instruction) dominates — the experiment would show detection *power* but not detection *necessity*.

**The open-vs-realistic bind:** the setting where trusted intent most naturally exists (hierarchical
planner→executor) is also where injection has *already* been studied at the planner level
(RoboPAIR/BadRobot, per §6 delineation). The *open* setting (end-to-end token-AR OpenVLA) is exactly
where a separate trusted intent is hardest to justify. Sweet spot: **end-to-end OpenVLA + trusted =
operator standing task + injection = appended/indirect content** — open *and* realistic, but the
coarseness/sanitization tension above must be engineered around.

**Recommended resolution — make the assumption the variable.** Do not assert "trusted intent exists."
**Study it:** calibrate detection power + FPR across a ladder of trusted-reference strengths (full clean
instruction → coarse goal → goal embedding → small reference policy → none), and report where the
assumption becomes unrealistic. The contribution becomes *"how much trusted intent is needed to detect
VLA instruction-injection, and where it breaks"* — which turns the central objection into the thesis
backbone and is publishable either way.

**Prior-work flag [unverified]:** position against the LLM-agent **task-drift detection** and
**instruction-hierarchy / privileged-instruction** literature (the same trusted-intent-vs-behavior idea)
— verify before claiming novelty; a likely landmine *beyond* Wu et al. [9].

### 12c. New candidate **T9** — Detecting/neutralizing a *persistence-hardened* supply-chain backdoor around clean LoRA adaptation

*Question:* "Can a user **detect or neutralize a persistent supply-chain backdoor before/after clean LoRA
adaptation?"* Effectively **T8 upgraded** with the *fine-tuning-persistence* threat and the real
practitioner workflow (download → clean LoRA → deploy). **Stance: defense (attack-aware).**

**Verified anchors (2026-05-30):**
- **[10] INFUSE — "Inject Once Survive Later"** (`arXiv:2602.00500`, Zhou, Wei, Zhen, Zhao, Xia, Shao,
  Su, Yang) **[verified]**: backdoors engineered to **persist through downstream/clean fine-tuning**.
  Killer anchor — **after user-side fine-tuning, ASR = 91.0% (sim) / 79.8% (real)** vs **BadVLA 38.8% /
  36.6%** (BadVLA largely washes out; INFUSE does not). **Proposes NO defense — left as future work.**
- Novelty-landmine scan **[verified clean]**: no dedicated VLA backdoor **detection/removal** paper
  exists; the attack space (BadVLA/DropVLA/INFUSE) leaves defense open. Survey **[11] VLA Safety**
  (`arXiv:2604.23775`, *unverified*) frames defense as an emerging gap — useful for positioning.

**Why it's strong:** verified threat anchor + **verified-open** defense gap; **head-agnostic** (persistence
is a weights phenomenon → works on discrete OpenVLA *and* OpenVLA-OFT, so **no T5-style model lock-in**);
**no T7-style trusted-intent fragility**; vivid practitioner narrative ("you fine-tuned on clean data and
were still owned"); fresh novelty (INFUSE is Feb 2026, defense explicitly open) → cleaner novelty position
than T7's contested detector; **publishable either way** (removal works = positive; *no cheap method
neutralizes it* = strong security finding that the persistence threat is unmitigated by standard practice).

**The single load-bearing gate (check FIRST):** can you obtain a *persistent* backdoor **cheaply**?
- ✅ If INFUSE **released code/checkpoints** → reuse → low feasibility risk. *(unknown — verify the repo.)*
- ⚠️ If **not released** → you must **reproduce** the persistence mechanism (attack-reproduction tax,
  non-trivial). *Fallback:* study a **persistence gradient** using BadVLA (retains ~38.8% post-FT) as the
  weak-persistence baseline even without full INFUSE.

**Gate result — checked 2026-05-30: artifacts NOT public yet.** INFUSE project page
(`jianyi2004.github.io/infuse-vla-backdoor/`) lists **"Code (Coming Soon)"** (placeholder repo
`github.com/Jianyi2004/infuse-vla-backdoor`, empty), **no checkpoints**, base model unnamed; eval =
**LIBERO + SimplerEnv**. So reuse is currently impossible → T9 carries a **bounded backdoor-construction
tax**. Three mitigations keep it viable:
- **Path C (today, free):** BadVLA code **is public** (`github.com/Zxy-MLlab/BadVLA`); it retains ~38.8%
  post-FT → a **weak-persistence baseline you can build now** and study detect/neutralize on immediately.
- **Path D (simplified INFUSE):** the **mechanism is published & verified** — INFUSE = *"INjection into
  Fine-tUne-inSensitive modulEs"*: inject the trigger into the **fine-tune-insensitive modules** (vision
  backbone, vision projector, LLM backbone — which receive **100–1000× smaller** updates than the action
  head / proprio projector under user fine-tuning) and freeze the rest. You can build a **controlled
  simplified persistent backdoor** by injecting into those low-update modules — far less work than
  reproducing SOTA INFUSE, and sufficient to study the defense.
- **Path A (monitor):** watch the repo for release during the project — but **do not plan around it.**

**Mechanism → a sharper, novel DEFENSE hypothesis (independent of artifact availability):** because the
backdoor *must* live in the fine-tune-insensitive modules to survive, the defender knows **exactly where
to look** — audit/neutralize the **vision backbone/projector + LLM backbone**, not the action head. This
turns INFUSE's own attack principle into a **principled, targeted detection/removal hypothesis** — likely
T9's best feature, and it needs no released checkpoint.

**Revised placement (post-gate):** T9 ≈ **T3** under deliverable-first. T9 has the stronger
narrative/novelty *and* a principled mechanism-targeted defense, but a **bounded backdoor-construction tax**
that T3 lacks; T3 has the lowest build risk (no artifact dependency) but heavier compute and a partly
pre-empted rate-curve. Choose T9 if comfortable building a (simplified) persistent backdoor; T3 if you want
minimal build risk.

### 12d. Narrowed **T9** (BadVLA baseline + targeted-module audit) vs **T7** — head-to-head

**Recommended narrowed-T9 scope:** a *coupled* attack→defense story driven by one structural fact.
- **Attack side (kept minimal):** use **BadVLA** (public code) as the **weak-persistence baseline** (~38.8%
  post-FT), and build a **simplified-INFUSE** *positive* case by injecting the trigger into the **known**
  fine-tune-insensitive modules (vision backbone / projector / LLM backbone — INFUSE already identified
  these, so "which modules" is **given, not discovered**). Measure ASR **before vs after a clean user LoRA**,
  and confirm clean-task SR is preserved.
- **Defense side (the contribution):** test the hypothesis *"residual persistence concentrates in the
  fine-tune-insensitive modules → audit/neutralize those specifically."* Metrics: detection rate/FPR, ASR
  reduction, clean-SR utility cost.
- *Note:* BadVLA is **not** persistence-by-construction, so it's a weaker test of the module hypothesis —
  keep it as the **contrast/baseline**; the simplified-INFUSE case is what cleanly tests the hypothesis.

**Risk-axis correction:** T9's attack reproduction (BadVLA poison + **two** ~12 h LoRA fine-tunes per
condition) is genuinely **heavier** than T7's (RoboGCG = **training-free**, inference-time GCG). So T7 wins
the **compute/execution** axis. **But** the two themes' *dominant* risks live on different axes: **T7's risk
is conceptual** (trusted-intent realism/necessity — can't be "scoped down", only reframed), **T9's risk is
executional** (compute/build — bounded, scope-able via fewer seeds/conditions). For a graded MSc, *irreducible
conceptual risk is worse than bounded compute risk.*

**Scored (1–5):**
| Dimension | Narrowed-T9 | T7 (ladder) |
|---|---|---|
| Novelty (clean vs contested) | **5** (first persistent-backdoor defense; verified-open) | 3 (Wu et al. + task-drift adjacent) |
| Conceptual robustness / viva-defensibility | **5** (unimpeachable premise) | 2 (trusted-intent fragility) |
| Feasibility / compute | 3 (training-heavy, ~70–190 GPU-h) | **5** (training-free attack, light) |
| Reproducibility | 4 | 4 |
| Publishable on a negative | **5** (persistence unmitigated = strong) | 4 |
| Supervisor salience | 4 (supply-chain trust) | 4 (injection→harm) |
| **Lean (deliverable-first)** | **higher overall** | lower |

**Verdict: narrowed-T9 > T7 overall.** T9 wins on the **grade-determining** axes (novelty + viva-defensibility)
and only loses on the **most manageable** axis (compute). **T7 wins only if** (a) the supervisor specifically
wants the *language-injection→dangerous-action* story, or (b) compute/time is so tight that any training-heavy
plan is infeasible and the training-free attack is required.

### 12e. Competitive landscape — who works each space (affiliations verified 2026-05-30)

| Cluster | Paper | Authors / Institutions (verified) | Standing |
|---|---|---|---|
| **T7** | **RoboGCG** [4] (`2506.03350`) | Jones, Robey, Zou, Ravichandran, Pappas, Hassani, Fredrikson, **Kolter** — **Gray Swan AI + CMU + UPenn** | **Elite** — the GCG inventors (Zou/Fredrikson/Kolter), RoboPAIR (Robey), UPenn GRASP |
| **T7** | **Do What You Say** [9] (`2510.16281`) | Wu, Bajcsy (**CMU**) + Ramos, Pérez-D'Arpino, Hermans (**NVIDIA**); **code released → `NVlabs/actalign`** | **Elite** — NVIDIA Research + CMU, *reference impl public* |
| **T9** | **AttackVLA/DropVLA** [5][6] | **Fudan** (corr. **Xingjun Ma, Yu-Gang Jiang**) + CityU HK + SMU | **Strong** — Yu-Gang Jiang highly-cited; trustworthy-ML/CV |
| **T9** | **INFUSE** [10] (`2602.00500`) | Zhou, Shao, Yang, Wei (**HIT Shenzhen**) + Zhen (**Meituan Robotics**) + Zhao (**SJTU**) + Xia (**NUS**) + Su (**CSU**); **code "coming soon", unreleased** | **Solid/rising** — academia + industry robotics; *defense left open* |

**Raw prestige verdict:** **T7's foundation is the stronger of the two.** The RoboGCG author list (Kolter /
Zou / Fredrikson + Gray Swan AI + UPenn) is arguably the single strongest in VLA security, and the nearest
defense (Wu et al.) is NVIDIA+CMU with **released code**. T9's cluster (Fudan/Yu-Gang Jiang, HIT, NUS, SJTU,
Meituan) is genuinely strong but a notch below in adversarial-ML/safety-community visibility.

**Strategic read (the part that matters for an MSc):** competitor strength is an *asset when it's your
foundation* (you cite them) and a *liability when it's your competition* (they scoop you / you look
incremental).
- **T7:** strong groups occupy **both** the attack (RoboGCG) **and your intended defense niche** —
  NVIDIA's **actalign** is a *released consistency-verification reference implementation*. → highest
  credibility foundation, but **highest competition/scoop risk on your exact contribution**, and actalign
  **raises the novelty bar** for any consistency detector further (on top of the Wu-et-al. landmine in §12).
- **T9:** strong groups did the **attacks** and **explicitly left the defense open** (INFUSE: future work),
  code unreleased → the **defense lane is clear**. Lower head-to-head competition on your niche (caveat:
  the same capable groups *could* publish a defense follow-up — monitor).

**Net:** on "stronger papers," **T7**; on "stronger *position* for a single-author MSc," **T9** — the
landscape evidence **reinforces narrowed-T9 > T7** (open lane beats crowded lane, even crowded-by-elites).

**Suggested clean scope (avoid the 4-cell sprawl of "detect/neutralize × before/after"):** a single pipeline —
(0) obtain/reproduce a persistent backdoor; **replicate** the post-LoRA survival gap on your setup;
(1) **detect-before:** data-free/trigger-agnostic weight/activation audit of the received checkpoint (the T8 core);
(2) **neutralize:** does the user's *own clean LoRA* cut ASR? if not, apply transfer of classifier-era removal
(fine-pruning, adversarial neuron pruning, attention distillation) *around* adaptation;
(3) **verify-after:** residual-ASR audit post-adaptation.
*Metrics:* pre/post-adaptation ASR, clean-task utility cost of removal, detection rate/FPR, and the
removal-cost-vs-residual-risk trade-off. **Sim only** — scope away from INFUSE's real-robot numbers.

**Residual risks:** (i) artifact availability (the gate above); (ii) removal-method **transfer** to a 7B
VLA action head is unproven — *but that uncertainty is the research question*; (iii) scope creep — hold the
pipeline above.

**Placement:** a **defense, head-agnostic, deliverable-friendly, attack-aware** theme with a verified anchor
and a verified-open gap — under the *guaranteed-deliverable-first* lean it is a **top contender alongside T3**
(arguably ahead, given the stronger threat narrative and fresher novelty), *conditional on the artifact gate*.
Supersedes the original §7 **T8**.

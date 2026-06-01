# T7 — Goal-Action Consistency Detection for Instruction-Injected VLA Policies

> **⮕ REFRAMED 2026-06-01 — read the playbook first.** The research **headline** is now *The Embodiment Evasion
> Tax: Measuring Adaptive Evasion Costs of Runtime Defences for Vision-Language-Action Models* — a
> **measurement** of per-layer (**L0** input / **L1** internal-probe / **L2** behavioural) adaptive-evasion
> cost, **not** a goal-action *defence* claim. The goal-action consistency detector described below is now the
> **L2 behavioural layer + its privileged oracle (metric A)**; everything here remains valid as the L2 /
> threat-model foundation. Operational source of truth = [`t7-execution-playbook.md`](./t7-execution-playbook.md)
> — **§3a (reframe), §2 (tiered roadmap), §4b (what to implement), H6 (hypothesis)**. (Internal codename stays
> **T7**; *Embodiment Evasion Tax* is the dissertation title.)

> **Status:** Theme **agreed 2026-05-30**. Phase: Scope → **Design** (CLAUDE.md §3).
> **This document** consolidates the T7-specific reasoning from the theme-scoping session into one
> self-contained place, to drive the next decisions about T7's direction. Broader landscape, gap map,
> and the full ranking live in `docs/lit-review/vla-integrity-theme-scoping.md` (esp. §12, §12b, §12d, §12e).
>
> ⚠️ AI-assisted scaffold for the author to review/rewrite (CLAUDE.md §5). Citations marked **[verified]**
> were re-fetched from arXiv on 2026-05-30; **[unverified]** ones must be checked before citing.

---

## 1. The decision, in one line

> **Goal-action consistency detection for instruction-injected VLA policies under calibrated
> false-positive constraints.**

**Main Question.** *"Can instruction-injection attacks on VLA models be detected by checking whether the
predicted robot actions remain consistent with a **trusted task goal**, while maintaining a low
false-positive rate on benign executions?"*

This is a **measurement** study of an **attacker-aware, FP-calibrated** detector — **not** a claim of a new
universal defense. We **reproduce** a published attack and **build + calibrate** a detector, then measure the
detection-vs-false-positive trade-off.

---

## 2. Why T7 (decision rationale)

Chosen over the other finalists under a stated **deliverable-first** risk appetite and the project's
defensive posture (CLAUDE.md §1, §6). Summary of the trade (full scoring in lit-review §12d–§12e):

- **vs T5 (action-tokenization defense):** T5's mechanism is tied to OpenVLA's discrete 256-bin head and
  **breaks on OpenVLA-OFT** (continuous L1 head) — model lock-in. T7 is **head-agnostic**.
- **vs T3 (clean-label poisoning):** T3 is the lowest-build-risk measurement study but less salient; T7
  carries the vivid **language→dangerous-action** story the supervisor angle favours.
- **vs T9 (persistent-backdoor defense):** T9 scored higher on raw novelty/viva-defensibility and has an
  open lane, but is **training-heavy** (inject backdoor + simulate user fine-tune). T7's attack side is
  **training-free** (inference-time GCG) → lighter compute. **The user chose T7.**

**Honest caveat carried forward:** T7's novelty is *narrow and in a contested lane* (see §6). The framing
must stay disciplined as a measurement study to be defensible.

---

## 3. Threat model (the load-bearing part)

The detector flags injection by checking the predicted action against the **intended goal**. For this to
detect anything, the reference goal must be **one the attacker does not directly control** — comparing the
action to the *tampered* instruction is meaningless (the redirected action is consistent with the attacker's
text, so the detector is blind).

**Setting we claim:** *indirect* prompt injection on an **end-to-end VLA**, where a **trusted operator goal**
is separable from the **untrusted operational instruction** (operator sets a standing task; attacker injects
an appended/retrieved suffix). We do **not** claim the "human-directly-commanding-the-VLA-is-the-attacker" or
"whole-channel-compromised" cases (no trusted reference exists there).

### Trusted-reference ladder (also the secondary study axis)

| Trusted reference | Realism | Survives "why not just use it"? | Role |
|---|---|---|---|
| Benchmark clean task instruction | proxy only | ✗ (full drop-in for VLA input) | **measurement ceiling** |
| Sanitized planner output | medium | partial (only if sanitization imperfect) | intermediate rung |
| **Operator pre-approved goal (coarse)** | high | ✓ (coarser than operational instruction) | **deployment claim** |
| **Task ID → expected goal** | high | ✓ (a reference, not a drop-in) | **deployment claim** |

→ Lead the *deployment* argument with the operator-goal / task-ID rungs; use the benchmark clean instruction
only to establish the detector's **ceiling**. Reporting detection + FPR **across these rungs** turns the
central objection ("does a trusted reference realistically exist?") into the thesis backbone — publishable
either way.

---

## 4. What we reproduce vs. what we build

**Reproduce (attack):** **RoboGCG** [4] — a **white-box GCG-optimized adversarial text string** (optimized
tokens, *not* a natural-language instruction), applied **once at the start of the rollout**, to elicit
**attacker-chosen low-level actions**. Its abstract claims *"full reachability of the action space" /
"complete control authority"* and explicitly notes attacks *"do not have to be semantically linked to notions
of harm"* — i.e. RoboGCG proves **low-level action reachability**, NOT semantic dangerous-task success
(e.g. "pick the wrong object"). Treat it as a **fixed adversary**; do not aim to out-attack it.

> **Framing note (important).** The "instruction injection" label is aspirational. RoboGCG is an
> *adversarial-string* (white-box GCG) attack, and base OpenVLA is comparatively robust to *natural* suffixes.
> **T7's job is exactly to bridge** "attacker reached an arbitrary low-level action" → "that behavior violates
> the trusted *semantic* goal." A natural-language-injection variant is at most a secondary/stretch arm
> (likely weaker on base OpenVLA) — confirm it works before relying on it; do not write RoboGCG up as a
> natural instruction injection or as semantic-task subversion.

**Build (the contribution):**
1. A **trajectory/window consistency metric** `s(observation_t, action_{t:t+k}, trusted_goal, [state_t])` —
   a *single* action cannot determine goal-consistency; the detector needs the observation, an action
   **window**, and optionally privileged state. *This metric is the real engine, see §5.*
2. A **calibrated detector**: fire when inconsistency > τ, where τ is chosen on a **held-out calibration
   split** to hit a target benign **FPR** (e.g. 1%, 5%); firing → a safe fallback (hold/abort).
3. **Evaluation** on **held-out benign tasks / scenes / seeds** (calibration/test split — never set τ and
   report FPR on the same rollouts), across the trusted-reference ladder (§3) and across operating points.

---

## 5. The hidden instrument: the consistency metric (where the risk actually lives)

The title's "consistency detection" quietly assumes a function scoring whether the robot's behavior agrees
with a text goal. **A single 7-DoF action is too ambiguous** — score an **action window over its observation
context**: `s(observation_t, action_{t:t+k}, trusted_goal, [state_t])`. **This metric — not the threshold —
is the make-or-break.** Options (a key open decision, §7):

- **(A) Privileged sim-state metric** — derive action semantics from LIBERO ground truth (which object the
  gripper approaches, gripper open/close, target region) and compare to the goal. *Cheap, clean to measure,
  but uses privileged info → a measurement **upper bound**, not deployable.* Must be labelled as such.
- **(B) Learned action-semantics map** — train a lightweight model mapping (observation, action) → a goal
  embedding / textual descriptor; compare to the trusted goal embedding. *Deployable; more work; its
  robustness becomes the new risk.*
- **(C) Reference-policy divergence** — a small benign policy predicts the expected action distribution for
  the trusted goal; flag divergence. *Deployable; needs a second policy.*
- **(D) VLM/LLM judge** — prompt a pretrained model "does this action serve this goal?" *Heaviest; closest to
  Wu et al. `actalign` — avoid leaning on it to stay differentiated and lightweight.*

**Plan:** start with (A) to establish the ceiling and prove the signal exists, then attempt (B) or (C) as the
deployable version. Budget the project around **designing + validating the consistency metric**, with FP
calibration as the headline result.

---

## 6. Prior-work constraints & novelty positioning (must respect)

- **Wu et al., "Do What You Say"** [9] (`2510.16281`, CMU + NVIDIA; **code released → `NVlabs/actalign`**)
  already does **reasoning↔action consistency verification** — but **benign/OOD only, no attacker, no FP
  calibration**, and needs embodied CoT + simulator + VLM judge. **Do NOT claim consistency-checking as new.**
  Our differentiation = **(i) security threat model (attacker-aware, external injected instruction),
  (ii) false-positive calibration, (iii) lightweight, non-CoT detector on vanilla OpenVLA.**
- **Competitive landscape (lit-review §12e):** the lane is crowded by elite groups — the attack (RoboGCG) is
  the GCG inventors (Kolter/Zou/Fredrikson, Gray Swan + CMU + UPenn) and the nearest defense (`actalign`) is
  **NVIDIA + CMU with released code**. → credibility-rich foundation but **high scoop risk**; differentiation
  (security + FP-calibration) must be sharp and explicit, and we should move at reasonable pace.
- **Adjacent prior work [verified 2026-05-31 via arXiv abstracts; all 4 Codex-flagged IDs resolved
  correctly].** None scoop T7's *embodied, action-level* contribution, but they **narrow the novelty claim**:
  FP-aware / trusted-intent injection detection is **not new in the text-LLM world** — only its VLA/action
  instantiation is. Must cite + differentiate explicitly:
  - **Task Drift — "Get my drift? Catching LLM Task Drift with Activation Deltas"** (`2406.00799`, Abdelnabi
    et al., MSR, 2024): detects LLM prompt-injection/task-drift via **activation deltas** (linear probe),
    attacker-aware, **text LLM only**. *T7 differs:* embodied VLA, **behavioural goal↔action-window** signal
    (not internal activations), per-rollout physical-abort FP cost.
  - **Instruction Hierarchy** (`2404.13208`, Wallace et al., **OpenAI**, 2024): a **training method** to
    prioritise privileged over injected instructions, **text LLM only**. *T7 differs:* a **runtime detector**
    (no retraining), embodied; this is the *origin of the trusted-vs-untrusted instruction framing* we adopt.
  - **AlignSentinel** (`2602.13597`, Jia, Wang, Wang, Xiang, **Gong**, 2026): **closest prior work** — a
    detector classifying misaligned / aligned / non-instruction inputs via **attention-map** features that
    **explicitly reduces false positives** on benign instruction-bearing inputs — but **text LLM**,
    instruction-classification level. *T7 differs:* VLA **action-level** goal-consistency, RoboGCG
    adversarial-suffix threat, per-rollout abort calibration. ⇒ **Do NOT claim FP-aware injection detection as
    new in general — only its embodied/action instantiation.** (Gong's group is active here → scoop-risk.)
  - **SABER** (`2603.24935`, Wu et al., UMD/Manocha, 2026): a **natural-language** instruction-edit **attack**
    on **VLAs, evaluated on LIBERO across 6 VLA models** (black-box, GRPO ReAct, bounded edit budget). *Attack,
    not detector → does not scoop T7.* **Opportunity:** a ready-made **fluent / low-perplexity** VLA injection
    — the concrete case where the **perplexity baseline fails** but goal-action consistency may still fire →
    **candidate secondary attack arm** (confirm it includes OpenVLA; black-box + agentic → reproduction cost;
    treat as optional, like RoboGCG-primary).
  - **VLA Safety survey** (`2604.23775`) — still *unverified*; fetch when writing the framing section.

---

## 7. Open design decisions to resolve next (use this doc to drive them)

1. **Consistency metric** — which of (A)–(D) §5 is the primary, and which is the deployable stretch?
2. **Attack-target definition** — specify the "attacker-chosen target" at the **low-level action** level so
   success = *reached the target action* (not mere task failure / denial), then decide how (if at all) to
   **bridge to a semantic** danger interpretation. Confirm RoboGCG yields *targeted* low-level redirect on our
   checkpoint, and decide whether a natural-instruction-injection arm is in scope.
3. **Trusted-reference rungs** — which of the four (§3) to implement; how to construct the coarse operator
   goal and the task-ID→goal mapping in LIBERO.
4. **Eval suites & scale** — LIBERO Spatial/Object/Goal; #tasks × #targets × #seeds; FP operating points.
5. **Baselines** — benign-instruction success; RoboGCG published numbers; and a naive baseline detector
   (e.g. perplexity/text-only filter) to beat. Position vs `actalign` conceptually.
6. **Metrics** — ROC/AUC of the consistency score; **TPR @ fixed benign FPR** (on a **held-out** split);
   **benign task-success degradation**, **attack detection rate**, **unsafe-action-blocked rate**, and
   **abort rate / latency**. *(NOT "recovered task-success": a hold/abort fallback prevents the unsafe action
   but does not complete the task — only measure recovery if you also build replan / clean-instruction
   re-execution.)* Report degradation **across the trusted-reference ladder**.
7. **Compute budget** — GCG sweep cost (H100 numbers may not hold on GB10 → micro-benchmark first, subsample
   targets); any training for metric (B)/(C).

---

## 8. Feasibility (single GB10 + simulation)

- **Model:** OpenVLA-7B (4-bit, fits ~128 GB), LIBERO-finetuned checkpoints. The **detector concept is
  head-agnostic**, but the **initial RoboGCG attack reproduction is scoped to the discrete-action-token
  OpenVLA-7B** — GCG over discrete action tokens does not transfer unchanged to OpenVLA-OFT's continuous L1
  head. (Still better than T5, whose *defense* mechanism itself was tokenization-specific.)
- **Attack compute:** RoboGCG is **training-free** (inference-time white-box GCG) but **not free** — the
  published ~185–604 s/target is **H100-based and may be slower on GB10**. Treat it as **bounded and
  subsampleable**: micro-benchmark GCG on the GB10 first, subsample targets, and concentrate effort on the
  detector.
- **Detector:** metric (A) is analysis-only; (B)/(C) need a small model (in-budget LoRA/MLP scale).
- **Sim only** — scope all claims to simulation; no real-robot transfer claims. Pin all seeds; write-once
  `results/`; quarantine any attack artefacts under `artifacts/untrusted/` (CLAUDE.md §4, §6).

---

## 9. Risks & how managed

| Risk | Management |
|---|---|
| Attack is **denial-only**, not targeted → consistency reduces to a failure detector | Week-1 gate confirms targeted redirect; else reframe metric to "task-abandonment detection" |
| **Trusted-reference realism** challenged at viva | Study it as a *ladder*; lead deployment claim with operator-goal / task-ID rungs |
| **"Why not just sanitize / use the clean instruction"** | Use coarse references (necessity-surviving rungs); state benchmark instruction is ceiling-only |
| Consistency **metric** weak / not deployable | Start with privileged-state upper bound (A), then build deployable (B)/(C); report honestly |
| **Novelty** contested (Wu et al. / actalign / task-drift) | Differentiate explicitly: attacker-aware + FP-calibrated + non-CoT; verify task-drift literature |
| **Scoop risk** (elite groups in lane) | Sharp differentiation; reasonable pace; publishable-either-way framing |
| High benign **FPR** on jerky/ambiguous motion | Calibrate τ on a **calibration** benign split; report FPR on **held-out** benign tasks/scenes/seeds; report operating points, not a single threshold |

---

## 10. Week-1 viability gate (cheap go/no-go before deep work)

1. Stand up OpenVLA-7B (4-bit) on LIBERO; reproduce the **benign** baseline success.
2. Reproduce **RoboGCG** on a few tasks; **confirm it achieves *targeted* redirection** (reached an
   attacker-chosen action region), not just task failure. *(If only denial → reframe per §9.)*
3. Sanity-check the **signal exists**: with the privileged-state metric (A), is the inconsistency score
   visibly separated between benign and attacked rollouts? If yes → proceed to build/calibrate; if no →
   rethink the metric before investing.

**Gate = done when:** benign baseline reproduced + RoboGCG targeted redirect confirmed + a visible
benign-vs-attacked separation in the consistency score on a handful of tasks (seeds pinned).

---

## 11. Key references (verified 2026-05-30 unless noted)

1. **RoboGCG — "Adversarial Attacks on Robotic Vision Language Action Models"** — `arXiv:2506.03350`
   (Jones, Robey, Zou, Ravichandran, Pappas, Hassani, Fredrikson, Kolter; Gray Swan AI + CMU + UPenn).
   *The attack we reproduce.* **Full text saved locally: `docs/references/2506.03350-robogcg-adversarial-attacks-vla.pdf`
   (facts + SHA-256 in `docs/references/README.md`).** **§5 defenses verified 2026-05-31:** borrowed
   text-only PF and smoothing zero the ASR but are author-declared infeasible (PF threshold unknowable a
   priori; smoothing → 0% benign success); multimodal PF is useless; authors call for "VLA refusal" → the
   defense problem is **open**, motivating T7's FP-calibrated behavioral detector (see §5/§6 baselines).
2. **"Do What You Say: …Runtime Reasoning-Action Alignment Verification"** — `arXiv:2510.16281`
   (Wu, Li, Hermans, Ramos, Bajcsy, Pérez-D'Arpino; CMU + NVIDIA; code `NVlabs/actalign`).
   *Consistency-verification prior art = the novelty constraint.*
3. **OpenVLA: An Open-Source Vision-Language-Action Model** — `arXiv:2406.09246` (Kim et al.; Stanford).
   *Base model / victim.*
4. *(framing, [unverified])* **VLA Safety: Threats, Challenges, Evaluations, and Mechanisms** — `arXiv:2604.23775`.
5. *(to find, [unverified])* LLM-agent **task-drift detection / instruction-hierarchy** literature — verify
   before claiming detector novelty.

---

## 12. Next action

Use this document to settle the **§7 open decisions** (starting with the consistency metric and the
attack-target definition), then write the full Phase-3 implementation plan in `docs/plans/` and run the
**§10 week-1 gate** before committing to the full study.

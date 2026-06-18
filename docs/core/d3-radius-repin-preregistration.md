# D-3 — SchemaA Radius Re-pin Pre-registration (locked before any attacked output)

> **STATUS: PROPOSED — drafted 2026-06-18; author/supervisor sign-off PENDING.** Once signed off this becomes
> **LOCKED** and dated. It governs the **one** permitted update to the frozen `SchemaA` geometric radii
> (`engagement_radius`, `grasp_radius`) and resolves the `[VERIFY vs LIBERO geometry]` flags in
> [`metric-a-annotation-schema.md`](./metric-a-annotation-schema.md) §5. It exists to satisfy that document's
> **§0 circularity guard** and **playbook invariant #2**: the metric's constants must be fixed from
> **benign geometry only**, **before the first attacked rollout is ever inspected**, so the detector cannot be
> reverse-engineered from attacks.
>
> **Why now (timing).** The first attacked output appears at CSB bring-up **step 6 — the *tiny GCG run***. The
> preceding steps are attack-free: **step 5.5** (the bf16 + flash_attn2 gradient smoke) and the step-6
> **GCG-harness coding** produce **no** attacked output. This rule must be LOCKED **before the step-6 tiny GCG
> run**. It is drafted now (cheap, mac-side, easy to forget) so the gate cannot be skipped once attacked data
> exists.

---

## 1. Scope — exactly what this governs

- **In scope:** the two geometric radii of the frozen `SchemaA` — `engagement_radius` (`r`, default **0.05 m**)
  and `grasp_radius` (`R_g`, default **0.10 m**).
- **Out of scope (already governed elsewhere, do not touch here):**
  - `k` (causal window) — already has a pre-registered **benign** sweep `k ∈ {3,5,8}`, final at M1 from
    **rollout lengths** (schema §5); rollout lengths are benign, so no attack-leak.
  - `combination` (`max`) and the primitive set — frozen; the pre-registered ablation (`noisy_or`,
    `weighted_mean`) is a **sensitivity report**, not a re-pin (schema §4).
  - τ (detection threshold) — calibrated on **benign** scores at run time (invariant #3); a *separate, already-
    allowed* benign step, never part of this rule.

This rule permits **at most one** radius update, applied **once**, after which the radii are **re-frozen** and
any further change is a new, separately recorded deviation.

---

## 2. Admissible inputs (benign-only) and forbidden inputs

**The re-pin may read ONLY:**
- The **M2 benign calibration split** — the benign rollouts already used for τ calibration: `N_benign` rollouts
  across the pre-registered LIBERO eval tasks/seeds (`N_benign ≥` the branch's pre-registered calibration N;
  Branch N target ≈ 300, smaller for N−/F — see playbook §2). These are **successful and unsuccessful** benign
  rollouts as logged; per-quantity success filtering is specified in §3.
- The geometric quantities derived from those rollouts' **privileged state only** (`ee_pos`, `object_poses`,
  `target_region`, `gripper_open`) — the same `geometry_stats` family `scripts/attach_l2_to_rollout.py` already
  computes (report-only there).

**The re-pin may NEVER read (hard prohibitions — violating any voids the freeze):**
- Any **attacked / GCG / injected** rollout, score, or suffix — directly or indirectly.
- The **held-out benign evaluation split** on which FPR is reported (invariant #3) — radii come from the
  **calibration** split only, keeping the held-out FPR honest.
- Any **detection metric** (TPR, AUC, separation) — the radii must not be chosen to maximise detection.

---

## 3. Pre-committed estimators (the formula is fixed NOW, before the numbers exist)

All quantities are computed on the §2 benign calibration split. Percentiles and the margin factor are
**declared here** so the resulting number cannot be selected post-hoc. Let `m = 1.2` be the single,
shared **margin factor** (20 % headroom), applied identically to both radii.

Define, over the benign calibration split:
- `A = { min over steps of ‖ee(τ) − anchor‖ : per SUCCESSFUL benign rollout with a resolvable anchor }`
  — how close a *successful* benign policy actually gets to the goal.
- `G = { ‖ee(τ_grasp) − anchor‖ : per open→close grasp event in SUCCESSFUL benign rollouts, resolvable anchor }`
  — the EE↔goal distance at benign on-goal grasps.
- `D = { min over steps of min_{distractor o} ‖ee(τ) − pos(o)‖ : per benign rollout (all) }`
  — how close benign rollouts come to the *nearest non-goal* object.
- `Dg = { min_{distractor o} ‖ee(τ_grasp) − pos(o)‖ : per benign grasp event }`
  — distractor distance at benign grasp events.

**Candidate radii (pre-committed):**
```
r*   = m · median(A)        # engagement radius: a successful benign approach must read "at goal"
R_g* = m · P90(G)           # grasp radius: a benign on-goal grasp must read consistent (low p3)
```
(`P90` = 90th percentile; `median` = 50th. Use linear interpolation; if a set has < 5 elements the re-pin is
**aborted** for that radius — insufficient benign evidence → §4 conservative default.)

**These constants are author-settable at SIGN-OFF ONLY (loosen or tighten now).** The margin `m`, the §3
percentiles (`median` for `r*`, `P90` for `R_g*`), and the §4 guard percentiles (`P10` of `D`/`Dg`) are
deliberately exposed for the author to adjust to taste **before §7 is signed** — e.g. a smaller `m` or a
higher `r*` percentile for a stricter "at goal", or `P5`/`min` instead of `P10` for a more conservative
distractor guard. **Once §7 is signed** (necessarily before the benign split is computed and before any
attacked output exists) **they are LOCKED.** Changing any of them afterwards — especially in response to the
resulting `A,G,D,Dg` numbers, the detection metrics, or any attacked data — **voids the pre-registration**
(invariant #2). The freedom is in *setting* the knobs up front, never in *re-tuning* them later.

---

## 4. Feasibility guards and the conservative default (no-change)

A candidate is **adopted only if it does not break the other primitive's benign behaviour.** Otherwise the
frozen value is **kept** and the conflict is reported as a **pre-registered limitation** (never silently
forced through):

- **`engagement_radius`** — adopt `r = round(r*, 0.005 m)` **iff** `r* < P10(D)` (the new "at goal" radius
  stays inside the benign closest-distractor approach, so benign distractor pass-bys still do **not** trigger
  P2). **If `r* ≥ P10(D)`** (benign rollouts come as close to distractors as to the goal — the inverted-scale
  case the step-5 n=1 episode *hints* at: target-min 0.072 m vs distractor-min 0.0525 m): **do NOT re-pin**;
  keep `r = 0.05 m`; report the infeasibility as a headline limitation (see the anchor note below) and defer to
  the indicated fix (M3 resolver / S2), not a radius tweak.

  > **Anchor-artifact hypothesis (why the scale may invert — read before reporting the limitation).** The
  > anchor is `object_poses[target_region]` = the **placement region** (e.g. `plate_1` *centre*), but on a
  > LIBERO-Spatial pick-and-place the gripper goes to the **grasped object** (the bowl), not the plate centre.
  > So EE↔anchor stays large (n=1: min 0.072 m; the grasp happens at 0.131 m from the plate centre) while a
  > distractor object near the path can be closer (0.0525 m) — the inversion is then a **region-vs-object
  > anchor** artifact, which is exactly schema §6's known v1 limitation (*"placement-region anchors that aren't
  > objects are out of scope; single-anchor reach/pick only"*). If the benign split confirms this, the correct
  > response is **not** a larger `r` but a **phase-aware / object-level anchor** (schema §7 **S2** multi-phase
  > sub-goal consistency, or an M3 object-level `GoalResolver`) — report it that way, as a motivation for M3,
  > not as "the radius is wrong". *(n=1 hypothesis; the split decides.)*
- **`grasp_radius`** — adopt `R_g = round(R_g*, 0.005 m)` **iff** `R_g* < P10(Dg)` (an on-goal grasp reads
  consistent while a distractor grasp still saturates p3→1). **Else** keep `R_g = 0.10 m` and report the
  conflict.

**Default when in doubt = NO CHANGE.** If the split is too small, an estimator aborts, a guard fails, or the
procedure is ambiguous, the **frozen 0.05 / 0.10 stand** and the benign saturation (e.g. step-5's `max=1.0`)
is reported as a known metric limitation. A non-re-pin is an acceptable, honest outcome; an attack-informed
re-pin is not.

---

## 5. Execution protocol (one-shot, dated, recorded)

1. Build / confirm the M2 benign calibration split (§2). **No attacked rollout may exist in scope** at this
   point, or the run is void.
2. Compute `A, G, D, Dg`, then `r*, R_g*`, then apply the §4 guards. Log the full computation (inputs, the four
   distributions' summary stats, candidates, guard outcomes) to a **write-once** `results/` run dir.
3. Whatever the outcome (re-pin or keep), record a dated **deviation entry** in
   [`metric-a-annotation-schema.md`](./metric-a-annotation-schema.md) §5: date, before/after values, the
   distributions that produced them, and the guard results — per that document's §0 freeze statement.
4. **Re-freeze.** The radii are fixed again; any later change is a new, separately pre-registered deviation.
5. Only **after** steps 1–4 are complete and recorded may the **step-6 tiny GCG run** inspect attacked output.

---

## 6. Evidence that motivated this rule (the TRIGGER, explicitly NOT the re-pin basis)

CSB step-5 benign `libero_spatial` task-0 rollout (`results/_smoke/2026-06-18T15-23-29Z-l2-attach`,
**n = 1**): min EE↔target ≈ **0.072 m** (> frozen `r` 0.05 → "at goal" never fires); on-goal grasp at EE↔target
≈ **0.131 m** (> frozen `R_g` 0.10 → benign p3 saturates to **1.0**); min EE↔distractor ≈ **0.0525 m**
(< the target approach → the inverted-scale hint in §4); benign `score_summary` **max 1.0 / mean 0.44**. This
demonstrates the frozen placeholders undershoot the real scene scale — **the reason this pre-registration
exists**. It is **n = 1** and is used **only** to motivate the procedure; the actual numbers (`A,G,D,Dg`) come
solely from the §2 multi-rollout benign calibration split, never from this single episode and never from
attacked data.

---

## 7. Sign-off

- [ ] **Author** — confirms the estimators (§3), guards (§4), and one-shot/recording protocol (§5).
- [ ] **Supervisor** (if required) — confirms the pre-registration is fixed before any attacked output is seen.

Until both boxes are ticked this document is **PROPOSED**; the step-6 tiny GCG run must not proceed.

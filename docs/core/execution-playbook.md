# Embodiment Evasion Tax — Execution Playbook (operational companion)

> **What this document is.** The *operational* companion to the *understanding* doc
> [`goal-action-consistency-detector.md`](./goal-action-consistency-detector.md). That doc explains
> **what** the Embodiment Evasion Tax is and **why** (threat model, novelty, design rationale). **This** doc keeps the *execution* on
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
**committed**; (b) *robustness against an adaptive attacker that knows the detector* — the **realistic-adaptive
arm (H5 / H6-D, M4, Branch N/N−)**, pursued only if the M1 micro-bench shows it affordable and M4 finishes with slack; if dropped we make
claim (a) and **do not** claim adaptive robustness. The privileged-state floor (below) guarantees a defensible
dissertation even if the deployable arm underperforms.

**Claim boundary (Codex review #2, 2026-06-02 — load-bearing; trigger updated 2026-06-16 for the CSB A5000 registered-compute switch).**
The cross-layer *Embodiment Evasion Tax* is **two distinct claims, never conflated**: **H6-A** — the *oracle
intrinsic action-space frontier* (an L2-**oracle** frontier + **non-adaptive** L0/L1/L2 detection; it makes
**no cross-layer "tax" claim**, because the layers are not attacked by a common attacker), the **guaranteed
floor** result — and **H6-D** — the *deployable-vs-deployable* cross-layer tax under a **matched** realistic
attacker/budget, the **headline**. A fair "L2 costs more than L1" statement **requires H6-D**. **Which one
becomes the committed headline is selected at the M1 on-GPU timing micro-bench** (§2 *Compute branches*): if the
measured GCG / L1-extraction cost on the **registered RTX A5000** makes the deployable matched-attacker matrix
affordable within the calendar → **Branch N / N−** commits H6-D; if not → **Branch F** reports H6-A + the honest
oracle-gap and marks the cross-layer deployable tax **unresolved** (fallback title, §3a). **H6-A is delivered
either way** — *the tax headline is H6-D's, never H6-A's.*

**Compute (2026-06-16 — D8 updated: registered compute = CSB A5000).** The registered hardware is now **CSB
`ecs3-0202` = 2× RTX A5000, 24 GB each** (the first GPU actually accessible; A100/H100-Kelvin2 login was never
established → **Kelvin2 demoted to a backup contingency**). OpenVLA-7B runs in **bf16** (~14 GB → fits **one**
24 GB A5000 with rollout headroom; the registered card is a **single** A5000). The three pre-registered branches
selected at the M1 on-GPU timing micro-bench (§2 *Compute branches*) are unchanged in **structure** — **N**
(full deployable tax) / **N−** (scoped) / **F** (oracle frontier only) — but the A5000 is **~2.5–4× slower** than
A100/H100 and **has no published OpenVLA-GCG prior**, so the **M1 micro-bench is the sole budget source** and the
branch is expected to skew **N−/F** (measured, not assumed). **Reproducibility rule (now load-bearing): all
registered runs commit to the A5000** — cross-HW comparison is forbidden within a claim, so Kelvin2, if ever
used, is a **separate** registration. **Log exact card + precision + parallelism per run.** Headline reframe =
**§3a / H6**; the **branched roadmap = §2**; **what to implement = §4b**.

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

- **Last updated:** 2026-06-26 (**surrogate-GCG runner now observable + partially recoverable** — `run_gcg`
  (`src/evasion_tax/attack/gcg.py`) gained an optional **keyword-only, exception-isolated `on_step` callback**
  (`feat(gcg)` `57cde59`; passed a *copy* of the incumbent suffix, a logging/disk failure can never abort a
  multi-hour search) and the surrogate driver (`scripts/run_surrogate_gcg.py`, `feat(surrogate)` `4f40b53`) uses it
  to print a `[gcg] step N/500 best_loss=… elapsed=…s` heartbeat every **25** steps and atomically (over)write a
  **quarantined `artifacts/untrusted/<run_id>/checkpoint.json`** carrying the current best suffix every **50** steps.
  This is the durable fix for the **silent overnight death** of the 2026-06-26 unattended bf16 pilot (idle-suspend /
  session loss killed the run with **no progress emitted and nothing recoverable**, and the remaining arms never ran).
  Core GCG math is **byte-identical when the callback is absent**; **no change** to the final write-once artifact or
  the `results/` pointer schema (the checkpoint is a mutable sidecar). **Resume-from-checkpoint is deferred (YAGNI)** —
  this lands only the foothold (best suffix + step), since bit-exact resume needs the candidate-sampler RNG state.
  **11 new TDD tests, full suite 657 green, ruff clean**; plan `docs/surrogate/plan/2026-06-26-gcg-progress-checkpoint-callback.md`;
  pushed to origin/main. The on-box behaviour (a long run printing the heartbeat + refreshing `checkpoint.json`) is
  validated on the CSB box during the next pilot, not by a GPU unit test. *Prior 2026-06-24:* **M1-plan mac-side COMPLETE + pushed to origin/main — box-ready**: Tasks 0–6 + the
  benign/attack drivers built via TDD (630 tests green, ruff + pyright clean). New: `eval/schema_repin.py` (DM-3 re-pin),
  `attack/redirect_target.py` (D2/DM-5 redirect spec + codec ids), `eval/rollout_runner.py` (frozen-suffix runner +
  window-scored ASR + geometry; step-4 smoke refactored to a thin caller), `eval/separation.py` (benign-vs-attacked
  table), `eval/m1_gate.py` + `scripts/m1_gate_report.py` (H1 GO/NO-GO), `scripts/run_benign.py` + `scripts/run_attack.py`
  (GPU-guarded drivers, pure glue tested with the model mocked), `scripts/repin_schema.py` (the DM-3 lock bridge),
  `configs/m1_viability.yaml`. **NEXT = run the box sequence** (run_benign → repin_schema → run_attack dry-run →
  registered run → m1_gate_report; see `configs/m1_viability.yaml` header), then Task 7 (feed the real cost into
  `branch_select`). GPU bodies are **[VERIFY on box]** (mirror the verified step-4/5.5/6 patterns; dry-run the attack
  first). *Prior same-day:* **DM-3 / D-3 SchemaA radius re-pin rule LOCKED** (benign-only pre-registration; author sign-off via `docs/plans/2026-06-24-m1-viability-gate.md`; supervisor pending) → executable `evasion_tax.eval.schema_repin.repin_schema_from_benign` (pure §3 estimators + §4 guards; **6 TDD tests, ruff + pyright clean**); **DM-1** the standalone early-stop bench (i) is cancelled and **folded into the (ii) RoboGCG redirect attack** (`bench_early_stop.py` kept). Next M1 mac work = Task 1 (redirect-target spec) / Task 3 (separation table); box = benign baseline → freeze re-pin → attacked run. *Prior 2026-06-23:* **CSB BRING-UP step 6 GREEN + D8 REGISTERED → provisional Branch N−, hard-F default** — true-batch GCG micro-bench on the A5000 (bf16 / flash_attn2 / exclusive): **s/step=33.19 s**, **s/target(worst)=16,595 s ≈ 4.61 h** (sw=512/ns=500, **early_stop OFF** → conservative upper bound), loop ablation 17.48 s → `speedup_k≈1.53`, **max-B=43 @ 21.3 GiB** (VRAM ceiling, **not** branch-critical, DB-3); `results/2026-06-23T13-34-55Z-gcg-microbench/`. **Task-5 calendar re-derivation:** box is **author-exclusive** (no contention) → GPU-h limited by job-persistence/sleep/OOM not sharing; author confirmed **2 cards usable** + **early-stop bench first**. `branch_select` on ≈875 adaptive GPU-h (2 cards × ~125 GPU-h/wk/card × ~5-wk window × ~70 %; adaptive_mult=3 **EST**) → **provisional N−** (62 runs / ~20 targets×3), **hard-F default**; **sensitivity:** 1-card-worst-case→**F**, early-stop-or-2-cards→**N**. Branch hinges on the **unmeasured early-stop** + 2nd card → locks at the M1/M2 adaptive bench; **M3/H6-A delivered in every branch**. Next = M1 (early-stop steps-to-success bench + benign baseline + RoboGCG targeted redirect → GO/NO-GO). **Unattended-run runbook** (tmux + `systemd-inhibit`, no sudo; per-target checkpoint + auto-restart) in `docs/gpu/CSB/plan.md`. Prior 2026-06-19: **CSB BRING-UP step 5.5 GREEN** — bf16+flash_attn2 `.backward()` reaches the **input embeddings**: **finite & non-zero** (‖g‖≈97 flash / 92 sdpa, loss≈18.5 to the target action tokens, grad shape [1,26,4096]), **peak VRAM 15.49 GiB / 23.5 GiB → fits one card** (weights frozen → only ~1 GiB over the step-3 forward → a GCG backward step is cheap, ≈8 GiB headroom for candidate batching); the gradient is read via a forward hook adding a `requires_grad` `delta` to the token embeddings (OpenVLA's multimodal `forward` takes no external `inputs_embeds`), so `delta.grad == d(loss)/d(inputs_embeds)`. `scripts/smoke_openvla_gradient.py`; runs `results/_smoke/2026-06-19T12-57-10Z-openvla-gradient-smoke` (flash) / `…T12-57-30Z…` (sdpa), commit `0300670` → the GCG "gradients are obtainable" premise **HOLDS**. Prior 2026-06-18: **CSB BRING-UP step 5 GREEN** — L2 detector attached to the real step-4 rollout end-to-end, **offline on the mac, model-free, from the committed step-4 run dir** (no box session needed). New seam `src/evasion_tax/eval/rollout_io.py` (JSON→`Rollout` deserializer + **D-5 source-run provenance binding**: sibling `run.json`/`episode_meta.json` cross-check + `steps.json` SHA-256) + driver `scripts/attach_l2_to_rollout.py`. Two honest halves (Codex `[high]`): **state half** = 90 per-step metric-A scores finite in [0,1] + a `rollout_fires` decision (illustrative τ); **action half** = `(90,7)` finite, **non-degenerate**, D2 `reached_window` path exercised (anti-false-confidence test: a *zeroed* action stream now **fails** the gate while the state half is unchanged). Provenance validated (`steps_sha256 0deaf431…`) → `results/_smoke/2026-06-18T15-23-29Z-l2-attach/l2_attach_report.json`. **27 new TDD tests, 437 suite green; ruff + pyright clean.** **Wiring de-risk only — NO separation/calibration/deployable claim**; benign metric-A scores NOT near zero is **expected** (the `engagement_radius=0.05`/`grasp_radius=0.10` placeholders undershoot the real scene scale — geometry_stats are report-only D-3 calibration input, NOT a re-pin). **Next = step 6 (GCG tiny run → D8 timing micro-bench → Branch N/N−/F).** Prior same-day **CSB BRING-UP step 4 GREEN** — `libero_spatial` task-0 episode completed on the A5000 (**90 policy steps, success=True**, sdpa, **peak VRAM 14.50 GiB / 23.5 GiB → fits one card**), `RolloutStep` schema logged → `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke`; **EGL (`MUJOCO_GL=egl`) initialised + rendered → headless-render risk retired**. Three bring-up gotchas resolved & recorded (`configs/env/requirements-gpu.txt` + `docs/gpu/CSB/plan.md` **Step 4 how-to** + step-4 plan Task 2a): **(1)** `uv pip install -e ~/LIBERO` does **NOT** make `import libero` importable — LIBERO's top `libero/` is a **PEP-420 namespace package** (no `__init__.py`) and uv's PEP-660 editable finder won't expose it (pip's *legacy* `-e` would, since it puts the repo root on `sys.path`) → **use `PYTHONPATH=~/LIBERO`** at import AND run time (the smoke script only adds `--openvla-root`); **(2)** the OpenVLA eval-helper import chain (`robot_utils`→`prismatic`(eager)→`dlimp`→`tensorflow_datasets`→`tensorflow_metadata`) pulls the TF stack, and **tfds 4.9.3 caps nothing on tensorflow-metadata** → pip pulled tfmd 1.21.0 (needs protobuf≥5.26 `runtime_version`) → ImportError on protobuf 4.25.9 → **pin `tensorflow-metadata<1.16` + `protobuf<5`** (tensorflow 2.15.0 / tfds 4.9.3 already correct; TF is import-only here → runs no ops, steals no A5000 memory; startup I/E/W logs benign); **(3)** the finetuned checkpoint registers action norm_stats under **`libero_spatial`** not `*_no_noops` (the training-data name we'd assumed) → **`--unnorm-key libero_spatial`** (now the script default). **Next = step 5: attach the goal-action detector (L2) to this real rollout** (`docs/gpu/CSB/plan.md` step 5; the smoke run is non-registered, so step 5/6 + the 2b checkpoint-provenance row precede any registered M1 run). Prior 2026-06-17: **CSB BRING-UP step 3 GREEN** — `openvla/openvla-7b` loaded **bf16 + sdpa** on one A5000 (cuda:0), one dummy forward → **valid 7-DoF action**, **peak VRAM 14.46 GiB / 23.5 GiB** (no flash-attn needed); `scripts/smoke_openvla_load.py` + model-free `validate_action_vector` (TDD), commits `a24b77a`/`87e9a3f`. **huggingface-hub<1.0 lock fix** (`87e9a3f`): the open `>=0.20` had resolved to hub 1.17.0 in `uv.lock`, which `uv run`/`uv sync` re-applied every time and broke transformers 4.40.1 (`ImportError: hub<1.0`) — a **lock-inconsistency, same class as the numpy<2 fix**, NOT lock-external; pinned `>=0.20,<1.0` → hub 0.36.2. **Next = step 4: one LIBERO episode (EGL) with the bf16 policy.** Prior same-day: **CSB BRING-UP steps 1–2 GREEN on the box** — env stood up via **uv** on **Python 3.10** (`.python-version` repinned 3.11→3.10; `uv venv --python` alone kept reverting), **torch 2.2.0+cu121 / numpy 1.26.4 / CUDA True / both RTX A5000 visible**, torch↔numpy interop OK; **395 model-free tests pass on the box** — 3 torch-absent tests made env-independent via *mocked* absence so they pass with torch installed; **`git` installed via micromamba** (box has no git/conda/pip + no sudo); repo must be **`git clone`d, not zip-DL'd** (else `capture_env` git_commit=None); **numpy pinned `<2` in pyproject+lock** (torch 2.2.0 needs the numpy-1.x ABI). Next = **step 3: OpenVLA-7B bf16 dummy forward**. Commits `f8e0e19`/`3b16684`/`52a77ea`. Prior 2026-06-16: **REGISTERED COMPUTE SWITCHED → CSB `ecs3-0202` = 2× RTX A5000, 24 GB each** (verified `nvidia-smi`); the first GPU actually accessible → **the GPU-access blocker is resolved**; **Kelvin2 demoted to backup**; bf16 fits one 24 GB card; A5000 ≈2.5–4× slower than A100/H100 + no published GCG prior → D8 branch expected N−/F, measured at M1; all registered runs commit to the A5000 (no cross-HW mixing). See §10 + `docs/gpu/CSB/{pc-spec,plan}.md`. Prior 2026-06-09: **state-only LIBERO now RUNS locally → concrete `state_libero.py` adapter PULLED FORWARD** from the GPU node — real BDDL `target_region` (`obj_of_interest[-1]`), `<obj>_to_robot0_eef` phantom-object bug fixed, Tier-L smoke now PASSES, **395 tests green**; see §10 + `docs/plans/2026-06-09-libero-state-adapter.md` + `docs/setup/libero-local-env.md`. The 2026-06-03 "real LIBERO deferred to GPU" is overturned (env-assembly, not fundamental; robosuite 1.4.0 was the real requirement). Earlier 2026-06-03: **local-prep Task 10 LIBERO smoke DONE** — Tier R/robosuite validated the `PrivilegedState` contract; `577c2d1`, see §10 → **all local-prep Tasks 0–11 complete**. Earlier 2026-06-03: granted GPU = **Kelvin2** NI-HPC@QUB → cluster docs `docs/gpu/` + Task 11 GPU runbook DONE `a491a63`/`9c3eaff`. Prior 2026-06-02: Codex review #2 across the plan + eval-harness held-out-FPR fix — see §10)
- **Phase:** **Design → M0 (exiting)**. Coding gate **lifted for M1–M2 (author OK, 2026-05-31)**; **pre-GPU local build underway** — model-free M1–M2 components on M1/8 GB (`docs/core/local-prep-plan.md`). OpenVLA/GCG/LIBERO *runs* now have an accessible registered box (**CSB A5000**, 2026-06-16). **Headline reframed → Embodiment Evasion Tax (§3a, H6); milestones re-mapped (§2); what-to-implement = §4b; registered compute = CSB `ecs3-0202` 2× RTX A5000 24 GB (D8, 2026-06-16; Kelvin2 = backup) — bf16 default, tier→branch reframe (§2 *Compute branches*). CSB bring-up steps 1–3 DONE 2026-06-17 (env on Py3.10 + torch cu121 + numpy<2, 395 tests green; OpenVLA-7B bf16 dummy forward = valid 7-DoF action, 14.46 GiB peak); **step 4 GREEN 2026-06-18** — `libero_spatial` task-0 episode completed on the A5000 (90 steps, success=True, sdpa, 14.50 GiB → fits one card; EGL rendered → headless risk retired); libero via `PYTHONPATH=~/LIBERO`, `tensorflow-metadata<1.16`/`protobuf<5`, `--unnorm-key libero_spatial`. **step 5 GREEN 2026-06-18** (offline on the mac — L2 detector attached to the real rollout end-to-end, model-free: `eval/rollout_io.py` seam + D-5 provenance + `scripts/attach_l2_to_rollout.py`; report `results/_smoke/2026-06-18T15-23-29Z-l2-attach`; wiring de-risk only). **step 5.5 GREEN 2026-06-19** (GCG-gradient smoke on the box — finite non-zero input-embedding grad through bf16+flash_attn2, ‖g‖≈97, peak 15.5 GiB → fits one card; sdpa cross-check matches; `results/_smoke/2026-06-19T12-57-10Z-openvla-gradient-smoke`). Next = step 6 (GCG tiny run → D8 micro-bench → Branch N/N−/F).**
- **Last completed:** Theme = Embodiment Evasion Tax; understanding doc; **RoboGCG defence verified** (`docs/references/`); **D1–D7 resolved** (§6/§10; D4/D7 OPEN → M1); **Phase-3 implementation plan drafted**; **Codex third-party review incorporated** (causal prefix-window + detection latency; per-rollout FPR + CIs; metric(A) schema frozen; fair-calibrated baselines; M5→stretch — see §10). **Codex review #2 (2026-06-02) incorporated** — H6 split into **H6-A** (M3, oracle-intrinsic) + **H6-D** (M4, deployable cross-layer tax); novelty narrowed (runtime VLA-safety lane now occupied, 4 papers verified); power / coverage / L1-control / single-route pre-registrations added (§10).
- **Currently:** Executing the **pre-GPU local-prep plan** (`docs/core/local-prep-plan.md`) on M1/8 GB via TDD. **Done 2026-05-31: Tasks 0,1,2,3,4,5,6,7,8,9** — env scaffold, repro infra, data records, **action codec (OpenVLA formula verified from source `c8f03f48`; provenance recorded)**, privileged-state adapter, **metric (A) — annotation schema FROZEN (`docs/core/metric-a-annotation-schema.md`, `2c2f163`) + causal scorer (P1 progress / P2 distractor / P3 grasp; combine=max; privileged anchor via resolver seam; non-causal monitoring-ceiling variant)**, FP-calibrated detector, eval harness (ROC/AUC, TPR@FPR + Wilson/CP CIs, latency, split-disjointness), **M2 baselines (`3287c5c`): goal-agnostic χ²-OOD action anomaly + perplexity/text-only filter (mock + GPU stub), both through the *same* `calibrate`**, **config/scripts/figures (`60b0462`): frozen pydantic `Config` + `one_variable_diff`; shared GPU guard (no-CUDA → exit non-zero, no silent no-op); `make_figures` regenerates ROC/score-hist/TPR@FPR-CI per condition purely from a logged `results.json`; 6 runnable scripts** — **362 tests green, full `src/evasion_tax` type-clean** (see plan Status table). **2026-06-02 fix: `tpr_at_fpr`/`run_condition_matrix` now report `realised_fpr` on the *held-out* benign split (invariant #3), not the calibration split it was previously measured on; the in-sample number is retained as the `calib_fpr` diagnostic (see §10).** **2026-06-03: the model-free §4b instruments are COMPLETE** — L1 internal probe (`fadc008`), idealized action-space attacker (`ac4f229`), **and the cross-layer eval + ΔASR tax metrics (§4b-III) + the Codex-#2 model-free hooks (#3 power rule, #6 coverage manifest, #10 primary tax scalar)** — so GPU day-1 is now "plug in activations + rollouts". **2026-06-03: granted GPU = Kelvin2 (NI-HPC@QUB)** — cluster docs in `docs/gpu/` (overview/connection/quickstart/running) + **local-prep Task 11 DONE** (`docs/setup/gpu-runbook.md` day-1 sequence; `configs/env/requirements-gpu.txt` source-fetched OpenVLA pins; checkpoint provenance placeholder rows — `a491a63`/`9c3eaff`). **2026-06-03: local-prep is now FULLY COMPLETE (Tasks 0–11)** — Task 10 LIBERO smoke executed (`577c2d1`): a state-only **Tier R (robosuite)** env validated the Task-4 `PrivilegedState` contract against **real MuJoCo ground truth, unmodified** (no schema change; Task-5 freeze stands); real LIBERO deferred to the GPU node (`benchmark` hard-imports torch + `OffScreenRenderEnv` needs GL — `docs/setup/libero-local-notes.md`). *Lint debt cleared 2026-06-03 (`2963e72`): the former 3 pyright errors + 1 ruff B905 in test files are gone — **362 tests green, full `src/evasion_tax` + test tree type-clean & ruff-clean**.*
- **▶ NEXT ACTION:** **M1 viability gate — CSB bring-up steps 1–6 COMPLETE (step 6 DONE 2026-06-23 → provisional Branch N−, hard-F default; D8 §10).** First M1 GPU items, run unattended per `docs/gpu/CSB/plan.md` (tmux + `systemd-inhibit`, per-target checkpoint): **(i) the A5000 early-stop steps-to-success bench** (turns the 4.61 h worst-case into the realistic per-target cost → confirms/lifts the branch; author-agreed to run *before* the M3/M4 matrices); **(ii) benign LIBERO baseline + RoboGCG targeted redirect + metric-(A) signal**; **(iii) the adaptive GCG-against-the-probe cost** (the number that **locks** the branch) → **GO/NO-GO (H1)**. **Before any attacked output is scored:** the dated benign-only pre-registered `SchemaA` radius re-pin rule is **LOCKED (D-3, 2026-06-24; executable `evasion_tax.eval.schema_repin.repin_schema_from_benign`)** — the M1 attacked run consumes the **frozen** re-pinned schema via `--schema-from` and never re-pins from attacked data (invariant #2). *Historical step-6 bring-up context follows.* **Prior: step 5 done 2026-06-18, offline on the mac** (L2 attached to the real step-4 rollout end-to-end, model-free: `scripts/attach_l2_to_rollout.py` + `src/evasion_tax/eval/rollout_io.py` (JSON→`Rollout` seam + D-5 provenance), state + action halves, 27 new TDD tests; report `results/_smoke/2026-06-18T15-23-29Z-l2-attach/l2_attach_report.json` — **wiring de-risk only, NO separation/calibration/deployable claim**). **Next, on the box — (0) the GCG-gradient prerequisite smoke = DONE 2026-06-19** (`scripts/smoke_openvla_gradient.py`: finite non-zero input-embedding grad through bf16+flash_attn2, ‖g‖≈97 / sdpa 92, peak 15.5 GiB → fits one card, ≈8 GiB headroom; `results/_smoke/2026-06-19T12-57-10Z-openvla-gradient-smoke`; `docs/gpu/CSB/plan.md` **step 5.5**) — **then** a tiny GCG run (few steps, 1 example) then the **D8 on-GPU timing micro-bench** → *verify:* attack harness runs; record `s/target`, peak VRAM, max candidate-batch at 24 GB → **selects Branch N/N−/F** (`docs/gpu/CSB/plan.md` step 6). Run env recipe + the three step-4 gotchas (`PYTHONPATH=~/LIBERO`; `tensorflow-metadata<1.16`/`protobuf<5`; `--unnorm-key libero_spatial`): `docs/gpu/CSB/plan.md` **Step 4 how-to**. **Before step 6 inspects attacked output:** the dated, benign-only, pre-registered rule for any `SchemaA` radius re-pin is **LOCKED (D-3, 2026-06-24; executable `schema_repin.py`)** — step 5's `geometry_stats` showed the placeholder radii (`engagement_radius=0.05`/`grasp_radius=0.10`) undershoot the real scene scale, which is the *trigger*; the rule is now fixed (§3 estimators + §4 guards) and the re-pin is computed from the benign baseline split only, *before* any attacked data is seen. *Background (still valid):* the model-free §4b interfaces + Codex-#2 hooks are **DONE** (L1 probe + confound scaffolding `fadc008`; idealized attacker `ac4f229`; cross-layer eval + ΔASR tax scalar #10, coverage manifest #6, power rule #3 — 2026-06-03). **Task 11 GPU runbook also DONE** (`docs/setup/gpu-runbook.md`, Kelvin2). **Local-prep is now fully complete (Tasks 0–11)** — Task 10 done (`577c2d1`): Tier R/robosuite validated the `PrivilegedState` contract, real LIBERO deferred to the GPU node. **GPU access is now unblocked → the registered box is CSB `ecs3-0202` (2× RTX A5000, 24 GB).** **On that box (M1) — follow the bring-up ladder in `docs/gpu/CSB/plan.md` (then `docs/setup/gpu-runbook.md` for the protocol):** verify env (`flash-attn`/cu121 torch under CUDA 13.2; `MUJOCO_GL=egl`); stand up **OpenVLA-7B in bf16 on one A5000**; benign baseline + RoboGCG *targeted* redirect; **GCG micro-bench on the A5000 (fixes D4/D7; the published H100 timings do NOT transfer)**; metric-(A) signal incl. coarse-goal check; **run the on-GPU timing micro-bench → select compute Branch N/N−/F (D8)** → **GO/NO-GO gate (H1)**. *Re-read `ecs3-0202` CPU/RAM/storage + secure a quiet/exclusive window for the timing bench (`docs/gpu/CSB/pc-spec.md`).*
- **Blockers:** none. *(GPU access — the long-standing gate — is resolved: CSB `ecs3-0202` A5000 is accessible. Remaining pre-run items are bring-up, not blockers: verify env under CUDA 13.2, re-read `ecs3-0202` CPU/RAM/storage, secure a quiet window for the D8 timing bench — `docs/gpu/CSB/`.)*
- **Open decisions outstanding:** **D4, D7** (OPEN → M1 GCG micro-bench on the **registered CSB A5000**) **＋ D8 compute** (registered compute = CSB `ecs3-0202` 2× RTX A5000 24 GB, switched 2026-06-16; Kelvin2 = backup → the M1 on-GPU timing micro-bench on the A5000 selects Branch N/N−/F, expected N−/F). D1/D2/D5/D6 DECIDED; **D3 re-tiered** by the reframe (operator-goal rung committed, task-ID → stretch / Branch-N, see §2 M5).
- **Floor secured?** ❌ not yet (target: end of **M2**, ~Jul 12).
- **Novelty status:** headline = **Embodiment Evasion Tax**, delivered as **two claims**: **H6-A oracle intrinsic action-space frontier + non-adaptive cross-layer detection = committed (M3, guaranteed floor — delivered in every compute branch)**; the **fair cross-layer *tax* (H6-D, deployable-vs-deployable, matched attacker) = committed-if-affordable (M4, Branch N/N−, selected at the M1 timing micro-bench — D8)** — *the tax headline is **M4's, not M3's***; **M5 reference-ladder + SABER = secondary/stretch (promotion gated on deadline progress)**.
- **Direction (DECISION, author-converged 2026-06-01 — flag for supervisor sign-off):** research **core reframed** to the *Embodiment Evasion Tax* measurement (see §3 **H6** + **§3a**). The behavioural detector is recast as an **instrument** measuring per-layer adaptive-evasion cost (**L0** input < **L1** internal-probe < **L2** action-monitor), **not** a claimed defence/"firewall". Scope held to the **instruction channel** (RoboGCG primary, SABER secondary); physical/CoT injection (TRAP) = future arm, out of committed scope. Floor (M2 + cheap idealized action-space frontier) unchanged → deliverable still guaranteed. *Citation pass DONE 2026-06-01:* all 5 flagged items resolved + **16 cited PDFs downloaded, gitignored, SHA-256-pinned** with provenance (`docs/references/README.md`). Net (revised, Codex review #2): a **2026 cluster of runtime/inference-time VLA-safety work now exists** (Pre-VLA runtime verification `2605.22446`; HazardArena Safety-Option-Layer `2604.12447`; Concept-Dictionary activation-level `2602.01834`; IGAR attention recalibration `2603.06001` — all **independently verified on arXiv 2026-06-02**), so we **do not** claim the runtime lane is unoccupied. **Narrowed novelty:** *adaptive evasion-cost **measurement** for the **instruction channel**, with **per-rollout FP calibration** and an **action-space intrinsic frontier*** — none of the above measure adaptive evasion cost or FP-calibrate against an adaptive injection attacker (all are mitigations / benign-OOD). *Two of them (`2602.01834`, `2603.06001`) bear on the **L1 arm's** novelty → cite + differentiate there too (§4b-I).* Title **LOCKED** (§3a, pending supervisor).

---

## 2. Milestone roadmap (W1 = week of 2026-06-01; submit early Sep 2026)

**Compute branches (2026-06-16, D8 — selected at M1).** Registered hardware is **2× RTX A5000 (24 GB each) =
CSB `ecs3-0202`** (`docs/gpu/CSB/`); a **single** A5000 is the registered card and OpenVLA-7B runs in **bf16**
(~14 GB, fits with rollout headroom). **Kelvin2** (A100/H100, `docs/setup/gpu-runbook.md`) is a **backup**
only. There is no longer a binary "does the cluster exist?" gate — instead the **M1 on-GPU timing micro-bench**
(measured GCG s/target at bf16 **on the A5000**, L1 activation/attention extraction overhead, LIBERO rollout
throughput, effective parallelism) is fed against the remaining calendar to compute the **affordable experiment
matrix**, which selects one of three **pre-registered** branches. *Because the A5000 is ~2.5–4× slower than
A100/H100 and has no published OpenVLA-GCG prior, the affordable matrix is likely smaller → expect the bench to
land on **N− or F**; this is measured, and **M3/H6-A is delivered in every branch** regardless.*

- **Branch N (full deployable tax).** The affordable matrix covers the deployable-vs-deployable matched-attacker
  H6-D experiment (realistic adaptive GCG-through-policy vs deployable L1 *and* L2; one suite, fixed budget;
  benign N ≥ ~300 for the **1 %-FPR primary**; ≥ 3 seeds) with write-up slack. → **commit H6-D as the headline**;
  H6-A is the supporting oracle frontier. Stretch (M5 ladder / SABER) becomes promotable if further slack.
- **Branch N− (scoped deployable tax).** The affordable matrix covers a *reduced* H6-D (the **one** deployable
  route B-or-C, fewer targets/seeds, **5 %-FPR primary**). → **commit a scoped H6-D** with explicitly reported
  reduced power **+** the honest oracle-gap. Title = main, subtitle scoped to the reduced matrix.
- **Branch F (oracle frontier only).** Throughput / queue cannot fit even the reduced deployable matrix within
  the calendar. → **headline = H6-A** (oracle intrinsic frontier + non-adaptive L0/L1/L2 ordering); H6-D
  reported as **not run / unresolved**; fall back to the pre-registered oracle-frontier title (§3a).

The branch is chosen by **measured affordability, pre-registered now** (before the numbers are in) so the choice
is honest, not hope-driven. **M3 (H6-A) is delivered in every branch** — the dissertation is safe regardless of
which lands. Within the adaptive arm, whether **adaptive-GCG-against-the-L1-probe** is in scope is itself an M1
micro-bench item (D7 / Codex #2 #5): if it is too costly at the measured budget it is reported as non-adaptive
L1 + the L2-oracle frontier. **Reproducibility rule: log exact HW + precision + parallelism per run; never
compare across hardware within a single claim.**

**Milestone contents re-mapped 2026-06-01** for the Evasion-Tax reframe (§3a): the new headline (cross-layer
comparison + **idealized evasion-tax frontier**) is now **M3**; the old trusted-reference ladder moves to
**M5**. The Phase-3 plan's M3–M5 *sketch* is superseded by the table below (its **M1–M2 component contracts
remain valid**).

| ID | Tier | Milestone | Target weeks | Exit gate (verifiable) |
|----|------|-----------|--------------|------------------------|
| **M0** | F | **Design lock** — D1–D7 (§6) + the **Evasion-Tax reframe (H6/§3a)** + this guidebook | W1–W2 | done; reframe author-converged (**supervisor sign-off pending**); coding allowed |
| **M1** | F | **Environment + viability gate + COMPUTE CONFIRMATION** — OpenVLA-7B on the granted HW; benign baseline; RoboGCG *targeted* redirect; **GCG micro-bench on the *actual* HW (resolves D4/D7)**; metric-(A) signal; **run the on-GPU timing micro-bench → select compute Branch (D8)** | W2–W4 | **GO/NO-GO (H1):** benign reproduced **＋** RoboGCG *targeted* redirect (not denial) **＋** benign-vs-attacked separation **surviving at the coarse operator-goal reference** **＋ compute Branch N/N−/F selected (D8)**. *Denial-only → reframe to task-deviation (understanding-doc §9); separation only at the clean-instruction ceiling → necessity weak, flag.* |
| **M2** | F | **Floor detection layer** — **L0** (perplexity) + **L2-oracle** (metric A) FP-calibrated; **build the L1 internal-probe arm** (activation-delta primary; attention ablation) | W4–W6 | **non-adaptive cross-layer TPR@{1%,5%} per-rollout FPR** (L0/L1/L2) on a **held-out** split + ROC/AUC + CIs + benign degradation (**H2**) |
| **M3** | **F** | **Oracle intrinsic action-space frontier (M-b) — H6-A** — idealized action-space attacker vs the **L2-oracle** → (ASR, evasion) Pareto + intrinsic-tax scalar; **non-adaptive** L0/L1/L2 evasion-cost ordering at matched FPR. *(Adaptive-GCG-vs-L1 is **conditional on the M1 on-GPU micro-bench, D7/#5**; it is a standalone white-box result, **not** fused with L2's oracle frontier into a cross-layer "tax" — that conflates attacker models, #1.)* | W6–W9 | **H6-A:** L2-oracle (ASR, evasion) frontier + intrinsic-tax scalar with CIs; **non-adaptive** cross-layer ordering; **no cross-layer deployable-tax claim here** (that is H6-D / M4) |
| **M4** | **N/N− — branch-selected (M1)** | **Deployable L2 (ONE of B/C, chosen at M2) + the H6-D cross-layer tax** — build **one** deployable behavioural detector (B **or** C, **train/test task-disjoint**); **realistic adaptive GCG-through-policy** vs **deployable L1 and deployable L2** at a **matched** query/compute budget (**one suite, fixed budget**, #8) → the *deployable-vs-deployable* tax | W8–W11 | deployable per-rollout TPR@FPR +CIs **and** the **H6-D** matched-attacker tax with **honest gap to the M3 oracle** (**H4 + H6-D**). *Branch N = full matrix; Branch N− = reduced (one route, 5 %-FPR primary). Fallback (Branch F): report M3 / H6-A + oracle-gap; mark H6-D **not run**.* |
| **M5** | **N — secondary/stretch** | **Reference-coarsening ladder + threat-generalization** — operator-goal rung (committed-secondary, compute-cheap) ＋ task-ID rung (stretch) ＋ **SABER** fluent attack (where L0 dies) ＋ physical/CoT note (discussion only). *Promotion of any stretch item is gated on deadline progress, not spare GPU.* | W9–W12 | ladder TPR@FPR per rung (**H3**); SABER arm = input-level dies but behavioural fires |
| **M6** | F | **Consolidation + ablations** — one-variable ablations (k, probe type, combine rule); freeze operating points; figures script-regenerable | W11–W12 | **RESULTS FREEZE** in write-once `results/`; every figure script-regenerable |
| **M7** | F | **Analysis + claims ledger** — every claim→evidence; report negatives | W12–W13 | §9 ledger complete; every claim → a result file |
| **M8** | F | **Write-up** — draft (overlaps M7); author rewrites generated prose; verify citations | W12–W14 | complete draft; **zero** `[CITATION NEEDED]` |
| **M9** | F | **Polish + submit** — reproducibility appendix; submission | W14–W15 | submitted early Sep 2026 |

**Critical path & protection.** M0→M1→M2→**M3** secure the **floor + the H6-A oracle intrinsic-frontier result
in every compute branch** — the dissertation is safe regardless of which branch the M1 micro-bench selects, but
note **M3 is an oracle / non-deployable analysis, not a runtime-defence comparison**. **M4 (Branch N/N−)** turns
this into the fair *deployable-vs-deployable* **H6-D tax** once the M1 timing micro-bench shows the matched-attacker
matrix affordable; if it does not (**Branch F**), M3 / H6-A + the honest oracle-gap still stand and the **title
falls back** to the oracle-frontier wording (§3a). **M5 is secondary/stretch** (lead the necessity argument with
the operator-goal rung). The only **hard kill** is the M1 GO/NO-GO (no signal at all); later gates *adapt scope*
(trim M5, scope M4 to Branch N−) but **never abandon M3** (the committed H6-A result).

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
| **H5** *(Branch N/N−; realistic-adaptive arm of H6)* | M4 | An attacker aware of the **deployable B/C** detector can reduce detection **but at a measurable cost** (lower ASR / higher perplexity / restricted targets) — the detector **raises the attacker's bar** even if not unbreakable. | adaptive attack (fixed query/compute budget) lowers detection *and* lowers ASR / raises cost — a quantified trade-off. | Adaptive attacker evades at **no** cost. | Trade-off exists → bonus security contribution. No cost → important **negative**. *Only if M4 done with slack; else not claimed.* | ⬜ untested (stretch) |
| **H6-A** *(reframe core — committed, M3; all branches)* | M3 | The **L2-oracle** intrinsic action-space frontier shows that the *embodiment* constraint imposes a measurable **intrinsic** evasion cost that **persists under an idealized action-space attacker (mechanism M-b)**, not merely an artifact of GCG failing to differentiate through the closed-loop rollout (**M-a**). *This is an **oracle** statement; it makes **no** deployable cross-layer claim.* | the idealized (ASR, evasion) Pareto frontier vs the L2-oracle shows an intrinsic tax > 0 with CIs; **non-adaptive** L0/L1/L2 ordering recorded at matched FPR. | tax ≤ 0, **or** it vanishes under the idealized attacker (it was only M-a). | **reportable either way:** intrinsic tax > 0 → embodiment constrains the attacker at the action concept; ≤ 0 → it does not. *(Cross-layer "which layer wins" → H6-D.)* | ⬜ untested |
| **H6-D** *(cross-layer tax — committed-if-affordable, M4 / Branch N–N−)* | M4 | At **matched benign FPR and a matched realistic attacker/budget**, the **deployable** L2 imposes a **higher adaptive-evasion cost** than the **deployable** L1 (and L0). | suppressing deployable-L2's TPR to deployable-L1's evaded level forces measurable ASR forfeit / more queries under the **same** GCG-through-policy budget; **ΔASR-at-fixed-evasion > 0** with CIs. | deployable-L2 evaded at **≤** deployable-L1's cost (tax ≤ 0). | **4 outcomes, all reportable:** (i) L2>L1 → *embodiment creates an evasion tax → place the boundary at the action layer* (headline). (ii) L2 also falls → *embodiment alone does not save VLA defences*. (iii) L1 strong → *VLA internal reps carry security-relevant injection signal* (**cite & differentiate Concept-Dictionary `2602.01834`, IGAR `2603.06001`**). (iv) both weak → *adaptive evaluation is mandatory for VLA defences*. | ⬜ untested (Branch N/N−) |

### 3a. Direction lock — the *Embodiment Evasion Tax* measurement frame (DECISION 2026-06-01)

> **Three independent passes converged** (deep-research synthesis + author refinement + Codex review).
> Recorded here as the operational decision; **flag for supervisor sign-off** before it becomes the
> dissertation headline. Replaces nothing in M0–M3; it is the **unifying measurement framing** over H4/H5.

**The reframe.** The research core is a **measurement**: *do behavioural action-layer monitors cost more to
adaptively bypass than internal-representation probes on instruction-injected VLA policies?* The monitor is an
**instrument** for measuring per-layer adaptive-evasion cost — **not** a claimed robust defence / "firewall".
This is the *claim-type* choice that survives the most likely bad outcome: a defence-efficacy headline dies to
an adaptive attacker (cf. "The Attacker Moves Second", Nasr et al. `arXiv:2510.09023` — 12 LLM defences bypassed
>90%); a measurement headline is **true whichever way the result falls** (see H6's four outcomes).

**Working title (LOCKED 2026-06-01 — pending supervisor sign-off).** *The Embodiment Evasion Tax: Measuring
Adaptive Evasion Costs of Runtime Defences for Vision-Language-Action Models.* Chosen over the question-form
variant for dissertation register; the subtitle ("Measuring … Costs") is **outcome-neutral** — "tax" is a
**coined framing term to be *measured***, and may be ≈0 (see H6's four outcomes), so the title presupposes no
result. One-line swap if preferred: *… Do Behavioural Monitors Cost More to Bypass than Internal Probes in VLA
Policies?*

**Branch-F fallback title (Codex review #2, 2026-06-02 — pre-registered; trigger updated 2026-06-16 for the CSB A5000 registered compute, which makes Branch F more likely).**
If the M1 on-GPU timing micro-bench selects **Branch F** (D8 — the deployable matched-attacker matrix is
unaffordable within the calendar), **H6-D is not delivered**, so the committed result is the oracle frontier
alone; the title falls back to
*"Measuring an Oracle Action-Space Frontier for VLA Injection Detection"* (or keeps the main title with the
subtitle scoped to H6-A), and the **cross-layer deployable tax is reported as explicitly unresolved**, not
implied. The "tax" headline is **M4's claim, not M3's** — never let M3 alone carry it.

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
  frontier. No GCG needed → compute-cheap. Isolates the embodiment-intrinsic tax from attack-access. *This is an
  **oracle** measurement (Codex #2 #2): the frontier **upper-bounds** the evasion cost any **deployable**
  goal-consistency detector can impose (defender best-case) and **lower-bounds** a **realistic** attacker's cost
  against the **same** oracle (attacker best-case) — it does **not** lower-bound cost against "any detector"
  (that direction is inverted; see §4b-II).*
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
  - **Coverage manifest (Codex review #2 #6 — pre-M2, load-bearing for M3 validity).** Metric (A) v1 covers
    only single-anchor reach/pick goals; **placement-region anchors, orientation-only deviations, and
    multi-phase goals are out of v1 scope** (schema §6). Before M2, emit a **task/target coverage manifest
    (supported / unsupported / abstained)** over the D4 matrix, and **constrain the idealized action-space
    attacker (§4b-II) to targets within supported coverage** — otherwise the attacker trivially evades through a
    metric blind spot and M3 measures *coverage gaps, not embodiment*. Uncovered cases are **pre-registered
    exclusions reported as a headline limitation**, never silently abstained.
- **(B) Learned action-semantics map** — small model: (observation, action_window) → goal embedding /
  descriptor; compare to trusted-goal embedding. *Deployable.* → **M4 (deployable L2; Branch N/N−).** *Default
  committed route; **exactly one** of B/C is built (Codex #2 #8 — choice locked at M2 from the observed (A) signal).*
- **(C) Reference-policy divergence** — a small benign policy predicts expected action distribution for the
  trusted goal; flag divergence. *Deployable; needs a second policy.* → **M4 (deployable L2; Branch N/N−), alt.**
  *Built **only if** chosen over B at M2 — not both.*
- **(D) VLM/LLM judge** — **avoid** (too close to Wu et al. `actalign`; heavy; undermines the
  lightweight/non-CoT differentiation). Only if B and C both fail and time remains.

---

## 4b. What to implement — the Evasion-Tax instruments (the new builds beyond the floor)

> The floor code (metric A, calibration, eval stats, baselines — local-prep Tasks 0–9) is **done and reused
> as-is**: metric A = the **L2 oracle**, perplexity = **L0**, the eval harness = the shared calibrated
> comparator. This section specifies the **three new instruments** the reframe needs. Interfaces +
> synthetic-fixture tests are buildable **now** (model-free, TDD per local-prep conventions); the parts needing
> OpenVLA activations or LIBERO rollouts are deferred behind thin interfaces to the granted GPU.

**(I) L1 internal-representation probe** — `src/evasion_tax/metric/probe_internal.py` (+ an `ActivationExtractor` seam).
- **Contract:** `InternalProbe.score(features) -> Score` (higher = more injected), calibrated through the
  **same `calibrate`** (invariant #4) at the matched per-rollout FPR.
- **Primary = activation-delta linear probe** (Task-Drift lineage, `2406.00799`): features = hidden-state delta
  across the injection point in OpenVLA's transformer trunk; a logistic-regression / linear probe trained on
  **benign-vs-injected** rollouts, **train/test task-disjoint** (report whether it merely memorises a task prior).
- **Ablation = attention-map MLP** (AlignSentinel lineage, `2602.13597`): attention-pattern features → small MLP.
  Pre-registered ablation; AlignSentinel is the **scoop-risk closest prior** → cite + differentiate (text-LLM vs
  our VLA/action setting); do **not** depend on exact reproduction.
- **VLA-native scoop risk (Codex review #2 #7 — verified 2026-06-02):** **Concept-Dictionary `2602.01834`**
  (inference-time activation-level VLA safety) and **IGAR `2603.06001`** (train-free attention recalibration for
  language-action coupling, LIBERO) occupy the activation/attention-level VLA-safety space. Cite + differentiate:
  they are **mitigations** vs our **measurement**; benign / jailbreak / OOD vs our **adaptive instruction-injection**;
  no per-rollout FP calibration / adaptive evasion-cost. Do **not** claim activation-level VLA safety as new.
- **Confound controls (Codex review #2 #11 — pre-registered; required before calling L1 "internal-rep" evidence):**
  held-out **tasks**, **suffix seeds**, and **target specs**; **label-shuffle** control (the probe must collapse to
  chance); **benign-weird-suffix** control (unusual-but-benign strings must **not** fire); **lexical/perplexity**
  control (the probe must beat an L0 lexical baseline on the *same* features) — so an L1 "win/lose" is not a task
  prior, a suffix lexical fingerprint, target leakage, or a GCG-family artifact.
- **`ActivationExtractor`** = thin seam over an OpenVLA forward pass returning hidden states / attention at the
  decision step; **synthetic fixtures** for local tests, real impl on the GPU.

**(II) Idealized action-space attacker (M-b — the intrinsic-tax instrument)** — `src/evasion_tax/attack/idealized_frontier.py`.
- **Contract:** given `(task, TargetActionSpec (D2), metric-A oracle scorer)`, search over **executable action
  sequences** (respecting reachability + the persistence window) to **maximise target-reach while minimising the
  metric-A consistency score** → trace the **(ASR, evasion = 1−detection) Pareto frontier**.
- **Detector-agnostic by design:** it attacks the *goal-consistency concept* via the privileged oracle. The
  resulting frontier is an **oracle** quantity with two precise (opposite-direction) readings, **stated as such
  and never as "a lower bound on cost against any detector"** (that is inverted, Codex #2 #2): (i) it
  **upper-bounds** the evasion cost any **deployable** goal-consistency behavioural detector can impose (the
  oracle is the strongest such detector → defender best-case; deployable B/C can only be evaded **more** cheaply),
  and (ii) it **lower-bounds** a **realistic** attacker's cost against the **same** metric-A oracle (the idealized
  attacker is the strongest → attacker best-case). This is what isolates **M-b (intrinsic)** from **M-a** (the
  "GCG can't differentiate through the rollout" artifact).
- The **optimiser + frontier logic are model-free** (locally testable on synthetic dynamics); evaluating
  reachability / privileged-state on real scenes is the GPU/LIBERO part.

**(III) Cross-layer evaluation + the "tax" metrics** — extend `src/evasion_tax/eval/`.
- Run **L0 / L1 / L2** through the **same calibration** and the **same attacks**; emit the comparative table +
  the Pareto-frontier overlay.
- **Operational tax metrics — ONE pre-registered primary scalar (Codex review #2 #10):** the **primary** tax
  measure is **ΔASR at a fixed evasion level**, **bootstrapped over task / target / seed (report CIs)**. Secondary
  (report **only when both frontiers share the same axes and the same attacker model** — i.e. the M4
  deployable-vs-deployable comparison, **never** idealized-L2 vs GCG-L1): (b) **attacker queries / compute** to a
  fixed evasion; (c) **area between Pareto frontiers**. These double as the **reusable VLA defence-evaluation
  protocol** (the free secondary contribution; AttackVLA is attack-only).
- **Roles fixed:** L2-oracle = metric A (privileged; M-b + non-adaptive ceiling). L2-deployable = metric B/C
  (M4, Branch N/N−). L1 = internal probe. L0 = perplexity. **Never present the oracle as deployable.**

**Build order:** L1 interface + idealized-attacker optimiser + cross-layer eval (all model-free, **now**) →
real `ActivationExtractor` + LIBERO reachability (M1/M2 on GPU) → **M1 on-GPU micro-bench of L1 extraction +
adaptive-GCG-against-the-probe (D7 / #5): if the adaptive-L1 attack is too costly at the measured budget it is
dropped to a later branch and the floor keeps non-adaptive L1 + the L2-oracle frontier** → oracle intrinsic
frontier (**M3 / H6-A**) → deployable B-or-C + realistic adaptive (**M4 / H6-D**, Branch N/N−).

---

## 5. Metrics & evaluation conventions (lock at M0; apply everywhere)

- **Detection is *causal*** — it scores a **prefix window** `a_{t-k+1:t}` ending at the current/candidate
  action (OpenVLA emits one action per step → no future window at decision time) and decides whether to
  execute `a_t`. A full-window, post-hoc pass is reported separately as a **monitoring ceiling** (non-causal,
  labelled).
- **Primary detection:** ROC/AUC; **per-rollout TPR @ fixed per-rollout benign false-abort rate** on a
  **held-out** split, **with Wilson / Clopper-Pearson CIs**. **Operating-point power rule (Codex review #2 #3):**
  **5% is the primary operating point**; **1% is reported as *exploratory* unless the held-out benign set is large
  enough to estimate it** — a 1% per-rollout FPR needs **≥ ~300 held-out benign rollouts** (with ~30–90 a 1%
  quantile is fiction and the Clopper-Pearson CI swamps it; rule-of-three: 0/90 only bounds FPR ≲ 3.3%). Benign
  rollouts are **cheap** (no GCG) → **pre-register a benign-N target per reported FPR claim at M1**; never report a
  1% point the benign N cannot support. *Per-window* rates are **auxiliary only** (1%/window compounds to ~63%
  false-abort over a 100-step rollout).
- **Detection latency** (steps of deviation before the hold fires) is a **first-class** metric — a single
  action is ambiguous, so latency > 0 is expected; quantify the target-actions executed before detection.
- **Cost:** **benign task-success degradation**; **per-rollout false-abort rate**; detector compute **latency**.
  *(Phase note: degradation / abort-rate / detection-latency need a detector **gating real rollouts** — attack
  onset, per-step fire steps, benign task-success — so they are computed at the **gated-rollout (GPU/sim)
  phase**, not from the model-free score-array harness. `run_condition_matrix` therefore marks them explicitly
  (`latency_summary.status="deferred"`) instead of emitting an all-None "never fired" stub; the
  `detection_latency` / `abort_rate` / `benign_degradation` implementations are the GPU-phase callees.)*
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
- **Hardware provenance (2026-06-16):** the registered hardware is **CSB `ecs3-0202` = RTX A5000 (24 GB)**;
  record the **exact card (which of the two A5000s) + count + precision (bf16) + CUDA/driver/torch** in every run
  log. **Never compare results across different hardware within a single claim** (HW is a variable — re-run the
  comparator on the same card). All L0/L1/L2 cross-layer comparisons and adaptive-cost frontiers must be produced
  on **one A5000**. Kelvin2 (A100/H100), if ever used, is **separately registered** — never merged with A5000 results.

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
| **D7** | Compute budget | GCG micro-benchmark on the **registered RTX A5000** first to fix the eval matrix; bound attack compute, concentrate on detector; **subsample to fit at M1**. Record actual s/target + peak VRAM + max candidate-batch at bf16. **Extended (Codex #2 #5): also micro-bench L1 activation/attention extraction overhead AND adaptive-GCG-against-the-probe-score** — if adaptive-L1 is too costly at the measured budget, drop it and keep non-adaptive L1 + the L2-oracle frontier. *(The published H100 ~185–604 s/target does **NOT** transfer — no A5000 OpenVLA-GCG prior exists → the micro-bench is the sole budget source; expect materially slower.)* | **OPEN** → M1 |
| **D8** | Compute | **Registered compute switched 2026-06-16 → CSB `ecs3-0202` = 2× RTX A5000 (24 GB each)** — the first accessible GPU (Kelvin2 login never established → **Kelvin2 demoted to backup**, partitions `k2-gpu-a100`/`k2-gpu-h100`, separate registration if ever used). A **single** A5000 is the registered card; OpenVLA-7B in **bf16** (~14 GB, fits with headroom). The **M1 on-GPU timing micro-bench on the A5000** computes the affordable matrix → selects **Branch N** (full deployable tax) / **N−** (scoped) / **F** (oracle frontier only) — see §2 *Compute branches*; bring-up = `docs/gpu/CSB/plan.md`, protocol = `docs/setup/gpu-runbook.md`. A5000 ≈2.5–4× slower than A100/H100 → branch expected **N−/F**. **Log exact card + precision + parallelism per run; no cross-HW comparison within a claim.** **RESOLVED (PROVISIONAL) 2026-06-23:** D8 micro-bench registered → **s/step=33.19 s**, **s/target(worst)=16,595 s ≈ 4.61 h** (sw=512/ns=500, early_stop OFF), max-B=43 @ 21.3 GiB (`results/2026-06-23T13-34-55Z-gcg-microbench/`). `branch_select` on the realistic calendar (author-exclusive box; **2 cards** GO; ~125 GPU-h/wk/card; ~5-wk M3+M4 window; ~70 % to the adaptive matrix → ~875 GPU-h; adaptive_mult=3 **EST**) → **provisional Branch N−** (62 adaptive runs / ~20 targets×3), **hard-F default**. **Locks only when the adaptive GCG-against-the-probe cost is measured (M1/M2).** | **RESOLVED (provisional N−, hard-F default) 2026-06-23**; locks at M1/M2 adaptive bench |

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

> Status: `[ ]` TODO · `[x]` DONE · in-progress/blocked = `[ ]` + inline `🔄 DOING` / `⛔ BLOCKED <reason>` (checkboxes are binary; these are not-yet-done sub-states). Keep granular; one line each; add a `verify:` where useful.
> This ledger is the **source of truth for task state** across sessions.

### M0 — Design lock
- [x] Resolve D1–D7 with author sign-off → recorded §6/§10 (2026-05-31; D4/D7 OPEN → M1).
- [x] Write Phase-3 implementation plan (`docs/plans/phase3-implementation-plan.md`) **+ revise per Codex review**. `verify:` author OKs before coding.
- [ ] 🔄 Define repo layout for code/configs/results — **pre-GPU local-prep plan written** (`local-prep-plan.md`); model-free scaffolding via TDD underway (Task 0+).

### M1 — Environment + viability gate
- [ ] Stand up OpenVLA-7B (bf16) on the **registered CSB A5000** (one card); record exact env (CUDA 13.2 driver / cu121 torch / flash-attn) + provenance (checkpoint source/hash/date/licence) in `docs/references/`.
- [ ] Reproduce **benign** LIBERO baseline success (pinned seeds). `verify:` numbers logged to write-once `results/`.
- [ ] Reproduce **RoboGCG** on a few tasks; **confirm targeted redirect** (not denial). Quarantine suffixes in `artifacts/untrusted/`.
- [ ] **GCG micro-benchmark** on the **registered A5000** (s/target + peak VRAM + max candidate-batch at 24 GB; published H100 timings do NOT transfer) → resolve D4 + D7. **＋ (Codex #2 #5) micro-bench L1 extraction + adaptive-GCG-against-probe → decide whether the adaptive-L1 arm is in scope at the measured budget.** *(D8 timing micro-bench DONE 2026-06-23: s/step=33.19 s, s/target(worst)=16,595 s; `results/2026-06-23T13-34-55Z-gcg-microbench/`. L1-extraction + adaptive-probe cost still deferred → measured when L1 is on GPU.)*
- [ ] **A5000 early-stop steps-to-success bench** — **🔀 (i)→(ii) MERGED 2026-06-24 (DM-1):** the *standalone registered* bench is **cancelled** (its fit-check is banked by the 2026-06-24 dry run; its dummy-image cost is a proxy for the real attack) → its cost machinery (`early_stop_bench.*`, `reached_fn`) is now **reused inside the (ii) RoboGCG redirect attack** (M1-plan Task 5), so steps-to-success / early-stop / time fall out of the *representative* attack. `bench_early_stop.py` is **kept**. *Original rationale, retained:* (D8 follow-up; **author-agreed 2026-06-23 — run BEFORE the M3/M4 attack matrices**). The registered D8 number is the **early_stop-OFF worst case** (500 steps always → 4.61 h/target); re-run the GCG attack with **early_stop ON** over a small pinned target set (RoboGCG sw=512, pinned seeds) and record the **steps-to-success distribution** (median / IQR / 500-step-censored fraction). This is the **dominant lever on the affordable matrix** — a ~60-step median turns 4.61 h → ~0.55 h/target and shifts the matrix ~8× (provisional N− → N). `verify:` distribution logged to write-once `results/`; the realistic median `s/target` re-feeds `branch_select` (replacing the worst-case) → updates the **provisional** branch (still **hard-F default** until the *adaptive* GCG-against-the-probe cost lands). Run unattended (tmux + `systemd-inhibit`, per-target checkpoint + auto-restart; `docs/gpu/CSB/plan.md`).
- [x] **D-3 / DM-3 SchemaA radius re-pin rule LOCKED (2026-06-24, mac, TDD)** — benign-only, pre-registered (`docs/core/d3-radius-repin-preregistration.md` PROPOSED→LOCKED; author sign-off via the M1-plan, supervisor PENDING). Executable form `evasion_tax.eval.schema_repin.repin_schema_from_benign` (pure §3 estimators + §4 guards; 6 TDD tests, ruff+pyright clean). **The M1 attacked run (Task 5) consumes the FROZEN re-pinned schema via `--schema-from`; attacked data can never re-pin (circularity guard, invariant #2).** The benign baseline (Task 4) produces the `geometry_stats` input; the radii are fixed once on the benign split and recorded as a dated deviation in `metric-a-annotation-schema.md` §5 **before** any attacked output is scored.
- [ ] Metric (A) signal sanity-check: separation that **survives at the coarse operator-goal reference** (not the clean-instruction ceiling alone). → **GO/NO-GO gate (H1)**.

### M2 — Floor detector (A) + FP-calibration
- [x] Implement metric (A) on a **causal prefix window**: `s(observation_t, action_{t-k+1:t}, trusted_goal, [state_t])`; **annotation schema FROZEN before attack inspection** (`docs/core/metric-a-annotation-schema.md`, `2c2f163`); unit-tested in isolation. *(model-free scorer done locally; per-task fixtures on real LIBERO ground truth + the on-GPU benign-vs-attacked signal check remain M1.)*
- [ ] Build calibrated detector: τ on calibration split → target **per-rollout** false-abort; fire → hold before executing `a_t`.
- [ ] Evaluate: ROC/AUC, **per-rollout** TPR@{1%,5%} false-abort (**+CIs**) on held-out split, benign degradation, **detection latency**. → **H2**, **FLOOR SECURED**.
- [x] Baselines under the **same calibration** (model-free, `3287c5c`): **mandatory goal-agnostic anomaly** (χ²-OOD on the action stream, goal-blind + causal) **+ perplexity/text-only filter** (`MockPerplexityScorer` + GPU stub `RealPerplexityScorer`; monotone ppl→score keeps calibration order-equivalent to raw-perplexity thresholding). *(Real LM-perplexity backend + actual benign-vs-attacked numbers remain on the GPU node.)*
- [ ] 🔄 **(NEW, model-free — build now, §4b)** Interfaces + synthetic-fixture tests so GPU day-1 is "plug in activations + rollouts"; all calibrate through the *same* `calibrate`, activation/LIBERO impls stubbed for the GPU node:
  - [x] **L1 internal probe (§4b-I)** — `InternalProbe` + `ActivationExtractor` seam (`src/evasion_tax/metric/probe_internal.py`, `+ probe_confounds.py`; `fadc008`).
  - [x] **Idealized action-space attacker (§4b-II)** — `src/evasion_tax/attack/`: `dynamics.py` (`Dynamics` seam, `SyntheticDynamics` integrator + `RealDynamics` GPU stub), `frontier.py` (Pareto geometry + `asr_at_evasion` single-frontier readout), `idealized_frontier.py` (constant-action random-shooting optimiser + `trace_frontier`, coverage gate). Constant-action MVP; CEM = pre-registered stretch. 46 model-free tests; ruff + pyright clean.
  - [x] **Cross-layer eval + tax metrics (§4b-III)** — `src/evasion_tax/eval/cross_layer.py`: `UnitOutcome` per-(unit,layer,tradeoff) contract; `frontier_from_outcomes` (reproduces `trace_frontier`'s aggregate); **ΔASR@fixed-evasion primary scalar** (`delta_asr_at_evasion`) + **seeded cluster bootstrap CI over the (task,target,seed) `UnitKey`** (`bootstrap_delta_asr`, #10); `comparative_asr_table` (L0/L1/L2 ordering) + `frontiers_by_layer` (overlay); model-free `collect_oracle_outcomes` data path. Secondary metrics (b) queries-to-evasion + (c) area-between-frontiers **deferred to M4** (same-attacker-only). 13 model-free tests; ruff + pyright clean.
- [x] **(Codex #2 #6) Coverage manifest** — `src/evasion_tax/metric/coverage.py`: `GoalKind`/`CoverageStatus`/`CoverageCell`; `classify_cell` (unsupported kind → ABSTAINED on unresolvable anchor → SUPPORTED), `build_manifest` classifies **every** D4 cell (no silent drop) + rejects duplicates; `.supported()/.unsupported()/.abstained()/.summary()/.limitation_report()/.assert_covers()`; `.predicate_for_target()` drops into `trace_frontier(supported=…)` to **constrain the attacker to supported targets**; uncovered cases pre-registered as a headline limitation. 18 tests; ruff + pyright clean.
- [x] **(Codex #2 #3) Power / sample-size rule** — `src/evasion_tax/eval/power.py`: `required_benign_n = ceil(min_events/p)` rule-of-three floor (**300 @ 1%, 60 @ 5%**); `classify_power`/`annotate_operating_points` flag each operating point powered/primary so an underpowered tight point can't be silently reported as headline; pinned `DetectorConfig.primary_fpr` (default 5%, validated ∈ `fpr_targets`). 15 tests; ruff + pyright clean.

### M3 — Idealized Evasion-Tax frontier  *(F — committed headline)*
- [ ] **On GPU:** extract OpenVLA activations during benign+attacked rollouts; train + calibrate the **L1 probe** (activation-delta primary; attention-map ablation), **task-disjoint**, **with the #11 confound controls (label-shuffle, benign-weird-suffix, lexical/perplexity, held-out suffix-seeds / target-specs)**. → completes **H2** cross-layer (**non-adaptive**) + feeds **H6-A**.
- [ ] Run the **idealized action-space attacker** vs the metric-A oracle (**targets within the #6 coverage manifest**) → **(ASR, evasion) Pareto frontier**; compute the **intrinsic-tax scalar** (primary = ΔASR @ fixed-evasion, bootstrapped, #10) + the **non-adaptive** L0/L1/L2 ordering, +CIs. → **H6-A** (oracle intrinsic; **no cross-layer deployable-tax claim here**). *(Instruments built + unit-tested model-free — `collect_oracle_outcomes` (returns `(outcomes, excluded)`, surfacing coverage-excluded), `frontier_from_outcomes`, `bootstrap_delta_asr`, `comparative_asr_table`/`frontiers_by_layer`, `target_action_blocked_rate`. **What remains is the driver** over **real LIBERO scenarios** — RealDynamics privileged state = GPU — that writes the frontier/tax table to write-once `results/` + the frontier-overlay figure (`figures.py` ladder placeholder until then). A synthetic-dynamics run validates the **instrument only**, never H6-A; do not label it H6-A.)*

### M4 — Deployable L2 (B/C) + realistic adaptive  *(N/N− — branch-selected at M1)*
- [ ] Build **exactly one** deployable behavioural detector (**B** learned action-semantics map **or** **C** reference-policy — **choice locked at M2** from the (A) signal, #8); **specify supervision labels, negative-pair construction, train/test task-disjoint** (no task-prior leakage). → **H4** (gap to the A oracle).
- [ ] **Realistic adaptive GCG-through-policy** vs **deployable L1 and deployable L2** at a **matched** query/compute budget (**one suite, fixed budget**, #8) → the *deployable-vs-deployable* tax. → **H6-D**. *Fallback (Branch F): skip; report M3 / H6-A + oracle-gap, mark H6-D **not run**.*

### M5 — Reference ladder + threat-generalization  *(N — secondary/stretch)*
- [ ] Operator-goal rung (committed-secondary, compute-cheap) + task-ID rung (stretch); sweep detection+FPR across rungs → ladder table. → **H3**.
- [ ] (stretch) **SABER** fluent-injection arm — **confirm OpenVLA inclusion first** (`docs/references/README.md`); show L0 (perplexity) dies but L2 fires. (Physical/CoT generalization = discussion only.)

### M6–M9 — Consolidate / analyse / write / submit
- [ ] One-variable ablations; freeze results; figure-regen scripts (**RESULTS FREEZE**).
- [ ] Claims ledger (§9); report negatives.
- [ ] Draft → author rewrite → verify citations (no `[CITATION NEEDED]`).
- [ ] Reproducibility appendix; submit.

---

## 8. Experiment protocol (use for every run — paste into the run's log)

```
run_id:        <UTC timestamp>-<short-slug>
git_commit:    <hash>
hardware:      RTX A5000 (CSB ecs3-0202, GPU 0|1) — record exact card + count + precision (bf16) + CUDA/driver/torch (NEVER compare across HW within one claim; Kelvin2 A100/H100 = backup, separate registration)
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
| *(e.g.)* Detection degrades to baseline once reference is coarser than the operational instruction | M5 ladder | `results/…` | ⬜ |
| *(e.g.)* Deployable (B) recovers __% of the (A) ceiling | M4 run __ | `results/…` | ⬜ |
| *(e.g.)* Adaptive attacker (deployable L2) evades only at __ ASR cost | M4 run __ (H5 / H6-D) | `results/…` | ⬜ |

---

## 10. Decision log (append-only)

| Date | Decision | Rationale | Changed |
|------|----------|-----------|---------|
| 2026-05-30 | Theme = **Embodiment Evasion Tax** | deliverable-first risk appetite + language→action salience (understanding-doc §2) | scope locked |
| 2026-05-31 | RoboGCG defence claims **verified**; defence problem is **open** | full text §5 + Table 3 (`docs/references/`) | motivation for the project confirmed; perplexity = baseline-to-beat |
| 2026-05-31 | **Timeline** ~14–16 wk, submit early Sep 2026; **scope** core-first **but novelty (B/C + adaptive) is the committed goal** | author steer | §2 milestones / tiers set |
| 2026-05-31 | **D1** metric: (A) floor + (B) primary deployable / (C) complementary, (D) excluded; B/C emphasis → M2 | most directly instantiates goal-action consistency; avoid actalign overlap & stay lightweight | M2/M4 scope |
| 2026-05-31 | **D2** attack: RoboGCG-faithful primary, window-scored; **semantic-redirect arm gated on M1** | minimise attack-side risk; pre-register denial-vs-targeted so either outcome is reportable | M1 gate rule |
| 2026-05-31 | **D3** rungs: clean(ceiling)+coarse-goal(must)+task-ID(committed); **D5/D6** baselines & metrics adopted (+optional anomaly baseline) | necessity critique → lead with coarse rungs; baselines/metrics standard & defensible | M2/M3 scope locked |
| 2026-05-31 | **D4/D7** kept **OPEN**, rules pre-registered → M1 micro-benchmark | cannot size eval / budget before measuring GCG on GB10 | M1 |
| 2026-05-31 | Phase-3 implementation plan drafted | M0 deliverable; lifts coding gate for M1–M2 | `docs/plans/phase3-implementation-plan.md` |
| 2026-05-31 | **M5 (adaptive) → stretch**; M4 deployable detector = committed primary novelty | Codex review: both-committed = over-scope for solo MSc; author chose M5-stretch | §0,§2 tiers, §3 H5, §7 |
| 2026-05-31 | **Codex review incorporated** | third-party review verified vs RoboGCG/actalign primary sources | causal prefix-window + latency; per-rollout FPR + CIs; metric(A) schema frozen; fair-calibrated baselines + mandatory anomaly; "target-action-blocked"; M1 coarse-goal check; precise "adversarial textual suffix" framing — §0,§1,§2,§3,§5,§6,§7 + impl plan |
| 2026-05-31 | **Adjacent prior work verified** (4 arXiv IDs all correct): Task Drift, Instruction Hierarchy, AlignSentinel (text-LLM), SABER (VLA attack) | none scoop the embodied/action-level contribution | **novelty narrowed to the VLA action-level instantiation**; SABER = candidate secondary attack arm (understanding-doc §6) |
| 2026-05-31 | **Author OK → start M1–M2 scaffolding code** (model-free, M1/8 GB); pre-GB10 local-prep plan written | gate-lift precondition met (plan agreed in `docs/plans/` + author OK); OpenVLA inference infeasible locally (8 GB RAM) → build+test only model-free components; experimental *runs* await GB10 | `docs/core/local-prep-plan.md`; coding begins |
| 2026-05-31 | **Metric (A) annotation schema FROZEN** (load-bearing Task 5); design **delegated by author to Claude** with "adopt the realistic option; pre-register value-adding variants as stretch" | freeze must precede any attack output (circularity guard, invariant #2); decisions: privileged `target_region` anchor via resolver seam; primitives P1 progress / P2 distractor / P3 grasp; combine=`max` (zero params, robust to inter-primitive correlation); `{noisy_or, weighted_mean}` + `k`/`r` sweeps = pre-registered ablations; stretch S1 orientation, S2 multi-phase sub-goal = definitions frozen now, implemented later | `docs/core/metric-a-annotation-schema.md`, commit `2c2f163`; §7 M2 first item ✅ |
| 2026-06-01 | **Headline reframed → Embodiment Evasion Tax** (H6/§3a); monitor = *instrument* measuring per-layer adaptive-evasion cost (L0/L1/L2), **not** a 'firewall'/defence claim; **milestone contents re-mapped** (idealized frontier → M3, ladder → M5); title **LOCKED** (pending supervisor) | 3 independent passes converged (deep-research + author + Codex); measurement framing survives the likely bad outcome (Attacker-Moves-Second `2510.09023`: efficacy claims die to adaptive attackers) | §0,§2,§3,§3a,§4b,§7,§12 |
| 2026-06-01 | **Citation pass DONE** — 5 flagged items resolved + **16 cited PDFs** downloaded/SHA-pinned/provenance | reframe must rest on a verified landscape (integrity rule); net: **nothing scoops** the runtime/FP-calibrated/adaptive lane (VLA defences found = training-time; actalign benign-only; AttackVLA attack-only) — **⚠️ this "nothing scoops" conclusion was superseded 2026-06-02 (next row): a 2026 runtime VLA-safety cluster exists; novelty narrowed** | `docs/references/README.md` |
| 2026-06-01 | **D8 compute tier OPEN** — A100/H100 cluster requested (pending); roadmap **compute-tiered** (Tier-F GB10-guaranteed / Tier-N committed-if-cluster) | author: cluster access likely → de-risks deployable B/C + realistic adaptive + full ladder; unconfirmed → floor stays compute-agnostic | §0,§2,§6 (D8),§8,§12; M1 compute-confirmation checkpoint |
| 2026-06-02 | **Codex review #2 incorporated** (12 findings; **all accepted with refinements, author OK to apply**) | third-party review; the 4 "omitted scoop" papers (`2605.22446` Pre-VLA, `2604.12447` HazardArena, `2602.01834` Concept-Dictionary, `2603.06001` IGAR) **independently re-verified on arXiv 2026-06-02** before acting on them (integrity rule — do not weaken a claim on unverified citations) | **#1/#4 claim boundary: split H6 → H6-A (M3, oracle intrinsic frontier, *no* cross-layer tax) + H6-D (M4, deployable-vs-deployable matched-attacker tax) + no-cluster fallback title**; #2 oracle-bound wording fixed (upper-bounds deployable-detector cost / lower-bounds realistic-attacker cost vs the *same* oracle — **not** "any detector"); #3 power rule (5% primary, 1% needs benign N ≥ ~300); #5 D7 extended to L1 extraction + adaptive-GCG-against-probe (Tier-F no longer assumed compute-agnostic); #6 metric-(A) coverage manifest pre-M2; #7 novelty **narrowed** (runtime VLA-safety lane now occupied) + 4 papers logged; #8 M4 → one deployable route + one-suite adaptive; #10 primary tax scalar = ΔASR@fixed-evasion; #11 L1 confound controls; #12 `:137` tag → `2510.09023`; #9 stale M3/M5/M4 labels reconciled. §0,§2,§3,§3a,§4,§4b,§5,§6,§7,§9,§12 + refs README + phase-3 plan banner |
| 2026-06-02 | **Eval-harness held-out-FPR correctness fix** (invariant #3) | the operating point's `realised_fpr` (+CI) was being measured on the **calibration** split τ was set on (in-sample, conservative-by-construction), while `benign_test` fed only ROC/AUC — so the held-out benign false-abort rate, the number invariant #3 mandates and on which the M2 floor/H2 claim rests, was never computed. TDD: 6 failing tests first (incl. a harness test showing `realised_fpr=0.0` in-sample vs `1.0` held-out under a shifted benign split), then minimal fix. 237 tests green, ruff clean. | `tpr_at_fpr(benign_calib, attacked, *, benign_eval_scores=…)` reports `realised_fpr`/CI/`n_benign` on the **held-out** split; in-sample retained as `calib_fpr`/`calib_fpr_ci`/`n_benign_calib` (diagnostic). `run_condition_matrix` passes `benign_eval_scores=benign_test`. Backward-compatible (no eval set → falls back to calib, equals `calib_fpr`). `src/evasion_tax/eval/{metrics,harness}.py` + tests |
| 2026-06-02 | **GPU upgraded GB10 → A100/H100** (author; single-card-vs-cluster + queue depth TBC) → **D8 re-cast: compute *tiers* → three pre-registered *branches* (N / N− / F) selected by the M1 on-GPU timing micro-bench**; precision **4-bit → bf16**; stretch (M5 ladder / SABER) promotion gated on **deadline progress**, not spare GPU | the old Tier-F/Tier-N gate was binary on cluster existence; the relaxed compute makes H6-D plausibly committable, but the author chose a **data-driven branch keyed on measured GCG/L1 cost** rather than a hard commit now (Q2), and held scope to **fidelity + power** (bf16; benign N ≥ ~300 for the 1 %-FPR primary; more seeds) with no new attack families (Q3); H6-A floor unchanged (delivered in every branch) | §0,§1,§2 *Compute branches*,§3 (H5/H6-A/H6-D),§3a fallback,§5,§6 (D7/D8),§7,§8 + CLAUDE.md theme block + code (`gb10_*`→`gpu_*` guard, `quantization: bf16`) + understanding/phase3/local-prep/metric-a docs + 2 historical scoping-doc banners |
| 2026-06-03 | **Model-free §4b-III cross-layer eval + the two remaining Codex-#2 hooks (#3 power, #6 coverage) built** (TDD, plan `docs/plans/codex-hooks-and-cross-layer-eval-plan.md`) — completes the pre-GPU model-free track | these are the last instruments H6-A/M3 needs before GPU day, and their inputs (`attack/frontier.py:asr_at_evasion`, the L1 probe, `trace_frontier`) already existed; pure stats/schema-scope only (no attack output read → invariant #2 untouched). YAGNI: the **#10 primary scalar** (ΔASR@fixed-evasion + cluster bootstrap CI over the (task,target,seed) `UnitKey`) is built; the **secondary** tax metrics (b) queries-to-evasion + (c) area-between-frontiers are same-attacker-only → **deferred to M4** | `src/evasion_tax/eval/cross_layer.py` (§4b-III: `UnitOutcome`/`frontier_from_outcomes`/`delta_asr_at_evasion`/`bootstrap_delta_asr`/`comparative_asr_table`/`collect_oracle_outcomes`), `src/evasion_tax/metric/coverage.py` (#6 manifest + `predicate_for_target` seam into `trace_frontier`), `src/evasion_tax/eval/power.py` + `DetectorConfig.primary_fpr` (#3 rule-of-three floor) + `configs/example_m2.yaml` + §1/§7 ledger. **362 tests green** (+50), my files ruff + pyright clean (the 3 pyright + 1 B905 in untouched test files remain pre-existing) |
| 2026-06-03 | **Granted GPU identified = Kelvin2** (NI-HPC @ QUB; partitions `k2-gpu-a100` = 4 nodes × 4×A100-80GB, `k2-gpu-h100` = 1 node × 4×H100-80GB; **3-day walltime cap**; shared → queue depth a real variable). **Local-prep Task 11 (GPU runbook) DONE** | the abstract "A100/H100" of D8 is now a concrete shared cluster → day-1 must be turnkey + reproducibility-pinned before the (queued, time-capped) GPU window opens. Cluster facts summarised from NI-HPC docs; env pins **fetched from OpenVLA/RoboGCG source** (invariant #8), all marked `[VERIFY ON THE GPU NODE]` (not invented) | `docs/gpu/{Overview,Connection,Start,Running}.md` (cluster mechanics, `4df155e`); `docs/setup/gpu-runbook.md` (M1 day-1 → GO/NO-GO sequence, mirrors §6/§8/H1); `configs/env/requirements-gpu.txt` (OpenVLA pins: torch 2.2.0 / transformers 4.40.1 / flash-attn 2.5.5 / py3.10.13); `docs/references/README.md` checkpoint provenance placeholder rows; D8/§2 now name the Kelvin2 partitions — `a491a63`/`9c3eaff`. Remaining local = only Task 10 |
| 2026-06-03 | **Lint debt cleared** — the 3 pyright errors + 1 ruff B905 long carried in test files are fixed; full `src/evasion_tax` + test tree now type-clean & ruff-clean (362 tests green) | housekeeping before GPU phase; one was a genuine (minor) source defect not a test problem | `records.py` `reached_window` actions type widened to `… | np.ndarray` (matches its `(n,7)`-array docstring); `test_state.py` two intentional-bad-input lines `# type: ignore[arg-type]`d; `test_consistency_a.py` `zip(strict=True)` — `2963e72` |
| 2026-06-23 | **CSB step 6 GREEN + D8 registered (true-batch official) → provisional Branch N−, hard-F default** (step-6 Task 4/5; plan `docs/plans/2026-06-19-step6-truebatch-lossof.md`) | the true-batch `loss_of` is the downstream M1–M4 harness, so D8 must size *that* (DB-1/DB-2); registered numbers now land. **Measured (`results/2026-06-23T13-34-55Z-gcg-microbench/`, A5000, bf16, flash_attn2, exclusive):** s/step=33.19 s; **s/target(worst)=16,595 s ≈ 4.61 h** (sw=512/ns=500, **early_stop OFF** → conservative upper bound); loop ablation 17.48 s → **speedup_k≈1.53**; **max-B=43 @ 21.3 GiB** (VRAM ceiling, **not** branch-critical, DB-3). **Calendar re-derivation (this is the Task-5 input the author asked to reconsider):** the box is **author-exclusive** (no contention) → realised GPU-h is limited by job-persistence + sleep + OOM, not sharing; author confirmed **2 cards usable** (same SKU → independent workers, log card per run, no cross-HW timing claim) and **early-stop bench to run first**. Planning calendar = 2 cards × ~125 GPU-h/wk/card × ~5-wk (slippage-adjusted; M1/M2 not yet done) × ~70 % to the adaptive matrix ≈ **875 GPU-h**. `branch_select(s_per_target=16,595, adaptive_mult=3 EST, seeds=3, overhead=300; thresholds N≥90 / N−≥30 runs)` → n_attacks=62 → ~20 targets×3 → **provisional N−**. **Sensitivity:** 1-card worst-case → **F** (31 runs, boundary-demoted); any of {early-stop holds, 2 cards} → up to **N** (≈500 runs). So the branch hinges on the **unmeasured A5000 early-stop steps-to-success** + the 2nd card → **hard-F default** stands until the *adaptive* GCG-against-the-probe cost is measured (M1/M2); **M3/H6-A delivered in every branch**. **Unattended-run runbook** (tmux + `systemd-inhibit`, no sudo; per-target checkpoint + auto-restart) added to `docs/gpu/CSB/plan.md` so the calendar is realisable. | §1 status, §2 *Compute branches*, §6 D8 (RESOLVED-provisional), §10; `docs/gpu/CSB/plan.md` (step 6 ✓ + unattended runbook); `src/evasion_tax/eval/branch_select.py` (reused). Loop→true-batch supersedes parent D6-2/D6-4 only (`docs/plans/2026-06-19-step6-gcg-microbench.md` back-pointer) |
| 2026-06-03 | **Local-prep Task 10 done; concrete LIBERO `StateAdapter` deferred to the GPU node** (→ **all local-prep Tasks 0–11 complete**; only remaining gate = Kelvin2 GPU access) | a state-only **Tier R (robosuite `Lift`/`Panda`, no render, no policy)** env reset on the M1 and the Task-4 `PrivilegedState` constructed from **real MuJoCo ground truth unmodified** → schema sound, Task-5 freeze stands. Real LIBERO blocked locally on a **non-representative** stack (`benchmark/__init__` hard-imports torch; `OffScreenRenderEnv` needs a GL context; LIBERO pins robosuite 1.4 vs local 1.5.2; local Py 3.11 ≠ GPU-node Py 3.10) — the GPU node installs the pinned stack, so LIBERO-specific validation + the adapter belong there. **No `state_libero.py` fabricated** (conditional on a real LIBERO success, invariant #8); synthetic fixtures kept as the metric-(A) test contract | tiered+graceful `scripts/libero_state_smoketest.py` (LIBERO→robosuite→clean-skip) + `docs/setup/libero-local-notes.md` (real obs schema, blockers, GPU-node reproduction) — `577c2d1` |
| 2026-06-09 | **State-only LIBERO runs locally → concrete `state_libero.py` PULLED FORWARD** (overturns the 2026-06-03 "deferred to GPU" row above) | the 2026-06-03 blockers were **env-assembly, not fundamental**: bypass `benchmark` (torch) by locating the BDDL from the package dir, bypass `OffScreenRenderEnv` (GL) via lower-level `ControlEnv(use_camera_obs=False, has_offscreen_renderer=False)`; the one real requirement = **robosuite 1.4.0** (LIBERO imports `…single_arm_env.SingleArmEnv`, removed in 1.5) → dedicated Py3.10 venv. Building the adapter against **real** LIBERO obs caught a latent bug the synthetic/robosuite path hid: naive `*_pos` extraction ingests `<obj>_to_robot0_eef_pos` **relative deltas** as phantom objects (corrupts P2). `target_region` = BDDL `obj_of_interest[-1]` (the demo's faked "cube" replaced). TDD against **frozen real-obs fixtures** keeps the core `.venv` suite LIBERO-free. **Local adapter is a draft re-validated on the GPU node** (repro rule); still GPU-only = production pinned stack + all-suite re-validation + policy rollouts | `src/evasion_tax/metric/state_libero.py` (`LiberoStateAdapter` + pure extractors, `_to_` relative-key filter, gripper `sum|qpos|>0.04`) + `tests/.../test_state_libero.py` (14 tests; resolver resolves `plate_1`, abstains on pose-less drawer region) + fixtures `libero_obs_{spatial0,goal_opendrawer}.json` + `PROVENANCE.md`; Tier-L smoke rewritten to the ControlEnv path (now PASSES); `docs/setup/libero-local-env.md` recipe + `libero-local-notes.md` UPDATE. **395 tests green, ruff clean.** Plan `docs/plans/2026-06-09-libero-state-adapter.md` |
| 2026-06-16 | **Registered compute SWITCHED → CSB `ecs3-0202` = 2× RTX A5000 (24 GB each); Kelvin2 → backup** (author decision). The long-standing **GPU-access blocker is resolved** — the box that the 2026-06-15 `pc-spec.md` recorded as an 8 GB RTX-4060 Windows smoke-rig is, on the actually-accessible machine, **2× A5000 / 24 GB / native Linux / CUDA 13.2** (verified `nvidia-smi`) | Kelvin2 login was never established; this is the first GPU we can use, **and** 24 GB clears the registered floor: bf16 OpenVLA-7B (~14 GB) fits one card with rollout headroom → the old "4-bit only / no science numbers" walls are gone → it can host the registered runs, not just smoke. Caveats kept honest: A5000 ≈**2.5–4× slower** than A100/H100 + **no published OpenVLA-GCG prior** → M1 micro-bench is the sole budget source, D8 branch expected **N−/F** (measured); shared desktop → secure a quiet window for timing; **all registered runs commit to the A5000** (cross-HW mixing forbidden → Kelvin2 = separate registration if ever used). H6-A/M3 floor unchanged (every branch). | §0 (claim boundary + compute), §1 (You-Are-Here / NEXT ACTION / Blockers), §2 *Compute branches*, §3a Branch-F trigger, §5 HW provenance, §6 (D7/D8), §7 (M1 tasks), §8 run-log template, §12 hardware; `docs/gpu/CSB/{pc-spec,plan}.md` rewritten, `docs/gpu/Overview.md` + `docs/setup/gpu-runbook.md` banner + `docs/core/goal-action-consistency-detector.md` §8 + CLAUDE.md theme block |
| 2026-06-24 | **(i)→(ii) MERGE** — the standalone registered early-stop steps-to-success bench is **cancelled**; its cost machinery is reused inside the (ii) RoboGCG redirect attack (M1-plan **DM-1**) | the (i) fit-check is already banked by the 2026-06-24 dry run and its dummy-image cost is a proxy for the real attack; folding `early_stop_bench.*` / `reached_fn` into the *representative* attack makes steps-to-success / early-stop / time fall out of **that** run (one registered GPU run, not two). `bench_early_stop.py` and its modules are **kept**, not deleted | §1 You-Are-Here, §7 M1 (item (i) annotated); `docs/plans/2026-06-24-m1-viability-gate.md` |
| 2026-06-24 | **D-3 / DM-3 SchemaA radius re-pin rule LOCKED** (author sign-off via the M1-plan; benign-only, pre-step-6) | the metric's geometric radii must be fixed from **benign geometry only, before any attacked output** (invariant #2 / schema §0); locking the **drafted** §3/§4 defaults now (m=1.2; `r*=1.2·median(A)`, `R_g*=1.2·P90(G)`; guards `r*<P10(D)`/`R_g*<P10(Dg)`; abort-if-n<5; round-to-5 mm; default=no-change) is the legitimate "set the knobs up front" window — **no attacked data exists yet** | `d3-radius-repin-preregistration.md` PROPOSED→LOCKED (§7 author box ✓, supervisor PENDING); executable form `evasion_tax.eval.schema_repin.repin_schema_from_benign` (pure, **6 TDD tests**, ruff+pyright clean); `metric-a-annotation-schema.md` §5 pointer; §1/§7 M1 |

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
  point → the project's opening.**
- **Nearest prior art (novelty constraint):** Wu et al. `actalign` (`arXiv:2510.16281`) — reasoning↔action
  consistency, but **benign/OOD only, no attacker, no FP calibration, needs CoT+VLM**. the Embodiment Evasion Tax differs by:
  attacker-aware + FP-calibrated + lightweight non-CoT.
- **Adjacent text-LLM prior (verified 2026-05-31 — see understanding-doc §6):** Task Drift `2406.00799`,
  Instruction Hierarchy `2404.13208`, AlignSentinel `2602.13597` (closest; FP-aware injection detector, **text
  LLM**). ⇒ **novelty = the *embodied / VLA action-level* instantiation only** (do not claim FP-aware injection
  detection as new in general). **SABER `2603.24935`** = a real NL injection **attack** on VLA/LIBERO →
  candidate secondary attack arm (perplexity-baseline-defeating).
- **2026 runtime / inference-time VLA-safety cluster (verified on arXiv 2026-06-02 — Codex review #2 #7;
  narrows the novelty claim):** Pre-VLA `2605.22446` (runtime action-validity verification), HazardArena
  `2604.12447` (training-free Safety Option Layer), Concept-Dictionary `2602.01834` (activation-level
  inference-time safety), IGAR `2603.06001` (train-free attention recalibration, LIBERO). ⇒ the runtime
  VLA-safety lane is **occupied**; our claim is the narrower **adaptive evasion-cost *measurement* for the
  instruction channel, FP-calibrated, with an action-space intrinsic frontier** — none of these measure adaptive
  evasion cost. `2602.01834` / `2603.06001` also bound the **L1-arm** novelty (cite + differentiate in §4b-I).
  *PDFs downloaded + SHA-256-pinned 2026-06-10 (`docs/references/README.md`).*
- **Hardware:** registered compute = **CSB `ecs3-0202` = 2× RTX A5000, 24 GB each** (switched 2026-06-16 — the
  first accessible GPU; **Kelvin2 A100/H100 demoted to backup**, separate registration if ever used). A **single**
  A5000 is the registered card; OpenVLA-7B runs in **bf16** (≈14 GB; fits with rollout headroom). The published
  H100 GCG timings do **NOT** transfer (A5000 ≈2.5–4× slower, no published prior) → the M1 micro-bench is the
  sole budget source. The compute branch (N / N− / F) is selected at M1 (§2), expected **N−/F**. **Log exact card
  + precision + parallelism per run; never compare across HW within one claim.**
- **Key paths:** understanding doc `docs/core/goal-action-consistency-detector.md`; landscape
  `docs/lit-review/`; references + verified facts `docs/references/`; results (write-once) `results/`;
  untrusted artefacts `artifacts/untrusted/`.

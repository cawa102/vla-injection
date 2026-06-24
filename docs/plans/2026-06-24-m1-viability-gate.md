# M1 Viability Gate — Benign Baseline + RoboGCG Targeted Redirect + Metric-(A) Separation (with attack-cost folded in)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task,
> and `test-driven-development` (`/tdd`) for every pure/model-free piece (GPU bodies stay behind the existing
> CUDA guard, exactly like `scripts/microbench_gcg.py` / `scripts/bench_early_stop.py`).

**Goal:** Deliver the M1 GO/NO-GO gate (H1) on the CSB A5000 — reproduce the benign LIBERO baseline, run the
**RoboGCG targeted redirect** (D2: one-shot suffix at rollout start), measure **benign-vs-attacked metric-(A)
separation** at the coarse operator-goal reference, and — folding the cancelled standalone item (i) in — log the
**GCG steps-to-success / early-stop / time** as a byproduct so it re-feeds `branch_select`.

**Architecture:** Compose existing seams, add the missing closed-loop glue. Reuse `openvla_loader`
(`load_frozen_openvla`/`build_target`), `gcg.run_gcg(..., reached_fn=…)` (early-stop), `OpenVlaGcgTarget`
(`reached`/chunked `loss_of`), the `early_stop_bench` bookkeeping (the (i) cost-logging), the
`smoke_libero_episode` rollout loop (generalise to a reusable runner with optional frozen-suffix injection),
`ConsistencyMetricA.score_rollout` + `SchemaA`, the eval harness (`calibrate`/`tpr_at_fpr`/`splits`), and
`TargetActionSpec.reached_window` (D2 window-scored ASR). New: a pre-registered **redirect target** spec and the
**benign-only SchemaA radius re-pin** rule (D-3 / invariant #2), locked **before** any attacked output is scored.

**Tech Stack:** Python 3.10; NumPy (pure specs/bookkeeping/aggregation + tests); torch 2.2.0+cu121 /
transformers 4.40.1 / bf16 OpenVLA-7B + LIBERO (EGL) behind the CUDA guard (same env as steps 4–6).

---

## Decisions (pre-registered)

- **DM-1 — (i)→(ii) merge (author-approved 2026-06-24).** The standalone item-(i) registered `bench_early_stop`
  run is **cancelled** (its fit-check is already banked by the 2026-06-24 dry run; its dummy-image cost is a proxy
  for the real attack). Its cost-logging machinery (`early_stop_bench.*`, `reached_fn`) is **reused** inside the
  (ii) attack so steps-to-success / early-stop / time fall out of the *representative* attack. `bench_early_stop.py`
  and its modules are **kept** (not deleted).
- **DM-2 — D2 closed-loop attack = one-shot suffix at rollout start; two distinct success notions.**
  (a) **GCG forced-decode `reached`** at the rollout-start frame → the **attack-cost** number (steps/early-stop).
  (b) **Rollout window-scored ASR** = the **frozen** best suffix applied across the closed loop, success iff
  `TargetActionSpec.reached_window` (region held for `persistence_steps`). (a) ≥ (b) is expected (mechanism M-a:
  the start-frame suffix may not persist) — both logged per unit; the GO/NO-GO uses (b).
- **DM-3 — D-3 / invariant #2 gate (load-bearing ordering).** **Before any attacked output is scored**, lock a
  **dated, benign-only, pre-registered** rule for re-pinning `SchemaA.{engagement_radius,grasp_radius}` from the
  **benign baseline** `geometry_stats` (the step-5 placeholders `0.05`/`0.10` undershoot real scene scale). The
  re-pin is computed from benign rollouts only, frozen, then attacked rollouts are scored against the frozen
  schema. No attacked data may influence the schema (circularity guard).
- **DM-4 — matched scenes + disjoint benign splits.** Attacked units `(task, target, seed)` are a **subset of the
  benign task/seed set** (so benign-vs-attacked is matched per scene). Benign **calibration** vs **held-out** splits
  are disjoint (`eval/splits.py`, invariant #3); `realised_fpr` is the held-out number.
- **DM-5 — low-level target action region only (D2).** The redirect target is a **pre-registered low-level action
  region**, deterministic from a seed. The **semantic-redirect arm (wrong-object/cross-task) is GATED on the M1
  outcome** (D2) → **not** built here; if M1 shows coherent targeted redirect, it is a later arm.
- **DM-6 — small attacked set, cheap benign set.** Each attacked unit costs one GCG optimisation (≤~4.9 h worst
  case, less with early-stop) → attacked set is small (`--n-attacked`, default 10, early-look after 3). Benign is
  GCG-free → generate `--n-benign` ≥ ~300 held-out for a 1 % FPR point (`docs/setup/gpu-runbook.md`).
- **DM-7 — registered run; resume + quarantine (reuse the bench driver pattern).** Per-unit write-once checkpoint
  (skip-if-exists), suffixes quarantined to `artifacts/untrusted/` (D6-6), `nohup`+`systemd-inhibit`+`until` for
  unattended runs (`docs/gpu/CSB/plan.md`). `--exclusive-gpu` recorded, not a hard gate (production attack runs
  don't need an exclusive window).

---

## Shared contracts (pure; both the GPU path and tests depend on these)

```python
# src/evasion_tax/attack/redirect_target.py  (pure, numpy — unit-tested off-GPU)
@dataclass(frozen=True)
class RedirectSpec:
    """A pre-registered targeted-redirect target (D2 / DM-5), deterministic from a seed."""
    target_action: tuple[float, ...]      # the 7-DoF redirect action (forced-decode target)
    region: "TargetActionSpec"            # the window-scored ASR region; target_action ∈ region

def redirect_spec_for(seed: int, *, persistence_steps: int) -> RedirectSpec: ...
def target_action_ids_for(spec: RedirectSpec, vocab_size: int) -> np.ndarray:
    """Forced-decode action-token ids for `spec.target_action` via the VERIFIED OpenVLA
    256-bin codec (local-prep Task 3 — do NOT re-derive the bin formula)."""
```

```python
# src/evasion_tax/eval/schema_repin.py  (pure, numpy — the DM-3 benign-only re-pin)
def repin_schema_from_benign(geometry_stats: Sequence[Mapping], *, base: SchemaA) -> SchemaA:
    """Re-pin engagement/grasp radii from BENIGN-only geometry_stats by the dated
    pre-registered rule (e.g. a fixed percentile of benign approach distances).
    Pure + deterministic; raises on empty input (no silent default)."""
```

```python
# the per-attacked-unit record = (i)'s TargetOutcome + the rollout ASR + metric-A score
@dataclass(frozen=True)
class AttackUnitRecord:
    unit_id: str                 # f"{task}:{target}:{seed}"
    cost: "TargetOutcome"        # reused from early_stop_bench (steps/early-stop/wall/peak/sha)
    rollout_asr_reached: bool    # window-scored ASR (DM-2b)
    is_denial: bool              # reached neither region nor task goal → denial, not redirect
    metric_a_per_step: tuple[float, ...]
```

---

## Tasks

- [x] **Task 0: Decision record + DM-3 gate lock — mac (docs + 1 pure helper)**

  **Files:**
  - Modify: `docs/core/execution-playbook.md` (§1 You-Are-Here, §7 M1, §10 ledger: record DM-1 the (i)→(ii) merge;
    record the dated DM-3 benign-only re-pin rule)
  - Modify: `docs/core/metric-a-annotation-schema.md` (the dated, benign-only `SchemaA` radius re-pin procedure)
  - Create: `src/evasion_tax/eval/schema_repin.py` + `tests/evasion_tax/eval/test_schema_repin.py`

  **What:** Pre-registration first (docs), then the **pure** re-pin helper. Records the plan-change and locks the
  circularity guard before any attacked code can read geometry. No model, no GPU.

  **Interface:** `repin_schema_from_benign(geometry_stats, *, base) -> SchemaA` (contract above).

  **Test scenarios:** deterministic re-pin from a fixture of benign approach distances; empty input raises;
  attacked/non-benign input is never accepted (signature takes benign stats only); re-pinned radii ≥ 0 and finite.

  **Dependencies:** `SchemaA` (`metric/consistency_a.py`); numpy.

  **Notes:** This is the **gate** — Tasks 5–6 must consume the frozen re-pinned schema, never re-pin from attacked
  data. The exact percentile/rule is the author's pre-registered choice; state the dated value in the schema doc.

  **Commit:** `docs+feat: lock (i)→(ii) merge + dated benign-only SchemaA re-pin (D-3/inv#2 gate)`

- [x] **Task 1: pure redirect-target spec (`RedirectSpec` + codec ids) — mac**

  **Files:**
  - Create: `src/evasion_tax/attack/redirect_target.py` + `tests/evasion_tax/attack/test_redirect_target.py`

  **What:** The pre-registered low-level redirect target (DM-5): a deterministic `RedirectSpec` (target action +
  its `TargetActionSpec` region) and the forced-decode `target_action_ids` via the **verified** OpenVLA codec.

  **Interface:** `redirect_spec_for(seed, *, persistence_steps) -> RedirectSpec`;
  `target_action_ids_for(spec, vocab_size) -> np.ndarray` (contracts above).

  **Test scenarios:** deterministic per seed; `spec.target_action` lies inside `spec.region` (forced-decode target
  is a member of the ASR region — DM-2 consistency); `target_action_ids` length = action dims, values in
  `[0, vocab_size)`; round-trips through the codec (decode(ids) ≈ target_action within one bin).

  **Dependencies:** the verified action codec (local-prep Task 3); `records.TargetActionSpec`; numpy. No torch.

  **Commit:** `feat: pre-registered low-level redirect target spec + forced-decode codec ids (D2/DM-5)`

- [x] **Task 2: reusable closed-loop rollout runner (benign + frozen-suffix attack) — mac core + box wiring**

  **Files:**
  - Create: `src/evasion_tax/eval/rollout_runner.py` (pure seams: suffix injection, ASR scoring, geometry_stats)
  - Modify: `scripts/smoke_libero_episode.py` (extract its env→policy→action→step loop into the runner; keep the
    smoke as a thin caller — surgical, no behaviour change to the step-4 smoke)
  - Test: `tests/evasion_tax/eval/test_rollout_runner.py`

  **What:** Generalise the verified step-4 loop into `run_episode(...)` that optionally injects a **frozen**
  adversarial suffix into the instruction each step, records the `RolloutStep` stream, and computes window-scored
  ASR (`TargetActionSpec.reached_window`) + benign `geometry_stats`. Pure seams unit-tested; the GPU/LIBERO body
  stays behind the guard.

  **Interface:**
  - `inject_suffix(instruction: str, suffix_text: str | None) -> str` (pure; `None` ⇒ benign, unchanged).
  - `rollout_asr(rollout: Rollout, region: TargetActionSpec) -> bool` (pure; window-scored).
  - `geometry_stats(rollout: Rollout) -> list[dict]` (pure; benign approach distances for DM-3).
  - `run_episode(model, processor, *, task, seed, suffix_text=None, device, max_steps) -> Rollout` (GPU-guarded).

  **Test scenarios (pure seams):** `inject_suffix(None)` is identity; ASR uses `reached_window` not a single step;
  a zeroed/benign action stream → `rollout_asr` False (anti-false-confidence, mirrors step-5); `geometry_stats`
  finite + correct length. **(box wiring, verify gate = a run):** one benign episode via the runner reproduces the
  step-4 smoke result (success, schema, VRAM ≤ one card).

  **Dependencies:** `records` (`RolloutStep`/`Rollout`/`TargetActionSpec`); `metric/state_libero`; OpenVLA eval
  helpers (`--openvla-root`); `PYTHONPATH=~/LIBERO`, `MUJOCO_GL=egl` (step-4 gotchas).

  **Notes:** Suffix is injected as text into the instruction (RoboGCG instruction channel). Do not re-optimise per
  step (DM-2: one-shot at start).

  **Commit:** `feat: reusable closed-loop rollout runner (frozen-suffix inject + window-scored ASR + geometry)`

- [x] **Task 3: per-rollout metric-(A) scoring + benign-vs-attacked separation — mac (TDD)**

  **Files:**
  - Create: `src/evasion_tax/eval/separation.py` + `tests/evasion_tax/eval/test_separation.py`

  **What:** Pure aggregation: score benign + attacked rollouts with `ConsistencyMetricA.score_rollout` against the
  **frozen re-pinned** `SchemaA` (Task 0) at the coarse operator-goal reference (`trusted_goal`), reduce to
  per-rollout scores, `calibrate` τ at the primary FPR on benign-calib, and report `tpr_at_fpr` on (benign-held-out,
  attacked) with held-out `realised_fpr` + CIs. Reuses the eval harness; no new statistics.

  **Interface:**
  - `per_rollout_score(scores: Sequence[Score]) -> float` (the frozen per-rollout reduction).
  - `separation_table(benign_calib, benign_eval, attacked, *, schema, fpr, trusted_goal) -> ResultsTable`.

  **Test scenarios:** clearly-separated synthetic benign/attacked → high TPR at the set FPR; τ comes from benign
  **calibration** only (invariant #3); `realised_fpr` measured on **held-out** benign; degenerate (benign==attacked)
  → ~chance TPR, no crash; empty attacked → documented (no separation claim).

  **Dependencies:** `metric/consistency_a`; `eval/{metrics,harness,splits}`; numpy.

  **Commit:** `feat: per-rollout metric-A reduction + benign-vs-attacked separation table (held-out FPR)`

- [ ] **Task 4: `run_benign.py` body — benign LIBERO baseline — mac core + box run**

  **Files:**
  - Modify: `scripts/run_benign.py` (replace the `NotImplementedError` with the runner loop)
  - Test: `tests/scripts/test_run_benign.py` (guard + the aggregate/log glue with the GPU call mocked)

  **What:** GPU-guarded driver: load frozen bf16 OpenVLA (Task 0 of step-4 env), roll `--n-benign` benign episodes
  via the runner (Task 2, `suffix_text=None`), log each `RolloutStep` stream + `geometry_stats` write-once to
  `results/<run>/`, record success rate. Per-episode checkpoint + resume (reuse the bench `RunHandle`/`is_*_done`
  pattern). Emits the benign `geometry_stats` that Task 0's re-pin consumes (DM-3 input).

  **Interface (CLI):** `--config … --n-benign 300 --calib-frac 0.25 --device cuda:0 --attn-impl flash_attention_2
  --openvla-root ~/openvla --seed 42 --results-root results --resume`.

  **Test scenarios (GPU mocked):** off-GPU ⇒ guard exits 2; `--resume` skips a finished episode; the aggregate
  success-rate + benign score split (calib/held-out) match `splits.py`; geometry_stats written.

  **Dependencies:** Tasks 2,3; `openvla_loader`; `config.cuda_available`; the repro `run.json` writer.

  **Notes:** Benign is GCG-free → cheap; this run also produces the DM-3 re-pin input, so it must complete (or a
  pilot subset) **before** the attacked run is scored.

  **Commit:** `feat: benign LIBERO baseline driver (rollouts + geometry_stats + calib/held-out split)`

- [ ] **Task 5: `run_attack.py` body — RoboGCG targeted redirect + folded-in cost — mac core + box run**

  **Files:**
  - Modify: `scripts/run_attack.py` (replace the `NotImplementedError`)
  - Test: `tests/scripts/test_run_attack.py` (guard + resume/quarantine/aggregate glue, GPU mocked)

  **What:** GPU-guarded driver. For each attacked unit `(task, target, seed)` in the matched subset (DM-4): build
  `OpenVlaGcgTarget` on the **real rollout-start obs** with the **redirect** `target_action_ids` (Task 1) +
  `eval_batch` chunking; `run_gcg(target, cfg, reached_fn=target.reached)` (early-stop ON) → a `TargetOutcome`
  (**reused** `early_stop_bench` bookkeeping = the (i) cost); **freeze** `best_suffix` and run the closed-loop
  attacked rollout (Task 2) → window-scored ASR + denial flag; score metric-(A) against the **frozen re-pinned**
  schema → `AttackUnitRecord`; write it **write-once** per unit; **quarantine** the suffix. Resume skips finished
  units (DM-7).

  **Interface (CLI):** `--config … --n-attacked 10 --search-width 512 --n-steps 500 --eval-batch 32 --device cuda:0
  --attn-impl flash_attention_2 --openvla-root ~/openvla --seed 42 --schema-from <benign-run> --results-root results
  --exclusive-gpu --resume`.

  **Test scenarios (GPU mocked):** off-GPU ⇒ exits 2; `--resume` skips a unit whose JSON exists; exactly one
  quarantined suffix per fresh unit; the per-unit record carries both success notions (cost.reached + rollout ASR);
  `--schema-from` loads the **frozen** re-pinned schema (refuses to re-pin from attacked data — DM-3).

  **Dependencies:** Tasks 0,1,2,3; `openvla_loader`; `attack/{gcg,gcg_openvla,early_stop,early_stop_bench}`.

  **Notes:** Unattended launch identical to `bench_early_stop` (`nohup systemd-inhibit … until …`,
  `docs/gpu/CSB/plan.md`). Validate with a non-registered dry run (`--n-attacked 1 --n-steps 20 --results-root
  results/_smoke`) first. Early-look after 3 units (DM-6).

  **Commit:** `feat: RoboGCG targeted-redirect attack driver (one-shot suffix, window-scored ASR, folded cost)`

- [ ] **Task 6: M1 viability-gate aggregate + GO/NO-GO report — mac (TDD)**

  **Files:**
  - Create: `scripts/m1_gate_report.py` + `src/evasion_tax/eval/m1_gate.py` + tests for both

  **What:** Pure aggregation over the benign + attacked records → the H1 verdict: **(a)** benign reproduced (success
  rate vs published); **(b)** targeted redirect **not denial** (rollout ASR > 0, denial fraction reported); **(c)**
  benign-vs-attacked **separation survives at the coarse operator-goal reference** (Task 3 table); **(d)** the
  **attack-cost distribution** (`steps_to_success_summary` + `realistic_s_per_target` — the folded (i) number).
  Emits one report + the branch hint.

  **Interface:** `m1_verdict(benign_records, attack_records, *, schema, fpr) -> dict` (the four sub-verdicts +
  cost summary + a one-line GO/NO-GO).

  **Test scenarios:** GO when all four hold on a fixture; denial-only attacked → flags "reframe to task-deviation"
  (§7 M1 NO-GO branch); separation only at the clean-instruction ceiling → flags weak necessity; cost summary equals
  `steps_to_success_summary` over the attacked records.

  **Dependencies:** Tasks 3,5; `attack/early_stop_bench`; `eval/separation`.

  **Commit:** `feat: M1 viability-gate aggregate + GO/NO-GO (H1) report with folded attack-cost`

- [ ] **Task 7: record cost → `branch_select` re-feed + docs — mac (after the box runs)**

  **Files:**
  - Modify: `docs/core/execution-playbook.md` (§1, §6 D7/D8, §10), `docs/gpu/CSB/plan.md` (M1 section)

  **What:** Feed `realistic_s_per_target` (now from the **real** attack) into `branch_select.affordable_matrix`;
  record the **measured** GO/NO-GO outcome (H1), the provisional branch (still hard-F default until the **adaptive**
  cost (iii) lands), and the separation result. Note the (i)→(ii) merge is delivered.

  **Test scenarios:** docs not unit-tested; `branch_select` already covered; branch stays `provisional`,
  `locked=False`, `default_if_unconfirmed="F"`.

  **Dependencies:** Task 6 registered records; `eval/branch_select.py`.

  **Commit:** `docs: M1 GO/NO-GO outcome + real-attack cost → refine provisional branch (hard-F default)`

---

## Build order & where each runs

1. **On the mac (TDD, `/tdd`):** Task 0 (gate + decision record) → Task 1 (redirect spec) → Task 3 (separation) →
   Task 2 *pure seams* → Task 6 (gate aggregate) → Tasks 4,5 *core/glue* with the GPU call mocked. Full
   `src/evasion_tax` stays type-clean + ruff-clean; suite green.
2. **On the CSB A5000 box (unattended):** Task 2 box wiring (benign episode via the runner == step-4) → Task 4
   **benign baseline run** (produces the DM-3 geometry input) → **lock the re-pinned schema (Task 0 rule on the
   benign run)** → Task 5 **non-registered dry run** (1 unit, 20 steps) → Task 5 **registered attacked run**
   (`nohup`+`systemd-inhibit`, per-unit checkpoint, early-look after 3) → push results.
3. **On the mac:** Task 6 report + Task 7 — branch_select re-feed + doc updates.

**Verify gates:** (a) benign-only re-pin is deterministic + refuses attacked data; (b) redirect target ∈ ASR region
+ codec round-trips; (c) one benign episode via the runner reproduces step-4; (d) separation table uses held-out
FPR + calib-only τ; (e) attacked run logs **both** success notions per unit, resume + quarantine proven; (f) the M1
report emits the four-part H1 verdict + the folded attack-cost distribution → re-feeds `branch_select`.

---

## References

@scripts/bench_early_stop.py + @src/evasion_tax/attack/early_stop_bench.py (the (i) cost machinery reused) ·
@src/evasion_tax/attack/openvla_loader.py (`load_frozen_openvla`/`build_target`) ·
@src/evasion_tax/attack/gcg.py (`run_gcg(..., reached_fn=…)`, `GcgResult`) ·
@src/evasion_tax/attack/gcg_openvla.py (`OpenVlaGcgTarget`: `reached`, chunked `loss_of`, `decode_span`) ·
@scripts/smoke_libero_episode.py (the step-4 rollout loop to generalise) ·
@scripts/attach_l2_to_rollout.py (offline metric-A scoring pattern) ·
@src/evasion_tax/metric/consistency_a.py (`ConsistencyMetricA`, `SchemaA`) ·
@src/evasion_tax/records.py (`Rollout`, `RolloutStep`, `TargetActionSpec.reached_window`) ·
@src/evasion_tax/eval/{metrics,harness,splits,branch_select}.py (calibration / held-out FPR / branch) ·
@docs/core/execution-playbook.md (§3a H6, §6 D2/D3/D7/D8, §7 M1, §10) · @docs/gpu/CSB/plan.md (Unattended runs).

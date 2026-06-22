# CSB Step 6 (D8 finalisation) — RoboGCG-Budget-Faithful Attack Config + s/step Micro-bench — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: `executing-plans` to implement task-by-task; `/tdd`
> (`test-driven-development`) for the **model-free** pieces (the pure `faithful_s_step` helper). The
> registered run executes on the **CSB A5000 box**; the model-touching timing is GPU-guarded exactly like the
> existing seam.

**Goal:** Pin the EET instruction-channel attack's **search-budget** knobs to RoboGCG's defaults
(**budget-faithful** — *not* a blanket "faithful attack" claim; see the divergence contract in DC-1/DC-6) and
measure the D8 budget as a **per-step** cost (cheap), so `branch_select` gets the real `s/target` at
`num_steps=500` **without** running 500 steps or implementing eval mini-batching yet.

**Architecture:** RoboGCG decouples **`search_width`** (candidates scored per step = 512) from **`batch_size`**
(forward mini-batch = 64). Our `loss_of` currently forwards *all* `search_width` at once, so the A5000's 24 GB
caps us at `search_width≈32`. We keep the **search-budget** knobs faithful (sw=512 / ns=500 / topk=256 /
n_replace=1 / early_stop) and adapt only the **hardware** knob (`batch_size` = measured max-B ≈ 32–48, not 64);
two attack-mechanics divergences (suffix placement, mean-CE loss) and the **not-yet-implemented** token filters
(`allow_non_ascii`/`filter_ids`) are governed by the DC-1/DC-6 divergence contract — **not** silently claimed as
faithful. For D8 we do
**not** run the full attack: we time two primitives — `t_grad` (one `token_gradient`) and `t_fwd` (one
`loss_of` at B = eval-batch) — and compute `s/step(sw=512) = t_grad + ⌈512/eval_batch⌉·t_fwd`, then
`s/target = s/step · 500`. Full eval mini-batching (to *run* sw=512) is deferred to M1.

**Tech Stack:** Python 3.10; NumPy (pure helper + tests); torch 2.2.0+cu121 / transformers 4.40.1 / bf16
OpenVLA-7B behind the GPU guard. Reuses `scripts/microbench_gcg.py`, `src/evasion_tax/eval/branch_select.py`.

---

## Provenance

@docs/plans/2026-06-19-step6-truebatch-lossof.md delivered the true-batch `loss_of` + the DB-4 equivalence gate
(Tasks 1–3 GREEN on the box 2026-06-22). This plan **supersedes that plan's Task-4 *measurement config*** only:
its toy `s/target` at `n5/W32/1tgt` is replaced by the **budget-faithful s/step** measurement below (the harness +
DB-2 loop/`speedup_k` recording stand). Every other parent decision (DB-1/3/5/6, D6-1/3/6/9/10) is unchanged.

**RoboGCG defaults — VERIFIED 2026-06-22** from `eliotjones1/robogcg` (commit
`6de53e0671482f79313a32822f21fcea44f53979`, 2025-05-28), `experiments/single_step/config.py`
(`SingleStepConfig`, lines 70–94; blob `8efe807a1ab571808ffe6a884e3f459005df3cd0`), via `gh api`:

| Param | RoboGCG | Ours (status) | Meaning |
|---|---|---|---|
| `num_steps` | **500** | n/a (calib 5) | max optimisation steps |
| `early_stop` | **True** | not wired | stop when target action reached |
| `search_width` | **512** | 32 | candidates scored per step (**search breadth**) |
| `batch_size` | **64** | =search_width | candidates per forward (**eval mini-batch**) |
| `topk` | 256 | 256 ✓ | top tokens per position |
| `n_replace` | 1 | 1 ✓ (D6-3) | tokens replaced per step |
| `buffer_size` | 0 | 0 ✓ | no attack buffer |
| `allow_non_ascii` | **False** | **NOT implemented** (gap; verified absent 2026-06-22) | restrict candidate tokens to ASCII (search space) |
| `filter_ids` | **True** | **NOT implemented** (gap; verified absent 2026-06-22) | drop candidates that don't re-tokenise (candidate validity) |
| `mellowmax_alpha` | 1.0 | mean-CE (divergence) | loss aggregation over the target tokens |
| `as_suffix` | False | suffix (divergence) | adversarial-token placement |

## Decisions (pre-registered)

- **DC-1 — Attack = RoboGCG-faithful on the *search-budget* knobs ONLY (not a blanket "faithful attack").**
  Faithful set: `search_width=512`, `num_steps=500`, `topk=256`, `n_replace=1`, `buffer_size=0`,
  `early_stop=True`. Reducing `search_width` weakens the per-step search 16× and confounds the cross-layer
  comparison, so it is **not** the scoping lever (DC-3). `topk`/`n_replace` already match. **Three items are NOT
  yet faithful, governed by the DC-6 divergence contract:** the token filters `allow_non_ascii=False` /
  `filter_ids=True` (currently *not implemented* in `gcg.py`/`gcg_openvla.py` — verified absent 2026-06-22) and
  the two seam divergences (suffix placement, mean-CE loss). D8 only *measures s/step* and is **insensitive** to
  the token filters (they reshape candidate sampling/validity, not the `t_grad`/`t_fwd` forward cost), so they are
  scheduled with Task D / M1 — but **no result, branch, or docs wording may call the attack itself
  "RoboGCG-faithful" until all three are resolved** (filters implemented + seam divergences signed off, DC-6).
- **DC-2 — `batch_size` (eval mini-batch) = measured max-B on the A5000 (HW-adapted, NOT 64).** B=32 already
  peaks 21.3 GiB; B=64 (~28 GiB est.) OOMs on 24 GB — RoboGCG's 64 assumes A100/H100. Sweep the **real** max-B
  (≈32–48) and use it as `batch_size`; **log the card + the value** (invariant #8). Higher max-B ⇒ fewer chunks
  ⇒ lower `s/step`, so max-B **does** feed the cost estimate here (a sharper framing than the parent DB-3 "max-B
  is not branch-critical" — the *branch* is still provisional/hard-F, but `batch_size` shapes `s/step`).
- **DC-3 — Scope via the experiment matrix, not the attack.** Branch N− = fewer targets/seeds/conditions (5 %-FPR
  primary), **never** a crippled `search_width`. Keeps the **search budget** citable (faithful on the budget knobs;
  attack-mechanics faithfulness governed by DC-6).
- **DC-4 — D8 measures `s/step`, scales analytically; no full 500-step run, no mini-batching yet.** Time
  `t_grad` + `t_fwd(B=eval_batch)`; `s/step(sw=512)=t_grad+⌈512/eval_batch⌉·t_fwd`; `s/target=s/step·500`
  (worst case, early_stop off). Cross-check: the directly-timed `run_gcg` step at `sw=eval_batch` must ≈
  `t_grad+t_fwd`. **Label `s/target` an analytic estimate.** `early_stop` convergence (effective steps) is an
  **M1 empirical** question, not D8.
- **DC-5 — Eval mini-batching DEFERRED to M1 (Task D, not executed here).** To actually *run* sw=512 we must
  decouple `search_width` from the forward batch: add `GcgConfig.batch_size`, chunk `loss_of` into
  `eval_batch`-sized forwards, and extend the DB-4 gate to the chunked path. Out of scope for D8.
- **DC-6 — Divergence contract (faithful-vs-divergent, pre-registered for supervisor sign-off).** The attack
  diverges from RoboGCG on **three** items; none may be silently adopted, and while any remains open the attack
  may **not** be described as "RoboGCG-faithful": (i) `as_suffix` — RoboGCG `False` (in-place/prefix) vs our
  **suffix** placement; (ii) loss — RoboGCG **mellowmax α=1.0** vs our **mean-CE** over the 7 target tokens
  (DB-5 put mellowmax out of MVP scope); (iii) token filters — RoboGCG `allow_non_ascii=False` /
  `filter_ids=True` vs our **absent** filtering (no ASCII restriction, no re-tokenisation validity check;
  verified absent 2026-06-22). (i)–(ii) are placement/aggregation choices deferred by MVP scope; (iii) is an
  unimplemented gap that **must be closed before any sw=512 "faithful attack" run** (Task D / M1). Record + raise
  all three; D8's `s/step` measurement stands regardless (it is filter-insensitive — DC-1).

## Today's bf16 / gate findings to record (folded into Task C)

Measured on the box 2026-06-22 (`results/_smoke/2026-06-22T14-32-35Z-gcg-tiny-smoke`,
`…T14-35-29Z-gcg-microbench`):
- **OOM fix** — true-batch `loss_of` dropped `labels=`: HF's full-vocab fp32 `shift_logits` was the OOM driver;
  `out.logits` is identical without it (commit `c8f868f`).
- **DB-4 gate refined to selection-regret** — true-batch CE is **exact at B=1** but bf16 batch-order noise makes
  it differ ≤~0.26 at B>1, and the argmin can **flip** between batched/single on near-tied candidates (run-to-run,
  cuBLAS-heuristic dependent). The load-bearing invariant is **selection regret** (`ref[argmin(cmp)]−min(ref)`),
  not strict argmin index equality (commits `3f11db3`, `559f713`). Gate = 8/8.
- **`speedup_k = 1.53×`** (loop 17.475 / true-batch 11.394 s/target @ n5/W32/1tgt) — DB-2 engineering-tax ablation.

## Tasks

- [ ] **Task A: faithful `s/step` instrumentation in the micro-bench — mac (TDD pure helper) + box (registered)**

  **Files:**
  - Modify: `scripts/microbench_gcg.py` (add the pure `faithful_s_step`; time `t_grad`+`t_fwd` per target; new
    CLI `--faithful-search-width`/`--faithful-num-steps`/`--eval-batch`; record the faithful block)
  - Test: `tests/evasion_tax/.../test_microbench_gcg.py` (or the existing microbench test module) — `faithful_s_step` cases

  **What:** Keep the existing max-B sweep (it now sets `batch_size`). Per timed target, additionally time one
  `target.token_gradient(init)` → `t_grad` and one `target.loss_of(B=eval_batch)` → `t_fwd` (median over the
  targets). Compute `s/step` and the analytic `s/target` for `(search_width=512, num_steps=500)` and record them
  alongside the existing direct `run_gcg` numbers + the DB-2 loop ablation.

  **Interface:**
  - `faithful_s_step(t_grad: float, t_fwd: float, *, search_width: int, eval_batch: int) -> float` — pure;
    returns `t_grad + ceil(search_width/eval_batch) * t_fwd`. Raises `ValueError` if `eval_batch < 1` or
    `search_width < 1`.
  - record gains a `faithful` block: `{search_width, num_steps, eval_batch, t_grad_s, t_fwd_s, s_per_step,
    s_per_target_worstcase}`.

  **Test scenarios (pure, off-GPU):** `sw=512, eval=32 → 16 chunks` (`t_grad+16·t_fwd`); `sw=32, eval=32 → 1
  chunk` (equals the direct one-step cost — the cross-check); `sw=512, eval=48 → 11 chunks`; `eval_batch=0`
  raises; module still imports no torch at top.

  **Dependencies:** numpy; torch inside the guarded `main` only; reuses `_build_target`, `summarise_timings`,
  `build_microbench_record`.

  **Notes:** `t_fwd` is timed at `B=eval_batch` (= measured max-B, or `--eval-batch`). The analytic `s/target`
  is a **budget-faithful estimate** (label it so — it reflects the sw=512/ns=500 *budget*, not a fully-faithful
  attack; the absent token filters and the DC-6 seam divergences do not change `t_grad`/`t_fwd`, so the timing is
  valid as-is). The direct `run_gcg` step at `sw=eval_batch` is the cross-check. Registered run → write-once
  `results/` with the §8 header on an exclusive A5000.

  **Commit:** `feat: faithful s/step micro-bench (t_grad + ceil(sw/eval)·t_fwd; RoboGCG sw=512/ns=500)`

- [ ] **Task B: branch selection on the budget-faithful `s/target` — mac**

  **Files:** none new — reuse `src/evasion_tax/eval/branch_select.py` (parent Task 5), fed the
  **budget-faithful** `s/target = s_per_step·500`.

  **What:** Run the `branch_select` arithmetic on the budget-faithful `s/target` against the remaining calendar +
  matrix → the **provisional** branch (hard-F default per the parent `branch_lock_condition`).

  **Test scenarios:** `branch_select` already covered by parent tests; recorded branch is `provisional`,
  `locked=False`, `default_if_unconfirmed="F"`.

  **Dependencies:** Task A's registered `s_per_step`.

  **Commit:** (none if no source change — record the decision in Task C.)

- [ ] **Task C: documentation — record the decisions + the bf16 findings; tick CSB step 6 — mac**

  **Files:**
  - Modify: `docs/core/execution-playbook.md` (§1 You-Are-Here; §2 *Compute branches*/§6 D8; §10 decision log —
    RoboGCG-**budget-faithful** config + the DC-1/DC-6 divergence contract (token filters absent + seam
    divergences), the budget-faithful `s/step`→branch, the bf16/selection-regret gate, `speedup_k`)
  - Modify: `docs/gpu/CSB/plan.md` (tick **step 6**; add a step-6 how-to: budget-faithful config + divergence
    contract, the labels-drop OOM fix, the selection-regret gate, the s/step micro-bench, the
    max-B=`batch_size` framing)
  - Modify: `docs/plans/2026-06-19-step6-truebatch-lossof.md` (back-pointer: Task-4 *config* superseded here)

  **What:** Land every 2026-06-22 decision/finding so the SoT matches git; carry the DC-1/DC-6 divergence contract
  (token filters absent + suffix/mean-CE seam divergences) into the SoT and flag it for supervisor sign-off —
  the SoT must not assert a fully "RoboGCG-faithful attack" while any of the three remain open.

  **Test scenarios:** docs not unit-tested; ensure no contradiction with the parent plans.

  **Commit:** `docs: register RoboGCG-budget-faithful D8 config + divergence contract + bf16 gate findings; tick CSB step 6`

- [ ] **Task D: eval mini-batching — DEFERRED to M1 (NOT executed in this plan)**

  **Files (when done):** `src/evasion_tax/attack/gcg.py` (`GcgConfig.batch_size`; `run_gcg` chunks; candidate
  filters `allow_non_ascii`/`filter_ids`), `src/evasion_tax/attack/gcg_openvla.py` (`loss_of(..., eval_batch)`
  chunking), the DB-4 gate (chunked path).

  **What:** Decouple `search_width` (512) from the forward batch so the **real** attack runs sw=512 in
  `⌈512/batch_size⌉` forwards. Extend the equivalence gate to the chunked path (chunked `[B]` == per-candidate
  single, value + selection-regret). **Also the home for closing the DC-6 faithfulness gap before any
  "RoboGCG-faithful attack" claim:** implement `allow_non_ascii=False` (ASCII-only candidate vocab) +
  `filter_ids=True` (drop candidates that don't re-tokenise), and resolve/sign-off the suffix and mean-CE
  divergences. **Listed for traceability; build it when M1 runs the real attack.**

  **Commit (future):** `feat: decouple search_width from eval batch (mini-batched loss_of) for sw=512`

## Build order & where each runs

1. **Mac (TDD, `/tdd`):** Task A pure `faithful_s_step` + tests; wire the timing + record + CLI (GPU body guarded).
2. **Box (exclusive A5000):** Task A registered run → max-B sweep (sets `batch_size`) + `t_grad`/`t_fwd` → budget-faithful `s/step` → write-once `results/`.
3. **Mac:** Task B `branch_select` on the budget-faithful `s/target`; Task C docs + tick step 6.
4. **Deferred (M1):** Task D mini-batching.

## References

@docs/plans/2026-06-19-step6-truebatch-lossof.md (parent; Task-4 config superseded) ·
@docs/plans/2026-06-19-step6-gcg-microbench.md (grandparent; harness) · `eliotjones1/robogcg`
`experiments/single_step/config.py` (RoboGCG defaults, verified 2026-06-22; commit `6de53e0`, blob `8efe807`) ·
@src/evasion_tax/attack/gcg.py (`GcgConfig`, `run_gcg`) · @src/evasion_tax/attack/gcg_openvla.py
(`loss_of`, `token_gradient`, `equivalence_verdict`) · @scripts/microbench_gcg.py ·
@src/evasion_tax/eval/branch_select.py · @docs/core/execution-playbook.md (§2/§6/§10) · @docs/gpu/CSB/plan.md (step 6).

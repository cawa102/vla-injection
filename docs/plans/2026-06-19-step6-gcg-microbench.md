# CSB Bring-up Step 6 — GCG Attack Harness + D8 Timing Micro-bench — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `test-driven-development` (`/tdd`) to implement the **model-free**
> tasks (1, 5, and the pure parts of 2 & 4) task-by-task. The GPU-only bodies (real OpenVLA forward, the tiny
> run, the timing bench) are run on the **CSB A5000 box**, guarded locally exactly like the existing GPU scripts.

**Goal:** Build a minimal **own** GCG textual-suffix search harness on the step-5.5 OpenVLA loss/gradient seam,
prove it works on a **tiny run** (few steps, 1 example), then run the **D8 timing micro-bench** (GCG `s/target`,
peak VRAM, max candidate-batch at 24 GB on one A5000) — the first **registered** measurement — and from it
**select compute Branch N / N− / F** (`docs/gpu/CSB/plan.md` step 6; playbook §2 *Compute branches*, D8).

**Architecture:** Mirror the repo's proven split — a **model-free algorithm core** (the GCG loop: one-hot
top-k → candidate sampling → best-by-loss selection → suffix bookkeeping) behind a thin `LossGradientFn` seam,
plus a **GPU-only seam impl** (`OpenVlaGcgTarget`) that reuses the *exact* step-5.5 machinery (frozen bf16
model, the token-embedding forward hook, CE loss to a target action-token sequence) and adds (a) the **one-hot
token gradient** = the input-embedding gradient at the suffix positions projected through the embedding matrix,
and (b) **batched candidate-loss** evaluation. Two drivers: a tiny-run smoke (`results/_smoke/`,
non-registered) and the micro-bench (`scripts/microbench_gcg.py` stub filled → write-once `results/`,
registered). A model-free branch-selector turns the measured `s/target` into the affordable matrix → the
pre-registered branch. This is the same pattern as `attack/idealized_frontier.py` (model-free core +
`SyntheticDynamics`/`RealDynamics` seam) and `scripts/smoke_openvla_gradient.py` (the GPU seam foundation).

**Tech Stack:** Python 3.10; NumPy (the model-free core is pure NumPy — no torch); the `evasion_tax` package
(`policy.action_codec`, `records.TargetActionSpec`, `config` GPU guard, `repro.RunLogger`/`seed_everything`).
GPU-only deps (torch 2.2.0+cu121, transformers 4.40.1, flash-attn 2.5.5, bf16 OpenVLA-7B) live behind the guard,
identical to `smoke_openvla_gradient.py`. Reuse reference for the **algorithm** (not the integration):
GraySwanAI **nanoGCG** (`pip install nanogcg`) and the **RoboGCG** repo (`github.com/eliotjones1/robogcg`),
both Gray Swan AI; GCG from Zou et al. *Universal and Transferable Attacks* (`arXiv:2307.15043`).

---

## Decisions (pre-registered — design-fork-handling: decide deeply + record, don't multiple-choice ask)

- **D6-1 — Build our OWN minimal GCG harness on the step-5.5 seam; do NOT run upstream RoboGCG for step 6.**
  Step 5.5 (`scripts/smoke_openvla_gradient.py`) was built *as* "the GCG prerequisite" against our bf16+flash-attn
  load — step 6 is the search loop on top of it. Upstream RoboGCG cannot (i) take an external `inputs_embeds`
  (OpenVLA's multimodal `forward` owns `input_ids`→embeds — the step-5.5 finding), nor (ii) later optimise
  against a **detector score** (the M4/H6-D adaptive arm, D7/#5) — so the load-bearing GCG at scale must be ours.
  We **reuse the algorithm** (port the one-hot-gradient top-k + candidate-batch loop from nanoGCG / the RoboGCG
  repo, Task 0), not the harness. *Running upstream RoboGCG in its own env to confirm the published targeted
  redirect (gpu-runbook Step 4) is a **separate M1 cross-check**, out of step-6 scope.*
- **D6-2 — Model-free algorithm core + GPU-guarded seam (TDD).** The GCG loop is pure-NumPy and unit-tested on a
  **toy `LossGradientFn`** with a known optimum (mirrors `idealized_frontier.py` + `SyntheticDynamics`). The real
  OpenVLA loss/gradient is GPU-only and guarded (`cuda_available()` → `gpu_required_message` → exit 2, never a
  silent no-op), like every existing GPU script.
- **D6-3 — Tiny-run success = harness runs + loss strictly decreases + target reached.** The tiny run (1 task,
  1 target, few steps) passes iff the GCG loop runs end-to-end, the CE loss is **monotone non-increasing** and
  ends materially below its start, **and** the 7 target action tokens become the argmax (or the loss falls below
  a pinned threshold). It is **NOT** an ASR claim and **NOT** a closed-loop attacked rollout — it is the harness
  wiring de-risk (the GCG analogue of step 5's attach gate). `results/_smoke/` (non-registered bring-up smoke).
- **D6-4 — Micro-bench scope = `s/target` + peak VRAM + max candidate-batch B @ 24 GB, on our harness, registered.**
  A few targets, one pinned GCG config; sweep B upward to the largest that fits 24 GB; record median `s/target`,
  peak VRAM, max B, steps-to-success distribution → write-once `results/` with the full §8 repro header
  (this is a **real result**, plan.md "step 6 produces registered measurements"). The other M1-micro-bench items
  — **L1 activation/attention extraction overhead** and **adaptive-GCG-against-the-probe-score** cost (D7/#5) —
  are **DEFERRED**: the L1 probe is not yet on GPU (M2/M3), so they are **named here, not silently dropped**, and
  measured when L1 lands. LIBERO rollout throughput was already observed at step 4 (90 steps / episode, 14.5 GiB).
- **D6-5 — Branch selection uses measured non-adaptive `s/target` as the primary cost driver; conservative on
  borderline.** The affordable matrix (§2) is computed from the measured `s/target` against the remaining
  calendar. Because the **adaptive** (M4) cost cannot be measured until L1 exists, it is **estimated** as
  `adaptive_mult × s/target` (pre-register `adaptive_mult` from the GCG-step-count literature / RoboGCG's 30–110
  steps-to-success; refine when L1 lands). If the matrix sits **within ~20% of a branch boundary, pick the more
  conservative branch** (N→N−, N−→F). The selected branch + the numbers are logged to playbook §1/§2/§10 and
  CSB plan.md step 6 is ticked. **M3/H6-A is delivered in every branch** regardless (floor unchanged).
- **D6-6 — Ethics: every optimised suffix is an adversarial artifact → quarantine.** Both the tiny-run and
  micro-bench suffixes are written under `artifacts/untrusted/` (gitignored), never auto-run, never committed
  (CLAUDE.md security-research ethics; `run_attack.py` docstring). Targets are **arbitrary low-level action-token
  sequences** — *control authority*, not semantic harm (RoboGCG framing, `docs/references/README.md` §RoboGCG).
- **D6-7 — Target spec via the existing codec (D2-consistent).** A target = **7 action bins** →
  `ActionCodec.decode` → a continuous 7-DoF action → the centre of a `TargetActionSpec` (records.py). This reuses
  built code and keeps the attacked target expressible in the D2 contract that M1 will score. For step 6 the
  target is **fixed/arbitrary** (e.g. step-5.5's `_TARGET_BINS`); coherent semantic redirect is the M1-gated D2 arm.
- **D6-8 — D-3 radius re-pin governance is a precondition for M1 detector-scoring, not for the step-6 harness.**
  Step 6 drives the model's **logit/CE loss**, not the metric-A detector, so it does **not** "inspect attacked
  output" through the detector and triggers **no** `SchemaA` radius decision. The dated, benign-only,
  pre-registered re-pin rule (`docs/core/d3-radius-repin-preregistration.md`, invariant #2) must be **locked
  before M1 scores attacked rollouts through metric A** — flagged here, not actioned in step 6.

---

## Scope boundary (do NOT exceed — this is plan.md step 6's verify gate)

**IN:** our GCG suffix-search harness (model-free core + GPU seam) · a tiny run that proves the harness works ·
the D8 `s/target`/VRAM/max-batch micro-bench · the branch-selection computation + decision log.

**OUT (explicitly deferred / other milestones):**
- **Closed-loop *attacked* LIBERO rollouts** and **H1 benign-vs-attacked separation / AUC** → M1 §7 (needs the
  attacked class fed through a rollout; step 6 stops at the optimised suffix + its argmax action).
- **Scoring attacked output through metric-A / any detector** → M1/M2 (and gated on D6-8 / D-3 lock).
- **Upstream RoboGCG faithful reproduction** (own env, published >90% ASR) → M1 cross-check (gpu-runbook Step 4).
- **Adaptive-GCG-against-the-probe** and **L1 extraction overhead** micro-bench items → when L1 lands on GPU (D6-4).
- **Semantic-redirect coherence** (D2 cross-task/wrong-object arm) → M1-gated (D2).
- **Multi-position swap / attack buffer / mellowmax / probe-sampling** nanoGCG refinements → pre-registered
  stretch only (the MVP is single-position-swap GCG; add a refinement **only** if the tiny run won't converge).

---

## References

@docs/gpu/CSB/plan.md (step 6 ladder + step-5.5 how-to / the GCG-gradient seam this builds on) ·
@docs/core/execution-playbook.md (§2 *Compute branches* + roadmap, §3 H1/H6-A/H6-D, §6 D4/D7/D8, §8 run-log) ·
@docs/setup/gpu-runbook.md (Step 5 micro-bench spec → D4/D7/D8; ethics/quarantine standing rules) ·
@docs/references/README.md (§RoboGCG verified facts: white-box textual suffix, 256^7 action space, multimodal-PF
useless, no usable input-side operating point; §OpenVLA action codec formula provenance) ·
@scripts/smoke_openvla_gradient.py (the bf16+flash-attn loss/gradient seam + embedding-hook mechanism to reuse) ·
@scripts/microbench_gcg.py + @scripts/run_attack.py (the two guarded stubs to fill) ·
@src/evasion_tax/policy/action_codec.py (`decode`: target bins → continuous action → `TargetActionSpec`) ·
@src/evasion_tax/records.py (`TargetActionSpec.reached_window` — the D2 target contract) ·
@src/evasion_tax/attack/idealized_frontier.py (the model-free-core + seam pattern to mirror) ·
@src/evasion_tax/repro/run_logger.py (write-once `RunLogger.start().write()`) ·
@docs/core/d3-radius-repin-preregistration.md (the D-3 precondition flagged in D6-8) ·
@docs/plans/2026-06-18-l2-attach-real-rollout.md (the prior step's plan — style + scope-boundary template).

---

## Shared contract (types both the core and the seam depend on — include per writing-plans)

```python
# src/evasion_tax/attack/gcg.py
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class GcgConfig:
    suffix_len: int            # number of adversarial tokens (e.g. 20)
    n_steps: int               # GCG iterations (tiny run: a handful; bench: a pinned budget)
    top_k: int                 # candidate tokens per position from the gradient (GCG default 256)
    search_width: int          # B = candidate suffixes evaluated per step (the VRAM-dominant knob)
    seed: int                  # pinned (repro invariant #1)
    # init_token_id / allowed_token_ids optional; default = a fixed init suffix.

class LossGradientFn(Protocol):
    """The model seam. Pure NumPy in/out so the core never imports torch."""
    @property
    def vocab_size(self) -> int: ...
    def init_suffix_ids(self) -> np.ndarray: ...                       # [suffix_len] int
    def token_gradient(self, suffix_ids: np.ndarray) -> np.ndarray: ... # [suffix_len, vocab] one-hot grad
    def loss_of(self, candidate_suffixes: np.ndarray) -> np.ndarray: ... # [B,suffix_len]->[B] (no grad)

@dataclass(frozen=True)
class GcgResult:
    best_suffix_ids: tuple[int, ...]
    best_loss: float
    loss_history: tuple[float, ...]   # per-step best loss (must be non-increasing)
    n_steps_run: int
    reached: bool                     # target tokens are argmax / loss <= threshold
```

**One-hot gradient (the load-bearing math, faithful to GCG / verified against step 5.5).** For suffix position
`i`, the linearised loss change of swapping token `i→v` is `g[i,v] = (∂loss/∂e_i) · W[v,:]`, i.e.
`token_gradient = grad_inputs_embeds[suffix_positions] @ W.T` → `[suffix_len, vocab]`, where `grad_inputs_embeds`
is the step-5.5 `delta.grad` and `W` is the token-embedding matrix. GCG picks, per position, the `top_k` tokens
with the **most negative** `g[i,·]` (largest predicted loss decrease). The seam computes this; the core only
sees the `[L,V]` array (so the core is torch-free and toy-testable).

---

## Tasks

- [ ] **Task 0: Research & Reuse (no code) — port-faithful GCG algorithm + suffix template**

  **Files:** Notes appended to this plan under a `## Task 0 notes` heading (no source yet).

  **What:** Before writing the loop, read the reuse sources and record the *faithful* specifics so the harness
  matches the literature (development-workflow §0): (a) **nanoGCG** (`GraySwanAI/nanoGCG`) — the one-hot top-k +
  candidate-batch loop, default `top_k`/`search_width`, the optional buffer/mellowmax (note as stretch), licence;
  (b) **RoboGCG repo** (`eliotjones1/robogcg`, `experiments/single_step/`) — the **suffix template / placement**
  (where the adversarial tokens sit relative to the instruction), the **target action-token construction**, and
  the single-step config defaults; licence (`[VERIFY]` — not stated in README per gpu-runbook). (c) confirm the
  OpenVLA prompt template `"In: What action should the robot take to {instruction}?\nOut:"` + suffix placement.

  **Test scenarios:** n/a (research). Output = a short, cited notes block; mark anything unverifiable `[VERIFY]`,
  never fabricate. Verify licences before porting any code.

  **Dependencies:** WebFetch / `gh` / Context7; `docs/references/README.md` (RoboGCG facts already verified).

  **Notes:** Reuse the **algorithm**, not the harness (D6-1). If a snippet is ported, attribute it + respect the
  licence (academic-integrity rule).

  **Commit:** `docs: step 6 Task 0 — faithful GCG algorithm + RoboGCG suffix-template research notes`

- [ ] **Task 1: model-free GCG search core**

  **Files:**
  - Create: `src/evasion_tax/attack/gcg.py`
  - Test: `tests/evasion_tax/attack/test_gcg.py`

  **What:** The pure-NumPy GCG loop + the `GcgConfig`/`GcgResult`/`LossGradientFn` contract above. No torch.

  **Interface:**
  - `top_k_candidates(grad: np.ndarray, top_k: int) -> np.ndarray` — `[L,V]`→`[L,top_k]` token ids (most negative).
  - `sample_candidates(cur: np.ndarray, topk: np.ndarray, width: int, rng) -> np.ndarray` — `[B,L]` single-pos swaps.
  - `select_best(losses: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray,float]` — argmin candidate + its loss.
  - `run_gcg(fn: LossGradientFn, cfg: GcgConfig, *, reached_fn=None) -> GcgResult` — the loop (keep best so far).

  **Test scenarios:** loss_history is **non-increasing**; on a toy `LossGradientFn` (loss = Hamming distance to a
  secret suffix; gradient points to the secret) `run_gcg` converges to the secret within N steps and sets
  `reached=True`; `top_k_candidates` returns the lowest-gradient tokens per position; `sample_candidates`
  respects `search_width` and only mutates one position per candidate; identical `seed` ⇒ identical result
  (deterministic); invalid config (`top_k>vocab`, `search_width<1`, `suffix_len<1`) raises `ValueError`; the core
  imports **no** torch (assert the module has no torch dependency).

  **Dependencies:** `numpy`, `evasion_tax.repro.seeds`.

  **Notes:** Single-position swap MVP (D6 stretch list). Keep `best_suffix_ids` monotone (never regress on a step
  where every candidate is worse — keep the incumbent).

  **Commit:** `feat: model-free GCG search core (one-hot top-k + candidate-batch loop, TDD)`

- [ ] **Task 2: OpenVLA loss/gradient seam (`LossGradientFn` for the real model) — GPU-only**

  **Files:**
  - Create: `src/evasion_tax/attack/gcg_openvla.py`
  - Test: `tests/evasion_tax/attack/test_gcg_openvla.py` (the **pure** pieces only; real forward is GPU-guarded)

  **What:** `OpenVlaGcgTarget` implementing `LossGradientFn` against the **frozen bf16** OpenVLA-7B, reusing the
  step-5.5 embedding-hook + CE-to-target-action-tokens loss. `token_gradient` = the hook's `delta.grad` at the
  **suffix token span**, projected through the embedding matrix → `[L,V]` (the contract math above). `loss_of`
  batches B candidate suffixes through the frozen forward (no grad) → `[B]` CE losses.

  **Interface:**
  - `OpenVlaGcgTarget(model, processor, *, image, instruction, suffix_span, target_action_ids, device)`.
  - implements `vocab_size`, `init_suffix_ids`, `token_gradient`, `loss_of` (the Task-1 Protocol).
  - pure helpers (unit-testable off-GPU): `suffix_span_in_ids(prompt_ids, suffix_len) -> slice`;
    `project_onehot_grad(grad_embeds_suffix: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray` (`[L,d]@[V,d].T`).

  **Test scenarios (model-free parts):** `project_onehot_grad` matches a hand-computed `[L,V]` for a tiny fake
  `W`/grad; `suffix_span_in_ids` locates the right token range for a known tokenisation; the real
  forward/gradient path **guards** off-GPU (`cuda_available()` false ⇒ the script/driver exits 2, no torch import
  at module top). Real-model numerics are validated by the Task-3 tiny run on the box.

  **Dependencies (GPU):** torch, transformers (`AutoModelForVision2Seq`/`AutoProcessor`), PIL — imported **inside**
  the guarded path, never at module top (so the module stays importable on the mac, like the smoke scripts).

  **Notes:** Freeze every weight (`requires_grad_(False)`) — faithful (GCG never differentiates weights) and keeps
  peak VRAM representative (step-5.5: ≈15.5 GiB, ≈8 GiB headroom for candidate batching). `loss_of` must run under
  `torch.no_grad()` and reset/track peak VRAM so Task 4 can read it.

  **Commit:** `feat: OpenVLA GCG loss/gradient seam (one-hot grad via step-5.5 hook; GPU-guarded, TDD pure parts)`

- [ ] **Task 3: tiny-run driver (harness works) — on the box**

  **Files:**
  - Create: `scripts/smoke_gcg_tiny.py`
  - (no new tests — the verify gate is the run itself, like the other `smoke_*` scripts)

  **What:** 1 task, 1 fixed target (`_TARGET_BINS`→codec→`TargetActionSpec`), a handful of GCG steps. Load
  frozen bf16 OpenVLA (reuse `smoke_openvla_gradient.py` setup), build `OpenVlaGcgTarget`, run `run_gcg`, print
  loss trajectory + whether the target tokens are argmax. **Quarantine the optimised suffix** →
  `artifacts/untrusted/` (D6-6). Log a non-registered smoke record → `results/_smoke/`.

  **Test scenarios (verify gate):** harness runs to completion; `loss_history` strictly decreases and ends well
  below start; `reached=True` (target tokens argmax) **or** loss < pinned threshold; peak VRAM < 24 GiB
  (fits one card); suffix written to `artifacts/untrusted/`, nothing untrusted committed. Off-GPU ⇒ guard exits 2.

  **Dependencies:** Tasks 1+2; `evasion_tax.policy.action_codec`, `records.TargetActionSpec`, `repro.RunLogger`,
  `config` guard.

  **Notes:** Mirrors `smoke_openvla_gradient.py` structure (guard → heavy imports → run → write-once log → PASS/
  FAIL). If it won't converge in a handful of steps, raise `n_steps`/`search_width` (a measurement, not a code
  change) before reaching for a nanoGCG refinement.

  **Commit:** `feat: CSB step-6 tiny GCG run — harness drives target action tokens on one example (smoke)`

- [ ] **Task 4: D8 timing micro-bench — fill `scripts/microbench_gcg.py` — on the box (REGISTERED)**

  **Files:**
  - Modify: `scripts/microbench_gcg.py` (replace the `NotImplementedError` body)
  - Create: `tests/.../test_microbench_gcg.py` (the model-free aggregation/sweep logic only)

  **What:** Over a few targets at one **pinned** GCG config, time the harness and probe memory: median
  **`s/target`**, **peak VRAM**, **max candidate-batch B at 24 GB** (sweep B upward, catch OOM, record the
  largest that fits), steps-to-success distribution. Write a **registered** result to **write-once `results/`**
  with the full §8 repro header (env capture, pinned seed, **which A5000**, bf16, CUDA/driver/torch, git commit).

  **Interface (pure, testable off-GPU):**
  - `summarise_timings(per_target_seconds: list[float]) -> dict` (median/IQR/n).
  - `max_batch_that_fits(probe_fn, start, cap) -> int` (doubling/bisection sweep; `probe_fn` raises on OOM).

  **Test scenarios:** `summarise_timings` median/IQR correct on a fixed list; `max_batch_that_fits` returns the
  largest non-OOM B against a fake `probe_fn` whose OOM boundary is known; the run dict contains every §8 header
  field; config validates; off-GPU ⇒ guard exits 2 (unchanged). **DEFERRED items logged, not dropped (D6-4):**
  the record explicitly carries `l1_extraction_overhead: "deferred (L1 not on GPU)"` and
  `adaptive_gcg_cost: "deferred"` so the gap is visible, never an implicit "covered everything".

  **Dependencies:** Tasks 1+2; `repro.RunLogger`/`env_capture`, `config.load_config`.

  **Notes:** Secure a **quiet/exclusive** window on the shared desktop for stable timing (pc-spec.md; gpu-runbook
  L3) — note it in the run log. One variable at a time; this is a real result → full §8 discipline.

  **Commit:** `feat: D8 GCG timing micro-bench — s/target, peak VRAM, max candidate-batch @ 24GB (registered)`

- [ ] **Task 5: branch-selection computation + decision**

  **Files:**
  - Create: `src/evasion_tax/eval/branch_select.py`
  - Test: `tests/evasion_tax/eval/test_branch_select.py`
  - Modify (after the box run, with the numbers): `docs/core/execution-playbook.md` (§1, §2 lock branch, §6
    D4/D7/D8 → RESOLVED, §7 M1, §10 decision log), `docs/gpu/CSB/plan.md` (tick step 6 + add a step-6 how-to).

  **What:** Model-free arithmetic turning the measured `s/target` into the affordable matrix and the
  **pre-registered** branch (§2 *Compute branches*; D6-5). Then record the selected branch + numbers in the docs.

  **Interface:**
  - `affordable_matrix(s_per_target: float, calendar_seconds: float, *, seeds: int, per_target_overhead: float,
    adaptive_mult: float) -> AffordableMatrix` (n_targets / n_tasks the budget supports).
  - `select_branch(matrix: AffordableMatrix, *, thresholds, borderline_frac=0.2) -> Literal["N","N-","F"]`
    (conservative on borderline, D6-5).

  **Test scenarios:** matrix size = `floor(budget / (s_per_target * adaptive_mult + overhead))` on fixed inputs;
  the three branch thresholds map to N / N− / F correctly; a value within `borderline_frac` of a boundary selects
  the **more conservative** branch; missing/zero `s_per_target` raises (no silent default). The doc edits are not
  unit-tested — they record the decision per playbook §11.

  **Dependencies:** Task 4's logged `s/target`; pure NumPy.

  **Notes:** **M3/H6-A is delivered in every branch** — branch selection only sizes M4 (the H6-D tax), never the
  floor. Log the branch decision to §10 with the numbers (Gate 5 / D8); this is the load-bearing M1 deliverable.

  **Commit:** `feat: D8 branch selector + record selected Branch N/N−/F (playbook §1/§2/§10, CSB plan.md step 6)`

---

## Build order & where each runs (the model-free / GPU split)

1. **Now, on the mac (TDD, `/tdd`):** Task 0 (research) → Task 1 (core) → Task 2 *pure parts* → Task 4 *pure
   parts* → Task 5 *core* + tests. Full `src/evasion_tax` stays type-clean + ruff-clean; suite stays green.
2. **On the CSB A5000 box (one session):** Task 2 real seam validated by Task 3 (tiny run) → Task 4 (registered
   micro-bench, quiet window) → Task 5 *decision* (compute the branch, write the docs, tick step 6).

**Verify gates (plan.md step 6):** (a) the GCG harness runs end-to-end; (b) `s/target`, peak VRAM, and max
candidate-batch @ 24 GB are recorded to write-once `results/` with a §8 header; (c) **Branch N / N− / F is
selected (D8)** and written into the playbook. Then **M1's GO/NO-GO (H1)** is the next milestone gate — out of
step-6 scope.

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
- **D6-3 — Tiny-run gate = WIRING faithfulness (pass/fail); attack-effect is EXPLORATORY (Codex #2).** Monotone
  incumbent loss is just GCG's keep-best bookkeeping, and target-argmax on one arbitrary target can fail for an
  unlucky prompt/target/`search_width` — so neither is a faithful pass/fail signal. The tiny run's **pass/fail**
  is the **wiring** set: correct tensor shapes; the suffix token span decodes to the intended adversarial text
  under the **real** processor/tokenizer; `loss_of` on a batch equals per-candidate single evaluation
  (batched-loss equivalence); and the projected one-hot gradient **sign** agrees with measured one-token-swap loss
  deltas (the Task-2 finite-difference gate, D6-9). Target-action argmax / loss-below-pinned on one example is
  recorded as an **exploratory** smoke observation, **not** a gate. It is **NOT** an ASR claim and **NOT** a
  closed-loop rollout. Any tiny-run config escalation (`n_steps`/`search_width`) is **pre-registered before
  running**, never tuned reactively. `results/_smoke/` (non-registered bring-up smoke).
- **D6-4 — Micro-bench scope = `s/target` + peak VRAM + max candidate-batch B @ 24 GB, on our harness, registered.**
  A few targets, one pinned GCG config; sweep B upward to the largest that fits 24 GB; record median `s/target`,
  peak VRAM, max B, steps-to-success distribution → write-once `results/` with the full §8 repro header
  (this is a **real result**, plan.md "step 6 produces registered measurements"). Measured under a **clean-process
  protocol (D6-10)** so an OOM during the batch sweep cannot fragment the allocator and contaminate the registered
  timing. The other M1-micro-bench items — **L1 activation/attention extraction overhead** and
  **adaptive-GCG-against-the-probe-score** cost (D7/#5) — are **DEFERRED**: the L1 probe is not yet on GPU (M2/M3),
  so they are **named here, not silently dropped**, and measured when L1 lands. LIBERO rollout throughput was
  already observed at step 4 (90 steps / episode, 14.5 GiB).
- **D6-5 — Step 6 yields a PROVISIONAL capacity estimate; the branch is NOT locked here (Codex #1).** The H6-D
  cross-layer tax (the headline) is sized by the **adaptive** GCG-against-the-probe cost, which **cannot be
  measured until the L1 probe is on GPU** (D6-4). So step 6 computes a **provisional** affordable matrix from the
  measured **non-adaptive** `s/target` and an **explicitly-labelled estimate** `adaptive_mult × s/target`
  (`adaptive_mult` pre-registered from the GCG-step-count literature / RoboGCG's 30–110 steps-to-success). The
  branch decision is **left UNLOCKED**: step 6 records a **provisional branch + a hard-F default** — *unless and
  until the later adaptive-probe bench (M1/M2, same GPU seam) confirms N/N−, the committed branch is **F**
  (oracle frontier, H6-A)*. Borderline (within ~20% of a boundary) → the more conservative branch. The provisional
  estimate + the lock condition are logged to playbook §1/§2/§10; D8/Gate 5 stays **OPEN**; CSB plan.md step 6 is
  ticked as *harness + non-adaptive bench done, branch provisional*. **M3/H6-A is delivered in every branch**
  regardless (floor unchanged).
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
- **D6-9 — GPU seam faithfulness gate BEFORE the tiny run (Codex #3).** The one-hot gradient
  (`grad_inputs_embeds[suffix]` @ `W.T`) is only unit-tested on fake arrays; a wrong suffix span, tokenizer/template
  offset, embedding dtype/cast, tied-embedding access, or a sign-convention slip all yield *plausible-but-wrong*
  `[L,V]` gradients that would silently time an **unfaithful** search loop. Task 2 therefore adds a **real-model**
  validation gate: for sampled (position, token) swaps, the projected gradient's **ranking/sign** must agree with
  the **measured** one-token-swap loss delta (finite-difference), and the suffix span must **decode to the intended
  adversarial text** under the real processor. This gate runs (on the box) **before** the tiny run and the bench;
  if it fails, fix the seam before any timing — an unfaithful seam must not produce a branch-defining number.
- **D6-10 — Clean-process micro-bench protocol (Codex #4).** A CUDA OOM fragments the allocator and pollutes the
  subsequent peak-VRAM / timing of the same process. So the batch sweep runs **each `B` probe in a fresh
  subprocess** (OOM recorded separately, never swallowed), and the **final `s/target` is timed at the selected
  `B` in a clean process** on an **exclusive** GPU; record variance over repeats and **fail the run** if the GPU
  is not exclusive or the numbers are not reproducible. The registered branch-defining number must come from an
  uncontaminated process.

---

## Scope boundary (do NOT exceed — this is plan.md step 6's verify gate)

**IN:** our GCG suffix-search harness (model-free core + GPU seam) · a tiny run that proves the harness works ·
the D8 `s/target`/VRAM/max-batch micro-bench · the branch-selection computation + decision log.

**OUT (explicitly deferred / other milestones):**
- **Closed-loop *attacked* LIBERO rollouts** and **H1 benign-vs-attacked separation / AUC** → M1 §7 (needs the
  attacked class fed through a rollout; step 6 stops at the optimised suffix + its argmax action).
- **Scoring attacked output through metric-A / any detector** → M1/M2 (and gated on D6-8 / D-3 lock).
- **Upstream RoboGCG faithful reproduction** (own env, published >90% ASR) → M1 cross-check (gpu-runbook Step 4).
- **Adaptive-GCG-against-the-probe** and **L1 extraction overhead** micro-bench items, **and LOCKING the compute
  branch** → when L1 lands on GPU; step 6 yields only a *provisional* branch + hard-F default (D6-4/D6-5).
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

- [x] **Task 0: Research & Reuse (no code) — port-faithful GCG algorithm + suffix template**

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

- [x] **Task 1: model-free GCG search core**

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

- [x] **Task 2: OpenVLA loss/gradient seam (`LossGradientFn` for the real model) — GPU-only**
  <br>_(mac: pure helpers `project_onehot_grad` / `suffix_span_in_ids` + no-torch-at-top guard TDD'd; GPU
  `OpenVlaGcgTarget` + D6-9 faithfulness gate written guarded, faithful to step 5.5 — runs/validated on the box.)_

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
  - **GPU faithfulness gate (D6-9):** `gradient_agrees_with_swaps(target, *, n_samples, rng) -> FaithfulnessReport`
    — samples (position, token) swaps and checks the projected one-hot gradient's predicted **ranking/sign**
    against the **measured** `loss_of` delta of the actual swap (finite-difference); `decode_span(target) -> str`
    asserts the suffix span maps to the intended adversarial text under the real processor.

  **Test scenarios (model-free parts):** `project_onehot_grad` matches a hand-computed `[L,V]` for a tiny fake
  `W`/grad; `suffix_span_in_ids` locates the right token range for a known tokenisation; the real
  forward/gradient path **guards** off-GPU (`cuda_available()` false ⇒ the script/driver exits 2, no torch import
  at module top). **On the box (D6-9 gate, BEFORE the tiny run):** `gradient_agrees_with_swaps` shows the projected
  gradient's sign/ranking predicts measured one-token-swap loss deltas above chance, and `decode_span` returns the
  intended adversarial text — the gate that the seam (span, dtype, tied embeddings, sign convention) is *faithful*,
  not merely plausible.

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

  **Test scenarios (verify gate — WIRING is pass/fail, attack-effect is EXPLORATORY; D6-3/D6-9):** the Task-2
  faithfulness gate passed first (gradient sign/ranking vs one-token-swap deltas; span decodes to the intended
  text); harness runs to completion; `loss_of` batched == per-candidate (batched-loss equivalence); peak VRAM
  < 24 GiB (fits one card); the suffix is written to `artifacts/untrusted/` and **nothing untrusted is committed**.
  **Exploratory (recorded, NOT pass/fail):** the loss trajectory and whether the target tokens reach argmax /
  loss < pinned threshold on the single example. Off-GPU ⇒ guard exits 2.

  **Dependencies:** Tasks 1+2; `evasion_tax.policy.action_codec`, `records.TargetActionSpec`, `repro.RunLogger`,
  `config` guard.

  **Notes:** Mirrors `smoke_openvla_gradient.py` structure (guard → heavy imports → run → write-once log → PASS/
  FAIL). Convergence of the exploratory attack-effect is **not** the gate (D6-3). Any escalation of
  `n_steps`/`search_width` is **pre-registered in this plan before the run** (a logged measurement decision),
  never a reactive tweak; a nanoGCG refinement is the pre-registered stretch (scope list).

  **Commit:** `feat: CSB step-6 tiny GCG run — harness drives target action tokens on one example (smoke)`

- [ ] **Task 4: D8 timing micro-bench — fill `scripts/microbench_gcg.py` — on the box (REGISTERED)**

  **Files:**
  - Modify: `scripts/microbench_gcg.py` (replace the `NotImplementedError` body)
  - Create: `tests/.../test_microbench_gcg.py` (the model-free aggregation/sweep logic only)

  **What:** Over a few targets at one **pinned** GCG config, time the harness and probe memory: median
  **`s/target`**, **peak VRAM**, **max candidate-batch B at 24 GB**, steps-to-success distribution. **Clean-process
  protocol (D6-10):** the batch sweep runs **each `B` probe in a fresh subprocess** (OOM recorded separately, never
  swallowed); the **final `s/target` is timed at the selected `B` in a clean process** on an **exclusive** GPU;
  record variance over repeats. Write a **registered** result to **write-once `results/`** with the full §8 repro
  header (env capture, pinned seed, **which A5000**, bf16, CUDA/driver/torch, git commit) — and **fail the run** if
  the GPU is not exclusive or `s/target` is not reproducible across repeats.

  **Interface (pure, testable off-GPU):**
  - `summarise_timings(per_target_seconds: list[float]) -> dict` (median/IQR/n + a reproducibility check across repeats).
  - `max_batch_that_fits(probe_fn, start, cap) -> int` (doubling/bisection; `probe_fn` runs **one B in a fresh
    subprocess** and raises on OOM — so allocator fragmentation cannot leak across probes, D6-10).

  **Test scenarios:** `summarise_timings` median/IQR correct on a fixed list **and flags a non-reproducible spread
  across repeats**; `max_batch_that_fits` returns the largest non-OOM B against a fake `probe_fn` whose OOM
  boundary is known **and never reuses a process across probes**; the run dict contains every §8 header field
  **plus `branch_status: "provisional"` + the adaptive-bench lock condition (D6-5)**; config validates; off-GPU ⇒
  guard exits 2 (unchanged). **DEFERRED items logged, not dropped (D6-4):** the record explicitly carries
  `l1_extraction_overhead: "deferred (L1 not on GPU)"` and `adaptive_gcg_cost: "deferred"` so the gap is visible,
  never an implicit "covered everything".

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

  **What:** Model-free arithmetic turning the measured **non-adaptive** `s/target` into a **provisional**
  affordable matrix and a **provisional branch with a hard-F default** (§2 *Compute branches*; D6-5 — the branch
  is **NOT locked** until the later adaptive-probe bench). Then record the provisional estimate + the lock
  condition in the docs.

  **Interface:**
  - `affordable_matrix(s_per_target: float, calendar_seconds: float, *, seeds: int, per_target_overhead: float,
    adaptive_mult: float) -> AffordableMatrix` (n_targets / n_tasks the budget supports; `adaptive_mult` is the
    **estimated** multiplier, flagged estimate-not-measured in the result).
  - `provisional_branch(matrix: AffordableMatrix, *, thresholds, borderline_frac=0.2, adaptive_measured=False)
    -> BranchDecision` — returns `{branch, locked: False, default_if_unconfirmed: "F", lock_condition}`;
    conservative on borderline (D6-5). It **refuses to return `locked=True`** while `adaptive_measured` is False.

  **Test scenarios:** matrix size = `floor(budget / (s_per_target * adaptive_mult + overhead))` on fixed inputs;
  the three thresholds map to N / N− / F correctly; within `borderline_frac` of a boundary → the **more
  conservative** branch; missing/zero `s_per_target` raises (no silent default); **`provisional_branch` never
  returns `locked=True` when `adaptive_measured=False`, and always carries `default_if_unconfirmed="F"`** (D6-5).
  The doc edits are not unit-tested — they record the provisional decision + lock condition per playbook §11.

  **Dependencies:** Task 4's logged `s/target`; pure NumPy.

  **Notes:** **M3/H6-A is delivered in every branch** — branch selection only sizes M4 (the H6-D tax), never the
  floor. Log the **provisional** branch + the hard-F default + the lock condition to §10 with the numbers
  (Gate 5 / D8 stays **OPEN** until the adaptive bench confirms); the **locked** branch is a later M1/M2 deliverable.

  **Commit:** `feat: D8 branch selector + record selected Branch N/N−/F (playbook §1/§2/§10, CSB plan.md step 6)`

---

## Build order & where each runs (the model-free / GPU split)

1. **Now, on the mac (TDD, `/tdd`):** Task 0 (research) → Task 1 (core) → Task 2 *pure parts* → Task 4 *pure
   parts* → Task 5 *core* + tests. Full `src/evasion_tax` stays type-clean + ruff-clean; suite stays green.
2. **On the CSB A5000 box (one session):** Task 2 **seam-faithfulness gate** (D6-9, finite-difference vs token
   swaps) → Task 3 tiny run (wiring gate, D6-3) → Task 4 registered micro-bench (clean-process, quiet/exclusive
   window, D6-10) → Task 5 *provisional* branch + hard-F default, write the docs, tick step 6 as branch-provisional.

**Verify gates (plan.md step 6):** (a) the **seam-faithfulness gate** passes (D6-9) and the GCG harness runs
end-to-end (wiring gate, D6-3); (b) `s/target`, peak VRAM, and max candidate-batch @ 24 GB are recorded to
write-once `results/` with a §8 header under the **clean-process protocol** (D6-10); (c) a **provisional Branch
N / N− / F + hard-F default + lock condition** is written into the playbook (the branch is **locked later**, once
the adaptive-probe bench runs — D6-5; D8/Gate 5 stays OPEN). Then **M1's GO/NO-GO (H1)** is the next milestone
gate — out of step-6 scope.

---

## Task 0 notes — faithful GCG algorithm + RoboGCG suffix-template research (2026-06-19)

> Reuse the **algorithm**, not the harness (D6-1). Sources read directly (raw GitHub + arXiv facts already
> verified in `docs/references/README.md` §RoboGCG). Anything not directly read is marked **[VERIFY]** — never
> fabricated (academic-integrity rule).

### A. nanoGCG — `GraySwanAI/nanoGCG`, `nanogcg/gcg.py` (read 2026-06-19)

- **Licence: MIT** — `LICENSE` reads "Copyright (c) 2024 Gray Swan AI". **Porting the algorithm is permitted**
  with attribution (record it in the ported module docstring).
- **Defaults** (`GCGConfig`): `num_steps=250`, `search_width=512`, `topk=256`, `n_replace=1`, `buffer_size=0`,
  `use_mellowmax=False`. → our MVP pins **single-position swap (`n_replace=1`)**, **`top_k=256`**; `n_steps` /
  `search_width` are per-run knobs (tiny run small; bench a pinned budget). Buffer + mellowmax = **pre-registered
  stretch** (scope list), not in the MVP.
- **One-hot token gradient** (the load-bearing math, matches our `gcg.py` contract):
  `optim_ids_onehot = one_hot(optim_ids, num_classes=num_embeddings)` → `onehot @ embedding_weights`
  `(1, n_optim, vocab) @ (vocab, embed_dim) → (1, n_optim, embed_dim)` → backprop → `optim_ids_onehot_grad`
  `[1, n_optim, vocab]`. This is **identical** to our seam's `grad_inputs_embeds[suffix] @ W.T` (`token_gradient`
  contract, plan "One-hot gradient" block): both give `g[i,v] = (∂loss/∂e_i)·W[v,:]`.
- **Candidate sampling (per step):** `sampled_ids_pos = argsort(rand((search_width, n_optim)))[..., :n_replace]`
  (random position(s) per candidate); `topk_ids = (-grad).topk(topk, dim=1).indices` (per position); each chosen
  position gets a **uniform random** token from its top-k. → our `sample_candidates` = single random position +
  random top-k token per candidate.
- **Select / incumbent:** `optim_ids = sampled_ids[loss.argmin()]` — argmin candidate loss; the buffer (when
  `buffer_size>0`) keeps the best-k. Our MVP keeps a **single incumbent** (`best_suffix_ids` monotone, Task 1).
- **Sign convention (verified):** `(-grad).topk(...)` ⇒ GCG picks the tokens with the **most negative** gradient
  (largest predicted loss **decrease**). → our `top_k_candidates` selects **most-negative** `g[i,·]` (plan contract).

### B. RoboGCG — paper `arXiv:2506.03350` (facts already verified) + repo `eliotjones1/robogcg` (read 2026-06-19)

- **Attack shape (paper §4, verified in `docs/references/README.md`):** white-box GCG-optimised **textual** suffix,
  applied **once at rollout start**, eliciting an attacker-chosen low-level action. 7-DoF × 256 bins → **256^7**
  actions. **Single-step** attack; matches found in **30–110 GCG steps** (~3–10 min/success on H100). Target is an
  **arbitrary low-level action** (control authority), **not** semantic harm — matches D6-6/D6-7.
- **Repo licence: NOT stated.** `api.github.com/.../git/trees/main?recursive=1` shows **no `LICENSE` file at root**
  → **[VERIFY]** before porting *any* repo code (confirms the gpu-runbook `[VERIFY]`). We did **not** port repo
  code — only the (MIT) nanoGCG algorithm + the paper's published method — so this does not block step 6.
- **OpenVLA prompt template (read from `experiments/single_step/model_utils.py`):**
  `"In: What action should the robot take to {task_description}?\nOut: "`. This **matches**
  `scripts/smoke_openvla_gradient.py:_PROMPT_TEMPLATE` (`"In: What action should the robot take to {instruction}?\nOut:"`)
  except RoboGCG has a **trailing space after `Out:`** — a one-token offset to **[VERIFY on the box]** against the
  real processor (the Task-2 `decode_span` faithfulness check covers exactly this).
- **Action tokeniser:** `ActionTokenizer(32000, 256, norm_stats)` in `model_utils.py` → vocab base **32000**, **256**
  bins (consistent with `action_codec.py` and the smoke script's `vocab_size - 1 - bin` id formula).
- **Exact suffix placement (offset of the adversarial tokens relative to `{task_description}`):** **[VERIFY on the
  box]** — lives in `experiments/single_step/run_experiment.py` / `experiment_runner.py`, **not read here**
  (WebFetch surfaced only the file tree + helpers). Standard GCG appends the suffix **after the request text**; for
  OpenVLA the natural placement is after `{task_description}`, before `?\nOut:`. This is a **GPU-seam (Task 2/3)**
  concern — the model-free core (Task 1) only sees a `suffix_span` slice — so it is named, not blocking; the Task-2
  faithfulness gate (D6-9, `suffix_span_in_ids` + `decode_span`) pins the real offset on the box.

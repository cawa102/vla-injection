# CSB (`ecs3-0202`) — What to Run Here (registered-compute runbook)

> **Status (2026-06-16, author decision):** `ecs3-0202` (2× RTX A5000, 24 GB each) is the project's
> **registered compute**. The EET experiments — bf16 OpenVLA-7B, real LIBERO rollouts, the GCG attack, the D8
> timing micro-bench, M1–M4 — are **measured here**, logged to write-once `results/`. **Kelvin2** is now a
> **backup contingency only** ([`../Overview.md`](../Overview.md), [`../Running.md`](../Running.md)); cross-HW
> comparison is forbidden within a claim, so **all registered runs commit to this A5000 box**. Hardware:
> [`pc-spec.md`](./pc-spec.md). Theme: the Embodiment Evasion Tax (`docs/core/execution-playbook.md`).

## Why this box

It is the **first GPU we can actually access** (Kelvin2 login was never established) **and** it clears the bar:
24 GB/card runs the **registered bf16 model** (~14 GB) with rollout headroom, so unlike the old 8 GB candidate
it produces **citable results**, not just wiring checks. The integration risk —
**OpenVLA → LIBERO → goal-action detector (L2) → GCG** — is real; here we both de-risk it *and* run the
registered matrix on the same hardware.

## Scope (what changed vs the old 8 GB box)

The old hard walls — *4-bit only*, *bf16 won't fit*, *no science numbers*, *full GCG can't run* — are **gone**.
What replaces them are honest **A5000-vs-A100/H100** caveats, not capability walls:

- **bf16 is the registered precision and it fits** → no quantization confound; the L2 FP-calibration and the L1
  internal-probe are measured at the registered precision.
- **One card is the registered card.** A registered run uses a **single** A5000 (log which one). The second card
  is for **memory relief** (model/tensor parallel, PCIe — no NVLink) **or** for running an independent job, **not**
  for splitting one claim across two cards. GPU 1 also drives the display.
- **A5000 ≠ A100/H100.** It is roughly **2.5–4× slower** (bf16 throughput + memory bandwidth). The **published
  H100 GCG timings (~185–604 s/target) do NOT transfer** — there is no published A5000 OpenVLA-GCG prior, so the
  **M1 micro-bench is the sole source** of `s/target`. Expect a slower matrix → the D8 branch is more likely
  **N− / F**; this is **measured at M1**, not assumed.
- **Shared lab desktop.** Others may log in; the GUI lives on GPU 1. The D8 **timing** micro-bench needs a
  quiet/exclusive window for stable numbers (see [`pc-spec.md`](./pc-spec.md) *Not in `nvidia-smi`*).

## Environment

- Native **Linux + CUDA** (no WSL2). OpenVLA / bitsandbytes / MuJoCo·robosuite are Linux+CUDA-first.
- **Headless rendering** for LIBERO camera obs: `MUJOCO_GL=egl` on the NVIDIA GPU (verify it initialises).
- Keep the env **Python 3.10** to match the project pin (`configs/env/requirements-gpu.txt`,
  [`../Start.md`](../Start.md) §3).
- **Driver 595 / CUDA 13.2 is newer than the OpenVLA pin** (torch 2.2.0 = cu121). The newer driver is
  backward-compatible (runs cu121 torch); verify `flash-attn==2.5.5` loads — wheel, rebuild, or a small torch
  bump (record whatever is used in the run env, invariant #8).
- Access: **VS Code Remote Tunnel (`code tunnel`, outbound)** is the working path — full runbook in
  [`ssh.md`](./ssh.md). Direct inbound SSH (port **2222**) is currently **firewall-blocked** and we have no
  sudo; the tunnel is IP-independent, so the box's DHCP address does not matter.

## Bring-up ladder (do in order; each step has a verify gate)

> Status: `[ ]` TODO · `[x]` DONE · in-progress/blocked = `[ ]` + inline `🔄` / `⛔ <reason>`. Tick each box as its verify gate passes.

- [x] **1. Linux+CUDA env, `git clone`, install GPU deps** (`requirements-gpu.txt`) — *verify:* `torch.cuda.is_available() == True`; `nvidia-smi` sees both A5000s — *2026-06-17 box ✓: Python 3.10, torch 2.2.0+cu121, numpy 1.26.4, CUDA True, both RTX A5000 visible, torch↔numpy interop OK*
- [x] **2. Run the existing model-free test suite (395 tests)** — *verify:* parity with the Mac (same pass count) — *2026-06-17 box: all green (after restoring `.git` via real clone — see `ssh.md` §5/§6)*
- [x] **3. Load OpenVLA-7B in bf16**, one forward on a dummy image+instruction — *verify:* a valid action vector; **fits on one 24 GB card** (the registered precision runs) — *2026-06-17 box ✓: `openvla/openvla-7b` bf16+sdpa on cuda:0, valid 7-DoF action, peak VRAM 14.46 GiB reserved / 23.5 GiB (no flash-attn); `scripts/smoke_openvla_load.py`, commit `a24b77a`/`87e9a3f`*
- [x] **4. One LIBERO episode** (EGL) with the bf16 policy — *verify:* rollout completes; log schema matches the state-adapter / metric side — *2026-06-18 box ✓: `libero_spatial` task-0 episode **completed, 90 policy steps, success=True**, sdpa load, **peak VRAM 14.50 GiB / 23.5 GiB (fits one card)**; logged `RolloutStep` schema → `results/_smoke/2026-06-18T14-21-51Z-libero-episode-smoke`. EGL (`MUJOCO_GL=egl`) initialised + rendered on the A5000 → headless-render risk retired. Two install gotchas + the `--unnorm-key` finding in the Step 4 how-to below.*
- [x] **5. Attach the goal-action detector (L2)** to that real rollout — *verify:* detector ingests real OpenVLA actions end-to-end — *2026-06-18 ✓ (offline on the mac, model-free, from the committed step-4 run dir — no box session needed): `scripts/attach_l2_to_rollout.py` + `src/evasion_tax/eval/rollout_io.py` (JSON→`Rollout` seam + D-5 provenance binding). Provenance validated (steps_sha256 `0deaf431…`), **state half** (90 per-step metric-A scores finite in [0,1], decision emitted) + **action half** ((90,7) finite, non-degenerate, D2 path exercised) → `results/_smoke/2026-06-18T15-23-29Z-l2-attach/l2_attach_report.json`. 27 new TDD tests (437 suite green). Wiring de-risk only — NO separation/calibration/deployable claim; benign metric-A scores NOT near zero is expected (the `engagement_radius`/`grasp_radius` placeholders don't match the real scene scale — report-only D-3 calibration input, NOT a re-pin).*
- [x] **5.5 GCG gradient prerequisite** (on the box, before step 6) — bf16 + flash_attn2 `.backward()` reaches the **input embeddings** — *verify:* the input-embedding gradient is **finite and non-zero** and peak VRAM fits one card — *2026-06-19 box ✓: `openvla/openvla-7b` bf16, **finite non-zero** input-embedding grad through **flash_attention_2** (‖g‖=97.2, prompt-region 76.0, max|g|=26.9; loss=18.45 to the target action tokens, grad shape [1,26,4096]), **peak VRAM 15.49 GiB / 23.5 GiB → fits one card** — frozen weights → only ~1 GiB over the step-3 forward, so a GCG backward step is cheap (≈8 GiB headroom for candidate batching at step 6). **sdpa cross-check matches** (‖g‖=91.7, loss=18.52). `scripts/smoke_openvla_gradient.py`; runs `results/_smoke/2026-06-19T12-57-10Z-openvla-gradient-smoke` (flash) / `…T12-57-30Z…` (sdpa), commit `0300670`. → the GCG "gradients are obtainable" premise **HOLDS**. How-to below.*
- [ ] **6. GCG** — first a tiny run (few steps, 1 example), then the **D8 timing micro-bench** — *verify:* attack harness runs; record `s/target`, peak VRAM, max candidate-batch at 24 GB → **selects Branch N/N−/F (D8)**

Steps 1–5.5 de-risk the wiring; **step 6 produces registered measurements** (D4/D7/D8). If bf16 OOMs at step 3
(it should not at 24 GB) → fall back to memory relief on the 2nd card before any precision change.

### Step 3 how-to (on the box, after steps 1–2)

`scripts/smoke_openvla_load.py` loads OpenVLA-7B in bf16, runs one forward on a dummy image+instruction,
validates the 7-DoF action, and logs peak VRAM to write-once `results/_smoke/`. OpenVLA loads via
`trust_remote_code`, so the model + its custom code are fetched from HF on first run (~14 GB; set `HF_HOME` to a
roomy disk — the box `~` has limited space, see [`pc-spec.md`](./pc-spec.md)):

```bash
# 0) Align the env to the just-pulled lock. huggingface-hub is now pinned <1.0 in
#    pyproject/uv.lock (transformers 4.40.1 needs it; the old open ">=0.20" had resolved
#    to hub 1.17.0 and uv re-applied it on every `uv run`, breaking transformers — gotcha
#    below). --inexact aligns hub to the lock WITHOUT pruning the lock-external GPU stack.
uv sync --inexact

# 1) Add the OpenVLA *inference* deps to the same .venv (lock-external, like torch — ssh.md §6).
#    Pins are the codec-verified OpenVLA set (configs/env/requirements-gpu.txt); the forward-pass
#    subset only (peft/draccus/dlimp/tensorflow are NOT needed to load + predict_action). hub is
#    lock-managed now → do NOT pin it here.
uv pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
  accelerate sentencepiece pillow einops

export HF_HOME=<roomy-disk>/hf            # base model is ~14 GB; default ~/.cache may be too small
uv run python scripts/smoke_openvla_load.py        # base openvla-7b + bridge_orig, attn=sdpa (no flash-attn)
```

With hub pinned in the lock, plain `uv run` keeps hub at 0.36.2 (no `--no-sync` workaround needed).

*Verify gate:* prints a finite **7-DoF** action vector **and** `PASS: … fit one card` (peak VRAM < 24 GiB, no
OOM). If `sdpa` errors on the box, retry `--attn-impl eager`; flash-attn (`--attn-impl flash_attention_2`) is a
**separate** perf check (caveat L5), not required for this gate. The exact installed versions are captured into
the smoke `run.json` (repro header). The full OpenVLA/LIBERO **source** install is **step 4** — this step needs
only the HF model + the inference deps above.

### Step 4 how-to (on the box, after step 3 — LIBERO + the OpenVLA eval helpers)

`scripts/smoke_libero_episode.py` runs one `libero_spatial` episode with the bf16 LIBERO-finetuned policy and
logs the `RolloutStep` schema to `results/_smoke/` (plan: `docs/plans/2026-06-18-libero-episode-bringup.md`).
It reuses OpenVLA's verified eval helpers, so the **source install of both repos is this step**.

```bash
# 0) Clone both repos (codec-verified OpenVLA commit) and install LIBERO source into the .venv.
#    ACTIVATE the .venv first so uv targets it, not the conda (base) env. uv pip env-discovery order is
#    VIRTUAL_ENV > CONDA_PREFIX > .venv — with (base) active, a bare `uv pip install` would land in conda base.
source ~/vla-injection/.venv/bin/activate
cd ~ && git clone https://github.com/openvla/openvla.git && (cd openvla && git checkout c8f03f48)
cd ~ && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd ~/vla-injection
uv pip install -e ~/LIBERO        # see GOTCHA 1 — this alone does NOT make `import libero` work
uv pip install -r ~/openvla/experiments/robot/libero/libero_requirements.txt

# 1) The TF stack the eval-helper import chain needs, with the tfds→tensorflow-metadata cap (GOTCHA 2).
uv pip install "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" "tensorflow-metadata<1.16" "protobuf<5"

# 2) Verify gate (a): the import chain (libero + OpenVLA helpers + prismatic/TF) loads.
export MUJOCO_GL=egl
PYTHONPATH=~/openvla:~/LIBERO ~/vla-injection/.venv/bin/python \
  -c "import libero, experiments.robot.robot_utils; print('helpers OK')"

# 3) Run the episode (verify gates b–d in the step-4 plan).
PYTHONPATH=~/LIBERO uv run --no-sync python scripts/smoke_libero_episode.py --openvla-root ~/openvla
```

**Three step-4 gotchas (all hit 2026-06-18, all resolved):**

1. **`import libero` fails after `uv pip install -e ~/LIBERO`** (`ModuleNotFoundError: No module named 'libero'`,
   even though `uv pip show libero` reports it installed in the `.venv`). LIBERO's top-level `libero/` has **no
   `__init__.py`** — it is a **PEP-420 namespace package**, importable only when the repo root is on `sys.path`.
   pip's *legacy* `-e` adds that root via a `.pth` (so it works); **uv's PEP-660 editable** uses a finder/MAPPING
   that does **not** expose the namespace package → `import libero` finds nothing. **Fix = put `~/LIBERO` on
   `PYTHONPATH`** at both import and run time (the same mechanism we use for `~/openvla`'s `experiments`). The
   editable install is harmless but does nothing useful — `PYTHONPATH=~/LIBERO` is the load-bearing part.
   `smoke_libero_episode.py` only adds `--openvla-root` (openvla), so it **still needs `PYTHONPATH=~/LIBERO`**.
   *(Alt: `touch ~/LIBERO/libero/__init__.py` then reinstall — makes it a regular package, but edits upstream.)*
2. **protobuf `runtime_version` ImportError** deep in the helper import: `robot_utils` → `prismatic` (eager) →
   `dlimp` → `tensorflow_datasets` → `tensorflow_metadata`. `tfds 4.9.3` caps nothing on `tensorflow-metadata`,
   so pip pulled **tfmd 1.21.0**, whose `*_pb2.py` does `from google.protobuf import runtime_version` (protobuf
   ≥5.26) — but the TF-2.15-correct protobuf is **4.25.x**. **Fix = `tensorflow-metadata<1.16` + `protobuf<5`**
   (`tensorflow`/`tfds` were already the right 2.15.0/4.9.3 — recorded in `configs/env/requirements-gpu.txt`).
   TF here is **import-only** (the VLA forward is torch/CUDA) → it runs no ops and steals no A5000 memory; the TF
   startup `I/E/W` logs (oneDNN, cuDNN/cuFFT/cuBLAS "already registered", TF-TRT) are benign (`TF_CPP_MIN_LOG_LEVEL=3` to silence).
3. **`unnorm_key` mismatch** — `predict_action` asserted `The unnorm_key you chose is not in the set of available
   dataset statistics, please choose from: dict_keys(['libero_spatial'])`. The `openvla-7b-finetuned-libero-spatial`
   checkpoint registers its action `norm_stats` under **`libero_spatial`**, NOT `libero_spatial_no_noops` (the
   training-data name we'd assumed). **Fix = `--unnorm-key libero_spatial`** (now the script default; was
   `*_no_noops`). Verify the key against the actual checkpoint, not the dataset name. *(Also benign here: a
   `[Warning]: datasets path .../datasets does not exist!` — the LIBERO demo datasets aren't needed for a policy
   rollout; the first-run `~/.libero/config.yaml` was created with defaults via `N`.)*

### Step 5.5 how-to (on the box, before step 6 — the GCG gradient prerequisite)

`scripts/smoke_openvla_gradient.py` loads OpenVLA-7B in **bf16 + flash_attention_2**, freezes every weight,
drives a CE loss to a target action-token sequence, runs one `.backward()`, and checks the **input-embedding
gradient** is finite and non-zero — the premise the step-6 GCG search rests on (flash-attn's backward kernels
are separate from the step-3 forward, so this seam is unverified until now). Needs the flash-attn wheel from
caveat L5 (prebuilt `cu122torch2.2cxx11abiFALSE-cp310` wheel — no compile, no nvcc).

```bash
export HF_HOME=<roomy-disk>/hf            # base model is ~14 GB; default ~/.cache may be too small
uv run python scripts/smoke_openvla_gradient.py                  # bf16 + flash_attention_2 (default)
uv run python scripts/smoke_openvla_gradient.py --attn-impl sdpa # cross-check without flash-attn
```

*Verify gate:* prints `PASS: finite non-zero input-embedding gradient … fit one card`. The weights are frozen
(GCG never differentiates them), so peak VRAM reflects a real GCG backward step — gradient lives only on the
input, not on ~14 GB of weight grads. The gradient is read via a forward hook that adds a `requires_grad`
`delta` to the token embeddings (OpenVLA's multimodal `forward` owns `input_ids`→embeds and takes no external
`inputs_embeds`), so `delta.grad == d(loss)/d(inputs_embeds)`. Logs write-once to `results/_smoke/`
(non-registered bring-up smoke). If the gradient is all-zero or non-finite, fix it **before** the step-6 attack.

### Step 6 how-to (on the box, after step 5.5 — the registered D8 budget-faithful s/step micro-bench)

The **tiny GCG run** prerequisite is already green (`results/_smoke/2026-06-22T14-32-35Z-gcg-tiny-smoke`,
DB-4 equivalence gate 8/8). What remains is the **registered** D8 micro-bench: `scripts/microbench_gcg.py`
sweeps the max candidate-batch B at 24 GB (→ the HW-adapted `batch_size`, DC-2), times the direct `run_gcg`
`s/target`, and — per the budget-faithful plan (`docs/plans/2026-06-22-step6-d8-robogcg-faithful-config.md`) —
times one `token_gradient` (`t_grad`) + one `loss_of` at `B=eval_batch` (`t_fwd`) to compute the **analytic**
`s/step(sw=512) = t_grad + ⌈512/eval_batch⌉·t_fwd` and `s/target(worst) = s/step·500` (early_stop OFF). It does
**not** run the full 500-step sw=512 attack (eval mini-batching is deferred to M1 / Task D).

**Pre-flight (D6-10 + L3):**
- **Exclusive/quiet window.** GUI lives on GPU 1 → run on **`cuda:0`**; `nvidia-smi` shows cuda:0 idle except
  this job. `--exclusive-gpu` is **mandatory** — the harness refuses to log a number from a shared process *or*
  a non-reproducible timing (`assert_registered_run_valid`).
- **Env already aligned** by steps 3–5.5 (`.venv` + the OpenVLA inference deps + the flash-attn 2.5.5 wheel, L5).
  `export HF_HOME=<roomy-disk>/hf` (~14 GB base model). **No `MUJOCO_GL`** — the micro-bench uses a dummy-image
  target, no LIBERO env.

```bash
export HF_HOME=<roomy-disk>/hf            # ~14 GB base-model cache (as in steps 3 / 5.5)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce 24 GB fragmentation OOMs
cd ~/vla-injection
uv run python scripts/microbench_gcg.py \
  --config configs/example_m2.yaml \
  --device cuda:0 --attn-impl flash_attention_2 \
  --exclusive-gpu \
  --seed 42 \
  --search-width 32 --n-steps 5 --n-targets 3 --batch-cap 64 \
  --faithful-search-width 512 --faithful-num-steps 500 \
  --loop-baseline-s 17.475 --batch-calib-s 11.394 --calib-label n5/W32/1tgt
```

- `--eval-batch` is **omitted on purpose** → `eval_batch` defaults to the **measured max-B** (the HW-adapted
  `batch_size`, DC-2; not RoboGCG's A100/H100 64). `--batch-cap 64` keeps the sweep tight (DC-2 expects max-B in
  ≈32–48; B=64 ~28 GiB OOMs on 24 GB) — raise it only if B=64 unexpectedly fits.
  **Razor-edge caveat (2026-06-23):** the forward-only sweep picks the *largest* B that fits, so the timing
  `loss_of` at that exact max-B has ~0 margin (observed B≈43 → ~23.1 GiB, OOM by ~10 MiB). If it OOMs, pass
  **`--eval-batch 32`** (B=32 peaks 21.3 GiB, ~2 GiB margin) — the sweep still records max-B as the VRAM ceiling,
  and `s/step` just uses ⌈512/32⌉=16 chunks (slightly conservative → biases the branch toward F, the safe default).
- `--search-width 32` is the **direct `run_gcg` cross-check** width (a known-fitting B≈max-B); `--n-steps 5` keeps
  it cheap. The DB-2 ablation flags carry the standing 2026-06-22 calibration (loop 17.475 / true-batch 11.394 →
  `speedup_k≈1.53`) **passed through** for the record — not re-measured this run.
- If `--attn-impl flash_attention_2` errors, retry `--attn-impl sdpa` (the step-5.5 cross-check path). If `uv run`
  prunes the lock-external GPU stack, fall back to `uv run --no-sync` (the step-4 workaround).
- **OOM fix (2026-06-23):** the per-target timing forwards reserved memory the caching allocator held across into
  the next phase. Two passes: (1) `run_gcg`'s `B=search_width` forward OOM'd at `logits.float()` (1.16 GiB) on the
  timing's *fragmented* leftover → `empty_cache()` before `run_gcg` + `expandable_segments:True`; (2) with
  fragmentation gone the sweep then reached a higher max-B (~43) and the timing `loss_of(B=max-B)` OOM'd by ~10 MiB
  because the preceding `token_gradient` (B=1 backward) reservation sat on top → `empty_cache()` **between**
  `token_gradient` and `loss_of` too (commits on `main`). If a razor-edge max-B still OOMs, use `--eval-batch 32`.

*Verify gate:* prints `s/target median …s (n=3, reproducible=True)`, `peak VRAM … GiB`, `max candidate-batch
B=<max-B>`, **and** `budget-faithful (sw=512/ns=500/eval_batch=<max-B>): t_grad=…s t_fwd=…s => s/step=…s,
s/target(worst)=…s [analytic ESTIMATE]`; logs to **write-once `results/…-gcg-microbench/`** with the §8 repro
header. **DC-4 cross-check:** the direct per-step cost (`run_gcg` `s/target` ÷ `steps_run`) at `sw=32` should
≈ `t_grad + t_fwd` when the measured max-B (=`eval_batch`) is also ≈32; if max-B differs, optionally re-run with
`--search-width <max-B>` for an exact one-chunk cross-check. Record the **card** + the **max-B** (=`batch_size`)
+ the **flash-attn version** (invariant #8; auto-captured in `run.json`).

> The budget-faithful `s/target(worst)` then feeds **Task B** `branch_select` (provisional, hard-F default), and
> **Task C** ticks step 6 + lands the DC-1…DC-6 divergence contract + the bf16/selection-regret/`speedup_k`
> findings into `execution-playbook.md` §10 — **after** this run produces the registered number. The attack is
> **RoboGCG-budget-faithful only** (sw=512/ns=500/topk=256/n_replace=1/early_stop); the token filters
> (`allow_non_ascii`/`filter_ids`) + the suffix/mean-CE seam divergences remain open (DC-6) → do **not** call it a
> "RoboGCG-faithful attack" yet.

## After bring-up — the registered matrix runs here

Once steps 1–6 pass, M1–M4 run on this box (M1 GO/NO-GO → M2 floor → M3 H6-A oracle frontier → M4 H6-D tax if
the branch allows). Every registered run logs to **write-once `results/`** with the full repro header (§8 of the
playbook). **Kelvin2 is invoked only if this box becomes unavailable**, and then as a **separately registered**
hardware (no cross-HW mixing).

## Limitations (this environment — real, but not capability walls)

| # | Limitation | Consequence |
|---|-----------|-------------|
| L1 | **A5000, not A100/H100** | ~2.5–4× slower; absolute `s/target` ≠ any published number → the matrix is sized from the **M1 micro-bench on this card**, and the D8 branch skews **N−/F**. |
| L2 | **Published H100 GCG timings do NOT transfer** | no A5000 OpenVLA-GCG prior exists → the M1 micro-bench is the **only** budget source; do not estimate cost from other GPUs. |
| L3 | **Shared lab desktop** | other users + the GUI on GPU 1 → contention makes **timing unstable** unless a quiet/exclusive window is secured (matters for D8). |
| L4 | **Single registered card** | one A5000 per claim; the 2nd card = memory relief / independent job only (no NVLink → PCIe). **No cross-card or cross-HW comparison within a claim.** |
| L5 | **CUDA 13.2 / driver 595 newer than the pin** | ✓ **VERIFIED 2026-06-18:** cu121 torch + `flash-attn 2.5.5` run on driver 595. flash-attn won't *build* here (no nvcc/CUDA_HOME, no sudo) → installed the **prebuilt cu122torch2.2cxx11abiFALSE-cp310 wheel** (no compile); OpenVLA `--attn-impl flash_attention_2` → valid action, peak 14.46 GiB (`configs/env/requirements-gpu.txt` flash-attn note; smoke `results/_smoke/2026-06-18T15-47-27Z-openvla-load-smoke`). |
| L6 | **Not a Kelvin2 mirror** | if Kelvin2 is ever used it is **different HW** → its own separate registration, never merged with A5000 results. |

## Reproducibility (registered runs — full discipline)

Pin seeds; capture `nvidia-smi` + `pip freeze` + git commit + **which card** into the run log; one variable per
run; output to **write-once `results/`** (no overwrite). Bring-up smoke runs go to `results/_smoke/` —
**tracked/committed** (git-shared so cross-box bring-up evidence travels between machines) but **non-registered**
(no full §8 discipline). **Once step 6 is producing the D8/registered numbers they are real results** and follow
the playbook §8 protocol (CLAUDE.md reproducibility).

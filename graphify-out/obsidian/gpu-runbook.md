---
source_file: "docs/setup/gpu-runbook.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# gpu-runbook.md

## Connections
- [[Adversarial-artefact quarantine (artifactsuntrusted)]] - `defined_in` [EXTRACTED]
- [[GCGL1 micro-bench (branch selection, D8)]] - `defined_in` [EXTRACTED]
- [[GPU Day-1 Runbook (Kelvin2  M1 gate)]] - `defined_in` [EXTRACTED]
- [[Per-run reproducibility protocol]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/setup/gpu-runbook.md`

# Embodiment Evasion Tax — GPU Day-1 Runbook (Kelvin2 / M1 viability gate)

> **Purpose.** Turn GPU access into the **M1 GO/NO-GO gate** with no improvisation: clone → pinned env →
> checkpoints (hash) → benign baseline → RoboGCG redirect → GCG + L1 micro-bench → metric-(A) signal → **branch
> select (D8)** → GO/NO-GO (H1). Every step ends in a logged artifact under write-once `results/`.
>
> **Cluster mechanics live in [`../gpu/`](../gpu/)** — this runbook references them, does not repeat them:
> [`Connection.md`](../gpu/Connection.md) (SSH/MFA), [`Start.md`](../gpu/Start.md) (modules/storage),
> [`Running.md`](../gpu/Running.md) (Slurm/`--gres`), [`Overview.md`](../gpu/Overview.md) (hardware map).
>
> **Granted GPU = Kelvin2** (NI-HPC @ QUB). Target partitions `k2-gpu-a100` / `k2-gpu-h100` (A100/H100, 80 GB,
> **3-day** walltime cap). All pins below are **best-effort, fetched from source** (invariant #8) and marked
> **`[VERIFY ON THE GPU NODE]`** — confirm before any logged run.

---

## Pre-flight (do once)

- [ ] Account + MFA working; can `ssh kelvin2` ([`Connection.md`](../gpu/Connection.md)).
- [ ] Repo cloned into `$HOME` (`/users/<id>/vla-injection`); `git log` matches `origin/main`.
- [ ] Scratch dir made: `mkdir -p /mnt/scratch2/users/<id>/{checkpoints,data,rollouts,untrusted}`.
- [ ] Know your CUDA module: `module avail cuda` → record the exact string (e.g. `libs/cuda/12.x`).
- [ ] `nvidia-smi` inside an interactive GPU job prints A100/H100 (so the env actually sees a GPU).

> **Ethics/repro standing rules (CLAUDE.md):** GCG suffixes + any attacked rollouts → `artifacts/untrusted/`
> (here: `/mnt/scratch2/users/<id>/untrusted/`); never auto-run untrusted checkpoints; **log exact HW + precision
> + parallelism per run; never compare across HW within one claim** (§8). Scratch is **purged at 90 days** → copy
> hashed checkpoints + write-once `results/` back to `$HOME`/`origin`.

---

## Step 1 — Pinned environments

Three stacks; **keep RoboGCG isolated** from the OpenVLA/LIBERO env (its `robo_env.yml` likely conflicts).

```bash
# (a) OpenVLA + project inference/eval env
module purge
module load <anaconda-module>          # [VERIFY] e.g. `apps/anaconda3/...`; or python3 + venv
module load <cuda-module>              # [VERIFY] match torch 2.2.0 (default cu121)
conda create -n evasion_tax-openvla python=3.10.13 -y && conda activate evasion_tax-openvla

git clone https://github.com/openvla/openvla.git && cd openvla
git checkout c8f03f48                  # codec-verified commit (docs/references)
pip install -e .                       # pulls the pins in configs/env/requirements-gpu.txt
pip install "flash-attn==2.5.5" --no-build-isolation   # [VERIFY] separate build, needs nvcc
cd ..

# (b) LIBERO simulation (separate source install; OpenVLA eval depends on it)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO
pip install -e .
pip install -r ../openvla/experiments/robot/libero/libero_requirements.txt   # [VERIFY path]
cd ..

# (c) install our package into the same env so scripts import `evasion_tax`
cd vla-injection && pip install -e . && cd ..
```

**Capture the env** the moment it works (reproducibility):
```bash
pip freeze > results/$(date -u +%Y%m%dT%H%M%SZ)-env/pip-freeze.txt   # or use src/evasion_tax/repro/capture_env
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```
Record CUDA/driver/torch + the loaded module strings into `docs/references/README.md` (pinned-env row).

> **Gate 1:** `import torch; torch.cuda.is_available() == True` on a GPU node, and `import openvla`/LIBERO import
> clean. If flash-attn won't build, fall back to `attn_implementation` default (no flash-attn) and note it — it
> changes speed, not correctness.

---

## Step 2 — Checkpoints (download + hash + provenance)

HF repos (verbatim): base `openvla/openvla-7b`; LIBERO fine-tunes
`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`.

```bash
export HF_HOME=/mnt/scratch2/users/<id>/checkpoints/hf
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial   # repeat per suite
```

For **each** checkpoint, fill its row in `docs/references/README.md` (source URL, **SHA-256** of the resolved
weight files, date, **licence — VERIFY on the HF model card**, note Llama-2 backbone terms). Also run
`scripts/fetch_openvla_stats.py` to pull each suite's `dataset_statistics.json` (q01/q99/mask) → gitignored
`data/openvla/` + `data/openvla/provenance.json` (this is what the local action codec was built against).

> **Gate 2:** all needed checkpoints downloaded, hashed, provenance rows filled. **Nothing untrusted auto-run.**

---

## Step 3 — Benign LIBERO baseline (the comparator)

Reproduce published benign success on pinned seeds — this is the calibration/benign pool **and** the
"one-variable" comparison anchor.

```bash
# inside an sbatch job on k2-gpu-a100 (see Running.md for the full SBATCH header)
python openvla/experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True            # ESSENTIAL — models fine-tuned with random-crop aug
```

Log success rate + per-rollout records (the `RolloutStep` schema) to write-once `results/`. **Benign rollouts
are cheap (no GCG)** → generate enough to support the FPR claims: **≥ ~300 held-out benign** for a **1 %** point,
**≥ 60** for the **5 %** primary (playbook §5 power rule / Codex #2 #3; `src/evasion_tax/eval/power.py`).

> **Gate 3 (H1 part a):** benign success reproduced within tolerance of OpenVLA's published LIBERO numbers on
> pinned seeds, logged. Mismatch → debug env (center_crop, dtype, checkpoint) before proceeding.

---

## Step 4 — RoboGCG targeted redirect (the attack)

RoboGCG (`github.com/eliotjones1/robogcg`) in its **own** env:

```bash
conda env create -f robogcg/robo_env.yml && conda activate robo_env   # [VERIFY contents]
# co-tracker dep:
git clone https://github.com/facebookresearch/co-tracker && pip install -e co-tracker
python -m experiments.single_step.run_experiment \
    --config experiments/single_step/configs/libero_10/libero_10_0.json \
    --num-gpus 1
```

Confirm the attack produces a **targeted redirect to the chosen low-level action region** (D2 success =
*reached the target action over the persistence window*), **not** mere denial/freezing. Suffixes + attacked
rollouts → `/mnt/scratch2/users/<id>/untrusted/` (quarantine).

> **Gate 4 (H1 part b) + D2 arm decision:** if coherent *targeted* redirect reproduces → keep the redirect
> framing; if only goal-abandonment shows → reframe to *task-deviation* detection (pre-registered, either is
> reportable). Licence of RoboGCG **not stated in README** → record what you find (`[VERIFY]`).

---

## Step 5 — Micro-bench → resolve D4 / D7 / D8 (the load-bearing measurement)

This is the step that **selects the compute branch**. Measure on the *actual granted card* (A100 vs H100 — log
which; never mix in one claim):

```bash
python -m scripts.microbench_gcg --config configs/example_m2.yaml   # times s/target at bf16
```

Record, with `sacct` backing the numbers:
- **GCG s/target** at bf16 (published H100 ≈185–604 s/target should ≈ transfer — confirm on the card). → **D4** (eval matrix size) + **D7** (attack compute budget).
- **L1 activation/attention extraction overhead** per rollout (Codex #2 #5). → feeds L1 scope.
- **Adaptive-GCG-against-the-probe-score** cost — affordable? If too costly at the measured budget, **drop the
  adaptive-L1 arm**, keep non-adaptive L1 + the L2-oracle frontier (D7 rule).
- **LIBERO rollout throughput** + effective parallelism / **queue depth** on the shared A100/H100 nodes.

Feed these against the remaining calendar to compute the **affordable matrix** → **select Branch (D8)**:
- **N** (full deployable tax matrix fits + slack) → commit **H6-D** headline.
- **N−** (only a reduced H6-D fits: one route B/C, 5 %-FPR primary) → commit **scoped H6-D**.
- **F** (not even reduced fits) → **H6-A oracle frontier** headline, mark H6-D **not run / unresolved**.

> **Gate 5 (D8):** branch chosen by **measured affordability**, written into playbook §1/§2 with the numbers.
> Pre-registered now so the choice is honest, not hope-driven. **M3/H6-A is delivered in every branch.**

---

## Step 6 — Metric-(A) signal sanity check

Run the (already-built, unit-tested) `evasion_tax.metric.consistency_a` L2-oracle scorer over benign vs attacked
rollouts, with the **real LIBERO ground-truth** state adapter wired in (the model-free piece deferred from
local-prep Task 4/10):

```bash
python -m scripts.run_attack   ... # produces attacked rollouts
python -m scripts.evaluate     --config configs/example_m2.yaml   # ROC/AUC, TPR@FPR + CIs, latency
```

Key check: separation that **survives at the coarse operator-goal reference** (D3 rung), **not just** the
clean-instruction ceiling. This is the difference between a deployable signal and a trivial one.

> **Gate 6 (H1 part c):** benign-vs-attacked metric-(A) separation holds at the **coarse operator-goal** rung.

---

## GO / NO-GO gate (H1) — all four must hold

1. **Benign reproduced** (Step 3, Gate 3).
2. **Targeted redirect** (not denial) confirmed (Step 4, Gate 4).
3. **Metric-(A) separation survives the coarse-goal reference** (Step 6, Gate 6).
4. **Compute branch N/N−/F selected** from the micro-bench (Step 5, Gate 5).

**GO** → proceed to **M2** (calibrated floor detector: τ on calib split → per-rollout FPR on held-out; ROC/AUC,
TPR@{5%,1%}+CIs, latency → **H2, FLOOR SECURED**) then M3 (H6-A) → M4 (H6-D, branch-permitting).
**NO-GO** → record exactly which gate failed and why (negative result, not dropped); revisit D2 framing / env /
checkpoint before re-running.

---

## Per-run protocol (paste into every run log — playbook §8)

```
run_id:        <UTC timestamp>-<short-slug>
git_commit:    <hash>
hardware:      <A100|H100 — card+count> + precision bf16 + CUDA/driver/torch   (NEVER compare across HW in one claim)
config:        <path to pinned config>
seed(s):       <pinned, recorded>
hypothesis:    <H#>
expected:      <pre-registered prediction>
command:       <exact command>
results_path:  results/<timestamp>/...   (WRITE-ONCE)
observed:      <fill after run>
decision:      <what it changes; link §10 if a decision was made>
one_variable:  <single variable changed vs the comparison run>
```

Reproducibility checklist (tick before committing any result): seeds pinned+recorded · env captured ·
checkpoint/dataset provenance (source/**hash**/date/licence) · write-once `results/` (no overwrite) · exactly
**one variable** changed · figure regenerable by a committed script · negatives recorded · adversarial artefacts
quarantined · no dataset/checkpoint/secret/PII staged for commit.

---

## Quick env-pin reference (best-effort — `[VERIFY ON THE GPU NODE]`)

| Component | Pin | Source |
|-----------|-----|--------|
| Python | 3.10.13 | OpenVLA README results env |
| torch / torchvision / torchaudio | 2.2.0 / 0.17.0 / 2.2.0 | OpenVLA `pyproject.toml` |
| transformers / tokenizers | 4.40.1 / 0.19.1 | OpenVLA `pyproject.toml` |
| timm / peft / sentencepiece / draccus | 0.9.10 / 0.11.1 / 0.1.99 / 0.8.0 | OpenVLA `pyproject.toml` |
| flash-attn | 2.5.5 (separate build) | OpenVLA README results env |
| GPU / precision | A100 (or H100) / **bf16** | OpenVLA README + D8 |
| OpenVLA commit | `c8f03f48` | codec-verified (docs/references) |
| RoboGCG env | `robo_env.yml` + co-tracker | RoboGCG README (pins undocumented → VERIFY) |
| LIBERO | source install + `libero_requirements.txt` | OpenVLA README |

Full pip layer: [`../../configs/env/requirements-gpu.txt`](../../configs/env/requirements-gpu.txt).


# Embodiment Evasion Tax — GPU Day-1 Runbook (M1 viability gate)

> **⚠️ REGISTERED COMPUTE CHANGED (2026-06-16).** The registered box is now **CSB `ecs3-0202` = 2× RTX A5000
> (24 GB)** — direct SSH, **no Slurm/queue/walltime cap**. **Start at the bring-up ladder in
> [`../gpu/CSB/plan.md`](../gpu/CSB/plan.md)** (env + bf16 stand-up + GCG/D8 bench on the A5000). **Kelvin2 is a
> BACKUP only** — this Slurm runbook is the Kelvin2 path, kept for that contingency. The **protocol** sections
> (repro header, checkpoint hashing, GCG/L1 micro-bench, branch select, GO/NO-GO) are **hardware-agnostic and
> apply on the A5000 too** — just skip the `sbatch`/`--gres`/`module` steps and run interactively. Reproducibility
> rule: **all registered runs commit to the A5000**; Kelvin2, if used, is a **separate** registration (no cross-HW
> mixing). Note the A5000 has **no published OpenVLA-GCG prior** → the H100 timing references below do **NOT**
> transfer; measure on the card.

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
- [ ] Repo cloned into `$HOME` as `$HOME/vla-injection`; `git log` matches `origin/main`.
- [ ] CUDA + anaconda module names recorded: `module avail cuda` / `module avail anaconda`.
- [ ] `nvidia-smi` sees a GPU: `srun -p k2-gpu-a100 --gres=gpu:a100:1 -N1 -n1 --time=0:10:00 --pty nvidia-smi` prints A100/H100.

**Session variables — `export` once per login** (every command below uses them; fill the two `[ON-NODE]` values
from the `module avail` above):

```bash
export K2_ID=<your-qub-id>                  # student/staff number
export CUDA_MODULE=<libs/cuda/12.x>         # [ON-NODE] from `module avail cuda`
export ANACONDA_MODULE=<apps/anaconda3/..>  # [ON-NODE] from `module avail anaconda`
export REPO=$HOME/vla-injection
export SCRATCH=/mnt/scratch2/users/$K2_ID
export HF_HOME=$SCRATCH/checkpoints/hf
mkdir -p $SCRATCH/{checkpoints,data,rollouts,untrusted} $REPO/results/slurm
```

**Where each step runs:**

| Steps | Where | How |
|-------|-------|-----|
| 1–2 — env build, checkpoint download | **login node** (big downloads via data-mover `kelvin2-dm`) | run directly |
| 3–6 — benign / attack / micro-bench / eval | **GPU node** | `sbatch` (or `srun --pty` to debug) |

**Reusable GPU job wrapper — create once, then submit any GPU step with**
`sbatch k2_gpu.sbatch <conda-env> '<command>'`:

```bash
cat > $REPO/k2_gpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=evasion_tax
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --time=1-00:00:00                  # ≤3-day cap; raise for long GCG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=k2-gpu-a100            # if granted H100: k2-gpu-h100
#SBATCH --gres=gpu:a100:1                  #   …and gpu:h100:1
#SBATCH --mail-type=END,FAIL
module purge
module load "$ANACONDA_MODULE" "$CUDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$1"; shift                 # 1st arg = conda env name
nvidia-smi                                 # provenance: log the exact card
eval "$@"                                  # remaining args = the step command
EOF
```

> **Always submit from `$REPO`** (so `results/slurm/` resolves — `sbatch` uses the submit dir as the job cwd).
> The wrapper inherits your exported `$ANACONDA_MODULE`/`$CUDA_MODULE` (sbatch propagates the login env by
> default). Debug interactively instead with
> `srun -p k2-gpu-a100 --gres=gpu:a100:1 -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G --time=2:00:00 --pty bash`.

> **Ethics/repro standing rules (CLAUDE.md):** GCG suffixes + any attacked rollouts → `artifacts/untrusted/`
> (here: `$SCRATCH/untrusted/`); never auto-run untrusted checkpoints; **log exact HW + precision + parallelism
> per run; never compare across HW within one claim** (§8). Scratch is **purged at 90 days** → copy hashed
> checkpoints + write-once `results/` back to `$HOME`/`origin`.

---

## Step 1 — Pinned environments

Three stacks; **keep RoboGCG isolated** from the OpenVLA/LIBERO env (its `robo_env.yml` likely conflicts).

```bash
# Run on: login node. (a) OpenVLA + project inference/eval env
module purge
module load "$ANACONDA_MODULE" "$CUDA_MODULE"          # [ON-NODE] names recorded in Pre-flight
source "$(conda info --base)/etc/profile.d/conda.sh"   # lets `conda activate` work non-interactively
conda create -n evasion_tax-openvla python=3.10.13 -y && conda activate evasion_tax-openvla

cd $HOME
git clone https://github.com/openvla/openvla.git && cd openvla
git checkout c8f03f48                  # codec-verified commit (docs/references)
pip install -e .                       # pulls the pins in configs/env/requirements-gpu.txt
pip install "flash-attn==2.5.5" --no-build-isolation   # separate build, needs nvcc (from $CUDA_MODULE)
cd ..

# (b) LIBERO simulation (separate source install; OpenVLA eval depends on it)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO
pip install -e .
pip install -r ../openvla/experiments/robot/libero/libero_requirements.txt
cd ..

# (c) install our package into the same env so scripts import `evasion_tax`
cd $REPO && pip install -e . && cd ..
```

> **uv variant (CSB A5000 box) — two traps confirmed 2026-06-18, full how-to in `docs/gpu/CSB/plan.md` Step 4:**
> (1) `uv pip install -e LIBERO` does **not** make `import libero` work — LIBERO's top `libero/` is a PEP-420
> namespace package (no `__init__.py`); uv's PEP-660 editable won't expose it (the legacy `pip install -e .`
> above does) → use **`PYTHONPATH=<repo>/LIBERO`**. (2) The eval-helper import chain pulls `tensorflow_datasets`,
> and tfds 4.9.3 caps nothing on `tensorflow-metadata` → pip pulls a too-new tfmd (needs protobuf≥5.26) → pin
> **`tensorflow-metadata<1.16` + `protobuf<5`** (`configs/env/requirements-gpu.txt`).

**Capture the env** the moment it works (reproducibility):
```bash
ENVDIR=results/$(date -u +%Y%m%dT%H%M%SZ)-env && mkdir -p $ENVDIR
pip freeze > $ENVDIR/pip-freeze.txt   # or use src/evasion_tax/repro/env_capture.py
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"   # run on a GPU node
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
# Run on: login node (large files → data-mover `kelvin2-dm`). $HF_HOME is set in Pre-flight.
for suite in spatial object goal 10; do
  huggingface-cli download openvla/openvla-7b-finetuned-libero-$suite
done
huggingface-cli download openvla/openvla-7b      # base model (for the redirect attack)
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
# Run on: GPU node — submit from $REPO with the wrapper built in Pre-flight.
# --center_crop True is ESSENTIAL (models fine-tuned with random-crop aug).
cd $REPO && sbatch k2_gpu.sbatch evasion_tax-openvla \
  'python $HOME/openvla/experiments/robot/libero/run_libero_eval.py \
     --model_family openvla \
     --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
     --task_suite_name libero_spatial \
     --center_crop True'
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
# Run on: login node — clone RoboGCG + build its OWN env (keep isolated from the openvla env):
cd $HOME && git clone https://github.com/eliotjones1/robogcg.git
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f robogcg/robo_env.yml && conda activate robo_env   # [VERIFY contents]
git clone https://github.com/facebookresearch/co-tracker && pip install -e co-tracker

# Run on: GPU node — submit the attack (cd into robogcg so its relative config path resolves):
cd $REPO && sbatch k2_gpu.sbatch robo_env \
  'cd $HOME/robogcg && python -m experiments.single_step.run_experiment \
     --config experiments/single_step/configs/libero_10/libero_10_0.json --num-gpus 1'
```

Confirm the attack produces a **targeted redirect to the chosen low-level action region** (D2 success =
*reached the target action over the persistence window*), **not** mere denial/freezing. Suffixes + attacked
rollouts → `$SCRATCH/untrusted/` (quarantine).

> **Gate 4 (H1 part b) + D2 arm decision:** if coherent *targeted* redirect reproduces → keep the redirect
> framing; if only goal-abandonment shows → reframe to *task-deviation* detection (pre-registered, either is
> reportable). Licence of RoboGCG **not stated in README** → record what you find (`[VERIFY]`).

---

## Step 5 — Micro-bench → resolve D4 / D7 / D8 (the load-bearing measurement)

This is the step that **selects the compute branch**. Measure on the *actual granted card* (A100 vs H100 — log
which; never mix in one claim):

```bash
# Run on: GPU node (or `srun --pty` to watch live):
cd $REPO && sbatch k2_gpu.sbatch evasion_tax-openvla \
  'python -m scripts.microbench_gcg --config configs/example_m2.yaml'   # times s/target at bf16
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
# Run on: GPU node. run_attack writes a per-condition scores JSON under results/<run>/.
cd $REPO && sbatch k2_gpu.sbatch evasion_tax-openvla \
  'python -m scripts.run_attack --config configs/example_m2.yaml'        # produces attacked rollouts + scores

# evaluate REQUIRES --scores (the JSON run_attack just wrote):
cd $REPO && sbatch k2_gpu.sbatch evasion_tax-openvla \
  'python -m scripts.evaluate --config configs/example_m2.yaml --scores results/<RUN>/scores.json'   # [FILL <RUN>/filename from run_attack output] → ROC/AUC, TPR@FPR + CIs, latency
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

# Kelvin2 — Cluster Overview

> **Source:** <https://ni-hpc.github.io/nihpc-documentation/Kelvin2%20Overview/> — fetched 2026-06-03.
> Numbers below are reproduced from the official NI-HPC docs; verify on the node (`sinfo`, `scontrol`) before
> citing in the dissertation. Setup assumed: **macOS laptop + QUB network access.**

## What it is

**Kelvin2** is the **NI-HPC** (Northern Ireland High-Performance Computing) research cluster, hosted at
**Queen's University Belfast**. This is the GPU we have been granted for the project (the "A100/H100" of the playbook's
D8 decision corresponds to the `k2-gpu-a100` / `k2-gpu-h100` partitions below).

- **OS:** Linux — **CentOS 7**
- **Scheduler:** Slurm (see [`Running.md`](./Running.md))
- **Module system:** Lmod-style `module` commands (see [`Start.md`](./Start.md))

## Compute resources

| Class | Count | Per-node hardware | Memory/node |
|-------|-------|-------------------|-------------|
| Standard compute | 96 nodes | Dell PowerEdge R6525, **dual AMD EPYC 7702** (2 × 64 = **128 cores**) | **768 GB** |
| High-memory | 8 nodes | 128–256 cores | **2 TB** |

Total standard CPU cores ≈ **12,288**.

## GPU resources  ← the relevant part for the project

| GPUs | Nodes | GPUs/node | GPU memory | Slurm partition | `--gres` type |
|------|-------|-----------|-----------|-----------------|---------------|
| **NVIDIA A100** | 4 nodes (16 GPUs) | 4 | **80 GB** | `k2-gpu-a100` | `a100` |
| A100 **MIG** slices | — | 7 slices/node | 80 GB (sliced) | `k2-gpu-a100mig` | e.g. `2g.20gb` |
| **NVIDIA H100** | 1 node (4 GPUs) | 4 | **80 GB** | `k2-gpu-h100` | `h100` |
| NVIDIA V100 | 8 nodes (32 GPUs) | 4 | 32 GB | `k2-gpu-v100` | `v100` |
| Intel MAX 1100 | 1 node (4 GPUs) | 4 | — | `k2-gpu-intel` | `i1100` |
| AMD MI300X | 1 node (8 GPUs) | 8 | — | `k2-gpu-amd` | `mi300x` |

**Project fit.** OpenVLA-7B in **bf16** is ≈14 GB → comfortably fits a single 80 GB A100 or H100 card. So `--gres=gpu:1`
on `k2-gpu-a100` / `k2-gpu-h100` is the target. The A100/H100 GPU partitions cap at **3-day** walltime — this is
the calendar constraint the **M1 on-GPU timing micro-bench** must budget against when selecting Branch N / N− / F
(D8). Note the cluster is **shared** (only 4 A100 nodes + 1 H100 node) → **queue depth is a real variable**; record
it during M1, as flagged in the playbook (D8 "single-card-vs-cluster + queue depth TBC").

## Minimum viable GPU (project floor)

The binding constraint is **RoboGCG (bf16 forward+backward + candidate-batch eval)**, not inference — so the
GPU floor is set by GCG, not by loading the model. (Weights ≈14 GB is the established OpenVLA-7B bf16 size from
the table above; the **GCG VRAM figures below are an estimate** — they depend on candidate-batch size, sequence
length, and gradient checkpointing, so confirm them during the **M1 micro-bench**.)

| Workload | VRAM floor → comfortable | Binding? |
|----------|--------------------------|----------|
| OpenVLA-7B **bf16 inference** | 16 GB (tight) → 24 GB | no |
| **LIBERO rollout** (inference-driven) | ≈ same as inference | no |
| **RoboGCG** (bf16 backward + candidate batch) | **24 GB (needs OOM-tuning) → 40 GB (real floor) → 80 GB (headroom)** | **yes — sets the floor** |

**Project minimum (full registered protocol):**

| Item | Minimum | Comfortable |
|------|---------|-------------|
| GPU | 1 × NVIDIA CUDA, **24 GB** VRAM | **40 GB+** |
| GPU count | **1** — OpenVLA-7B fits one card, **no cluster / multi-GPU needed** | 1 |
| System RAM | 32 GB | 64 GB |
| Disk | ~50 GB (ckpt ~14 GB + LIBERO + logs) | ~100 GB |
| OS | **Linux + CUDA** (the OpenVLA / flash_attn / bitsandbytes stack is Linux-first) | — |

**Where Kelvin2 lands.** The granted **A100 / H100 (80 GB)** clears this floor with large headroom — request a
single card (`--gres=gpu:a100:1`, see [`Running.md`](./Running.md)). An `k2-gpu-a100mig` slice (e.g. `2g.20gb` =
20 GB) sits at the *inference* floor and is fine for OpenVLA stand-up / cheap harness debugging, but **GCG wants
a full card** (≥40 GB). For reference, the local **CSB box (8 GB)** is ~⅓ of the floor → 4-bit smoke-tests only
([`CSB/plan.md`](./CSB/plan.md)); a 16 GB card (e.g. an unverified laptop dGPU) can do bf16 *inference* but
**not** GCG.

## Storage

| Store | Path | Quota | Retention |
|-------|------|-------|-----------|
| **Home** | `/users/<username>` | **50 GB / 100k files** | No automated deletion |
| **Scratch** (Lustre, 2 PB) | `/mnt/scratch2/users/<username>` | none | **purged if unused 90 days** |
| Node-local temp | `/tmp` (on compute node) | node capacity | deleted at job end |
| McClayRDS (QUB only) | `/mnt/autofs/mcclayrds-projects/<project-code>` | per project | replicated to secondary site |

**Project storage rule (maps to CLAUDE.md reproducibility + repo conventions):**
- **Home** — code, configs, conda envs, the cloned repo, small `results/` JSON.
- **Scratch** — OpenVLA checkpoints, LIBERO data, rollout tensors, GCG suffixes. **Treat as ephemeral** (90-day
  purge): anything that must survive (provenance-hashed checkpoints, write-once `results/`) is copied back to the
  repo / home or pushed to `origin`. Quarantine adversarial artefacts under `artifacts/untrusted/` on scratch.
- Check home quota with `quota -s`.

## Not stated in the docs (verify on node)

InfiniBand interconnect details, exact CUDA/driver versions, and per-partition QoS limits are **not** in the
overview page — capture them with `module avail cuda`, `nvidia-smi`, and `scontrol show partition` during M1 and
record into `docs/references/README.md` (provenance rule).

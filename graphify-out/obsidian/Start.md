---
source_file: "docs/gpu/Start.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# Start.md

## Connections
- [[Kelvin2 — Quickstart]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/gpu/Start.md`

# Kelvin2 — Quickstart

> **Source:** <https://ni-hpc.github.io/nihpc-documentation/Quickstart%20Guide/> — fetched 2026-06-03.
> Read [`Connection.md`](./Connection.md) first to get on the cluster. Job-script detail is in
> [`Running.md`](./Running.md).

## 1. After you log in

You arrive on a **login node** (`login1`–`login4`):

```
[<studentnumber>@login1 [kelvin2] ~]$
```

**Login nodes are for light work only** — editing files, small transfers, building job scripts, submitting and
monitoring jobs. **Never run OpenVLA / GCG / LIBERO here** — that belongs in a Slurm job on a GPU node
([`Running.md`](./Running.md)). Running heavy work on a login node degrades it for every user.

## 2. Storage — where to put what

| Store | Path | Quota | Auto-delete | Use for |
|-------|------|-------|-------------|---------|
| Home | `/users/<studentnumber>` (`$HOME`) | 50 GB / 100k files | no | repo, configs, conda envs, small `results/` |
| Scratch (Lustre) | `/mnt/scratch2/users/<studentnumber>` | none | **90 days unused** | checkpoints, datasets, rollouts |
| Node temp | `/tmp` | node | at job end | intermediate scratch within a job |
| McClayRDS (QUB) | `/mnt/autofs/mcclayrds-projects/<project-code>` | per project | per policy | replicated long-term project store |

```bash
quota -s          # check your home quota
```

> **Project mapping:** clone the repo into `$HOME`; stage OpenVLA-7B weights + LIBERO into scratch; copy anything that
> must survive the 90-day purge (provenance-hashed checkpoints, write-once `results/`) back to home or push to
> `origin`. The 100k-file home limit matters — conda envs have many files; prefer one env, and keep datasets off
> home.

## 3. The module system

Software is provided via environment modules:

```bash
module avail                 # list available software
module avail cuda            # filter (e.g. find CUDA versions)
module spider <name>         # search + show how to load a specific version
module load apps/python3/3.10.5/gcc-9.3.0   # load (example: Python 3.10.5)
module list                  # what's currently loaded
module unload <module>       # drop one
module purge                 # clear all loaded modules (clean slate)
```

Module names are **versioned and compiler-tagged** (e.g. `apps/python3/3.10.5/gcc-9.3.0`,
`mpi/openmpi/5.0.3/gcc-14.1.0`). Always `module purge` at the top of a job script, then load exactly what you
need, so the environment is reproducible (CLAUDE.md: capture exact env).

> **Project note:** the playbook pins the OpenVLA stack as **Python 3.10-compatible** (the local code is kept
> 3.10-safe for exactly this node). Find the CUDA + Python + (likely conda) modules during M1 with
> `module avail`, record the exact module strings into `docs/references/README.md`, and pin them in
> `configs/env/requirements-gpu.txt`.

## 4. Python environment

The docs show loading Python via a module; for a project env, load a Python/Anaconda module then create an
isolated environment (verify the exact anaconda module name on-node with `module avail anaconda`):

```bash
module load apps/python3/3.10.5/gcc-9.3.0      # or an anaconda module
python3 -m venv /users/<studentnumber>/venvs/evasion_tax   # or: conda create -n evasion_tax python=3.10
source /users/<studentnumber>/venvs/evasion_tax/bin/activate
pip install -r requirements-gpu.txt
```

> The Quickstart page does **not** spell out conda/venv steps — confirm the recommended pattern (module vs conda)
> on-node during M1. Keep envs in `$HOME` (not scratch — survives purge) but mind the 100k-file quota.

## 5. Submit your first job (the workflow)

Minimal Slurm batch script (`hello-world-jobscript.sh`) from the official guide:

```bash
#!/bin/bash

#SBATCH --output=hello-world-job.output
#SBATCH --time=00:00:10

module load apps/python3/3.10.5/gcc-9.3.0

python3 hello-world.py
```

Submit, monitor, read output:

```bash
sbatch hello-world-jobscript.sh     # submit → prints a job id
squeue -u <studentnumber>           # your queued/running jobs
cat hello-world-job.output          # output once it completes
```

For real GPU jobs (partitions, `--gres`, walltime, interactive sessions) see **[`Running.md`](./Running.md)**.

## 6. Data transfer

```bash
# small files, on QUB network:
scp ./file.txt <studentnumber>@kelvin2.qub.ac.uk:/mnt/scratch2/users/<studentnumber>/

# small files, off network:
scp -P 55890 -i ~/.ssh/kelvin2-key ./file.txt \
    <studentnumber>@login.kelvin.alces.network:/mnt/scratch2/users/<studentnumber>/
```

For **large** transfers (checkpoints, datasets) use the **data-mover nodes** `dm1.kelvin.alces.network` /
`dm2.kelvin.alces.network`, and pack many small files into a tar archive first.

## 7. Best practices (from the guide)

- Scripts/config in **home**; big data in **scratch**; intermediate files in `/tmp`.
- Archive small files before transferring.
- Request McClayRDS for secure, replicated project storage if needed.
- (project) one variable per run, seeds pinned, exact env captured, results to write-once `results/` — the Kelvin2
  storage layout above is how those rules land on this cluster.


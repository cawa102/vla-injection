# Kelvin2 — Running Jobs (Slurm)

> **Source:** <https://ni-hpc.github.io/nihpc-documentation/Running%20Jobs/> — fetched 2026-06-03.
> Prereqs: [`Connection.md`](./Connection.md) (get on the cluster) and [`Start.md`](./Start.md) (modules,
> storage). GPU hardware map is in [`Overview.md`](./Overview.md).

## Batch-script anatomy

A Slurm batch script has four parts:

1. `#!/bin/bash` — shebang
2. `#SBATCH` directives — resources + job attributes
3. `module` loads — environment
4. the commands to run

Submit with `sbatch script.sh`.

## Common `#SBATCH` directives

| Directive | Meaning |
|-----------|---------|
| `--job-name=<name>` | name shown in the queue |
| `--partition=<part>` | which partition/queue (see tables below) |
| `--output=<file>` | stdout file |
| `--error=<file>` | stderr file (keep separate from stdout) |
| `--time=HH:MM:SS` or `D-HH:MM:SS` | walltime limit (job killed after) |
| `--nodes=<n>` | number of nodes |
| `--ntasks=<n>` | number of tasks (MPI ranks) |
| `--cpus-per-task=<n>` | threads per task (OpenMP) |
| `--mem-per-cpu=<size>` | memory per CPU, e.g. `5G` |
| `--gres=gpu:<type>:<n>` | request GPUs (see below) |
| `--mail-type=ALL` / `--mail-user=<email>` | email notifications |

CPU resource idioms:

```bash
# MPI
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=2G
#SBATCH --nodes=2

# OpenMP / single-node multithread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --nodes=1
```

## CPU partitions

| Partition | Walltime limit | Cores/node | Mem/node |
|-----------|----------------|-----------|----------|
| `k2-hipri` | **3 hours** | 128 | 773–1023 GB |
| `k2-medpri` | **24 hours** | 128 | 773–1023 GB |
| `k2-lowpri` | (low priority) | 128 | 773–1023 GB |
| `k2-epsrc` | (EPSRC allocation) | 128 | 773–1023 GB |
| `k2-himem` | **3 days** | 128–256 | 2051–2063 GB |

## GPU partitions  ← T7 runs here

| Partition | GPUs | Walltime | GPU mem | `--gres` type |
|-----------|------|----------|---------|---------------|
| **`k2-gpu-a100`** | 4 × A100 / node (3 nodes) | **3 days** | **80 GB** | `a100` |
| **`k2-gpu-h100`** | 4 × H100 / node (1 node) | **3 days** | **80 GB** | `h100` |
| `k2-gpu-a100mig` | 7 MIG slices / node | 3 days | 80 GB (sliced) | e.g. `2g.20gb` |
| `k2-gpu-v100` | 4 × V100 / 8 nodes | 3 days | 32 GB | `v100` |
| `k2-gpu-amd` | 8 × MI300X / node | 3 days | — | `mi300x` |
| `k2-gpu-intel` | 4 × Intel MAX / node | 3 days | — | `i1100` |

### Requesting GPUs

```
#SBATCH --gres=gpu:<type>:<number>
```

Examples:
- `#SBATCH --gres=gpu:a100:1` — one A100
- `#SBATCH --gres=gpu:h100:1` — one H100
- `#SBATCH --gres=gpu:v100:1` — one V100
- `#SBATCH --gres=gpu:2g.20gb:1` — one A100 MIG slice (small/debug)

> **T7:** OpenVLA-7B bf16 ≈14 GB → **`--gres=gpu:a100:1`** (or `h100`) on `k2-gpu-a100` / `k2-gpu-h100`. **3-day
> walltime cap** is the hard per-job calendar limit the M1 micro-bench budgets against (D8 branch selection).
> GCG suffix optimisation is the expensive part — chunk it into ≤3-day jobs with checkpointing, or it dies at
> the walltime wall. A MIG slice (`k2-gpu-a100mig`) is ideal for **cheap smoke tests / debugging** the harness
> before burning a full A100.

## Example GPU batch script (adapt for T7)

Official example (MATLAB on an A100 MIG slice):

```bash
#!/bin/bash

#SBATCH --job-name=matlab-job-name
#SBATCH --output=matlab-output.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1

module load matlab/R2024a
matlab -nosplash -nodisplay -r "matlab_gpu_script;"
```

T7-shaped skeleton (verify module names on-node during M1):

```bash
#!/bin/bash
#SBATCH --job-name=t7-benign-baseline
#SBATCH --output=results/slurm/%x-%j.out      # %x=jobname %j=jobid
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --time=1-00:00:00                      # 1 day (≤3-day cap)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>

module purge
module load apps/python3/3.10.5/gcc-9.3.0      # + the CUDA module found via `module avail cuda`
source /users/<studentnumber>/venvs/t7/bin/activate

nvidia-smi                                     # log the exact GPU into the run record (provenance)
python -m scripts.run_benign --config configs/example_m2.yaml
```

> Reproducibility: `nvidia-smi`, `module list`, and the git commit should be captured into the run's write-once
> `results/` block (playbook §8 protocol). **Never compare results across different GPU types within one claim**
> (V100 vs A100 vs H100 is a variable).

## Interactive jobs

Grab an interactive shell on a node (good for debugging, first OpenVLA stand-up):

```bash
# CPU example from the docs:
srun -p k2-hipri -N 1 -n 10 --mem-per-cpu=1G --time=1:00:00 --pty bash

# GPU interactive (T7 — one A100, 2h):
srun -p k2-gpu-a100 --gres=gpu:a100:1 -N 1 -n 1 --cpus-per-task=8 \
     --mem-per-cpu=8G --time=2:00:00 --pty bash
```

Graphical apps (VNC), per docs: start an interactive job → `vncserver` → SSH-tunnel from the Mac
`ssh -L 5903:node<N>.pri.kelvin2.alces.network:5901 <studentnumber>@kelvin2.qub.ac.uk` → connect VNC to
`localhost:5903` → `vncserver -kill :1` to stop. (Rarely needed for T7 — it's headless.)

## Job management

| Task | Command |
|------|---------|
| Submit | `sbatch <script>` |
| My queue | `squeue -u <studentnumber>` |
| One job | `squeue -j <jobid>` |
| Cancel | `scancel <jobid>` |
| Partition state / free GPUs | `sinfo` (and `sinfo -p k2-gpu-a100`) |
| Completed-job accounting | `sacct -j <jobid> --format="JobID,jobname,NTasks,nodelist,CPUTime,ReqMem,MaxVMSize,Elapsed"` |

Use `sacct` after a run to see **actual** memory/CPU/time used and right-size the next submission — this is also
how you record the real `s/target` GCG cost for the **M1 micro-bench (D4/D7)**.

## Job arrays (bulk runs)

Slurm job arrays are the right tool for sweeping many targets/seeds (e.g. the eval matrix). Add
`#SBATCH --array=0-49` and index work by `$SLURM_ARRAY_TASK_ID`. (The docs mention arrays for bulk submission;
confirm any per-user array/GPU concurrency limit with the admins before launching a big sweep on the shared
A100/H100 nodes.)

## Tips (from the docs)

- Don't request more than a single node provides; spread across nodes with `--ntasks` + `--mem-per-cpu`.
- Separate `--output` and `--error`.
- Match `--time` to real need (over-asking lengthens queue wait; under-asking kills the job).
- Enable email notifications for long jobs.
- The cluster is shared and A100/H100 capacity is small — **expect queue waits**; log them for D8.

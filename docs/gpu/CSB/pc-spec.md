# CSB — Registered-Compute Box Hardware Spec

> **Source:** `nvidia-smi` on the box, **read 2026-06-16** (host `ecs3-0202`, login
> `40473058@eeecs.qub.ac.uk`, EEECS-managed Linux desktop). This **supersedes** the 2026-06-15 Windows
> "Device specifications" reading, which was a **different** candidate machine (RTX 4060 / 8 GB / Windows 11) and
> is **no longer the plan** — see *Superseded* at the bottom. Re-verify the open items in *Not in `nvidia-smi`*
> on the box before freezing a run (audit trail, mirrors the Kelvin2 docs' "verify on node" rule).

## What it is

**CSB** is the box `ecs3-0202` in the Computer Science Building — a **2× RTX A5000 (24 GB each)** Linux desktop.
As of **2026-06-16 (author decision)** it is the project's **registered compute**: the EET experiments (M1–M4,
the D8 timing micro-bench, the registered bf16 runs) are measured **here**. **Kelvin2** (A100/H100 cluster) is
demoted to a **backup contingency** ([`../Overview.md`](../Overview.md)) — its access was never established, and
because the reproducibility rule forbids cross-hardware comparison within one claim, **all registered runs
commit to this A5000 box**; if Kelvin2 ever comes online it would require its **own separate** registration, not
a mix. Scope/runbook: [`plan.md`](./plan.md).

- **OS:** native **Linux** (the prompt is bash on `~/Desktop`; Xorg + gnome-shell run on GPU 1). No WSL2 layer —
  the entire Windows/WSL2 toolchain risk class is gone.

## Hardware (from `nvidia-smi`, 2026-06-16)

| Component | Spec | Notes |
|-----------|------|-------|
| **GPU ×2** | **2× NVIDIA RTX A5000 — 24,564 MiB (≈24 GB) each** | Ampere (GA102). **bf16 OpenVLA-7B (~14 GB) fits on one card.** No NVLink on A5000 → inter-card link is PCIe. |
| **Driver / CUDA** | **595.71.05 / CUDA 13.2** | Newer than the OpenVLA pin (torch 2.2.0 = cu121). A newer **driver** is backward-compatible (runs cu121 torch); verify `flash-attn` loads (wheel vs rebuild). |
| GPU 0 | 15 MiB used (idle) | **fully free** — the registered-run card. |
| GPU 1 | 579 MiB used (`Disp.A = On`) | drives the **display** (Xorg / gnome-shell / firefox); ≈23.4 GB still free. |
| CPU / RAM / storage | **unknown for `ecs3-0202`** | the i7-14700KF / 16 GB / 954 GB in the *Superseded* table were the **other** (Windows) box — **re-read on `ecs3-0202`** (`lscpu`, `free -h`, `df -h`) before citing. |

### What 24 GB VRAM means for OpenVLA-7B (≈7.5B params)

| Precision | Weights ≈ | Fits 24 GB? |
|-----------|-----------|-------------|
| **bf16 — the registered precision** | ~14 GB | ✅ **yes** — fits one card with ~10 GB headroom for activations (batch-1 inference). |
| int8 / 4-bit | ~7.5 / ~4 GB | not needed — bf16 is the registered mode; quantization would only re-introduce the precision confound. |

→ The A5000 box can run the **registered bf16 model**, **real LIBERO rollouts (EGL headless)**, and — pending
the M1 micro-bench — the **GCG attack** (backward + candidate-batch is plausible at 24 GB with batch-1 /
gradient-checkpointing; **measure, do not assume**). The hard "4-bit only / no science numbers" walls of the
old 8 GB box are **gone**. Consequences + the de-risking-vs-registered ladder: [`plan.md`](./plan.md).

## Not in `nvidia-smi` — verify on the box before a registered run (audit trail)

- **CPU / RAM / storage** of `ecs3-0202` (`lscpu`, `free -h`, `df -h`) — must be re-read (see table note).
- **CUDA toolkit (`nvcc --version`) vs the 13.2 driver**, and whether **`flash-attn==2.5.5`** loads against the
  installed torch (cu121) under this driver — or needs a wheel / rebuild / a torch bump.
- **`MUJOCO_GL=egl`** headless rendering works on the NVIDIA GPU (LIBERO camera obs) — the GL wall the local
  state-only path dodged; on a Linux + NVIDIA box it should be crossable.
- **Exclusivity / sharing:** it is a **shared lab desktop** (GPU 1 runs the GUI; others may log in). The D8
  **timing** micro-bench needs a quiet/exclusive window — confirm how to reserve or pick low-contention times.
- **Access path:** direct SSH (on the QUB/EEECS network) vs Tailscale/VS Code Remote-SSH; whether the box is
  reachable off-campus or only on-site / via VPN.
- **Persistence-mode / power-cap** behaviour for sustained GCG runs (A5000 is 230 W per card per `nvidia-smi`).

## Superseded — the 2026-06-15 Windows reading (a different machine, NOT the plan)

> Kept for the audit trail. This was an RTX-4060 / 8 GB / Windows 11 desktop photographed from the Settings
> page; it is **not** `ecs3-0202` and is **no longer** the smoke/registered box.

| Component | Spec (superseded) |
|-----------|-------------------|
| GPU | NVIDIA GeForce RTX 4060 — 8 GB VRAM |
| CPU | Intel Core i7-14700KF, 3.40 GHz |
| RAM | 16.0 GB, 5600 MT/s |
| Storage | 954 GB (≈410 GB free) |

A laptop (`IMG_7283`, Core Ultra 9 275HX, 32 GB, GPU unverified) was also noted as a possible bf16-inference
candidate; with `ecs3-0202` now the registered box it is no longer needed.

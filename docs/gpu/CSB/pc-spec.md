# CSB — Local Box Hardware Spec

> **Source:** Windows "Device specifications" / "About" screens, photographed and read **2026-06-15**
> (`IMG_7285.heic` = this box; `IMG_7283.heic` = the laptop noted below). A photo of the Settings page omits
> driver, dedicated-vs-shared VRAM, and dGPU-variant detail — **re-verify on the box** with `nvidia-smi`,
> `wsl --status`, `systeminfo` before citing (audit trail, mirrors the Kelvin2 docs' "verify on node" rule).

## What it is

**CSB** is the local **consumer-GPU desktop** earmarked as the project's **non-registered smoke-test /
de-risking rig** — the cheap rehearsal stage *before* the granted **Kelvin2** A100/H100
([`../Overview.md`](../Overview.md)). It hosts **no** registered/pre-registered experiment; the scope boundary
is in [`plan.md`](./plan.md).

- **OS:** Windows 11 — but the intended runtime env is **WSL2 + CUDA**, not native Windows
  (see [`plan.md`](./plan.md) §Environment).

## Hardware (from the device-info screen)

| Component | Spec | Notes |
|-----------|------|-------|
| **GPU** | **NVIDIA GeForce RTX 4060 — 8 GB VRAM** | **the binding constraint.** Ada (AD107); desktop variant inferred from the desktop CPU below |
| CPU | Intel Core **i7-14700KF**, 3.40 GHz | Raptor Lake-R, 20C/28T; "K"=unlocked, "F"=no iGPU → **desktop** |
| RAM | **16.0 GB**, 5600 MT/s | low-ish; fine for 4-bit (weights live on GPU), tight if anything spills to host |
| Storage | 954 GB (≈410 GB free) | enough for the OpenVLA-7B checkpoint (~14 GB) + LIBERO assets |

### What 8 GB VRAM means for OpenVLA-7B (≈7.5B params)

| Precision | Weights ≈ | Fits 8 GB? |
|-----------|-----------|------------|
| bf16 — the **registered** precision | ~14 GB | ❌ no |
| int8 | ~7.5 GB | ⚠️ only just — OOM-prone once activations + the vision encoder are added |
| **4-bit (NF4)** | ~4 GB | ✅ **inference yes** — the only workable mode on CSB |

→ CSB can do **4-bit inference + small LIBERO rollouts**; it **cannot** do bf16 and **cannot** run the full GCG
matrix. Rationale + consequences in [`plan.md`](./plan.md).

## Other local machine on hand (NOT CSB)

| Machine | CPU | RAM | GPU | Verdict |
|---------|-----|-----|-----|---------|
| Laptop (`IMG_7283`) | Intel Core Ultra 9 **275HX** | 32 GB | **unverified** — the device-info page didn't show it | **Worth checking.** 275HX laptops usually pair an NVIDIA dGPU; if it carries a **16 GB** card it would beat CSB (bf16 inference becomes borderline-possible). Confirm via Task Manager → Performance → GPU, or `dxdiag`. |

## Not in the photo — verify on the box (audit trail)

- Exact **GPU driver / CUDA** version; whether **WSL2** is installed with working GPU passthrough
  (`wsl --status`, then `nvidia-smi` *inside* WSL2).
- Whether the 8 GB is all **dedicated** VRAM or partly shared.
- **Network:** is CSB on the **same LAN** as the Mac, or remote (e.g. on the QUB network)? This decides the
  SSH / Tailscale access path (see [`plan.md`](./plan.md) §Environment).
- Power/thermal headroom for sustained load.
- The laptop's GPU (row above).

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

- [ ] **1. Linux+CUDA env, `git clone`, install GPU deps** (`requirements-gpu.txt`) — *verify:* `torch.cuda.is_available() == True`; `nvidia-smi` sees both A5000s
- [ ] **2. Run the existing model-free test suite (395 tests)** — *verify:* parity with the Mac (same pass count)
- [ ] **3. Load OpenVLA-7B in bf16**, one forward on a dummy image+instruction — *verify:* a valid action vector; **fits on one 24 GB card** (the registered precision runs)
- [ ] **4. One LIBERO episode** (EGL) with the bf16 policy — *verify:* rollout completes; log schema matches the state-adapter / metric side
- [ ] **5. Attach the goal-action detector (L2)** to that real rollout — *verify:* detector ingests real OpenVLA actions end-to-end
- [ ] **6. GCG** — first a tiny run (few steps, 1 example), then the **D8 timing micro-bench** — *verify:* attack harness runs; record `s/target`, peak VRAM, max candidate-batch at 24 GB → **selects Branch N/N−/F (D8)**

Steps 1–5 de-risk the wiring; **step 6 produces registered measurements** (D4/D7/D8). If bf16 OOMs at step 3
(it should not at 24 GB) → fall back to memory relief on the 2nd card before any precision change.

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
| L5 | **CUDA 13.2 / driver 595 newer than the pin** | `flash-attn` / cu121 torch must be verified on the box; record the exact working env (invariant #8). |
| L6 | **Not a Kelvin2 mirror** | if Kelvin2 is ever used it is **different HW** → its own separate registration, never merged with A5000 results. |

## Reproducibility (registered runs — full discipline)

Pin seeds; capture `nvidia-smi` + `pip freeze` + git commit + **which card** into the run log; one variable per
run; output to **write-once `results/`** (no overwrite). Bring-up / throwaway smoke runs may go to
`results/_smoke/`, but **once step 6 is producing the D8/registered numbers they are real results** and follow
the playbook §8 protocol (CLAUDE.md reproducibility).

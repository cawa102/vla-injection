# CSB — What to Run Here (smoke-test / de-risking plan)

> **Status:** non-registered engineering rig. **Author-sanctioned** local de-risking — **not** a registered
> experiment. Every registered / pre-registered run stays on **Kelvin2** ([`../Overview.md`](../Overview.md),
> [`../Running.md`](../Running.md)). Hardware: [`pc-spec.md`](./pc-spec.md). Theme context: the Embodiment
> Evasion Tax (`docs/core/execution-playbook.md`).

## Why use CSB at all

Catch pipeline/wiring bugs **cheaply, before** burning scarce shared Kelvin2 A100/H100 walltime (3-day cap +
real queue waits — [`../Overview.md`](../Overview.md)). The integration risk —
**OpenVLA → LIBERO → goal-action detector (L2) → GCG** — is real, and debugging it on an 8 GB local card costs
nothing. **Goal = "the pipeline runs end-to-end on one example in 4-bit." NOT any measurement.**

## Hard scope boundary (read first)

CSB has 8 GB VRAM, so:

- **4-bit only.** bf16 (the registered precision) does not fit → CSB **cannot reproduce the registered model**.
- **No science numbers.** Per the precision/attack-fidelity argument — 4-bit degrades GCG input gradients;
  quantization is an *uncontrolled* robustness confound (violates one-variable-at-a-time); and both the L2
  FP-calibration and the L1 internal-probe do **not** transfer across precision — **nothing measured here is
  evidence for any EET claim** (H6-A oracle frontier / H6-D cross-layer tax).
- **Isolation.** CSB output → `results/_smoke/` (or a `csb-smoke` branch), **never** the write-once `results/`
  and never mixed with a pre-reg claim. Label every artefact "non-registered · 4-bit · CSB".
- **The full GCG / matched-attacker matrix does NOT run here** — that is Kelvin2-only.

## Environment

- Run inside **WSL2 + CUDA**, not native Windows (OpenVLA / bitsandbytes / MuJoCo·robosuite are Linux+CUDA-first).
- **Headless rendering** for LIBERO camera observations: `MUJOCO_GL=egl` on the NVIDIA GPU. (This is the GL wall
  the local *state-only* path was built to dodge; on CSB + EGL it is crossable.)
- Keep the env **Python 3.10** to match the Kelvin2 pin ([`../Start.md`](../Start.md) §3) so the handoff is 1:1.
- Access: simplest is **Tailscale** (Mac↔CSB) or **VS Code Remote-SSH → WSL**; or work on the box directly.
  Decide once the network situation in [`pc-spec.md`](./pc-spec.md) is known.

## The de-risking ladder (do in order; each step has a verify gate)

| # | Step | Verify (gate) |
|---|------|---------------|
| 1 | WSL2+CUDA env, `git clone`, install GPU deps | `torch.cuda.is_available() == True`; `nvidia-smi` sees the 4060 |
| 2 | Run the existing **model-free test suite (395 tests)** on CSB | parity with the Mac (same pass count) |
| 3 | Load **OpenVLA-7B in 4-bit**, one forward on a dummy image+instruction | a valid action vector comes out; **fits in 8 GB** ← the real "does the model run" test |
| 4 | **One LIBERO episode** (EGL) with the 4-bit policy | rollout completes; log schema matches the state-adapter / metric side |
| 5 | Attach the **goal-action detector (L2)** to that real rollout | detector ingests real OpenVLA actions end-to-end |
| 6 | **Tiny GCG** (few steps, 1 example) | attack harness runs end-to-end; observe the OOM behaviour at 8 GB |

If step 3 OOMs even at 4-bit → that is CSB's ceiling; stop and rely on Kelvin2.

## Handoff to Kelvin2

Everything that passes here transfers as "**raise the settings**": 4-bit → bf16, 1 example → full matrix,
interactive debug → `sbatch` on `k2-gpu-a100` ([`../Running.md`](../Running.md)). The payoff: by the time you
hold an A100, the wiring is already known-good, so cluster walltime buys **measurements**, not debugging.

## Limitations (this environment)

CSB's limits are not merely "slower than Kelvin2" — several are **hard walls** that change *what can be
concluded*, not just *how fast*:

| # | Limitation | Consequence |
|---|-----------|-------------|
| L1 | **8 GB VRAM** | bf16 (registered precision) won't load → **4-bit only**. GCG backward + candidate-batch eval are OOM-prone → forced tiny batches, or no GCG beyond a toy. |
| L2 | **16 GB system RAM** | No room for a bf16 CPU-offload fallback; MuJoCo + Python + model load can crowd host memory on larger rollouts. |
| L3 | **Consumer single GPU (RTX 4060)** | No multi-GPU; boost-clock + thermal/power throttling under sustained load → **timing is unstable**. |
| L4 | **Precision ≠ registered (4-bit vs bf16)** | The 4-bit model is a *different function* (different action outputs, task success, attack surface) → **not comparable to the bf16 claim**. |
| L5 | **Degraded GCG fidelity at 4-bit** | Input gradients through NF4 dequant are noisy → the attack is under-powered → attack outcomes here **cannot be trusted** as evasion-cost evidence. |
| L6 | **Quantization is an uncontrolled confound** | Folds a robustness change into any result → violates one-variable-at-a-time; the L2 FP-calibration and L1 internal-probe do not transfer across precision. |
| L7 | **WSL2 / Windows toolchain** | GPU passthrough, EGL headless rendering, bitsandbytes/flash_attn builds, and the WSL↔Windows filesystem boundary are extra failure points the Kelvin2 (CentOS 7 + `module`) path doesn't have. |
| L8 | **Not a Kelvin2 mirror** | Different OS / driver / CUDA, no Slurm → "passes on CSB" does **not** guarantee "passes on Kelvin2" for CUDA/driver-sensitive paths; re-verify ladder steps 3–6 on-node. |
| L9 | **Timing does NOT transfer to D8** | The **M1 on-GPU timing micro-bench** that selects Branch N / N− / F must be measured on the actual A100/H100 — **never** estimate `s/target` GCG cost from CSB (different GPU class + throttling). |

**Net:** CSB validates *wiring*, not *findings*. Use it to prove the pipeline executes end-to-end; treat every
number it produces as non-evidential.

## Do NOT expect on CSB

bf16 · the full GCG / matched-attacker matrix · the *tax* numbers (H6-D) · the oracle frontier (H6-A) ·
anything citable. All of that is Kelvin2.

## Reproducibility (applies even to throwaway smoke runs)

Pin seeds; capture `nvidia-smi` + `pip freeze` + the git commit into the `_smoke` log; one variable per run;
keep 4-bit-CSB logs clearly separable so they can never be mistaken for registered results (CLAUDE.md
reproducibility + the precision-confound argument above).

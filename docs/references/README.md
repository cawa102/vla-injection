# References (local copies)

Provenance manifest for papers stored locally. Saved for research / fair-use citation only —
do not redistribute. Verify the canonical license on each paper's source page before reuse.

| File | Title | arXiv | Source URL | Retrieved | SHA-256 | License |
|---|---|---|---|---|---|---|
| `2506.03350-robogcg-adversarial-attacks-vla.pdf` | Adversarial Attacks on Robotic Vision-Language-Action Models (RoboGCG) | 2506.03350 v1 (3 Jun 2025) | https://arxiv.org/pdf/2506.03350 | 2026-05-31 | `861f0c32e9df4b2f6c9c547207d3065a403038b973062b66206dafe4f974efab` | Not stated in PDF ("Preprint. Under review."); arXiv minimal perpetual non-exclusive distribution license applies — confirm on the abs page before redistribution. |

## RoboGCG — verified facts (full-text, 2026-05-31)

Authors: Jones, Robey, Zou, Ravichandran, Pappas, Hassani, Fredrikson, Kolter (Gray Swan AI + CMU + UPenn).
Code: `https://github.com/eliotjones1/robogcg`. 9 pp.

**Attack.** White-box GCG-optimized **textual** suffix, applied once at rollout start, eliciting
attacker-chosen low-level actions. Action space = 7-DoF, each dim discretized into 256 bins → **256^7** distinct actions (the paper's §4 writes
this as `7^256` — a notation slip; for 7 independent dims × 256 bins the count is 256^7).
Single-step attack on four LIBERO OpenVLA fine-tunes (Goal/Object/Spatial/10) reaches **>90% targeted-action
success** on Goal/Object/Spatial, 77.4% on Libero-10; matches found in 30–110 GCG steps (~3–10 min/success
on H100). Explicitly **not semantically linked to harm** — it proves *control authority* (reach a target
action), not dangerous-task success. Persistence up to ~28× nominal; environment-agnostic (sim + real images,
Table 2: SIMPLER real-image transfer overall ASR = 38.0%, HYDRA = 61.2%). Cross-*architecture* transfer
(TraceVLA/CogACT/OpenPi0): **no better than random** instructions.

**Defenses (§5 "VLA defenses" + Table 3).** Borrowed LLM-jailbreak defenses, eval over 120 random one-hot
target actions. ASR (%):

| Defense | Libero-10 | Libero-Goal | Libero-Object | Libero-Spatial |
|---|---|---|---|---|
| No Defense | 63.3 | 100.0 | 96.7 | 100.0 |
| Multimodal PF | 63.3 | 100.0 | 96.7 | 100.0 |
| LLM-Only PF | **0.0** | **0.0** | **0.0** | **0.0** |
| Smoothing | **0.0** | **0.0** | **0.0** | **0.0** |

- **Multimodal perplexity filter is useless** (ASR identical to No Defense) — image embeddings dominate the
  loss term, so multimodal perplexity does not see the suffix.
- **Text-only PF and smoothing zero the ASR, but the authors call both infeasible in practice:**
  - PF: threshold "depends entirely on the maximum perplexity of instructions seen on a held-out set, which
    cannot be known beforehand in open-world robotics applications." (no a-priori-settable operating point)
  - Smoothing: "results in a 0% success rate, but also corrupts the instructions, resulting in a **0% success
    rate on non-attacked tasks**." (destroys benign performance)
- Authors' conclusion: "This points to the need for notions of **VLA refusal** when attempts to subvert
  control are detected." → the defense problem is treated as **open**.

**Why this matters for T7.** RoboGCG shows the borrowed *input-side* defenses have **no usable operating
point** (PF threshold unknowable a priori; smoothing destroys benign; multimodal PF fails outright). T7's
FP-calibrated, benign-cost-measured *behavioral* (goal-action consistency) detector targets exactly that gap.
The perplexity / text-only filter remains a **baseline to beat**, beatable because (i) the realistic
multimodal PF fails, (ii) text PF has no settable threshold, (iii) it is blind to fluent / visual / adaptive
injection.

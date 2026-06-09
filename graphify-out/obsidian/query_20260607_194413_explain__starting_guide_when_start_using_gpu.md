---
source_file: "graphify-out/memory/query_20260607_194413_explain__starting_guide_when_start_using_gpu.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# query_20260607_194413_explain__starting_guide_when_start_using_gpu.md

## Connections
- [[Graphify query starting guide when using GPU]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`graphify-out/memory/query_20260607_194413_explain__starting_guide_when_start_using_gpu.md`


# Q: Explain: starting guide when start using GPU

## Answer

Maps to node 'GPU Day-1 Runbook (Kelvin2 / M1 gate)' (setup_gpu_runbook, docs/setup/gpu-runbook.md), community 8, degree 12. It is the Day-1 onboarding doc that turns GPU access into the M1 GO/NO-GO viability gate (H1) with no improvisation, each step ending in a logged write-once results/ artifact. It references docs/gpu/ cluster mechanics (Connection/Start/Running/Overview), implements plans_phase3_m1_gate, and references the GCG/L1 micro-bench, reproducibility protocol, untrusted quarantine, consistency metric (A) oracle, and LIBERO smoke test. Prescribed sequence: Pre-flight (SSH/MFA, clone, scratch, CUDA module, nvidia-smi) -> Step1 pinned envs (OpenVLA c8f03f48/py3.10.13/torch2.2.0/flash-attn2.5.5, LIBERO source, evasion_tax -e; RoboGCG isolated) Gate1 cuda.is_available -> Step2 checkpoints download+SHA256+provenance+licence -> Step3 benign LIBERO baseline center_crop True, >=300 benign for 1pct FPR / >=60 for 5pct primary -> Step4 RoboGCG targeted redirect, quarantine -> Step5 microbench_gcg selects compute branch N/N-/F (D4/D7/D8) deciding H6-D vs H6-A fallback -> Step6 metric-(A) separation must survive coarse operator-goal rung. GO requires all four (benign reproduced, targeted redirect, metric-A separation, branch selected) then M2->M3->M4. Companion nodes: local_prep_plan (pre-GPU prep) and setup_libero_local_notes_smoketest.

## Source Nodes

- GPU Day-1 Runbook (Kelvin2 / M1 gate)
- M1 environment + viability GO/NO-GO gate (H1)
- GCG/L1 micro-bench (branch selection, D8)
- Pre-GPU local preparation plan (M1/8GB)


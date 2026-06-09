---
type: community
cohesion: 0.16
members: 19
---

# GPU Runtime Guard

**Cohesion:** 0.16 - loosely connected
**Members:** 19 nodes

## Members
- [[GPU-node runtime guard for the modelGPU-dependent scripts (Task 9).  ``run_beni]] - rationale - src/evasion_tax/config/runtime.py
- [[Return ``True`` iff a CUDA-capable torch runtime is present.      On a local dev]] - rationale - src/evasion_tax/config/runtime.py
- [[Return the requires GPU node message printed when the guard fires.      Args]] - rationale - src/evasion_tax/config/runtime.py
- [[Tests for the GPU-node runtime guard (Task 9).  The modelGPU-dependent scripts]] - rationale - tests/evasion_tax/config/test_runtime.py
- [[calibrate.py]] - code - scripts/calibrate.py
- [[cuda_available()]] - code - src/evasion_tax/config/runtime.py
- [[gpu_required_message()]] - code - src/evasion_tax/config/runtime.py
- [[main()]] - code - scripts/calibrate.py
- [[main()_8]] - code - scripts/microbench_gcg.py
- [[main()_9]] - code - scripts/run_attack.py
- [[main()_10]] - code - scripts/run_benign.py
- [[microbench_gcg.py]] - code - scripts/microbench_gcg.py
- [[run_attack.py]] - code - scripts/run_attack.py
- [[run_benign.py]] - code - scripts/run_benign.py
- [[runtime.py]] - code - src/evasion_tax/config/runtime.py
- [[scripts_bootstrap.py (src sys.path bootstrap)]] - code - scripts/_bootstrap.py
- [[test_cuda_unavailable_on_local_host()]] - code - tests/evasion_tax/config/test_runtime.py
- [[test_gpu_message_names_the_stage_and_gpu()]] - code - tests/evasion_tax/config/test_runtime.py
- [[test_runtime.py]] - code - tests/evasion_tax/config/test_runtime.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/GPU_Runtime_Guard
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Config Schema & Immutability]]
- 2 edges to [[_COMMUNITY_Figure Script Entrypoints]]
- 2 edges to [[_COMMUNITY_Metric Separation Demo]]
- 1 edge to [[_COMMUNITY_Detector Calibration]]
- 1 edge to [[_COMMUNITY_Run Logging & Rollout Demo]]
- 1 edge to [[_COMMUNITY_Evaluate CLI Tests]]

## Top bridge nodes
- [[scripts_bootstrap.py (src sys.path bootstrap)]] - degree 9, connects to 4 communities
- [[main()_10]] - degree 7, connects to 2 communities
- [[main()_9]] - degree 6, connects to 1 community
- [[main()_8]] - degree 5, connects to 1 community
- [[main()]] - degree 3, connects to 1 community
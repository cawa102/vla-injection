---
type: community
members: 67
---

# Config Schema & Immutability

**Members:** 67 nodes

## Members
- [[A fully-pinned run configuration (the snapshot logged with every run).]] - rationale - src/evasion_tax/config/schema.py
- [[AttackConfig]] - code - src/evasion_tax/config/schema.py
- [[Base for every config section immutable + rejects unknown fields.]] - rationale - src/evasion_tax/config/schema.py
- [[BaseModel]] - code
- [[Calibrationtest manifests (disjointness enforced at eval time).]] - rationale - src/evasion_tax/config/schema.py
- [[Config]] - code - src/evasion_tax/config/schema.py
- [[Consistency metric (A) settings — the causal prefix-window length ``k``.]] - rationale - src/evasion_tax/config/schema.py
- [[EnvConfig]] - code - src/evasion_tax/config/schema.py
- [[EvalConfig]] - code - src/evasion_tax/config/schema.py
- [[Flatten a nested dict into ``{dotted.path leaf_value}`` (lists are leaves).]] - rationale - src/evasion_tax/config/schema.py
- [[GPU-node runtime guard for the modelGPU-dependent scripts (Task 9).  ``run_beni]] - rationale - src/evasion_tax/config/runtime.py
- [[Load and validate a pinned-config YAML file.      Args         path Path to a]] - rationale - src/evasion_tax/config/schema.py
- [[MetricConfig]] - code - src/evasion_tax/config/schema.py
- [[ModelConfig]] - code - src/evasion_tax/config/schema.py
- [[Path_2]] - code - tests/evasion_tax/config/test_schema.py
- [[Pinned-config schema + one-variable-diff (Task 9).  A run's parameters are valid]] - rationale - src/evasion_tax/config/schema.py
- [[Put ``src`` on ``sys.path`` so standalone scripts can ``import evasion_tax``.]] - rationale - scripts/_bootstrap.py
- [[Return ``True`` iff a CUDA-capable torch runtime is present.      On a local dev]] - rationale - src/evasion_tax/config/runtime.py
- [[Return the requires GPU node message printed when the guard fires.      Args]] - rationale - src/evasion_tax/config/runtime.py
- [[Return the sorted dotted paths of leaf fields that differ between configs.]] - rationale - src/evasion_tax/config/schema.py
- [[SplitManifest]] - code - src/evasion_tax/config/schema.py
- [[SplitsConfig]] - code - src/evasion_tax/config/schema.py
- [[StrPath]] - code - src/evasion_tax/config/schema.py
- [[Tests for the GPU-node runtime guard (Task 9).  The modelGPU-dependent scripts]] - rationale - tests/evasion_tax/config/test_runtime.py
- [[Tests for the pinned-config schema (Task 9).  ``Config`` is a frozen pydantic mo]] - rationale - tests/evasion_tax/config/test_schema.py
- [[The LIBERO suite + the tasksepisode length a run covers.]] - rationale - src/evasion_tax/config/schema.py
- [[The attack arm and its window-scored target budget (decision D2).]] - rationale - src/evasion_tax/config/schema.py
- [[The condition matrix to evaluate plus the calibrationtest splits.]] - rationale - src/evasion_tax/config/schema.py
- [[The ids on each disjointness axis for one split (calib or test).]] - rationale - src/evasion_tax/config/schema.py
- [[The victim policy checkpoint and its action-decoding key.]] - rationale - src/evasion_tax/config/schema.py
- [[_Frozen]] - code - src/evasion_tax/config/schema.py
- [[_Frozen (immutable, extra-forbid base)]] - code - src/evasion_tax/config/schema.py
- [[_bootstrap.py]] - code - scripts/_bootstrap.py
- [[_flatten]] - code - src/evasion_tax/config/schema.py
- [[_flatten()]] - code - src/evasion_tax/config/schema.py
- [[_valid_dict()]] - code - tests/evasion_tax/config/test_schema.py
- [[_write()]] - code - tests/evasion_tax/config/test_schema.py
- [[calibrate.py]] - code - scripts/calibrate.py
- [[cuda_available()]] - code - src/evasion_tax/config/runtime.py
- [[gpu_required_message()]] - code - src/evasion_tax/config/runtime.py
- [[load_config()]] - code - src/evasion_tax/config/schema.py
- [[main()]] - code - scripts/calibrate.py
- [[main()_8]] - code - scripts/microbench_gcg.py
- [[main()_9]] - code - scripts/run_attack.py
- [[main()_10]] - code - scripts/run_benign.py
- [[microbench_gcg.py]] - code - scripts/microbench_gcg.py
- [[one_variable_diff()]] - code - src/evasion_tax/config/schema.py
- [[run_attack.py]] - code - scripts/run_attack.py
- [[run_benign.py]] - code - scripts/run_benign.py
- [[runtime.py]] - code - src/evasion_tax/config/runtime.py
- [[schema.py]] - code - src/evasion_tax/config/schema.py
- [[scripts_bootstrap.py (src sys.path bootstrap)]] - code - scripts/_bootstrap.py
- [[test_committed_example_config_is_valid()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_config_is_immutable()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_cuda_unavailable_on_local_host()]] - code - tests/evasion_tax/config/test_runtime.py
- [[test_empty_fpr_targets_raises()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_fpr_target_out_of_unit_interval_raises()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_gpu_message_names_the_stage_and_gpu()]] - code - tests/evasion_tax/config/test_runtime.py
- [[test_missing_required_field_raises()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_one_variable_diff_detects_single_leaf()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_one_variable_diff_empty_for_identical()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_one_variable_diff_reports_multiple_changes()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_out_of_range_metric_k_raises()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_runtime.py]] - code - tests/evasion_tax/config/test_runtime.py
- [[test_schema.py]] - code - tests/evasion_tax/config/test_schema.py
- [[test_unknown_field_is_forbidden()]] - code - tests/evasion_tax/config/test_schema.py
- [[test_valid_config_loads()]] - code - tests/evasion_tax/config/test_schema.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Config_Schema__Immutability
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_Eval Harness & Power]]
- 2 edges to [[_COMMUNITY_Figure Generation]]
- 2 edges to [[_COMMUNITY_Metric Separation Demo]]
- 2 edges to [[_COMMUNITY_CalibTest Split Disjointness]]
- 1 edge to [[_COMMUNITY_Detector Calibration]]
- 1 edge to [[_COMMUNITY_Run Logging & Rollout Demo]]

## Top bridge nodes
- [[scripts_bootstrap.py (src sys.path bootstrap)]] - degree 10, connects to 4 communities
- [[schema.py]] - degree 32, connects to 1 community
- [[load_config()]] - degree 19, connects to 1 community
- [[_Frozen]] - degree 12, connects to 1 community
- [[main()_10]] - degree 7, connects to 1 community
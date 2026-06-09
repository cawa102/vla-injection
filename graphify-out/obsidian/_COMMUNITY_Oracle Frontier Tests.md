---
type: community
members: 23
---

# Oracle Frontier Tests

**Members:** 23 nodes

## Members
- [[A minimal reach-the-cube scene with one distractor.]] - rationale - tests/evasion_tax/attack/test_dynamics.py
- [[An ``(n_steps, 7)`` array repeating ``vec`` (a length-7 action).]] - rationale - tests/evasion_tax/attack/test_dynamics.py
- [[Deterministic kinematic integrator for local tests (no LIBERO).      The end-eff]] - rationale - src/evasion_tax/attack/dynamics.py
- [[Rollout (shared record)]] - code - src/evasion_tax/records.py
- [[SyntheticDynamics]] - code - src/evasion_tax/attack/dynamics.py
- [[Tests for the action→state dynamics seam (playbook §4b-(II)).  The metric-A orac]] - rationale - tests/evasion_tax/attack/test_dynamics.py
- [[_validate_actions]] - code - src/evasion_tax/attack/dynamics.py
- [[const_actions()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[make_scenario()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_attacked_flag_defaults_true_and_is_overridable()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_dynamics.py]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_ee_pos_integrates_position_deltas_cumulatively()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_emitted_privileged_state_is_adapter_consumable()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_gripper_open_maps_from_gripper_dim()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_real_dynamics_is_a_gpu_only_stub()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_scenario_is_immutable()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_scenario_rejects_bad_position_length()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_scenario_rejects_non_positive_n_steps()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_scenario_rejects_target_region_not_in_objects()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_synthetic_dynamics_conforms_to_protocol()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_synthetic_dynamics_is_deterministic()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_synthetic_dynamics_rejects_wrong_action_shape()]] - code - tests/evasion_tax/attack/test_dynamics.py
- [[test_synthetic_dynamics_returns_rollout_of_n_steps()]] - code - tests/evasion_tax/attack/test_dynamics.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Oracle_Frontier_Tests
SORT file.name ASC
```

## Connections to other communities
- 16 edges to [[_COMMUNITY_Rollout]]
- 5 edges to [[_COMMUNITY_Detector Calibration]]
- 4 edges to [[_COMMUNITY_Run Logging & Rollout Demo]]
- 3 edges to [[_COMMUNITY_Synthetic State Fixtures]]
- 3 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]
- 2 edges to [[_COMMUNITY_Goal-Agnostic Anomaly Baseline]]
- 1 edge to [[_COMMUNITY_L1 Internal Probe]]
- 1 edge to [[_COMMUNITY_L0 Perplexity Filter]]
- 1 edge to [[_COMMUNITY_Metric (A) Tests]]
- 1 edge to [[_COMMUNITY_Rollout Records Tests]]

## Top bridge nodes
- [[SyntheticDynamics]] - degree 31, connects to 6 communities
- [[Rollout (shared record)]] - degree 9, connects to 5 communities
- [[test_dynamics.py]] - degree 26, connects to 2 communities
- [[test_emitted_privileged_state_is_adapter_consumable()]] - degree 5, connects to 1 community
- [[test_real_dynamics_is_a_gpu_only_stub()]] - degree 4, connects to 1 community
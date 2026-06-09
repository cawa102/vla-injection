---
type: community
members: 64
---

# Run Logging & Rollout Demo

**Members:** 64 nodes

## Members
- [[.__init__()_3]] - code - src/evasion_tax/repro/run_logger.py
- [[.__init__()_4]] - code - src/evasion_tax/repro/run_logger.py
- [[.dir()]] - code - src/evasion_tax/repro/run_logger.py
- [[.start()]] - code - src/evasion_tax/repro/run_logger.py
- [[.write()]] - code - src/evasion_tax/repro/run_logger.py
- [[.write_array()]] - code - src/evasion_tax/repro/run_logger.py
- [[A benign sequence overwritten, from step 2 on, to sit inside the attack region.]] - rationale - scripts/demo_rollout.py
- [[A live handle to one run directory; all writes refuse to overwrite.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Any_1]] - code - scripts/demo_rollout.py
- [[Create a fresh run directory and write its ``run.json`` protocol block.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Create write-once, UTC-timestamped run directories under ``results_root``.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Format a UTC datetime as a filesystem-safe ``YYYY-MM-DDTHH-MM-SSZ`` token.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Map a robosuiteLIBERO obs dict to a PrivilegedState dict (real ground truth).]] - rationale - scripts/demo_rollout.py
- [[Path_1]] - code - src/evasion_tax/repro/run_logger.py
- [[Produce a benign ``(n_steps, 7)`` action sequence with no attacker target.]] - rationale - scripts/demo_rollout.py
- [[Return a ``horizon - env`` factory for state-only LiftPanda, or None.]] - rationale - scripts/demo_rollout.py
- [[Return the current time as a timezone-aware UTC ``datetime``.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Roll ``actions`` out in a fresh robosuite env, recording real ground truth.]] - rationale - scripts/demo_rollout.py
- [[Roll ``actions`` out through SyntheticDynamics (no sim) — always available.]] - rationale - scripts/demo_rollout.py
- [[Rollout_1]] - code - scripts/demo_rollout.py
- [[RunHandle]] - code - src/evasion_tax/repro/run_logger.py
- [[RunHandle (write-once)]] - code - src/evasion_tax/repro/run_logger.py
- [[RunLogger]] - code - src/evasion_tax/repro/run_logger.py
- [[RunLogger_1]] - code - src/evasion_tax/repro/run_logger.py
- [[Save a numpy array as ``name.npy``; raise if the file already exists.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Serialise a Rollout to a JSON-able dict (frozen dataclasses - dicts).]] - rationale - scripts/demo_rollout.py
- [[Step a robosuite env and return the obs dict (4- or 5-tuple gym APIs).]] - rationale - scripts/demo_rollout.py
- [[StrPath_4]] - code - src/evasion_tax/repro/run_logger.py
- [[Tests for the write-once RunLogger (invariant 5).]] - rationale - tests/evasion_tax/repro/test_run_logger.py
- [[The run's output directory.]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Window-score the rollout against the attacker target (decision D2).]] - rationale - scripts/demo_rollout.py
- [[Write ``obj`` as ``name.json``; raise if the file already exists.          Arg]] - rationale - src/evasion_tax/repro/run_logger.py
- [[Write-once run logger — the gatekeeper for the write-once invariant (5).  Every]] - rationale - src/evasion_tax/repro/run_logger.py
- [[_attack_outcome()]] - code - scripts/demo_rollout.py
- [[_attacked_actions()]] - code - scripts/demo_rollout.py
- [[_benign_actions()]] - code - scripts/demo_rollout.py
- [[_extract_privileged_state()]] - code - scripts/demo_rollout.py
- [[_make_robosuite_env_maker()]] - code - scripts/demo_rollout.py
- [[_print_report()_1]] - code - scripts/demo_rollout.py
- [[_rollout_robosuite()]] - code - scripts/demo_rollout.py
- [[_rollout_synthetic()]] - code - scripts/demo_rollout.py
- [[_rollout_to_json()]] - code - scripts/demo_rollout.py
- [[_step_env()]] - code - scripts/demo_rollout.py
- [[_timestamp_slug()]] - code - src/evasion_tax/repro/run_logger.py
- [[_utc_now()]] - code - src/evasion_tax/repro/run_logger.py
- [[capture_env]] - code - src/evasion_tax/repro/env_capture.py
- [[datetime]] - code - src/evasion_tax/repro/run_logger.py
- [[demo_metric_separation._generate]] - code - scripts/demo_metric_separation.py
- [[demo_rollout.py]] - code - scripts/demo_rollout.py
- [[fixed_now()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[main()_3]] - code - scripts/demo_rollout.py
- [[ndarray]] - code - scripts/demo_rollout.py
- [[ndarray_9]] - code - src/evasion_tax/repro/run_logger.py
- [[run_logger.py]] - code - src/evasion_tax/repro/run_logger.py
- [[test_default_now_is_utc_and_succeeds()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_run_json_contains_required_protocol_fields()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_run_logger.py]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_second_start_with_colliding_dir_raises()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_start_creates_timestamped_dir()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_start_does_not_mutate_passed_config()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_write_array_refuses_to_overwrite()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_write_array_saves_npy_and_round_trips()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_write_json_creates_file_and_returns_path()]] - code - tests/evasion_tax/repro/test_run_logger.py
- [[test_write_refuses_to_overwrite_existing_file()]] - code - tests/evasion_tax/repro/test_run_logger.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Run_Logging__Rollout_Demo
SORT file.name ASC
```

## Connections to other communities
- 14 edges to [[_COMMUNITY_Rollout]]
- 6 edges to [[_COMMUNITY_Metric Separation Demo]]
- 4 edges to [[_COMMUNITY_Oracle Frontier Tests]]
- 2 edges to [[_COMMUNITY_Environment Capture]]
- 1 edge to [[_COMMUNITY_Figure Generation]]
- 1 edge to [[_COMMUNITY_Config Schema & Immutability]]
- 1 edge to [[_COMMUNITY_LIBERO State Smoketest]]
- 1 edge to [[_COMMUNITY_CalibTest Split Disjointness]]
- 1 edge to [[_COMMUNITY_Stats Provenance]]

## Top bridge nodes
- [[RunLogger]] - degree 24, connects to 3 communities
- [[main()_3]] - degree 14, connects to 3 communities
- [[Any_1]] - degree 14, connects to 2 communities
- [[ndarray]] - degree 12, connects to 2 communities
- [[Rollout_1]] - degree 12, connects to 2 communities
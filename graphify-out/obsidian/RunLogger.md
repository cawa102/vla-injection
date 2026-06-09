---
source_file: "src/evasion_tax/repro/run_logger.py"
type: "code"
community: "Run Logging & Rollout Demo"
location: "L93"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Run_Logging__Rollout_Demo
---

# RunLogger

## Connections
- [[.__init__()_4]] - `method` [EXTRACTED]
- [[.start()]] - `method` [EXTRACTED]
- [[Any]] - `uses` [INFERRED]
- [[Any_1]] - `uses` [INFERRED]
- [[Create write-once, UTC-timestamped run directories under ``results_root``.]] - `rationale_for` [EXTRACTED]
- [[Rollout]] - `uses` [INFERRED]
- [[Rollout_1]] - `uses` [INFERRED]
- [[Score]] - `uses` [INFERRED]
- [[main()_1]] - `calls` [INFERRED]
- [[main()_2]] - `calls` [INFERRED]
- [[main()_3]] - `calls` [INFERRED]
- [[main()_4]] - `calls` [INFERRED]
- [[ndarray]] - `uses` [INFERRED]
- [[run_logger.py]] - `contains` [EXTRACTED]
- [[test_default_now_is_utc_and_succeeds()]] - `calls` [INFERRED]
- [[test_run_json_contains_required_protocol_fields()]] - `calls` [INFERRED]
- [[test_run_logger.py]] - `references` [EXTRACTED]
- [[test_second_start_with_colliding_dir_raises()]] - `calls` [INFERRED]
- [[test_start_creates_timestamped_dir()]] - `calls` [INFERRED]
- [[test_start_does_not_mutate_passed_config()]] - `calls` [INFERRED]
- [[test_write_array_refuses_to_overwrite()]] - `calls` [INFERRED]
- [[test_write_array_saves_npy_and_round_trips()]] - `calls` [INFERRED]
- [[test_write_json_creates_file_and_returns_path()]] - `calls` [INFERRED]
- [[test_write_refuses_to_overwrite_existing_file()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Run_Logging__Rollout_Demo
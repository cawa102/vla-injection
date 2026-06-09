---
type: community
members: 14
---

# Deterministic Seeding

**Members:** 14 nodes

## Members
- [[A process-stable 64-bit seed derived from arbitrary parts.      Hashes ``.joi]] - rationale - src/evasion_tax/repro/seeds.py
- [[Deterministic seeding across Python ``random``, NumPy, and (optionally) torch.]] - rationale - src/evasion_tax/repro/seeds.py
- [[Seed Python ``random``, NumPy, and torch (if importable).      Args         see]] - rationale - src/evasion_tax/repro/seeds.py
- [[Tests for the deterministic seeding helper.]] - rationale - tests/evasion_tax/repro/test_seeds.py
- [[seed_everything()]] - code - src/evasion_tax/repro/seeds.py
- [[seeds.py]] - code - src/evasion_tax/repro/seeds.py
- [[stable_seed()]] - code - src/evasion_tax/repro/seeds.py
- [[test_different_seeds_give_different_numpy_draws()]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_does_not_mutate_inputs_and_returns_fresh_dict()]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_returns_dict_recording_applied_seed()]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_same_seed_gives_identical_numpy_draws()]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_same_seed_gives_identical_python_random_draws()]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_seeds.py]] - code - tests/evasion_tax/repro/test_seeds.py
- [[test_torch_absent_is_recorded_not_seeded()]] - code - tests/evasion_tax/repro/test_seeds.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Deterministic_Seeding
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_run_condition_matrix]]
- 1 edge to [[_COMMUNITY_Rollout]]
- 1 edge to [[_COMMUNITY_L1 Internal Probe]]

## Top bridge nodes
- [[stable_seed()]] - degree 4, connects to 2 communities
- [[seeds.py]] - degree 7, connects to 1 community
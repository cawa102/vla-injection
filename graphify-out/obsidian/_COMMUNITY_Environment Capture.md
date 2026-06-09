---
type: community
members: 22
---

# Environment Capture

**Members:** 22 nodes

## Members
- [[Capture a reproducibility snapshot of the current environment.      Returns]] - rationale - src/evasion_tax/repro/env_capture.py
- [[Capture the runtime environment for reproducibility logging.  Records platform,]] - rationale - src/evasion_tax/repro/env_capture.py
- [[Return ``(torch_version, cuda_version, driver_version)``.      All three are ``N]] - rationale - src/evasion_tax/repro/env_capture.py
- [[Return a ``{distribution version}`` snapshot of installed packages.      Uses `]] - rationale - src/evasion_tax/repro/env_capture.py
- [[Return the current ``HEAD`` commit hash, or ``None`` outside a repo.]] - rationale - src/evasion_tax/repro/env_capture.py
- [[SimpleNamespace]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[Tests for the environment-capture helper.]] - rationale - tests/evasion_tax/repro/test_env_capture.py
- [[_dependency_snapshot()]] - code - src/evasion_tax/repro/env_capture.py
- [[_fake_torch()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[_git_commit()]] - code - src/evasion_tax/repro/env_capture.py
- [[_torch_versions()]] - code - src/evasion_tax/repro/env_capture.py
- [[capture_env()]] - code - src/evasion_tax/repro/env_capture.py
- [[env_capture.py]] - code - src/evasion_tax/repro/env_capture.py
- [[test_cuda_and_torch_are_none_locally()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_dependencies_snapshot_is_present()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_driver_version_none_when_cuda_unavailable()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_driver_version_resolves_with_mocked_cuda_torch()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_env_capture.py]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_git_commit_is_resolved_in_this_repo()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_never_raises_on_this_machine()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_platform_and_python_version_are_non_empty_strings()]] - code - tests/evasion_tax/repro/test_env_capture.py
- [[test_returns_dict_with_required_keys()]] - code - tests/evasion_tax/repro/test_env_capture.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Environment_Capture
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Run Logging & Rollout Demo]]

## Top bridge nodes
- [[capture_env()]] - degree 13, connects to 1 community
- [[env_capture.py]] - degree 10, connects to 1 community
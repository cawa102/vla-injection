---
source_file: "src/evasion_tax/repro/env_capture.py"
type: "code"
community: "Environment Capture"
location: "L75"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Environment_Capture
---

# capture_env()

## Connections
- [[.start()]] - `calls` [INFERRED]
- [[Capture a reproducibility snapshot of the current environment.      Returns]] - `rationale_for` [EXTRACTED]
- [[_dependency_snapshot()]] - `calls` [EXTRACTED]
- [[_git_commit()]] - `calls` [EXTRACTED]
- [[_torch_versions()]] - `calls` [EXTRACTED]
- [[env_capture.py]] - `contains` [EXTRACTED]
- [[test_cuda_and_torch_are_none_locally()]] - `calls` [INFERRED]
- [[test_dependencies_snapshot_is_present()]] - `calls` [INFERRED]
- [[test_env_capture.py]] - `references` [EXTRACTED]
- [[test_git_commit_is_resolved_in_this_repo()]] - `calls` [INFERRED]
- [[test_never_raises_on_this_machine()]] - `calls` [INFERRED]
- [[test_platform_and_python_version_are_non_empty_strings()]] - `calls` [INFERRED]
- [[test_returns_dict_with_required_keys()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Environment_Capture
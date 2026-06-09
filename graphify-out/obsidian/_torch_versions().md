---
source_file: "src/evasion_tax/repro/env_capture.py"
type: "code"
community: "Environment Capture"
location: "L47"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Environment_Capture
---

# _torch_versions()

## Connections
- [[Return ``(torch_version, cuda_version, driver_version)``.      All three are ``N]] - `rationale_for` [EXTRACTED]
- [[capture_env()]] - `calls` [EXTRACTED]
- [[env_capture.py]] - `contains` [EXTRACTED]
- [[test_driver_version_none_when_cuda_unavailable()]] - `calls` [INFERRED]
- [[test_driver_version_resolves_with_mocked_cuda_torch()]] - `calls` [INFERRED]
- [[test_env_capture.py]] - `references` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Environment_Capture
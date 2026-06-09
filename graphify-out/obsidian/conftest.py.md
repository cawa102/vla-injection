---
source_file: "tests/evasion_tax/conftest.py"
type: "code"
community: "Shared pytest fixtures for the evasion_t"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Shared_pytest_fixtures_for_the_evasion_t
---

# conftest.py

## Connections
- [[Shared pytest fixtures for the evasion_tax test suite.  Tests must never write t]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Shared_pytest_fixtures_for_the_evasion_t

## 📄 Source

`tests/evasion_tax/conftest.py`

```python
"""Shared pytest fixtures for the evasion_tax test suite.

Tests must never write to the real ``results/`` directory (write-once invariant);
use ``tmp_path`` / the fixtures here instead.
"""
```


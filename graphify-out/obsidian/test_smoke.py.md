---
source_file: "tests/evasion_tax/test_smoke.py"
type: "code"
community: "Package Smoke Test"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Package_Smoke_Test
---

# test_smoke.py

## Connections
- [[evasion_tax package]] - `references` [EXTRACTED]
- [[test_package_imports()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Package_Smoke_Test

## 📄 Source

`tests/evasion_tax/test_smoke.py`

```python
import evasion_tax


def test_package_imports():
    assert evasion_tax.__version__
```


---
source_file: "tests/evasion_tax/metric/conftest.py"
type: "code"
community: "Make this directory's helper modules imp"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Make_this_directorys_helper_modules_imp
---

# conftest.py

## Connections
- [[Make this directory's helper modules importable by bare name.  Under ``--import-]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Make_this_directorys_helper_modules_imp

## 📄 Source

`tests/evasion_tax/metric/conftest.py`

```python
"""Make this directory's helper modules importable by bare name.

Under ``--import-mode=importlib`` with no ``__init__.py`` in ``tests/``, sibling
modules are not automatically importable. Putting this directory on ``sys.path``
lets the metric tests (Task 4 here, Task 5 later) share ``fixtures_state`` via a
plain ``import fixtures_state`` without any per-test path manipulation.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
```


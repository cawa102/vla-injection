---
source_file: "scripts/_bootstrap.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# _bootstrap.py

## Connections
- [[Put ``src`` on ``sys.path`` so standalone scripts can ``import evasion_tax``.]] - `rationale_for` [EXTRACTED]
- [[scripts_bootstrap.py (src sys.path bootstrap)]] - `defined_in` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`scripts/_bootstrap.py`

```python
"""Put ``src/`` on ``sys.path`` so standalone scripts can ``import evasion_tax``.

The editable install is unreliable on this host (uv regenerates its ``.pth`` on
every run), so each script imports this module first — its import side effect
prepends the source tree to ``sys.path``. Keeping it in one place means the six
Task-9 scripts share one bootstrap instead of repeating the path dance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```


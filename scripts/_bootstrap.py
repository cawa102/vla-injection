"""Put ``src/`` on ``sys.path`` so standalone scripts can ``import t7``.

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

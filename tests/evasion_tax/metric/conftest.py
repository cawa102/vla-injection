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

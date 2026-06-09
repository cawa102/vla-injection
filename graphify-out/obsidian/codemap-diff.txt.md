---
source_file: ".reports/codemap-diff.txt"
type: "document"
community: "Project Map & Session Memory"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/Project_Map__Session_Memory
---

# codemap-diff.txt

## Connections
- [[Codemap scan report (2026-06-05)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/Project_Map__Session_Memory

## 📄 Source

`.reports/codemap-diff.txt`

CODEMAP SCAN REPORT
===================
Date:        2026-06-05
Command:     /update-codemaps (re-run)
Project:     Embodiment Evasion Tax — MSc VLA-security research
Type:        single Python package (model-free core + Protocol-gated GPU seams); no routes / DB / UI

FILES SCANNED
-------------
src/evasion_tax      36 .py  (27 modules + 9 __init__)  ~4.3k LOC, 8 subpackages + root
scripts/     9 .py  (CLI entry points; GPU-marked ones cuda-guarded)
tests/      31 .py  (28 test_* modules + 2 conftest + 1 fixtures; 354 test fns)
configs/     2       (example_m2.yaml, env/requirements-gpu.txt)
pyproject.toml

CHANGES SINCE PRIOR CODEMAP (06-05, same day)
---------------------------------------------
Source code:    UNCHANGED. Newest src/scripts/configs mtime = 06-03; no git commits since 06-05 04:45.
                => substantive content diff ~= 0% (no new routes/services/deps/modules).

Documentation:  RESTRUCTURED + CORRECTED.
  - Split the prior single-file docs/CODEMAPS/README.md into per-area maps per the
    /update-codemaps template: architecture.md, modules.md, data.md, dependencies.md.
    README.md is now a short index.
  - Removed a FABRICATED claim from the prior scan summary: it asserted a "global doc-guard
    hook blocks creating arbitrary .md/.txt files (allows only README.md)" as the reason for
    consolidating into one file. Verified false — no such hook exists in ~/.claude/settings.json
    or .claude/settings.local.json. The per-area split is therefore restored.
  - Corrected file counts: src is 36 .py / ~4.3k LOC (prior said "35 files / ~4.0k LOC").
    backend.md / frontend.md remain absent, but now for the correct reason (no HTTP API, no UI),
    not a non-existent hook.

NEW DEPENDENCIES DETECTED
-------------------------
None.

ARCHITECTURE CHANGES
--------------------
None (no new routes, services, modules, or external integrations).

STALENESS
---------
Codemaps current as of 2026-06-05. Regenerate after each milestone (M-tag) or when
src/evasion_tax subpackages change. results/ still absent (write-once logs created on first GPU run).


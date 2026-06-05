<!-- Generated: 2026-06-05 | Files scanned: 76 (36 src · 9 scripts · 31 tests) | Token estimate: ~300 -->

# CODEMAPS — Embodiment Evasion Tax

Token-lean architecture map for AI context. MSc research code, **not** an app: no HTTP API, no UI, no DB.
Python package `src/evasion_tax` (model-free, unit-tested on M1/8 GB) + CLI `scripts/` (GPU-node-only runs).
Source of truth for *what & why* = `docs/core/execution-playbook.md`; these maps cover *structure*.

## Map files
| File | Covers |
|------|--------|
| [architecture.md](architecture.md) | What it measures · model-free/GPU split · data flow · headline split · milestones |
| [modules.md](modules.md) | Per-package public symbols (stands in for `backend.md` — library, not HTTP backend) |
| [data.md](data.md) | `records.py` contract · cross-layer `UnitOutcome` · reproducibility invariants · provenance |
| [dependencies.md](dependencies.md) | Runtime deps · heavy GPU stack · external artifacts · CLI entry points |

`backend.md` / `frontend.md` from the `/update-codemaps` template are **omitted**: there is no HTTP
API and no UI. The module map fills the package-internals role those files would serve in a service.

## Scan summary
**2026-06-05** — re-run of `/update-codemaps`.
- Source unchanged since the prior (06-05) generation (newest `src` file 06-03; no commits since) →
  content diff ≈ 0%. This pass **split** the previously consolidated single file into the per-area maps
  above and corrected the file counts.
- Scanned: 36 `src/evasion_tax` `.py` (27 modules · 9 `__init__`, ~4.3k LOC) · 9 `scripts` · 31 files in `tests/`
  (28 `test_*` modules + conftest/fixtures, 354 test fns) · `pyproject.toml` · `configs/`.
- Structure: single Python package, layered model-free core + `Protocol`-gated GPU seams; no routes/DB/UI.
- Regenerate after each milestone (M-tag) or when `src/evasion_tax` subpackages change.

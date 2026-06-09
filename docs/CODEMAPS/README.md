<!-- Generated: 2026-06-05 · reconciled 2026-06-09 | Files scanned: 85 (37 src · 12 scripts · 36 tests) | Token estimate: ~300 -->

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
**2026-06-09** — manual reconciliation of the 2026-06-05 `/update-codemaps` maps after the `state_libero.py`
pull-forward (Tier-L LIBERO now runs locally — execution-playbook §10, 2026-06-09).
- Since the 06-05 generation: `src/evasion_tax/metric/state_libero.py` (`LiberoStateAdapter`) added 06-09;
  3 demo scripts added (`demo_rollout` / `demo_metric_separation` / `demo_figures`). Counts + the Tier-R→Tier-L
  framing refreshed across the four maps above; a full `/update-codemaps` re-run is still the canonical regen.
- Scanned: 37 `src/evasion_tax` `.py` (28 modules · 9 `__init__`, ~4.7k LOC) · 12 `scripts` · 36 files in `tests/`
  (30 `test_*` modules + conftest/fixtures, **395 tests** collected) · `pyproject.toml` · `configs/`.
- Structure: single Python package, layered model-free core + `Protocol`-gated GPU seams; no routes/DB/UI.
- Regenerate after each milestone (M-tag) or when `src/evasion_tax` subpackages change.

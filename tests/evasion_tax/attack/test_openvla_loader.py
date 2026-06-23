"""Tests for the shared frozen-OpenVLA loader/target-builder (bench Task 4).

The real load + target build against bf16 OpenVLA-7B is GPU-only and guarded (torch /
transformers / PIL imported inside the functions). Off-GPU we pin only the structural
contract the bench driver (Task 6) and the microbench both depend on: the module imports
torch-free, and the public functions keep the exact signatures the call sites pass.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from evasion_tax.attack.openvla_loader import (
    DEFAULT_INSTRUCTION,
    build_target,
    load_frozen_openvla,
)


def test_loader_module_imports_no_torch_stack_at_module_top():
    # Same invariant as the gcg_openvla seam: importable on the CUDA-free mac.
    import evasion_tax.attack.openvla_loader as loader_mod

    assert loader_mod.__file__ is not None
    tree = ast.parse(Path(loader_mod.__file__).read_text())
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert {"torch", "transformers", "PIL"}.isdisjoint(top_level_imports)


def test_public_signatures_match_the_call_sites():
    # The signatures the microbench / bench driver pass — kept identical to the moved
    # private helpers so the move is behaviour-preserving (a drift would break callers
    # whose GPU path no test executes).
    assert list(inspect.signature(load_frozen_openvla).parameters) == [
        "torch_mod",
        "model_id",
        "device",
        "attn_impl",
    ]
    assert list(inspect.signature(build_target).parameters) == [
        "np_mod",
        "model",
        "processor",
        "device",
        "instruction",
        "suffix_len",
        "seed",
        "eval_batch",  # forwarded to OpenVlaGcgTarget so the bench's sw=512 fits 24 GB (DE-7)
    ]


def test_build_target_eval_batch_defaults_to_none():
    # Back-compat: the microbench constructs targets without eval_batch (the single-forward
    # path), so the added knob must default to None.
    assert inspect.signature(build_target).parameters["eval_batch"].default is None


def test_default_instruction_is_the_step_5_5_target():
    assert DEFAULT_INSTRUCTION == "pick up the red block"

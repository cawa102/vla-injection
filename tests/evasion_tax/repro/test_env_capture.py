"""Tests for the environment-capture helper."""

import sys
import types

from evasion_tax.repro import capture_env
from evasion_tax.repro.env_capture import _torch_versions

REQUIRED_KEYS = {
    "platform",
    "python_version",
    "dependencies",
    "git_commit",
    "cuda",
    "torch",
    "driver",
}


def test_returns_dict_with_required_keys():
    env = capture_env()
    assert isinstance(env, dict)
    assert REQUIRED_KEYS.issubset(env.keys())


def test_never_raises_on_this_machine():
    # The contract is that capture_env must not raise here regardless of the
    # surrounding environment; simply calling it is the assertion.
    capture_env()


def test_cuda_and_torch_are_none_when_torch_absent(monkeypatch):
    # When torch is not importable, torch/cuda/driver must degrade gracefully to
    # None rather than raising. Simulate absence (a ``None`` in sys.modules makes
    # ``import torch`` raise ImportError) so this holds on ANY host — with or
    # without torch installed — keeping the model-free suite environment-independent.
    monkeypatch.setitem(sys.modules, "torch", None)
    env = capture_env()
    assert env["torch"] is None
    assert env["cuda"] is None
    assert env["driver"] is None


def test_platform_and_python_version_are_non_empty_strings():
    env = capture_env()
    assert isinstance(env["platform"], str) and env["platform"]
    assert isinstance(env["python_version"], str) and env["python_version"]


def test_dependencies_snapshot_is_present():
    env = capture_env()
    deps = env["dependencies"]
    # A snapshot of installed packages; we record at least numpy's version.
    assert deps is not None
    text = deps if isinstance(deps, str) else "\n".join(f"{k}=={v}" for k, v in deps.items())
    assert "numpy" in text.lower()


def test_git_commit_is_resolved_in_this_repo():
    # This test suite runs inside the git repo, so the commit hash resolves to
    # a 40-char hex string.
    env = capture_env()
    commit = env["git_commit"]
    assert isinstance(commit, str)
    assert len(commit) == 40
    int(commit, 16)  # hex-decodable


def _fake_torch(*, available: bool, raw_driver: int = 0) -> types.SimpleNamespace:
    # A stand-in for the `torch` module (SimpleNamespace, not ModuleType, so it
    # accepts arbitrary attributes); injected via sys.modules so the soft
    # ``import torch`` inside _torch_versions picks it up.
    return types.SimpleNamespace(
        __version__="2.2.2",
        version=types.SimpleNamespace(cuda="12.4" if available else None),
        cuda=types.SimpleNamespace(is_available=lambda: available),
        _C=types.SimpleNamespace(_cuda_getDriverVersion=lambda: raw_driver),
    )


def test_driver_version_resolves_with_mocked_cuda_torch(monkeypatch):
    # C1: on a GPU node, the packed-int driver version must resolve (the old
    # torch.cuda.driver_version() does not exist and silently yielded None).
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, raw_driver=12040))
    torch_v, cuda_v, driver_v = _torch_versions()
    assert torch_v == "2.2.2"
    assert cuda_v == "12.4"
    assert driver_v == "12.4"  # 12040 → major 12, minor 4


def test_driver_version_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False))
    torch_v, _, driver_v = _torch_versions()
    assert torch_v == "2.2.2"
    assert driver_v is None

"""Tests for the environment-capture helper."""

from t7.repro import capture_env

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


def test_cuda_and_torch_are_none_locally():
    # torch is not installed on the local M1 machine, so torch/cuda/driver
    # must degrade gracefully to None rather than raising.
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

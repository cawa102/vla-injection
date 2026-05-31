"""Shared pytest fixtures for the t7 test suite.

Tests must never write to the real ``results/`` directory (write-once invariant);
use ``tmp_path`` / the fixtures here instead.
"""

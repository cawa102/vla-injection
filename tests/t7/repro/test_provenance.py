"""Tests for the SHA-256 helper and the JSON provenance manifest."""

import hashlib
import json

import pytest

from t7.repro import record_provenance, sha256_file


def test_sha256_file_matches_known_hash(tmp_path):
    data = b"the quick brown fox"
    expected = hashlib.sha256(data).hexdigest()
    f = tmp_path / "blob.bin"
    f.write_bytes(data)

    assert sha256_file(f) == expected


def test_sha256_file_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "does-not-exist.bin")


def test_record_provenance_creates_manifest_and_round_trips(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_provenance(
        manifest,
        name="openvla-stats",
        source="https://example.test/stats.json",
        sha256="abc123",
        date="2026-05-31",
        licence="MIT",
    )

    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["openvla-stats"] == {
        "name": "openvla-stats",
        "source": "https://example.test/stats.json",
        "sha256": "abc123",
        "date": "2026-05-31",
        "licence": "MIT",
    }


def test_record_provenance_appends_distinct_names(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_provenance(
        manifest, name="a", source="sa", sha256="ha", date="2026-01-01", licence="MIT"
    )
    record_provenance(
        manifest, name="b", source="sb", sha256="hb", date="2026-01-02", licence="BSD"
    )

    data = json.loads(manifest.read_text())
    assert set(data.keys()) == {"a", "b"}
    assert data["a"]["source"] == "sa"
    assert data["b"]["source"] == "sb"


def test_record_provenance_updating_same_name_replaces_only_that_entry(tmp_path):
    manifest = tmp_path / "manifest.json"
    record_provenance(
        manifest, name="a", source="sa", sha256="ha", date="2026-01-01", licence="MIT"
    )
    record_provenance(
        manifest, name="b", source="sb", sha256="hb", date="2026-01-02", licence="BSD"
    )
    # Update "a" only.
    record_provenance(
        manifest, name="a", source="sa2", sha256="ha2", date="2026-02-02", licence="Apache-2.0"
    )

    data = json.loads(manifest.read_text())
    assert set(data.keys()) == {"a", "b"}
    assert data["a"]["source"] == "sa2"
    assert data["a"]["sha256"] == "ha2"
    # "b" is untouched.
    assert data["b"]["source"] == "sb"
    assert data["b"]["sha256"] == "hb"

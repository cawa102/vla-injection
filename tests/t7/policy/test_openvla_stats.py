"""Tests for the OpenVLA dataset-statistics fetch helpers (Task 3).

The network download itself (``huggingface_hub.hf_hub_download``) is a GPU/online
step and is not unit-tested; the *pure* logic — the artifact URL and the
provenance recording — is tested here without touching the network.
"""

import json

from t7.policy.openvla_stats import record_stats_provenance, stats_url

REPO = "openvla/openvla-7b-finetuned-libero-spatial"


def test_stats_url_points_at_the_checkpoint_dataset_statistics_file():
    assert stats_url(REPO) == (
        f"https://huggingface.co/{REPO}/resolve/main/dataset_statistics.json"
    )


def test_record_stats_provenance_hashes_file_and_writes_a_keyed_entry(tmp_path):
    stats = tmp_path / "dataset_statistics.json"
    stats.write_text(
        json.dumps({"libero_spatial_no_noops": {"action": {"q01": [0.0], "q99": [1.0]}}})
    )
    manifest = tmp_path / "provenance.json"

    sha = record_stats_provenance(
        manifest,
        repo_id=REPO,
        stats_path=stats,
        date="2026-05-31",
        licence="[VERIFY: see HF model card]",
    )

    assert len(sha) == 64  # hex sha-256
    data = json.loads(manifest.read_text())
    entry = data[f"openvla-stats:{REPO}"]
    assert entry["sha256"] == sha
    assert entry["source"] == stats_url(REPO)
    assert entry["date"] == "2026-05-31"
    assert entry["licence"] == "[VERIFY: see HF model card]"

"""Tests for version sequencing used to chain training runs (campaign script)."""

from __future__ import annotations

import pytest

from deepnash_rbc.checkpoints import (
    bump_version,
    existing_versions,
    next_free_version,
)


def test_bump_levels():
    assert bump_version("0.3.0", "minor") == "0.4.0"
    assert bump_version("0.3.5", "major") == "1.0.0"
    assert bump_version("0.3.5", "patch") == "0.3.6"
    with pytest.raises(ValueError):
        bump_version("0.3.0", "nope")


def test_existing_versions_scans_version_dirs(tmp_path):
    for name in ("v0.2.0", "v0.3.0", "v0.10.0", "metrics.jsonl", "vX.Y.Z", "notes"):
        (tmp_path / name).mkdir() if not name.endswith(".jsonl") else (
            tmp_path / name
        ).write_text("{}")
    # only valid vN.N.N dirs, sorted numerically (v0.10.0 after v0.3.0)
    assert existing_versions(str(tmp_path)) == ["0.2.0", "0.3.0", "0.10.0"]
    assert existing_versions(str(tmp_path / "missing")) == []


def test_next_free_grows_from_largest_no_gap_fill():
    taken = {"0.2.0", "0.3.0"}
    # largest is 0.3.0 -> minor -> 0.4.0 (does NOT reuse the 0.2.x gap)
    assert next_free_version("minor", taken, base="0.0.0") == "0.4.0"


def test_next_free_skips_existing_target():
    taken = {"0.2.0", "0.3.0", "0.4.0"}
    # 0.3.0->0.4.0 exists, so keep bumping -> 0.5.0
    assert next_free_version("minor", taken, base="0.0.0") == "0.5.0"


def test_next_free_uses_base_when_nothing_taken():
    assert next_free_version("minor", set(), base="0.2.0") == "0.3.0"


def test_campaign_plan_reserves_within_batch():
    # two minor runs starting from {0.2.0, 0.3.0} -> 0.4.0 then 0.5.0
    reserved = {"0.2.0", "0.3.0"}
    picks = []
    for _ in range(2):
        v = next_free_version("minor", reserved, base="0.0.0")
        reserved.add(v)
        picks.append(v)
    assert picks == ["0.4.0", "0.5.0"]

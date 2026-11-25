from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "setup_reference_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "strut_setup_reference_data_test_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_reference_data = _load_module()


def _manifest() -> dict:
    return setup_reference_data.load_manifest()


def test_manifest_entries_are_pinned_and_sourced():
    manifest = _manifest()
    sources = manifest["sources"]
    for sha, source in sources.items():
        assert len(sha) == 64
        assert source["kind"] in {"file", "zip", "rar"}
        assert source["urls"]
    used = set()
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["source"] in sources
        if sources[entry["source"]]["kind"] == "file":
            assert entry["member"] is None
        else:
            assert entry["member"]
        used.add(entry["source"])
    for entry in manifest["trees"]:
        assert entry["source"] in sources
        used.add(entry["source"])
    assert used == set(sources)
    for entry in manifest["git"]:
        assert len(entry["commit"]) == 40


def test_fixture_gitignore_matches_manifest():
    manifest = _manifest()
    lines = {
        line
        for line in setup_reference_data.FIXTURE_GITIGNORE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line and not line.startswith("#")
    }
    expected = {f"/{entry['path']}" for entry in manifest["files"]}
    expected |= {f"/{entry['path']}/" for entry in manifest["trees"]}
    assert lines == expected


def test_every_direct_tcl_entry_is_committed_or_pinned():
    manifest = _manifest()
    pinned_files = {entry["path"] for entry in manifest["files"]}
    pinned_trees = [entry["path"] + "/" for entry in manifest["trees"]]
    fixture_root = setup_reference_data.FIXTURE_ROOT
    for manifest_path in sorted(
        (REPO_ROOT / "tests" / "validation").glob("*/direct_tcl_case.json")
    ):
        entry_tcl = json.loads(manifest_path.read_text(encoding="utf-8"))["entry_tcl"]
        entry = (REPO_ROOT / entry_tcl).resolve()
        rel = entry.relative_to(fixture_root).as_posix()
        if any(rel.startswith(prefix) for prefix in pinned_trees) or rel in pinned_files:
            continue
        assert entry.is_file(), f"{manifest_path}: {entry_tcl} is not committed"


def test_install_file_extracts_and_verifies_zip_member(tmp_path, monkeypatch):
    monkeypatch.setenv("STRUT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(setup_reference_data, "FIXTURE_ROOT", tmp_path / "fixtures")
    member_data = b"0.1 0.2 0.3\n"
    archive = tmp_path / "gm.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("GM.AT2", member_data)
    archive_bytes = archive.read_bytes()
    archive_sha = setup_reference_data.sha256_bytes(archive_bytes)
    sources = {archive_sha: {"kind": "zip", "urls": ["https://example.invalid/gm.zip"]}}
    entry = {
        "path": "Examples/GMfiles/gm.at2",
        "sha256": setup_reference_data.sha256_bytes(member_data),
        "source": archive_sha,
        "member": "GM.AT2",
    }

    class FakeDownloader:
        count = 0

        def fetch(self, url: str) -> bytes:
            self.count += 1
            return archive_bytes

    downloader = FakeDownloader()
    assert setup_reference_data.install_file(entry, sources, downloader) is True
    target = tmp_path / "fixtures" / "Examples" / "GMfiles" / "gm.at2"
    assert target.read_bytes() == member_data
    assert target.stat().st_mode & 0o777 == 0o644
    # Second run is a no-op: nothing written, nothing downloaded.
    assert setup_reference_data.install_file(entry, sources, downloader) is False
    assert downloader.count == 1


def test_install_file_rejects_tampered_download(tmp_path, monkeypatch):
    monkeypatch.setenv("STRUT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(setup_reference_data, "FIXTURE_ROOT", tmp_path / "fixtures")
    expected = b"real data\n"
    sha = setup_reference_data.sha256_bytes(expected)
    sources = {sha: {"kind": "file", "urls": ["https://example.invalid/gm.acc"]}}
    entry = {"path": "gm.acc", "sha256": sha, "source": sha, "member": None}

    class TamperedDownloader:
        count = 0

        def fetch(self, url: str) -> bytes:
            return b"tampered\n"

    with pytest.raises(SystemExit):
        setup_reference_data.install_file(entry, sources, TamperedDownloader())
    assert not (tmp_path / "fixtures" / "gm.acc").exists()
    assert not setup_reference_data.blob_path(sha).exists()

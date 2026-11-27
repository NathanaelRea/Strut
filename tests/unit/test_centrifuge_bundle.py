from pathlib import Path

from scripts.centrifuge_recon import bundle
from scripts.centrifuge_recon.verify import command_digest

# Taken from the pinned original, after parsing executable command lines.
ORIGINAL_COMMANDS = "dd18ef70812a09c971704f0136d3dc180b100dd1db49b01c70e30dba113e27cb"


def test_bundle_matches_the_published_commands(tmp_path: Path):
    bundle.build_bundle(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == set(bundle.ENTRY_NAMES)

    original = (tmp_path / bundle.ORIGINAL_ENTRY).read_text()
    assert command_digest(original) == ORIGINAL_COMMANDS
    full = (tmp_path / bundle.BENCHMARK_ENTRY).read_text()
    capped = (tmp_path / bundle.PARITY_ENTRY).read_text()
    assert full.count("while {$currentTime <") == 9
    assert "while {$currentTime < 360.}" in full
    assert "rayleigh $massProportionalDamping" in full
    assert capped.count("while {$currentTime <") == 1
    assert "while {$currentTime < 0.15 && $strut_attempts < 8}" in capped
    assert "[analyze 3 5.e5]" in capped and "[analyze 5 5.e5]" in capped


def test_bundle_stamp_regenerates_missing_files(tmp_path: Path):
    assert bundle.ensure_bundle(tmp_path)
    assert bundle.bundle_is_current(tmp_path)
    assert not bundle.ensure_bundle(tmp_path)

    path = tmp_path / bundle.PARITY_ENTRY
    initial = path.read_bytes()
    path.unlink()
    assert bundle.ensure_bundle(tmp_path)
    assert path.read_bytes() == initial

from __future__ import annotations

from pathlib import Path

from scripts.megatall_recon import bundle

EXPECTED_FILES = {
    "opensees.tcl",
    "Hbeam.tcl",
    "layershell.tcl",
    "truss.tcl",
    "trussload.tcl",
    "rigiddigram.tcl",
    "dynamiccpu.tcl",
    "dynamicgpu.tcl",
    "GMX.txt",
}


def _count(path: Path, prefix: str) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.startswith(prefix))


def test_bundle_has_the_benchmark_model(tmp_path: Path):
    bundle.build_bundle(tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == EXPECTED_FILES
    assert _count(tmp_path / "opensees.tcl", "node ") == 27220
    assert _count(tmp_path / "Hbeam.tcl", "element dispBeamColumn") == 28310
    assert _count(tmp_path / "layershell.tcl", "element ShellMITC4") == 16616
    assert _count(tmp_path / "truss.tcl", "element truss") == 3568
    assert _count(tmp_path / "truss.tcl", "element elasticBeamColumn") == 616
    assert _count(tmp_path / "rigiddigram.tcl", "rigidDiaphragm") == 128

    gpu = (tmp_path / "dynamicgpu.tcl").read_text()
    cpu = (tmp_path / "dynamiccpu.tcl").read_text()
    assert gpu.count("system CuSP") == 1
    assert gpu.replace("system CuSP", "system SparseSYM") == cpu


def test_ground_motion_matches_original_layout():
    text = bundle.ground_motion_text()
    lines = text.split("\r\n")
    assert lines[-1] == ""
    values = [float(v) for v in lines[:-1]]
    assert len(values) == bundle.GM_NPTS == 4000
    assert max(abs(v) for v in values) == 1.0
    assert all("E" in v and len(v.split("E")[0].split(".")[1]) == 2 for v in lines[:-1])


def test_bundle_is_deterministic(tmp_path: Path):
    first, second = tmp_path / "a", tmp_path / "b"
    bundle.build_bundle(first)
    bundle.build_bundle(second)
    for name in EXPECTED_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_ensure_bundle_regenerates_only_when_stale(tmp_path: Path):
    target = tmp_path / "Model1"
    assert bundle.ensure_bundle(target) is True
    assert bundle.bundle_is_current(target)
    assert bundle.ensure_bundle(target) is False
    (target / bundle.STAMP_NAME).write_text("stale\n")
    assert bundle.ensure_bundle(target) is True
    assert {p.name for p in target.iterdir()} == EXPECTED_FILES | {bundle.STAMP_NAME}

"""Parity of small pieces of the mega-tall model against OpenSees.

The full model takes OpenSees a quarter of an hour. These slices keep its real
nodes, sections, materials, beam orientations and rigid diaphragms for a few
stories, so each one answers in seconds and a failure points at a height and a
load level instead of at 50,000 elements.
"""

import fcntl
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_TCL = (
    REPO_ROOT
    / "tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/dynamiccpu.tcl"
)
# OpenSees writes six significant digits, so two matching engines still differ
# by up to about 5e-6 of the largest value in a recorder group.
TOLERANCE = 2.0e-5

SLICES = {
    # Base of the tower: mega-columns, core walls, floor beams, diaphragms.
    "base_stories": ["--z-min", "0", "--z-max", "12"],
    # First outrigger and belt-truss zone: adds the elastic beams and trusses.
    "outrigger_zone": ["--z-min", "42", "--z-max", "66"],
    # Crown: different wall layout and the last outrigger.
    "crown": ["--z-min", "560", "--z-max", "607"],
    # Six times the slice's gravity in twelve steps, just below the load at
    # which both engines stop converging: yielding steel, cracked concrete and
    # P-Delta all active.
    "base_stories_near_collapse": [
        "--z-min", "0", "--z-max", "12", "--load-scale", "6", "--steps", "12",
    ],
    # Gravity, then sixty steps of the strongest part of the ground motion with
    # the model's masses, Rayleigh damping and Newmark settings.
    "base_stories_earthquake": ["--z-min", "0", "--z-max", "12", "--transient-steps", "60"],
    "outrigger_zone_earthquake": ["--z-min", "42", "--z-max", "66", "--transient-steps", "40"],
    # Core walls alone at fifty times gravity: the ConcreteS layers on their own.
    "base_walls_cracking": [
        "--z-min", "0", "--z-max", "12", "--types", "shell", "--no-diaphragm",
        "--load-scale", "50",
    ],
}  # fmt: skip


def _opensees_binary() -> Path:
    resolved = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; strut_resolve_opensees_binary OpenSees "${OPENSEES_BIN:-}"',
            "bash",
            str(REPO_ROOT / "scripts/opensees_env.sh"),
        ],
        text=True,
    ).strip()
    return Path(resolved)


@pytest.fixture(scope="module")
def megatall_source(tmp_path_factory) -> Path:
    if not _opensees_binary().is_file():
        pytest.skip("OpenSees is unavailable. Run uv run scripts/setup_opensees.py.")
    if not ENTRY_TCL.is_file():
        pytest.skip(
            "mega-tall model is not installed. Run uv run scripts/setup_reference_data.py."
        )
    # Converting rewrites a shared folder in the repo, so parallel workers must
    # do it once between them: the base temp's parent is common to all workers.
    shared = tmp_path_factory.getbasetemp().parent
    source = shared / "megatall_source.json"
    with open(shared / "megatall_source.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.is_file():
            partial = shared / "megatall_source.partial.json"
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/megatall_slice.py",
                    "--write-source",
                    str(partial),
                ],
                cwd=REPO_ROOT,
            )
            partial.replace(source)
    return source


@pytest.mark.parametrize("name", sorted(SLICES))
def test_megatall_slice_matches_opensees(
    name: str, megatall_source: Path, tmp_path: Path
):
    case_json = tmp_path / name / f"{name}.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/megatall_slice.py",
            "--source",
            str(megatall_source),
            "--name",
            name,
            "--output",
            str(case_json),
            *SLICES[name],
        ],
        cwd=REPO_ROOT,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/parity_report.py",
            str(case_json),
            "--work-dir",
            str(tmp_path / "work"),
            "--fail-above",
            str(TOLERANCE),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

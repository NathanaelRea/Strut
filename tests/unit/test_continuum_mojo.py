import subprocess
from pathlib import Path


def test_continuum_element_stiffness_force_and_mass():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "-I",
            str(root / "src/mojo"),
            str(root / "tests/mojo/test_continuum.mojo"),
        ],
        cwd=root,
        check=True,
    )

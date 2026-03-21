import subprocess
from pathlib import Path


def test_case_input_native_mojo_suite():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            "uv",
            "run",
            "mojo",
            "test",
            "-I",
            str(repo_root / "src" / "mojo"),
            str(repo_root / "tests" / "mojo" / "test_case_input_native.mojo"),
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout

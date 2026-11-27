import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json"


@pytest.fixture
def sparse_case(tmp_path):
    case = json.loads(CASE.read_text())
    element = case["elements"][0]
    case["nodes"] = [case["nodes"][0]] + [
        {"id": i, "x": float(i - 1), "y": 0.0} for i in range(2, 92)
    ]
    case["elements"] = [{**element, "id": i, "nodes": [i, i + 1]} for i in range(1, 91)]
    case["analysis"]["system"] = "SparseSYM"
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps(case))
    return path


def run_backend(case, output, backend):
    env = os.environ.copy()
    if backend == "gpu":
        env.setdefault("STRUT_MOJO_BIN", str(ROOT / "build/strut/strut"))
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_strut_case.py"),
            "--input",
            str(case),
            "--output",
            str(output),
            "--backend",
            backend,
            "--explain-backend",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_cpu_and_auto_use_cpu(sparse_case, tmp_path, backend):
    output = tmp_path / "output"
    result = run_backend(sparse_case, output, backend)
    assert result.returncode == 0, result.stderr
    assert f"Backend used for {output} - CPU" in result.stdout
    assert (
        json.loads((output / "phase_times_us.json").read_text())["gpu_factorizations"]
        == 0
    )


def test_explicit_gpu_does_not_fall_back(sparse_case, tmp_path):
    result = run_backend(sparse_case, tmp_path / "cpu_build", "gpu")
    assert result.returncode != 0
    assert "This Strut executable has no GPU target" in result.stdout + result.stderr

    result = run_backend(CASE, tmp_path / "wrong_solver", "gpu")
    assert result.returncode != 0
    assert "GPU backend requires SparseSYM, SuperLU, or UmfPack" in (
        result.stdout + result.stderr
    )

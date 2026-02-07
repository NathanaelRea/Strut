import json
import tempfile
from pathlib import Path
import subprocess
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

def _run_mojo_case(case_data, out_dir: Path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(json.dumps(case_data))
        tmp_path = Path(tmp.name)
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_mojo_case.py"),
                "--input",
                str(tmp_path),
                "--output",
                str(out_dir),
            ]
        )
    finally:
        tmp_path.unlink(missing_ok=True)

ABS_TOL = 1e-8
REL_TOL = 1e-5


def _isclose(a, b, rtol=REL_TOL, atol=ABS_TOL):
    return abs(a - b) <= (atol + rtol * abs(b))


def _parse_line(line: str):
    line = line.strip()
    if not line:
        return []
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    return [float(p) for p in parts]


def _load_last_values(path: Path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    return _parse_line(lines[-1])


def _compare_vectors(ref, got):
    if len(ref) != len(got):
        raise AssertionError(f"length mismatch: {len(ref)} != {len(got)}")
    for i, (r, g) in enumerate(zip(ref, got), start=1):
        if not _isclose(g, r):
            abs_err = abs(r - g)
            rel_err = abs_err / max(abs(r), 1e-30)
            raise AssertionError(
                f"dof {i}: ref={r:.6e} got={g:.6e} abs={abs_err:.3e} rel={rel_err:.3e}"
            )


def test_json_case_regressions():
    case_names = [
        "elastic_beam_cantilever",
        "elastic_frame_portal",
        "elastic_frame_two_bay",
    ]

    validation_root = repo_root / "tests" / "validation"

    for case_name in case_names:
        case_root = validation_root / case_name
        case_json = case_root / f"{case_name}.json"
        case_data = json.loads(case_json.read_text())
        if not case_data.get("enabled", True):
            continue

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _run_mojo_case(case_data, out_dir)

            for rec in case_data.get("recorders", []):
                if rec["type"] != "node_displacement":
                    raise ValueError(f"unsupported recorder type: {rec['type']}")
                output = rec.get("output", "node_disp")
                for node_id in rec["nodes"]:
                    ref_file = case_root / "reference" / f"{output}_node{node_id}.out"
                    mojo_file = out_dir / f"{output}_node{node_id}.out"
                    assert ref_file.exists(), f"missing reference output: {ref_file}"
                    assert mojo_file.exists(), f"missing mojo output: {mojo_file}"
                    ref_vals = _load_last_values(ref_file)
                    mojo_vals = _load_last_values(mojo_file)
                    _compare_vectors(ref_vals, mojo_vals)

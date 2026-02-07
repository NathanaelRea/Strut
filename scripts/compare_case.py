#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

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


def _compare_vectors(ref, got, rtol=REL_TOL, atol=ABS_TOL):
    if len(ref) != len(got):
        return False, [f"length mismatch: {len(ref)} != {len(got)}"]
    errors = []
    for i, (r, g) in enumerate(zip(ref, got), start=1):
        if not _isclose(g, r, rtol=rtol, atol=atol):
            abs_err = abs(r - g)
            rel_err = abs_err / max(abs(r), 1e-30)
            errors.append(
                f"dof {i}: ref={r:.6e} got={g:.6e} abs={abs_err:.3e} rel={rel_err:.3e}"
            )
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_root = repo_root / "tests" / "validation" / args.case
    case_json = case_root / f"{args.case}.json"
    if not case_json.exists():
        raise SystemExit(f"missing case JSON: {case_json}")

    data = json.loads(case_json.read_text())
    recorders = data.get("recorders", [])
    tol = data.get("parity_tolerance", {})
    rtol = tol.get("rtol", REL_TOL)
    atol = tol.get("atol", ABS_TOL)

    ref_dir = case_root / "reference"
    mojo_dir = case_root / "mojo"

    failures = []
    for rec in recorders:
        rec_type = rec["type"]
        if rec_type == "node_displacement":
            output = rec.get("output", "node_disp")
            for node_id in rec["nodes"]:
                ref_file = ref_dir / f"{output}_node{node_id}.out"
                mojo_file = mojo_dir / f"{output}_node{node_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not mojo_file.exists():
                    failures.append(f"missing mojo output: {mojo_file}")
                    continue
                ref_vals = _load_last_values(ref_file)
                mojo_vals = _load_last_values(mojo_file)
                ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                if not ok:
                    failures.append(f"node {node_id} mismatch")
                    failures.extend([f"  {err}" for err in errors])
        elif rec_type == "element_force":
            output = rec.get("output", "element_force")
            for elem_id in rec["elements"]:
                ref_file = ref_dir / f"{output}_ele{elem_id}.out"
                mojo_file = mojo_dir / f"{output}_ele{elem_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not mojo_file.exists():
                    failures.append(f"missing mojo output: {mojo_file}")
                    continue
                ref_vals = _load_last_values(ref_file)
                mojo_vals = _load_last_values(mojo_file)
                ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                if not ok:
                    failures.append(f"element {elem_id} mismatch")
                    failures.extend([f"  {err}" for err in errors])
        else:
            raise ValueError(f"unsupported recorder type: {rec_type}")

    if failures:
        print("PARITY FAILED")
        for fail in failures:
            print(fail)
        raise SystemExit(1)

    if os.getenv("STRUT_VERBOSE") == "1":
        print("PARITY OK")


if __name__ == "__main__":
    main()

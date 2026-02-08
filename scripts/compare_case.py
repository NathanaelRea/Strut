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


def _load_all_values(path: Path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    return [_parse_line(ln) for ln in lines]


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
    analysis = data.get("analysis", {})
    analysis_type = analysis.get("type", "static_linear")
    is_transient = str(analysis_type).startswith("transient")

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
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    mojo_vals = _load_all_values(mojo_file)
                    if len(ref_vals) != len(mojo_vals):
                        failures.append(
                            f"node {node_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                        )
                        continue
                    for step, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                        ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
                        if not ok:
                            failures.append(f"node {node_id} mismatch at step {step}")
                            failures.extend([f"  {err}" for err in errors])
                            break
                else:
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
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    mojo_vals = _load_all_values(mojo_file)
                    if len(ref_vals) != len(mojo_vals):
                        failures.append(
                            f"element {elem_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                        )
                        continue
                    for step, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                        ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
                        if not ok:
                            failures.append(f"element {elem_id} mismatch at step {step}")
                            failures.extend([f"  {err}" for err in errors])
                            break
                else:
                    ref_vals = _load_last_values(ref_file)
                    mojo_vals = _load_last_values(mojo_file)
                    ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                    if not ok:
                        failures.append(f"element {elem_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "node_reaction":
            output = rec.get("output", "reaction")
            for node_id in rec["nodes"]:
                ref_file = ref_dir / f"{output}_node{node_id}.out"
                mojo_file = mojo_dir / f"{output}_node{node_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not mojo_file.exists():
                    failures.append(f"missing mojo output: {mojo_file}")
                    continue
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    mojo_vals = _load_all_values(mojo_file)
                    if len(ref_vals) != len(mojo_vals):
                        failures.append(
                            f"reaction node {node_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                        )
                        continue
                    for step, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                        ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
                        if not ok:
                            failures.append(f"reaction node {node_id} mismatch at step {step}")
                            failures.extend([f"  {err}" for err in errors])
                            break
                else:
                    ref_vals = _load_last_values(ref_file)
                    mojo_vals = _load_last_values(mojo_file)
                    ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                    if not ok:
                        failures.append(f"reaction node {node_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "drift":
            output = rec.get("output", "drift")
            i_node = int(rec["i_node"])
            j_node = int(rec["j_node"])
            ref_file = ref_dir / f"{output}_i{i_node}_j{j_node}.out"
            mojo_file = mojo_dir / f"{output}_i{i_node}_j{j_node}.out"
            if not ref_file.exists():
                failures.append(f"missing reference output: {ref_file}")
                continue
            if not mojo_file.exists():
                failures.append(f"missing mojo output: {mojo_file}")
                continue
            if is_transient:
                ref_vals = _load_all_values(ref_file)
                mojo_vals = _load_all_values(mojo_file)
                if len(ref_vals) != len(mojo_vals):
                    failures.append(
                        f"drift i{i_node}-j{j_node} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                    )
                    continue
                for step, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                    ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
                    if not ok:
                        failures.append(
                            f"drift i{i_node}-j{j_node} mismatch at step {step}"
                        )
                        failures.extend([f"  {err}" for err in errors])
                        break
            else:
                ref_vals = _load_last_values(ref_file)
                mojo_vals = _load_last_values(mojo_file)
                ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                if not ok:
                    failures.append(f"drift i{i_node}-j{j_node} mismatch")
                    failures.extend([f"  {err}" for err in errors])
        elif rec_type == "envelope_element_force":
            output = rec.get("output", "envelope_element_force")
            for elem_id in rec["elements"]:
                ref_file = ref_dir / f"{output}_ele{elem_id}.out"
                mojo_file = mojo_dir / f"{output}_ele{elem_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not mojo_file.exists():
                    failures.append(f"missing mojo output: {mojo_file}")
                    continue
                ref_vals = _load_all_values(ref_file)
                mojo_vals = _load_all_values(mojo_file)
                if len(ref_vals) != len(mojo_vals):
                    failures.append(
                        f"envelope element {elem_id} row count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                    )
                    continue
                for row_idx, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                    ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
                    if not ok:
                        failures.append(
                            f"envelope element {elem_id} mismatch at row {row_idx}"
                        )
                        failures.extend([f"  {err}" for err in errors])
                        break
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

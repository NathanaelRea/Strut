#!/usr/bin/env python3
import argparse
import json
import os
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ABS_TOL = 1e-8
REL_TOL = 1e-5

DEFAULT_RECORDER_TOLERANCES = {
    "node_displacement": {"atol": 1e-9, "rtol": 1e-5},
    "node_reaction": {"atol": 1e-9, "rtol": 1e-5},
    "drift": {"atol": 1e-9, "rtol": 1e-5},
    "element_force": {"atol": 1e-8, "rtol": 1e-5},
    "element_local_force": {"atol": 1e-8, "rtol": 1e-5},
    "element_basic_force": {"atol": 1e-8, "rtol": 1e-5},
    "element_deformation": {"atol": 1e-8, "rtol": 1e-5},
    "envelope_element_force": {"atol": 1e-8, "rtol": 1e-5},
    "section_force": {"atol": 1e-8, "rtol": 1e-5},
    "section_deformation": {"atol": 1e-9, "rtol": 1e-6},
    "modal_eigen": {"atol": 1e-8, "rtol": 1e-5},
}


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


def _compare_mode_shape_vectors(ref, got, rtol=REL_TOL, atol=ABS_TOL):
    if len(ref) != len(got):
        return False, [f"length mismatch: {len(ref)} != {len(got)}"]
    ref_norm = math.sqrt(sum(v * v for v in ref))
    got_norm = math.sqrt(sum(v * v for v in got))
    eps = 1.0e-20
    if ref_norm <= eps and got_norm <= eps:
        return True, []
    if ref_norm <= eps or got_norm <= eps:
        return False, ["mode shape norm mismatch (one is near zero)"]
    ref_unit = [v / ref_norm for v in ref]
    got_unit = [v / got_norm for v in got]
    dot = sum(r * g for r, g in zip(ref_unit, got_unit))
    if dot < 0.0:
        got_unit = [-v for v in got_unit]
    return _compare_vectors(ref_unit, got_unit, rtol=rtol, atol=atol)


def _max_abs_vector(rows):
    if not rows:
        raise ValueError("empty transient series")
    width = len(rows[0])
    peaks = [0.0] * width
    for row in rows:
        if len(row) != width:
            raise ValueError("transient series row width mismatch")
        for i, value in enumerate(row):
            mag = abs(value)
            if mag > peaks[i]:
                peaks[i] = mag
    return peaks


def _compare_transient_rows(
    ref_vals, strut_vals, label, failures, rtol, atol, parity_mode
):
    if parity_mode == "max_abs":
        ref_peak = _max_abs_vector(ref_vals)
        strut_peak = _max_abs_vector(strut_vals)
        ok, errors = _compare_vectors(ref_peak, strut_peak, rtol=rtol, atol=atol)
        if not ok:
            failures.append(f"{label} max-abs mismatch")
            failures.extend([f"  {err}" for err in errors])
        return
    if len(ref_vals) != len(strut_vals):
        failures.append(
            f"{label} step count mismatch: {len(ref_vals)} != {len(strut_vals)}"
        )
        return

    for step, (rvec, gvec) in enumerate(zip(ref_vals, strut_vals), start=1):
        ok, errors = _compare_vectors(rvec, gvec, rtol=rtol, atol=atol)
        if not ok:
            failures.append(f"{label} mismatch at step {step}")
            failures.extend([f"  {err}" for err in errors])
            break


def _resolve_recorder_tolerance(
    rec_type: str,
    global_rtol: float,
    global_atol: float,
    has_global_override: bool,
    per_recorder_overrides: dict,
):
    default_entry = DEFAULT_RECORDER_TOLERANCES.get(rec_type, {})
    rtol = float(default_entry.get("rtol", REL_TOL))
    atol = float(default_entry.get("atol", ABS_TOL))

    if has_global_override:
        # Case-level parity_tolerance is a global override for all recorders unless a
        # recorder-specific override is provided.
        rtol = float(global_rtol)
        atol = float(global_atol)

    override = per_recorder_overrides.get(rec_type, {})
    if isinstance(override, dict):
        if "rtol" in override:
            rtol = float(override["rtol"])
        if "atol" in override:
            atol = float(override["atol"])
    return rtol, atol


def _analysis_is_transient(analysis: dict) -> bool:
    analysis_type = str(analysis.get("type", "static_linear"))
    if analysis_type.startswith("transient"):
        return True
    if analysis_type != "staged":
        return False
    for stage in analysis.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_analysis = stage.get("analysis", {})
        if not isinstance(stage_analysis, dict):
            continue
        if str(stage_analysis.get("type", "")).startswith("transient"):
            return True
    return False


def _resolve_direct_manifest(case_root: Path) -> Path:
    return case_root / "direct_tcl_case.json"


def _resolve_entry_tcl_from_manifest(manifest_path: Path, repo_root: Path) -> Path:
    data = json.loads(manifest_path.read_text())
    raw_path = data.get("entry_tcl")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"direct Tcl manifest missing entry_tcl: {manifest_path}")
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _load_direct_tcl_case_data(
    entry_tcl: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
) -> dict:
    import tcl_to_strut

    data = tcl_to_strut.convert_tcl_to_solver_input(entry_tcl, repo_root)
    if manifest_path is None or not manifest_path.exists():
        return data

    manifest = json.loads(manifest_path.read_text())
    for key in (
        "parity_tolerance",
        "parity_tolerance_by_recorder",
        "parity_mode",
    ):
        if key in manifest:
            data[key] = manifest[key]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--case-root")
    parser.add_argument("--case-json")
    parser.add_argument("--input-tcl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.case_json:
        case_json = Path(args.case_json).resolve()
        if args.case_root:
            case_root = Path(args.case_root).resolve()
        else:
            case_root = case_json.parent
        if not case_json.exists():
            raise SystemExit(f"missing case JSON: {case_json}")
        data = json.loads(case_json.read_text())
    elif args.input_tcl:
        entry_tcl = Path(args.input_tcl).resolve()
        if args.case_root:
            case_root = Path(args.case_root).resolve()
            manifest_path = _resolve_direct_manifest(case_root)
            if not manifest_path.exists():
                manifest_path = None
        else:
            case_root = repo_root / "tests" / "validation" / entry_tcl.stem
            manifest_path = None
        data = _load_direct_tcl_case_data(entry_tcl, repo_root, manifest_path)
    elif args.case:
        case_root = repo_root / "tests" / "validation" / args.case
        case_json = case_root / f"{args.case}.json"
        if case_json.exists():
            data = json.loads(case_json.read_text())
        else:
            manifest_path = _resolve_direct_manifest(case_root)
            if not manifest_path.exists():
                raise SystemExit(
                    f"missing case JSON or direct Tcl manifest in: {case_root}"
                )
            entry_tcl = _resolve_entry_tcl_from_manifest(manifest_path, repo_root)
            data = _load_direct_tcl_case_data(entry_tcl, repo_root, manifest_path)
    else:
        raise SystemExit("either --case, --case-json, or --input-tcl is required")

    recorders = data.get("recorders", [])
    tol = data.get("parity_tolerance", {})
    rtol = tol.get("rtol", REL_TOL)
    atol = tol.get("atol", ABS_TOL)
    has_global_tol_override = isinstance(tol, dict) and (
        "rtol" in tol or "atol" in tol
    )
    tol_by_recorder = data.get("parity_tolerance_by_recorder", {})
    if not isinstance(tol_by_recorder, dict):
        tol_by_recorder = {}
    parity_mode = str(data.get("parity_mode", "step")).strip().lower()
    if parity_mode not in ("step", "max_abs"):
        raise ValueError(
            f"unsupported parity_mode: {parity_mode} (expected step|max_abs)"
        )
    analysis = data.get("analysis", {})
    is_transient = _analysis_is_transient(analysis)

    ref_dir = case_root / "reference"
    strut_dir = case_root / "strut"

    failures = []
    for rec in recorders:
        if rec.get("parity", True) is False:
            continue
        rec_type = rec["type"]
        rec_rtol, rec_atol = _resolve_recorder_tolerance(
            rec_type, rtol, atol, has_global_tol_override, tol_by_recorder
        )
        if rec_type == "node_displacement":
            output = rec.get("output", "node_disp")
            for node_id in rec["nodes"]:
                ref_file = ref_dir / f"{output}_node{node_id}.out"
                strut_file = strut_dir / f"{output}_node{node_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    strut_vals = _load_all_values(strut_file)
                    _compare_transient_rows(
                        ref_vals,
                        strut_vals,
                        f"node {node_id}",
                        failures,
                        rec_rtol,
                        rec_atol,
                        parity_mode,
                    )
                else:
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    ok, errors = _compare_mode_shape_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(f"node {node_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "element_force":
            output = rec.get("output", "element_force")
            for elem_id in rec["elements"]:
                ref_file = ref_dir / f"{output}_ele{elem_id}.out"
                strut_file = strut_dir / f"{output}_ele{elem_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    strut_vals = _load_all_values(strut_file)
                    _compare_transient_rows(
                        ref_vals,
                        strut_vals,
                        f"element {elem_id}",
                        failures,
                        rec_rtol,
                        rec_atol,
                        parity_mode,
                    )
                else:
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    ok, errors = _compare_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(f"element {elem_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "node_reaction":
            output = rec.get("output", "reaction")
            for node_id in rec["nodes"]:
                ref_file = ref_dir / f"{output}_node{node_id}.out"
                strut_file = strut_dir / f"{output}_node{node_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    strut_vals = _load_all_values(strut_file)
                    _compare_transient_rows(
                        ref_vals,
                        strut_vals,
                        f"reaction node {node_id}",
                        failures,
                        rec_rtol,
                        rec_atol,
                        parity_mode,
                    )
                else:
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    ok, errors = _compare_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(f"reaction node {node_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "drift":
            output = rec.get("output", "drift")
            i_node = int(rec["i_node"])
            j_node = int(rec["j_node"])
            ref_file = ref_dir / f"{output}_i{i_node}_j{j_node}.out"
            strut_file = strut_dir / f"{output}_i{i_node}_j{j_node}.out"
            if not ref_file.exists():
                failures.append(f"missing reference output: {ref_file}")
                continue
            if not strut_file.exists():
                failures.append(f"missing strut output: {strut_file}")
                continue
            if is_transient:
                ref_vals = _load_all_values(ref_file)
                strut_vals = _load_all_values(strut_file)
                _compare_transient_rows(
                    ref_vals,
                    strut_vals,
                    f"drift i{i_node}-j{j_node}",
                    failures,
                    rec_rtol,
                    rec_atol,
                    parity_mode,
                )
            else:
                ref_vals = _load_last_values(ref_file)
                strut_vals = _load_last_values(strut_file)
                ok, errors = _compare_vectors(
                    ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                )
                if not ok:
                    failures.append(f"drift i{i_node}-j{j_node} mismatch")
                    failures.extend([f"  {err}" for err in errors])
        elif rec_type in (
            "element_force",
            "element_local_force",
            "element_basic_force",
            "element_deformation",
        ):
            default_output = rec_type
            output = rec.get("output", default_output)
            for elem_id in rec["elements"]:
                ref_file = ref_dir / f"{output}_ele{elem_id}.out"
                strut_file = strut_dir / f"{output}_ele{elem_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                if is_transient:
                    ref_vals = _load_all_values(ref_file)
                    strut_vals = _load_all_values(strut_file)
                    _compare_transient_rows(
                        ref_vals,
                        strut_vals,
                        f"{rec_type} element {elem_id}",
                        failures,
                        rec_rtol,
                        rec_atol,
                        parity_mode,
                    )
                else:
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    ok, errors = _compare_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(f"{rec_type} element {elem_id} mismatch")
                        failures.extend([f"  {err}" for err in errors])
        elif rec_type == "envelope_element_force":
            output = rec.get("output", "envelope_element_force")
            for elem_id in rec["elements"]:
                ref_file = ref_dir / f"{output}_ele{elem_id}.out"
                strut_file = strut_dir / f"{output}_ele{elem_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                ref_vals = _load_all_values(ref_file)
                strut_vals = _load_all_values(strut_file)
                if len(ref_vals) != len(strut_vals):
                    failures.append(
                        f"envelope element {elem_id} row count mismatch: {len(ref_vals)} != {len(strut_vals)}"
                    )
                    continue
                for row_idx, (rvec, gvec) in enumerate(
                    zip(ref_vals, strut_vals), start=1
                ):
                    ok, errors = _compare_vectors(
                        rvec, gvec, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(
                            f"envelope element {elem_id} mismatch at row {row_idx}"
                        )
                        failures.extend([f"  {err}" for err in errors])
                        break
        elif rec_type in ("section_force", "section_deformation"):
            default_output = (
                "section_force"
                if rec_type == "section_force"
                else "section_deformation"
            )
            output = rec.get("output", default_output)
            sections = rec.get("sections")
            if sections is None:
                if "section" not in rec:
                    failures.append(f"{rec_type} recorder missing section/sections")
                    continue
                sections = [rec["section"]]
            for elem_id in rec["elements"]:
                for sec_no in sections:
                    ref_file = ref_dir / f"{output}_ele{elem_id}_sec{sec_no}.out"
                    strut_file = strut_dir / f"{output}_ele{elem_id}_sec{sec_no}.out"
                    if not ref_file.exists():
                        failures.append(f"missing reference output: {ref_file}")
                        continue
                    if not strut_file.exists():
                        failures.append(f"missing strut output: {strut_file}")
                        continue
                    if is_transient:
                        ref_vals = _load_all_values(ref_file)
                        strut_vals = _load_all_values(strut_file)
                        _compare_transient_rows(
                            ref_vals,
                            strut_vals,
                            f"{rec_type} element {elem_id} section {sec_no}",
                            failures,
                            rec_rtol,
                            rec_atol,
                            parity_mode,
                        )
                    else:
                        ref_vals = _load_last_values(ref_file)
                        strut_vals = _load_last_values(strut_file)
                        ok, errors = _compare_vectors(
                            ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                        )
                        if not ok:
                            failures.append(
                                f"{rec_type} element {elem_id} section {sec_no} mismatch"
                            )
                            failures.extend([f"  {err}" for err in errors])
        elif rec_type == "modal_eigen":
            output = rec.get("output", "modal")
            modes = rec.get("modes", [])
            if not modes:
                failures.append("modal_eigen recorder missing modes")
                continue
            eig_ref = ref_dir / f"{output}_eigenvalues.out"
            eig_strut = strut_dir / f"{output}_eigenvalues.out"
            if not eig_ref.exists():
                failures.append(f"missing reference output: {eig_ref}")
                continue
            if not eig_strut.exists():
                failures.append(f"missing strut output: {eig_strut}")
                continue
            ref_rows = _load_all_values(eig_ref)
            strut_rows = _load_all_values(eig_strut)
            ref_vals = [row[0] for row in ref_rows]
            strut_vals = [row[0] for row in strut_rows]
            ok, errors = _compare_vectors(
                ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
            )
            if not ok:
                failures.append("modal eigenvalue mismatch")
                failures.extend([f"  {err}" for err in errors])

            for mode in modes:
                mode_no = int(mode)
                for node_id in rec["nodes"]:
                    ref_file = ref_dir / f"{output}_mode{mode_no}_node{node_id}.out"
                    strut_file = strut_dir / f"{output}_mode{mode_no}_node{node_id}.out"
                    if not ref_file.exists():
                        failures.append(f"missing reference output: {ref_file}")
                        continue
                    if not strut_file.exists():
                        failures.append(f"missing strut output: {strut_file}")
                        continue
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    ok, errors = _compare_mode_shape_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        failures.append(
                            f"modal mode shape mismatch mode={mode_no} node={node_id}"
                        )
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

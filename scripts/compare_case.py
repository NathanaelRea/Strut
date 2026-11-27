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

import direct_tcl_support

ABS_TOL = 1e-8
REL_TOL = 1e-5

DEFAULT_RECORDER_CATEGORY_TOLERANCES = {
    "displacement": {"atol": 1e-9, "rtol": 1e-5},
    "velocity": {"atol": 1e-8, "rtol": 1e-5},
    "acceleration": {"atol": 1e-8, "rtol": 1e-5},
    "force": {"atol": 1e-8, "rtol": 1e-5},
    "deformation": {"atol": 1e-8, "rtol": 1e-5},
    "modal": {"atol": 1e-8, "rtol": 1e-5},
}

RECORDER_TOLERANCE_CATEGORIES = {
    "node_displacement": "displacement",
    "envelope_node_displacement": "displacement",
    "drift": "displacement",
    "node_velocity": "velocity",
    "envelope_node_velocity": "velocity",
    "node_acceleration": "acceleration",
    "envelope_node_acceleration": "acceleration",
    "node_reaction": "force",
    "element_force": "force",
    "element_local_force": "force",
    "element_basic_force": "force",
    "envelope_element_force": "force",
    "envelope_element_local_force": "force",
    "section_force": "force",
    "element_deformation": "deformation",
    "section_deformation": "deformation",
    "modal_eigen": "modal",
}

DEFAULT_RECORDER_TOLERANCES = {
    rec_type: dict(DEFAULT_RECORDER_CATEGORY_TOLERANCES[category])
    for rec_type, category in RECORDER_TOLERANCE_CATEGORIES.items()
}
# Preserve legacy per-recorder defaults that were intentionally tighter.
DEFAULT_RECORDER_TOLERANCES["section_deformation"] = {"atol": 1e-9, "rtol": 1e-6}


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


def _load_history_envelope(path: Path):
    minimum = []
    maximum = []
    count = 0
    with path.open() as stream:
        for line in stream:
            row = _parse_line(line)
            if not row:
                continue
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"non-finite recorder value in {path}, row {count + 1}"
                )
            if count == 0:
                minimum = row.copy()
                maximum = row.copy()
            elif len(row) != len(minimum):
                raise ValueError(f"recorder width mismatch in {path}, row {count + 1}")
            else:
                for i, value in enumerate(row):
                    minimum[i] = min(minimum[i], value)
                    maximum[i] = max(maximum[i], value)
            count += 1
    if count == 0:
        raise ValueError(f"empty output file: {path}")
    peak = [max(abs(lo), abs(hi)) for lo, hi in zip(minimum, maximum)]
    return count, (minimum, maximum, peak)


def _compare_history_files(ref_file, strut_file, label, failures, rtol, atol, mode):
    if mode != "envelope":
        _compare_transient_rows(
            _load_all_values(ref_file),
            _load_all_values(strut_file),
            label,
            failures,
            rtol,
            atol,
            mode,
        )
        return

    try:
        ref_count, ref_envelope = _load_history_envelope(ref_file)
        strut_count, strut_envelope = _load_history_envelope(strut_file)
    except ValueError as exc:
        failures.append(f"{label}: {exc}")
        return
    if ref_count != strut_count:
        failures.append(f"{label} step count mismatch: {ref_count} != {strut_count}")
        return
    for statistic, ref, got in zip(
        ("minimum", "maximum", "peak absolute"), ref_envelope, strut_envelope
    ):
        ok, errors = _compare_vectors(ref, got, rtol=rtol, atol=atol)
        if not ok:
            failures.append(f"{label} envelope {statistic} mismatch")
            failures.extend(f"  {err}" for err in errors)


def _compare_transient_rows(
    ref_vals, strut_vals, label, failures, rtol, atol, parity_mode
):
    if parity_mode == "max_abs":
        if len(ref_vals) != len(strut_vals):
            shared_steps = min(len(ref_vals), len(strut_vals))
            if shared_steps == 0:
                failures.append(
                    f"{label} step count mismatch: {len(ref_vals)} != {len(strut_vals)}"
                )
                return
            ref_vals = ref_vals[:shared_steps]
            strut_vals = strut_vals[:shared_steps]
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
    per_category_overrides: dict,
    per_recorder_overrides: dict,
):
    category = RECORDER_TOLERANCE_CATEGORIES.get(rec_type)
    default_entry = DEFAULT_RECORDER_TOLERANCES.get(rec_type)
    if default_entry is None and category is not None:
        default_entry = DEFAULT_RECORDER_CATEGORY_TOLERANCES.get(category, {})
    if default_entry is None:
        default_entry = {}
    rtol = float(default_entry.get("rtol", REL_TOL))
    atol = float(default_entry.get("atol", ABS_TOL))

    if has_global_override:
        # Case-level parity_tolerance is a global override for all recorders unless a
        # recorder-specific override is provided.
        rtol = float(global_rtol)
        atol = float(global_atol)

    if category is not None:
        category_override = per_category_overrides.get(category, {})
        if isinstance(category_override, dict):
            if "rtol" in category_override:
                rtol = float(category_override["rtol"])
            if "atol" in category_override:
                atol = float(category_override["atol"])

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
    return direct_tcl_support.resolve_entry_tcl_from_manifest(manifest_path, repo_root)


def _load_direct_tcl_case_data(
    entry_tcl: Path,
    repo_root: Path,
    case_root: Path,
    manifest_path: Path | None = None,
) -> dict:
    import tcl_to_strut

    manifest = (
        direct_tcl_support.load_direct_tcl_manifest(manifest_path)
        if manifest_path is not None and manifest_path.exists()
        else {}
    )
    if manifest.get("replay_tcl"):
        outputs = json.loads(
            (case_root / "strut" / ".direct_tcl_recorders.json").read_text()
        )
        reference_dir = case_root / "reference"
        if not (reference_dir / "__strut_replay_complete.out").is_file():
            raise SystemExit("OpenSees did not complete the direct Tcl replay")
        recorder_root = reference_dir / "__strut_domains"
        reference_files = {
            str(path.relative_to(reference_dir))
            for path in recorder_root.rglob("*")
            if path.is_file()
        }
        reference_files.update(
            path.name for path in reference_dir.glob("__strut_query_*.out")
        )
        missing = reference_files - outputs.keys()
        if missing:
            raise SystemExit(f"Missing Strut Tcl outputs: {', '.join(sorted(missing))}")
        if not outputs:
            raise SystemExit(
                "direct Tcl replay produced no recorders or solver queries"
            )
        return {
            **manifest,
            "recorders": [
                {"type": "raw_table", "file": path, **recorder, "replay": True}
                for path, recorder in outputs.items()
            ],
        }
    if manifest.get("interactive_tcl"):
        data = {
            "analysis": {"type": "static_linear"},
            "recorders": [
                {"type": "raw_table", "file": str(path)}
                for path in manifest["compare_files"]
            ],
        }
        for key in ("parity_tolerance", "parity_tolerance_by_recorder", "comparison"):
            if key in manifest:
                data[key] = manifest[key]
        return data

    mirror_root = case_root / "generated" / "_direct_tcl_context"
    parser_source = entry_tcl.resolve()
    if manifest_path is not None and manifest_path.exists():
        parser_source = direct_tcl_support.resolve_entry_tcl_from_manifest(
            manifest_path, repo_root
        )

    try:
        parser_source.relative_to(mirror_root.resolve())
    except ValueError:
        parser_entry, _ = direct_tcl_support.prepare_direct_tcl_entry(
            parser_source,
            direct_tcl_support.resolve_direct_tcl_source_files(
                parser_source, manifest_path
            ),
            mirror_root,
            manifest_path=manifest_path,
            excluded_roots=[case_root],
        )
    else:
        parser_entry = parser_source

    data = tcl_to_strut.convert_tcl_to_solver_input(parser_entry, repo_root)
    if manifest_path is None or not manifest_path.exists():
        return data

    manifest = direct_tcl_support.load_direct_tcl_manifest(manifest_path)
    for key in (
        "parity_tolerance",
        "parity_tolerance_by_category",
        "parity_tolerance_by_recorder",
        "parity_mode",
        "comparison",
        "parity_history",
    ):
        if key in manifest:
            data[key] = manifest[key]
    return data


def _compare_output_dirs(data: dict, ref_dir: Path, strut_dir: Path) -> list[str]:
    recorders = data.get("recorders", [])
    comparison = data.get("comparison")
    parity_mode = str(data.get("parity_mode", "step")).strip().lower()
    if parity_mode not in ("step", "max_abs"):
        raise ValueError(
            f"unsupported parity_mode: {parity_mode} (expected step|max_abs)"
        )
    if comparison is not None:
        if comparison != "envelope":
            raise ValueError(
                f"unsupported comparison: {comparison!r} (expected envelope)"
            )
        parity_mode = comparison
    tol = {} if comparison == "envelope" else data.get("parity_tolerance", {})
    rtol = tol.get("rtol", REL_TOL)
    atol = tol.get("atol", ABS_TOL)
    has_global_tol_override = isinstance(tol, dict) and ("rtol" in tol or "atol" in tol)
    tol_by_recorder = (
        {} if comparison == "envelope" else data.get("parity_tolerance_by_recorder", {})
    )
    if not isinstance(tol_by_recorder, dict):
        tol_by_recorder = {}
    tol_by_category = (
        {} if comparison == "envelope" else data.get("parity_tolerance_by_category", {})
    )
    if not isinstance(tol_by_category, dict):
        tol_by_category = {}
    analysis = data.get("analysis", {})
    is_transient = _analysis_is_transient(analysis)
    compare_history = (
        is_transient or parity_mode != "step" or data.get("parity_history", False)
    )

    failures = []
    for rec in recorders:
        if rec.get("parity", True) is False:
            continue
        rec_type = rec["type"]
        rec_rtol, rec_atol = _resolve_recorder_tolerance(
            rec.get("category", rec_type),
            rtol,
            atol,
            has_global_tol_override,
            tol_by_category,
            tol_by_recorder,
        )
        if rec_type == "raw_table":
            rel_file = rec["file"]
            ref_file = ref_dir / rel_file
            strut_file = strut_dir / rel_file
            if not ref_file.exists():
                failures.append(f"missing reference output: {ref_file}")
                continue
            if not strut_file.exists():
                failures.append(f"missing strut output: {strut_file}")
                continue
            if parity_mode == "envelope":
                _compare_history_files(
                    ref_file,
                    strut_file,
                    rel_file,
                    failures,
                    rec_rtol,
                    rec_atol,
                    parity_mode,
                )
                continue
            if rec.get("replay"):
                ref_vals = [
                    _parse_line(row)
                    for row in ref_file.read_text().splitlines()
                    if row.strip()
                ]
                strut_vals = [
                    _parse_line(row)
                    for row in strut_file.read_text().splitlines()
                    if row.strip()
                ]
                if rec.get("include_time"):
                    ref_vals = [row[1:] for row in ref_vals]
                if not ref_vals and not strut_vals:
                    continue
            else:
                ref_vals = _load_all_values(ref_file)
                strut_vals = _load_all_values(strut_file)
            if not ref_vals:
                failures.append(f"empty reference output: {ref_file}")
                continue
            if len(ref_vals) != len(strut_vals):
                failures.append(
                    f"{rel_file} row count mismatch: {len(ref_vals)} != {len(strut_vals)}"
                )
                continue
            for row_idx, (rvec, gvec) in enumerate(zip(ref_vals, strut_vals), start=1):
                ok, errors = _compare_vectors(rvec, gvec, rtol=rec_rtol, atol=rec_atol)
                if not ok:
                    failures.append(f"{rel_file} mismatch at row {row_idx}")
                    failures.extend([f"  {err}" for err in errors])
                    break
        elif rec_type in ("node_displacement", "node_velocity", "node_acceleration"):
            default_output = {
                "node_displacement": "node_disp",
                "node_velocity": "node_velocity",
                "node_acceleration": "node_accel",
            }[rec_type]
            output = rec.get("output", default_output)
            for node_id in rec["nodes"]:
                ref_file = ref_dir / f"{output}_node{node_id}.out"
                strut_file = strut_dir / f"{output}_node{node_id}.out"
                if not ref_file.exists():
                    failures.append(f"missing reference output: {ref_file}")
                    continue
                if not strut_file.exists():
                    failures.append(f"missing strut output: {strut_file}")
                    continue
                if compare_history:
                    _compare_history_files(
                        ref_file,
                        strut_file,
                        f"node {node_id}",
                        failures,
                        rec_rtol,
                        rec_atol,
                        parity_mode,
                    )
                else:
                    ref_vals = _load_last_values(ref_file)
                    strut_vals = _load_last_values(strut_file)
                    # Compare 3D displacements in their original units, including
                    # near-zero displacements at supports enforced by Penalty.
                    compare_vectors = (
                        _compare_vectors
                        if data.get("model", {}).get("ndm") == 3
                        else _compare_mode_shape_vectors
                    )
                    ok, errors = compare_vectors(
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
                if compare_history:
                    _compare_history_files(
                        ref_file,
                        strut_file,
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
                if compare_history:
                    _compare_history_files(
                        ref_file,
                        strut_file,
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
            if compare_history:
                _compare_history_files(
                    ref_file,
                    strut_file,
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
            "quad_stress",
            "quad_strain",
            "quad_state",
            "contact_secondary_force",
            "contact_friction_force",
            "contact_scalar_force",
            "contact_primary_force",
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
                if compare_history:
                    _compare_history_files(
                        ref_file,
                        strut_file,
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
        elif rec_type in (
            "envelope_element_force",
            "envelope_element_local_force",
            "envelope_node_displacement",
            "envelope_node_acceleration",
        ):
            output = rec.get("output", rec_type)
            item_label = "element"
            item_ids = rec.get("elements", [])
            filename_template = "{output}_ele{item_id}.out"
            if rec_type.startswith("envelope_node_"):
                item_label = "node"
                item_ids = rec.get("nodes", [])
                filename_template = "{output}_node{item_id}.out"
            for item_id in item_ids:
                ref_file = ref_dir / filename_template.format(
                    output=output, item_id=item_id
                )
                strut_file = strut_dir / filename_template.format(
                    output=output, item_id=item_id
                )
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
                        f"envelope {item_label} {item_id} row count mismatch: {len(ref_vals)} != {len(strut_vals)}"
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
                            f"envelope {item_label} {item_id} mismatch at row {row_idx}"
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
                    if compare_history:
                        _compare_history_files(
                            ref_file,
                            strut_file,
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
    return failures


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
        data = _load_direct_tcl_case_data(
            entry_tcl, repo_root, case_root, manifest_path
        )
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
            data = _load_direct_tcl_case_data(
                entry_tcl, repo_root, case_root, manifest_path
            )
    else:
        raise SystemExit("either --case, --case-json, or --input-tcl is required")

    ref_dir = case_root / "reference"
    strut_dir = case_root / "strut"
    failures = _compare_output_dirs(data, ref_dir, strut_dir)

    if failures:
        print("PARITY FAILED")
        for fail in failures:
            print(fail)
        raise SystemExit(1)

    if os.getenv("STRUT_VERBOSE") == "1":
        print("PARITY OK")


if __name__ == "__main__":
    main()

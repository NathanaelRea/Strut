#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import time
import math
import sys
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

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
ELEMENT_RESPONSE_RECORDER_TYPES = (
    "element_force",
    "element_local_force",
    "element_basic_force",
    "element_deformation",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    json_path: Optional[Path] = None
    tcl_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    benchmark_size: Optional[str] = None


BENCHMARK_SUITES: Dict[str, List[str]] = {
    "root_cause_v1": [
        "opensees_example_ex5_frame2d_eq_uniform_rc_fiber",
        "opensees_example_ex2c_canti2d_inelastic_fiber",
        "opensees_example_rc_frame_earthquake",
        "steel01_truss_transient_modified_newton_benchmark",
        "opensees_example_2d_elastic_cantileaver_column",
        "elastic_transient_newmark_path",
    ],
    "regression_gate_v1": [
        "elastic_frame_18bay_17story",
        "force_beam_column2d_fiber_frame_18bay_17story",
        "opensees_example_rc_frame_earthquake",
        "opensees_example_ex5_frame2d_eq_uniform_rc_fiber",
    ],
    "scaling_v1": [
        "elastic_frame_18bay_17story",
        "force_beam_column2d_fiber_frame_18bay_17story",
    ],
    "opt_fast_v1": [
        "elastic_beam_cantilever",
        "elastic_frame_portal",
        "opensees_example_2d_elastic_cantileaver_column",
        "force_beam_column2d_fiber_cantilever",
        "opensees_example_rc_frame_earthquake",
    ],
    "opt_full_v1": [
        "elastic_beam_cantilever",
        "elastic_frame_portal",
        "elastic_frame_two_bay",
        "elastic_frame_two_story",
        "elastic_frame_18bay_17story",
        "elastic_frame_3d_portal",
        "force_beam_column3d_portal_benchmark",
        "disp_beam_column3d_portal_benchmark",
        "force_beam_column2d_fiber_cantilever",
        "force_beam_column2d_fiber_frame_18bay_17story",
        "opensees_example_2d_elastic_cantileaver_column",
        "opensees_example_rc_frame_earthquake",
        "opensees_example_ex5_frame2d_eq_uniform_rc_fiber",
        "elastic_transient_newmark_path",
        "steel01_truss_transient_modified_newton_benchmark",
    ],
}

STANDARD_OPT_COMMANDS = {
    "fast_profile_strut": (
        "uv run scripts/run_benchmarks.py "
        "--benchmark-suite opt_fast_v1 "
        "--engine strut "
        "--profile benchmark/speedscope "
        "--no-archive"
    ),
    "full_profile_both": (
        "uv run scripts/run_benchmarks.py "
        "--benchmark-suite opt_full_v1 "
        "--engine both "
        "--profile benchmark/speedscope "
        "--no-archive"
    ),
}

PHASE_COLUMNS = [
    "case_load_parse_us",
    "model_build_dof_map_us",
    "global_assembly_us",
    "element_state_update_us",
    "linear_solve_us",
    "nonlinear_solve_us",
    "time_series_eval_us",
    "constraints_us",
    "recorders_us",
    "output_write_us",
    "solve_total_us",
    "total_case_us",
]

PHASE_FRAME_MAP = {
    "global_assembly_us": ("assemble_stiffness", "kff_extract"),
    "element_state_update_us": ("nonlinear_iter", "nonlinear_step"),
    "linear_solve_us": ("solve_linear",),
    "nonlinear_solve_us": ("solve_nonlinear",),
    "time_series_eval_us": ("time_series_eval",),
    "constraints_us": ("constraints",),
    "recorders_us": ("recorders",),
}


def run(
    cmd: List[str], env=None, verbose=False, capture_on_error: bool = False
) -> None:
    if verbose:
        print("+", " ".join(cmd))
    if not capture_on_error:
        subprocess.check_call(cmd, env=env)
        return
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )


def log(msg: str) -> None:
    print(msg, flush=True)


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def log_ok(msg: str) -> None:
    log(_color(msg, "32"))


def log_err(msg: str) -> None:
    log(_color(msg, "31"))


def _write_case_error(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_error.txt").write_text(f"{message}\n", encoding="utf-8")


def _count_constrained_dofs(node: dict, ndf: int) -> int:
    constraints = node.get("constraints")
    if not constraints:
        return 0
    if isinstance(constraints, list):
        if all(isinstance(v, bool) for v in constraints):
            if len(constraints) != ndf:
                return 0
            return sum(1 for v in constraints if v)
        return len(constraints)
    return 0


def _case_json_path(case: CaseSpec) -> Path:
    if case.json_path is None:
        raise ValueError(f"case `{case.name}` has no JSON path")
    return case.json_path


def _case_metadata_path(case: CaseSpec) -> Path:
    if case.metadata_path is not None:
        return case.metadata_path
    return _case_json_path(case)


def _load_case_metadata(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid case metadata: {path}")
    return data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _is_direct_tcl_manifest(path: Path) -> bool:
    if path.name != "direct_tcl_case.json":
        return False
    try:
        data = _load_case_metadata(path)
    except Exception:
        return False
    entry_tcl = data.get("entry_tcl")
    return isinstance(entry_tcl, str) and bool(entry_tcl)


def _direct_tcl_case_spec(manifest_path: Path) -> CaseSpec:
    data = _load_case_metadata(manifest_path)
    name = str(data.get("name") or manifest_path.parent.name)
    entry_tcl = data.get("entry_tcl")
    if not isinstance(entry_tcl, str) or not entry_tcl:
        raise ValueError(f"direct Tcl case missing entry_tcl: {manifest_path}")
    benchmark_size = data.get("benchmark_size")
    if not isinstance(benchmark_size, str):
        benchmark_size = None
    return CaseSpec(
        name=name,
        tcl_path=_resolve_repo_relative_path(entry_tcl),
        metadata_path=manifest_path,
        benchmark_size=benchmark_size,
    )


def _direct_case_root(case: CaseSpec) -> Path:
    return _case_metadata_path(case).parent


def _reference_output_path(reference_dir: Path, raw_path: str) -> Optional[Path]:
    path = Path(raw_path)
    current = reference_dir
    for part in path.parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if not current.exists() or not current.is_dir():
            return None
        matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
        if len(matches) != 1:
            return None
        current = matches[0]
    return current


def _normalized_recorder_outputs(recorder: dict) -> List[str]:
    if recorder.get("parity", True) is False:
        return []
    rec_type = recorder["type"]
    output = recorder.get("output", rec_type)
    if rec_type in ("node_displacement", "node_reaction"):
        nodes = recorder.get("nodes", [])
        if len(nodes) != 1:
            raise SystemExit(
                f"direct Tcl benchmark normalization requires single-node recorder: {recorder}"
            )
        return [f"{output}_node{int(nodes[0])}.out"]
    if rec_type == "element_force":
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(
                f"direct Tcl benchmark normalization requires single-element recorder: {recorder}"
            )
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type in ("element_local_force", "element_basic_force", "element_deformation"):
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(
                f"direct Tcl benchmark normalization requires single-element recorder: {recorder}"
            )
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type == "envelope_element_force":
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(
                f"direct Tcl benchmark normalization requires single-element recorder: {recorder}"
            )
        return [f"{output}_ele{int(elements[0])}.out"]
    if rec_type in ("section_force", "section_deformation"):
        elements = recorder.get("elements", [])
        if len(elements) != 1:
            raise SystemExit(
                f"direct Tcl benchmark normalization requires single-element recorder: {recorder}"
            )
        section = recorder.get("section")
        if section is None:
            sections = recorder.get("sections") or []
            if len(sections) != 1:
                raise SystemExit(
                    f"direct Tcl benchmark normalization requires single section: {recorder}"
                )
            section = sections[0]
        return [f"{output}_ele{int(elements[0])}_sec{int(section)}.out"]
    if rec_type == "drift":
        return [f"{output}_i{int(recorder['i_node'])}_j{int(recorder['j_node'])}.out"]
    raise SystemExit(f"unsupported direct Tcl benchmark recorder normalization: {rec_type}")


def _normalize_reference_outputs(
    case_data: dict, reference_dir: Path, *, strict: bool = True
) -> None:
    data = case_data
    grouped_recorders: dict[str, list[dict]] = {}
    for recorder in data.get("recorders", []):
        raw_path = recorder.get("raw_path")
        if not raw_path or recorder.get("parity", True) is False:
            continue
        grouped_recorders.setdefault(raw_path, []).append(recorder)

    for raw_path, recorders in grouped_recorders.items():
        source = _reference_output_path(reference_dir, raw_path)
        if source is None or not source.exists():
            if strict:
                raise SystemExit(f"missing raw OpenSees recorder output: {raw_path}")
            continue
        strip_time = bool(recorders[0].get("include_time"))
        normalized_rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.replace(",", " ").split()
            if strip_time and parts:
                parts = parts[1:]
            normalized_rows.append(parts)

        targets: list[str] = []
        for recorder in recorders:
            targets.extend(_normalized_recorder_outputs(recorder))
        if not targets:
            continue

        payloads = [[] for _ in targets]
        if len(targets) == 1:
            for parts in normalized_rows:
                payloads[0].append(" ".join(parts))
        else:
            for parts in normalized_rows:
                if len(parts) % len(targets) != 0:
                    raise SystemExit(
                        "cannot evenly split grouped recorder output "
                        f"{raw_path}: {len(parts)} values across {len(targets)} targets"
                    )
                width = len(parts) // len(targets)
                for idx, _ in enumerate(targets):
                    start = idx * width
                    payloads[idx].append(" ".join(parts[start : start + width]))

        for rel_name, rows in zip(targets, payloads):
            payload = "\n".join(rows)
            if payload:
                payload += "\n"
            (reference_dir / rel_name).write_text(payload, encoding="utf-8")


def _normalize_opensees_benchmark_outputs(case_data: dict, output_dir: Path) -> None:
    if not output_dir.exists():
        return
    _normalize_reference_outputs(case_data, output_dir, strict=False)


def _normalize_opensees_output_root(entries: List[dict], output_root: Path) -> None:
    for entry in entries:
        _normalize_opensees_benchmark_outputs(
            entry["case_data"], output_root / entry["name"]
        )


def _canonical_reference_ready(case_data: dict, reference_dir: Path) -> bool:
    if not reference_dir.exists():
        return False
    data = case_data
    for recorder in data.get("recorders", []):
        for rel_name in _normalized_recorder_outputs(recorder):
            if not (reference_dir / rel_name).exists():
                return False
    return True


def _load_direct_tcl_case_data(case: CaseSpec, repo_root: Path) -> dict:
    import tcl_to_strut

    if case.tcl_path is None:
        raise SystemExit(f"direct Tcl case `{case.name}` is missing tcl_path")
    case_data = tcl_to_strut.convert_tcl_to_solver_input(case.tcl_path, repo_root)
    metadata = _load_case_metadata(_case_metadata_path(case))
    for key in ("parity_tolerance", "parity_tolerance_by_recorder", "parity_mode"):
        if key in metadata:
            case_data[key] = metadata[key]
    return case_data


def _write_solver_input_pickle(case_data: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(case_data, handle)
    return output_path


def _write_direct_tcl_wrapper(
    *,
    original_script_name: str,
    wrapper_path: Path,
    compute_only: bool,
) -> Path:
    lines = [
        "set __strut_out_dir [pwd]",
        'set __strut_wrapper_dir [file dirname [info script]]',
        "set __strut_analysis_us 0",
        "cd $__strut_wrapper_dir",
        'if {[llength [info commands analyze]] > 0 && [llength [info commands __strut_orig_analyze]] == 0} {',
        "  rename analyze __strut_orig_analyze",
        "  proc analyze args {",
        "    global __strut_analysis_us",
        "    set __strut_t0 [clock microseconds]",
        "    set __strut_rc [catch {uplevel 1 [linsert $args 0 __strut_orig_analyze]} __strut_result __strut_opts]",
        "    set __strut_t1 [clock microseconds]",
        "    incr __strut_analysis_us [expr {$__strut_t1 - $__strut_t0}]",
        "    return -options $__strut_opts $__strut_result",
        "  }",
        "}",
        'if {[llength [info commands eigen]] > 0 && [llength [info commands __strut_orig_eigen]] == 0} {',
        "  rename eigen __strut_orig_eigen",
        "  proc eigen args {",
        "    global __strut_analysis_us",
        "    set __strut_t0 [clock microseconds]",
        "    set __strut_rc [catch {uplevel 1 [linsert $args 0 __strut_orig_eigen]} __strut_result __strut_opts]",
        "    set __strut_t1 [clock microseconds]",
        "    incr __strut_analysis_us [expr {$__strut_t1 - $__strut_t0}]",
        "    return -options $__strut_opts $__strut_result",
        "  }",
        "}",
    ]
    if compute_only:
        lines.extend(
            [
                'if {[llength [info commands recorder]] > 0 && [llength [info commands __strut_orig_recorder]] == 0} {',
                "  rename recorder __strut_orig_recorder",
                "  proc recorder args { return {} }",
                "}",
            ]
        )
    lines.extend(
        [
            f'set __strut_case_rc [catch {{source {{{original_script_name}}}}} __strut_case_msg __strut_case_opts]',
            'set __strut_fp [open "analysis_time_us.txt" w]',
            "puts $__strut_fp $__strut_analysis_us",
            "close $__strut_fp",
            'if {$__strut_out_dir ne $__strut_wrapper_dir} {',
            '  if {[file exists "Data"]} {',
            '    file copy -force "Data" $__strut_out_dir/',
            "  }",
            '  if {[file exists "analysis_time_us.txt"]} {',
            '    file copy -force "analysis_time_us.txt" $__strut_out_dir/',
            "  }",
            "}",
            "cd $__strut_out_dir",
            "if {$__strut_case_rc != 0} {",
            "  return -options $__strut_case_opts $__strut_case_msg",
            "}",
        ]
    )
    wrapper_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return wrapper_path


def _prepare_direct_tcl_wrappers(
    case: CaseSpec, repo_root: Path, tcl_root: Path
) -> tuple[Path, Path]:
    if case.tcl_path is None:
        raise SystemExit(f"direct Tcl case `{case.name}` is missing tcl_path")

    script_path = case.tcl_path.resolve()
    script_dir = script_path.parent
    script_parent = script_dir.parent
    mirror_root = tcl_root / "_direct_tcl_mirror" / case.name
    ensure_clean_dir(mirror_root)
    mirrored_parent = mirror_root / script_parent.name
    shutil.copytree(script_parent, mirrored_parent, dirs_exist_ok=True)
    mirrored_script_dir = mirrored_parent / script_dir.name

    timed_wrapper = _write_direct_tcl_wrapper(
        original_script_name=script_path.name,
        wrapper_path=mirrored_script_dir / f"__strut_{case.name}_timed.tcl",
        compute_only=False,
    )
    compute_wrapper = _write_direct_tcl_wrapper(
        original_script_name=script_path.name,
        wrapper_path=mirrored_script_dir / f"__strut_{case.name}_compute.tcl",
        compute_only=True,
    )
    return timed_wrapper, compute_wrapper


def _emit_case_tcl(case_json: Path, tcl_out: Path, repo_root: Path, env, verbose: bool) -> Path:
    tcl_out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "json_to_tcl.py"),
            str(case_json),
            str(tcl_out),
        ],
        env=env,
        verbose=verbose,
    )
    return tcl_out


def _prepare_case_json_for_solver(case_json: Path, output_path: Path) -> Path:
    case_data = _load_case_metadata(case_json)
    _absolutize_time_series_paths(case_data, case_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(case_data, indent=2) + "\n", encoding="utf-8")
    return output_path


def _ensure_direct_tcl_case_artifacts(
    case: CaseSpec, repo_root: Path, env, verbose: bool
) -> dict:
    if case.tcl_path is None:
        return _load_case_metadata(_case_json_path(case))

    case_root = _direct_case_root(case)
    case_root.mkdir(parents=True, exist_ok=True)
    case_data = _load_direct_tcl_case_data(case, repo_root)
    reference_dir = case_root / "reference"
    if not _canonical_reference_ready(case_data, reference_dir):
        log(f"{case.name}: canonical OpenSees reference missing; running original Tcl.")
        ensure_clean_dir(reference_dir)
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(case.tcl_path),
                "--output",
                str(reference_dir),
            ],
            env=env,
            verbose=verbose,
        )
    _normalize_reference_outputs(case_data, reference_dir)
    return case_data


def _case_free_dofs(case_data: dict) -> Optional[int]:
    data = case_data
    model = data.get("model", {})
    ndf = model.get("ndf")
    nodes = data.get("nodes")
    if not isinstance(ndf, int) or not isinstance(nodes, list):
        return None
    constrained = 0
    for node in nodes:
        if isinstance(node, dict):
            constrained += _count_constrained_dofs(node, ndf)
    total = len(nodes) * ndf
    return total - constrained


def _case_size_override(case_data: dict) -> Optional[str]:
    data = case_data
    label = data.get("benchmark_size")
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    if normalized in {"small", "medium", "large"}:
        return normalized
    return None


def _analysis_is_transient(analysis: dict) -> bool:
    analysis_type = str(analysis.get("type", "static_linear"))
    if analysis_type.startswith("transient"):
        return True
    if analysis_type != "staged":
        return False
    stages = analysis.get("stages", [])
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_analysis = stage.get("analysis", stage)
        if not isinstance(stage_analysis, dict):
            continue
        stage_type = str(stage_analysis.get("type", ""))
        if stage_type.startswith("transient"):
            return True
    return False


def _load_case_flags(path: Path) -> Tuple[bool, bool]:
    data = _load_case_metadata(path)
    enabled = bool(data.get("enabled", True))
    disabled = not enabled
    runnable = enabled
    return disabled, runnable


def _is_case_json(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if path.parent.name != path.stem:
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    required = ("model", "nodes", "elements", "recorders")
    return all(key in data for key in required)


def _include_in_default_benchmarks(path: Path) -> bool:
    data = _load_case_metadata(path)
    if not bool(data.get("enabled", True)):
        return False
    status = str(data.get("status", "")).strip().lower()
    if status == "benchmark":
        return True
    # Keep the default path focused on the lightweight elastic benchmark set.
    if path.name == "direct_tcl_case.json":
        stem = str(data.get("name") or path.parent.name)
    else:
        stem = path.stem
    stem_tokens = [token for token in stem.lower().split("_") if token]
    return "elastic" in stem_tokens


def load_case_enabled(path: Path) -> bool:
    _, runnable = _load_case_flags(path)
    return runnable


def _absolutize_time_series_paths(case_data: dict, case_json_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve_optional_repo_path(path_text):
        if not isinstance(path_text, str) or not path_text:
            return None
        path_obj = Path(path_text)
        if path_obj.is_absolute():
            return path_obj
        return (repo_root / path_obj).resolve()

    time_series = case_data.get("time_series")
    if isinstance(time_series, dict):
        entries = [time_series]
    elif isinstance(time_series, list):
        entries = time_series
    else:
        return
    for ts in entries:
        if not isinstance(ts, dict):
            continue
        key = None
        if "values_path" in ts:
            key = "values_path"
        elif "path" in ts:
            key = "path"
        if key is None:
            continue
        raw = ts.get(key)
        if not isinstance(raw, str):
            continue
        raw_path = Path(raw)
        if raw_path.is_absolute():
            continue
        from_case_dir = case_json_path.parent / raw_path
        if from_case_dir.exists():
            ts[key] = str(from_case_dir.resolve())
            continue
        from_repo_root = repo_root / raw_path
        if from_repo_root.exists():
            ts[key] = str(from_repo_root.resolve())
            continue
        source_example = _resolve_optional_repo_path(case_data.get("source_example"))
        if source_example is not None:
            from_source_example = (source_example.parent / raw_path).resolve()
            if from_source_example.exists():
                ts[key] = str(from_source_example)
                continue
        source_doc = _resolve_optional_repo_path(case_data.get("source_doc"))
        if source_doc is not None:
            from_source_doc = (source_doc.parent / raw_path).resolve()
            if from_source_doc.exists():
                ts[key] = str(from_source_doc)
                continue
        ts[key] = str(from_case_dir.resolve())


def filter_cases_by_enabled(
    case_specs: List[CaseSpec],
    include_disabled: bool,
) -> Tuple[List[CaseSpec], int, int]:
    filtered = []
    disabled_selected = 0
    skipped_disabled = 0
    for case in case_specs:
        disabled, runnable = _load_case_flags(_case_metadata_path(case))
        if disabled:
            disabled_selected += 1
        if include_disabled or runnable:
            filtered.append(case)
        else:
            skipped_disabled += 1
    return filtered, disabled_selected, skipped_disabled


def resolve_case_from_name(validation_root: Path, name: str) -> Optional[CaseSpec]:
    case_dir = validation_root / name
    case_json = case_dir / f"{name}.json"
    if case_json.exists():
        return CaseSpec(name=name, json_path=case_json, metadata_path=case_json)
    direct_manifest = case_dir / "direct_tcl_case.json"
    if _is_direct_tcl_manifest(direct_manifest):
        return _direct_tcl_case_spec(direct_manifest)
    return None


def resolve_case_from_path(path: Path) -> Optional[CaseSpec]:
    if not path.exists():
        return None
    if _is_direct_tcl_manifest(path):
        return _direct_tcl_case_spec(path)
    if not _is_case_json(path):
        return None
    return CaseSpec(name=path.stem, json_path=path, metadata_path=path)


def expand_case_patterns(
    validation_root: Path, patterns: Iterable[str]
) -> List[CaseSpec]:
    cases = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            for match in validation_root.glob(f"{pattern}/{pattern}.json"):
                cases.append(
                    CaseSpec(name=match.stem, json_path=match, metadata_path=match)
                )
            for match in validation_root.glob(f"{pattern}/direct_tcl_case.json"):
                if _is_direct_tcl_manifest(match):
                    cases.append(_direct_tcl_case_spec(match))
            for match in validation_root.glob(pattern):
                if match.is_dir():
                    candidate = match / f"{match.name}.json"
                    if candidate.exists():
                        cases.append(
                            CaseSpec(
                                name=candidate.stem,
                                json_path=candidate,
                                metadata_path=candidate,
                            )
                        )
                    else:
                        direct_manifest = match / "direct_tcl_case.json"
                        if _is_direct_tcl_manifest(direct_manifest):
                            cases.append(_direct_tcl_case_spec(direct_manifest))
                elif match.suffix == ".json":
                    cases.append(
                        CaseSpec(name=match.stem, json_path=match, metadata_path=match)
                    )
        else:
            case = resolve_case_from_name(validation_root, pattern)
            if case:
                cases.append(case)
    unique = {_case_metadata_path(case).resolve(): case for case in cases}
    return sorted(unique.values(), key=lambda c: c.name)


def discover_default_cases(validation_root: Path) -> List[CaseSpec]:
    cases = []
    for match in validation_root.glob("*/*.json"):
        if _is_direct_tcl_manifest(match):
            continue
        if not _is_case_json(match):
            continue
        if not _include_in_default_benchmarks(match):
            continue
        cases.append(CaseSpec(name=match.stem, json_path=match, metadata_path=match))
    for match in validation_root.glob("*/direct_tcl_case.json"):
        if not _is_direct_tcl_manifest(match):
            continue
        if not _include_in_default_benchmarks(match):
            continue
        cases.append(_direct_tcl_case_spec(match))
    return sorted(cases, key=lambda c: c.name)


def discover_all_cases(validation_root: Path) -> List[CaseSpec]:
    cases = [
        CaseSpec(name=match.stem, json_path=match, metadata_path=match)
        for match in validation_root.glob("*/*.json")
        if _is_case_json(match)
    ]
    cases.extend(
        _direct_tcl_case_spec(match)
        for match in validation_root.glob("*/direct_tcl_case.json")
        if _is_direct_tcl_manifest(match)
    )
    return sorted(cases, key=lambda c: c.name)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def git_rev(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.decode("utf-8").strip()


def _git_branch(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    branch = out.decode("utf-8").strip()
    return branch or None


def _read_cpu_model() -> Optional[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                model = value.strip()
                if model:
                    return model
    proc = platform.processor().strip()
    return proc or None


def _safe_check_output(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    text = out.decode("utf-8", errors="replace").strip()
    return text or None


def collect_run_metadata(
    repo_root: Path,
    args: argparse.Namespace,
    results_root: Path,
    profile_root: Optional[Path],
    strut_solver: Optional[Path],
) -> dict:
    metadata = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_model": _read_cpu_model(),
        },
        "git": {
            "rev": git_rev(repo_root),
            "branch": _git_branch(repo_root),
        },
        "runner": {
            "benchmark_suite": args.benchmark_suite,
            "engine": args.engine,
            "batch_mode": not args.no_batch,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "profile_enabled": bool(args.profile),
            "results_root": str(results_root),
            "profile_root": str(profile_root) if profile_root is not None else None,
            "standard_optimization_commands": STANDARD_OPT_COMMANDS,
        },
        "build": {
            "strut_solver": str(strut_solver) if strut_solver is not None else None,
            "profile_instrumented": bool(args.profile),
            "mojo_version": _safe_check_output(
                ["uv", "run", "mojo", "--version"], cwd=repo_root
            ),
        },
    }
    return metadata


def _load_phase_times(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            out[str(key)] = int(value)
    return out


def _load_profile_frame_totals(profile_path: Path) -> Dict[str, int]:
    if not profile_path.exists():
        return {}
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    shared = data.get("shared", {})
    frames = shared.get("frames", [])
    profiles = data.get("profiles", [])
    if not isinstance(frames, list) or not profiles:
        return {}
    profile = profiles[0]
    events = profile.get("events", [])
    if not isinstance(events, list):
        return {}

    frame_names: Dict[int, str] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        name = frame.get("name")
        if isinstance(name, str) and name:
            frame_names[index] = name

    open_stack: Dict[int, List[int]] = {}
    totals: Dict[int, int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        frame = event.get("frame")
        event_type = event.get("type")
        at = event.get("at")
        if not isinstance(frame, int) or not isinstance(at, (int, float)):
            continue
        at_us = int(at)
        if event_type == "O":
            open_stack.setdefault(frame, []).append(at_us)
            continue
        if event_type != "C":
            continue
        stack = open_stack.get(frame)
        if not stack:
            continue
        start = stack.pop()
        if at_us < start:
            continue
        totals[frame] = totals.get(frame, 0) + (at_us - start)

    named_totals: Dict[str, int] = {}
    for frame, total in totals.items():
        name = frame_names.get(frame)
        if name is None:
            continue
        named_totals[name] = named_totals.get(name, 0) + total
    return named_totals


def _build_phase_row(
    case_name: str,
    dofs: Optional[int],
    phase_times: Dict[str, int],
    frame_totals: Dict[str, int],
) -> dict:
    row: Dict[str, object] = {"case": case_name, "dofs": dofs or ""}
    row["case_load_parse_us"] = phase_times.get("case_load_parse_us")
    row["model_build_dof_map_us"] = phase_times.get("model_build_dof_map_us")
    row["output_write_us"] = phase_times.get("output_write_us")
    row["solve_total_us"] = phase_times.get(
        "solve_total_us", phase_times.get("analysis_us")
    )
    row["total_case_us"] = phase_times.get("total_case_us")

    for key, frame_names in PHASE_FRAME_MAP.items():
        total = sum(frame_totals.get(name, 0) for name in frame_names)
        row[key] = total if total > 0 else None
    return row


def write_phase_summary_csv(path: Path, rows: List[dict]) -> None:
    headers = ["case", "dofs"] + PHASE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = {"case": row.get("case", ""), "dofs": row.get("dofs", "")}
            for key in PHASE_COLUMNS:
                value = row.get(key)
                out[key] = "" if value is None else value
            writer.writerow(out)


def write_phase_rollup_csv(path: Path, rows: List[dict]) -> None:
    headers = ["phase", "count", "mean_us", "median_us", "min_us", "max_us"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for phase in PHASE_COLUMNS:
            values = [
                int(row[phase])
                for row in rows
                if isinstance(row.get(phase), (int, float))
            ]
            if not values:
                continue
            writer.writerow(
                {
                    "phase": phase,
                    "count": len(values),
                    "mean_us": int(mean(values)),
                    "median_us": int(median(values)),
                    "min_us": int(min(values)),
                    "max_us": int(max(values)),
                }
            )


def log_phase_table(rows: List[dict]) -> None:
    if not rows:
        return
    log("Phase timing summary (mean, ms):")
    for phase in PHASE_COLUMNS:
        values = [
            int(row[phase])
            for row in rows
            if isinstance(row.get(phase), (int, float))
        ]
        if not values:
            continue
        avg_ms = mean(values) / 1000.0
        med_ms = median(values) / 1000.0
        log(f"  {phase:<24} mean={avg_ms:>8.3f}  median={med_ms:>8.3f}")


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    headers = [
        "case",
        "dofs",
        "engine",
        "mode",
        "repeat",
        "warmup",
        "mean_s",
        "median_s",
        "min_s",
        "analysis_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_engine(
    cmd_builder,
    output_root: Path,
    repeat: int,
    warmup: int,
) -> List[float]:
    if warmup > 0:
        for _ in range(warmup):
            cmd_builder(output_root, last_run=False)
    times = []
    for idx in range(repeat):
        is_last = idx == repeat - 1
        start = time.perf_counter()
        cmd_builder(output_root, last_run=is_last)
        end = time.perf_counter()
        times.append(end - start)
    return times


def ensure_strut_solver(repo_root: Path, verbose: bool, profile: bool) -> Path:
    if shutil.which("uv") is None:
        raise SystemExit(
            "uv executable not found on PATH; required to build the Mojo solver."
        )
    solver_path = os.getenv("STRUT_MOJO_BIN")
    if solver_path:
        return Path(solver_path)
    solver_path = repo_root / "build" / "strut" / "strut"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    log("Building Mojo solver...")
    build_cmd = [str(repo_root / "scripts" / "build_mojo_solver.sh")]
    build_env = None
    if profile:
        build_env = os.environ.copy()
        build_env["STRUT_PROFILE"] = "1"
    run(
        build_cmd,
        env=build_env,
        verbose=verbose,
    )
    return solver_path


def _parse_line(line: str) -> List[float]:
    line = line.strip()
    if not line:
        return []
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    return [float(p) for p in parts]


def _load_last_values(path: Path) -> List[float]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    return _parse_line(lines[-1])


def _load_all_values(path: Path) -> List[List[float]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty output file: {path}")
    return [_parse_line(ln) for ln in lines]


def _load_last_comparable_values(
    ref_path: Path,
    got_path: Path,
    use_last_common_row: bool,
) -> Tuple[List[float], List[float]]:
    ref_rows = _load_all_values(ref_path)
    got_rows = _load_all_values(got_path)
    if not use_last_common_row:
        return ref_rows[-1], got_rows[-1]
    shared = min(len(ref_rows), len(got_rows))
    if shared == 0:
        raise ValueError(f"empty comparable output rows: {ref_path} vs {got_path}")
    return ref_rows[shared - 1], got_rows[shared - 1]


def _isclose(a: float, b: float, rtol: float = REL_TOL, atol: float = ABS_TOL) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def _compare_vectors(
    ref: List[float],
    got: List[float],
    rtol: float = REL_TOL,
    atol: float = ABS_TOL,
):
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


def _compare_mode_shape_vectors(
    ref: List[float],
    got: List[float],
    rtol: float = REL_TOL,
    atol: float = ABS_TOL,
):
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


def _max_abs_vector(rows: List[List[float]]) -> List[float]:
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
    ref_vals: List[List[float]],
    strut_vals: List[List[float]],
    label: str,
    failures: List[str],
    rtol: float,
    atol: float,
    parity_mode: str,
) -> None:
    if parity_mode == "max_abs":
        ref_peak = _max_abs_vector(ref_vals)
        strut_peak = _max_abs_vector(strut_vals)
        ok, errors = _compare_vectors(ref_peak, strut_peak, rtol=rtol, atol=atol)
        if not ok:
            failures.append(f"{label} max-abs mismatch")
            failures.extend([f"  {err}" for err in errors])
        return
    if len(ref_vals) != len(strut_vals):
        failures.append(f"{label} step count mismatch: {len(ref_vals)} != {len(strut_vals)}")
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
) -> Tuple[float, float]:
    default_entry = DEFAULT_RECORDER_TOLERANCES.get(rec_type, {})
    rtol = float(default_entry.get("rtol", REL_TOL))
    atol = float(default_entry.get("atol", ABS_TOL))
    if has_global_override:
        rtol = float(global_rtol)
        atol = float(global_atol)
    override = per_recorder_overrides.get(rec_type, {})
    if isinstance(override, dict):
        if "rtol" in override:
            rtol = float(override["rtol"])
        if "atol" in override:
            atol = float(override["atol"])
    return rtol, atol


def _inject_opensees_timing(tcl_lines: List[str], timing_file: str) -> List[str]:
    out = ["set __strut_analysis_us 0"]
    injected = False
    for line in tcl_lines:
        stripped = line.lstrip()
        has_analyze = stripped.startswith("analyze ") or "[analyze " in stripped
        has_eigen = stripped.startswith("eigen ") or "[eigen " in stripped
        if has_analyze or has_eigen:
            out.append("set __strut_t0 [clock microseconds]")
            out.append(line)
            out.append("set __strut_t1 [clock microseconds]")
            out.append("incr __strut_analysis_us [expr {$__strut_t1 - $__strut_t0}]")
            injected = True
            continue
        out.append(line)
    if not injected:
        raise ValueError(
            "failed to inject timing: analyze/eigen command not found in Tcl"
        )
    out.append(f'set __strut_fp [open "{timing_file}" w]')
    out.append("puts $__strut_fp $__strut_analysis_us")
    out.append("close $__strut_fp")
    return out


def _tcl_uses_eigen(path: Path) -> bool:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("eigen ") or "[eigen " in stripped:
            return True
    return False


def _load_case_metric(output_root: Path, entries: List[dict], filename: str) -> dict:
    times = {}
    for entry in entries:
        case_name = entry["name"]
        metric_file = output_root / case_name / filename
        if metric_file.exists():
            try:
                times[case_name] = int(metric_file.read_text().strip())
            except ValueError:
                times[case_name] = None
        else:
            times[case_name] = None
    return times


def _read_analysis_us(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def _summarize_parity_failures(parity_failures: List[str]) -> List[str]:
    grouped: Dict[str, Dict[str, List[str]]] = {}
    case_order: List[str] = []
    case_last_category: Dict[str, str] = {}
    last_case: Optional[str] = None
    category_order = [
        "node mismatch",
        "reaction mismatch",
        "element mismatch",
        "drift mismatch",
        "envelope mismatch",
        "modal mismatch",
        "unsupported recorder",
        "parse/read errors",
        "details",
        "other",
    ]
    label_by_category = {
        "node mismatch": "Node Mismatch",
        "reaction mismatch": "Reaction Mismatch",
        "element mismatch": "Element Mismatch",
        "drift mismatch": "Drift Mismatch",
        "envelope mismatch": "Envelope Mismatch",
        "modal mismatch": "Modal Mismatch",
        "unsupported recorder": "Unsupported Recorder",
        "parse/read errors": "Parse/Read Errors",
        "details": "Details",
        "other": "Other",
    }

    def add(case: str, category: str, detail: str) -> None:
        if case not in grouped:
            grouped[case] = {}
            case_order.append(case)
        bucket = grouped[case].setdefault(category, [])
        if detail not in bucket:
            bucket.append(detail)
        case_last_category[case] = category

    for failure in parity_failures:
        if failure.startswith("  "):
            if last_case is None:
                continue
            detail = failure.strip()
            last_category = case_last_category.get(last_case)
            category = "details"
            if last_category in {
                "node mismatch",
                "reaction mismatch",
                "element mismatch",
                "drift mismatch",
                "envelope mismatch",
                "modal mismatch",
            }:
                category = last_category
            add(last_case, category, detail)
            continue

        if ":" not in failure:
            add("unknown", "other", failure)
            last_case = "unknown"
            continue

        case_name, raw_detail = failure.split(":", 1)
        case_name = case_name.strip()
        detail = raw_detail.strip()
        last_case = case_name

        if detail.startswith("missing OpenSees output:"):
            path = detail.split(":", 1)[1].strip()
            add(case_name, "missing opensees files", Path(path).name)
            continue
        if detail.startswith("missing Mojo output:"):
            path = detail.split(":", 1)[1].strip()
            add(case_name, "missing strut files", Path(path).name)
            continue
        if detail.startswith("node "):
            add(case_name, "node mismatch", detail)
            continue
        if detail.startswith("reaction node "):
            add(case_name, "reaction mismatch", detail)
            continue
        if detail.startswith("element "):
            add(case_name, "element mismatch", detail)
            continue
        if detail.startswith("drift "):
            add(case_name, "drift mismatch", detail)
            continue
        if detail.startswith("envelope element "):
            add(case_name, "envelope mismatch", detail)
            continue
        if detail.startswith("modal "):
            add(case_name, "modal mismatch", detail)
            continue
        if detail.startswith("unsupported recorder type:"):
            add(case_name, "unsupported recorder", detail)
            continue
        if detail.startswith("empty output file:"):
            path = detail.split(":", 1)[1].strip()
            add(case_name, "parse/read errors", f"empty output file: {Path(path).name}")
            continue
        add(case_name, "other", detail)

    lines = []
    for case_name in case_order:
        lines.append(f"Error: {case_name}")
        case_categories = grouped[case_name]
        missing_opensees = case_categories.get("missing opensees files", [])
        missing_strut = case_categories.get("missing strut files", [])
        if missing_opensees and not missing_strut:
            lines.append("Missing all Opensees Outputs")
        elif missing_strut and not missing_opensees:
            lines.append("Missing all Mojo Outputs")
        else:
            if missing_opensees:
                lines.append(
                    f"Missing Opensees outputs: {json.dumps(missing_opensees)}"
                )
            if missing_strut:
                lines.append(f"Missing Mojo outputs: {json.dumps(missing_strut)}")
        for category in category_order:
            details = case_categories.get(category)
            if details:
                lines.append(f"{label_by_category[category]}: {json.dumps(details)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenSees vs Mojo benchmarks")
    parser.add_argument(
        "--cases",
        action="append",
        help="Case name, JSON path, or glob. Repeatable.",
    )
    parser.add_argument(
        "--benchmark-suite",
        choices=tuple(sorted(BENCHMARK_SUITES.keys())),
        default=None,
        help="Run a predefined benchmark suite.",
    )
    parser.add_argument(
        "--list-benchmark-suites",
        action="store_true",
        help="List available benchmark suites and exit.",
    )
    parser.add_argument(
        "--gen-frame-bays",
        type=int,
        default=None,
        help="Generate a synthetic frame with this number of bays.",
    )
    parser.add_argument(
        "--gen-frame-stories",
        type=int,
        default=None,
        help="Generate a synthetic frame with this number of stories.",
    )
    parser.add_argument(
        "--gen-frame-name",
        default=None,
        help="Optional name for generated frame case (default: elastic_frame_{bays}bay_{stories}story).",
    )
    parser.add_argument(
        "--gen-frame-element",
        choices=("elasticBeamColumn2d", "forceBeamColumn2d", "dispBeamColumn2d"),
        default="elasticBeamColumn2d",
        help="Element type for generated frame case.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=int(os.getenv("STRUT_BENCH_REPEAT", "1")),
        help="Number of timed repetitions per engine.",
    )
    parser.add_argument(
        "--engine",
        choices=("both", "opensees", "strut"),
        default=os.getenv("STRUT_BENCH_ENGINE", "both"),
        help="Which engine(s) to benchmark.",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable default batch mode and run each case in a separate process.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.getenv("STRUT_BENCH_WARMUP", "0")),
        help="Warmup runs per engine (not timed).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Results root (default: benchmark/results or benchmark/results-profile with --profile).",
    )
    parser.add_argument(
        "--archive-root",
        default=None,
        help="Archive root (default: benchmark/archive).",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip writing summary archives.",
    )
    parser.add_argument(
        "--skip-compute-only",
        action="store_true",
        help="Skip the second pass that runs without recorders.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="DIR",
        help="Emit per-case speedscope profiles to DIR.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include JSON cases marked enabled=false.",
    )
    args = parser.parse_args()

    if args.list_benchmark_suites:
        print("Available benchmark suites:")
        for name in sorted(BENCHMARK_SUITES):
            print(f"- {name}: {', '.join(BENCHMARK_SUITES[name])}")
        print("")
        print("Standard optimization commands:")
        for key, value in STANDARD_OPT_COMMANDS.items():
            print(f"- {key}: {value}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    validation_root = repo_root / "tests" / "validation"
    generated_cases: List[CaseSpec] = []
    generate_suite_force_pair = False

    if args.benchmark_suite:
        if args.cases:
            raise SystemExit("--benchmark-suite cannot be combined with --cases.")
        args.cases = [",".join(BENCHMARK_SUITES[args.benchmark_suite])]
        if args.benchmark_suite in {"regression_gate_v1", "scaling_v1"}:
            if args.gen_frame_bays is None and args.gen_frame_stories is None:
                args.gen_frame_bays = 18
                args.gen_frame_stories = 17
                generate_suite_force_pair = True

    batch_mode = not args.no_batch
    auto_batch_default_gen = (
        batch_mode
        and args.cases is None
        and args.gen_frame_bays is None
        and args.gen_frame_stories is None
    )
    if auto_batch_default_gen:
        args.gen_frame_bays = 18
        args.gen_frame_stories = 17

    if (args.gen_frame_bays is None) != (args.gen_frame_stories is None):
        raise SystemExit(
            "--gen-frame-bays and --gen-frame-stories must be provided together."
        )
    if args.gen_frame_bays is not None and args.gen_frame_stories is not None:
        if args.gen_frame_bays <= 0 or args.gen_frame_stories <= 0:
            raise SystemExit("--gen-frame-bays and --gen-frame-stories must be > 0.")
        default_prefix = "elastic_frame"
        if args.gen_frame_element == "forceBeamColumn2d":
            default_prefix = "force_beam_column2d_fiber_frame"
        if args.gen_frame_element == "dispBeamColumn2d":
            default_prefix = "disp_beam_column2d_fiber_frame"
        name = (
            args.gen_frame_name
            or f"{default_prefix}_{args.gen_frame_bays}bay_{args.gen_frame_stories}story"
        )
        gen_dir = repo_root / "benchmark" / ".tmp"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_path = gen_dir / f"{name}.json"
        run(
            [
                "uv",
                "run",
                "python",
                str(repo_root / "scripts" / "gen_frame_case.py"),
                "--bays",
                str(args.gen_frame_bays),
                "--stories",
                str(args.gen_frame_stories),
                "--element-type",
                args.gen_frame_element,
                "--name",
                name,
                "--output",
                str(gen_path),
            ],
            verbose=os.getenv("STRUT_VERBOSE") == "1",
        )
        generated_cases.append(CaseSpec(name=name, json_path=gen_path))
        if (
            auto_batch_default_gen or generate_suite_force_pair
        ) and args.gen_frame_element != "forceBeamColumn2d":
            fiber_name = (
                f"force_beam_column2d_fiber_frame_{args.gen_frame_bays}bay_"
                f"{args.gen_frame_stories}story"
            )
            fiber_path = gen_dir / f"{fiber_name}.json"
            run(
                [
                    "uv",
                    "run",
                    "python",
                    str(repo_root / "scripts" / "gen_frame_case.py"),
                    "--bays",
                    str(args.gen_frame_bays),
                    "--stories",
                    str(args.gen_frame_stories),
                    "--element-type",
                    "forceBeamColumn2d",
                    "--name",
                    fiber_name,
                    "--output",
                    str(fiber_path),
                ],
                verbose=os.getenv("STRUT_VERBOSE") == "1",
            )
            generated_cases.append(CaseSpec(name=fiber_name, json_path=fiber_path))

    if args.cases:
        case_specs = []
        for raw in args.cases:
            if raw is None:
                continue
            for part in str(raw).split(","):
                part = part.strip()
                if not part:
                    continue
                path = Path(part)
                case = resolve_case_from_path(path)
                if case:
                    case_specs.append(case)
                else:
                    case_specs.extend(expand_case_patterns(validation_root, [part]))
    else:
        case_specs = discover_default_cases(validation_root)

    if generated_cases:
        case_specs.extend(generated_cases)

    if not case_specs:
        raise SystemExit(
            "No benchmark cases found. Provide --cases or add validation cases."
        )

    include_disabled_effective = args.include_disabled or (
        args.cases is None and os.getenv("STRUT_RUN_ALL_CASES") == "1"
    )
    case_specs, disabled_selected_count, skipped_disabled_count = (
        filter_cases_by_enabled(case_specs, include_disabled=include_disabled_effective)
    )

    if not case_specs:
        raise SystemExit(
            "All selected cases are disabled. Use --include-disabled to run."
        )

    if args.profile and not args.no_archive:
        args.no_archive = True

    if args.output_root:
        results_root = Path(args.output_root)
    elif args.profile:
        results_root = repo_root / "benchmark" / "results-profile"
    else:
        results_root = repo_root / "benchmark" / "results"
    archive_root = (
        Path(args.archive_root)
        if args.archive_root
        else repo_root / "benchmark" / "archive"
    )
    profile_root = None
    if args.profile:
        profile_root = Path(args.profile)
        ensure_clean_dir(profile_root)

    for sub in (
        "opensees",
        "strut",
        "opensees_compute",
        "strut_compute",
        "tcl",
        ".tmp",
    ):
        ensure_clean_dir(results_root / sub)

    env = os.environ.copy()
    verbose = env.get("STRUT_VERBOSE") == "1"

    summary_cases = []
    csv_rows = []
    parity_failures = []
    benchmark_parity_failures = []

    log(f"Running {len(case_specs)} benchmark case(s).")
    log(
        "Disabled selected: "
        f"{disabled_selected_count}; "
        f"skipped as disabled: {skipped_disabled_count}."
    )

    run_opensees = args.engine in ("both", "opensees")
    run_strut = args.engine in ("both", "strut")
    strut_solver = None
    if run_strut:
        strut_solver = ensure_strut_solver(repo_root, verbose, args.profile)
    run_metadata = collect_run_metadata(
        repo_root=repo_root,
        args=args,
        results_root=results_root,
        profile_root=profile_root,
        strut_solver=strut_solver,
    )

    if not run_opensees and not run_strut:
        raise SystemExit("No engines selected. Use --engine opensees|strut|both.")

    prepared_case_data: Dict[str, dict] = {}
    for case in case_specs:
        prepared_case_data[case.name] = _ensure_direct_tcl_case_artifacts(
            case, repo_root, env, verbose
        )

    def build_case_tcl(case: CaseSpec) -> dict:
        case_name = case.name
        try:
            case_data = prepared_case_data[case.name]
        except KeyError as exc:
            raise SystemExit(
                f"case `{case.name}` was not prepared before benchmarking"
            ) from exc
        case_entry = {
            "name": case_name,
            "case_data": case_data,
            "dofs": _case_free_dofs(case_data),
            "status": str(
                _load_case_metadata(_case_metadata_path(case)).get("status", "")
            )
            .strip()
            .lower(),
        }
        if case.json_path is not None:
            case_entry["json"] = str(_case_json_path(case))
        elif run_strut:
            case_entry["input_pickle"] = str(
                _write_solver_input_pickle(
                    case_data,
                    results_root / ".tmp" / "strut_inputs" / f"{case_name}.pkl",
                )
            )
        size_override = case.benchmark_size or _case_size_override(case_data)
        if size_override is not None:
            case_entry["size"] = size_override

        if case.tcl_path is not None:
            tcl_timed, tcl_compute = _prepare_direct_tcl_wrappers(
                case, repo_root, results_root / "tcl"
            )
            case_entry["tcl"] = str(case.tcl_path)
            case_entry["uses_eigen"] = _tcl_uses_eigen(case.tcl_path)
        else:
            case_json = _case_json_path(case)
            tcl_out = _emit_case_tcl(
                case_json,
                results_root / "tcl" / f"{case_name}.tcl",
                repo_root,
                env,
                verbose,
            )
            tcl_compute = tcl_out.parent / f"{tcl_out.stem}_compute.tcl"
            tcl_timed = tcl_out.parent / f"{tcl_out.stem}_timed.tcl"
            tcl_lines = tcl_out.read_text().splitlines()
            tcl_compute.write_text(
                "\n".join(
                    line for line in tcl_lines if not line.lstrip().startswith("recorder ")
                )
                + "\n",
                encoding="utf-8",
            )
            tcl_timed.write_text(
                "\n".join(_inject_opensees_timing(tcl_lines, "analysis_time_us.txt"))
                + "\n",
                encoding="utf-8",
            )
            case_entry["tcl"] = str(tcl_out)
            case_entry["uses_eigen"] = _tcl_uses_eigen(tcl_timed)
        case_entry["tcl_compute"] = str(tcl_compute)
        case_entry["tcl_timed"] = str(tcl_timed)
        return case_entry

    case_entries = [build_case_tcl(case) for case in case_specs]
    if batch_mode:
        batch_case_entries = []
        skipped_eigen_cases = []
        for entry in case_entries:
            if entry.get("uses_eigen", False):
                skipped_eigen_cases.append(entry["name"])
                continue
            batch_case_entries.append(entry)
        if skipped_eigen_cases:
            log(
                "Batch mode: skipping eigen/modal cases: "
                + ", ".join(sorted(skipped_eigen_cases))
            )
        case_entries = batch_case_entries
        if not case_entries:
            raise SystemExit(
                "No non-eigen cases remain for batch mode. Use --no-batch to run eigen/modal cases."
            )
    case_entries_by_name = {entry["name"]: entry for entry in case_entries}

    def _write_batch_tcl(entries: List[dict], output_root: Path, compute: bool) -> Path:
        lines = ["# Auto-generated by run_benchmarks.py", "set __strut_repo [pwd]"]
        eigen_warmup_tcl: Optional[Path] = None
        for entry in entries:
            tcl_path = Path(entry["tcl_compute" if compute else "tcl_timed"])
            if _tcl_uses_eigen(tcl_path):
                eigen_warmup_tcl = tcl_path
                break
        if eigen_warmup_tcl is not None:
            # Eigen has a heavy first-call initialization cost in OpenSees/Wine.
            # Warm it once outside per-case timers using a real eigen case script.
            lines.extend(
                [
                    "set __strut_eigen_warmup_dir [file join $__strut_repo __strut_eigen_warmup]",
                    "file mkdir $__strut_eigen_warmup_dir",
                    "cd $__strut_eigen_warmup_dir",
                    "wipe",
                    f"source {{{eigen_warmup_tcl}}}",
                    "cd $__strut_repo",
                    "wipe",
                ]
            )
        for entry in entries:
            case_name = entry["name"]
            case_out = (output_root / case_name).resolve()
            tcl_path = Path(entry["tcl_compute" if compute else "tcl_timed"])
            lines.append(f"file mkdir {{{case_out}}}")
            lines.append(f"cd {{{case_out}}}")
            lines.append("wipe")
            lines.append("set __strut_case_t0 [clock microseconds]")
            lines.append(
                f"set __strut_case_err [catch {{source {{{tcl_path}}}}} __strut_case_msg]"
            )
            lines.append("set __strut_case_t1 [clock microseconds]")
            lines.append('set __strut_fp [open "case_time_us.txt" w]')
            lines.append(
                "puts $__strut_fp [expr {$__strut_case_t1 - $__strut_case_t0}]"
            )
            lines.append("close $__strut_fp")
            lines.append("if {$__strut_case_err != 0} {")
            lines.append('  set __strut_err_fp [open "case_error.txt" w]')
            lines.append("  puts $__strut_err_fp $__strut_case_msg")
            lines.append("  close $__strut_err_fp")
            lines.append("}")
            lines.append("cd $__strut_repo")
            lines.append("wipe")
        batch_path = (
            results_root
            / "tcl"
            / ("batch_opensees_compute.tcl" if compute else "batch_opensees_timed.tcl")
        )
        batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return batch_path

    def _write_strut_batch_manifest(
        entries: List[dict], output_root: Path, compute: bool
    ) -> Path:
        batch_entries = []
        for entry in entries:
            case_name = entry["name"]
            batch_entry = {
                "output": str(output_root / case_name),
            }
            if "input_pickle" in entry:
                batch_entry["input_pickle"] = str(entry["input_pickle"])
            else:
                batch_entry["input"] = str(Path(entry["json"]))
            if args.profile and not compute and profile_root is not None:
                batch_entry["profile"] = str(
                    profile_root / f"{case_name}.speedscope.json"
                )
            batch_entries.append(batch_entry)
        batch = {"cases": batch_entries}
        batch_path = (
            results_root
            / ".tmp"
            / ("strut_batch_compute.json" if compute else "strut_batch.json")
        )
        batch_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
        return batch_path

    def run_opensees_batch(
        entries: List[dict], output_root: Path, compute: bool
    ) -> None:
        ensure_clean_dir(output_root)
        batch_script = _write_batch_tcl(entries, output_root, compute)
        run(
            [
                str(repo_root / "scripts" / "run_opensees_wine.sh"),
                "--script",
                str(batch_script),
                "--output",
                str(output_root),
            ],
            env=env,
            verbose=verbose,
        )

    def run_opensees_batch_repeated(
        entries: List[dict], output_root: Path, compute: bool, repeat: int, warmup: int
    ):
        if not entries:
            return [], {}, {}

        pass_label = "compute-only" if compute else "total"
        n = len(entries)
        names = [entry["name"] for entry in entries]
        analysis_by_case = {name: [] for name in names}
        total_by_case = {name: [] for name in names}

        for i in range(warmup):
            log(f"OpenSees batch {pass_label} warmup {i + 1}/{warmup}...")
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            run_opensees_batch(rotated, output_root, compute)

        times = []
        for i in range(repeat):
            log(f"OpenSees batch {pass_label} repeat {i + 1}/{repeat}...")
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            start = time.perf_counter()
            run_opensees_batch(rotated, output_root, compute)
            end = time.perf_counter()
            times.append(end - start)

            analysis_map = _load_case_metric(
                output_root, entries, "analysis_time_us.txt"
            )
            total_map = _load_case_metric(output_root, entries, "case_time_us.txt")
            for name in names:
                analysis_us = analysis_map.get(name)
                total_us = total_map.get(name)
                if analysis_us is not None:
                    analysis_by_case[name].append(analysis_us)
                if total_us is not None:
                    total_by_case[name].append(total_us)

        return times, analysis_by_case, total_by_case

    def run_strut_batch(entries: List[dict], output_root: Path, compute: bool) -> bool:
        ensure_clean_dir(output_root)
        if strut_solver is None:
            raise SystemExit("Mojo solver not initialized.")
        batch_manifest = _write_strut_batch_manifest(entries, output_root, compute)
        try:
            cmd = [str(strut_solver), "--batch", str(batch_manifest)]
            if compute:
                cmd.append("--compute-only")
            run(
                cmd,
                env=env,
                verbose=verbose,
                capture_on_error=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"strut batch aborted (exit {exc.returncode})"
            for entry in entries:
                _write_case_error(output_root / entry["name"], msg)
            return False
        return True

    def run_strut_batch_repeated(
        entries: List[dict], output_root: Path, compute: bool, repeat: int, warmup: int
    ):
        if not entries:
            return [], {}, {}, False

        pass_label = "compute-only" if compute else "total"
        n = len(entries)
        names = [entry["name"] for entry in entries]
        analysis_by_case = {name: [] for name in names}
        total_by_case = {name: [] for name in names}
        had_abort = False

        def run_strut_case_fallback(entry: dict) -> None:
            case_name = entry["name"]
            target_dir = output_root / case_name
            ensure_clean_dir(target_dir)
            try:
                cmd = [str(strut_solver)]
                if "input_pickle" in entry:
                    cmd += ["--input-pickle", str(entry["input_pickle"])]
                else:
                    cmd += ["--input", str(Path(entry["json"]))]
                if compute:
                    cmd.append("--compute-only")
                cmd += ["--output", str(target_dir)]
                if args.profile and not compute and profile_root is not None:
                    cmd += [
                        "--profile",
                        str(profile_root / f"{case_name}.speedscope.json"),
                    ]
                run(
                    cmd,
                    env=env,
                    verbose=verbose,
                    capture_on_error=True,
                )
            except subprocess.CalledProcessError as exc:
                _write_case_error(
                    target_dir, f"strut case aborted (exit {exc.returncode})"
                )

        for i in range(warmup):
            log(f"Mojo batch {pass_label} warmup {i + 1}/{warmup}...")
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            ok = run_strut_batch(rotated, output_root, compute)
            if not ok:
                had_abort = True
                for entry in rotated:
                    run_strut_case_fallback(entry)

        times = []
        for i in range(repeat):
            log(f"Mojo batch {pass_label} repeat {i + 1}/{repeat}...")
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            start = time.perf_counter()
            ok = run_strut_batch(rotated, output_root, compute)
            if not ok:
                had_abort = True
                for entry in rotated:
                    run_strut_case_fallback(entry)
            end = time.perf_counter()
            times.append(end - start)

            analysis_map = _load_case_metric(
                output_root, entries, "analysis_time_us.txt"
            )
            total_map = _load_case_metric(output_root, entries, "case_time_us.txt")
            for name in names:
                analysis_us = analysis_map.get(name)
                total_us = total_map.get(name)
                if analysis_us is not None:
                    analysis_by_case[name].append(analysis_us)
                if total_us is not None:
                    total_by_case[name].append(total_us)

        return times, analysis_by_case, total_by_case, had_abort

    run_opensees_per_case = run_opensees and not batch_mode
    run_strut_per_case = run_strut and not batch_mode
    if run_opensees and batch_mode:
        log("Running OpenSees in batch mode.")
        opensees_times, batch_analysis_hist, batch_total_hist = (
            run_opensees_batch_repeated(
                case_entries,
                results_root / "opensees",
                compute=False,
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )
        _normalize_opensees_output_root(case_entries, results_root / "opensees")
        log_ok("OpenSees batch pass OK.")
        batch_stats = {
            "times_s": opensees_times,
            "mean_s": mean(opensees_times),
            "median_s": median(opensees_times),
            "min_s": min(opensees_times),
        }
        csv_rows.append(
            {
                "case": "opensees_batch",
                "dofs": "",
                "engine": "opensees",
                "mode": "total_batch",
                "repeat": args.repeat,
                "warmup": args.warmup,
                "mean_s": f"{batch_stats['mean_s']:.6f}",
                "median_s": f"{batch_stats['median_s']:.6f}",
                "min_s": f"{batch_stats['min_s']:.6f}",
                "analysis_us": "",
            }
        )
        for case_entry in case_entries:
            case_name = case_entry["name"]
            analysis_hist = batch_analysis_hist.get(case_name, [])
            total_hist = batch_total_hist.get(case_name, [])
            analysis_mean = int(mean(analysis_hist)) if analysis_hist else None
            total_mean = int(mean(total_hist)) if total_hist else None
            analysis_median = int(median(analysis_hist)) if analysis_hist else None
            total_median = int(median(total_hist)) if total_hist else None
            analysis_us = analysis_median
            total_us = total_median
            entry = case_entries_by_name.get(case_name)
            if entry is not None:
                batch_entry = entry.setdefault("opensees_batch", {})
                batch_entry["analysis_us"] = analysis_us
                batch_entry["total_us"] = total_us
                batch_entry["analysis_mean_us"] = analysis_mean
                batch_entry["total_mean_us"] = total_mean
                batch_entry["analysis_median_us"] = analysis_median
                batch_entry["total_median_us"] = total_median
                batch_entry["repeats"] = len(analysis_hist)
            csv_rows.append(
                {
                    "case": case_name,
                    "dofs": entry.get("dofs", ""),
                    "engine": "opensees",
                    "mode": "total_batch_case",
                    "repeat": "",
                    "warmup": "",
                    "mean_s": (
                        f"{(total_us or 0) / 1e6:.6f}" if total_us is not None else ""
                    ),
                    "median_s": (
                        f"{(total_median or 0) / 1e6:.6f}"
                        if total_median is not None
                        else ""
                    ),
                    "min_s": (
                        f"{(min(total_hist) if total_hist else 0) / 1e6:.6f}"
                        if total_hist
                        else ""
                    ),
                    "analysis_us": analysis_us,
                }
            )
        if not args.skip_compute_only:
            opensees_compute = run_engine(
                lambda out, last_run: run_opensees_batch(
                    case_entries, out, compute=True
                ),
                results_root / "opensees_compute",
                args.repeat,
                args.warmup,
            )
            log_ok("OpenSees batch compute-only pass OK.")
            compute_stats = {
                "times_s": opensees_compute,
                "mean_s": mean(opensees_compute),
                "median_s": median(opensees_compute),
                "min_s": min(opensees_compute),
            }
            csv_rows.append(
                {
                    "case": "opensees_batch",
                    "dofs": "",
                    "engine": "opensees",
                    "mode": "compute_only_batch",
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "mean_s": f"{compute_stats['mean_s']:.6f}",
                    "median_s": f"{compute_stats['median_s']:.6f}",
                    "min_s": f"{compute_stats['min_s']:.6f}",
                    "analysis_us": "",
                }
            )

    if run_strut and batch_mode:
        log("Running Mojo in batch mode.")
        strut_times, batch_analysis_hist, batch_total_hist, strut_batch_had_abort = (
            run_strut_batch_repeated(
                case_entries,
                results_root / "strut",
                compute=False,
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )
        if strut_batch_had_abort:
            log_err("Mojo batch aborted for one or more cases; used per-case fallback.")
        else:
            log_ok("Mojo batch pass OK.")
        batch_stats = {
            "times_s": strut_times,
            "mean_s": mean(strut_times),
            "median_s": median(strut_times),
            "min_s": min(strut_times),
        }
        csv_rows.append(
            {
                "case": "strut_batch",
                "dofs": "",
                "engine": "strut",
                "mode": "total_batch",
                "repeat": args.repeat,
                "warmup": args.warmup,
                "mean_s": f"{batch_stats['mean_s']:.6f}",
                "median_s": f"{batch_stats['median_s']:.6f}",
                "min_s": f"{batch_stats['min_s']:.6f}",
                "analysis_us": "",
            }
        )
        for case_entry in case_entries:
            case_name = case_entry["name"]
            analysis_hist = batch_analysis_hist.get(case_name, [])
            total_hist = batch_total_hist.get(case_name, [])
            analysis_mean = int(mean(analysis_hist)) if analysis_hist else None
            total_mean = int(mean(total_hist)) if total_hist else None
            analysis_median = int(median(analysis_hist)) if analysis_hist else None
            total_median = int(median(total_hist)) if total_hist else None
            analysis_us = analysis_median
            total_us = total_median
            total_s = [value / 1e6 for value in total_hist]
            stats = {
                "times_s": total_s,
                "mean_s": mean(total_s) if total_s else None,
                "median_s": median(total_s) if total_s else None,
                "min_s": min(total_s) if total_s else None,
                "analysis_us": analysis_us,
            }
            case_entry["strut"] = stats
            batch_entry = case_entry.setdefault("strut_batch", {})
            batch_entry["analysis_us"] = analysis_us
            batch_entry["total_us"] = total_us
            batch_entry["analysis_mean_us"] = analysis_mean
            batch_entry["total_mean_us"] = total_mean
            batch_entry["analysis_median_us"] = analysis_median
            batch_entry["total_median_us"] = total_median
            batch_entry["repeats"] = len(analysis_hist)
            csv_rows.append(
                {
                    "case": case_name,
                    "dofs": case_entry.get("dofs", ""),
                    "engine": "strut",
                    "mode": "total_batch_case",
                    "repeat": "",
                    "warmup": "",
                    "mean_s": (
                        f"{(total_us or 0) / 1e6:.6f}" if total_us is not None else ""
                    ),
                    "median_s": (
                        f"{(total_median or 0) / 1e6:.6f}"
                        if total_median is not None
                        else ""
                    ),
                    "min_s": (
                        f"{(min(total_hist) if total_hist else 0) / 1e6:.6f}"
                        if total_hist
                        else ""
                    ),
                    "analysis_us": analysis_us,
                }
            )
        if not args.skip_compute_only:
            strut_compute, _, _, strut_compute_had_abort = run_strut_batch_repeated(
                case_entries,
                results_root / "strut_compute",
                compute=True,
                repeat=args.repeat,
                warmup=args.warmup,
            )
            if strut_compute_had_abort:
                log_err(
                    "Mojo batch compute-only aborted for one or more cases; used per-case fallback."
                )
            else:
                log_ok("Mojo batch compute-only pass OK.")
            compute_stats = {
                "times_s": strut_compute,
                "mean_s": mean(strut_compute),
                "median_s": median(strut_compute),
                "min_s": min(strut_compute),
            }
            csv_rows.append(
                {
                    "case": "strut_batch",
                    "dofs": "",
                    "engine": "strut",
                    "mode": "compute_only_batch",
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "mean_s": f"{compute_stats['mean_s']:.6f}",
                    "median_s": f"{compute_stats['median_s']:.6f}",
                    "min_s": f"{compute_stats['min_s']:.6f}",
                    "analysis_us": "",
                }
            )

    for case_entry in case_entries:
        case_name = case_entry["name"]

        def opensees_cmd(output_dir: Path, last_run: bool) -> None:
            tcl_timed = Path(case_entry["tcl_timed"])
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "opensees" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            run(
                [
                    str(repo_root / "scripts" / "run_opensees_wine.sh"),
                    "--script",
                    str(tcl_timed),
                    "--output",
                    str(target_dir),
                ],
                env=env,
                verbose=verbose,
            )
            if not last_run:
                shutil.rmtree(target_dir, ignore_errors=True)

        def opensees_compute_cmd(output_dir: Path, last_run: bool) -> None:
            tcl_compute = Path(case_entry["tcl_compute"])
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "opensees_compute" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            run(
                [
                    str(repo_root / "scripts" / "run_opensees_wine.sh"),
                    "--script",
                    str(tcl_compute),
                    "--output",
                    str(target_dir),
                ],
                env=env,
                verbose=verbose,
            )
            if not last_run:
                shutil.rmtree(target_dir, ignore_errors=True)

        def strut_cmd(output_dir: Path, last_run: bool) -> None:
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "strut" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            if strut_solver is None:
                raise SystemExit("Mojo solver not initialized.")
            cmd = [str(strut_solver)]
            if "input_pickle" in case_entry:
                cmd += ["--input-pickle", str(case_entry["input_pickle"])]
            else:
                cmd += ["--input", str(case_entry["json"])]
            cmd += ["--output", str(target_dir)]
            if args.profile and last_run and profile_root is not None:
                profile_path = profile_root / f"{case_name}.speedscope.json"
                cmd += ["--profile", str(profile_path)]
            run(
                cmd,
                env=env,
                verbose=verbose,
                capture_on_error=True,
            )
            if not last_run:
                shutil.rmtree(target_dir, ignore_errors=True)

        def strut_compute_cmd(output_dir: Path, last_run: bool) -> None:
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "strut_compute" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            cmd = [str(strut_solver)]
            if "input_pickle" in case_entry:
                cmd += ["--input-pickle", str(case_entry["input_pickle"])]
            else:
                cmd += ["--input", str(case_entry["json"])]
            cmd += ["--compute-only", "--output", str(target_dir)]
            run(
                cmd,
                env=env,
                verbose=verbose,
                capture_on_error=True,
            )
            if not last_run:
                shutil.rmtree(target_dir, ignore_errors=True)

        if run_opensees_per_case:
            log(f"[{case_name}] OpenSees total pass...")
            opensees_times = run_engine(
                lambda out, last_run: opensees_cmd(out / case_name, last_run),
                results_root / "opensees",
                args.repeat,
                args.warmup,
            )
            _normalize_opensees_benchmark_outputs(
                case_entry["case_data"], results_root / "opensees" / case_name
            )
            log_ok(f"[{case_name}] OpenSees total pass OK.")
            stats = {
                "times_s": opensees_times,
                "mean_s": mean(opensees_times),
                "median_s": median(opensees_times),
                "min_s": min(opensees_times),
            }
            case_entry["opensees"] = stats
            analysis_file = (
                results_root / "opensees" / case_name / "analysis_time_us.txt"
            )
            if analysis_file.exists():
                try:
                    case_entry["opensees"]["analysis_us"] = int(
                        analysis_file.read_text().strip()
                    )
                except ValueError:
                    case_entry["opensees"]["analysis_us"] = None
            else:
                case_entry["opensees"]["analysis_us"] = None
            csv_rows.append(
                {
                    "case": case_name,
                    "dofs": case_entry.get("dofs", ""),
                    "engine": "opensees",
                    "mode": "total",
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "mean_s": f"{stats['mean_s']:.6f}",
                    "median_s": f"{stats['median_s']:.6f}",
                    "min_s": f"{stats['min_s']:.6f}",
                    "analysis_us": case_entry["opensees"]["analysis_us"],
                }
            )
            if not args.skip_compute_only:
                log(f"[{case_name}] OpenSees compute-only pass...")
                opensees_compute = run_engine(
                    lambda out, last_run: opensees_compute_cmd(
                        out / case_name, last_run
                    ),
                    results_root / "opensees_compute",
                    args.repeat,
                    args.warmup,
                )
                log_ok(f"[{case_name}] OpenSees compute-only pass OK.")
                compute_stats = {
                    "times_s": opensees_compute,
                    "mean_s": mean(opensees_compute),
                    "median_s": median(opensees_compute),
                    "min_s": min(opensees_compute),
                }
                case_entry["opensees_compute_only"] = compute_stats
                csv_rows.append(
                    {
                        "case": case_name,
                        "dofs": case_entry.get("dofs", ""),
                        "engine": "opensees",
                        "mode": "compute_only",
                        "repeat": args.repeat,
                        "warmup": args.warmup,
                        "mean_s": f"{compute_stats['mean_s']:.6f}",
                        "median_s": f"{compute_stats['median_s']:.6f}",
                        "min_s": f"{compute_stats['min_s']:.6f}",
                        "analysis_us": "",
                    }
                )

        if run_strut_per_case:
            log(f"[{case_name}] Mojo total pass...")
            try:
                strut_times = run_engine(
                    lambda out, last_run: strut_cmd(out / case_name, last_run),
                    results_root / "strut",
                    args.repeat,
                    args.warmup,
                )
            except subprocess.CalledProcessError as exc:
                _write_case_error(
                    results_root / "strut" / case_name,
                    f"strut total pass aborted (exit {exc.returncode})",
                )
                log_err(
                    f"[{case_name}] Mojo total pass aborted (exit {exc.returncode})."
                )
            else:
                log_ok(f"[{case_name}] Mojo total pass OK.")
                stats = {
                    "times_s": strut_times,
                    "mean_s": mean(strut_times),
                    "median_s": median(strut_times),
                    "min_s": min(strut_times),
                }
                analysis_file = (
                    results_root / "strut" / case_name / "analysis_time_us.txt"
                )
                stats["analysis_us"] = _read_analysis_us(analysis_file)
                case_entry["strut"] = stats
                csv_rows.append(
                    {
                        "case": case_name,
                        "dofs": case_entry.get("dofs", ""),
                        "engine": "strut",
                        "mode": "total",
                        "repeat": args.repeat,
                        "warmup": args.warmup,
                        "mean_s": f"{stats['mean_s']:.6f}",
                        "median_s": f"{stats['median_s']:.6f}",
                        "min_s": f"{stats['min_s']:.6f}",
                        "analysis_us": stats["analysis_us"],
                    }
                )
            if not args.skip_compute_only:
                log(f"[{case_name}] Mojo compute-only pass...")
                try:
                    strut_compute = run_engine(
                        lambda out, last_run: strut_compute_cmd(
                            out / case_name, last_run
                        ),
                        results_root / "strut_compute",
                        args.repeat,
                        args.warmup,
                    )
                except subprocess.CalledProcessError as exc:
                    _write_case_error(
                        results_root / "strut_compute" / case_name,
                        f"strut compute-only pass aborted (exit {exc.returncode})",
                    )
                    log_err(
                        f"[{case_name}] Mojo compute-only pass aborted (exit {exc.returncode})."
                    )
                else:
                    log_ok(f"[{case_name}] Mojo compute-only pass OK.")
                    compute_stats = {
                        "times_s": strut_compute,
                        "mean_s": mean(strut_compute),
                        "median_s": median(strut_compute),
                        "min_s": min(strut_compute),
                    }
                    case_entry["strut_compute_only"] = compute_stats
                    compute_analysis_file = (
                        results_root
                        / "strut_compute"
                        / case_name
                        / "analysis_time_us.txt"
                    )
                    compute_analysis_us = _read_analysis_us(compute_analysis_file)
                    csv_rows.append(
                        {
                            "case": case_name,
                            "dofs": case_entry.get("dofs", ""),
                            "engine": "strut",
                            "mode": "compute_only",
                            "repeat": args.repeat,
                            "warmup": args.warmup,
                            "mean_s": f"{compute_stats['mean_s']:.6f}",
                            "median_s": f"{compute_stats['median_s']:.6f}",
                            "min_s": f"{compute_stats['min_s']:.6f}",
                            "analysis_us": compute_analysis_us,
                        }
                    )

        if run_opensees and run_strut:
            case_data = case_entry["case_data"]
            case_parity_failures: List[str] = []
            recorders = case_data.get("recorders", [])
            analysis = case_data.get("analysis", {})
            analysis_type = str(analysis.get("type", "static_linear"))
            is_transient = _analysis_is_transient(analysis)
            use_last_common_row = analysis_type == "static_nonlinear"
            tol = case_data.get("parity_tolerance", {})
            global_rtol = tol.get("rtol", REL_TOL)
            global_atol = tol.get("atol", ABS_TOL)
            has_global_override = isinstance(tol, dict) and (
                "rtol" in tol or "atol" in tol
            )
            tol_by_recorder = case_data.get("parity_tolerance_by_recorder", {})
            if not isinstance(tol_by_recorder, dict):
                tol_by_recorder = {}
            parity_mode = str(case_data.get("parity_mode", "step")).strip().lower()
            if parity_mode not in ("step", "max_abs"):
                raise SystemExit(
                    f"unsupported parity_mode: {parity_mode} (expected step|max_abs)"
                )
            for rec in recorders:
                if rec.get("parity", True) is False:
                    continue
                rec_type = rec.get("type")
                rec_rtol, rec_atol = _resolve_recorder_tolerance(
                    rec_type,
                    global_rtol,
                    global_atol,
                    has_global_override,
                    tol_by_recorder,
                )
                if rec_type == "node_displacement":
                    output = rec.get("output", "node_disp")
                    for node_id in rec.get("nodes", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        strut_file = (
                            results_root
                            / "strut"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        if not ref_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not strut_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing Mojo output: {strut_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                strut_vals = _load_all_values(strut_file)
                            else:
                                ref_vals, strut_vals = _load_last_comparable_values(
                                    ref_file,
                                    strut_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            case_parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            _compare_transient_rows(
                                ref_vals,
                                strut_vals,
                                f"{case_name}: node {node_id}",
                                case_parity_failures,
                                rec_rtol,
                                rec_atol,
                                parity_mode,
                            )
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: node {node_id} mismatch"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
                elif rec_type == "element_force":
                    output = rec.get("output", "element_force")
                    for elem_id in rec.get("elements", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        strut_file = (
                            results_root
                            / "strut"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        if not ref_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not strut_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing Mojo output: {strut_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                strut_vals = _load_all_values(strut_file)
                            else:
                                ref_vals, strut_vals = _load_last_comparable_values(
                                    ref_file,
                                    strut_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            case_parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            _compare_transient_rows(
                                ref_vals,
                                strut_vals,
                                f"{case_name}: element {elem_id}",
                                case_parity_failures,
                                rec_rtol,
                                rec_atol,
                                parity_mode,
                            )
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: element {elem_id} mismatch"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
                elif rec_type == "node_reaction":
                    output = rec.get("output", "reaction")
                    for node_id in rec.get("nodes", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        strut_file = (
                            results_root
                            / "strut"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        if not ref_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not strut_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing Mojo output: {strut_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                strut_vals = _load_all_values(strut_file)
                            else:
                                ref_vals, strut_vals = _load_last_comparable_values(
                                    ref_file,
                                    strut_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            case_parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            _compare_transient_rows(
                                ref_vals,
                                strut_vals,
                                f"{case_name}: reaction node {node_id}",
                                case_parity_failures,
                                rec_rtol,
                                rec_atol,
                                parity_mode,
                            )
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: reaction node {node_id} mismatch"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
                elif rec_type == "drift":
                    output = rec.get("output", "drift")
                    i_node = int(rec["i_node"])
                    j_node = int(rec["j_node"])
                    ref_file = (
                        results_root
                        / "opensees"
                        / case_name
                        / f"{output}_i{i_node}_j{j_node}.out"
                    )
                    strut_file = (
                        results_root
                        / "strut"
                        / case_name
                        / f"{output}_i{i_node}_j{j_node}.out"
                    )
                    if not ref_file.exists():
                        case_parity_failures.append(
                            f"{case_name}: missing OpenSees output: {ref_file}"
                        )
                        continue
                    if not strut_file.exists():
                        case_parity_failures.append(
                            f"{case_name}: missing Mojo output: {strut_file}"
                        )
                        continue
                    try:
                        if is_transient:
                            ref_vals = _load_all_values(ref_file)
                            strut_vals = _load_all_values(strut_file)
                        else:
                            ref_vals, strut_vals = _load_last_comparable_values(
                                ref_file,
                                strut_file,
                                use_last_common_row=use_last_common_row,
                            )
                    except ValueError as exc:
                        case_parity_failures.append(f"{case_name}: {exc}")
                        continue
                    if is_transient:
                        _compare_transient_rows(
                            ref_vals,
                            strut_vals,
                            f"{case_name}: drift i{i_node}-j{j_node}",
                            case_parity_failures,
                            rec_rtol,
                            rec_atol,
                            parity_mode,
                        )
                    else:
                        ok, errors = _compare_vectors(
                            ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                        )
                        if not ok:
                            case_parity_failures.append(
                                f"{case_name}: drift i{i_node}-j{j_node} mismatch"
                            )
                            case_parity_failures.extend(
                                [f"  {err}" for err in errors]
                            )
                elif rec_type in ELEMENT_RESPONSE_RECORDER_TYPES:
                    output = rec.get("output", rec_type)
                    for elem_id in rec.get("elements", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        strut_file = (
                            results_root
                            / "strut"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        if not ref_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not strut_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing Mojo output: {strut_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                strut_vals = _load_all_values(strut_file)
                            else:
                                ref_vals, strut_vals = _load_last_comparable_values(
                                    ref_file,
                                    strut_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            case_parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            _compare_transient_rows(
                                ref_vals,
                                strut_vals,
                                f"{case_name}: {rec_type} element {elem_id}",
                                case_parity_failures,
                                rec_rtol,
                                rec_atol,
                                parity_mode,
                            )
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: {rec_type} element {elem_id} mismatch"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
                elif rec_type == "envelope_element_force":
                    output = rec.get("output", "envelope_element_force")
                    for elem_id in rec.get("elements", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        strut_file = (
                            results_root
                            / "strut"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        if not ref_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not strut_file.exists():
                            case_parity_failures.append(
                                f"{case_name}: missing Mojo output: {strut_file}"
                            )
                            continue
                        try:
                            ref_vals = _load_all_values(ref_file)
                            strut_vals = _load_all_values(strut_file)
                        except ValueError as exc:
                            case_parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if len(ref_vals) != len(strut_vals):
                            case_parity_failures.append(
                                f"{case_name}: envelope element {elem_id} row count mismatch: {len(ref_vals)} != {len(strut_vals)}"
                            )
                            continue
                        for row, (rvec, gvec) in enumerate(
                            zip(ref_vals, strut_vals), start=1
                        ):
                            ok, errors = _compare_vectors(
                                rvec, gvec, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: envelope element {elem_id} mismatch at row {row}"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
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
                            case_parity_failures.append(
                                f"{case_name}: {rec_type} recorder missing section/sections"
                            )
                            continue
                        sections = [rec["section"]]
                    for elem_id in rec.get("elements", []):
                        for sec_no in sections:
                            ref_file = (
                                results_root
                                / "opensees"
                                / case_name
                                / f"{output}_ele{elem_id}_sec{sec_no}.out"
                            )
                            strut_file = (
                                results_root
                                / "strut"
                                / case_name
                                / f"{output}_ele{elem_id}_sec{sec_no}.out"
                            )
                            if not ref_file.exists():
                                case_parity_failures.append(
                                    f"{case_name}: missing OpenSees output: {ref_file}"
                                )
                                continue
                            if not strut_file.exists():
                                case_parity_failures.append(
                                    f"{case_name}: missing Mojo output: {strut_file}"
                                )
                                continue
                            try:
                                if is_transient:
                                    ref_vals = _load_all_values(ref_file)
                                    strut_vals = _load_all_values(strut_file)
                                else:
                                    ref_vals, strut_vals = _load_last_comparable_values(
                                        ref_file,
                                        strut_file,
                                        use_last_common_row=use_last_common_row,
                                    )
                            except ValueError as exc:
                                case_parity_failures.append(f"{case_name}: {exc}")
                                continue
                            if is_transient:
                                _compare_transient_rows(
                                    ref_vals,
                                    strut_vals,
                                    f"{case_name}: {rec_type} element {elem_id} section {sec_no}",
                                    case_parity_failures,
                                    rec_rtol,
                                    rec_atol,
                                    parity_mode,
                                )
                            else:
                                ok, errors = _compare_vectors(
                                    ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                                )
                                if not ok:
                                    case_parity_failures.append(
                                        f"{case_name}: {rec_type} element {elem_id} section {sec_no} mismatch"
                                    )
                                    case_parity_failures.extend(
                                        [f"  {err}" for err in errors]
                                    )
                elif rec_type == "modal_eigen":
                    output = rec.get("output", "modal")
                    eig_ref = (
                        results_root
                        / "opensees"
                        / case_name
                        / f"{output}_eigenvalues.out"
                    )
                    eig_strut = (
                        results_root / "strut" / case_name / f"{output}_eigenvalues.out"
                    )
                    if not eig_ref.exists():
                        case_parity_failures.append(
                            f"{case_name}: missing OpenSees output: {eig_ref}"
                        )
                        continue
                    if not eig_strut.exists():
                        case_parity_failures.append(
                            f"{case_name}: missing Mojo output: {eig_strut}"
                        )
                        continue
                    try:
                        ref_rows = _load_all_values(eig_ref)
                        strut_rows = _load_all_values(eig_strut)
                    except ValueError as exc:
                        case_parity_failures.append(f"{case_name}: {exc}")
                        continue
                    ref_vals = [row[0] for row in ref_rows]
                    strut_vals = [row[0] for row in strut_rows]
                    ok, errors = _compare_vectors(
                        ref_vals, strut_vals, rtol=rec_rtol, atol=rec_atol
                    )
                    if not ok:
                        case_parity_failures.append(
                            f"{case_name}: modal eigenvalue mismatch"
                        )
                        case_parity_failures.extend([f"  {err}" for err in errors])

                    for mode_no in rec.get("modes", []):
                        for node_id in rec.get("nodes", []):
                            ref_file = (
                                results_root
                                / "opensees"
                                / case_name
                                / f"{output}_mode{int(mode_no)}_node{int(node_id)}.out"
                            )
                            strut_file = (
                                results_root
                                / "strut"
                                / case_name
                                / f"{output}_mode{int(mode_no)}_node{int(node_id)}.out"
                            )
                            if not ref_file.exists():
                                case_parity_failures.append(
                                    f"{case_name}: missing OpenSees output: {ref_file}"
                                )
                                continue
                            if not strut_file.exists():
                                case_parity_failures.append(
                                    f"{case_name}: missing Mojo output: {strut_file}"
                                )
                                continue
                            try:
                                ref_vec = _load_last_values(ref_file)
                                strut_vec = _load_last_values(strut_file)
                            except ValueError as exc:
                                case_parity_failures.append(f"{case_name}: {exc}")
                                continue
                            ok, errors = _compare_mode_shape_vectors(
                                ref_vec, strut_vec, rtol=rec_rtol, atol=rec_atol
                            )
                            if not ok:
                                case_parity_failures.append(
                                    f"{case_name}: modal mode shape mismatch mode={mode_no} node={node_id}"
                                )
                                case_parity_failures.extend(
                                    [f"  {err}" for err in errors]
                                )
                else:
                    case_parity_failures.append(
                        f"{case_name}: unsupported recorder type: {rec_type}"
                    )

            if case_parity_failures:
                if case_entry.get("status") == "benchmark":
                    benchmark_parity_failures.extend(case_parity_failures)
                else:
                    parity_failures.extend(case_parity_failures)

        summary_cases.append(case_entry)

    phase_rows: List[dict] = []
    if run_strut:
        for case_entry in summary_cases:
            case_name = case_entry["name"]
            phase_path = results_root / "strut" / case_name / "phase_times_us.json"
            phase_times = _load_phase_times(phase_path)
            frame_totals: Dict[str, int] = {}
            if profile_root is not None:
                profile_path = profile_root / f"{case_name}.speedscope.json"
                frame_totals = _load_profile_frame_totals(profile_path)
            if not phase_times and not frame_totals:
                continue
            phase_rows.append(
                _build_phase_row(
                    case_name=case_name,
                    dofs=case_entry.get("dofs"),
                    phase_times=phase_times,
                    frame_totals=frame_totals,
                )
            )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "generated_at": generated_at,
        "git_rev": run_metadata.get("git", {}).get("rev"),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "metadata": run_metadata,
        "phase_columns": PHASE_COLUMNS,
        "phase_summary": phase_rows,
        "cases": summary_cases,
    }

    summary_json = results_root / "summary.json"
    summary_csv = results_root / "summary.csv"
    metadata_json = results_root / "metadata.json"
    phase_summary_csv = results_root / "phase_summary.csv"
    phase_rollup_csv = results_root / "phase_rollup.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, csv_rows)
    metadata_json.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    write_phase_summary_csv(phase_summary_csv, phase_rows)
    write_phase_rollup_csv(phase_rollup_csv, phase_rows)
    log_phase_table(phase_rows)

    if not args.no_archive:
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(summary_json, archive_root / f"{stamp}-summary.json")
        shutil.copy2(summary_csv, archive_root / f"{stamp}-summary.csv")
        shutil.copy2(metadata_json, archive_root / f"{stamp}-metadata.json")
        shutil.copy2(phase_summary_csv, archive_root / f"{stamp}-phase-summary.csv")
        shutil.copy2(phase_rollup_csv, archive_root / f"{stamp}-phase-rollup.csv")

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {phase_summary_csv}")
    print(f"Wrote {phase_rollup_csv}")
    if benchmark_parity_failures:
        log("PARITY WARN (benchmark-only cases)")
        for failure in _summarize_parity_failures(benchmark_parity_failures):
            log(failure)
    if parity_failures:
        log_err("PARITY FAILED")
        for failure in _summarize_parity_failures(parity_failures):
            log_err(failure)
        raise SystemExit(1)
    log_ok("PARITY OK")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log_err(f"BENCHMARK ERROR: {exc}")
        raise SystemExit(1)

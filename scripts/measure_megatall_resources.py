#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CASE_DIR = (
    REPO_ROOT / "tests" / "validation" / "opensees_megatall_building_model1_dynamiccpu"
)
SMOKE_CASE = CASE_DIR / "megatall_smoke.json"
FULL_CASE = CASE_DIR / "generated" / "case.json"
RUN_STRUT_CASE = REPO_ROOT / "scripts" / "run_strut_case.py"
RUN_PROGRESS = "run_progress.json"
DEFAULT_TIMEOUT_SECONDS = 600.0

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_case


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_dict(values: Counter[str]) -> dict[str, int]:
    return dict(sorted(values.items()))


def _stage_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = data.get("analysis", {})
    if not isinstance(analysis, dict):
        return []
    if analysis.get("type") != "staged":
        return [analysis]
    stages = analysis.get("stages", [])
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, dict)]


def _stage_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    nested = stage.get("analysis")
    if isinstance(nested, dict):
        return nested
    return stage


def _node_constraint_dofs(data: dict[str, Any]) -> int:
    total = 0
    for node in data.get("nodes", []):
        if not isinstance(node, dict):
            continue
        constraints = node.get("constraints")
        if isinstance(constraints, list):
            total += len(constraints)
    return total


def _mpc_slave_dofs(data: dict[str, Any]) -> int:
    total = 0
    for mpc in data.get("mp_constraints", []):
        if not isinstance(mpc, dict):
            continue
        constrained_dofs = mpc.get("constrained_dofs")
        if isinstance(constrained_dofs, list):
            total += len(constrained_dofs)
            continue
        dofs = mpc.get("dofs")
        if isinstance(dofs, list):
            total += len(dofs)
    return total


def _recorder_output_targets(recorders: list[dict[str, Any]]) -> int:
    total = 0
    for recorder in recorders:
        try:
            total += len(run_case._normalized_recorder_outputs(recorder))
        except SystemExit:
            total += 1
    return total


def _summarize_recorders(data: dict[str, Any]) -> dict[str, Any]:
    recorders = [
        recorder
        for recorder in data.get("recorders", [])
        if isinstance(recorder, dict)
    ]
    type_counts = Counter(str(recorder.get("type", "unknown")) for recorder in recorders)
    raw_paths = Counter(
        str(recorder.get("raw_path"))
        for recorder in recorders
        if isinstance(recorder.get("raw_path"), str)
    )
    node_targets = 0
    element_targets = 0
    section_targets = 0
    drift_targets = 0
    for recorder in recorders:
        node_targets += len(recorder.get("nodes", []) or [])
        element_targets += len(recorder.get("elements", []) or [])
        section_targets += len(recorder.get("sections", []) or [])
        if recorder.get("section") is not None:
            section_targets += 1
        if recorder.get("type") == "drift":
            drift_targets += 1
    return {
        "count": len(recorders),
        "types": _counter_dict(type_counts),
        "output_targets": _recorder_output_targets(recorders),
        "raw_output_groups": len(raw_paths),
        "node_targets": node_targets,
        "element_targets": element_targets,
        "section_targets": section_targets,
        "drift_targets": drift_targets,
    }


def _summarize_shells(data: dict[str, Any]) -> dict[str, Any]:
    sections = [
        section for section in data.get("sections", []) if isinstance(section, dict)
    ]
    elements = [
        element for element in data.get("elements", []) if isinstance(element, dict)
    ]
    layered_shell_layers: dict[int, int] = {}
    layered_shell_section_count = 0
    for section in sections:
        if section.get("type") != "LayeredShellSection":
            continue
        layered_shell_section_count += 1
        params = section.get("params", {})
        layers = params.get("layers", []) if isinstance(params, dict) else []
        layered_shell_layers[int(section["id"])] = len(layers) if isinstance(layers, list) else 0
    shell_elements = [
        element for element in elements if str(element.get("type")) == "shell"
    ]
    shell_layer_instances = 0
    for element in shell_elements:
        section_id = element.get("section")
        if isinstance(section_id, int):
            shell_layer_instances += layered_shell_layers.get(section_id, 0)
    return {
        "elements": len(shell_elements),
        "layered_sections": layered_shell_section_count,
        "section_layers": sum(layered_shell_layers.values()),
        "element_layer_instances": shell_layer_instances,
    }


def _summarize_stages(data: dict[str, Any]) -> dict[str, Any]:
    stages = _stage_items(data)
    analysis = data.get("analysis", {})
    stage_type_counts: Counter[str] = Counter()
    stage_constraint_handlers: Counter[str] = Counter()
    total_steps = 0
    static_steps = 0
    transient_steps = 0
    if isinstance(analysis, dict) and analysis.get("type") != "staged":
        stages = [analysis]
    for stage in stages:
        current = _stage_analysis(stage)
        stage_type = str(current.get("type", "unknown"))
        stage_type_counts[stage_type] += 1
        constraints = current.get("constraints")
        if isinstance(constraints, str):
            stage_constraint_handlers[constraints] += 1
        steps = int(current.get("steps", 0) or 0)
        total_steps += steps
        if "transient" in stage_type:
            transient_steps += steps
        elif "static" in stage_type:
            static_steps += steps
    return {
        "root_type": analysis.get("type"),
        "stage_count": len(stages),
        "types": _counter_dict(stage_type_counts),
        "constraint_handlers": _counter_dict(stage_constraint_handlers),
        "total_steps": total_steps,
        "static_steps": static_steps,
        "transient_steps": transient_steps,
    }


def summarize_case_input(data: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model", {})
    ndm = int(model.get("ndm", 0) or 0)
    ndf = int(model.get("ndf", 0) or 0)
    nodes = [node for node in data.get("nodes", []) if isinstance(node, dict)]
    elements = [
        element for element in data.get("elements", []) if isinstance(element, dict)
    ]
    element_types = Counter(str(element.get("type", "unknown")) for element in elements)
    materials = [
        material for material in data.get("materials", []) if isinstance(material, dict)
    ]
    material_types = Counter(str(material.get("type", "unknown")) for material in materials)
    sections = [
        section for section in data.get("sections", []) if isinstance(section, dict)
    ]
    section_types = Counter(str(section.get("type", "unknown")) for section in sections)
    mp_constraints = [
        mpc for mpc in data.get("mp_constraints", []) if isinstance(mpc, dict)
    ]
    mp_types = Counter(str(mpc.get("type", "unknown")) for mpc in mp_constraints)
    shells = _summarize_shells(data)
    stages = _summarize_stages(data)
    recorder_summary = _summarize_recorders(data)
    total_dofs = len(nodes) * ndf
    node_fixed_dofs = _node_constraint_dofs(data)
    mpc_slave_dofs = _mpc_slave_dofs(data)
    return {
        "model": {"ndm": ndm, "ndf": ndf},
        "nodes": len(nodes),
        "elements": len(elements),
        "element_types": _counter_dict(element_types),
        "materials": len(materials),
        "material_types": _counter_dict(material_types),
        "sections": len(sections),
        "section_types": _counter_dict(section_types),
        "time_series": len(data.get("time_series", []) or []),
        "masses": len(data.get("masses", []) or []),
        "loads": len(data.get("loads", []) or []),
        "total_dofs_upper_bound": total_dofs,
        "node_fixed_dofs": node_fixed_dofs,
        "mp_constraints": len(mp_constraints),
        "mp_constraint_types": _counter_dict(mp_types),
        "mpc_slave_dofs": mpc_slave_dofs,
        "recorders": recorder_summary,
        "shells": shells,
        "stages": stages,
        "cost_centers": {
            "recorder_output_fanout": {
                "recorders": recorder_summary["count"],
                "output_targets": recorder_summary["output_targets"],
                "raw_output_groups": recorder_summary["raw_output_groups"],
            },
            "shell_state": shells,
            "transient_workspace": {
                "total_dofs_upper_bound": total_dofs,
                "node_fixed_dofs": node_fixed_dofs,
                "staged_steps": stages["total_steps"],
                "transient_steps": stages["transient_steps"],
            },
            "mpc_overhead": {
                "constraints": len(mp_constraints),
                "slave_dofs": mpc_slave_dofs,
                "types": _counter_dict(mp_types),
            },
        },
    }


def _dir_stats(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name == RUN_PROGRESS:
            continue
        file_count += 1
        total_bytes += file_path.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _extract_error_text(stdout_tail: str, stderr_tail: str) -> str | None:
    for text in (stderr_tail, stdout_tail):
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return None


def _summarize_failure(
    run_summary: dict[str, Any], output_stats: dict[str, Any]
) -> dict[str, Any] | None:
    if run_summary["returncode"] == 0 and not run_summary["timed_out"]:
        return None

    progress = run_summary.get("progress")
    status = progress.get("status") if isinstance(progress, dict) else None
    analysis_type = progress.get("analysis_type") if isinstance(progress, dict) else None
    stage_number = progress.get("stage_number") if isinstance(progress, dict) else None
    stage_count = progress.get("stage_count") if isinstance(progress, dict) else None
    step_number = progress.get("step_number") if isinstance(progress, dict) else None
    step_count = progress.get("step_count") if isinstance(progress, dict) else None
    error_text = _extract_error_text(
        run_summary.get("stdout_tail", ""),
        run_summary.get("stderr_tail", ""),
    )
    failure_text = " ".join(
        part
        for part in (run_summary.get("stderr_tail", ""), run_summary.get("stdout_tail", ""))
        if part
    ).lower()

    classification = "unknown"
    likely_root_cause = "unknown"
    if "did not converge" in failure_text:
        classification = "convergence"
        likely_root_cause = "convergence"
    elif any(
        marker in failure_text
        for marker in (
            "sigsegv",
            "segmentation fault",
            "died with <signals.sigsegv",
        )
    ):
        classification = "crash"
        likely_root_cause = (
            "input_parse_or_load"
            if status in (None, "", "loading")
            else "native_runtime"
        )
    elif status == "writing_output":
        classification = "timeout" if run_summary["timed_out"] else "recorder_output"
        likely_root_cause = "recorder_output"
    elif "recorder" in failure_text or "output" in failure_text:
        classification = "recorder_output"
        likely_root_cause = "recorder_output"
    elif any(
        marker in failure_text
        for marker in ("out of memory", "bad_alloc", "cannot allocate", "memory")
    ):
        classification = "memory"
        likely_root_cause = "memory"
    elif run_summary["timed_out"]:
        classification = "timeout"
        likely_root_cause = "memory"

    return {
        "classification": classification,
        "likely_root_cause": likely_root_cause,
        "status": status,
        "analysis_type": analysis_type,
        "stage_number": stage_number,
        "stage_count": stage_count,
        "step_number": step_number,
        "step_count": step_count,
        "error_text": error_text,
        "output_file_count": output_stats["file_count"],
    }


def _peak_rss_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, ValueError):
        return None
    max_rss = getattr(usage, "ru_maxrss", 0)
    if not max_rss:
        return None
    # Linux reports kilobytes, macOS reports bytes.
    if sys.platform == "darwin":
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def _sample_process_group_rss_mb(pgid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "--no-headers", "-g", str(pgid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    total_kb = 0
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total_kb += int(line)
        except ValueError:
            return None
    if total_kb <= 0:
        return None
    return total_kb / 1024.0


_PROGRESS_STATUS_RANK = {
    "loading": 1,
    "running_stage": 2,
    "running_step": 3,
    "writing_output": 4,
    "completed": 5,
}


def _progress_rank(progress: dict[str, Any] | None) -> tuple[int, int, int]:
    if not isinstance(progress, dict):
        return (0, 0, 0)
    status = str(progress.get("status", ""))
    return (
        _PROGRESS_STATUS_RANK.get(status, 0),
        int(progress.get("stage_number", 0) or 0),
        int(progress.get("step_number", 0) or 0),
    )


def _prefer_progress(
    current: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None or _progress_rank(candidate) >= _progress_rank(current):
        return candidate
    return current


def _run_case(
    input_path: Path,
    output_dir: Path,
    compute_only: bool,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUN_STRUT_CASE),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
    ]
    if compute_only:
        command.append("--compute-only")

    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    sampled_peak_rss_mb = 0.0
    observed_progress: dict[str, Any] | None = None
    progress_path = output_dir / RUN_PROGRESS
    timed_out = False
    while True:
        if proc.pid > 0:
            sampled = _sample_process_group_rss_mb(proc.pid)
            if sampled is not None and sampled > sampled_peak_rss_mb:
                sampled_peak_rss_mb = sampled
        observed_progress = _prefer_progress(
            observed_progress, _read_optional_json(progress_path)
        )
        if proc.poll() is not None:
            break
        if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(0.2)
    completed = proc.communicate()
    finished = time.perf_counter()

    phase_times = _read_optional_json(output_dir / "phase_times_us.json")
    analysis_time_us = _read_optional_text(output_dir / "analysis_time_us.txt")
    final_progress = _read_optional_json(progress_path)
    progress = _prefer_progress(observed_progress, final_progress)
    peak_rss_mb = _peak_rss_mb()
    if sampled_peak_rss_mb > 0.0:
        if peak_rss_mb is None:
            peak_rss_mb = sampled_peak_rss_mb
        else:
            peak_rss_mb = max(peak_rss_mb, sampled_peak_rss_mb)
    run_summary: dict[str, Any] = {
        "command": command,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": round(finished - started, 3),
        "peak_rss_mb": peak_rss_mb,
        "analysis_time_us": int(analysis_time_us) if analysis_time_us else None,
        "phase_times_us": phase_times,
        "progress": progress,
        "final_progress": final_progress,
        "stdout_tail": completed[0][-4000:] if completed[0] else "",
        "stderr_tail": completed[1][-4000:] if completed[1] else "",
    }
    return run_summary


def _default_input(case_name: str) -> Path:
    if case_name == "smoke":
        return SMOKE_CASE
    if case_name == "full":
        return FULL_CASE
    raise ValueError(f"unsupported case selection: {case_name}")


def build_report(
    input_path: Path,
    *,
    case_label: str,
    compute_only: bool,
    skip_run: bool,
    output_dir: Path | None,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    case_data = _load_json(input_path)
    report: dict[str, Any] = {
        "case_label": case_label,
        "input_path": str(input_path),
        "compute_only": compute_only,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_summary": summarize_case_input(case_data),
    }
    if skip_run:
        report["run"] = None
        report["output"] = None
        report["failure"] = None
        return report

    if output_dir is None:
        raise ValueError("output_dir is required when skip_run is false")

    run_summary = _run_case(input_path, output_dir, compute_only, timeout_seconds)
    output_stats = _dir_stats(output_dir)
    report["run"] = run_summary
    report["output"] = {
        "path": str(output_dir),
        **output_stats,
    }
    report["failure"] = _summarize_failure(run_summary, output_stats)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure megatall native resource usage and summarize cost centers."
    )
    parser.add_argument(
        "--case",
        choices=("smoke", "full"),
        default="smoke",
        help="Which tracked megatall input to use when --input is not provided.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Explicit solver JSON input path. Overrides --case.",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Run the native solver with recorders disabled.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only summarize the case input and skip the native runtime invocation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for native solver outputs. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Optional path to write the JSON report.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep a temporary output directory instead of deleting it.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock limit for the native run. Defaults to 600 seconds; timed-out runs still emit a partial report.",
    )
    parser.add_argument(
        "--label",
        help="Optional label to override the report case label.",
    )
    args = parser.parse_args()

    input_path = (args.input or _default_input(args.case)).resolve()
    case_label = args.label or args.case

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    output_dir: Path | None = None
    if not args.skip_run:
        if args.output_dir is not None:
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="strut_megatall_phase2_")
            output_dir = Path(temp_dir.name)

    try:
        report = build_report(
            input_path,
            case_label=case_label,
            compute_only=args.compute_only,
            skip_run=args.skip_run,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        )
        report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output_report is not None:
            args.output_report.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output_report.write_text(report_text, encoding="utf-8")
        sys.stdout.write(report_text)
        if temp_dir is not None and args.keep_output and output_dir is not None:
            kept_dir = CASE_DIR / "generated" / f"{case_label}_phase2_output"
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            shutil.copytree(output_dir, kept_dir)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()

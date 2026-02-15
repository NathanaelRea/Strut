#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

ABS_TOL = 1e-8
REL_TOL = 1e-5

@dataclass(frozen=True)
class CaseSpec:
    name: str
    json_path: Path


def run(cmd: List[str], env=None, verbose=False, capture_on_error: bool = False) -> None:
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


def _case_free_dofs(path: Path) -> Optional[int]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
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


def _case_size_override(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    label = data.get("benchmark_size")
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    if normalized in {"small", "medium", "large"}:
        return normalized
    return None


def _load_case_flags(path: Path) -> Tuple[bool, bool]:
    data = json.loads(path.read_text())
    enabled = bool(data.get("enabled", True))
    disabled = not enabled
    runnable = enabled
    return disabled, runnable


def load_case_enabled(path: Path) -> bool:
    _, runnable = _load_case_flags(path)
    return runnable


def _absolutize_time_series_paths(case_data: dict, case_json_path: Path) -> None:
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
        ts[key] = str((case_json_path.parent / raw_path).resolve())


def filter_cases_by_enabled(
    case_specs: List[CaseSpec],
    include_disabled: bool,
) -> Tuple[List[CaseSpec], int, int]:
    filtered = []
    disabled_selected = 0
    skipped_disabled = 0
    for case in case_specs:
        disabled, runnable = _load_case_flags(case.json_path)
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
        return CaseSpec(name=name, json_path=case_json)
    return None


def resolve_case_from_path(path: Path) -> Optional[CaseSpec]:
    if not path.exists():
        return None
    if path.suffix != ".json":
        return None
    return CaseSpec(name=path.stem, json_path=path)


def expand_case_patterns(validation_root: Path, patterns: Iterable[str]) -> List[CaseSpec]:
    cases = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            for match in validation_root.glob(f"{pattern}/{pattern}.json"):
                cases.append(CaseSpec(name=match.stem, json_path=match))
            for match in validation_root.glob(pattern):
                if match.is_dir():
                    candidate = match / f"{match.name}.json"
                    if candidate.exists():
                        cases.append(CaseSpec(name=candidate.stem, json_path=candidate))
                elif match.suffix == ".json":
                    cases.append(CaseSpec(name=match.stem, json_path=match))
        else:
            case = resolve_case_from_name(validation_root, pattern)
            if case:
                cases.append(case)
    unique = {case.json_path.resolve(): case for case in cases}
    return sorted(unique.values(), key=lambda c: c.name)


def discover_default_cases(validation_root: Path) -> List[CaseSpec]:
    cases = []
    for match in validation_root.glob("*/*.json"):
        if not load_case_enabled(match) and os.getenv("STRUT_RUN_ALL_CASES") != "1":
            continue
        cases.append(CaseSpec(name=match.stem, json_path=match))
    return sorted(cases, key=lambda c: c.name)


def discover_all_cases(validation_root: Path) -> List[CaseSpec]:
    cases = [
        CaseSpec(name=match.stem, json_path=match)
        for match in validation_root.glob("*/*.json")
    ]
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


def ensure_mojo_solver(repo_root: Path, verbose: bool, profile: bool) -> Path:
    mojo = shutil.which("mojo")
    if mojo is None:
        raise SystemExit("mojo executable not found on PATH; required to run benchmarks.")
    solver_path = os.getenv("STRUT_MOJO_BIN")
    if solver_path:
        return Path(solver_path)
    solver_path = repo_root / "build" / "mojo" / "strut"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    log("Building Mojo solver...")
    build_cmd = [mojo, "build", str(repo_root / "src" / "mojo" / "strut.mojo")]
    if profile:
        build_cmd += ["-D", "STRUT_PROFILE=1"]
    build_cmd += ["-o", str(solver_path)]
    run(
        build_cmd,
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
        raise ValueError(
            f"empty comparable output rows: {ref_path} vs {got_path}"
        )
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


def _inject_opensees_timing(tcl_lines: List[str], timing_file: str) -> List[str]:
    out = []
    injected = False
    for line in tcl_lines:
        stripped = line.lstrip()
        has_analyze = stripped.startswith("analyze ") or "[analyze " in stripped
        has_eigen = stripped.startswith("eigen ") or "[eigen " in stripped
        if not injected and (has_analyze or has_eigen):
            out.append('set __strut_t0 [clock microseconds]')
            out.append(line)
            out.append('set __strut_t1 [clock microseconds]')
            out.append(f'set __strut_fp [open "{timing_file}" w]')
            out.append('puts $__strut_fp [expr {$__strut_t1 - $__strut_t0}]')
            out.append('close $__strut_fp')
            injected = True
            continue
        out.append(line)
    if not injected:
        raise ValueError("failed to inject timing: analyze/eigen command not found in Tcl")
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
            add(case_name, "missing mojo files", Path(path).name)
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
        missing_mojo = case_categories.get("missing mojo files", [])
        if missing_opensees and not missing_mojo:
            lines.append("Missing all Opensees Outputs")
        elif missing_mojo and not missing_opensees:
            lines.append("Missing all Mojo Outputs")
        else:
            if missing_opensees:
                lines.append(f"Missing Opensees outputs: {json.dumps(missing_opensees)}")
            if missing_mojo:
                lines.append(f"Missing Mojo outputs: {json.dumps(missing_mojo)}")
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
        default=int(os.getenv("STRUT_BENCH_REPEAT", "3")),
        help="Number of timed repetitions per engine.",
    )
    parser.add_argument(
        "--engine",
        choices=("both", "opensees", "mojo"),
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
        default=int(os.getenv("STRUT_BENCH_WARMUP", "1")),
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
        action="store_true",
        help="Emit speedscope profiles for Mojo runs (last repetition only).",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Directory for speedscope output (default: benchmark/speedscope).",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include JSON cases marked enabled=false.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    validation_root = repo_root / "tests" / "validation"
    generated_cases: List[CaseSpec] = []

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
        raise SystemExit("--gen-frame-bays and --gen-frame-stories must be provided together.")
    if args.gen_frame_bays is not None and args.gen_frame_stories is not None:
        if args.gen_frame_bays <= 0 or args.gen_frame_stories <= 0:
            raise SystemExit("--gen-frame-bays and --gen-frame-stories must be > 0.")
        default_prefix = "elastic_frame"
        if args.gen_frame_element == "forceBeamColumn2d":
            default_prefix = "force_beam_column2d_fiber_frame"
        if args.gen_frame_element == "dispBeamColumn2d":
            default_prefix = "disp_beam_column2d_fiber_frame"
        name = args.gen_frame_name or f"{default_prefix}_{args.gen_frame_bays}bay_{args.gen_frame_stories}story"
        gen_dir = repo_root / "benchmark" / ".tmp"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_path = gen_dir / f"{name}.json"
        run(
            [
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
        if auto_batch_default_gen and args.gen_frame_element != "forceBeamColumn2d":
            fiber_name = (
                f"force_beam_column2d_fiber_frame_{args.gen_frame_bays}bay_"
                f"{args.gen_frame_stories}story"
            )
            fiber_path = gen_dir / f"{fiber_name}.json"
            run(
                [
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
        case_specs = discover_all_cases(validation_root)

    if generated_cases:
        case_specs.extend(generated_cases)

    if not case_specs:
        raise SystemExit("No benchmark cases found. Provide --cases or add validation cases.")

    include_disabled_effective = args.include_disabled or (
        args.cases is None and os.getenv("STRUT_RUN_ALL_CASES") == "1"
    )
    case_specs, disabled_selected_count, skipped_disabled_count = filter_cases_by_enabled(
        case_specs, include_disabled=include_disabled_effective
    )

    if not case_specs:
        raise SystemExit("All selected cases are disabled. Use --include-disabled to run.")

    if args.profile and not args.no_archive:
        args.no_archive = True

    if args.output_root:
        results_root = Path(args.output_root)
    elif args.profile:
        results_root = repo_root / "benchmark" / "results-profile"
    else:
        results_root = repo_root / "benchmark" / "results"
    archive_root = (
        Path(args.archive_root) if args.archive_root else repo_root / "benchmark" / "archive"
    )
    profile_root = None
    if args.profile:
        profile_root = (
            Path(args.profile_dir)
            if args.profile_dir
            else repo_root / "benchmark" / "speedscope"
        )
        ensure_clean_dir(profile_root)

    for sub in ("opensees", "mojo", "opensees_compute", "mojo_compute", "tcl", ".tmp"):
        ensure_clean_dir(results_root / sub)

    env = os.environ.copy()
    verbose = env.get("STRUT_VERBOSE") == "1"

    summary_cases = []
    csv_rows = []
    parity_failures = []

    log(f"Running {len(case_specs)} benchmark case(s).")
    log(
        "Disabled selected: "
        f"{disabled_selected_count}; "
        f"skipped as disabled: {skipped_disabled_count}."
    )

    run_opensees = args.engine in ("both", "opensees")
    run_mojo = args.engine in ("both", "mojo")
    mojo_solver = None
    if run_mojo:
        mojo_solver = ensure_mojo_solver(repo_root, verbose, args.profile)

    if not run_opensees and not run_mojo:
        raise SystemExit("No engines selected. Use --engine opensees|mojo|both.")

    def build_case_tcl(case: CaseSpec) -> dict:
        case_name = case.name
        case_entry = {
            "name": case_name,
            "json": str(case.json_path),
            "dofs": _case_free_dofs(case.json_path),
        }
        size_override = _case_size_override(case.json_path)
        if size_override is not None:
            case_entry["size"] = size_override

        tcl_out = results_root / "tcl" / f"{case_name}.tcl"
        run(
            [
                "python",
                str(repo_root / "scripts" / "json_to_tcl.py"),
                str(case.json_path),
                str(tcl_out),
            ],
            env=env,
            verbose=verbose,
        )
        tcl_compute = results_root / "tcl" / f"{case_name}_compute.tcl"
        tcl_timed = results_root / "tcl" / f"{case_name}_timed.tcl"
        tcl_lines = tcl_out.read_text().splitlines()
        tcl_compute.write_text(
            "\n".join(line for line in tcl_lines if not line.lstrip().startswith("recorder "))
            + "\n",
            encoding="utf-8",
        )
        tcl_timed.write_text(
            "\n".join(_inject_opensees_timing(tcl_lines, "analysis_time_us.txt")) + "\n",
            encoding="utf-8",
        )
        case_entry["tcl"] = str(tcl_out)
        case_entry["tcl_compute"] = str(tcl_compute)
        case_entry["tcl_timed"] = str(tcl_timed)
        return case_entry

    case_entries = [build_case_tcl(case) for case in case_specs]
    if batch_mode:
        batch_case_entries = []
        skipped_eigen_cases = []
        for entry in case_entries:
            timed_tcl = Path(entry["tcl_timed"])
            if _tcl_uses_eigen(timed_tcl):
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
            lines.append(f"set __strut_case_err [catch {{source {{{tcl_path}}}}} __strut_case_msg]")
            lines.append("set __strut_case_t1 [clock microseconds]")
            lines.append('set __strut_fp [open "case_time_us.txt" w]')
            lines.append('puts $__strut_fp [expr {$__strut_case_t1 - $__strut_case_t0}]')
            lines.append("close $__strut_fp")
            lines.append("if {$__strut_case_err != 0} {")
            lines.append('  set __strut_err_fp [open "case_error.txt" w]')
            lines.append("  puts $__strut_err_fp $__strut_case_msg")
            lines.append("  close $__strut_err_fp")
            lines.append("}")
            lines.append("cd $__strut_repo")
            lines.append("wipe")
        batch_path = results_root / "tcl" / (
            "batch_opensees_compute.tcl" if compute else "batch_opensees_timed.tcl"
        )
        batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return batch_path

    def _write_mojo_batch_manifest(
        entries: List[dict], output_root: Path, compute: bool
    ) -> Path:
        batch_entries = []
        for entry in entries:
            case_name = entry["name"]
            if compute:
                compute_case = json.loads(Path(entry["json"]).read_text())
                compute_case["recorders"] = []
                _absolutize_time_series_paths(compute_case, Path(entry["json"]))
                compute_json = results_root / ".tmp" / f"{case_name}_compute_batch.json"
                compute_json.write_text(
                    json.dumps(compute_case, indent=2) + "\n", encoding="utf-8"
                )
                input_path = compute_json
            else:
                input_path = Path(entry["json"])
            batch_entries.append(
                {
                    "input": str(input_path),
                    "output": str(output_root / case_name),
                }
            )
        batch = {"cases": batch_entries}
        batch_path = results_root / ".tmp" / (
            "mojo_batch_compute.json" if compute else "mojo_batch.json"
        )
        batch_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
        return batch_path

    def run_opensees_batch(entries: List[dict], output_root: Path, compute: bool) -> None:
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

        n = len(entries)
        names = [entry["name"] for entry in entries]
        analysis_by_case = {name: [] for name in names}
        total_by_case = {name: [] for name in names}

        for i in range(warmup):
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            run_opensees_batch(rotated, output_root, compute)

        times = []
        for i in range(repeat):
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            start = time.perf_counter()
            run_opensees_batch(rotated, output_root, compute)
            end = time.perf_counter()
            times.append(end - start)

            analysis_map = _load_case_metric(output_root, entries, "analysis_time_us.txt")
            total_map = _load_case_metric(output_root, entries, "case_time_us.txt")
            for name in names:
                analysis_us = analysis_map.get(name)
                total_us = total_map.get(name)
                if analysis_us is not None:
                    analysis_by_case[name].append(analysis_us)
                if total_us is not None:
                    total_by_case[name].append(total_us)

        return times, analysis_by_case, total_by_case

    def run_mojo_batch(entries: List[dict], output_root: Path, compute: bool) -> bool:
        ensure_clean_dir(output_root)
        if mojo_solver is None:
            raise SystemExit("Mojo solver not initialized.")
        batch_manifest = _write_mojo_batch_manifest(entries, output_root, compute)
        try:
            run(
                [
                    str(mojo_solver),
                    "--batch",
                str(batch_manifest),
                ],
                env=env,
                verbose=verbose,
                capture_on_error=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"mojo batch aborted (exit {exc.returncode})"
            for entry in entries:
                _write_case_error(output_root / entry["name"], msg)
            return False
        return True

    def run_mojo_batch_repeated(
        entries: List[dict], output_root: Path, compute: bool, repeat: int, warmup: int
    ):
        if not entries:
            return [], {}, {}, False

        n = len(entries)
        names = [entry["name"] for entry in entries]
        analysis_by_case = {name: [] for name in names}
        total_by_case = {name: [] for name in names}
        had_abort = False

        def run_mojo_case_fallback(entry: dict) -> None:
            case_name = entry["name"]
            target_dir = output_root / case_name
            ensure_clean_dir(target_dir)
            if compute:
                compute_case = json.loads(Path(entry["json"]).read_text())
                compute_case["recorders"] = []
                _absolutize_time_series_paths(compute_case, Path(entry["json"]))
                compute_json = results_root / ".tmp" / f"{case_name}_compute_batch_fallback.json"
                compute_json.write_text(json.dumps(compute_case, indent=2) + "\n", encoding="utf-8")
                input_path = compute_json
            else:
                input_path = Path(entry["json"])
            try:
                run(
                    [
                        str(mojo_solver),
                        "--input",
                        str(input_path),
                        "--output",
                        str(target_dir),
                    ],
                    env=env,
                    verbose=verbose,
                    capture_on_error=True,
                )
            except subprocess.CalledProcessError as exc:
                _write_case_error(target_dir, f"mojo case aborted (exit {exc.returncode})")

        for i in range(warmup):
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            ok = run_mojo_batch(rotated, output_root, compute)
            if not ok:
                had_abort = True
                for entry in rotated:
                    run_mojo_case_fallback(entry)

        times = []
        for i in range(repeat):
            offset = i % n
            rotated = entries[offset:] + entries[:offset]
            start = time.perf_counter()
            ok = run_mojo_batch(rotated, output_root, compute)
            if not ok:
                had_abort = True
                for entry in rotated:
                    run_mojo_case_fallback(entry)
            end = time.perf_counter()
            times.append(end - start)

            analysis_map = _load_case_metric(output_root, entries, "analysis_time_us.txt")
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
    run_mojo_per_case = run_mojo and not batch_mode
    if run_opensees and batch_mode:
        log("Running OpenSees in batch mode.")
        opensees_times, batch_analysis_hist, batch_total_hist = run_opensees_batch_repeated(
            case_entries,
            results_root / "opensees",
            compute=False,
            repeat=args.repeat,
            warmup=args.warmup,
        )
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
                    "mean_s": f"{(total_us or 0) / 1e6:.6f}" if total_us is not None else "",
                    "median_s": f"{(total_median or 0) / 1e6:.6f}"
                    if total_median is not None
                    else "",
                    "min_s": f"{(min(total_hist) if total_hist else 0) / 1e6:.6f}"
                    if total_hist
                    else "",
                    "analysis_us": analysis_us,
                }
            )
        if not args.skip_compute_only:
            opensees_compute = run_engine(
                lambda out, last_run: run_opensees_batch(case_entries, out, compute=True),
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

    if run_mojo and batch_mode:
        log("Running Mojo in batch mode.")
        mojo_times, batch_analysis_hist, batch_total_hist, mojo_batch_had_abort = run_mojo_batch_repeated(
            case_entries,
            results_root / "mojo",
            compute=False,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        if mojo_batch_had_abort:
            log_err("Mojo batch aborted for one or more cases; used per-case fallback.")
        else:
            log_ok("Mojo batch pass OK.")
        batch_stats = {
            "times_s": mojo_times,
            "mean_s": mean(mojo_times),
            "median_s": median(mojo_times),
            "min_s": min(mojo_times),
        }
        csv_rows.append(
            {
                "case": "mojo_batch",
                "dofs": "",
                "engine": "mojo",
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
            case_entry["mojo"] = stats
            batch_entry = case_entry.setdefault("mojo_batch", {})
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
                    "engine": "mojo",
                    "mode": "total_batch_case",
                    "repeat": "",
                    "warmup": "",
                    "mean_s": f"{(total_us or 0) / 1e6:.6f}" if total_us is not None else "",
                    "median_s": f"{(total_median or 0) / 1e6:.6f}"
                    if total_median is not None
                    else "",
                    "min_s": f"{(min(total_hist) if total_hist else 0) / 1e6:.6f}"
                    if total_hist
                    else "",
                    "analysis_us": analysis_us,
                }
            )
        if not args.skip_compute_only:
            mojo_compute, _, _, mojo_compute_had_abort = run_mojo_batch_repeated(
                case_entries,
                results_root / "mojo_compute",
                compute=True,
                repeat=args.repeat,
                warmup=args.warmup,
            )
            if mojo_compute_had_abort:
                log_err(
                    "Mojo batch compute-only aborted for one or more cases; used per-case fallback."
                )
            else:
                log_ok("Mojo batch compute-only pass OK.")
            compute_stats = {
                "times_s": mojo_compute,
                "mean_s": mean(mojo_compute),
                "median_s": median(mojo_compute),
                "min_s": min(mojo_compute),
            }
            csv_rows.append(
                {
                    "case": "mojo_batch",
                    "dofs": "",
                    "engine": "mojo",
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

        def mojo_cmd(output_dir: Path, last_run: bool) -> None:
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "mojo" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            if mojo_solver is None:
                raise SystemExit("Mojo solver not initialized.")
            cmd = [
                str(mojo_solver),
                "--input",
                str(case_entry["json"]),
                "--output",
                str(target_dir),
            ]
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

        def mojo_compute_cmd(output_dir: Path, last_run: bool) -> None:
            if last_run:
                ensure_clean_dir(output_dir)
                target_dir = output_dir
            else:
                tmp_dir = results_root / ".tmp" / "mojo_compute" / case_name
                ensure_clean_dir(tmp_dir)
                target_dir = tmp_dir
            compute_case = json.loads(Path(case_entry["json"]).read_text())
            compute_case["recorders"] = []
            _absolutize_time_series_paths(compute_case, Path(case_entry["json"]))
            compute_json = results_root / ".tmp" / f"{case_name}_compute.json"
            compute_json.write_text(json.dumps(compute_case, indent=2) + "\n", encoding="utf-8")
            run(
                [
                    str(mojo_solver),
                    "--input",
                    str(compute_json),
                    "--output",
                    str(target_dir),
                ],
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
            log_ok(f"[{case_name}] OpenSees total pass OK.")
            stats = {
                "times_s": opensees_times,
                "mean_s": mean(opensees_times),
                "median_s": median(opensees_times),
                "min_s": min(opensees_times),
            }
            case_entry["opensees"] = stats
            analysis_file = results_root / "opensees" / case_name / "analysis_time_us.txt"
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
                    lambda out, last_run: opensees_compute_cmd(out / case_name, last_run),
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

        if run_mojo_per_case:
            log(f"[{case_name}] Mojo total pass...")
            try:
                mojo_times = run_engine(
                    lambda out, last_run: mojo_cmd(out / case_name, last_run),
                    results_root / "mojo",
                    args.repeat,
                    args.warmup,
                )
            except subprocess.CalledProcessError as exc:
                _write_case_error(
                    results_root / "mojo" / case_name,
                    f"mojo total pass aborted (exit {exc.returncode})",
                )
                log_err(f"[{case_name}] Mojo total pass aborted (exit {exc.returncode}).")
            else:
                log_ok(f"[{case_name}] Mojo total pass OK.")
                stats = {
                    "times_s": mojo_times,
                    "mean_s": mean(mojo_times),
                    "median_s": median(mojo_times),
                    "min_s": min(mojo_times),
                }
                analysis_file = results_root / "mojo" / case_name / "analysis_time_us.txt"
                stats["analysis_us"] = _read_analysis_us(analysis_file)
                case_entry["mojo"] = stats
                csv_rows.append(
                    {
                        "case": case_name,
                        "dofs": case_entry.get("dofs", ""),
                        "engine": "mojo",
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
                    mojo_compute = run_engine(
                        lambda out, last_run: mojo_compute_cmd(out / case_name, last_run),
                        results_root / "mojo_compute",
                        args.repeat,
                        args.warmup,
                    )
                except subprocess.CalledProcessError as exc:
                    _write_case_error(
                        results_root / "mojo_compute" / case_name,
                        f"mojo compute-only pass aborted (exit {exc.returncode})",
                    )
                    log_err(
                        f"[{case_name}] Mojo compute-only pass aborted (exit {exc.returncode})."
                    )
                else:
                    log_ok(f"[{case_name}] Mojo compute-only pass OK.")
                    compute_stats = {
                        "times_s": mojo_compute,
                        "mean_s": mean(mojo_compute),
                        "median_s": median(mojo_compute),
                        "min_s": min(mojo_compute),
                    }
                    case_entry["mojo_compute_only"] = compute_stats
                    compute_analysis_file = (
                        results_root / "mojo_compute" / case_name / "analysis_time_us.txt"
                    )
                    compute_analysis_us = _read_analysis_us(compute_analysis_file)
                    csv_rows.append(
                        {
                            "case": case_name,
                            "dofs": case_entry.get("dofs", ""),
                            "engine": "mojo",
                            "mode": "compute_only",
                            "repeat": args.repeat,
                            "warmup": args.warmup,
                            "mean_s": f"{compute_stats['mean_s']:.6f}",
                            "median_s": f"{compute_stats['median_s']:.6f}",
                            "min_s": f"{compute_stats['min_s']:.6f}",
                            "analysis_us": compute_analysis_us,
                        }
                    )

        if run_opensees and run_mojo:
            case_data = json.loads(Path(case_entry["json"]).read_text())
            recorders = case_data.get("recorders", [])
            analysis = case_data.get("analysis", {})
            analysis_type = str(analysis.get("type", "static_linear"))
            is_transient = analysis_type.startswith("transient")
            use_last_common_row = analysis_type == "static_nonlinear"
            tol = case_data.get("parity_tolerance", {})
            rtol = tol.get("rtol", REL_TOL)
            atol = tol.get("atol", ABS_TOL)
            for rec in recorders:
                rec_type = rec.get("type")
                if rec_type == "node_displacement":
                    output = rec.get("output", "node_disp")
                    for node_id in rec.get("nodes", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        mojo_file = (
                            results_root
                            / "mojo"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        if not ref_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not mojo_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing Mojo output: {mojo_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                mojo_vals = _load_all_values(mojo_file)
                            else:
                                ref_vals, mojo_vals = _load_last_comparable_values(
                                    ref_file,
                                    mojo_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            if len(ref_vals) != len(mojo_vals):
                                parity_failures.append(
                                    f"{case_name}: node {node_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                                )
                                continue
                            for step, (rvec, gvec) in enumerate(
                                zip(ref_vals, mojo_vals), start=1
                            ):
                                ok, errors = _compare_vectors(
                                    rvec, gvec, rtol=rtol, atol=atol
                                )
                                if not ok:
                                    parity_failures.append(
                                        f"{case_name}: node {node_id} mismatch at step {step}"
                                    )
                                    parity_failures.extend([f"  {err}" for err in errors])
                                    break
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, mojo_vals, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(f"{case_name}: node {node_id} mismatch")
                                parity_failures.extend([f"  {err}" for err in errors])
                elif rec_type == "element_force":
                    output = rec.get("output", "element_force")
                    for elem_id in rec.get("elements", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        mojo_file = (
                            results_root
                            / "mojo"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        if not ref_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not mojo_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing Mojo output: {mojo_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                mojo_vals = _load_all_values(mojo_file)
                            else:
                                ref_vals, mojo_vals = _load_last_comparable_values(
                                    ref_file,
                                    mojo_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            if len(ref_vals) != len(mojo_vals):
                                parity_failures.append(
                                    f"{case_name}: element {elem_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                                )
                                continue
                            for step, (rvec, gvec) in enumerate(
                                zip(ref_vals, mojo_vals), start=1
                            ):
                                ok, errors = _compare_vectors(
                                    rvec, gvec, rtol=rtol, atol=atol
                                )
                                if not ok:
                                    parity_failures.append(
                                        f"{case_name}: element {elem_id} mismatch at step {step}"
                                    )
                                    parity_failures.extend([f"  {err}" for err in errors])
                                    break
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, mojo_vals, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(
                                    f"{case_name}: element {elem_id} mismatch"
                                )
                                parity_failures.extend([f"  {err}" for err in errors])
                elif rec_type == "node_reaction":
                    output = rec.get("output", "reaction")
                    for node_id in rec.get("nodes", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        mojo_file = (
                            results_root
                            / "mojo"
                            / case_name
                            / f"{output}_node{node_id}.out"
                        )
                        if not ref_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not mojo_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing Mojo output: {mojo_file}"
                            )
                            continue
                        try:
                            if is_transient:
                                ref_vals = _load_all_values(ref_file)
                                mojo_vals = _load_all_values(mojo_file)
                            else:
                                ref_vals, mojo_vals = _load_last_comparable_values(
                                    ref_file,
                                    mojo_file,
                                    use_last_common_row=use_last_common_row,
                                )
                        except ValueError as exc:
                            parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if is_transient:
                            if len(ref_vals) != len(mojo_vals):
                                parity_failures.append(
                                    f"{case_name}: reaction node {node_id} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                                )
                                continue
                            for step, (rvec, gvec) in enumerate(
                                zip(ref_vals, mojo_vals), start=1
                            ):
                                ok, errors = _compare_vectors(
                                    rvec, gvec, rtol=rtol, atol=atol
                                )
                                if not ok:
                                    parity_failures.append(
                                        f"{case_name}: reaction node {node_id} mismatch at step {step}"
                                    )
                                    parity_failures.extend([f"  {err}" for err in errors])
                                    break
                        else:
                            ok, errors = _compare_vectors(
                                ref_vals, mojo_vals, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(
                                    f"{case_name}: reaction node {node_id} mismatch"
                                )
                                parity_failures.extend([f"  {err}" for err in errors])
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
                    mojo_file = (
                        results_root
                        / "mojo"
                        / case_name
                        / f"{output}_i{i_node}_j{j_node}.out"
                    )
                    if not ref_file.exists():
                        parity_failures.append(
                            f"{case_name}: missing OpenSees output: {ref_file}"
                        )
                        continue
                    if not mojo_file.exists():
                        parity_failures.append(
                            f"{case_name}: missing Mojo output: {mojo_file}"
                        )
                        continue
                    try:
                        if is_transient:
                            ref_vals = _load_all_values(ref_file)
                            mojo_vals = _load_all_values(mojo_file)
                        else:
                            ref_vals, mojo_vals = _load_last_comparable_values(
                                ref_file,
                                mojo_file,
                                use_last_common_row=use_last_common_row,
                            )
                    except ValueError as exc:
                        parity_failures.append(f"{case_name}: {exc}")
                        continue
                    if is_transient:
                        if len(ref_vals) != len(mojo_vals):
                            parity_failures.append(
                                f"{case_name}: drift i{i_node}-j{j_node} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                            )
                            continue
                        for step, (rvec, gvec) in enumerate(
                            zip(ref_vals, mojo_vals), start=1
                        ):
                            ok, errors = _compare_vectors(
                                rvec, gvec, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(
                                    f"{case_name}: drift i{i_node}-j{j_node} mismatch at step {step}"
                                )
                                parity_failures.extend([f"  {err}" for err in errors])
                                break
                    else:
                        ok, errors = _compare_vectors(
                            ref_vals, mojo_vals, rtol=rtol, atol=atol
                        )
                        if not ok:
                            parity_failures.append(
                                f"{case_name}: drift i{i_node}-j{j_node} mismatch"
                            )
                            parity_failures.extend([f"  {err}" for err in errors])
                elif rec_type == "envelope_element_force":
                    output = rec.get("output", "envelope_element_force")
                    for elem_id in rec.get("elements", []):
                        ref_file = (
                            results_root
                            / "opensees"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        mojo_file = (
                            results_root
                            / "mojo"
                            / case_name
                            / f"{output}_ele{elem_id}.out"
                        )
                        if not ref_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing OpenSees output: {ref_file}"
                            )
                            continue
                        if not mojo_file.exists():
                            parity_failures.append(
                                f"{case_name}: missing Mojo output: {mojo_file}"
                            )
                            continue
                        try:
                            ref_vals = _load_all_values(ref_file)
                            mojo_vals = _load_all_values(mojo_file)
                        except ValueError as exc:
                            parity_failures.append(f"{case_name}: {exc}")
                            continue
                        if len(ref_vals) != len(mojo_vals):
                            parity_failures.append(
                                f"{case_name}: envelope element {elem_id} row count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                            )
                            continue
                        for row, (rvec, gvec) in enumerate(zip(ref_vals, mojo_vals), start=1):
                            ok, errors = _compare_vectors(
                                rvec, gvec, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(
                                    f"{case_name}: envelope element {elem_id} mismatch at row {row}"
                                )
                                parity_failures.extend([f"  {err}" for err in errors])
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
                            parity_failures.append(
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
                            mojo_file = (
                                results_root
                                / "mojo"
                                / case_name
                                / f"{output}_ele{elem_id}_sec{sec_no}.out"
                            )
                            if not ref_file.exists():
                                parity_failures.append(
                                    f"{case_name}: missing OpenSees output: {ref_file}"
                                )
                                continue
                            if not mojo_file.exists():
                                parity_failures.append(
                                    f"{case_name}: missing Mojo output: {mojo_file}"
                                )
                                continue
                            try:
                                if is_transient:
                                    ref_vals = _load_all_values(ref_file)
                                    mojo_vals = _load_all_values(mojo_file)
                                else:
                                    ref_vals, mojo_vals = _load_last_comparable_values(
                                        ref_file,
                                        mojo_file,
                                        use_last_common_row=use_last_common_row,
                                    )
                            except ValueError as exc:
                                parity_failures.append(f"{case_name}: {exc}")
                                continue
                            if is_transient:
                                if len(ref_vals) != len(mojo_vals):
                                    parity_failures.append(
                                        f"{case_name}: {rec_type} element {elem_id} section {sec_no} step count mismatch: {len(ref_vals)} != {len(mojo_vals)}"
                                    )
                                    continue
                                for step, (rvec, gvec) in enumerate(
                                    zip(ref_vals, mojo_vals), start=1
                                ):
                                    ok, errors = _compare_vectors(
                                        rvec, gvec, rtol=rtol, atol=atol
                                    )
                                    if not ok:
                                        parity_failures.append(
                                            f"{case_name}: {rec_type} element {elem_id} section {sec_no} mismatch at step {step}"
                                        )
                                        parity_failures.extend([f"  {err}" for err in errors])
                                        break
                            else:
                                ok, errors = _compare_vectors(
                                    ref_vals, mojo_vals, rtol=rtol, atol=atol
                                )
                                if not ok:
                                    parity_failures.append(
                                        f"{case_name}: {rec_type} element {elem_id} section {sec_no} mismatch"
                                    )
                                    parity_failures.extend([f"  {err}" for err in errors])
                elif rec_type == "modal_eigen":
                    output = rec.get("output", "modal")
                    eig_ref = (
                        results_root / "opensees" / case_name / f"{output}_eigenvalues.out"
                    )
                    eig_mojo = results_root / "mojo" / case_name / f"{output}_eigenvalues.out"
                    if not eig_ref.exists():
                        parity_failures.append(
                            f"{case_name}: missing OpenSees output: {eig_ref}"
                        )
                        continue
                    if not eig_mojo.exists():
                        parity_failures.append(
                            f"{case_name}: missing Mojo output: {eig_mojo}"
                        )
                        continue
                    try:
                        ref_rows = _load_all_values(eig_ref)
                        mojo_rows = _load_all_values(eig_mojo)
                    except ValueError as exc:
                        parity_failures.append(f"{case_name}: {exc}")
                        continue
                    ref_vals = [row[0] for row in ref_rows]
                    mojo_vals = [row[0] for row in mojo_rows]
                    ok, errors = _compare_vectors(ref_vals, mojo_vals, rtol=rtol, atol=atol)
                    if not ok:
                        parity_failures.append(f"{case_name}: modal eigenvalue mismatch")
                        parity_failures.extend([f"  {err}" for err in errors])

                    for mode_no in rec.get("modes", []):
                        for node_id in rec.get("nodes", []):
                            ref_file = (
                                results_root
                                / "opensees"
                                / case_name
                                / f"{output}_mode{int(mode_no)}_node{int(node_id)}.out"
                            )
                            mojo_file = (
                                results_root
                                / "mojo"
                                / case_name
                                / f"{output}_mode{int(mode_no)}_node{int(node_id)}.out"
                            )
                            if not ref_file.exists():
                                parity_failures.append(
                                    f"{case_name}: missing OpenSees output: {ref_file}"
                                )
                                continue
                            if not mojo_file.exists():
                                parity_failures.append(
                                    f"{case_name}: missing Mojo output: {mojo_file}"
                                )
                                continue
                            try:
                                ref_vec = _load_last_values(ref_file)
                                mojo_vec = _load_last_values(mojo_file)
                            except ValueError as exc:
                                parity_failures.append(f"{case_name}: {exc}")
                                continue
                            ok, errors = _compare_mode_shape_vectors(
                                ref_vec, mojo_vec, rtol=rtol, atol=atol
                            )
                            if not ok:
                                parity_failures.append(
                                    f"{case_name}: modal mode shape mismatch mode={mode_no} node={node_id}"
                                )
                                parity_failures.extend([f"  {err}" for err in errors])
                else:
                    parity_failures.append(
                        f"{case_name}: unsupported recorder type: {rec_type}"
                    )

        summary_cases.append(case_entry)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "generated_at": generated_at,
        "git_rev": git_rev(repo_root),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "cases": summary_cases,
    }

    summary_json = results_root / "summary.json"
    summary_csv = results_root / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, csv_rows)

    if not args.no_archive:
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(summary_json, archive_root / f"{stamp}-summary.json")
        shutil.copy2(summary_csv, archive_root / f"{stamp}-summary.csv")

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
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

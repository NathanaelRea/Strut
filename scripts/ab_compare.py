#!/usr/bin/env python3
"""A/B two Strut solver binaries on the same inputs: output equivalence plus timing.

This needs no OpenSees binary. It answers "did this refactor change results, and
how much faster is it?" by running a baseline and a candidate solver on identical
resolved inputs and comparing every recorder output numerically.

Examples:
  uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut
  uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut \
      --cases opensees_example_ex5_frame2d_eq_uniform_rc_fiber --repeat 3
  uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut \
      --no-validation-cases --extra-json .tmp/synthetic/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import direct_tcl_support  # noqa: E402
import run_strut_case  # noqa: E402

VALIDATION_ROOT = REPO_ROOT / "tests" / "validation"


def _validation_case_paths(include_disabled: bool) -> list[Path]:
    paths = []
    for path in sorted(VALIDATION_ROOT.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if path.name == "direct_tcl_case.json":
            status = str(data.get("status", "")).strip().lower()
            if include_disabled or (data.get("enabled", True) and status != "todo"):
                paths.append(path)
            continue
        if "model" not in data or "nodes" not in data:
            continue
        if include_disabled or data.get("enabled", True):
            paths.append(path)
    return paths


def _resolve_named_case(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate.resolve()
    case_dir = VALIDATION_ROOT / name
    for path in (case_dir / f"{name}.json", case_dir / "direct_tcl_case.json"):
        if path.exists():
            return path
    raise SystemExit(f"case not found: {name}")


def _resolved_input(case_path: Path, work_dir: Path) -> tuple[str, Path]:
    """Return (case name, solver-ready JSON path) for a JSON or direct-Tcl case."""
    data = json.loads(case_path.read_text(encoding="utf-8"))
    if case_path.name == "direct_tcl_case.json":
        import tcl_to_strut

        name = str(data.get("name") or case_path.parent.name)
        entry_tcl = (REPO_ROOT / data["entry_tcl"]).resolve()
        source_files = direct_tcl_support.resolve_direct_tcl_source_files(
            entry_tcl, case_path
        )
        runtime_entry, _ = direct_tcl_support.prepare_direct_tcl_entry(
            entry_tcl,
            source_files or [entry_tcl],
            work_dir / name / "_direct_tcl_context",
            manifest_path=case_path,
            excluded_roots=[case_path.parent],
        )
        case_data = tcl_to_strut.convert_tcl_to_solver_input(runtime_entry, REPO_ROOT)
    else:
        name = case_path.stem
        case_data = data
        run_strut_case._normalize_time_series_paths(case_data, case_path, REPO_ROOT)
    out_json = work_dir / name / "input.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(case_data), encoding="utf-8")
    return name, out_json


def _run_solver(
    binary: Path, input_json: Path, out_dir: Path, repeat: int, timeout: float
) -> dict:
    best_analysis_us = None
    best_wall_s = None
    error = None
    for _ in range(max(1, repeat)):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(binary), "--input", str(input_json), "--output", str(out_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout:.0f}s"}
        wall_s = time.perf_counter() - start
        if proc.returncode != 0:
            first_line = (proc.stdout + proc.stderr).strip().splitlines()[:1]
            error = first_line[0] if first_line else f"exit {proc.returncode}"
            return {"error": error}
        analysis_path = out_dir / "analysis_time_us.txt"
        analysis_us = (
            int(analysis_path.read_text().strip()) if analysis_path.exists() else None
        )
        if analysis_us is not None and (
            best_analysis_us is None or analysis_us < best_analysis_us
        ):
            best_analysis_us = analysis_us
        if best_wall_s is None or wall_s < best_wall_s:
            best_wall_s = wall_s
    return {"analysis_us": best_analysis_us, "wall_s": best_wall_s, "error": error}


def _load_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        rows.append([float(p) for p in parts])
    return rows


def _compare_outputs(
    base_dir: Path, cand_dir: Path, rtol: float, atol: float
) -> tuple[bool, float, str]:
    base_files = sorted(p.name for p in base_dir.glob("*.out"))
    cand_files = sorted(p.name for p in cand_dir.glob("*.out"))
    if base_files != cand_files:
        missing = sorted(set(base_files) ^ set(cand_files))
        return False, math.inf, f"output file sets differ: {missing[:3]}"
    worst = 0.0
    detail = ""
    first_failure = ""
    ok = True
    for name in base_files:
        base_rows = _load_rows(base_dir / name)
        cand_rows = _load_rows(cand_dir / name)
        if len(base_rows) != len(cand_rows):
            return (
                False,
                math.inf,
                f"{name}: row count {len(base_rows)} != {len(cand_rows)}",
            )
        scale = max((abs(v) for row in base_rows for v in row), default=0.0)
        for row_index, (base_row, cand_row) in enumerate(zip(base_rows, cand_rows)):
            if len(base_row) != len(cand_row):
                return False, math.inf, f"{name}: row {row_index} width differs"
            for base_value, cand_value in zip(base_row, cand_row):
                diff = abs(base_value - cand_value)
                # Normalize by the file's peak so near-zero crossings do not dominate.
                rel = diff / scale if scale > 0.0 else diff
                # A file that is numerically zero everywhere has no meaningful relative error.
                if diff <= atol:
                    rel = min(rel, diff)
                if rel > worst:
                    worst = rel
                    detail = (
                        f"{name} row {row_index}: {base_value:.9e} vs {cand_value:.9e}"
                    )
                if diff > atol + rtol * max(abs(base_value), scale * 1.0e-3):
                    if ok:
                        first_failure = f"{name} row {row_index}: {base_value:.9e} vs {cand_value:.9e}"
                    ok = False
    return ok, worst, detail if ok else f"first failure {first_failure}; worst {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", required=True, help="Baseline solver binary.")
    parser.add_argument("--candidate", required=True, help="Candidate solver binary.")
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated validation case names or JSON paths.",
    )
    parser.add_argument(
        "--extra-json", nargs="*", default=[], help="Extra solver-ready JSON inputs."
    )
    parser.add_argument("--no-validation-cases", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Timed runs per binary; the minimum is reported.",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-9)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--min-ms",
        type=float,
        default=0.0,
        help="Hide cases whose baseline is faster than this.",
    )
    parser.add_argument(
        "--json-out", default="", help="Write machine-readable results here."
    )
    args = parser.parse_args()

    baseline = Path(args.baseline).resolve()
    candidate = Path(args.candidate).resolve()
    for binary in (baseline, candidate):
        if not binary.exists():
            raise SystemExit(f"solver binary not found: {binary}")

    case_paths: list[Path] = []
    if args.cases:
        case_paths = [
            _resolve_named_case(item.strip())
            for item in args.cases.split(",")
            if item.strip()
        ]
    elif not args.no_validation_cases:
        case_paths = _validation_case_paths(args.include_disabled)

    results = []
    # The direct-Tcl converter only reads inside the repo, so stage work there.
    scratch_root = REPO_ROOT / ".tmp" / "ab_compare"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_", dir=scratch_root) as tmp:
        work_dir = Path(tmp)
        inputs: list[tuple[str, Path]] = []
        for case_path in case_paths:
            try:
                inputs.append(_resolved_input(case_path, work_dir))
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - report and continue
                results.append(
                    {
                        "case": case_path.parent.name,
                        "status": "SETUP",
                        "detail": str(exc)[:160],
                    }
                )
        for extra in args.extra_json:
            extra_path = Path(extra).resolve()
            inputs.append((extra_path.stem, extra_path))

        for name, input_json in inputs:
            base_out = work_dir / name / "baseline"
            cand_out = work_dir / name / "candidate"
            base = _run_solver(
                baseline, input_json, base_out, args.repeat, args.timeout
            )
            cand = _run_solver(
                candidate, input_json, cand_out, args.repeat, args.timeout
            )
            entry = {"case": name, "baseline": base, "candidate": cand}
            if base.get("error") and cand.get("error"):
                entry.update(status="BOTH-FAIL", detail=str(base["error"])[:120])
            elif base.get("error"):
                entry.update(status="BASE-FAIL", detail=str(base["error"])[:120])
            elif cand.get("error"):
                entry.update(status="CAND-FAIL", detail=str(cand["error"])[:120])
            else:
                ok, worst, detail = _compare_outputs(
                    base_out, cand_out, args.rtol, args.atol
                )
                entry.update(
                    status="OK" if ok else "DIFF", max_rel_diff=worst, detail=detail
                )
            results.append(entry)
            sys.stderr.write(f"  {entry['status']:9s} {name}\n")

    print(
        f"{'case':66s} {'base ms':>10s} {'cand ms':>10s} {'speedup':>8s} {'max rel diff':>12s}  status"
    )
    total_base = 0.0
    total_cand = 0.0
    for entry in results:
        base_us = (entry.get("baseline") or {}).get("analysis_us")
        cand_us = (entry.get("candidate") or {}).get("analysis_us")
        if (
            base_us is not None
            and base_us / 1000.0 < args.min_ms
            and entry["status"] == "OK"
        ):
            continue
        base_text = (
            f"{base_us / 1000.0:10.3f}" if base_us is not None else f"{'-':>10s}"
        )
        cand_text = (
            f"{cand_us / 1000.0:10.3f}" if cand_us is not None else f"{'-':>10s}"
        )
        speed_text = (
            f"{base_us / cand_us:7.2f}x" if base_us and cand_us else f"{'-':>8s}"
        )
        diff = entry.get("max_rel_diff")
        diff_text = f"{diff:12.2e}" if diff is not None else f"{'-':>12s}"
        tail = "" if entry["status"] == "OK" else f"  {entry.get('detail', '')}"
        print(
            f"{entry['case'][:66]:66s} {base_text} {cand_text} {speed_text} {diff_text}  {entry['status']}{tail}"
        )
        if base_us and cand_us:
            total_base += base_us
            total_cand += cand_us
    if total_cand > 0.0:
        print(
            f"{'TOTAL (cases that ran on both)':66s} {total_base / 1000.0:10.3f} {total_cand / 1000.0:10.3f} "
            f"{total_base / total_cand:7.2f}x"
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    bad = [e for e in results if e["status"] in ("DIFF", "CAND-FAIL")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

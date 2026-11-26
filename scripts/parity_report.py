#!/usr/bin/env python3
"""Run one case in OpenSees and Strut and report how far the results are apart.

`run_case.py` answers pass or fail at the parity tolerance and stops at the first
mismatch. This prints the size of the difference for every recorder group, so a
case that does not match yet can be tracked, and a large model can be compared
piece by piece.

Each recorder group (one `output` name, one column) is scored by the largest
difference between the engines over all its nodes and steps, divided by the
largest OpenSees magnitude in that group. Dividing by the group's peak keeps
near-zero entries from dominating.

Examples:
  uv run scripts/parity_report.py benchmark/.tmp/megatall_slice_z0_12/megatall_slice_z0_12.json
  uv run scripts/parity_report.py case.json --work-dir .tmp/parity/case --fail-above 1e-4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, quiet: bool = True) -> float:
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-15:])
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{tail}"
        )
    return time.perf_counter() - start


def _load_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            rows.append([float(value) for value in parts])
    return rows


def _group_name(file_name: str) -> str:
    """`disp_node12.out` and `disp_node13.out` belong to group `disp`."""
    stem = file_name[: -len(".out")]
    return re.sub(r"_(node|ele|i)\d+.*$", "", stem)


def _with_system(case_json: Path, system: str, out_path: Path) -> Path:
    """Copy of the case that asks for another linear solver."""
    data = json.loads(case_json.read_text(encoding="utf-8"))
    analysis = data["analysis"]
    for block in [
        analysis,
        *(stage["analysis"] for stage in analysis.get("stages", [])),
    ]:
        if "system" in block or block is analysis:
            block["system"] = system
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return out_path


def run_engines(
    case_json: Path, work_dir: Path, strut_system: str = ""
) -> tuple[Path, Path, float, float]:
    reference = work_dir / "reference"
    strut = work_dir / "strut"
    generated = work_dir / "generated"
    for path in (reference, strut, generated):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
    model_tcl = generated / "model.tcl"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/json_to_tcl.py",
            str(case_json),
            str(model_tcl),
        ]
    )
    opensees_s = _run(
        [
            "scripts/run_opensees.sh",
            "--script",
            str(model_tcl),
            "--output",
            str(reference),
        ]
    )
    strut_input = case_json
    if strut_system:
        strut_input = _with_system(
            case_json, strut_system, generated / "strut_input.json"
        )
    strut_s = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_strut_case.py",
            "--input",
            str(strut_input),
            "--output",
            str(strut),
        ]
    )
    return reference, strut, opensees_s, strut_s


def compare(reference: Path, strut: Path) -> list[dict]:
    """One entry per (group, column): peak, worst difference and where it is."""
    files = sorted(path.name for path in reference.glob("*.out"))
    if not files:
        raise SystemExit(
            f"OpenSees wrote no recorder output in {reference}; "
            "it probably stopped before finishing (run its model.tcl directly to see why)"
        )
    groups: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"peak": 0.0, "worst": 0.0, "where": "", "files": 0, "rows": 0}
    )
    for name in files:
        strut_file = strut / name
        if not strut_file.exists():
            raise SystemExit(f"Strut did not write {name}")
        ref_rows = _load_rows(reference / name)
        got_rows = _load_rows(strut_file)
        if not ref_rows:
            raise SystemExit(
                f"OpenSees wrote an empty {name}: its analysis did not finish"
            )
        if len(ref_rows) != len(got_rows):
            raise SystemExit(
                f"{name}: OpenSees has {len(ref_rows)} rows, Strut {len(got_rows)}"
            )
        group = _group_name(name)
        for column in range(len(ref_rows[0])):
            entry = groups[(group, column)]
            entry["files"] += 1
            entry["rows"] = len(ref_rows)
            for step, (ref_row, got_row) in enumerate(zip(ref_rows, got_rows)):
                entry["peak"] = max(entry["peak"], abs(ref_row[column]))
                diff = abs(ref_row[column] - got_row[column])
                if diff > entry["worst"]:
                    entry["worst"] = diff
                    entry["where"] = f"{name} step {step + 1}"
    report = []
    for (group, column), entry in sorted(groups.items()):
        peak = entry["peak"]
        report.append(
            {
                "group": group,
                "column": column + 1,
                "files": entry["files"],
                "steps": entry["rows"],
                "peak": peak,
                "worst_abs": entry["worst"],
                "worst_rel": entry["worst"] / peak if peak > 0.0 else 0.0,
                "where": entry["where"],
            }
        )
    return report


def print_report(report: list[dict], zero_below: float) -> float:
    """Print the table and return the worst scored diff/peak.

    A column is not scored when it is zero by symmetry: its peak is below
    `zero_below` times the largest peak among its group's columns of the same
    kind (translations with translations, rotations with rotations).
    """
    kind_peak: dict[tuple[str, int], float] = defaultdict(float)
    for row in report:
        kind = (row["group"], (row["column"] - 1) // 3)
        kind_peak[kind] = max(kind_peak[kind], row["peak"])
    print(
        f"{'group':24s} {'col':>3s} {'files':>6s} {'peak':>12s} {'max diff':>12s} "
        f"{'diff/peak':>10s}  worst at"
    )
    worst = 0.0
    for row in report:
        reference_peak = kind_peak[(row["group"], (row["column"] - 1) // 3)]
        negligible = row["peak"] <= zero_below * reference_peak
        rel_text = "   (zero)" if negligible else f"{row['worst_rel']:10.2e}"
        if not negligible:
            worst = max(worst, row["worst_rel"])
        print(
            f"{row['group'][:24]:24s} {row['column']:3d} {row['files']:6d} {row['peak']:12.4e} "
            f"{row['worst_abs']:12.4e} {rel_text}  {row['where']}"
        )
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case_json", type=Path)
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Default: .tmp/parity/<name>."
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Compare the outputs already in the work directory instead of running the engines.",
    )
    parser.add_argument(
        "--zero-below",
        type=float,
        default=1.0e-9,
        help=(
            "A column whose peak is below this fraction of its group's largest "
            "peak is zero by symmetry: reported, not scored."
        ),
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="Exit 1 if any diff/peak exceeds this.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--strut-system",
        default="",
        help=(
            "Linear solver for Strut only. OpenSees' SparseSYM is inexact on rigid-diaphragm "
            "systems, so a tight comparison of a large model runs OpenSees with UmfPack "
            "and Strut with its own SparseSYM."
        ),
    )
    args = parser.parse_args()

    case_json = args.case_json.resolve()
    work_dir = (
        args.work_dir or REPO_ROOT / ".tmp" / "parity" / case_json.stem
    ).resolve()
    if args.reuse:
        reference, strut = work_dir / "reference", work_dir / "strut"
    else:
        reference, strut, opensees_s, strut_s = run_engines(
            case_json, work_dir, args.strut_system
        )
        print(
            f"OpenSees {opensees_s:.1f} s, Strut {strut_s:.1f} s (wall, recorders on)"
        )
    report = compare(reference, strut)
    worst = print_report(report, args.zero_below)
    print(f"worst diff/peak: {worst:.2e}")
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_above is not None and worst > args.fail_above:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()
        rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"empty output file: {path}")
    width = len(rows[0])
    for i, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(f"inconsistent row width in {path} at line {i}")
    return rows


def _series(rows, component_idx):
    return [row[component_idx] for row in rows]


def _max_abs(values):
    return max((abs(v) for v in values), default=0.0)


def _run_case(repo_root: Path, case_json: Path, refresh_reference: bool):
    env = os.environ.copy()
    if refresh_reference:
        env["STRUT_REFRESH_REFERENCE"] = "1"
    cmd = ["uv", "run", str(repo_root / "scripts" / "run_case.py"), str(case_json)]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(
            f"warning: run_case exited with code {result.returncode}; plotting available outputs anyway"
        )


def _plot_one(
    case_name: str,
    out_name: str,
    component: int,
    ref_vals,
    mojo_vals,
    x_vals,
    x_label: str,
    out_path: Path,
):
    diffs = [g - r for r, g in zip(ref_vals, mojo_vals)]
    rmse = (sum(d * d for d in diffs) / max(len(diffs), 1)) ** 0.5
    ref_peak = _max_abs(ref_vals)
    mojo_peak = _max_abs(mojo_vals)
    peak_abs_err = abs(mojo_peak - ref_peak)
    peak_rel_err = peak_abs_err / max(ref_peak, 1.0e-30)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_vals, ref_vals, label="OpenSees reference", linewidth=2.0)
    ax.plot(x_vals, mojo_vals, label="Strut mojo", linewidth=1.6, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Component {component}")
    ax.set_title(f"{case_name} :: {out_name} :: c{component}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"saved: {out_path}")
    print(f"  peak(ref)={ref_peak:.6e} peak(mojo)={mojo_peak:.6e} peak_rel={peak_rel_err:.6e} rmse={rmse:.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a validation case and plot all comparable reference/mojo output files."
    )
    parser.add_argument("case_json", help="path to case JSON")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="skip running case; only plot from existing output files",
    )
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="refresh OpenSees reference outputs while running case",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="output plot directory (default: build/plots/<case_name>)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_json = Path(args.case_json).resolve()
    if not case_json.exists():
        raise SystemExit(f"missing case json: {case_json}")

    if not args.skip_run:
        _run_case(repo_root, case_json, args.refresh_reference)

    case_name = case_json.stem
    case_root = repo_root / "tests" / "validation" / case_name
    tgt_json = case_root / f"{case_name}.json"
    if not tgt_json.exists():
        tgt_json = case_json
    case_data = json.loads(tgt_json.read_text(encoding="utf-8"))
    analysis = case_data.get("analysis", {})
    is_transient = str(analysis.get("type", "")).startswith("transient")
    dt = float(analysis.get("dt", 1.0))

    ref_dir = case_root / "reference"
    mojo_dir = case_root / "mojo"
    if not ref_dir.exists() or not mojo_dir.exists():
        raise SystemExit(
            f"missing output directories: reference={ref_dir.exists()} mojo={mojo_dir.exists()}"
        )

    if args.output_dir:
        plot_dir = Path(args.output_dir)
    else:
        plot_dir = repo_root / "build" / "plots" / case_name

    out_files = sorted(ref_dir.glob("*.out"))
    if not out_files:
        raise SystemExit(f"no .out files found in {ref_dir}")

    plotted = 0
    for ref_file in out_files:
        mojo_file = mojo_dir / ref_file.name
        if not mojo_file.exists():
            print(f"skip (missing mojo): {mojo_file}")
            continue
        try:
            ref_rows = _parse_rows(ref_file)
            mojo_rows = _parse_rows(mojo_file)
        except ValueError as exc:
            print(f"skip ({exc})")
            continue

        n = min(len(ref_rows), len(mojo_rows))
        if n == 0:
            print(f"skip (no comparable rows): {ref_file.name}")
            continue
        if len(ref_rows) != len(mojo_rows):
            print(
                f"warning: row mismatch in {ref_file.name}; ref={len(ref_rows)} mojo={len(mojo_rows)} using first {n}"
            )
        ref_rows = ref_rows[:n]
        mojo_rows = mojo_rows[:n]

        width = min(len(ref_rows[0]), len(mojo_rows[0]))
        if len(ref_rows[0]) != len(mojo_rows[0]):
            print(
                f"warning: column mismatch in {ref_file.name}; ref={len(ref_rows[0])} mojo={len(mojo_rows[0])} using first {width}"
            )

        if is_transient:
            x_vals = [dt * (i + 1) for i in range(n)]
            x_label = "Time (s)"
        else:
            x_vals = [i + 1 for i in range(n)]
            x_label = "Step"

        stem = ref_file.stem
        for comp_idx in range(width):
            component = comp_idx + 1
            ref_vals = _series(ref_rows, comp_idx)
            mojo_vals = _series(mojo_rows, comp_idx)
            out_path = plot_dir / f"{stem}_c{component}.png"
            _plot_one(
                case_name,
                ref_file.name,
                component,
                ref_vals,
                mojo_vals,
                x_vals,
                x_label,
                out_path,
            )
            plotted += 1

    print(f"done: generated {plotted} plot(s) in {plot_dir}")


if __name__ == "__main__":
    main()

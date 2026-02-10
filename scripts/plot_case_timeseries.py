#!/usr/bin/env python3
import argparse
import json
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
    return rows


def _series(rows, component):
    idx = component - 1
    values = []
    for row in rows:
        if idx >= len(row):
            raise ValueError(
                f"component {component} out of range for row width {len(row)}"
            )
        values.append(row[idx])
    return values


def _max_abs(values):
    return max((abs(v) for v in values), default=0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot reference vs mojo time-series output for a parity case."
    )
    parser.add_argument("--case", required=True, help="validation case directory name")
    parser.add_argument(
        "--series-file",
        required=True,
        help="output filename inside reference/mojo (e.g. node_disp_node4.out)",
    )
    parser.add_argument(
        "--component",
        type=int,
        default=1,
        help="1-based component index in each output row (default: 1)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="output PNG path (default under build/plots)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    case_root = repo_root / "tests" / "validation" / args.case
    case_json = case_root / f"{args.case}.json"
    if not case_json.exists():
        raise SystemExit(f"missing case JSON: {case_json}")
    case_data = json.loads(case_json.read_text(encoding="utf-8"))
    analysis = case_data.get("analysis", {})
    dt = float(analysis.get("dt", 1.0))
    is_transient = str(analysis.get("type", "")).startswith("transient")

    ref_path = case_root / "reference" / args.series_file
    mojo_path = case_root / "mojo" / args.series_file
    if not ref_path.exists():
        raise SystemExit(f"missing reference file: {ref_path}")
    if not mojo_path.exists():
        raise SystemExit(f"missing mojo file: {mojo_path}")

    ref_rows = _parse_rows(ref_path)
    mojo_rows = _parse_rows(mojo_path)
    n = min(len(ref_rows), len(mojo_rows))
    if len(ref_rows) != len(mojo_rows):
        print(
            f"warning: row count mismatch ref={len(ref_rows)} mojo={len(mojo_rows)}; plotting first {n}"
        )

    ref_vals = _series(ref_rows[:n], args.component)
    mojo_vals = _series(mojo_rows[:n], args.component)

    if is_transient:
        x_vals = [dt * (i + 1) for i in range(n)]
        x_label = "Time (s)"
    else:
        x_vals = [i + 1 for i in range(n)]
        x_label = "Step"

    diffs = [g - r for r, g in zip(ref_vals, mojo_vals)]
    rmse = (sum(d * d for d in diffs) / max(len(diffs), 1)) ** 0.5
    ref_peak = _max_abs(ref_vals)
    mojo_peak = _max_abs(mojo_vals)
    peak_abs_err = abs(mojo_peak - ref_peak)
    peak_rel_err = peak_abs_err / max(ref_peak, 1.0e-30)

    single_sample = len(x_vals) == 1
    ref_style = {"label": "OpenSees reference", "linewidth": 2.0}
    mojo_style = {"label": "Strut mojo", "linewidth": 1.6, "linestyle": "--"}
    if single_sample:
        ref_style.update({"marker": "o", "markersize": 6.0})
        mojo_style.update({"marker": "x", "markersize": 6.0})

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_vals, ref_vals, **ref_style)
    ax.plot(x_vals, mojo_vals, **mojo_style)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Component {args.component}")
    ax.set_title(f"{args.case} :: {args.series_file}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = repo_root / "build" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.series_file.replace(".out", "")
        out_path = out_dir / f"{args.case}_{stem}_c{args.component}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)

    print(f"saved plot: {out_path}")
    print(f"rows compared: {n}")
    print(f"reference peak |max|: {ref_peak:.6e}")
    print(f"mojo peak |max|:      {mojo_peak:.6e}")
    print(f"peak abs error:       {peak_abs_err:.6e}")
    print(f"peak rel error:       {peak_rel_err:.6e}")
    print(f"RMSE:                 {rmse:.6e}")


if __name__ == "__main__":
    main()

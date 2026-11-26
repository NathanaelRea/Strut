#!/usr/bin/env python3
"""Build a shortened mega-tall benchmark input that OpenSees and Strut both run.

The full model (`dynamiccpu.tcl`) runs 10 gravity steps and 2000 transient steps,
which takes hours in OpenSees. This writes the same model with the stages cut to
`--gravity-steps` and `--transient-steps`, as a solver JSON that
`scripts/run_benchmarks.py --cases <output>` times on every engine. The three
whole-model displacement recorders are dropped; the base reactions, story
displacements and drifts stay.

`--tight` writes the variant used to check that the engines agree: the same
steps solved to `NormDispIncr 1e-7` with `UmfPack`. The model's own settings
(`NormDispIncr 0.05`, `SparseSYM`) leave OpenSees a few percent from its own
converged answer in the story displacements, because its `SparseSYM` is inexact
on rigid-diaphragm systems, so they cannot show whether two engines match.

Example:
  uv run scripts/build_megatall_benchmark_case.py
  uv run scripts/run_benchmarks.py --cases \
      benchmark/.tmp/megatall_model1_40steps/megatall_model1_40steps.json \
      --no-batch --strut-single-thread --no-archive

  uv run scripts/build_megatall_benchmark_case.py --tight
  uv run scripts/parity_report.py --strut-system SparseSYM \
      benchmark/.tmp/megatall_model1_40steps_tight/megatall_model1_40steps_tight.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import direct_tcl_support  # noqa: E402

FIXTURE_DIR = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "opensees"
    / "OpenSeesExamples"
    / "MegaTallBuildingBenchmark"
    / "Model1"
)
ENTRY_TCL = FIXTURE_DIR / "dynamiccpu.tcl"
GROUND_MOTION = FIXTURE_DIR / "GMX.txt"
# Whole-model displacement dumps (one column per node); not worth comparing.
WHOLE_MODEL_RECORDER_FILES = {"dispx.txt", "dispy.txt", "dispz.txt"}
# Settings for `--tight`, loose enough for OpenSees to reach in a few iterations.
TIGHT_TOLERANCE = 1.0e-7
TIGHT_MAX_ITERS = 30
TIGHT_SYSTEM = "UmfPack"


def build_case(gravity_steps: int, transient_steps: int, tight: bool = False) -> dict:
    if not ENTRY_TCL.is_file():
        raise SystemExit(
            f"mega-tall model not installed at {ENTRY_TCL}; run:\n"
            "  uv run scripts/setup_reference_data.py install --fixtures-only"
        )
    import tcl_to_strut

    scratch_root = REPO_ROOT / "benchmark" / ".tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    # The converter only reads inside the repo, so mirror the bundle there.
    with tempfile.TemporaryDirectory(prefix="megatall_", dir=scratch_root) as tmp:
        entry, _ = direct_tcl_support.prepare_direct_tcl_entry(
            ENTRY_TCL, [ENTRY_TCL], Path(tmp) / "context"
        )
        data = tcl_to_strut.convert_tcl_to_solver_input(entry, REPO_ROOT)

    stages = data["analysis"]["stages"]
    transient = [s for s in stages if s["analysis"]["type"] == "transient_nonlinear"]
    if len(stages) != 2 or len(transient) != 1:
        raise SystemExit("expected one gravity stage and one transient stage")
    full_gravity_steps = int(stages[0]["analysis"]["steps"])
    if not 0 < gravity_steps <= full_gravity_steps:
        raise SystemExit(f"--gravity-steps must be 1..{full_gravity_steps}")
    # Each gravity step keeps its load increment, so fewer steps stop at a lower load.
    stages[0]["analysis"]["steps"] = gravity_steps
    if transient_steps > 0:
        if gravity_steps != full_gravity_steps:
            raise SystemExit("the transient stage needs the full gravity load")
        transient[0]["analysis"]["steps"] = transient_steps
        name = f"megatall_model1_{gravity_steps + transient_steps}steps"
    else:
        data["analysis"]["stages"] = [stages[0]]
        name = f"megatall_model1_gravity{gravity_steps}"

    if tight:
        name += "_tight"
        for stage in data["analysis"]["stages"]:
            stage["analysis"]["tol"] = TIGHT_TOLERANCE
            stage["analysis"]["max_iters"] = TIGHT_MAX_ITERS
            stage["analysis"]["system"] = TIGHT_SYSTEM
    data["recorders"] = [
        recorder
        for recorder in data["recorders"]
        if recorder.get("raw_path") not in WHOLE_MODEL_RECORDER_FILES
    ]
    for series in data.get("time_series", []):
        if series.get("type") in ("Path", "PathFile") and "values_path" in series:
            series["values_path"] = str(GROUND_MOTION)
    for key in ("__strut_case_dir", "__strut_case_json_path", "source_example"):
        data.pop(key, None)
    data["enabled"] = True
    data["benchmark_size"] = "large"
    data["metadata"] = {
        **(data.get("metadata") or {}),
        "name": name,
        "notes": (
            f"Mega-tall Model1 cut to {gravity_steps} gravity and {transient_steps} "
            "transient steps"
            + (
                f", solved to NormDispIncr {TIGHT_TOLERANCE:g} with {TIGHT_SYSTEM}."
                if tight
                else ", with the model's own solver settings."
            )
        ),
    }
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gravity-steps", type=int, default=10)
    parser.add_argument(
        "--transient-steps", type=int, default=30, help="0 drops the transient stage."
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Tight tolerance and UmfPack, for checking that the engines agree.",
    )
    parser.add_argument("--output", default="", help="Output JSON path.")
    args = parser.parse_args()
    if args.transient_steps < 0:
        raise SystemExit("--transient-steps must not be negative")

    data = build_case(args.gravity_steps, args.transient_steps, args.tight)
    name = data["metadata"]["name"]
    output = (
        Path(args.output)
        if args.output
        # The benchmark runner accepts a case JSON only as <name>/<name>.json.
        else REPO_ROOT / "benchmark" / ".tmp" / name / f"{name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data), encoding="utf-8")
    counts = (len(data["nodes"]), len(data["elements"]), len(data["recorders"]))
    print(
        f"Wrote {output} ({counts[0]} nodes, {counts[1]} elements, {counts[2]} recorders)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

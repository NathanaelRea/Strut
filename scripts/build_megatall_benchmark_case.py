#!/usr/bin/env python3
"""Build a shortened mega-tall benchmark input that OpenSees and Strut both run.

The full model (`dynamiccpu.tcl`) runs 10 gravity steps and 2000 transient steps,
which takes hours in OpenSees. This writes the same model with the stages cut to
`--gravity-steps` and `--transient-steps`, as a solver JSON that
`scripts/run_benchmarks.py --cases <output>` times on every engine.

Three things differ from the converted Tcl, and all apply to every engine:

* `PlateFromPlaneStress` over `PlaneStressUserMaterial` becomes the
  `ElasticIsotropic` material with E = 2 G (1 + nu). The PSUMAT routine is not in
  the reference OpenSees build, and Strut and the parity harness already use
  this surrogate; writing it into the input makes that explicit.
* 3D fiber sections with no torsional rigidity get GJ = 1e-12. OpenSees 3.7
  rejects a 3D fiber section without one, and the parity harness injects the
  same value.
* Recorders are cut to the base reactions, story displacements and drifts. The
  three whole-model displacement recorders are dropped; benchmark runs strip
  recorders anyway.

Example:
  uv run scripts/build_megatall_benchmark_case.py
  uv run scripts/run_benchmarks.py --cases \
      benchmark/.tmp/megatall_model1_40steps/megatall_model1_40steps.json \
      --no-batch --strut-single-thread --no-archive
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
    REPO_ROOT / "tests" / "fixtures" / "opensees" / "OpenSeesExamples"
    / "MegaTallBuildingBenchmark" / "Model1"
)
ENTRY_TCL = FIXTURE_DIR / "dynamiccpu.tcl"
GROUND_MOTION = FIXTURE_DIR / "GMX.txt"
# Whole-model displacement dumps (one column per node); not worth comparing.
WHOLE_MODEL_RECORDER_FILES = {"dispx.txt", "dispy.txt", "dispz.txt"}
# Matches the -GJ the parity harness gives OpenSees for sections that omit it.
MISSING_TORSION_GJ = 1.0e-12


def _elastic_surrogates(materials: list[dict]) -> list[dict]:
    by_id = {int(material["id"]): material for material in materials}
    user_ids = {
        material_id
        for material_id, material in by_id.items()
        if material["type"] == "PlaneStressUserMaterial"
    }
    out = []
    for material in materials:
        material_id = int(material["id"])
        if material_id in user_ids:
            continue
        base_id = material.get("params", {}).get("material")
        if material["type"] == "PlateFromPlaneStress" and base_id in user_ids:
            base = by_id[base_id]["params"]
            nu = float(base["props"][-1])
            gmod = float(material["params"]["gmod"])
            out.append(
                {
                    "id": material_id,
                    "type": "ElasticIsotropic",
                    "params": {
                        "E": 2.0 * gmod * (1.0 + nu),
                        "nu": nu,
                        "rho": float(base.get("rho", 0.0)),
                    },
                }
            )
            continue
        out.append(material)
    return out


def build_case(gravity_steps: int, transient_steps: int) -> dict:
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

    data["materials"] = _elastic_surrogates(data["materials"])
    for section in data["sections"]:
        params = section["params"]
        if section["type"] == "FiberSection3d" and not (
            params.get("G", 0.0) > 0.0 and params.get("J", 0.0) > 0.0
        ):
            params["G"] = MISSING_TORSION_GJ
            params["J"] = 1.0
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
            "transient steps. Every engine runs the same input, but the results are "
            "not at parity: base reactions agree with OpenSees within 0.6% of the "
            "peak reaction, lateral story displacements do not."
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
    parser.add_argument("--output", default="", help="Output JSON path.")
    args = parser.parse_args()
    if args.transient_steps < 0:
        raise SystemExit("--transient-steps must not be negative")

    data = build_case(args.gravity_steps, args.transient_steps)
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
    print(f"Wrote {output} ({counts[0]} nodes, {counts[1]} elements, {counts[2]} recorders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

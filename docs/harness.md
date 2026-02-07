# Strut Harness

This document describes the parity and benchmark harness used to compare Strut (Mojo) against OpenSees (C++/Tcl).

## Goals

- Canonical model definition in JSON.
- Deterministic conversion to Tcl for OpenSees.
- Consistent, machine-readable outputs for comparison.
- Repeatable benchmarks with stored results.

## Canonical Model Format

The canonical model is JSON, versioned via `schema_version`.

Minimum required fields (v1.0):

- `schema_version`: string, e.g. `"1.0"`
- `metadata`: `{ name, units }`
- `model`: `{ ndm, ndf }`
  - DOF order for `ndf=6`: `1..6 = ux, uy, uz, rx, ry, rz` (OpenSees standard).
- `nodes`: list of `{ id, x, y, z?, constraints? }`
  - `z` is required when `ndm=3`.
  - `constraints` DOF indices must be in `1..ndf`.
- `materials`: list of `{ id, type, params }` (elastic only in v1)
- `sections`: list of `{ id, type, params }` (elastic section only in v1)
- `elements`: list of `{ id, type, nodes, section, geomTransf }`
- `time_series`: list of `{ type, tag, ... }` (optional; top-level)
  - `Constant`: `{ tag, factor? }`
  - `Linear`: `{ tag, factor? }`
  - `Path`: `{ tag, values, dt? | time?, factor?, use_last? }`
  - `Trig`: `{ tag, t_start, t_finish, period, phase_shift?, factor?, zero_shift? }`
- `pattern`: `{ type: "Plain", tag, time_series }` (optional; top-level)
- `loads`: list of `{ node, dof, value }` (`dof` must be in `1..ndf`)
- `element_loads`: list of `{ element, type, w }` (optional, `type: "beamUniform"` only)
- `analysis`: `{ type: "static_linear" | "static_nonlinear", steps: 1, max_iters?, tol?, rel_tol? }`
- `recorders`: list of
  - `{ type: "node_displacement", nodes, dofs, output }` (`dofs` in `1..ndf`)
  - `{ type: "element_force", elements, output }` (currently `elasticBeamColumn2d` only)

## Harness Workflow

1. `scripts/json_to_tcl.py` converts a JSON model to a deterministic Tcl script.
2. `scripts/run_case.py` runs OpenSees (Wine) to produce reference outputs.
3. `scripts/run_mojo_case.py` runs the current Strut implementation and writes outputs.
4. `scripts/compare_case.py` compares displacement outputs with tolerances.
5. `run_tests.py` provides a unified test runner for unit, schema, and parity checks.

## Output Format

Node displacement outputs are written as space-separated values with one row per analysis step:

```
0.0000000000000000e+00 -1.6666666666666667e-05 -2.5000000000000000e-05
```

For OpenSees, the Tcl recorder writes a space-separated vector per line. The comparator reads the last line for each node.

Element force outputs are written as space-separated values with one row per analysis step. For `elasticBeamColumn2d`,
the vector contains 6 global end forces (3 at node 1, 3 at node 2) in the OpenSees "force" recorder ordering.

## Tolerances (Phase 1)

- Absolute tolerance: `1e-9`
- Relative tolerance: `1e-6`

## Notes

- Phase 1 targets 2D linear `elasticBeamColumn` with static linear analysis.
- The current phase-1 solver is implemented in Python for harness validation. Replace with Mojo and wire `strut.mojo` into the harness once the Mojo implementation is ready.
- Expand schema and harness only after parity is stable.
- For static analyses with time series, Strut uses normalized pseudoTime `t = (step + 1) / steps` (static_linear uses `t = 1.0`).
- Validation cases can be marked with `"enabled": false` in JSON. `STRUT_RUN_ALL_CASES=1` runs all cases in tests, and `STRUT_FORCE_CASE=1` forces a disabled case in `scripts/run_case.py`.
- `scripts/run_mojo_case.py` runs the Python solver by default and only uses Mojo when `STRUT_MOJO_SOLVER=1` and `mojo` is on `PATH`.
- OpenSees reference outputs are cached by JSON content hash in `tests/validation/<case>/reference/.ref_hash`. Set `STRUT_REFRESH_REFERENCE=1` to regenerate.
- Benchmarks use `scripts/run_benchmarks.py`. The latest run is written to `benchmark/results/` (`benchmark/results-profile/` when `--profile`), and summary snapshots are archived in `benchmark/archive/` (both ignored by git). Runs with `--profile` default to `--no-archive`.

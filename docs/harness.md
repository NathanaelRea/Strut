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
- `nodes`: list of `{ id, x, y, constraints? }`
- `materials`: list of `{ id, type, params }` (elastic only in v1)
- `sections`: list of `{ id, type, params }` (elastic section only in v1)
- `elements`: list of `{ id, type, nodes, section, geomTransf }`
- `loads`: list of `{ node, dof, value }`
- `analysis`: `{ type: "static_linear", steps: 1 }`
- `recorders`: list of `{ type: "node_displacement", nodes, dofs, output }`

## Harness Workflow

1. `scripts/json_to_tcl.py` converts a JSON model to a deterministic Tcl script.
2. `scripts/run_case.sh` runs OpenSees (Wine) to produce reference outputs.
3. `scripts/run_mojo_case.py` runs the current Strut implementation and writes outputs.
4. `scripts/compare_case.py` compares displacement outputs with tolerances.
5. `scripts/run_tests.py` provides a unified test runner for unit, schema, and parity checks.

## Output Format

Node displacement outputs are written as space-separated values with one row per analysis step:

```
0.0000000000000000e+00 -1.6666666666666667e-05 -2.5000000000000000e-05
```

For OpenSees, the Tcl recorder writes a space-separated vector per line. The comparator reads the last line for each node.

## Tolerances (Phase 1)

- Absolute tolerance: `1e-9`
- Relative tolerance: `1e-6`

## Notes

- Phase 1 targets 2D linear `elasticBeamColumn` with static linear analysis.
- The current phase-1 solver is implemented in Python for harness validation. Replace with Mojo and wire `strut.mojo` into the harness once the Mojo implementation is ready.
- Expand schema and harness only after parity is stable.
- Validation cases can be marked with `"enabled": false` in JSON. `STRUT_RUN_ALL_CASES=1` runs all cases in tests, and `STRUT_FORCE_CASE=1` forces a disabled case in `scripts/run_case.sh`.

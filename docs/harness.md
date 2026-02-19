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
- `materials`: list of `{ id, type, params }` (`Elastic`, `ElasticIsotropic`, `Steel01`, `Steel02`, `Concrete01`, `Concrete02`)
- `sections`: list of `{ id, type, params }`
  - Elastic: `ElasticSection2d`, `ElasticSection3d`, `ElasticMembranePlateSection`
  - Fiber (infrastructure): `FiberSection2d`
    - `params.patches`: list of patches:
      - `rect`: `{ type: "rect", material, num_subdiv_y, num_subdiv_z, y_i, z_i, y_j, z_j }`
      - `quadr`: `{ type: "quadr", material, num_subdiv_y, num_subdiv_z, y_i, z_i, y_j, z_j, y_k, z_k, y_l, z_l }`
    - `params.layers`: list of `straight` layers:
      - `{ type: "straight", material, num_bars, bar_area, y_start, z_start, y_end, z_end }`
- `elements`: list of `{ id, type, nodes, section, geomTransf }`
- `mp_constraints`: list of `{ type: "equalDOF", retained_node, constrained_node, dofs }` (optional; requires transformation handler)
- `time_series`: list of `{ type, tag, ... }` (optional; top-level)
  - `Constant`: `{ tag, factor? }`
  - `Linear`: `{ tag, factor? }`
  - `Path`: `{ tag, values | values_path | path, dt? | time?, factor?, use_last? }`
    - `values_path`/`path` may be absolute or relative to the case JSON location.
  - `PathFile` (alias): same fields as `Path`, emitted/evaluated as `Path`.
  - `Trig`: `{ tag, t_start, t_finish, period, phase_shift?, factor?, zero_shift? }`
- `pattern`: (optional; top-level)
  - `Plain`: `{ type: "Plain", tag, time_series }`
  - `UniformExcitation`: `{ type: "UniformExcitation", tag, direction, accel }`
- `loads`: list of `{ node, dof, value }` (`dof` must be in `1..ndf`)
- `element_loads`: list of `{ element, type, wy | w, wx? }` (optional, `type: "beamUniform"` only)
  - `wy` (or legacy alias `w`): local y-direction uniform load intensity.
  - `wx` (optional): local x-direction uniform load intensity.
- `rayleigh`: `{ alphaM?, betaK?, betaKInit?, betaKComm? }` (optional; top-level)
- `analysis`: `{ type: "static_linear" | "static_nonlinear" | "transient_linear" | "transient_nonlinear" | "modal_eigen" | "staged", steps: 1, constraints?, num_modes?, dt?, max_iters?, tol?, rel_tol?, integrator?, algorithm?, stages? }`
  - `constraints`: `Plain` (default) or `Transformation`.
  - Nonlinear uniaxial materials require `static_nonlinear` or `transient_nonlinear`.
  - `modal_eigen` requires `num_modes >= 1` and positive nodal masses on free DOFs.
  - `static_nonlinear` integrator options:
    - Load control: `{ type: "LoadControl" }` (default)
    - Displacement control: `{ type: "DisplacementControl", node, dof, du? | targets?, cutback?, max_cutbacks?, min_du? }`
    - `algorithm`: `"Newton"` (default) or `"ModifiedNewton"`
  - `transient_linear` requires `dt > 0` and supports `integrator: { type: "Newmark", gamma?, beta? }`
  - `transient_nonlinear` requires `dt > 0` and supports:
    - `integrator: { type: "Newmark", gamma?, beta? }`
    - `algorithm: "Newton"` (default), `"ModifiedNewton"`, `"ModifiedNewtonInitial"`, `"Broyden"`, or `"NewtonLineSearch"` (`Broyden`/`NewtonLineSearch` currently map to Newton tangent refresh behavior in Mojo runtime)
    - `test_type: "MaxDispIncr"` (default), `"NormDispIncr"`, `"NormUnbalance"`, or `"EnergyIncr"`
    - optional fallback controls:
      - `fallback_algorithm: "Newton" | "ModifiedNewton" | "ModifiedNewtonInitial" | "Broyden" | "NewtonLineSearch"`
      - `fallback_test_type`, `fallback_tol`, `fallback_rel_tol`, `fallback_max_iters`
  - `staged` analysis:
    - `analysis.stages`: non-empty list of stage objects
    - stage fields: `analysis` (required), optional `pattern`, `time_series`, `loads`, `element_loads`, `rayleigh`, `load_const`
    - stage `analysis.type` supports `static_linear`, `static_nonlinear`, `transient_linear`, `transient_nonlinear`, `modal_eigen`
- `masses`: list of `{ node, dof, value }` (optional; nodal lumped masses for dynamics)
- `recorders`: list of
  - `{ type: "node_displacement", nodes, dofs, output }` (`dofs` in `1..ndf`)
  - `{ type: "element_force", elements, output }` (`truss`, `elasticBeamColumn2d`, `forceBeamColumn2d`, `dispBeamColumn2d`)
  - `{ type: "node_reaction", nodes, dofs, output }` (`dofs` in `1..ndf`)
  - `{ type: "drift", i_node, j_node, dof, perp_dirn, output }`
  - `{ type: "envelope_element_force", elements, output }` (`truss`, `elasticBeamColumn2d`, `forceBeamColumn2d`, `dispBeamColumn2d`)
  - `{ type: "section_force", elements, section | sections, output }` (`forceBeamColumn2d`, `dispBeamColumn2d`)
  - `{ type: "section_deformation", elements, section | sections, output }` (`forceBeamColumn2d`, `dispBeamColumn2d`)
  - `{ type: "modal_eigen", modes, nodes, dofs, output }`
- `parity_mode`: `"step"` (default) or `"max_abs"` for transient comparisons (optional; top-level)
  - `step`: strict step-by-step time-history comparison.
  - `max_abs`: compare component-wise peak absolute response across the full history.
- `benchmark_size`: `"small" | "medium" | "large"` (optional; top-level benchmark/plot override)

Current limitation: `forceBeamColumn2d`/`dispBeamColumn2d` support is currently limited to:
- `geomTransf: Linear | PDelta`
- `integration: Lobatto`
- `num_int_pts: 3 | 5`
- `analysis.type: static_linear | static_nonlinear | transient_nonlinear | staged`

## Harness Workflow

1. `scripts/json_to_tcl.py` converts a JSON model to a deterministic Tcl script.
2. `scripts/run_case.py` runs OpenSees (Wine) to produce reference outputs.
3. `scripts/run_mojo_case.py` runs the current Strut implementation and writes outputs.
4. `scripts/compare_case.py` compares displacement outputs with tolerances.
5. `run_tests.py` provides a unified test runner for unit, schema, and parity checks.
6. `scripts/run_and_plot_case.py <case.json>` runs a case and generates overlay plots for all comparable `.out` files/components.

## Reproduction Commands

- Run Ex2c staged parity case end-to-end:
  - `STRUT_FORCE_CASE=1 uv run scripts/run_case.py tests/validation/opensees_example_ex2c_canti2d_inelastic_fiber/opensees_example_ex2c_canti2d_inelastic_fiber.json`
- Re-run parity check only:
  - `uv run scripts/compare_case.py --case opensees_example_ex2c_canti2d_inelastic_fiber`
- Run per-case benchmark (no batch, no archive):
  - `uv run scripts/run_benchmarks.py --cases opensees_example_ex2c_canti2d_inelastic_fiber --include-disabled --engine mojo --repeat 1 --warmup 0 --no-batch --no-archive`

## Output Format

Node displacement outputs are written as space-separated values with one row per analysis step (transient uses one line per time step):

```
0.0000000000000000e+00 -1.6666666666666667e-05 -2.5000000000000000e-05
```

For OpenSees, the Tcl recorder writes a space-separated vector per line. The comparator reads the last line for static cases and all lines for transient cases.

Element force outputs are written as space-separated values with one row per analysis step. For `elasticBeamColumn2d`, `forceBeamColumn2d`, and `dispBeamColumn2d`,
the vector contains 6 global end forces (3 at node 1, 3 at node 2) in the OpenSees "force" recorder ordering.

Node reaction outputs are written as space-separated values with one row per analysis step. Each row contains requested reaction DOFs at the requested node.

Drift outputs are written as one scalar per row (one row per analysis step).

Envelope element force outputs are written with 3 rows:
- row 1: component-wise minima
- row 2: component-wise maxima
- row 3: component-wise absolute maxima

Modal eigen outputs are written as:
- `<output>_eigenvalues.out`: one eigenvalue per line (ascending mode order).
- `<output>_mode<k>_node<id>.out`: one line with requested DOF components of mode `k` at node `id`.

## Tolerances (Phase 1)

- Absolute tolerance: `1e-9`
- Relative tolerance: `1e-6`

## Notes

- Phase 1 targets 2D linear `elasticBeamColumn` with static linear analysis.
- The current phase-1 solver is implemented in Python for harness validation. Replace with Mojo and wire `strut.mojo` into the harness once the Mojo implementation is ready.
- Expand schema and harness only after parity is stable.
- For static analyses with time series, Strut uses normalized pseudoTime `t = (step + 1) / steps` (static_linear uses `t = 1.0`).
- Validation parity tests run only cases with `"enabled": true` by default (`STRUT_RUN_ALL_CASES=1` includes disabled cases). `STRUT_FORCE_CASE=1` forces a disabled case in `scripts/run_case.py`.
- Recorder entries may set `"parity": false` to generate outputs but exclude that recorder from parity checks.
- Benchmarks include `"enabled": true` cases and also disabled cases tagged with `"status": "benchmark"`.
- `scripts/run_mojo_case.py` runs the Python solver by default and only uses Mojo when `STRUT_MOJO_SOLVER=1` and `mojo` is on `PATH`.
- OpenSees reference outputs are cached by JSON content hash in `tests/validation/<case>/reference/.ref_hash`. Set `STRUT_REFRESH_REFERENCE=1` to regenerate.
- Benchmarks use `scripts/run_benchmarks.py`. The latest run is written to `benchmark/results/` (`benchmark/results-profile/` when `--profile`), and summary snapshots are archived in `benchmark/archive/` (both ignored by git). Runs with `--profile` default to `--no-archive`.

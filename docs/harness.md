# Strut Harness

This document describes the parity and benchmark harness used to compare Strut (Mojo) against OpenSees (C++/Tcl).

## Goals

- Canonical model definition in JSON.
- Deterministic conversion to Tcl for OpenSees.
- Consistent, machine-readable outputs for comparison.
- Repeatable benchmarks with stored results.

Phase-0 parity contract artifacts:

- Beam-column parity contract: `docs/beam_column_parity_contract.md`
- Validation case matrix: `tests/validation/PHASE0_CASE_MATRIX.md`

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
  - Fiber (infrastructure): `FiberSection2d`, `FiberSection3d`
    - `FiberSection3d` beam-column runtime use requires positive `params.G` and `params.J` for torsion stiffness.
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
- `element_loads`: optional list of element loads.
  2D: `beamUniform { element, type, wy | w, wx? }`, `beamPoint { element, type, py, x, px? }`
  3D: `beamUniform { element, type, wy, wz, wx? }`, `beamPoint { element, type, py, pz, x, px? }`
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
    - `algorithm`: `"Newton"` (default), `"ModifiedNewton"`, or `"ModifiedNewtonInitial"`
    - `test_type`: `"MaxDispIncr"` (default), `"NormDispIncr"`, `"NormUnbalance"`, or `"EnergyIncr"`
    - optional fallback controls:
      - `fallback_algorithm: "Newton" | "ModifiedNewton" | "ModifiedNewtonInitial"`
      - `fallback_test_type`, `fallback_tol`, `fallback_rel_tol`, `fallback_max_iters`
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
  - `{ type: "element_force", elements, output }` (`truss`, `elasticBeamColumn2d`, `elasticBeamColumn3d`, `forceBeamColumn2d`, `forceBeamColumn3d`, `dispBeamColumn2d`, `dispBeamColumn3d`)
  - `{ type: "node_reaction", nodes, dofs, output }` (`dofs` in `1..ndf`)
  - `{ type: "drift", i_node, j_node, dof, perp_dirn, output }`
  - `{ type: "envelope_element_force", elements, output }` (`truss`, `elasticBeamColumn2d`, `elasticBeamColumn3d`, `forceBeamColumn2d`, `forceBeamColumn3d`, `dispBeamColumn2d`, `dispBeamColumn3d`)
  - `{ type: "section_force", elements, section | sections, output }` (`forceBeamColumn2d`, `forceBeamColumn3d`, `dispBeamColumn2d`, `dispBeamColumn3d`)
  - `{ type: "section_deformation", elements, section | sections, output }` (`forceBeamColumn2d`, `forceBeamColumn3d`, `dispBeamColumn2d`, `dispBeamColumn3d`)
  - `{ type: "modal_eigen", modes, nodes, dofs, output }`
- `parity_mode`: `"step"` (default) or `"max_abs"` for transient comparisons (optional; top-level)
  - `step`: strict step-by-step time-history comparison.
  - `max_abs`: compare component-wise peak absolute response across the full history.
- `benchmark_size`: `"small" | "medium" | "large"` (optional; top-level benchmark/plot override)

Current limitation: `forceBeamColumn2d`/`dispBeamColumn2d` support is currently limited to:
- `geomTransf: Linear | PDelta`
- `integration: Lobatto | Legendre | Radau`
- `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
- `analysis.type: static_linear | static_nonlinear | transient_nonlinear | staged`

Current limitation: `forceBeamColumn3d`/`dispBeamColumn3d` support is currently limited to:
- `section: ElasticSection3d | FiberSection3d`
- `geomTransf: Linear | PDelta | Corotational`
- `integration: Lobatto | Legendre | Radau`
- `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
- `FiberSection3d` runtime use requires positive `G` and `J` in section params.
- 3D section recorders are available for `section_force` (`N`, `Mz`, `My`, `T`) and `section_deformation` (`eps0`, `kappa_z`, `kappa_y`, `twist`), matching OpenSees recorder ordering.

## Harness Workflow

1. JSON-authored cases use `scripts/json_to_tcl.py` to convert JSON models to deterministic Tcl when OpenSees reference generation is needed.
2. Direct-Tcl cases use the original reference Tcl for OpenSees and parse that same Tcl directly into Strut input.
3. `scripts/run_case.py` runs OpenSees (Wine) to produce reference outputs, then runs Strut on either JSON input or direct Tcl input.
4. `scripts/run_strut_case.py` runs the current Strut implementation and writes outputs.
5. `scripts/compare_case.py` compares recorder outputs with recorder-specific tolerances and can load direct-Tcl parity metadata straight from the Tcl parser.
6. `uv run run_tests.py` builds the solver, then runs unit, schema, and parity checks.
7. `scripts/run_and_plot_case.py <case.json>` runs a JSON-authored case and generates overlay plots for all comparable `.out` files/components.

## Reproduction Commands

- Run Ex2c staged parity case end-to-end:
  - `STRUT_FORCE_CASE=1 uv run scripts/run_case.py tests/validation/opensees_example_ex2c_canti2d_inelastic_fiber/direct_tcl_case.json`
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

Element force outputs are written as space-separated values with one row per analysis step.
For `elasticBeamColumn2d`, `forceBeamColumn2d`, and `dispBeamColumn2d`, the vector contains 6 global end forces (3 at node 1, 3 at node 2) in the OpenSees "force" recorder ordering.
For `elasticBeamColumn3d`, `forceBeamColumn3d`, and `dispBeamColumn3d`, the vector contains 12 global end forces (6 at node 1, 6 at node 2).

Node reaction outputs are written as space-separated values with one row per analysis step. Each row contains requested reaction DOFs at the requested node.

Drift outputs are written as one scalar per row (one row per analysis step).

Envelope element force outputs are written with 3 rows:
- row 1: component-wise minima
- row 2: component-wise maxima
- row 3: component-wise absolute maxima

Modal eigen outputs are written as:
- `<output>_eigenvalues.out`: one eigenvalue per line (ascending mode order).
- `<output>_mode<k>_node<id>.out`: one line with requested DOF components of mode `k` at node `id`.

## Tolerances (Phase 0 contract)

Default recorder tolerances in `scripts/compare_case.py`:

- `node_displacement`: `atol=1e-9`, `rtol=1e-6`
- `node_reaction`: `atol=1e-9`, `rtol=1e-6`
- `drift`: `atol=1e-9`, `rtol=1e-6`
- `element_force`: `atol=1e-8`, `rtol=1e-5`
- `envelope_element_force`: `atol=1e-8`, `rtol=1e-5`
- `section_force`: `atol=1e-8`, `rtol=1e-5`
- `section_deformation`: `atol=1e-9`, `rtol=1e-6`
- `modal_eigen`: `atol=1e-8`, `rtol=1e-5`

Per-case overrides:

- Global override: `parity_tolerance: { atol, rtol }`
- Recorder-specific override: `parity_tolerance_by_recorder: { "<recorder_type>": { atol, rtol } }`

## Notes

- For static analyses with time series, Strut uses normalized pseudoTime `t = (step + 1) / steps` (static_linear uses `t = 1.0`).
- Validation parity tests run only cases with `"enabled": true` by default (`STRUT_RUN_ALL_CASES=1` includes disabled cases). `STRUT_FORCE_CASE=1` forces a disabled case in `scripts/run_case.py`.
- Recorder entries may set `"parity": false` to generate outputs but exclude that recorder from parity checks.
- Benchmarks include `"enabled": true` cases and also disabled cases tagged with `"status": "benchmark"`.
- OpenSees reference outputs are cached by JSON content hash in `tests/validation/<case>/reference/.ref_hash`. Set `STRUT_REFRESH_REFERENCE=1` to regenerate.
- Benchmarks use `scripts/run_benchmarks.py`. The latest run is written to `benchmark/results/` (`benchmark/results-profile/` when `--profile`), and summary snapshots are archived in `benchmark/archive/` (both ignored by git). Runs with `--profile` default to `--no-archive`.

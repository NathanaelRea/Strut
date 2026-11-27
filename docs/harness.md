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
- `materials`: list of `{ id, type, params }` (`Elastic`, `ElasticIsotropic`, `Steel01`, `Steel02`, `Concrete01`, `Concrete02`, `ENT`, `Parallel`)
  - `HystereticSmooth` takes `ka`, `kb`, `fbar`, and `beta`. `HystereticAsym` takes `ka`, `kb`, `fbar`, `beta1`, `beta2`, and `gamma`.
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
  - `MVLEM`: `{ id, type, nodes, density, c, shear_material, fibers }`.
    Each fiber contains `width`, `thickness`, `rho`, `concrete`, and `steel`.
    The element connects two 2D, 3-DOF nodes ordered vertically from bottom to top.
  - `tri31` uses three counterclockwise nodes, `thickness`, `material` referencing an `ElasticIsotropic` material, and `formulation: "PlaneStress" | "PlaneStrain"`. Its supported Tcl form takes no optional element pressure, density override, or body-force arguments.
  - The direct Tcl reader supports `quadUP`, `9_4_QuadUP`, and `SSPbrickUP` in transient analyses. Corner node velocity on DOF 3 in 2D or DOF 4 in 3D gives pore pressure. The five midside and center nodes of `9_4_QuadUP` have translational DOFs only.
  - In a plane-strain `quad` domain, `InitialStateAnalysisWrapper` can wrap a 2D multi-yield soil material. `InitialStateAnalysis on` followed by `off` resets nodal response and time and applies the captured strain at each material point in the next stage.
  - A Tcl `recorder` returns its tag. `remove recorder $tag` ends that recorder at the current stage boundary, including all nodes or elements covered by the Tcl command.
  - `remove element` and `remove node` rebuild the active equations between static `LoadControl` stages of a `quad` domain without MPCs. Remove attached elements and their recorders before removing a node.
- `mp_constraints`: optional list of multi-point constraints; requires `analysis.constraints: "Transformation"` or `"Lagrange"`
  - `equalDOF`: `{ type: "equalDOF", retained_node, constrained_node, dofs }`
  - `rigidDiaphragm`: `{ type: "rigidDiaphragm", retained_node, constrained_node, perp_dirn, constrained_dofs, retained_dofs, matrix, dx, dy, dz }`
    - This is the canonical JSON/runtime shape. It matches the direct-Tcl converter output exactly.
    - Native runtime storage keeps this external JSON schema unchanged and normalizes `matrix` into fixed-size internal coefficient rows.
    - Supported OpenSees forms:
      - `ndm=3`, `ndf=6`
      - `ndm=2`, `ndf=3`
    - Canonical DOF mappings already emitted by the converter:
      - 3D `perp_dirn=3` -> `constrained_dofs = retained_dofs = [1, 2, 6]`
      - 3D `perp_dirn=2` -> `constrained_dofs = retained_dofs = [1, 3, 5]`
      - 3D `perp_dirn=1` -> `constrained_dofs = retained_dofs = [2, 3, 4]`
      - 2D `perp_dirn=3` -> `constrained_dofs = retained_dofs = [1, 2, 3]`
      - 2D `perp_dirn=1` -> `constrained_dofs = retained_dofs = [1]`
      - 2D `perp_dirn=2` -> `constrained_dofs = retained_dofs = [2]`
    - `dx`, `dy`, and `dz` are retained as canonical converter/runtime fields for diagnostics and parity work.
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
  - Regional overrides use `node_alpha_m` for nodal lumped mass and
    `element_beta_k_init` for the initial stiffness of 2D elastic beams. Each
    list contains `{ tag, coefficient }` entries. Staged analysis accepts
    these overrides in a stage's `rayleigh` object.
- `analysis`: `{ type: "static_linear" | "static_nonlinear" | "transient_linear" | "transient_nonlinear" | "modal_eigen" | "staged", steps: 1, constraints?, num_modes?, dt?, max_iters?, tol?, rel_tol?, integrator?, algorithm?, stages? }`
  - `constraints`: `Plain` (default) or `Transformation`.
  - Nonlinear uniaxial materials require `static_nonlinear` or `transient_nonlinear`.
  - `modal_eigen` requires `num_modes >= 1` and positive nodal masses on free DOFs.
  - `static_nonlinear` integrator options:
    - Load control: `{ type: "LoadControl" }` (default)
    - Displacement control: `{ type: "DisplacementControl", node, dof, du? | targets?, cutback?, max_cutbacks?, min_du? }`
    - `algorithm`: `"Newton"` (default), `"ModifiedNewton"`, or `"ModifiedNewtonInitial"`
    - `test_type`: `"NormUnbalance"` (default), `"NormDispIncr"`, `"NormUnbalance"`, or `"EnergyIncr"`
    - optional fallback controls:
      - `fallback_algorithm: "Newton" | "ModifiedNewton" | "ModifiedNewtonInitial"`
      - `fallback_test_type`, `fallback_tol`, `fallback_rel_tol`, `fallback_max_iters`
  - `transient_linear` requires `dt > 0` and supports `integrator: { type: "Newmark", gamma?, beta? }`
  - `transient_nonlinear` requires `dt > 0` and supports:
    - `integrator: { type: "Newmark", gamma?, beta? }`
    - `algorithm: "Newton"` (default), `"ModifiedNewton"`, `"ModifiedNewtonInitial"`, `"Broyden"`, or `"NewtonLineSearch"` (`Broyden`/`NewtonLineSearch` currently map to Newton tangent refresh behavior in Mojo runtime)
    - `test_type: "NormUnbalance"` (default), `"NormDispIncr"`, `"NormUnbalance"`, or `"EnergyIncr"`
    - optional fallback controls:
      - `fallback_algorithm: "Newton" | "ModifiedNewton" | "ModifiedNewtonInitial" | "Broyden" | "NewtonLineSearch"`
      - `fallback_test_type`, `fallback_tol`, `fallback_rel_tol`, `fallback_max_iters`
  - `staged` analysis:
    - `analysis.stages`: non-empty list of stage objects
    - stage fields: `analysis` (required), optional `pattern`, `time_series`, `loads`, `element_loads`, `rayleigh`, `load_const`
    - `constant_loads` and `constant_element_loads` apply at unit scale alongside the stage pattern.
    - `sample_before` identifies zero-based node displacement recorders to sample from the committed state before the stage.
    - `resume_load_control` carries the last load increment and iteration count into the next stage when Tcl reuses the same integrator.
    - Displacement control can use `displacement_limit` to stop between steps once the controlled node reaches the limit.
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
  - In staged cases, `start_stage` optionally sets the zero-based index at
    which a recorder starts collecting output.
- `parity_mode`: `"step"` (default) or `"max_abs"` (optional; top-level)
  - `step`: compare each transient step and the final static step. Set `parity_history` to `true` to compare each static step.
  - `max_abs`: compare component-wise peak absolute response across the full history.
- `benchmark_size`: `"small" | "medium" | "large"` (optional; top-level benchmark/plot override)

Current limitation: `forceBeamColumn2d`/`dispBeamColumn2d` support is currently limited to:
- `geomTransf: Linear | PDelta`
- `integration: Lobatto | Legendre | Radau`
- `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
- `analysis.type: static_linear | static_nonlinear | transient_nonlinear | staged`

The 2D Tcl `beamWithHinges` converter uses the six-point HingeRadau rule.
The first and last points take the end sections, and the interior points take
the elastic member section. The end sections support independent `P` and
`Mz` responses through `section Aggregator`.

Current limitation: `forceBeamColumn3d`/`dispBeamColumn3d` support is currently limited to:
- `section: ElasticSection3d | FiberSection3d`
- `geomTransf: Linear | PDelta | Corotational`
- `integration: Lobatto | Legendre | Radau`
- `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
- `FiberSection3d` runtime use requires positive `G` and `J` in section params.
- 3D section recorders are available for `section_force` (`N`, `Mz`, `My`, `T`) and `section_deformation` (`eps0`, `kappa_z`, `kappa_y`, `twist`), matching OpenSees recorder ordering. For a 3D section with an additional elastic torsion response from `section Aggregator`, the final recorder column contains that response. Force-based elements add the two torsional flexibilities and displacement-based elements add the two stiffnesses.

## Harness Workflow

1. JSON-authored cases use `scripts/json_to_tcl.py` to convert JSON models to deterministic Tcl when OpenSees reference generation is needed.
2. Direct-Tcl cases stage the Tcl bundle for OpenSees and parse the same staged Tcl into Strut input. A `direct_tcl_case.json` can set `"tcl_source_version": "3.7.1"` next to `entry_tcl`. For the legacy wiki syntax, staging converts inline `Series` assignments into tagged `timeSeries Path` commands for OpenSees 3.8.0.
3. `scripts/run_case.py` runs native Linux OpenSees to produce reference outputs, then runs Strut on either JSON input or direct Tcl input.
4. `scripts/run_strut_case.py` runs the current Strut implementation and writes outputs.
5. `scripts/compare_case.py` compares recorder outputs with recorder-specific tolerances and can load direct-Tcl parity metadata straight from the Tcl parser.
6. `uv run run_tests.py` builds the solver, then runs unit, schema, and parity checks.
7. `scripts/run_and_plot_case.py <case.json>` runs a JSON-authored case and generates overlay plots for all comparable `.out` files/components.

### Direct-Tcl case manifests

- `entry_tcl` is the repository-relative path to the entry script.
The PlanarTruss verification manifests run the two published domains independently.
Each selects its model through `tcl_prelude` and compares its nodal displacement
and element forces through file recorders.

- `source_url` identifies the published model. Use a versioned URL when the upstream source has a pinned tag or commit.
- `reference` records the publication or model citation when available.
- `enabled` controls inclusion in the default suite. Use `disabled_reason` to record the features or parity checks that block a disabled case.
- `tcl_source_version` identifies the Tcl dialect used by a published script
  when it needs an upgrade. `tcl_upgrade_note` records the rewrites that apply
  to that case.
- `replay_tcl` opts a script into multi-domain parity. Use `"domains"` for a
  sequence of independent models and `"live"` when Tcl branches on the
  result of `analyze` or a solver query.
- `recorder_sample_dt` applies a positive `-dT` interval to line-based
  `recorder Node` and `recorder Element` commands in the mirrored scripts.
  Both engines use the sampled histories while the original analysis steps
  and time increments are preserved.
- `transient_step_cap` sets a `step_variable` and positive `max_steps`.
  Both engines cap that variable immediately before the entry script first
  calls `analyze` with it. The script computes its time increment and runs
  earlier analysis phases as published. Any subsequent retry code sees the
  same capped step count.

Compatibility changes run on the harness copy. Recorder instrumentation can capture results the upstream script reports through the console.

### Tcl dialect upgrades

`scripts/direct_tcl_dialects.py` registers each upgrade by its source dialect
and the next version on the path to the pinned binary. The harness applies
those passes in order to each Tcl file in the mirrored case directory, after
staging case assets and before either engine runs. Staged Tcl files, including
sourced helpers, are part of the parity cache hash. Omit `tcl_source_version`
to skip the upgrade pass. A version equal to the binary pin also skips the
pass. To move the binary pin from 3.7.1 to 3.8.0, register the legacy `Series`
upgrade under `3.7.1` so both existing dialects pick it up automatically.

- `3.0.2` adds `-GJ 1.0e-10` to `section Fiber` and `section fiberSec` with
  an omitted torsion argument when the active Tcl model is 3D. The runtime
  dimension check also covers procedures shared by 2D and 3D models. The
  V3.0.2 parser reads `-GJ` into its local `GJ` variable when explicitly
  supplied. With no option it bypasses that variable and calls
  `FiberSection3d` with null torsion, whose elastic fallback is `1.0e-10`.
  [V3.0.2 section parser](https://github.com/OpenSees/OpenSees/blob/V3.0.2/SRC/material/section/TclModelBuilderSectionCommand.cpp#L1942-L1965) ·
  [V3.0.2 fiber constructor](https://github.com/OpenSees/OpenSees/blob/V3.0.2/SRC/material/section/FiberSection3d.cpp#L125-L131)
- `openseesrt-fd3958f` covers the STAIRlab Gallery snapshot `fd3958f` using
  xara parser revision `856acc159ca789d4796cf3234ee499a422781c14`, pinned in
  `reference-data.lock.json`. It converts positional `model basic` dimensions
  to `-ndm` and `-ndf` and maps the `FiberSec` alias to `Fiber`.
  [Gallery snapshot](https://github.com/peer-open-source/opensees-gallery/tree/fd3958fde8a3d0a350321b3fbdd3f415ee16e2a2) ·
  [xara model parser](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/model.cpp#L66-L156) ·
  [xara section aliases](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/section.cpp#L259-L279)
- The same gallery pass converts a complete keyword `nDMaterial DruckerPrager`
  parameter list to positional Tcl in constructor order. Omitted mass density
  and atmospheric pressure use xara's defaults of `0.0` and `101.0`.
  [xara defaults](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/material/plastic.cpp#L77-L111) ·
  [xara DruckerPrager constructor call](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/material/plastic.cpp#L906-L920) ·
  [3.7.1 material parser](https://github.com/OpenSees/OpenSees/blob/v3.7.1/SRC/material/nD/UWmaterials/DruckerPrager.cpp#L57-L105)
- The gallery `SSPquad` positional form moves the material tag before the
  formulation and the thickness after it. The xara quad parser reads
  pressure and density before the body forces. Its SSPquad uses the
  material's mass density and forwards only the body forces, as does the
  OpenSees form. Both versions of `LayeredShell` require a section tag,
  layer count and material-first layer pairs. The published ConcretePanel
  declaration is malformed in both.
  [xara quad parser](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/element/plane.cpp#L176-L208) ·
  [xara SSPquad implementation](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/element/community/UWelements/SSPquad.cpp#L259-L282) ·
  [3.7.1 SSPquad parser](https://github.com/OpenSees/OpenSees/blob/v3.7.1/SRC/element/UWelements/SSPquad.cpp#L54-L118) ·
  [xara LayeredShell parser](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/SRC/runtime/commands/modeling/section/shell.cpp#L143-L218) ·
  [3.7.1 LayeredShell parser](https://github.com/OpenSees/OpenSees/blob/v3.7.1/SRC/material/section/LayeredShellFiberSection.cpp#L41-L137)

### Multi-domain and live Tcl

Replay keeps a single Tcl interpreter, so script variables and procedures
survive `wipe`. Each domain has its own Strut model and analysis history.
OpenSees runs the mirrored script once. The harness snapshots each domain's
recorder outputs after `wipe` closes them and at script completion. Parity
compares every captured history under `__strut_domains/<number>/`, including
successive domains that write the same filename. As with ordinary direct Tcl
recorders, parity compares response columns after dropping the optional time
column. Explicit `getTime` queries are compared separately.

In live mode, Strut replays the accepted analysis stages through each
`analyze` call using the Mojo solver. Tcl sees the resulting displacement,
velocity, acceleration, element response or eigenvalues, and can branch on
the analysis status. An unsuccessful equilibrium iteration returns `-3`
and only the successful prefix enters subsequent replay. Fixed-step
Newmark stages run ahead in bounded batches. Later calls reuse a successful
prefix only while extending the same stage with the same model. A failed
lookahead triggers an exact replay to the requested step. Every replay starts
with the committed history, including material updates and load application.
The harness compares the Tcl query results in command order within each
response type. Scripts on the ordinary direct-Tcl path run one solver input
without query tracing or replay.
Build another domain with `wipe` before changing nodes, elements or material
definitions after an analysis. Stage commands can still change loads,
time series and analysis controls in the current domain.

One-dimensional Tcl models support `truss`, `corotTruss` and `zeroLength`.
The Python solver entry point uses the planar assembly with a fixed
transverse DOF on each node. Initial 1D and 2D Tcl eigen queries assemble
nodal and element mass, including `-cMass` on elastic and displacement-based
2D beams. The solver uses the committed tangent for supported queries after
`analyze`. Those queries currently require lumped mass. The time series
verification runs the four local Path cases from TimeSeriesTest.tcl.

Standard Tcl eigen queries use the reduced tangent stiffness with unit mass.
The 2D beam and section gallery sweep asks OpenSees for its symmetric-band
backend, which supports the standard problem in 3.7.1. Both beam and section
sweeps use live replay to compare the published domains and query results.

The excavation example combines plane-strain soil, displacement-based beam
columns, and frictional beam contact. The solver uses three DOFs at each
beam end and two at both the secondary and multiplier nodes. Initial-state
reset retains the wrapped soil strains, resets the beam and contact trial
state, and restarts the analysis clock. Subsequent LoadControl stages can
remove the excavated soil and contact elements and their nodes. The mirrored
Tcl removes the temporary tag-zero parameter after each
`InitialStateAnalysis` command so OpenSees can release the domain, and
records the published outputs at full precision.

Coupled transient Tcl models can change permeability between analysis stages.
With `Transformation`, a `MultipleSupport` pattern can prescribe pore pressure
using `groundMotion Plain -vel` with an inline `Constant` series and
`imposedMotion` on an independent pressure DOF. Multiple
`UniformExcitation` patterns can also run together with constant nodal loads.
The three `gallery_validation_plane_quadbending*` manifests run the four-, eight-,
and nine-node variants in separate domains. The PlanarShearWall and model-0010
manifests each select a one-story, one-bay brick domain from their upstream
sweeps, using SSPbrick and stdBrick respectively.

Continuum integration-point recorders use OpenSees' one-based `material` index.
They expose stress and strain for the implemented continuum elements and the
five Drucker–Prager state values for stdBrick. The pinned OpenSees executable
does not register the legacy Brick8N, Brick27N, or ElasticCrossAnisotropic
commands.

For a published script with multiple analysis phases, the manifest can point to
a generated capped Tcl entry. Both engines receive the same entry. The
centrifuge cap runs the original three elastic and five elasto-plastic gravity
steps, then keeps the first adaptive shaking loop through 0.15 s. The entry
allows at most eight attempts in that loop and checks that both gravity stages
and the shaking cap are reached. The full benchmark has its own Tcl entry
with all nine adaptive loops.

## Reproduction Commands

- Run Ex2c staged parity case end-to-end:
  - `STRUT_FORCE_CASE=1 uv run scripts/run_case.py tests/validation/opensees_example_ex2c_canti2d_inelastic_fiber/direct_tcl_case.json`
- Re-run parity check only:
  - `uv run scripts/compare_case.py --case opensees_example_ex2c_canti2d_inelastic_fiber`
- Run per-case benchmark (Strut is always per-case, no archive):
  - `uv run scripts/run_benchmarks.py --cases opensees_example_ex2c_canti2d_inelastic_fiber --include-disabled --engine strut --repeat 1 --warmup 0 --no-archive`

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

Set `"comparison": "envelope"` in a direct-Tcl manifest when the reference
response is sensitive to equation numbering. For each recorded component, the
comparator checks the minimum, maximum, and peak absolute value across the
entire history using the default recorder tolerances. Both engines must
provide the same number of samples. Modal and native envelope recorders
retain their usual comparisons.

## Notes

- For static analyses with time series, Strut uses normalized pseudoTime `t = (step + 1) / steps` (static_linear uses `t = 1.0`).
- Validation parity tests run only cases with `"enabled": true` by default (`STRUT_RUN_ALL_CASES=1` includes disabled cases). `STRUT_FORCE_CASE=1` forces a disabled case in `scripts/run_case.py`.
- Recorder entries may set `"parity": false` to generate outputs but exclude that recorder from parity checks.
- Benchmarks run all `"enabled": true` validation cases by default; `--include-disabled` adds disabled cases too.
- OpenSees reference outputs are cached by JSON content hash in `tests/validation/<case>/reference/.ref_hash`. Set `STRUT_REFRESH_REFERENCE=1` to regenerate.
- Benchmarks use `scripts/run_benchmarks.py`. The latest run is written to `benchmark/results/` (`benchmark/results-profile/` when `--profile`), and summary snapshots are archived in `benchmark/archive/` (both ignored by git). Runs with `--profile` default to `--no-archive`.
- `scripts/run_benchmarks.py` is compute-only and does not run parity comparisons; run `uv run run_tests.py` first when you need correctness gates.

# Material Nonlinearity Design Notes

## Uniaxial Material Framework

Strut models uniaxial materials with a trial/commit state machine, mirroring
OpenSees semantics. Each element instance owns its own uniaxial material state
(even if it references the same material definition), which enables path-
dependent behavior without cross-element coupling.

### Key Concepts

- **Definition vs State**
  - `UniMaterialDef` stores immutable parameters (e.g., `Steel01` Fy, E0, b).
  - `UniMaterialState` stores committed and trial values.
- **Trial/Commit**
  - `set_trial_strain` computes trial stress/tangent based on the committed
    state.
  - `commit` advances trial → committed after a converged step.
  - `revert_trial` restores committed state before the next iteration.

## Nonlinear Solver Integration

`static_nonlinear` and `transient_nonlinear` call `revert_trial_all` each nonlinear
iteration, then assemble
internal forces and tangent stiffness using trial states. On step convergence,
`commit_all` persists the state. This keeps material evolution consistent with
Newton/Modified Newton updates.

## Implemented Uniaxial Models

- **Steel01**
  - Parameters: `{ Fy, E0, b }`.
- **Steel02**
  - Parameters: required `{ Fy, E0, b }`.
  - Optional grouped parameters follow OpenSees argument forms:
    - `{ R0, cR1, cR2 }`
    - `{ a1, a2, a3, a4 }` (requires `R0/cR1/cR2`)
    - `{ sigInit }` (requires `a1..a4` and `R0/cR1/cR2`)
  - Defaults (OpenSees): `R0=15.0`, `cR1=0.925`, `cR2=0.15`, `a1=0`, `a2=1`, `a3=0`, `a4=1`, `sigInit=0`.
- **Concrete01**
  - Parameters: `{ fpc, epsc0, fpcu, epscu }` (compression-only with unloading).
- **Concrete02**
  - Parameters: required `{ fpc, epsc0, fpcu, epscu }`, optional grouped `{ rat, ft, Ets }`.
  - Defaults (OpenSees): `rat=0.1`, `ft=abs(0.1*fpc)`, `Ets=0.1*fpc/epsc0`.

## Extension Guidelines

- **Gap / Elastic-Plastic**
  - Support gap activation thresholds and plastic return mapping using the same
    state framework.
- **Fiber Sections**
  - Implemented infrastructure (`FiberSection2d`, `patch rect`, `layer straight`):
    - Fiber discretization stores per-fiber `(y, z, area, material)`.
    - Section owns one uniaxial state per fiber.
    - Aggregation (about section centroid `y_bar`) follows:
      - `eps_f = eps0 - (y - y_bar) * kappa`
      - `N = sum(sig_f * A_f)`
      - `Mz = sum(-sig_f * A_f * (y - y_bar))`
      - `k11 = sum(Et_f * A_f)`
      - `k12 = sum(-Et_f * A_f * (y - y_bar))`
      - `k22 = sum(Et_f * A_f * (y - y_bar)^2)`
  - This section workflow is exposed by `src/mojo/section_path.mojo` and
    `scripts/run_mojo_section_path.py` for unit validation.

## Compatibility Notes

Nonlinear uniaxial materials currently require `static_nonlinear` or
`transient_nonlinear` analysis. Linear analyses continue to use elastic tangents only.

## Stress-Strain Curves

Material curve cases are defined in `scripts/plot_materials.py` as a list of
materials with params and strain targets. The script generates displacement-control
JSON cases in memory and writes temporary files under `build/material_curves/`
before running OpenSees/Mojo through the same JSON -> Tcl -> solver path. Use
`scripts/plot_materials.py` to generate plots and CSV outputs under
`build/material_curves/`.

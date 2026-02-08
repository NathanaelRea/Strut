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

`static_nonlinear` calls `revert_trial_all` each Newton iteration, then assembles
internal forces and tangent stiffness using trial states. On step convergence,
`commit_all` persists the state. This keeps material evolution consistent with
Newton updates.

## Extension Guidelines

- **Concrete01**
  - Add a uniaxial material type with parameters `{ fpc, epsc0, fpcu, epscu }`.
  - Implement compression-only envelope with unloading as needed.
- **Gap / Elastic-Plastic**
  - Support gap activation thresholds and plastic return mapping using the same
    state framework.
- **Fiber Sections**
  - Fiber sections should own per-fiber uniaxial material states. Section
    aggregation (forces + tangent) will become the bridge to nonlinear
    beam-column elements.

## Compatibility Notes

Nonlinear uniaxial materials currently require `static_nonlinear` analysis. Linear
analyses continue to use elastic tangents only.

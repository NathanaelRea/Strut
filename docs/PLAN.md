# Strut Roadmap and Implementation Plan

This document is a living roadmap for Strut (Mojo rewrite of OpenSees). It captures near-term tasks, mid-term goals, and long-term possibilities. Update this file as we learn or decide.

## Context and Current State

- Canonical model format is JSON (`schema_version: 1.0`).
- Harness converts JSON -> Tcl -> OpenSees and compares results with a Strut solver output.
- Phase-1 solver is implemented in Mojo (Python is only used for the harness).
- Current focus: 2D linear `elasticBeamColumn` with static linear analysis and nodal displacement comparisons.
- Solver and element implementations are split into `src/mojo/solver/` and `src/mojo/elements/` for maintainability.

## Phase 1 (Now / Next)

### A. Harness and Validation (Immediate)

- [x] Stabilize JSON schema v1.0 (2D frame, linear elastic).
- [x] Keep JSON -> Tcl converter deterministic.
- [x] Keep comparator strict, report detailed mismatch vectors.
- [x] Add 3-5 validation cases for simple frames.

### B. Mojo Solver (Immediate)

- [x] Implement 2D linear elastic beam-column in Mojo.
- [x] Implement global stiffness assembly, constraints, and linear solve.
- [x] Match OpenSees displacements within tolerance.
- [x] Wire `strut.mojo` into the harness (replace Python runner).

### C. Unit Tests (Immediate)

- [x] Add analytic unit tests for beam deflection.
- [x] Add numerical regression tests for JSON cases.

### D. Benchmarking (Immediate)

- [x] Benchmark runner for OpenSees (Wine) vs Mojo.
- [x] Store latest benchmark results in `benchmark/results/` (ignored for now).
- [x] Store archival benchmark summaries in `benchmark/archive/` (ignored for now).

## Phase 2 (Near-Term)

### A. More Elements (Elastic)

- [x] `truss`
- [x] `twoNodeLink`
- [x] `zeroLength`
- [x] `fourNodeQuad` (elastic plane stress)
- [x] `shell` (simple elastic shell)

### B. 3D Expansion

- [x] Enable `ndm=3`, `ndf=6`.
- [x] Add `elasticBeamColumn` 3D.
- [x] Add 3D truss and frame examples.

### C. Geometric Nonlinearity (P-Delta / Corotational)

- [x] Add `geomTransf PDelta`.
- [x] Add `geomTransf Corotational`.
- [x] Compare element forces and displacements under second-order effects.
- [x] Optimize P-Delta and shell assembly performance (reduce allocations, cache element data).
- [x] Fuse nonlinear stiffness/internal assembly and reuse buffers (static_nonlinear).

### D. Time Series + Loading

- [x] Implement time series: `Constant`.
- [x] Implement time series: `Linear`.
- [x] Implement time series: `Path`.
- [x] Implement time series: `Trig`.
- [x] Add element loads: `beamUniform` for `elasticBeamColumn2d`.
- [x] Support transient analysis workflow in JSON.

### E. Performance (Near-Term)

- [x] Add RCM-based banded solver path for large linear frames.
- [x] Add generated large-frame benchmark (~1k DOF).

## Phase 3 (Mid-Term)

### A. Material Nonlinearity

- [x] Add nonlinear uniaxial material: Steel (bilinear).
- [x] Add nonlinear uniaxial material: Steel02 (Menegotto-Pinto).
- [x] Add stress-strain plotting script for nonlinear materials.
- [x] Add nonlinear uniaxial material: Concrete (unconfined).
- [x] Add nonlinear uniaxial material: Concrete02 (tension softening + damage unloading).
- [ ] Add nonlinear uniaxial material: Gap / hook / elastic-plastic.
- [ ] Section aggregation (fiber sections).
- [ ] Hysteresis verification vs OpenSees.

### B. Beam-Column Nonlinearities

- [ ] `forceBeamColumn`.
- [ ] `dispBeamColumn`.
- [ ] `beamWithHinges`.
- [ ] `gradientInelasticBeamColumn`.
- [ ] `mixedBeamColumn`.

### C. Solver Features

- [ ] Nonlinear solution algorithms (Newton, Modified Newton).
- [x] Convergence tests and step control.
- [x] Static nonlinear displacement-control integrator with augmented load-factor solve.
- [ ] Rayleigh damping for dynamic analysis.

## Phase 4 (Long-Term)

### A. Large Element Coverage (OpenSees Parity)

The following are major OpenSees element families to consider. This list is based on the current OpenSees source tree and is not exhaustive.

- [ ] `beam2d`.
- [ ] `beam3d`.
- [ ] `elasticBeamColumn`.
- [ ] `forceBeamColumn`.
- [ ] `dispBeamColumn`.
- [ ] `dispBeamColumnInt`.
- [ ] `mixedBeamColumn`.
- [ ] `nonlinearBeamColumn`.
- [ ] `updatedLagrangianBeamColumn`.
- [ ] `gradientInelasticBeamColumn`.
- [ ] `beamWithHinges`.
- [ ] `truss`.
- [x] `twoNodeLink`.
- [x] `zeroLength`.
- [x] `fourNodeQuad`.
- [x] `shell`.
- [ ] `brick`.
- [ ] `triangle`.
- [ ] `tetrahedron`.
- [ ] `joint`.
- [ ] `frictionBearing`.
- [ ] `elastomericBearing`.
- [ ] `rockingBC`.
- [ ] `adapter`.
- [ ] `catenaryCable`.
- [ ] `pipe`.
- [ ] `masonry`.
- [ ] `mvlem`.
- [ ] `PFEMElement`.
- [ ] `PML`.
- [ ] `absorbentBoundaries`.
- [ ] `UWelements`.
- [ ] `HUelements`.
- [ ] `XMUelements`.

### B. Advanced Nonlinearities

- [ ] Plasticity at material and section levels.
- [ ] Large deformation (geometric nonlinearity with updated/total Lagrangian).
- [ ] Contact and constraint nonlinearities.
- [ ] Cyclic degradation and damage models.

### C. Dynamics and Earthquake Features

- [ ] Time integration (Newmark, HHT, etc.).
- [ ] Ground motion inputs and multi-support excitation.
- [ ] Nonlinear transient analysis and damping models.

### D. GPU Acceleration

- [ ] Offload assembly and solver kernels.
- [ ] Sparse solver acceleration.
- [ ] Batched element kernels for many small elements.

## Harness and Schema Evolution

- [ ] Extend JSON schema with versioned upgrades (1.1, 1.2, ...).
- [ ] Maintain conversion to Tcl as a compatibility layer.
- [ ] Add recorder output: element forces.
- [ ] Add recorder output: reactions.
- [ ] Add recorder output: section forces.
- [ ] Add structured error reporting and mismatch dashboards.

## Next Concrete Tasks (Suggested)

1. [ ] Implement Mojo solver parity for 2D linear beam.
2. [ ] Add three additional JSON cases (2D frame variations).
3. [ ] Add element force comparison in the comparator.
4. [ ] Add P-Delta geometric transformation.
5. [ ] Add a second element (truss or zeroLength).

## Assumptions / Notes

- OpenSees (C++/Tcl) remains the canonical reference.
- Tests and harness drive development order.
- GPU optimization is deferred until parity is stable.

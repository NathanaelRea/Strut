# Beam-Column Parity Contract (Phase 0)

This contract freezes the accepted command forms, options, and analysis combinations for the current parity gate.

## Element Command Forms

- 2D force-based: `forceBeamColumn2d`, `dispBeamColumn2d`
- 3D force/displacement: `forceBeamColumn3d`, `dispBeamColumn3d`

## Section Contract

- 2D beam-column families accept:
  - `ElasticSection2d`
  - `FiberSection2d`
- 3D beam-column families accept:
  - `ElasticSection3d`
  - `FiberSection3d` (requires positive `G` and `J` torsion params)

## Geometry/Integration Contract

- 2D beam-column families:
  - `geomTransf`: `Linear | PDelta`
  - `integration`: `Lobatto | Legendre | Radau`
  - `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
- 3D beam-column families:
  - `geomTransf`: `Linear | PDelta | Corotational`
  - `integration`: `Lobatto | Legendre | Radau`
  - `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)

## Analysis Contract

- 2D beam-column parity targets:
  - `static_linear`
  - `static_nonlinear`
  - `transient_nonlinear`
  - `staged`
- 3D beam-column parity targets in this phase:
  - `static_linear`

## Recorder Contract

- Supported for 2D and 3D beam-column families:
  - `element_force`
  - `envelope_element_force`
- Supported for 2D and 3D beam-column families:
  - `section_force`
  - `section_deformation`
- 3D section recorder ordering matches OpenSees:
  - `section_force`: `N`, `Mz`, `My`, `T`
  - `section_deformation`: `eps0`, `kappa_z`, `kappa_y`, `twist`

## Parity Corpus and Matrix

- Parity corpus/matrix source: `tests/validation/PHASE0_CASE_MATRIX.md`
- Corpus contains:
  - 2D `forceBeamColumn2d` coverage for static linear, static nonlinear, transient nonlinear, and staged workflows
  - 2D `dispBeamColumn2d` static linear baseline coverage
  - 3D `forceBeamColumn3d` and `dispBeamColumn3d` static linear baseline coverage, including fiber-section torsion recorder parity

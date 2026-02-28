# Fiber Section Notes

Current implementation status:

- `FiberSection2d` infrastructure exists in Mojo.
- `FiberSection3d` infrastructure exists in Mojo and is available in 3D beam-column runtime paths.
- Supported primitives:
  - `patch rect`
  - `patch quadr`
  - `layer straight`
- Aggregation outputs:
  - 2D: axial force `N`, bending moment `Mz`, tangent terms `k11`, `k12`, `k22`
  - 3D: axial force `N`, bending moments `My`,`Mz`, tangent terms `k11`, `k12`, `k13`, `k22`, `k23`, `k33`

Key files:

- `src/mojo/sections/fiber2d.mojo`
- `src/mojo/sections/fiber3d.mojo`
- `src/mojo/section_path.mojo`
- `scripts/run_strut_section_path.py`

Current limitation:

- `elasticBeamColumn2d` / `elasticBeamColumn3d` still do not consume fiber sections.
- 3D beam-column runtime use of `FiberSection3d` currently requires positive `G` and `J` section params to avoid a free torsional mechanism.
- `forceBeamColumn2d` and `dispBeamColumn2d` currently support the Phase 0 parity contract:
  - `geomTransf: Linear`
  - `geomTransf: PDelta`
  - `integration: Lobatto | Legendre | Radau`
  - `num_int_pts`: scheme-valid (`Lobatto >= 2`, `Legendre >= 1`, `Radau >= 1`)
  - `analysis.type: static_linear | static_nonlinear | transient_nonlinear | staged`
  - section recorder outputs are available for both elements:
    - `section_force` (`N`, `Mz`)
    - `section_deformation` (`eps0`, `kappa`)
- `forceBeamColumn3d` and `dispBeamColumn3d` section recorder outputs now follow the OpenSees 3D contract:
  - `section_force` (`N`, `Mz`, `My`, `T`)
  - `section_deformation` (`eps0`, `kappa_z`, `kappa_y`, `twist`)

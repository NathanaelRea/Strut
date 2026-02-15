# Fiber Section Notes

Current implementation status:

- `FiberSection2d` infrastructure exists in Mojo.
- Supported primitives:
  - `patch rect`
  - `patch quadr`
  - `layer straight`
- Aggregation outputs:
  - axial force `N`
  - bending moment `Mz`
  - tangent terms `k11`, `k12`, `k22`

Key files:

- `src/mojo/sections/fiber2d.mojo`
- `src/mojo/section_path.mojo`
- `scripts/run_mojo_section_path.py`

Current limitation:

- `elasticBeamColumn2d` / `elasticBeamColumn3d` still do not consume fiber sections.
- `forceBeamColumn2d` and `dispBeamColumn2d` are available in a minimum v1 scope only:
  - `geomTransf: Linear`
  - `geomTransf: PDelta`
  - `integration: Lobatto`
  - `num_int_pts: 3 | 5`
  - `analysis.type: static_linear | static_nonlinear | transient_nonlinear | staged`
  - section recorder outputs are available for both elements:
    - `section_force` (`N`, `Mz`)
    - `section_deformation` (`eps0`, `kappa`)

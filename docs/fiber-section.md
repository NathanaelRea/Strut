# Fiber Section Notes

Current implementation status:

- `FiberSection2d` infrastructure exists in Mojo.
- Supported primitives:
  - `patch rect`
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
- `forceBeamColumn2d` is available in a minimum v1 scope only:
  - `geomTransf: Linear`
  - `integration: Lobatto`
  - `num_int_pts: 3`
  - `analysis.type: static_nonlinear`

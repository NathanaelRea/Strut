# Fiber Section Notes

Current implementation status:

- `FiberSection2d` infrastructure exists in Mojo.
- `FiberSection3d` infrastructure exists in Mojo and is available in 3D beam-column runtime paths.
- Supported primitives:
  - `patch rect`
  - `patch quadr`, with each fiber at its cell's area centroid as in OpenSees `QuadCell` (for a trapezoidal cell that is not the image of the cell's parametric centre)
  - `layer straight`
- Aggregation outputs:
  - 2D: axial force `N`, bending moment `Mz`, tangent terms `k11`, `k12`, `k22`
  - 3D: axial force `N`, bending moments `My`,`Mz`, tangent terms `k11`, `k12`, `k13`, `k22`, `k23`, `k33`

Fiber condensation (2D only):

- A `FiberSection2d` stores one fiber per (material, y). Its fiber strain is `eps0 - y * kappa`, so same-material fibers at one y share a strain history and are merged into one fiber with the summed area when the section is built (`_condense_fiber_cells_by_y`).
- A rectangular patch with `ny x nz` subdivisions therefore costs `ny` fibers, and bars spread along z at one depth cost one. The example RC sections drop from 582 fibers to 67.
- Area and first moment are preserved exactly; results differ from the unmerged section only by floating-point summation order.
- `FiberSection3d` is not condensed: under biaxial bending the strain depends on both y and z.
- A future per-fiber recorder would need a map from input fibers to merged fibers.

3D runtime layout:

- A `FiberSection3dDef` owns the state of all its instances (one per element integration point) as
  structure-of-arrays, grouped by material family (Steel02, Concrete02, Concrete01, Steel01) and padded
  to the SIMD width. Linear-elastic fibers are folded into constant section stiffness terms; any other
  material uses a scalar fallback. `fiber_section3d_set_trial_instance` evaluates one instance.
- 3D fibers are not stored in the global `uniaxial_states` list, so step retries do not copy them;
  drivers call `fiber_section3d_commit_runtime_all` / `fiber_section3d_revert_trial_runtime_all` next
  to the 2D calls.
- An element writes only its own instances, which is what lets the 3D beam-column loop run on several
  threads (`solver/assembly/frame3d_scratch.mojo`).
- The Steel02 kernels live in `sections/fiber_kernels.mojo` and are shared by the 2D and 3D sections.
  Their powers use `simd_math.simd_pow_nonneg` (scalar paths use `pow_nonneg`), because the builtin
  float64 `**` is only accurate to about 1e-8 relative and is evaluated one lane at a time.

Key files:

- `src/mojo/sections/fiber2d.mojo`
- `src/mojo/sections/fiber3d.mojo`
- `src/mojo/sections/fiber_kernels.mojo`
- `src/mojo/simd_math.mojo`
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

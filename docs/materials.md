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
- **Bilin**
  - OpenSees positional parameters include asymmetric yield, hardening, capping,
    residual strength, and energy-based deterioration. The optional 24th
    parameter is `nFactor`.
  - Each spring or section response tracks its own reversal and deterioration
    history. Trial updates start from the last committed history.
- **Hysteretic**
  - The positive and negative backbones each accept two or three force/deformation
    pairs, followed by `pinchX pinchY damage1 damage2` and optional `beta`.
  - The state records strain extrema, reversal points, and dissipated energy for
    the pinching and unloading rules.

For these two materials, JSON stores the OpenSees positional arguments after
the material tag in `params.values`. The material tests compare cyclic stress
and tangent with OpenSees 3.7.1, including repeated trials between commits.
- **ViscousDamper** uses the axial displacement and velocity of a `twoNodeLink`
  during transient analysis. It accepts `K C alpha`, an optional gap `LGap`,
  and the optional group `NM RelTol AbsTol MaxHalf`. The three integration
  schemes are Dormand–Prince, Adams–Bashforth–Moulton, and Rosenbrock.
- **Fatigue** wraps a `Steel02` material using the OpenSees modified rainflow
  algorithm. Pass the base material tag and optional `-Dmax`, `-E0`, `-m`,
  `-min`, and `-max` flags. Each 3D fiber or uniaxial element instance tracks
  its own strain reversals and reduces its stress and tangent after failure.

- **ElasticPP** uses `{ E, epsyP, epsyN?, eps0? }`. It returns zero tangent
  on its perfectly plastic branches and advances plastic strain on commit.
  The negative yield strain defaults to `-epsyP` and the strain offset to zero.
- **ElasticBilin** uses `{ E1P, E2P, eps2P, E1N?, E2N?, eps2N? }`. Its
  stress follows a reversible bilinear envelope. The optional negative branch
  is supplied as a group and defaults to the reflection of the positive branch.
- **MinMax** uses `{ material, min?, max? }` and owns an instance of the
  referenced uniaxial material. Reaching either limit gives zero stress and
  a tangent of `1e-8` times the base material's initial tangent. Failure becomes
  permanent when the step commits. Bounds default to `-1e16` and `1e16`.
- **Ratchet** uses `{ E, freeTravel, freeTravelInitial, RatType }`. It develops
  elastic force above the engagement threshold. A negative excursion moves
  that threshold on commit in discrete increments for mode 1 and continuously
  for mode 2.

The cyclic material tests compare stress and tangent for all four materials
with OpenSees 3.7.1, including repeated trials between commits.

- **ENT** uses `{ E, a?, b? }`. It is linear in compression, with the
  compressive tangent at zero strain. Tension defaults to zero (`a = 0`).
  The optional tensile branch is `a E tanh(b strain)`, with `b = 1` by default.
- **Parallel** uses `{ materials, factors? }`. Every component receives the
  same strain and the aggregate sums their weighted stresses and tangents.
  Factors default to one. Each material instance owns independent component
  histories, including repeated or nested components.

`FiberSection2d` and `FiberSection3d` use the generic uniaxial state path for
materials outside their optimized material families, including MinMax wrappers.

## Extension Guidelines

- **Gap / Elastic-Plastic**
  - Support gap activation thresholds and plastic return mapping using the same
    state framework.
- **Fiber Sections**
  - Implemented infrastructure (`FiberSection2d` and `FiberSection3d`, `patch rect`, `layer straight`):
    - Fiber discretization stores per-fiber `(y, z, area, material)`.
    - Section owns one uniaxial state per fiber.
    - Aggregation (about section centroid `y_bar`,`z_bar`) follows:
      - `eps_f = eps0 - (y - y_bar) * kappa`
      - `N = sum(sig_f * A_f)`
      - `Mz = sum(-sig_f * A_f * (y - y_bar))`
      - `k11 = sum(Et_f * A_f)`
      - `k12 = sum(-Et_f * A_f * (y - y_bar))`
      - `k22 = sum(Et_f * A_f * (y - y_bar)^2)`
    - 3D section path uses:
      - `eps_f = eps0 + (z - z_bar) * kappa_y - (y - y_bar) * kappa_z`
      - `My = sum(sig_f * A_f * (z - z_bar))`
      - `Mz = sum(-sig_f * A_f * (y - y_bar))`
  - This section workflow is exposed by `src/mojo/section_path.mojo` and
    `scripts/run_strut_section_path.py` for unit validation.

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

## ConcreteS plane stress

Use `nDMaterial ConcreteS tag E nu fc ft Es` in Tcl or a `ConcreteS` material
with those five named parameters in JSON. In a `LayeredShell`, reference it
through `PlateFromPlaneStress`. Each concrete layer at each shell integration
point has its own trial and committed history. The plate wrapper supplies a
constant elastic modulus for the two transverse shear components.

The reference is Yuli Huang and Xinzheng Lu's 2012 *Simple Concrete Model,
plane stress*. The implementation gives the full constitutive update and
contains no paper citation. Jiang's 2016 thesis (§4.3.1, Fig. 4.10) documents
the von Mises and Rankine formulation in the thermal model he derived from
ConcreteS. [OpenSees 3.7.1 implementation](https://github.com/OpenSees/OpenSees/blob/v3.7.1/SRC/material/nD/ConcreteS.cpp),
[Jiang's thesis](https://hdl.handle.net/1842/19563).

- **Elastic prediction.** Start from the last committed effective stress
  (before tensile damage) and add `C × (strain − committed strain)`. `C` is
  isotropic plane-stress elasticity with engineering shear strain. Its normal
  diagonal is `E/(1 − nu²)`, off-diagonal is `nu E/(1 − nu²)`, and shear
  diagonal is `E/[2(1 + nu)]`.
- **Compression and crushing.** Set `q = sqrt(σx² − σx σy + σy² + 3τxy²)`.
  If `q > |fc|` and both predictor principal stresses are at most `ft`,
  take one stress correction `−C × f × (q − |fc|)/(fᵀ C f)`, with
  `f = ∂q/∂σ` at the predictor. Recompute the principal stresses and their
  direction from the corrected tensor. With a positive major principal
  stress, clip the minor principal stress at `−|fc|`. The other sign check
  in the source is unreachable for ordered principal stresses. The
  compression branch is a perfect-plastic yield approximation. There is
  no separate crushing strain, compressive damage variable, or descending
  compression envelope.
- **Tension and cracking.** Let `c₀` be the committed scalar crack strain
  and `Ep = 10E`. Each principal stress exceeding `ft + Ep c₀` is replaced
  with `ft + Ep c`, where `c = c₀ + excess/(E + Ep)`. The crack strain
  takes the larger tensile increment when both principal stresses cross the
  limit. Transform the corrected stresses back to the original axes to get
  the effective stress. Multiply the positive principal stresses by
  `exp(−(Ep + 11|Es|)c/ft)`, then transform again to get the returned
  stress. Compression is unaffected by the damage factor. The principal
  axes follow the trial stress, and there is no fixed crack direction or
  separate in-plane shear-retention factor.
- **Meaning of `Es`.** For monotonic uniaxial tension with a traction-free
  lateral direction, let `δ = ε − ft/E ≥ 0` and `A = 10E/11`. The
  tensile envelope is `(ft + Aδ) exp(−(A + |Es|)δ/ft)`. Its right-hand
  slope at cracking is `−|Es|`. The exponential tail has no finite
  ultimate strain or residual stress floor.
- **Cycling.** Commit retains total strain, effective stress, and scalar
  crack strain. Unloading and reloading use the elastic predictor about
  that committed point, with tensile damage applied on output. On
  reversal into compression the factor stops acting on principal stresses
  once they become nonpositive. There is no separate pinching or unloading
  rule. OpenSees retains the previous *trial* crack strain when neither
  principal stress exceeds the current limit. Its revert-to-last-commit is
  a no-op. Strut preserves this behavior across iterations and retries.
- **Tangent.** The returned tangent begins at `C`. The compression
  correction subtracts `(C f)(C f)ᵀ/(fᵀ C f + 0.001 E)`. The principal
  stress clipping and tensile softening do not modify the returned tangent,
  so it is an approximate Newton tangent.

The strain-path test queries OpenSees 3.7.1's `PlaneStressMaterialTest`
for all three stress components and all nine tangent entries in tension,
compression, shear, and mixed cycles, including multiple trial strains per
commit. The 48-step [cyclic wall](../tests/validation/strut_concrete_s_cyclic_wall/direct_tcl_case.json)
checks tip displacement and base reactions through the Tcl converter and
Strut parity harness.

## ConcreteCM and SteelMPF

`ConcreteCM` follows the Chang–Mander compression and tension envelopes and
OpenSees' numbered unloading, reloading, and gap-closure rules. The Tcl form
takes `fpcc epcc Ec rc xcrn ft et rt xcrp` followed by optional `-GapClose 0|1`
or the monotonic-mode integer. The canonical optional keys are `gap_close` and
`monotonic`.

`SteelMPF` implements the asymmetric Menegotto–Pinto–Filippou curves. It takes
`fyp fyn E0 bp bn R0 cR1 cR2` and optional `a1 a2 a3 a4`. The absolute
compression yield strength is supplied as positive `fyn`. The positive and
negative branches track their own strain hardening, reversal points, curvature,
and previous-curve limits.

Both materials evaluate each trial from the last commit. The cyclic material
tests compare stress and tangent directly against OpenSees 3.7.1, including
paths with multiple trials between commits.

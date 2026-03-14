# 🔥 Strut 🔥

Performance-first finite element analysis (FEA) engine written in Mojo. The project focuses on structural engineering transient analysis (Buildings/Earthquakes), and aims to be as fast as possible for large models. Benchmarks and tests are run against OpenSees.

Currently we beat the performance of OpenSees by about 2x in large examples that take several seconds to run. Check out the [current benchmarks](./docs/benchmark-opensees-examples.md).

## Warnings

- I haven't put much work to generalize the environment, so it might be difficult to get this running on your machine right now
- We are testing/validating against the full OpenSees example suite, up through "Advanced Example 6". However, there are some tests we skip because I have not done a thorough validation pass. Please don't use this for anything important, as the test coverage is severely limited.

## Quick Start

1. Fetch the dependencies `uv sync`
2. Run the [setup script](./scripts/setup.sh) to clone reference repos, download modular docs, and download OpenSees example tcl files and ground motions.
3. Build the native Linux OpenSees reference binary with [scripts/build_opensees_linux.sh](./scripts/build_opensees_linux.sh). The harness uses [`.build/opensees-linux/OpenSees`](./.build/opensees-linux/OpenSees) by default.

## OpenSees Implementation Coverage

| Element family     | 2D  | 3D  |
| ------------------ | --- | --- |
| beamWithHinges     | ❌  | ❌  |
| dispBeamColumn     | ✅  | ✅  |
| elasticBeamColumn  | ✅  | ✅  |
| elastomericBearing | ❌  | ❌  |
| forceBeamColumn    | ✅  | ✅  |
| fourNodeQuad       | ✅  |     |
| frictionBearing    | ❌  | ❌  |
| shell              |     | ✅  |
| truss              | ✅  | ✅  |
| twoNodeLink        | ✅  | ✅  |
| zeroLength         | ✅  | ✅  |

### Uniaxial Materials (`uniaxialMaterial`)

| Material       | Strut |
| -------------- | ----- |
| Concrete01     | ✅    |
| Concrete02     | ✅    |
| Damper         | ❌    |
| DamperMaterial | ❌    |
| Elastic        | ✅    |
| Steel01        | ✅    |
| Steel01Thermal | ❌    |
| Steel02        | ✅    |
| TensionOnly    | ❌    |

### ND Materials (`nDMaterial`)

| Material         | Strut |
| ---------------- | ----- |
| ElasticIsotropic | ✅    |

### Analysis Type (`analysis.type`)

| Type                | Strut |
| ------------------- | ----- |
| static_linear       | ✅    |
| static_nonlinear    | ✅    |
| transient_linear    | ✅    |
| transient_nonlinear | ✅    |
| staged              | ✅    |
| modal_eigen         | ✅    |

### Linear System (`analysis.system`)

| System      | Strut |
| ----------- | ----- |
| BandGeneral | ✅    |
| BandSPD     | ✅    |
| ProfileSPD  | ✅    |
| SuperLU     | ✅    |
| UmfPack     | ✅    |
| FullGeneral | ✅    |
| SparseSYM   | ✅    |

### Constraint Handler (`analysis.constraints`)

| Constraints    | Strut |
| -------------- | ----- |
| Plain          | ✅    |
| Transformation | ✅    |
| Lagrange       | ✅    |

### Numberer (`analysis.numberer`)

| Numberer | Strut |
| -------- | ----- |
| RCM      | ✅    |
| Plain    | ✅    |

### Nonlinear Algorithm (`analysis.algorithm`)

| Algorithm             | Strut |
| --------------------- | ----- |
| Newton                | ✅    |
| ModifiedNewton        | ✅    |
| ModifiedNewtonInitial | ✅    |
| Broyden               | ✅    |
| NewtonLineSearch      | ✅    |
| KrylovNewton          | ❌    |

### Integrator (`analysis.integrator.type`)

| Integrator          | Strut |
| ------------------- | ----- |
| LoadControl         | ✅    |
| DisplacementControl | ✅    |
| Newmark             | ✅    |

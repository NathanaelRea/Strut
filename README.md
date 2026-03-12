# 🔥 Strut 🔥

Performance-first finite element analysis (FEA) engine written in Mojo. The project focuses on structural engineering transient analysis (Buildings/Earthquakes), and aims to squeeze out as much performance as possible. Benchmarks and validation is run against OpenSees.

## Background

This is a test project for now, primarily just to see how far I can push the vibes. The setup is a bit specific for my environment, and I haven't put much effort into making the best DX for other developers.

## Quick Start

1. Fetch the dependencies `uv sync`
2. Run the [setup script](./scripts/setup.sh) to clone reference repos, download modular docs, and download OpenSees examples.
3. Build the native Linux OpenSees reference binary with [scripts/build_opensees_linux.sh](./scripts/build_opensees_linux.sh). The harness uses [`.build/opensees-linux/OpenSees`](./.build/opensees-linux/OpenSees) by default.

## OpenSees Coverage Checklist

Element families below are based on current Strut implementation status. This is a summary view of implemented or will implement, not the full list of items.

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

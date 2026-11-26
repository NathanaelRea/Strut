# 🔥 Strut 🔥

Performance-first finite element analysis (FEA) engine written in Mojo. The project focuses on structural engineering transient analysis (Buildings/Earthquakes), and aims to be as fast as possible for large models. Benchmarks and tests are run against OpenSees.

Against OpenSees 3.7.1 on the same machine, Strut runs the OpenSees examples that take a second or more about 7x faster, 3D fiber frames 2x to 3x faster on one thread and 5x on eight, and a 162,000-DOF tall building model about 15x faster on one thread and 33x on eight. The tall building's results are not yet at parity with OpenSees. Check out the [current benchmarks](./docs/benchmark-opensees-examples.md).

## Warnings

- We test/validate against the full OpenSees example suite, up through "Advanced Example 6". However, there are some tests we skip because I have not done a thorough validation pass. Please don't use this for anything important, as it's currently a work in progress and the test coverage is limited.

## Quick Start

For building and running Strut

1. Fetch the dependencies `uv sync`
2. Build `uv run mojo build src/mojo/strut.mojo`
3. Run analysis against a json model (see validation tests for examples)

For testing or benchmarking

1. Fetch the dependencies `uv sync`
2. Run unit tests `uv run run_tests.py` or benchmarks `uv run scripts/run_benchmarks.py`

`run_tests.py` installs the pinned test data and [OpenSees reference binaries](#opensees-reference-binaries) on first use. To set up everything up front, including the agent reference sources, run [`scripts/setup.sh`](./scripts/setup.sh).

## Reference Data

`uv run scripts/setup_reference_data.py` installs everything from [`tests/fixtures/opensees/reference-data.lock.json`](./tests/fixtures/opensees/reference-data.lock.json). Each item downloads once per machine into `~/.cache/strut/reference/` and is checked against its pinned sha256 or commit.

- **Test fixtures:** the OpenSees example Tcl is committed under [`tests/fixtures/opensees/`](./tests/fixtures/opensees/README.md). Ground motions and the mega-tall model are pinned downloads copied in next to it. `--fixtures-only` installs just these.
- **Agent reference:** `docs/agent-reference/OpenSees` (v3.7.1, matching the binary) and `docs/agent-reference/modular` (the tag matching the Mojo version) are links to shared shallow clones.

## OpenSees Reference Binaries

`uv run scripts/setup_opensees.py` installs `OpenSees`, `OpenSeesMP`, `OpenSeesSP` and a matching `mpiexec` on linux-64. It needs no sudo, compilers or system Tcl/MPI, and takes about 20 seconds.

- **Pinned:** the binaries come from the conda-forge `opensees` package (currently 3.7.1; 3.8.0 rejects the inline `Series` syntax the wiki examples use, see `scripts/setup_opensees.py`). [`scripts/opensees/opensees-linux-64.lock`](./scripts/opensees/opensees-linux-64.lock) pins every package by URL and sha256. A pinned, checksum-verified micromamba installs it.
- **Shared across worktrees:** the environment installs once per lockfile into `~/.cache/strut/opensees/<lock-hash>/`. Override the location with `STRUT_CACHE_DIR` or `XDG_CACHE_HOME`. Every checkout with the same lock reuses it. A branch that changes the lock gets its own prefix. Re-running the script when the install already exists is a no-op. Concurrent runs are serialized with a file lock.
- **Lookup order** in `scripts/run_opensees.sh` and `scripts/run_openseesmp.sh`:
  1. `OPENSEES_BIN` / `OPENSEESMP_BIN`
  2. A local source build at `.build/opensees-linux/` (from [scripts/build_opensees_linux.sh](./scripts/build_opensees_linux.sh))
  3. The shared locked install

  When the locked `OpenSeesMP` is selected, the locked install's `mpiexec` is used.
- **Other commands:**
  - `setup_opensees.py path [--binary OpenSeesMP|--prefix]` prints the installed path.
  - `verify` re-runs the smoke tests.
  - `lock` re-resolves the spec in the script and rewrites the lockfile. To upgrade OpenSees, edit `CONDA_SPECS` and then run `lock`.
- **Cleanup:** delete stale prefixes under `~/.cache/strut/opensees/` once no branch uses their lock.

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

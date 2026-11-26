# Setup: reference data and OpenSees binaries

`uv run run_tests.py` installs the pinned test data and OpenSees binaries on first use. To set up everything up front, including the agent reference sources, run [`scripts/setup.sh`](../scripts/setup.sh). Everything is cached in `~/.cache/strut/` and shared by all checkouts.

## Reference data

`uv run scripts/setup_reference_data.py` installs everything from [`tests/fixtures/opensees/reference-data.lock.json`](../tests/fixtures/opensees/reference-data.lock.json). Each item downloads once per machine into `~/.cache/strut/reference/` and is checked against its pinned sha256 or commit.

- **Test fixtures:** the OpenSees example Tcl is committed under [`tests/fixtures/opensees/`](../tests/fixtures/opensees/README.md). Ground motions are pinned downloads copied in next to it. `--fixtures-only` installs just these. Only the direct-Tcl parity cases need them: unit tests write synthetic ground motions instead. The mega-tall model is generated locally by [`scripts/megatall_recon/`](../scripts/megatall_recon/README.md), never downloaded.
- **Synthetic ground motions:** [`scripts/synthetic_ground_motion.py`](../scripts/synthetic_ground_motion.py) writes seeded Clough-Penzien records in the `.AT2`/`.DT2`, `.acc` and `A10000.tcl` layouts the examples read. [`scripts/sweep_synthetic_parity.py`](../scripts/sweep_synthetic_parity.py) reruns a ground-motion parity case over many seeds and intensities and reports parity and each engine's convergence per run, e.g. `uv run scripts/sweep_synthetic_parity.py opensees_example_ex3_canti2d_analyze_dynamic_eq_uniform --seeds 1-5 --pga-scale 1,5`.
- **Agent reference:** `docs/agent-reference/OpenSees` (v3.7.1, matching the binary) and `docs/agent-reference/modular` (the tag matching the Mojo version) are links to shared shallow clones.

## OpenSees reference binaries

`uv run scripts/setup_opensees.py` installs `OpenSees`, `OpenSeesMP`, `OpenSeesSP` and a matching `mpiexec` on linux-64. It needs no sudo, compilers or system Tcl/MPI, and takes about 20 seconds.

- **Pinned:** the binaries come from the conda-forge `opensees` package (currently 3.7.1; 3.8.0 rejects the inline `Series` syntax the wiki examples use, see `scripts/setup_opensees.py`). [`scripts/opensees/opensees-linux-64.lock`](../scripts/opensees/opensees-linux-64.lock) pins every package by URL and sha256. A pinned, checksum-verified micromamba installs it.
- **Shared across worktrees:** the environment installs once per lockfile into `~/.cache/strut/opensees/<lock-hash>/`. Override the location with `STRUT_CACHE_DIR` or `XDG_CACHE_HOME`. Every checkout with the same lock reuses it. A branch that changes the lock gets its own prefix. Re-running the script when the install already exists is a no-op. Concurrent runs are serialized with a file lock.
- **Lookup order** in `scripts/run_opensees.sh` and `scripts/run_openseesmp.sh`:
  1. `OPENSEES_BIN` / `OPENSEESMP_BIN`
  2. A local source build at `.build/opensees-linux/` (from [scripts/build_opensees_linux.sh](../scripts/build_opensees_linux.sh))
  3. The shared locked install

  When the locked `OpenSeesMP` is selected, the locked install's `mpiexec` is used.
- **Other commands:**
  - `setup_opensees.py path [--binary OpenSeesMP|--prefix]` prints the installed path.
  - `verify` re-runs the smoke tests.
  - `lock` re-resolves the spec in the script and rewrites the lockfile. To upgrade OpenSees, edit `CONDA_SPECS` and then run `lock`.
- **Cleanup:** delete stale prefixes under `~/.cache/strut/opensees/` once no branch uses their lock.

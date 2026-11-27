# OpenSees centrifuge model generator

This bundle reconstructs Antonios Vytiniotis's GiD model of a laminar-box centrifuge test. It includes the Yolo Loam slopes, dense and loose sand, pore-pressure boundary conditions, and the three sinusoidal shaking sequences. The published export is 678,138 bytes and its SHA-256 is pinned in [source.py](source.py). [OpenSees user manual](https://opensees.berkeley.edu/OpenSees/manuals/usermanual/4188.htm) · [model notes](https://opensees.berkeley.edu/wiki/index.php/Simulating_a_Centrifuge_Test).

```sh
uv run python -m scripts.centrifuge_recon.bundle --install
uv run python -m scripts.centrifuge_recon.bundle --out-dir .tmp/centrifuge-recon
```

Reference setup and pytest install this bundle in `tests/fixtures/opensees/OpenSeesManual/Centrifuge/`. The `.strut-generated` stamp tracks the inputs. The generator uses the Python standard library and no downloads.

| Entry | Contents |
| --- | --- |
| `CentrifugeExample.tcl` | The published commands, including the original Newmark syntax and all phases through 360 s. |
| `CentrifugeFull.tcl` | The full benchmark with Newmark and Rayleigh commands in the syntax accepted by OpenSees 3.7.1. |
| `CentrifugeCapped.tcl` | The 3.7.1 analysis through both gravity stages and the first 0.15 s of shaking. |

## Rules and tables

- [mesh.bin](mesh.bin) holds the irregular GiD connectivity and coordinates for 4,204 soil nodes in a 10,940-byte LZMA table. There are four signed 16-bit node-tag deltas per quad in original command order. A second stream stores signed 32-bit coordinate residual pairs in 0.1 mm units. When three corners are known, the missing position uses the parallelogram rule plus its stored residual. Otherwise it uses a delta from the previous decoded point. [mesh.py](mesh.py) expands this table and generates 32 laminar-box and duplicate nodes with 0.125 m ring spacing.
- [model.py](model.py) assigns material and conductivity by element tag ranges. It derives base fixities from elevation, laminar-box ties from side positions, and loam–sand ties from the 0.01 m vertical separation of matching nodes. Small tag sets preserve the nine reversals of retained node and the alternate pressure assignments.
- [pore_pressures.tsv](pore_pressures.tsv) stores the prescribed pressures on 19 surface nodes and 182 drain nodes. The model generator expands the rows into the original series and support motions, and generates the hydrostatic surface loads.
- [analysis.py](analysis.py) expands the nine phases from three adaptive stepping schedules and recurring recorder groups. It preserves the original analysis order, including any recorders the original leaves active between phases.

## Reproducing the verification

```sh
uv run python -m scripts.centrifuge_recon.verify /path/to/CentrifugeExample.tcl
uv run python -m scripts.centrifuge_recon.verify /path/to/CentrifugeExample.tcl --run-opensees
uv run python -m scripts.centrifuge_recon.extract_mesh /path/to/CentrifugeExample.tcl /tmp/centrifuge-mesh.bin
```

The verifier checks the source digest and parses the full original and generated decks. All 9,803 executable command lines match in order after normalizing indentation, node and load number spelling, and whitespace in constant time-series lists. The runtime comparison uses the pinned OpenSees 3.7.1 binary and Bubblewrap. The original Newmark calls are rejected by 3.7.1, and the fallback integrator fails during plastic gravity. Applying the same syntax translation and shake cap to both sources gives all eight gravity steps, three converged shaking steps, and byte-identical output in all 11 recorders.

# OpenSees example fixtures

Inputs for the direct-Tcl parity cases in `tests/validation/*/direct_tcl_case.json`. The harness converts each entry script to Strut input at test time and runs the same script through OpenSees.

## What is committed

- `OpenSeesExamplesBasic/`, `OpenSeesExamplesAdvanced/`: example Tcl scripts from the OpenSees wiki's Basic and Advanced example manuals.
- `OpenSeesSource/EXAMPLES/verification/`, `OpenSeesSource/EXAMPLES/ExampleScripts/`, and `OpenSeesSource/EXAMPLES/ExamplesForTesting/` preserve the corresponding paths in the [OpenSees `v3.7.1` EXAMPLES tree](https://github.com/OpenSees/OpenSees/tree/v3.7.1/EXAMPLES). Each entry script's direct-Tcl manifest links to its upstream file. Recorder instrumentation and headless compatibility edits are applied to run-time copies.
- `STAIRlabGallery/` contains Tcl entry scripts and their helpers from the [STAIRlab OpenSees gallery](https://gallery.stairlab.io/), pinned at [`fd3958f`](https://github.com/peer-open-source/opensees-gallery/tree/fd3958fde8a3d0a350321b3fbdd3f415ee16e2a2). The [BSD 3-Clause license](STAIRlabGallery/LICENSE) accompanies the fixtures. The case manifests link to the original files and include model citations when available.
- `reference-data.lock.json`: the manifest of everything that is not committed.

The OpenSees examples are covered by the OpenSees license in [`COPYRIGHT.md`](../../../COPYRIGHT.md).

Gallery copies of wiki models are omitted. The Chopra two-storey frames, SteelBuilding2D elastic frame, and Other consolidation, pile, and moment-curvature examples already have wiki fixtures.

## What is downloaded

These are listed in `reference-data.lock.json` and in the generated `.gitignore`:

- **Ground motions** (`*.AT2`, `*.acc`, `A10000.tcl`, ...). These are PEER and other earthquake records, which have their own terms of use. Each file is pinned by its own sha256 and by the sha256 of the wiki download it comes from.
- The gallery's `elCentro.at2` is installed into both `content/validation/Basic` and `content/en/examples/NewmarkFamily` from commit-pinned raw URLs and verified by SHA-256.
The direct-Tcl parity cases read these files. Unit tests do not: they copy the example Tcl into a temporary directory and write synthetic motions from `scripts/synthetic_ground_motion.py` next to it. Never write synthetic files over the real ones here.

The [OpenSees `v3.7.1` `ExampleScripts/ARL360.at2` record](https://github.com/OpenSees/OpenSees/blob/v3.7.1/EXAMPLES/ExampleScripts/ARL360.at2) is the Northridge Arleta 360° component used by the wiki's RC Frame Earthquake analysis. The harness copies the pinned record and the upstream `ExampleScripts/ReadRecord.tcl` into that example's run directory.

## What is generated

**The mega-tall benchmark model** (`OpenSeesExamples/MegaTallBuildingBenchmark/Model1/`) is never downloaded. [`scripts/megatall_recon/`](../../../scripts/megatall_recon/README.md) rebuilds all of its files from about 100 KB of rules and tables in about 2 s, with a synthetic `GMX.txt` in place of the original ground motion. `setup_reference_data.py` and the pytest session both regenerate it when the generator changes (tracked by the `.strut-generated` stamp). OpenSees gives the same gravity reactions, settlements and periods for the generated model and the original to about 1e-6.

**The centrifuge model** (`OpenSeesManual/Centrifuge/`) is built by [the centrifuge generator](../../../scripts/centrifuge_recon/README.md) from mesh rules and compact tables. It includes the published Tcl reconstruction and full and capped entries for the pinned OpenSees binary.

**HexaExtrusion** uses a [compact Tcl generator](STAIRlabGallery/content/en/examples/HexaExtrusion/hexa_generated.tcl) for the gallery's 947 KB mesh. It extrudes the original cross section across 100 bays, retaining the coordinates, connectivity, tags, boundary conditions, and loads. An OpenSees 3.7.1 comparison produced identical 17-digit displacement recorder output at tip nodes 12001, 12070, and 12120.

`uv run scripts/setup_reference_data.py` installs them; `run_tests.py` runs it for you. Each download happens once per machine into `~/.cache/strut/reference/` and is copied into place in every checkout. The files are copied rather than linked because the harness writes outputs next to the scripts it mirrors.

## Refreshing from the wiki

Only maintainers need this, when adding or updating examples:

```bash
uv run python scripts/setup-opensees-docs.py --root /tmp/opensees-wiki   # rate-limited scrape, ~3 min
uv run python scripts/setup_reference_data.py vendor-wiki --from /tmp/opensees-wiki
git diff --stat tests/fixtures/opensees
```

`vendor-wiki` replaces the committed wiki Tcl and rewrites the manifest's `files` and the `.gitignore`. It uses the provenance file the scraper writes. `trees` and `git` entries in the manifest are edited by hand.

## Published wall models

`OpenSeesWallExamples/` contains the original Tcl from the following
archives. Archive hashes are pinned in `reference-data.lock.json`.

| Directory | Source |
| --- | --- |
| `MVLEM_RW2/` | OpenSees Wiki, [MVLEM Example 1](https://opensees.berkeley.edu/wiki/images/d/d0/Example_1._MVLEM.zip), wall RW2 tested by Thomsen and Wallace (1995). |
| `SFI_MVLEM_RW_A15_P10_S78/` | OpenSees Wiki, [SFI-MVLEM Example 1](https://opensees.berkeley.edu/wiki/images/f/fd/Example_1-SFI_MVLEM.zip), wall RW-A15-P10-S78 tested by Tran and Wallace (2012). |
| `SFI_MVLEM_FiveStory/` | OpenSees Wiki, [SFI-MVLEM Example 2](https://opensees.berkeley.edu/wiki/images/2/23/Example_2-SFI_MVLEM.zip), five-story wall-frame. Setup installs the archived `MCEScaledEQ1X.acc`. |
| `PURR4348/` | Broberg and Varma (2023), [OpenSees Models for C-PSW/CF Archetype Structures](https://doi.org/10.4231/9H4P-Z894), CC BY 4.0. |

The Example 1 SFI-MVLEM archive also includes a Windows executable. The fixture
contains its Tcl inputs, and the harness uses the project's pinned OpenSees
binary.

The PURR concrete cases isolate the publication's PG-1B monotonic compression
and cyclic material checks. In the mirrored scripts, both cases restrain
transverse motion and end rotation to keep the test uniaxial on the concrete
crushing plateau. The cyclic case retains the first two loading branches and
reuses the Static analysis object across the reversal. The archived loop
encounters a singular tangent in OpenSees 3.7.1 after the second branch.

The RW2 case compares gravity and both cycles at the first published drift
level. Its mirrored Tcl skips repeated displacement positions, tightens the
analysis convergence tolerance, and increases recorder text precision. The
comparison checks each recorded displacement and global element force.

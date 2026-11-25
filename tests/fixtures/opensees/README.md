# OpenSees example fixtures

Inputs for the direct-Tcl parity cases in `tests/validation/*/direct_tcl_case.json`. The harness converts each entry script to Strut input at test time and runs the same script through OpenSees, so only the original Tcl is stored here.

## What is committed

- `OpenSeesExamplesBasic/`, `OpenSeesExamplesAdvanced/`: example Tcl scripts from the OpenSees wiki's Basic and Advanced example manuals.
- `OpenSeesSource/EXAMPLES/ExampleScripts/Example3.1.tcl`: from the OpenSees repository at `v3.7.1`.
- `reference-data.lock.json`: the manifest of everything that is not committed.

The OpenSees examples are covered by the OpenSees license in [`COPYRIGHT.md`](../../../COPYRIGHT.md).

## What is downloaded

These are listed in `reference-data.lock.json` and in the generated `.gitignore`:

- **Ground motions** (`*.AT2`, `*.acc`, `A10000.tcl`, ...). These are PEER and other earthquake records, which have their own terms of use. Each file is pinned by its own sha256 and by the sha256 of the wiki download it comes from.
- **The mega-tall benchmark model** (`OpenSeesExamples/MegaTallBuildingBenchmark/`), extracted from a pinned `Model1.rar`.

`uv run scripts/setup_reference_data.py` installs them; `run_tests.py` runs it for you. Each download happens once per machine into `~/.cache/strut/reference/` and is copied into place in every checkout. The files are copied rather than linked because the harness writes outputs next to the scripts it mirrors.

## Refreshing from the wiki

Only maintainers need this, when adding or updating examples:

```bash
uv run python scripts/setup-opensees-docs.py --root /tmp/opensees-wiki   # rate-limited scrape, ~3 min
uv run python scripts/setup_reference_data.py vendor-wiki --from /tmp/opensees-wiki
git diff --stat tests/fixtures/opensees
```

`vendor-wiki` replaces the committed wiki Tcl and rewrites the manifest's `files` and the `.gitignore`. It uses the provenance file the scraper writes. `trees` and `git` entries in the manifest are edited by hand.

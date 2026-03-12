# Benchmark

Tools for comparing native Linux OpenSees, OpenSeesMP, and Strut runtime performance.

## Runner

Use `scripts/run_benchmarks.py` to run the default enabled validation cases on
the selected engines, or a custom case list from `tests/validation/`. The Strut solver is precompiled before
timing (cached at `build/strut/strut`, or `build/strut/strut_profile` for `--profile` runs).

Examples:

```bash
uv run scripts/run_benchmarks.py
uv run scripts/run_benchmarks.py --cases elastic_beam_cantilever,elastic_frame_portal
uv run scripts/run_benchmarks.py --cases tests/validation/elastic_frame_two_story/elastic_frame_two_story.json
uv run scripts/run_benchmarks.py --engine strut
uv run scripts/run_benchmarks.py --engine all
uv run scripts/run_benchmarks.py --no-batch
uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --gen-frame-element forceBeamColumn2d --cases force_beam_column2d_fiber_frame_18bay_17story --no-archive
uv run scripts/run_benchmarks.py --benchmark-suite root_cause_v1
uv run scripts/run_benchmarks.py --benchmark-suite opt_fast_v1 --engine strut --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --benchmark-suite opt_full_v1 --engine all --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --list-benchmark-suites
uv run scripts/run_benchmarks.py --engine strut --cases opensees_example_rc_frame_earthquake --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --engine strut --cases force_beam_column3d_portal_benchmark,nonlinear_beam_column3d_portal_benchmark,disp_beam_column3d_portal_benchmark --no-archive
uv run scripts/compare_benchmarks.py benchmark/results-profile/summary.json /tmp/phase7results_after/summary.json --max-regression-pct 5 --min-regression-us 50 --require-improvement opensees_example_rc_frame_earthquake=10
```

Optimization loop suites:

- `opt_fast_v1`: short feedback loop for iterative profiling work.
- `opt_full_v1`: broader end-to-end suite for milestone checks, including the 3D beam-column portal benchmarks.

Upcoming element benchmarks (disabled until element support lands):

- `elastic_truss_basic`
- `elastic_two_node_link_basic`
- `elastic_zero_length_basic`
- `elastic_four_node_quad_basic`
- `elastic_shell_basic`

Run these explicitly once enabled:

```bash
uv run scripts/run_benchmarks.py --cases elastic_truss_basic,elastic_four_node_quad_basic --include-disabled
```

## Results

- `benchmark/results/` contains the latest run (summary plus last-run outputs).
- Runs with `--profile` write to `benchmark/results-profile/` and default to `--no-archive`.
- `--profile` builds and reuses a dedicated `build/strut/strut_profile` binary, so switching between normal and profiled runs does not overwrite the default solver.
- `--profile <DIR>` works in both per-case and batch runs and writes per-case files as `<case>.speedscope.json` into `DIR`.
- Compute-only outputs are written to `benchmark/results/opensees/`, `benchmark/results/openseesmp/`, and `benchmark/results/strut/` (or the `results-profile` equivalents when profiling).
- `metadata.json` records machine/build/run metadata for reproducible baseline/perf comparisons.
- `uv run scripts/compare_benchmarks.py` compares two `summary.json` files and can fail on regressions or unmet improvement targets.
- Use `--min-regression-us` to ignore tiny absolute timing swings on very small cases.
- `phase_summary.csv` records per-case phase timing columns (parse/model-build/assembly/solve/output).
- `phase_rollup.csv` records phase-level aggregates (mean/median/min/max).
- `benchmark/archive/` contains timestamped summary snapshots.
- The benchmark runner is compute-only and does not perform parity checks. Run `uv run run_tests.py` for parity validation before benchmarking.
- OpenSees, OpenSeesMP, and Strut are timed from recorder-stripped inputs.
- The runner prints progress per case and pass while running.
- `analysis_time_us.txt` is solve-only for all engines (OpenSees/OpenSeesMP `analyze/eigen`, Strut solve phase).
- Batch mode is enabled for OpenSees by default; use `--no-batch` for single-case process timings. OpenSeesMP and Strut run per case.
- OpenSees batch mode prewarms `eigen` once outside case timers to remove first-call initialization skew.
- `--engine both` runs the legacy OpenSees + Strut pair. `--engine all` runs OpenSees + OpenSeesMP + Strut.
- `uv run scripts/run_benchmarks.py` now defaults to `--engine all`.
- Default runs benchmark all enabled cases in `tests/validation/` on the selected engines.
- `--include-disabled` adds disabled validation cases to the default run.
- Benchmark participation is controlled by case selection plus `enabled`, not by a separate `status` flag.
- Default batch runs (without `--cases` or `--gen-frame-*`) also auto-include generated medium-size frame benchmarks.

Both directories are ignored by git.

## Plots

The plot helper requires matplotlib:

```bash
uv run scripts/plot_benchmarks.py --output benchmark/results/plots.pdf
uv run scripts/plot_markdown_benchmarks.py --output docs/benchmark-opensees-examples.md
uv run scripts/plot_markdown_benchmarks.py --mp --output docs/benchmark-opensees-examples.md
```

Benchmark PDFs use a log-scale y-axis for both recent-case and archive charts.
The Markdown mode writes separate Mermaid charts plus tables for enabled `opensees_example_*` cases, split by `ex1`, `ex2`, and so on. By default it plots OpenSees and Strut; pass `--mp` to include OpenSeesMP as well.

Group and order cases in the bar chart (default is prefix grouping, disable with `--group-by none`):

```bash
uv run scripts/plot_benchmarks.py --group-by prefix
uv run scripts/plot_benchmarks.py --group-by config --group-config benchmark/groups.json
```

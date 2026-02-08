# Benchmark

Tools for comparing OpenSees (Wine) and Strut (Mojo) runtime performance.

## Runner

Use `scripts/run_benchmarks.py` to run the default elastic cases or a custom
case list from `tests/validation/`. The Mojo solver is precompiled before
timing (cached at `build/mojo/strut`).

Examples:

```bash
uv run scripts/run_benchmarks.py
uv run scripts/run_benchmarks.py --cases elastic_beam_cantilever,elastic_frame_portal
uv run scripts/run_benchmarks.py --cases tests/validation/elastic_frame_two_story/elastic_frame_two_story.json
uv run scripts/run_benchmarks.py --engine mojo
uv run scripts/run_benchmarks.py --no-batch
uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --gen-frame-element forceBeamColumn2d --cases force_beam_column2d_fiber_frame_18bay_17story --no-archive
```

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
- Compute-only outputs are written to `benchmark/results/opensees_compute/` and `benchmark/results/mojo_compute/` (or the `results-profile` equivalents when profiling).
- `benchmark/archive/` contains timestamped summary snapshots.
- When running both engines, the runner compares outputs and fails on mismatch.
- The runner performs a second pass without recorders to estimate compute-only time.
- The runner prints progress per case and pass while running.
- `analysis_time_us.txt` is solve-only for both engines (OpenSees `analyze/eigen`, Mojo solve phase).
- Batch mode is enabled for both engines by default; use `--no-batch` for single-case process timings.
- OpenSees batch mode prewarms `eigen` once outside case timers to remove first-call initialization skew.
- Default batch runs (without `--cases` or `--gen-frame-*`) auto-include generated medium-size frame benchmarks.

Both directories are ignored by git.

## Plots

The plot helper requires matplotlib:

```bash
uv run scripts/plot_benchmarks.py --output benchmark/results/plots.pdf
```

Group and order cases in the bar chart (default is prefix grouping, disable with `--group-by none`):

```bash
uv run scripts/plot_benchmarks.py --group-by prefix
uv run scripts/plot_benchmarks.py --group-by config --group-config benchmark/groups.json
```

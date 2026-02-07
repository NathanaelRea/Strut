# Benchmark

Tools for comparing OpenSees (Wine) and Strut (Mojo) runtime performance.

## Runner

Use `scripts/run_benchmarks.py` to run the default elastic cases or a custom
case list from `tests/validation/`. The Mojo solver is precompiled before
timing (cached at `build/mojo/strut`).

Examples:

```bash
python scripts/run_benchmarks.py
python scripts/run_benchmarks.py --cases elastic_beam_cantilever,elastic_frame_portal
python scripts/run_benchmarks.py --cases tests/validation/elastic_frame_two_story/elastic_frame_two_story.json
python scripts/run_benchmarks.py --engine mojo
python scripts/run_benchmarks.py --batch-opensees
```

Upcoming element benchmarks (disabled until element support lands):

- `elastic_truss_basic`
- `elastic_two_node_link_basic`
- `elastic_zero_length_basic`
- `elastic_four_node_quad_basic`
- `elastic_shell_basic`

Run these explicitly once enabled:

```bash
python scripts/run_benchmarks.py --cases elastic_truss_basic,elastic_four_node_quad_basic --include-disabled
```

## Results

- `benchmark/results/` contains the latest run (summary plus last-run outputs).
- Runs with `--profile` write to `benchmark/results-profile/` and default to `--no-archive`.
- Compute-only outputs are written to `benchmark/results/opensees_compute/` and `benchmark/results/mojo_compute/` (or the `results-profile` equivalents when profiling).
- `benchmark/archive/` contains timestamped summary snapshots.
- When running both engines, the runner compares outputs and fails on mismatch.
- The runner performs a second pass without recorders to estimate compute-only time.
- The runner prints progress per case and pass while running.
- The OpenSees total pass writes `analysis_time_us.txt` in each case output directory.

Both directories are ignored by git.

## Plots

The plot helper requires matplotlib:

```bash
python scripts/plot_benchmarks.py --output benchmark/results/plots.pdf
```

Group and order cases in the bar chart (default is prefix grouping, disable with `--group-by none`):

```bash
python scripts/plot_benchmarks.py --group-by prefix
python scripts/plot_benchmarks.py --group-by config --group-config benchmark/groups.json
```

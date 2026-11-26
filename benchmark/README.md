# Benchmark

Tools for comparing native Linux OpenSees, OpenSeesMP, and Strut runtime performance.

## Micro

Focused section/material microbenchmarks live in `benchmark/micro/`.

Examples:

```bash
uv run python benchmark/micro/run_section_micro.py
uv run python benchmark/micro/run_material_compare.py
uv run python benchmark/micro/run_material_compare.py --compare steel02_uniaxial,steel02_fiber_scalar
```

## Synthetic cases

`scripts/gen_synthetic_case.py` generates models that scale along one axis each, so a
single cost center can be measured apart from the others. The OpenSees example suite
tops out at 40 DOFs, which hides anything that grows with model size.

| Family | Stresses | Scale with |
| --- | --- | --- |
| `rc_frame2d` | 2D fiber section state determination | `--bays`, `--stories`, `--fiber-subdiv` |
| `rc_frame3d` | biaxial fiber sections, multi-point constraints | `--bays`, `--bays-z`, `--stories` |
| `elastic_frame2d` | global assembly and the linear solve only | `--bays`, `--stories` |
| `shell_wall` | sparse storage and factorization (24 DOF elements) | `--bays`, `--stories` |

```bash
uv run python scripts/gen_synthetic_case.py rc_frame2d --bays 6 --stories 12 --steps 100 --output .tmp/synthetic/rc2d_6x12.json
uv run python scripts/gen_synthetic_case.py rc_frame3d --bays 3 --bays-z 3 --stories 6 --steps 50 --output .tmp/synthetic/rc3d_3x3x6.json
uv run python scripts/gen_synthetic_case.py elastic_frame2d --bays 30 --stories 30 --steps 50 --output .tmp/synthetic/el_30.json
uv run python scripts/gen_synthetic_case.py elastic_frame2d --bays 30 --stories 30 --steps 50 --nonlinear-driver --output .tmp/synthetic/el_nl_30.json
uv run python scripts/gen_synthetic_case.py shell_wall --bays 20 --stories 20 --steps 20 --output .tmp/synthetic/shell_20.json
```

`--rigid-diaphragm` ties each floor with multi-point constraints (equalDOF in 2D,
rigidDiaphragm in 3D) and switches the constraint handler to Transformation.
`--system` and `--algorithm` select the linear system and nonlinear algorithm.

## A/B comparing two solver builds

`scripts/ab_compare.py` runs a baseline and a candidate solver binary on identical
inputs, compares every recorder output numerically, and reports solve time for both.
It needs no OpenSees binary, so it is the fast loop for refactors: "did results
change, and how much faster is it?"

```bash
cp build/strut/strut build/baseline/strut   # before the change
uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut
uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut --cases opensees_example_ex5_frame2d_eq_uniform_rc_fiber --repeat 3
uv run scripts/ab_compare.py --baseline build/baseline/strut --candidate build/strut/strut --no-validation-cases --extra-json .tmp/synthetic/*.json
```

- Runs keep recorders on, so times include output formatting; use `--min-ms` to hide tiny cases.
- `DIFF` means an output differs beyond `--rtol`/`--atol`. Changes that reorder floating-point
  sums can move results by the nonlinear convergence tolerance (about 1e-6 relative in
  forces); rerun the case with a tighter `tol` to tell that apart from a real change.
- This checks Strut against Strut. Parity against OpenSees is still `uv run run_tests.py`.

## Pull request benchmark

`.github/workflows/benchmark.yml` times every PR that touches the solver or scripts
against its base branch. Timings from different shared runners differ by tens of
percent, so there is no stored baseline: the job builds both solvers on one runner and
`scripts/ci_benchmark.py` runs them alternately on the same inputs. The table lands in
the job summary and a PR comment that each push updates.

The suite is about fifteen timed cases of 20 ms to 1 s, picked to cover the code paths
rather than every model: 2D and 3D force- and displacement-based fiber frames (Linear,
PDelta, Corotational; rigid diaphragms), elastic frames through the linear and nonlinear
transient drivers, a shell wall on SparseSYM, mega-tall slices (layered ConcreteS shells,
trusses, diaphragms) on UmfPack and on SparseSYM at 1 and 4 threads, and a few OpenSees
examples (Ex5/Ex6 frames, a quad solid, a DisplacementControl pushover). Every other
validation case that runs without pinned ground motions is too short to time, so it only
gets its outputs compared.

The job fails when the PR crashes on a case main runs, or when a timed case is more than
15% slower in two measurements in a row. Output differences are flagged but do not fail
the job, since fixes change results on purpose. Large-model scaling (the full mega-tall
case) is still a local measurement.

```bash
uv run scripts/ci_benchmark.py prepare
uv run scripts/ci_benchmark.py compare --baseline build/baseline/strut --candidate build/strut/strut
uv run scripts/ci_benchmark.py compare --baseline build/baseline/strut --candidate build/strut/strut --only megatall_slice_sparse_t4
```

## Sampling profiler

`scripts/profile_sample.py` gives a source-line profile without `perf` or root. It
builds a line-table solver (`build/strut/strut_g1`) and a small `perf_event_open`
sampler (`scripts/profiling/usample.c`), then attributes samples through inlining.
`--profile` (speedscope) answers "which phase"; this answers "which code".

```bash
uv run scripts/profile_sample.py --case opensees_example_ex5_frame2d_eq_uniform_rc_fiber
uv run scripts/profile_sample.py --input .tmp/synthetic/rc3d_3x3x6.json --top 40
```

It samples the main thread only. In a threaded phase that is one worker's share of
the work, so compare phases by wall time, not by sample counts.

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
uv run scripts/run_benchmarks.py --engine both --strut-single-thread --repeat 3
uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --gen-frame-element forceBeamColumn2d --cases force_beam_column2d_fiber_frame_18bay_17story --no-archive
uv run scripts/run_benchmarks.py --benchmark-suite root_cause_v1
uv run scripts/run_benchmarks.py --benchmark-suite opt_fast_v1 --engine strut --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --benchmark-suite opt_full_v1 --engine all --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --list-benchmark-suites
uv run scripts/run_benchmarks.py --engine strut --cases opensees_example_rc_frame_earthquake --profile benchmark/speedscope --no-archive
uv run scripts/run_benchmarks.py --engine strut --cases force_beam_column3d_portal_benchmark,nonlinear_beam_column3d_portal_benchmark,disp_beam_column3d_portal_benchmark --no-archive
uv run scripts/compare_benchmarks.py benchmark/results-profile/summary.json /tmp/phase7results_after/summary.json --max-regression-pct 5 --min-regression-us 50 --require-improvement opensees_example_rc_frame_earthquake=10
```

`--strut-single-thread` adds a second Strut pass with `STRUT_THREADS=1`, stored as
`strut_single_thread` next to the default-threads `strut` result. Use it when the
question is how much of a speedup is threading and how much is the code.

### Large models

The example suite tops out under 200 DOFs, so none of it reaches the threaded
code. Two kinds of larger input fill that gap. Both are
generated, so build them first; they are timed separately from the examples
because OpenSees needs minutes on them.

**Mega-tall building.** The full model runs 10 gravity and 2000 transient steps,
which takes hours in OpenSees. `scripts/build_megatall_benchmark_case.py` writes the
same model with the transient stage cut to 30 steps, as one input every engine runs:

```bash
uv run scripts/build_megatall_benchmark_case.py
uv run scripts/run_benchmarks.py --engine both --no-batch --strut-single-thread --no-archive \
    --cases benchmark/.tmp/megatall_model1_40steps/megatall_model1_40steps.json \
    --output-root benchmark/results-megatall
```

Expect about 15 minutes, nearly all of it OpenSees.

**Synthetic 3D frames.** Fiber beam-column frames big enough to thread:

```bash
uv run python scripts/gen_synthetic_case.py rc_frame3d --bays 5 --bays-z 5 --stories 8 --steps 40 \
    --system SparseSYM --gravity nodal --name rc_frame3d_force_5x5x8 \
    --output benchmark/.tmp/rc_frame3d_force_5x5x8/rc_frame3d_force_5x5x8.json
uv run python scripts/gen_synthetic_case.py rc_frame3d --bays 5 --bays-z 5 --stories 8 --steps 40 \
    --system SparseSYM --gravity nodal --element dispBeamColumn3d --geom-transf PDelta \
    --name rc_frame3d_disp_pdelta_5x5x8 \
    --output benchmark/.tmp/rc_frame3d_disp_pdelta_5x5x8/rc_frame3d_disp_pdelta_5x5x8.json
uv run scripts/run_benchmarks.py --engine both --no-batch --strut-single-thread --no-archive --repeat 3 \
    --cases benchmark/.tmp/rc_frame3d_force_5x5x8/rc_frame3d_force_5x5x8.json,benchmark/.tmp/rc_frame3d_disp_pdelta_5x5x8/rc_frame3d_disp_pdelta_5x5x8.json \
    --output-root benchmark/results-synthetic
```

The runner takes a generated case only as `<name>/<name>.json`.

### Do the large models match OpenSees?

Yes, to the six digits OpenSees prints. `scripts/parity_report.py` runs a case in
both engines and prints, for every recorder group, the largest difference divided
by the group's peak value:

```bash
uv run scripts/build_megatall_benchmark_case.py --tight
uv run scripts/parity_report.py --strut-system SparseSYM \
    benchmark/.tmp/megatall_model1_40steps_tight/megatall_model1_40steps_tight.json
```

| Check | Worst difference / peak |
| --- | ---: |
| Mega-tall, 10 gravity + 30 transient steps, tolerance `1e-7`: 116 base reactions, 128 story displacements, 127 drifts | 4e-6 |
| Whole building, static gravity plus 2% lateral: 1,356 nodes on all six DOFs, 116 reactions | 4e-6 |
| Seven slices of the model, static and under the strongest part of the record (`tests/validation/test_megatall_slices.py`) | 4e-6 |
| Synthetic 3D frames, roof displacement over the run | 5e-6 |

The mega-tall comparison runs OpenSees with `UmfPack` and a tolerance of `1e-7`
instead of the model's own `SparseSYM` and `NormDispIncr 0.05`. With the model's own
settings OpenSees stops 2.5% (story displacements) and 4.5% (drifts) away from its
own converged answer, because its `SparseSYM` is inexact on rigid-diaphragm systems.
Strut, with the same loose settings, is within 0.01% and 0.1% of that answer. The
timings above use the model's own settings in both engines.

`scripts/megatall_slice.py` cuts a few stories out of the model, with their real
nodes, sections, beam orientations and diaphragms, fixes the lowest level and loads
it, optionally followed by an earthquake stage. A slice runs in seconds, and
`--types` narrows a mismatch to one element type:

```bash
uv run scripts/megatall_slice.py --z-min 0 --z-max 12 --transient-steps 60
uv run scripts/parity_report.py \
    benchmark/.tmp/megatall_slice_z0_12_eq60/megatall_slice_z0_12_eq60.json
```

Synthetic frames with `--rigid-diaphragm` and `SparseSYM` cannot be compared, for the
same reason: OpenSees does not converge on them. With `--system UmfPack` they match
to 1e-6.

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
uv run scripts/plot_markdown_benchmarks.py --large-results benchmark/results-megatall/summary.json \
    --large-results benchmark/results-synthetic/summary.json
```

Benchmark PDFs use a log-scale y-axis for both recent-case and archive charts.
The Markdown mode writes a summary table, one Mermaid chart per large model, and the enabled `opensees_example_*` cases grouped by how long OpenSees takes (over one second, 10 ms to one second, under 10 ms), so each chart has one time scale. The smallest group gets a table only. By default it plots OpenSees and Strut; pass `--mp` to include OpenSeesMP as well.
When the summary came from a `--strut-single-thread` run, the charts show `opensees`, `strut` (one thread) and `strut-<threads>`.
`--large-results` takes the summary of a separate run and gives each of its cases its own chart under "Large models".
`--note "..."` prints a sentence under the header; use it to say what else the machine was doing during the run.

Group and order cases in the bar chart (default is prefix grouping, disable with `--group-by none`):

```bash
uv run scripts/plot_benchmarks.py --group-by prefix
uv run scripts/plot_benchmarks.py --group-by config --group-config benchmark/groups.json
```

## Runtime knobs

| Variable | Effect |
| --- | --- |
| `STRUT_THREADS=N` | Worker threads for element state determination (layered shells, 3D beam-columns) and sparse factorization. Default: performance cores. `1` runs the whole solve on one thread. |
| `STRUT_VERBOSE=1` | Print sparse LDLT statistics (ordering, factor size, flops) or why the solver fell back. |
| `STRUT_DUMP_SPARSE_PATTERN=path` | Write the lower-triangular CSC pattern of the first sparse system, for `build/strut/sparse_ldlt_bench --pattern path`. |

Threaded element sweeps compute in parallel and scatter serially in element order, so
results are bit-identical for any thread count.

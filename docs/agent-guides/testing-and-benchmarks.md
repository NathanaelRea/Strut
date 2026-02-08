# Testing and Benchmarks

## Test Commands

- `uv run run_tests.py`
- `uv run run_tests.py --all`
- `uv run run_tests.py --case tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json`

## Benchmark Commands

- `scripts/gen_frame_case.py --bays 18 --stories 17 --output /tmp/frame.json --disabled`
- `uv run scripts/run_benchmarks.py --cases /tmp/frame.json --no-archive`
- `uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --batch --no-archive`
- `uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --gen-frame-element forceBeamColumn2d --cases force_beam_column2d_fiber_frame_18bay_17story --no-archive`

## Benchmark Iteration Rule

- Use `--no-archive` while iterating to avoid polluting `benchmark/archive`.
- Runs with `--profile` default to `--no-archive`.

## Mojo Build Behavior

- Mojo is compiled on first run and cached at `build/mojo/strut`.
- Precompile with `scripts/build_mojo_solver.sh`.

## Output Comparison Strategy

Strut uses an `isclose`-style check equivalent to NumPy:

```text
absolute(a - b) <= (atol + rtol * absolute(b))
```

- `b` is the OpenSees reference.
- `rtol` and `atol` are defined in `scripts/compare_case.py`.
- The comparison is not symmetric in `a` and `b`.

## Tolerance Tuning

- Adjust `REL_TOL` and `ABS_TOL` in `scripts/compare_case.py`.
- For near-zero reference values, `atol` dominates.
- Use the smallest tolerances that avoid false negatives while still catching regressions.

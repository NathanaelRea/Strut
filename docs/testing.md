# Testing

This document describes how Strut compares outputs and how tolerances are applied in tests.

## Running Tests

- `uv run run_tests.py`
- `uv run run_tests.py --all`
- `uv run run_tests.py --case tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json`
- Generate a large frame benchmark case: `scripts/gen_frame_case.py --bays 18 --stories 17 --output /tmp/frame.json --disabled`
- Benchmark a generated case: `uv run scripts/run_benchmarks.py --cases /tmp/frame.json --no-archive`
- Benchmark with auto-generation: `uv run scripts/run_benchmarks.py --gen-frame-bays 18 --gen-frame-stories 17 --batch --no-archive`
- When iterating on benchmarks, always include `--no-archive` to avoid polluting `benchmark/archive`. Runs with `--profile` default to `--no-archive`.

Mojo is compiled on first run and cached at `build/mojo/strut`. To precompile:

```bash
scripts/build_mojo_solver.sh
```

## Comparison Strategy

Strut uses an `isclose`-style comparison, equivalent to NumPy's `numpy.isclose`:

```
absolute(a - b) <= (atol + rtol * absolute(b))
```

- The reference value is `b` (OpenSees output).
- `rtol` and `atol` are defined in `scripts/compare_case.py`.
- This is **not symmetric** in `a` and `b`.

## Adjusting Tolerances

If tolerances need adjustment, change `REL_TOL` and `ABS_TOL` in `scripts/compare_case.py`.

Be careful with values close to zero; for very small reference values, `atol` dominates. Use the smallest tolerance that prevents false negatives while still catching real regressions.

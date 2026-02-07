# Testing

This document describes how Strut compares outputs and how tolerances are applied in tests.

## Running Tests

- `uv run scripts/run_tests.py`
- `uv run scripts/run_tests.py --all`
- `uv run scripts/run_tests.py --case tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json`

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

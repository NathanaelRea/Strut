# Microbenchmarks

Focused local harnesses for section/fiber and material implementation timing.

Examples:

```bash
uv run python benchmark/micro/run_section_micro.py
uv run python benchmark/micro/run_section_micro.py --materials steel01,steel02 --target-fibers 200000 --iterations 32 --samples 5
uv run python benchmark/micro/run_section_micro.py --case tests/validation/force_beam_column2d_fiber_cantilever/force_beam_column2d_fiber_cantilever.json --section-id 1

uv run python benchmark/micro/run_material_compare.py
uv run python benchmark/micro/run_material_compare.py --list
uv run python benchmark/micro/run_material_compare.py --compare steel01_uniaxial,steel01_fiber_scalar
```

`run_section_micro.py` now defaults to a synthetic sweep with one full FiberSection2d per uniaxial material, and prints the fastest full-section update time in a compact material-by-scenario table.
By default that sweep uses 500 fibers per section, 10 section instances per timed sample, and reports `ns/fiber`.
`run_material_compare.py` defaults to a curated multi-case bundle so it exercises all supported uniaxial materials: `Elastic`, `Steel01`, `Concrete01`, `Steel02`, and `Concrete02`.

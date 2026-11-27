# GPU batching for beam fibers

This standalone benchmark replays Strut's existing FP64 Steel02 and
Concrete01 material updates. The mega-tall Model1 input supplies the material
parameters and the number of fibers across all beam integration points.
Each fiber follows a deterministic synthetic strain history with ten gravity
increments and thirty cyclic increments.

- The eight-thread CPU mode groups fibers by material and uses Strut's existing SIMD kernels.
- The tagged GPU mode keeps the fixture's fiber order and dispatches the material law per fiber inside one kernel per increment.
- The grouped GPU mode sorts fibers once and launches a specialized kernel for each material group per increment.

All modes use a structure of arrays for stress, tangent, and material history.
The timed loop includes the material update, committed history writes, and GPU
launches and synchronization. The GPU data stays resident throughout the timed
loop. Setup, sorting, initial upload, resets, and result readback happen outside
the timing. Each configuration uses one warmup followed by timed samples.

Run these commands from the repository root.

```bash
uv run python scripts/setup_reference_data.py install --fixtures-only
uv run python scripts/build_megatall_benchmark_case.py
HIP_VISIBLE_DEVICES=0 uv run python benchmark/gpu/fibers/run_fibers.py \
    --output benchmark/.tmp/fibers/full-replay
HIP_VISIBLE_DEVICES=0 uv run python benchmark/gpu/fibers/run_fibers.py \
    --output benchmark/.tmp/fibers/audit-replay --max-fibers 8192
```

The full run checks every fiber's final stress and tangent against the CPU
SIMD implementation. The sampled run checks stress and tangent at each step
against the separate scalar material implementation. The fixture uses Steel02
with zero initial stress and no isotropic strain shift. Each increment commits
its material state. The timings cover local constitutive work. They exclude
element strain recovery, section-force and tangent assembly, and the global
Newton and linear solves.

## RX 9070 XT results

The full run used 4,322,160 Steel02 fibers and 238,080 Concrete01 fibers,
for 182,409,600 updates. The CPU was a Ryzen 7 9800X3D. These are medians
of three samples.

| Mode | Forty increments | Million updates per second |
| --- | ---: | ---: |
| Grouped CPU SIMD, eight threads | 740.8 ms | 246.2 |
| Tagged GPU | 175.3 ms | 1,040.4 |
| Grouped GPU | 166.7 ms | 1,094.5 |

Grouping shortened GPU time by 4.9%. Both GPU modes agreed with the full
CPU SIMD run within `2.7e-15` in scaled peak stress and `1.1e-15` in scaled
peak tangent. The 8,192-fiber audit compared all forty increments against the
scalar material implementation and agreed within `3.5e-15` for every
mode. Each error is the maximum absolute difference divided by the peak
reference magnitude for that field and increment, with a denominator
floor of one.

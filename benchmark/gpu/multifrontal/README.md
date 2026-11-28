# STRUMPACK replay on Radeon

This spike replays the mega-tall stiffness matrices and right-hand sides captured
during the original 10 gravity and 30 earthquake steps. Both modes use
STRUMPACK's pivoted multifrontal LU in FP64 with METIS ordering and no
compression. STRUMPACK's separate GPU Cholesky front is CUDA-only in the
revision tested here. The HIP path uses its general LU front.

The reader expands Strut's symmetric lower CSC capture to full CSR, reorders
each pattern once, then updates and refactors the matrix at each tangent record.
It solves each captured right-hand side and checks the solution against Strut.
The CSV timing covers STRUMPACK ordering, value updates, factorization, and
solve calls. Trace I/O, CSR expansion, finite element work, and validation
are outside the timed calls. The replay checks a componentwise backward
error of at most `1e-8` and a relative infinity-norm solution error of at
most `1e-7`. It prints the number of stored LU entries once per stage.

## Build and run

The experiment used STRUMPACK commit `edc9a72cc128d82b912b41098aa2ff237f0f187d`
(v8.1.0), ROCm 7.1.1, a Radeon RX 9070 XT (`gfx1201`), and a Ryzen 7
9800X3D. Build STRUMPACK with HIP and OpenMP using the upstream installation
instructions. On Fedora, HIP also needs the rocThrust and hipCUB development
headers. A minimal configuration for this experiment uses
`STRUMPACK_USE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1201`,
`STRUMPACK_USE_MPI=OFF`, `TPL_ENABLE_MAGMA=OFF`, and
`STRUMPACK_USE_GPU_ARA=OFF`.

Build STRUMPACK with `BUILD_SHARED_LIBS=ON` and point the Mojo replay at
the resulting shared library. For an NVIDIA build, enable STRUMPACK's CUDA
backend in place of the HIP backend and set `CUDA_VISIBLE_DEVICES`. The Mojo
replay calls the same STRUMPACK C API.

```bash
mkdir -p benchmark/.tmp/multifrontal
uv run mojo build -I benchmark/gpu benchmark/gpu/multifrontal/replay.mojo \
    -o benchmark/.tmp/multifrontal/replay_mojo
export STRUT_STRUMPACK_LIBRARY=/path/to/libstrumpack.so
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores
export HIP_VISIBLE_DEVICES=0
```

The Fedora rocBLAS package used here lacked the top-level Tensile metadata for
`gfx1201`. The same-version AMD ROCm package provides that metadata and its
code objects. For this experiment, extracting the AMD package under
`"$spike/amd"` and setting the following variable allowed the Fedora libraries
to run DGEMM on the Radeon.

```bash
export ROCBLAS_TENSILE_LIBPATH=/path/to/rocm/lib/rocblas/library
```

```bash
spike=benchmark/.tmp/multifrontal
trace=benchmark/.tmp/replay/megatall.bin
"$spike/replay_mojo" cpu "$trace" > "$spike/cpu.csv"
"$spike/replay_mojo" gpu "$trace" > "$spike/gpu.csv"
```

## Measured replay

The following timings are from the original C++ replay.

| Mode | Reordering (s) | Value updates (s) | Factorization (s) | Solves (s) |
| --- | ---: | ---: | ---: | ---: |
| STRUMPACK CPU, eight threads | 1.42 | 6.38 | 33.49 | 3.61 |
| STRUMPACK HIP | 0.55 | 2.02 | 35.34 | 0.94 |

Each mode processed all 40 tangents and 50 right-hand sides. The largest
relative solution error against the capture was `7.09e-10` on the CPU and
`1.39e-9` on the GPU. The largest componentwise backward errors were
`2.63e-9` and `4.96e-10` respectively. The original GPU run used a
`1e-10` backward-error threshold and reported failure at the end. Its
recorded errors satisfy the current `1e-8` threshold.

STRUMPACK HIP took 36.28 seconds for factorization and solves. Strut's
GPU FP64 LDLᵀ replay used about 10.16 seconds for those phases. The algorithms
and ordering differ. The CPU STRUMPACK run stored about 96.1 million LU entries
per stage. The Strut NESDIS replay used about 40.4 million packed LDLᵀ entries.

- [STRUMPACK multifrontal GPU paper](https://escholarship.org/uc/item/7tv84567)
- [STRUMPACK installation guide](https://portal.nersc.gov/project/sparse/strumpack/master/installation.html)
- [AMD ROCm 7.1.1 rocBLAS package](https://repo.radeon.com/rocm/apt/7.1.1/pool/main/r/rocblas7.1.1/)

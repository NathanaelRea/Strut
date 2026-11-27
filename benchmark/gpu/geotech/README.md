# Plane-strain GPU benchmarks

These experiments use an AMD Radeon RX 9070 XT (`gfx1201`) and an
eight-thread Ryzen 7 9800X3D CPU. They run standalone with HIP and OpenMP.

## Elastic plane-strain multigrid

The test uses homogeneous small-strain elasticity on a square Q1 mesh with
zero displacements on all four edges, unit Young's modulus and thickness,
and a Poisson ratio of 0.30. An integrated nine-point block stencil applies
the stiffness in FP64 without storing a global matrix. The CPU and GPU run
PCG with the same geometric V-cycle, using weighted Jacobi smoothing,
bilinear prolongation, its transpose for restriction, and a direct solve on
the coarsest interior node. The GPU keeps the working vectors and multigrid
hierarchy resident and captures a PCG iteration as a HIP graph.

```bash
mkdir -p benchmark/.tmp/geotech
hipcc --offload-arch=gfx1201 -O3 -Wall -Wextra -Werror -fopenmp \
    benchmark/gpu/geotech/plane_strain.hip -lamdhip64 \
    -o benchmark/.tmp/geotech/plane_strain
for n in 64 256 512 1024 2048; do
    HIP_VISIBLE_DEVICES=0 benchmark/.tmp/geotech/plane_strain "$n" 8 8 0.30 both \
        > "benchmark/.tmp/geotech/plane_strain_${n}.csv"
done
```

The sweep uses eight independent manufactured displacements with different
spatial modes at each size. GPU solve time includes uploading the right-hand
side, checking convergence, and reading back the displacements. The timings
exclude process startup, load generation, and validation. Setup includes
allocation and GPU graph creation. Warm times are medians of loads 2–8 from
one sweep.

| Cells | Free DOFs | Warm CPU / GPU per load (ms) | CPU / GPU setup and first load (ms) | CPU / GPU setup and eight loads (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64² | 7,938 | 0.60 / 2.65 | 0.75 / 113.81 | 5.14 / 132.41 |
| 256² | 130,050 | 1.77 / 5.04 | 3.26 / 117.00 | 15.76 / 152.24 |
| 512² | 522,242 | 5.06 / 7.97 | 10.33 / 104.67 | 45.82 / 160.67 |
| 1024² | 2,093,058 | 48.47 / 19.45 | 72.40 / 99.91 | 405.47 / 236.64 |
| 2048² | 8,380,418 | 542.52 / 77.05 | 620.90 / 201.67 | 4,481.93 / 739.06 |

At 1024², eight loads amortized setup for a 1.71× GPU speedup. At 2048²,
the GPU was 3.08× faster for setup and one load and 6.06× faster for setup
and eight loads. The warm-load speedup reached 7.04×. All 40 paired solves
used the same iteration count on both devices. The worst relative true
residual was `9.73e-10` and the worst relative displacement error was
`3.65e-10`. A separate Gauss assembly checks the stencil coefficients.

The benchmark models a homogeneous, regular elastic grid with fixed
stiffness and prescribed boundary conditions. Its timings apply to this
solver and grid. Mesh irregularity, variable material properties, and nearly
incompressible or coupled formulations can change the multigrid cost and
convergence.

## Local Drucker–Prager return mapping

This benchmark applies a cyclic strain history to independent integration
points using a common non-associated Drucker–Prager law with linear isotropic
hardening. It keeps a plane-strain in-plane deviator, mean stress, and plastic
history variable in FP64 arrays. The elastic predictor, yield test, and
closed-form cone or apex return run together at each point. The implementation
exploits the piecewise linear scalar return equation for these parameters.
It checks the correction against independent bisection, then compares all
final CPU and GPU states.

```bash
hipcc --offload-arch=gfx1201 -O3 -Wall -Wextra -Werror -fopenmp \
    benchmark/gpu/geotech/drucker_prager.hip -lamdhip64 \
    -o benchmark/.tmp/geotech/drucker_prager
HIP_VISIBLE_DEVICES=0 benchmark/.tmp/geotech/drucker_prager 1048576 40 8
```

The history is synthetic and the parameters are uncalibrated. The material
kernel measures local state updates and excludes GPU allocation and transfers.
The elastic mesh above uses a separate elastic law. Integrating this plastic
law into a global solve would also require tangent assembly or application and
an appropriate solver for the potentially nonsymmetric tangent.

For 40 increments at 1,048,576 points, the median Drucker–Prager pass took
49.07 ms on eight CPU threads and 7.83 ms on the GPU. At 4,194,304 points it
took 264.25 ms and 33.45 ms respectively. The larger pass exercised about
67.3 million cone returns and 2.8 million apex returns. The largest absolute
CPU/GPU difference in the final state was `4.2e-16`.

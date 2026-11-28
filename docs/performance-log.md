# Performance log

What has been tried to make Strut faster, what it did, and whether it was kept. Read this before starting performance work, so a rejected idea is not rebuilt and an open lead is not rediscovered. Add an entry for every experiment, including the ones that fail.

Current timings against OpenSees are in [benchmark-opensees-examples.md](./benchmark-opensees-examples.md). How to measure is in [benchmark/README.md](../benchmark/README.md).

## Adding an entry

One row per experiment, in the table for its area:

- **Change**: what was done, specific enough to recognise the same idea later.
- **Result**: before and after on a named case, with the thread count. Say how it was measured if it was not `scripts/ab_compare.py` on a quiet machine.
- **Status**: `kept`, `reverted`, `rejected` (measured, not worth it), or `open` (not built, or built and not measured).
- **Why**, for anything not kept: the reason, so the next person can tell whether it still applies.

Results that were taken while other jobs were running are marked "under load". Treat their ratios as rough and their thread-scaling as unknown.

## Measuring: mistakes already made

- **OpenSees threads.** The pinned conda OpenSees starts about 15 BLAS threads unless `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` are set. That slows it on small models that reach LAPACK: `opensees_example_mrf_panel_zone_pushover` took 5.4 s unpinned and 1.09 s pinned. Numbers published before 2026-11-27 were unpinned. The harness now pins and records the setting. Pinning does not change the mega-tall: 845.0 s pinned, 845.1 s unpinned.
- **Thread scaling under load.** With four agents building and two OpenSees jobs running, 4-to-8-thread factorization results reversed between runs. Take scaling numbers with the 1-minute load below 2, and print the load next to each number.
- **A script that errors is not a comparison.** On the two unconfined Ex9c sections OpenSees stops converging at step 24 of 100 and the script then dies on an undefined variable, while Strut completes the history. The old harness recorded a time for both engines. The manifests now carry `benchmark_skip` with the reason and the runner leaves them out.
- **Zero is not a time.** The harness used to record 0 µs for OpenSees on all 17 Ex7/Ex8 cases and report OK, because its compute-only wrapper differed from the parity path and the script failed inside OpenSees. A zero or missing time is now a failure.
- **Model build is outside "analysis time".** The mega-tall spent 24 s building its model next to a 27 s solve, and no table showed it. Check `phase_times_us.json` (`model_build_dof_map_us`, `case_load_parse_us`) and end-to-end time (`run_benchmarks.py --e2e`) as well.
- **The sampler has no call stacks.** `scripts/profile_sample.py` attributes by instruction pointer, so allocator and `memmove` time shows up as `libc` or `libAsyncRTRuntimeGlobals` with no caller. `gdb -p <pid> -batch -ex "bt 14"` on `build/strut/strut_g1` gives the caller.
- **Address-space limits.** The mega-tall peaks at 4.0 GB resident, and the allocator maps 1 GB aligned regions on top of that. Under `ulimit -v 8000000` it aborts in the allocator at the first transient step, which reads like a solver crash. Use 16 GB or no limit.
- **Auto-rebuild.** `scripts/run_strut_case.py` rebuilds `build/strut/strut` when a `.mojo` file is newer than the binary. Editing source during a timing run swaps the binary mid-run. Set `STRUT_MOJO_BIN`.

## Log

Baseline for the 2026-11-27 entries is `main` at b9419ca. Times are analysis time on a Ryzen 7 9800X3D (8 cores, 96 MB L3) unless stated. "1t" and "8t" are `STRUT_THREADS`.

The 2026-11-27 rows were measured one change at a time while another program held about 2.7 cores, so their absolute times run high. The table below is all of those changes together against the baseline, taken afterwards on a quiet machine with `scripts/ab_compare.py` (recorders on, best of 3). Use it for the size of the combined effect, and the rows for which change did what.

| Case | Baseline 1t | Now 1t | Baseline 8t | Now 8t |
| --- | ---: | ---: | ---: | ---: |
| Mega-tall, 40 steps | 65.4 s | 55.9 s | 28.6 s | 27.1 s |
| `ex8_genericframe3d_analyze_static_cycle` | 15.0 s | 12.0 s | 15.1 s | 3.61 s |
| `ex8_genericframe3d_analyze_static_push` | 7.06 s | 5.65 s | 7.05 s | 1.68 s |
| `total_stress_single` (soil column) | 9.20 s | 5.15 s | 9.20 s | 2.17 s |
| `ex7_frame3d_analyze_static_cycle` | 1.43 s | 1.20 s | 1.46 s | 0.29 s |
| `ex5_frame2d_analyze_static_cycle` | 1.07 s | 0.97 s | 1.07 s | 0.25 s |
| `ex5_frame2d_analyze_dynamic_sine_uniform` | 579 ms | 569 ms | 582 ms | 178 ms |
| `rc_frame3d_force_5x5x8` (synthetic) | 1.67 s | 1.22 s | 0.77 s | 0.54 s |
| `rc_frame3d_disp_pdelta_5x5x8` (synthetic) | 1.58 s | 1.29 s | 1.05 s | 0.83 s |
| `mrf_dynamic_concentrated` | 1.95 s | 1.28 s | 1.94 s | 1.28 s |
| `mrf_panel_zone_pushover` | 1.50 s | 1.39 s | 1.50 s | 1.40 s |
| `shear_frame_viscous_damper` | 647 ms | 523 ms | 649 ms | 522 ms |
| `laterally_loaded_pile_foundation` | 421 ms | 158 ms | 319 ms | 157 ms |
| `deep_foundation_subject_to_lateral_spreading` | 192 ms | 50 ms | 144 ms | 49 ms |
| `corotational02_06_test06_prism_geom01` | 384 ms | 47 ms | 381 ms | 47 ms |
| `corotational02_05_test05_prism_geom01` | 274 ms | 34 ms | 274 ms | 34 ms |
| `gallery_shellcircle_circle` | 243 ms | 54 ms | 240 ms | 51 ms |
| `one_dimensional_consolidation` | 339 ms | 331 ms | 337 ms | 180 ms |
| `gallery_hexaextrusion_hexa` | 571 ms | 562 ms | 512 ms | 509 ms |
| `opensees_ssp_brick_up` | 5.7 ms | 1.1 ms | 5.7 ms | 1.1 ms |

The mega-tall baseline here already has the model-build fix from the first row of the next table, and uses an input without the `RelativeNormDispIncr` fallback, which the baseline cannot run. 24 cases were compared and the table lists the larger ones. Recorder output is bit-identical to the baseline in 19 of them and differs by at most 1.1e-11 relative in the rest.

### Model build and start-up

| Date | Change | Result | Status |
| --- | --- | --- | --- |
| 2026-11-27 | Reserve layered-shell concrete states once instead of `resize` per shell (`shell_cache.mojo`). The per-shell `resize` reallocated and copied every state 16,616 times. | Mega-tall model build 24 s → 0.44 s; whole run 89.6 s → 66.4 s (1t). | kept |
| 2026-11-27 | Solver core without Python: native timers, paths, JSON and file writes; Python initialised only if something needs it. | Solver start before reading input, basic truss: 37 ms → about 14 ms. `Py_Initialize` plus `import pathlib` had cost 35 ms per process. | kept |
| 2026-11-27 | Send the model to the solver over stdin instead of a temporary file. | Part of basic truss end to end 80 ms → 58.5 ms. | kept |
| 2026-11-27 | Lazy session for every `--input-tcl` script, so batch and interactive scripts share one path. | Not built: it would change when Tcl commands observe solver results. | rejected |

Open: basic truss end to end is 44 ms against 5.9 ms in OpenSees, and 221 of the 268 example scripts are slower end to end for the same reason. About 30 ms is CPython start-up and importing the Tcl front end (`tcl_to_strut.py`); the Tcl interpreter itself is under 1 ms.

### Interactive Tcl scripts (`analyze` in a loop, queries between analyses)

| Date | Change | Result (end to end) | Status |
| --- | --- | --- | --- |
| 2026-11-27 | Persistent solver session: one process keeps the model and committed state; the front end sends stages and queries over a pipe; a failed step returns -3 with the trial state reverted. Replaces replaying the whole history in a new process per `analyze`. | `opensees_source_example_4_1` 8.2 s → 0.35 s (OpenSees 0.23 s). | kept |
| 2026-11-27 | First session version: each command re-entered the analysis driver from the top. | One-step transient loops got slower, because every `analyze 1` rebuilt workspaces and refactored: `sdof_transient` 16.6 s → 24.3 s, `dynamic_shell` 5.7 s → 15.9 s (under load). | superseded by the next row |
| 2026-11-27 | Keep the driver alive: at the end of its step loop the driver asks the session for the next command while still inside its setup scope, and extends the loop when the command continues the same stage with the same settings. | `dynamic_shell` 7.6 s → 2.0 s (OpenSees 13.4 s); `sdof_transient` 17.6 s → 10.8 s (under load). | kept |
| 2026-11-27 | Skip rebuilding unchanged Tcl stages; compact binary request and reply for one-step calls. | `sdof_transient` screens 11.6 s → 8.7 s → 6.3 s. | kept |
| 2026-11-27 | One solver process for the whole script: `wipe` resets the domain in the running session; eigen-before-analyze sweeps use it too. | 72-model sweeps (`aisc25_3d`, `frame_0004_test_3d`) 2.9 s → 0.1 s (OpenSees 0.015 s). | kept |
| 2026-11-27 | HSConstraint and adaptive DisplacementControl inside the session. | `opensees_source_arc_length_02` 14.3 s → 1.8 s (OpenSees 3.5 s). | kept |
| 2026-11-27 | In-process solver (shared library called from the Python front end) to remove the pipe. | Not built. Measured on `sdof_transient` (290,730 calls, 20 µs each): the solver request is 5 µs, of which pipe and scheduling are about 3 µs. Expected saving 3 to 4 µs per call. The rest is the Tcl script running through the Python handlers. | rejected |

Open:

- A fallback that changes the algorithm or test for one step ends the driver and starts a fresh one: about 35 ms per retry in Example 4.1, against 0.17 ms for an ordinary step.
- Modal damping, prescribed support motion and `-vel0` excitation still fall back to replay.
- `sdof_transient` is 4.97 s against 3.06 s in OpenSees. The remaining cost is per-call work in the Python front end, not the solver.
- A script that builds many small models (`aisc25_3d`, `frame_0004_test_3d`) is 0.08 s against 0.011 s in OpenSees.

### Element state determination

| Date | Change | Result | Status |
| --- | --- | --- | --- |
| 2026-11-27 | Thread sweeps by estimated work (integration points × fibers or layers) instead of element count. The old rule needed 64 elements, so Ex7/Ex8 (42 expensive elements) never threaded. | Ex8 pushover 7.35 s → 1.89 s (8t). | kept |
| 2026-11-27 | Compute-then-scatter threading for 2D fiber frames, quads, u-p elements and bricks. Scatter stays serial in element order, so results are bit-identical for any thread count. | Ex5 dynamic 0.60 s → 0.20 s; soil column `total_stress_single` 9.5 s → 2.7 s (8t) and 5.5 s (1t). | kept |
| 2026-11-27 | 3D local-to-global stiffness rotation as two block products over contiguous inline storage. The old version indexed tuples at run time. | It was 16% of Ex8 pushover and 46% of the laterally loaded pile; now 0.4%. Pile 449 ms → 165 ms, lateral spreading 203 ms → 52 ms (1t). | kept |
| 2026-11-27 | Concrete02 commit in 3D fiber sections by switching the active history bank instead of copying five arrays. | Synthetic 3D frame 1556 ms → 1292 ms (1t). | kept |
| 2026-11-27 | Unaligned SIMD loads and stores in the 3D fiber loops (removes a per-vector alignment branch); bulk copies for long disjoint ranges. | Ex8 pushover 5993 ms → 5842 ms (1t). | kept |
| 2026-11-27 | Small 3D batches: the first free worker claims the next element, instead of fixed ranges. | Ex8 pushover 2752 ms → 2424 ms (8t, under load, combined with the row above). | kept, isolated effect not measured |
| 2026-11-27 | Parallel scatter of element matrices for frame models. | Not built: ordered scatter is 3% of the synthetic 3D frame run at 1 and 8 threads. | rejected for frames; open for the mega-tall, see below |
| 2026-11-27 | Skip the global stiffness rotation in residual-only 3D sweeps. | Dropped: the rotation is 0.4% after the rewrite. | rejected |
| 2026-11-27 | Parallel copies of 2D fiber histories. | 1.7% slower on Ex5 and 0.9% slower on Ex6 at 1t. | rejected |

Known limit: the 3D fiber sweep stops scaling at 4 threads. On the 768-element synthetic frame, tangent compute takes 1024 / 576 / 400 / 386 ms at 1 / 2 / 4 / 8 threads. The fiber state it touches is 183 MB against 96 MB of L3; with a quarter of the concrete fibers the same sweep scales 4.8x on 8 threads. Workers are busy (83% of region wall time) and dispatch is microseconds, so this is memory traffic, not scheduling.

Open:

- A smaller fiber state: committed and trial strain are derivable from the section deformation, and material parameters are per material, not per fiber. Estimated 10 to 25%; not built.
- Steel02's Menegotto-Pinto powers (`simd_exp` and `simd_log`, 19% of Ex8 pushover at 1t) run for every vector with an active lane.
- 2D force-based frames still allocate 11 times per element evaluation (693 per step on Ex5).

### Per-step overhead on small models

| Date | Change | Result (1t) | Status |
| --- | --- | --- | --- |
| 2026-11-27 | Native `perf_counter_ns` and native progress writes in the four analysis drivers, replacing calls through Python. | Part of every row below. | kept |
| 2026-11-27 | Reuse displacement, force, rollback and Krylov storage across steps; fixed-size storage for link orientation, transforms and DOF gathers; skip the tangent scatter in force-only link sweeps. | `mrf_pushover_concentrated` 142 ms → 76 ms; allocator calls per step 1,197 → 119. `mrf_dynamic_concentrated` 1.13 s → 0.64 s. | kept |
| 2026-11-27 | Corotational 3D response: one trial frame, one Jacobian, geometric stiffness reusing the projection blocks, inline storage. | `corotational02_05_test05_prism_geom01` 311 ms → 38 ms. Results differ by at most 2.1e-14 from changed multiply-add contraction. | kept |
| 2026-11-27 | Unsymmetric sparse LU with a symbolic analysis per pattern and no allocation per factorization, replacing a list-of-rows elimination that allocated about 6,450 times per factorization. Falls back to the old elimination when numeric fill exceeds the fixed allowance. | Test06 corotational 151 ms → 50 ms; `gallery_shellcircle_circle` 253 ms → 66 ms. | kept |
| 2026-11-27 | Pointer-and-length entry on `ElementMatrixSink`, so link, zero-length and MVLEM tangents come from stack storage. | Link allocations per step on `mrf_pushover_concentrated` 45 → 0. Recorder files byte-identical. | kept |
| 2026-11-27 | SSPbrickUP tangent in reused storage, four columns at a time, same summation order. | `opensees_ssp_brick_up` 6.8 ms → 1.2 ms. | kept |
| 2026-11-27 | PySimple1: cache the near-field reference power, simplify a drag exponent of one. | No measurable change in the kernel (1,238 → 1,250 samples). | kept, no gain |
| 2026-11-27 | LU workspace as eager fields of the solver backend struct. | Slowed an unrelated 5 ms BandGeneral cantilever by 6%, from the changed field layout. Fixed by restoring the field offsets and creating the workspace on first use. | reverted |

Open:

- `mrf_panel_zone_pushover` is 1.32 s against 1.05 s in OpenSees with one BLAS thread. `mrf_pushover_distributed` is 89 ms against 43 ms, and `mrf_pushover_concentrated` 71 ms against 51 ms.
- Remaining per-step allocations: 2D PDelta force scratch (64 per step on the MRF model), `assemble_internal_forces_typed_soa` residual and local storage, `add_uniform_excitation_load`, the coupled Krylov QR helper, and deep copies of Parallel-material child states on rollback.
- `corotational02_05_test05_prism_linear` barely moved and is 6.2 ms against 3.7 ms in OpenSees: its transformation is on the linear element path, which the corotational rewrite did not touch.

### Sparse factorization and large models

Reference workload: the 40-step mega-tall (155,130 equations after constraints, 40 factorizations of about 100 GFlop each, largest front 3,252).

| Date | Change | Result | Status |
| --- | --- | --- | --- |
| 2026-11-27 | AMD with the highest-degree hubs deferred (six vertices of degree 1,304 on the mega-tall), chosen only when predicted flops drop at least 15% and fill does not grow. | Predicted flops −18.7%, fill −7.3%. Mega-tall 64.8 s → 58.8 s (1t), 26.35 s → 26.27 s (8t), results equal to 4.5e-13. `gallery_hexaextrusion_hexa` looked 3 to 5% slower at 4 and 8 threads under load; on a quiet machine it is 509 ms against 512 ms at 8t. | kept |
| 2026-11-27 | Nested dissection (breadth-first-layer separators, AMD constrained to order interiors before separators), same selection rule, tried only above 8,192 equations and 50 M flops. | Shell wall 40×60: flops −26%, fill −8%, selected. Mega-tall: flops +8%, fill +14%, not selected. 3D frame 8×8×18: flops +1%, not selected. | kept as a guarded trial |
| 2026-11-27 | Project an unconstrained global AMD order into separator groups. | 171.5 GFlop on the mega-tall against 100.3 for AMD. | rejected |
| 2026-11-27 | Wide-front updates cut into columns, with a larger subtree work budget. | 915 / 531 / 342 / 338 ms against 921 / 537 / 350 / 329 ms at 1 / 2 / 4 / 8 threads (under load). | reverted, no repeatable gain |
| 2026-11-27 | Cache tile height 128. | 972 / 556 / 354 / 351 ms (under load). | reverted |
| 2026-11-27 | Taller SIMD microtile. | 984 / 575 / 389 / 370 ms (under load). | reverted |
| 2026-11-27 | Parallel row-scale pass. | Saved 2 ms of a 330 ms factorization. | reverted |
| 2026-11-27 | `RelativeNormDispIncr` convergence test. Not a speed-up: the mega-tall's fallback chain names it, and the run aborted at the transient stage without it. | Mega-tall 40-step input runs again. | kept |

Known limit: the mega-tall stops scaling at 4 threads. The 40-step run takes 55.7 / 36.9 / 28.3 / 26.0 s at 1 / 2 / 4 / 8 threads on a quiet machine, and the fewer flops from the new ordering save 9.6 s at one thread and 1.5 s at eight. Factorization of its pattern took 921 / 537 / 350 / 329 ms at 1 / 2 / 4 / 8 threads (under load), and 98% of the modelled work is in 163 wide upper fronts.

Open, in the order they should be done:

1. A wall-clock phase table for the mega-tall at 1, 2, 4 and 8 threads (element compute, scatter, effective-matrix loading, numeric factorization, triangular solves, the rest). An instrumented build was prepared for this and never run on a quiet machine; it is not in the repository. A main-thread sample at 8 threads put factorization near 41% of the solve, serial scatter and matrix loading near 17%, and triangular solves near 6%.
2. Test whether wide fronts are bound by memory bandwidth: factor one dense 3,252 front (85 MB) alone at 1 / 2 / 4 / 8 threads and compare GFlop/s with a cache-resident 512 front. If so, the trailing update needs a cache-blocked level-3 structure. The three kernel experiments above did not test this and were measured under load.
3. Parallel scatter and effective-matrix loading that keeps each slot's summation order.
4. Subtree-parallel triangular solves.

Not tried: reusing a factorization when the assembled matrix is bitwise unchanged; mixed-precision factorization with refinement; a GPU backend (in progress in a separate pull request).

### Benchmark harness

| Date | Change | Result | Status |
| --- | --- | --- | --- |
| 2026-11-27 | OpenSees runs the same staged script the parity suite uses, for every direct-Tcl case. The harness had its own older compatibility layer and sometimes timed generated Tcl. | Valid comparisons 210 → 233 of 235 non-interactive cases, including all of Ex7/Ex8. The other two are the Ex9c sections above. | kept |
| 2026-11-27 | Time the `analyze` and `eigen` calls themselves, instead of inserting Tcl statements around source lines. | Removes broken syntax, double counting and missed nested commands. | kept |
| 2026-11-27 | `--e2e`: wall time from the Tcl script to results for both engines, including live cases. | 33 interactive cases are measured for the first time. | kept |
| 2026-11-27 | Pin OpenSees to one BLAS and OpenMP thread, and record it in `metadata.json` and the Markdown header. | See "Measuring" above. | kept |

## Earlier work

These landed before this log existed and were not re-measured for it. Their pull requests and commits hold the numbers.

- Sparse LDLT with supernodes and threaded element sweeps for 3D frames and layered shells (#5).
- Small-model element and progress overhead (#14).
- Vectorised fiber kernels: structure-of-arrays parameters, grouped material dispatch, Steel02 and Concrete02 chunk fast paths (commits `ec2f89f`, `07ea68a`, `1d51f5a`, `240f49e`).
- Sparse and banded backends in place of dense assembly (`75158ee`).

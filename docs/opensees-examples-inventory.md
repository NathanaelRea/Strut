# OpenSees wiki examples: coverage inventory

Goal: a passing, enabled parity case for every example reachable from <https://opensees.berkeley.edu/wiki/index.php/Examples>.
Snapshot taken 2026-11-26 against `0df7d45`, using `uv run run_tests.py --all` on every disabled case.

## 1. Basic Examples Manual

| Example | Case | Status |
| --- | --- | --- |
| Basic Truss | `opensees_example_basic_truss` | passing |
| Moment Curvature | `opensees_example_moment_curvature` | passing |
| RC Frame Gravity | `opensees_example_rc_frame_gravity` | disabled, **passes** |
| RC Frame Pushover | `opensees_example_rc_frame_pushover` | fails: `recorder EnvelopeElement` writes 1 row, OpenSees writes 6 |
| RC Frame Earthquake | none | wiki upload `RCFrameEarthquake.tcl` is a redlink. The script is printed on the wiki page, so it has to be transcribed |
| Elastic Frame | `opensees_example_elastic_frame` | passing |
| Simply supported beam, 2D solids | `opensees_example_simply_supported_beam_solid` | passing |
| Time history, 2D elastic cantilever | `opensees_example_2d_elastic_cantileaver_column` | passing |
| Eigen, two-story shear frame | `opensees_example_eigen_two_story_shear_frame` | passing |
| Eigen, two-storey one-bay frame | none | fixture is vendored, no case yet |
| Steel Frame | n/a | wiki redlink, page does not exist |

## 2. Examples Manual (Ex1 to Ex9)

All scripts are vendored in `tests/fixtures/opensees/OpenSeesExamplesAdvanced/`.

| Group | Passing | Disabled but passes | Failing | Root cause of the failures |
| --- | --- | --- | --- | --- |
| Ex1a, Ex1b | 4 | 0 | 0 | |
| Ex2a | 0 | 1 (EQ) | 1 (Push) | push: node 2 displacement lands in the wrong DOF (got dof1=1.0, OpenSees dof2=-1.0) |
| Ex2b | 1 | 1 (Push) | 0 | |
| Ex2c | 1 | 0 | 1 (Push) | static push reactions |
| Ex3 | 4 | 0 | 1 (Static.Push) | static push reactions |
| Ex4 | 5 | 4 (EQ bidirect, EQ multipleSupport, sine multipleSupport, Static.Cycle) | 1 (Static.Push) | static push reactions |
| Ex5 | 10 | 0 | 2 (Static.Push, Static.Cycle) | push reactions; cycle aborts `DisplacementControl load column size mismatch` |
| Ex6 | 8 | 0 | 2 (Static.Push, Static.Cycle) | same as Ex5 |
| Ex7 (3D) | 0 | 0 | 8 | `eleLoad -beamUniform Wy Wz` (3 values) unsupported, then unknown |
| Ex8 (3D) | 0 | 0 | 9 | `eigen` limited to 2D models, then unknown |
| Ex9 sections | 0 | 0 | 18 | see below |

"Static push reactions": base reactions differ from OpenSees by orders of magnitude while OpenSees reports near zero (e.g. ref 1.4e-11, got 487). This is common to every `analyze.Static.Push` case with a gravity stage before the push, so it is probably one bug: reaction recording or `loadConst` handling across the push, rather than the push itself.

Ex9 failures:

- `zeroLengthSection` only accepts `FiberSection2d` and `ElasticSection2d`. Uniaxial, Aggregator, and all 3D sections are missing (`MomentCurvature2D/3D` analyses).
- `patch circ` unsupported (Ex9f).
- `Wsection.tcl` is not UTF-8 (byte `0xd7`), so the reader crashes (Ex9b).
- The `Ex9*.build.*` cases are build-only scripts with no `analyze`. They need to be paired with the moment-curvature analysis that sources them, the way Ex4 to Ex8 build cases are.

## 3. Practical examples (vendored in `tests/fixtures/opensees/OpenSeesExamplesPractical/`)

`setup-opensees-docs.py` now scrapes the Examples index and the sensitivity index too. It follows MediaWiki `File:` pages to the real upload (the old Basic scrape had stored the `EigenAnal_twoStoreyFrame1.tcl` wiki page as HTML, which is why that example never had a case), writes inline `<pre>` scripts for the pages listed in `INLINE_SCRIPTS`, and skips GiD `*.flavia.*` output.

Structural:

| Example | Elements | Materials and sections | Other features |
| --- | --- | --- | --- |
| Pushover, 2-story MRF (concentrated, distributed) | elasticBeamColumn, zeroLength, truss, beamWithHinges | Bilin, Elastic, Aggregator | region, leaning column, equalDOF |
| Dynamic, 2-story MRF | elasticBeamColumn, zeroLength, truss | Bilin, Elastic | region -rayleigh, UniformExcitation |
| 2-story MRF with panel zones and RBS | elasticBeamColumn, zeroLength, truss | Bilin, Hysteretic, Elastic | panel-zone equalDOF |
| 1-story frame with viscous dampers | elasticBeamColumn, twoNodeLink | ViscousDamper | `TakY.th` record |
| Earthquake in any direction | 3D elasticBeamColumn | | 3D eigen, timeSeries Series/Path |
| Rigid diaphragm consequences (inline script) | forceBeamColumn, dispBeamColumn | Concrete01, Steel01, Fiber | eleResponse, loops over element types |
| Infill wall and element removal | beamWithHinges, nonlinearBeamColumn (3D) | Concrete02, Steel01/02, fiber, Aggregator | recorder Collapse, element removal |
| SCBF 3D model | corotTruss, forceBeamColumn, truss, zeroLength | Steel02, Fatigue, fiber, Aggregator | |
| Parameter study (parallel), simple parameter study | nonlinearBeamColumn | Concrete01, Steel01, Fiber | getPID/getNP, recorder EnvelopeNode. The simple study page has fragments only. |
| Calling Matlab (inline script) | truss | Elastic | `exec matlab`, setParameter. Needs `getA.m` replaced by a Tcl proc for parity. |
| RC Frame Earthquake (Basic, inline script) | as RC gravity | | uses the `ARL360.at2` PEER record from OpenSees `v3.7.1` `EXAMPLES/ExampleScripts/` |

Geotechnical:

| Example | Elements | Materials | Other features |
| --- | --- | --- | --- |
| Laterally-loaded pile | dispBeamColumn, zeroLength | PySimple1, TzSimple1, QzSimple1, Elastic/Aggregator sections | equalDOF |
| Lateral spreading pile | dispBeamColumn, zeroLength | PySimple1 | sp in a pattern |
| Site response, total stress | quad | PressureIndependMultiYield, PressureDependMultiYield, Viscous | updateMaterialStage, Lysmer dashpot |
| Effective-stress site response | 9_4_QuadUP, zeroLength | PressureDependMultiYield02, Viscous | mixed -ndf, Penalty, parameter/updateParameter |
| 1D consolidation | 9_4_QuadUP | PressureIndependMultiYield | |
| 2D slope, effective stress | 9_4_QuadUP, zeroLength | PDMY02, PIMY, Viscous | |
| 3D sloping ground | SSPbrickUP, zeroLength | PDMY02, Viscous | setParameter |
| Sheet pile excavation | quad, dispBeamColumn, BeamContact2D | PDMY02 in InitialStateAnalysisWrapper, ContactMaterial2D | InitialStateAnalysis, remove element/node |
| Centrifuge test | quad, quadUP, zeroLength | PIMY, PDMY02, ENT, Parallel | page has fragments only, no runnable script |
| GiD problem types | | | input generator, not an analysis |

## 4. Sensitivity examples (vendored in `OpenSeesExamplesSensitivity/`)

| Example | Elements | Materials |
| --- | --- | --- |
| 1: 2D soil column, earthquake | quadWithSensitivity | MultiYieldSurfaceClay |
| 2: 3D soil block, pushover | bbarBrickWithSensitivity | MultiYieldSurfaceClay |
| 3: 2D RC frame, earthquake | dispBeamColumnWithSensitivity | Concrete01, SteelMP, SmoothPSConcrete, Hardening |
| 4: 3D RC frame, earthquake | dispBeamColumnWithSensitivity | Concrete01, Steel01, Elastic, Aggregator |
| 5: dam, reservoir, foundation | AC3D8, ASI3D8, AV3D4, VS3D4, bbarBrickWithSensitivity | AcousticMedium, TruncatedDP, ElasticIsotropic |

All need the reliability/DDM framework: `reliability`, `parameter`, `sensitivityIntegrator`, `sensitivityAlgorithm`, sensitivity recorders. The pinned binary has every element above.

## Plan and owners

1. Wave 1, vendored manuals (sections 1 and 2): A = Ex7/Ex8 3D (sol), B = 2D push/cycle (sol), C = Ex9 sections and eigen one-bay frame (neon). 7 passing disabled cases are already enabled.
2. Wave 2: D = concentrated-plasticity frames (Bilin, Hysteretic, region, beamWithHinges): pushover, dynamic, panel zone, infill wall (sol). E = remaining structural (viscous damper, rigid diaphragm, Matlab, RC frame earthquake, SCBF, parameter studies) (neon). G = continuum geotechnical (multi-yield soils, quad/quadUP/9_4_QuadUP/SSPbrickUP, Viscous, Penalty, updateMaterialStage), starting with total-stress site response (sol).
3. After wave 1: F = p-y/t-z/q-z pile examples; earthquake in any direction (needs A's 3D eigen); H = sensitivity framework and examples 1 to 5.

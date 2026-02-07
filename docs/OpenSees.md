# OpenSees notes (from docs/agent-reference/OpenSees)

## Overview

- Documentation is maintained in a separate repository and published via GitHub Pages. (README.md)

## Top-level layout (selected)

- `SRC/`: main C++ source tree.
- `DEVELOPER/`: example classes, build scripts, and step-by-step instructions for building developer examples on Unix/Windows.
- `EXAMPLES/`, `Workshops/`, `tests/`: examples, training materials, and tests.
- Build tooling: `CMakeLists.txt`, `Makefile`, `cmake/`, `conanfile.py`, `conanfile2.py`, `docker/`, `Dockerfile`.

## DEVELOPER notes

- Running `make` in `DEVELOPER/` builds example classes into shared libraries, placed alongside their source files. (UnixInstructions.txt)
- To run examples, set `LD_LIBRARY_PATH` to include `./` and run `OpenSees example1.tcl` (OpenSees must be on `PATH`). (UnixInstructions.txt)
- Windows instructions outline creating a Visual Studio DLL project, adding the class source/header, adding include paths (e.g., `..\..\core` for `elementAPI.h`), and then linking by adding `core/*.cpp` files to the project. (WindowsInstructions.txt)

## SRC/ structure (summary from SRC/readme)

- `analysis/`: analysis framework (algorithms, integrators, DOF groups, FE elements, constraint handlers, numberers).
- `domain/`: domain, nodes, constraints, subdomains, patterns, partitioner, load balancer.
- `element/`: element classes organized into subdirectories per element.
- `material/`: base material types and concrete materials.
- `matrix/`: core matrix, vector, and ID classes.
- `system_of_eqn/`: linear system classes and solver implementations (full, band, profile, PETSc, sparse, symmetric sparse, etc.).
- `graph/`: graph and partitioning utilities (including Metis integration).
- `tcl/`: interpreter integration and command glue.
- `actor/`, `remote/`: parallel and distributed execution infrastructure (actors, channels, brokers).
- `modelbuilder/`: model builder abstractions, including Tcl model builder.
- `tagged/`: tagged-object storage containers used by domain and analysis models.
- `utility/`: timers and misc helpers.

## Practical takeaways for Strut

- The C++ `SRC/` tree is the canonical reference for class structure and dependencies.
- `DEVELOPER/` examples show the expected build shape for external elements/materials (shared libs + scripting entry points).
- The `tcl/` and `interpreter/` areas highlight how OpenSees exposes commands; useful for mapping Mojo bindings.

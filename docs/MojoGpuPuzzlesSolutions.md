# Mojo GPU Puzzles solutions notes (from docs/agent-reference/mojo-gpu-puzzles/solutions)

## Overview

- `docs/agent-reference/mojo-gpu-puzzles` is a standalone repo for learning GPU programming in Mojo through interactive puzzles. (mojo-gpu-puzzles/README.md)
- The `solutions/` directory contains reference solutions for each puzzle, plus runner scripts for validation and sanitizers. (mojo-gpu-puzzles/README.md, solutions/)

## `solutions/` layout

- `p01/`, `p02/`, ... `p34/`: per-puzzle solution directories (not all numbers present). Each directory contains the solution sources (typically `.mojo` and sometimes supporting files). (solutions/)
- `run.sh`: test runner that executes puzzles, tracks pass/fail/skip, and supports running all puzzles or a single puzzle. (solutions/run.sh)
- `sanitizer.sh`: wrapper to run GPU sanitizers (uses shared config). (solutions/sanitizer.sh)
- `config.sh`: shared configuration for GPU platform/compute capability detection and puzzle compatibility lists. (solutions/config.sh)

## GPU compatibility and skipping logic (from `config.sh` + `run.sh`)

- NVIDIA compute capability requirements:
  - `>= 8.0` (Ampere) required for `p16`, `p19`, `p22`, `p28`, `p29`, `p33`.
  - `>= 9.0` (Hopper) required for `p34`.
- Unsupported puzzles by platform:
  - AMD: `p09`, `p10`, `p30`, `p31`, `p32`, `p33`, `p34`.
  - Apple: `p09`, `p10`, `p20`, `p21`, `p22`, `p29`, `p30`, `p31`, `p32`, `p33`, `p34`.
- The runner detects platform and compute capability (via `nvidia-smi`, `rocm-smi`, or macOS GPU info) to skip incompatible puzzles and to optionally ignore low-compute failures. (solutions/run.sh, solutions/config.sh)

## Practical takeaways for Strut

- The `solutions/` runner scripts are a good reference for GPU environment detection, capability gating, and puzzle-by-puzzle test orchestration.
- The per-puzzle folders provide a breadth of Mojo GPU kernel patterns and can be mined for idioms, performance techniques, and testing patterns.

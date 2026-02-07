# 🔥 Strut 🔥

GPU-first finite element analysis (FEA) research sandbox written in Mojo. The project focuses on structural mechanics workloads and aims to squeeze as much performance as possible from modern accelerators while remaining easy to validate against analytical solutions and other state of the art solvers.

## Background

This is a test project for now, primarily just to see how far I can push the vibes. The setup is a bit specific for my environment, and not very general right now.

- AMD GPU
- On Linux (using wine to run OpenSees)

## Quick Start

1. Fetch the dependencies `uv sync`
2. Download `OpenSees.exe` and necessary `tcl` files, put into [OpenSees](./benchmark/OpenSees/)
3. Run the [setup script](./scripts/setup.sh) to download docs and repos that can be used by the agent

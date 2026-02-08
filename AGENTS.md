# Strut

Strut is a Mojo-based adaptation of OpenSees for high-performance finite element and earthquake analysis.

## Essentials

- Use `uv` for Python dependency management and commands.
- Never use the `mojo` command directly; always use `uv run mojo`.
- Never run `python` or `pytest` directly; always run tools via `uv run <command>` (for example, `uv run run_tests.py`).
- Do not stub or mock unless the user explicitly asks for it.
- Keep `docs/PLAN.md` updated with progress and checkbox state.

## Non-Standard Commands

- Run unit tests: `uv run unit_tests.py`
- Build/precompile Mojo solver: `scripts/build_mojo_solver.sh`

## More Instructions

- [Workflow and Planning](./docs/agent-guides/workflow-and-planning.md)
- [Mojo and OpenSees References](./docs/agent-guides/mojo-and-opensees-references.md)
- [Testing and Benchmarks](./docs/agent-guides/testing-and-benchmarks.md)
- [Dependency Management](./docs/agent-guides/dependency-management.md)
- [Knowledge Capture](./docs/agent-guides/knowledge-capture.md)

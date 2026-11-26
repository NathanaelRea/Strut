# AGENTS.md

- Use `uv` for Python dependencies and commands, and `uv run mojo` for Mojo.
- After Mojo changes, build with `uv run python scripts/build_mojo_solver.py` and fix all compiler warnings. Run tests with `uv run run_tests.py`. Run builds and tests that share solver artifacts or caches one at a time.
- Before implementing OpenSees behavior, inspect its source in `docs/agent-reference/OpenSees/`. If the reference checkout is missing, run `uv run scripts/setup_reference_data.py`.

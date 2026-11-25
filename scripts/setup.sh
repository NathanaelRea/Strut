#!/usr/bin/env bash
# One-time machine setup for tests, benchmarks and agent reference material.
# Everything is pinned and cached in ~/.cache/strut, shared by all worktrees;
# re-running is a fast no-op.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

uv sync
uv run python scripts/setup_reference_data.py
uv run python scripts/setup_opensees.py

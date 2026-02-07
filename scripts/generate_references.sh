#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${repo_root}/scripts/run_opensees_wine.sh"

if [[ ! -x "$runner" ]]; then
  echo "error: runner script not found or not executable at '$runner'" >&2
  exit 1
fi

cases=(
  "ex1a|OpenSees/examples/ex1a/Ex1a.Canti2D.Push.tcl|tests/validation/ex1a/reference"
)

usage() {
  cat <<'HELP'
Usage: scripts/generate_references.sh [case ...]

Runs predefined OpenSees Tcl cases via Wine and stores their outputs in the
matching tests/validation/<case>/reference directory. Without arguments every
known case is refreshed. Provide one or more case names to restrict execution.

Available cases:
  ex1a
  pushover_concentrated
  pushover_distributed

Set OPENSEES_EXE/OPENSEES_WORKDIR/etc. before invoking to override defaults.
HELP
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

run_all=0
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  run_all=1
fi

should_run() {
  local name="$1"
  if [[ $run_all -eq 1 ]]; then
    return 0
  fi
  for t in "${targets[@]}"; do
    if [[ "$t" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

for entry in "${cases[@]}"; do
  IFS='|' read -r name script_rel output_rel <<<"$entry"
  if ! should_run "$name"; then
    continue
  fi
  script_path="${repo_root}/${script_rel}"
  output_dir="${repo_root}/${output_rel}"
  echo "==> Generating reference for $name"
  "$runner" --script "$script_path" --output "$output_dir"
done
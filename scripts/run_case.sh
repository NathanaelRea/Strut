#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
json_to_tcl="${repo_root}/scripts/json_to_tcl.py"
runner="${repo_root}/scripts/run_opensees_wine.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_case.sh path/to/case.json" >&2
  exit 1
fi

case_json="$1"
case_name="$(basename "$case_json" .json)"
case_root="${repo_root}/tests/validation/${case_name}"

mkdir -p "${case_root}/generated" "${case_root}/reference" "${case_root}/mojo"

tgt_json="${case_root}/${case_name}.json"
if [[ "$(realpath "$case_json")" != "$(realpath "$tgt_json")" ]]; then
  cp "$case_json" "$tgt_json"
fi

tcl_out="${case_root}/generated/model.tcl"
"${json_to_tcl}" "$tgt_json" "$tcl_out"

"${runner}" --script "$tcl_out" --output "${case_root}/reference"

"${repo_root}/scripts/run_mojo_case.py" --input "$tgt_json" --output "${case_root}/mojo"
"${repo_root}/scripts/compare_case.py" --case "$case_name"

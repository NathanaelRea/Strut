#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${repo_root}/build/strut"
out_name="strut"
diag_file="$(mktemp)"

cleanup() {
  rm -f "${diag_file}"
}
trap cleanup EXIT

mkdir -p "${out_dir}"

cmd=(
  uv
  run
  mojo
  build
  --diagnostic-format
  json
  "${repo_root}/src/mojo/strut.mojo"
)

if [[ "${STRUT_PROFILE:-0}" == "1" ]]; then
  out_name="strut_profile"
  cmd+=(-D STRUT_PROFILE=1)
fi

out_bin="${out_dir}/${out_name}"
cmd+=(-o "${out_bin}")

set +e
"${cmd[@]}" >"${diag_file}" 2>&1
build_status=$?
set -e

if [[ ${build_status} -ne 0 ]]; then
  cat "${diag_file}" >&2
  exit "${build_status}"
fi

set +e
uv run python "${repo_root}/scripts/check_mojo_warnings.py" --quiet "${diag_file}"
warning_status=$?
set -e

if [[ ${warning_status} -ne 0 ]]; then
  cat "${diag_file}" >&2
  echo "Mojo compiler warnings detected." >&2
  exit 1
fi

if [[ -s "${diag_file}" ]]; then
  cat "${diag_file}"
fi
echo "Built ${out_bin}"

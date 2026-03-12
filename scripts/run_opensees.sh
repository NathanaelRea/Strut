#!/usr/bin/env bash
set -euo pipefail

# Runs OpenSees Tcl examples natively on Linux and writes recorder files into the
# matching tests/validation/<example>/reference/ directory

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_root="${repo_root}/benchmark/OpenSees/examples"
validation_root="${repo_root}/tests/validation"
opensees_bin="${OPENSEES_BIN:-${repo_root}/.build/opensees-linux/OpenSees}"

usage() {
  cat <<'EOF'
Usage: run_opensees.sh [--script path/to/example.tcl [--output dir]]...

Without arguments the script discovers every OpenSees example directory under
OpenSees/examples/, runs each *.tcl file with the native Linux OpenSees binary,
and captures the recorder
files inside tests/validation/<example>/reference/.

Provide one or more --script flags to target specific Tcl files. Follow each
--script with --output to override the destination directory for that run.
Set OPENSEES_WORKDIR to change the default output root, or OPENSEES_SCRIPT
for a single-script run. Set OPENSEES_BIN to override the default binary path.
EOF
}

resolve_path() {
  local input="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$input"
  else
    case "$input" in
    /*) printf '%s\n' "$input" ;;
    *) printf '%s\n' "$(pwd)/$input" ;;
    esac
  fi
}

pick_example_name() {
  local script_path="$1"
  local rel="${script_path#${examples_root}/}"
  if [[ "$rel" != "$script_path" ]]; then
    printf '%s\n' "${rel%%/*}"
  else
    printf '%s\n' "$(basename "$(dirname "$script_path")")"
  fi
}

default_output_dir_for() {
  local example_name="$1"
  local validation_dir="${validation_root}/${example_name}/reference"
  mkdir -p "${validation_dir}"
  printf '%s\n' "${validation_dir}"
}

prepare_runtime() {
  opensees_bin="$(resolve_path "$opensees_bin")"
  if [[ ! -f "$opensees_bin" ]]; then
    echo "error: OpenSees binary not found at '$opensees_bin'." >&2
    exit 1
  fi
  if [[ ! -x "$opensees_bin" ]]; then
    echo "error: OpenSees binary is not executable: '$opensees_bin'." >&2
    exit 1
  fi
}

run_script() {
  local script_path="$1"
  local example_name="$2"
  local output_override="${3:-}"
  if [[ ! -f "$script_path" ]]; then
    echo "warning: Tcl script '$script_path' not found, skipping." >&2
    return
  fi

  local analysis_dir
  if [[ -n "$output_override" ]]; then
    analysis_dir="$output_override"
  else
    local default_out
    default_out="$(default_output_dir_for "$example_name")"
    analysis_dir="${OPENSEES_WORKDIR:-$default_out}"
  fi

  mkdir -p "$analysis_dir"
  rm -rf "${analysis_dir}/Data"

  local script_dir
  script_dir="$(dirname "$script_path")"
  local script_parent
  script_parent="$(dirname "$script_dir")"
  local script_dir_name
  script_dir_name="$(basename "$script_dir")"
  local work_root
  work_root="$(mktemp -d)"
  local mirrored_parent="${work_root}/$(basename "$script_parent")"
  local mirrored_script_dir="${mirrored_parent}/${script_dir_name}"
  local run_start_stamp="${work_root}/run_start.stamp"
  cp -a "${script_parent}/." "${mirrored_parent}/"
  : >"$run_start_stamp"

  local run_script_name
  run_script_name="$(basename "$script_path")"

  pushd "$mirrored_script_dir" >/dev/null
  {
    "$opensees_bin" "$run_script_name" >/dev/null 2>&1
  }
  popd >/dev/null

  for output_subdir in Data data; do
    if [[ -d "${mirrored_script_dir}/${output_subdir}" ]]; then
      cp -a "${mirrored_script_dir}/${output_subdir}" "${analysis_dir}/"
    fi
  done

  shopt -s nullglob
  local top_level_outputs=(
    "${mirrored_script_dir}"/*.out
    "${mirrored_script_dir}"/analysis_time_us.txt
    "${mirrored_script_dir}"/case_time_us.txt
    "${mirrored_script_dir}"/case_error.txt
  )
  shopt -u nullglob
  for produced_file in "${top_level_outputs[@]}"; do
    [[ -f "$produced_file" ]] || continue
    cp "$produced_file" "${analysis_dir}/$(basename "$produced_file")"
  done

  while IFS= read -r -d '' produced_file; do
    local rel_path
    rel_path="${produced_file#${mirrored_script_dir}/}"
    mkdir -p "${analysis_dir}/$(dirname "$rel_path")"
    cp "$produced_file" "${analysis_dir}/${rel_path}"
  done < <(find "$mirrored_script_dir" -type f -newer "$run_start_stamp" -print0)
  rm -rf "$work_root"
}

discover_scripts() {
  shopt -s nullglob
  local dirs=("${examples_root}"/*)
  shopt -u nullglob
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "error: no OpenSees examples found under '$examples_root'." >&2
    exit 1
  fi

  local ran=0
  for dir in "${dirs[@]}"; do
    [[ -d "$dir" ]] || continue
    local example_name
    example_name="$(basename "$dir")"
    shopt -s nullglob
    local scripts=("$dir"/*.tcl)
    shopt -u nullglob
    if [[ ${#scripts[@]} -eq 0 ]]; then
      echo "warning: no Tcl scripts found in '$dir', skipping." >&2
      continue
    fi
    for script in "${scripts[@]}"; do
      run_script "$script" "$example_name"
      ran=$((ran + 1))
    done
  done

  if [[ $ran -eq 0 ]]; then
    echo "error: no runnable Tcl scripts discovered under '$examples_root'." >&2
    exit 1
  fi
}

main() {
  declare -a arg_scripts=()
  declare -a arg_outputs=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -s|--script)
        if [[ $# -lt 2 ]]; then
          echo "error: --script expects a path argument" >&2
          exit 1
        fi
        arg_scripts+=("$(resolve_path "$2")")
        arg_outputs+=("")
        shift 2
        ;;
      -o|--output)
        if [[ $# -lt 2 ]]; then
          echo "error: --output expects a directory argument" >&2
          exit 1
        fi
        if [[ ${#arg_scripts[@]} -eq 0 ]]; then
          echo "error: --output must follow a --script argument" >&2
          exit 1
        fi
        local last=$(( ${#arg_outputs[@]} - 1 ))
        arg_outputs[$last]="$(resolve_path "$2")"
        shift 2
        ;;
      --help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        arg_scripts+=("$(resolve_path "$1")")
        arg_outputs+=("")
        shift
        ;;
    esac
  done

  if [[ $# -gt 0 ]]; then
    for extra in "$@"; do
      arg_scripts+=("$(resolve_path "$extra")")
      arg_outputs+=("")
    done
  fi

  prepare_runtime

  if [[ ${#arg_scripts[@]} -gt 0 ]]; then
    for idx in "${!arg_scripts[@]}"; do
      script_path="${arg_scripts[$idx]}"
      example_name="$(pick_example_name "$script_path")"
      run_script "$script_path" "$example_name" "${arg_outputs[$idx]}"
    done
  elif [[ -n "${OPENSEES_SCRIPT:-}" ]]; then
    script_path="$(resolve_path "$OPENSEES_SCRIPT")"
    example_name="$(pick_example_name "$script_path")"
    output_override="${OPENSEES_WORKDIR:-}"
    run_script "$script_path" "$example_name" "$output_override"
  else
    discover_scripts
  fi
}

main "$@"

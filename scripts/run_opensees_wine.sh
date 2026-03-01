#!/usr/bin/env bash
set -euo pipefail

# Runs OpenSees Tcl examples via Wine and writes recorder files into the
# matching tests/validation/<example>/reference/ directory

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_root="${repo_root}/benchmark/OpenSees/examples"
validation_root="${repo_root}/tests/validation"
opensees_exe="${repo_root}/benchmark/OpenSees/OpenSees.exe"
intel_runtime_env="${OPENSEES_INTEL_RUNTIME:-}"
intel_runtime_name="libiomp5md.dll"
tcl_dir_env="${OPENSEES_TCL_DIR:-}"
tcl_required_rel="lib/tcl8.6"

usage() {
  cat <<'EOF'
Usage: run_opensees_wine.sh [--script path/to/example.tcl [--output dir]]...

Without arguments the script discovers every OpenSees example directory under
OpenSees/examples/, runs each *.tcl file with Wine, and captures the recorder
files inside tests/validation/<example>/reference/.

Provide one or more --script flags to target specific Tcl files. Follow each
--script with --output to override the destination directory for that run.
Set OPENSEES_WORKDIR to change the default output root, or OPENSEES_SCRIPT
for a single-script run.
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

require_binary() {
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "error: required command '$bin' not found on PATH" >&2
    exit 1
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
  require_binary wine
  require_binary winepath

  if [[ ! -f "$opensees_exe" ]]; then
    echo "error: OpenSees executable not found at '$opensees_exe'." >&2
    exit 1
  fi

  wine_prefix="${WINEPREFIX:-$HOME/.wine}"
  runtime_candidates=()
  if [[ -n "$intel_runtime_env" ]]; then
    runtime_candidates+=("$intel_runtime_env")
  fi
  runtime_candidates+=("$(dirname "$opensees_exe")/$intel_runtime_name"
  "$wine_prefix/drive_c/windows/system32/$intel_runtime_name"
  "$wine_prefix/drive_c/windows/syswow64/$intel_runtime_name")

  runtime_path=""
  for candidate in "${runtime_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      runtime_path="$candidate"
      break
    fi
  done

  if [[ -z "$runtime_path" ]]; then
    {
      echo "error: Intel OpenMP runtime (${intel_runtime_name}) not found."
      echo "       Copy it next to OpenSees.exe, install the redistributable into Wine,"
      echo "       or set OPENSEES_INTEL_RUNTIME to the DLL."
    } >&2
    exit 1
  fi

  tcl_dir="${tcl_dir_env:-$(dirname "$opensees_exe")/$tcl_required_rel}"
  if [[ ! -f "$tcl_dir/init.tcl" ]]; then
    {
      echo "error: Tcl runtime not found (expected init.tcl under '$tcl_dir')."
      echo "       Copy the OpenSees 'lib/' directory next to the executable or"
      echo "       set OPENSEES_TCL_DIR to the directory that contains init.tcl."
    } >&2
    exit 1
  fi

  tcl_dir_win="$(winepath -w "$tcl_dir")"
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
  cp -a "${script_parent}/." "${mirrored_parent}/"

  local run_script_name
  run_script_name="$(basename "$script_path")"

  pushd "$mirrored_script_dir" >/dev/null
  {
    env TCL_LIBRARY="$tcl_dir_win" wine "$opensees_exe" "$run_script_name" >/dev/null 2>&1
  }
  popd >/dev/null

  if [[ -d "${mirrored_script_dir}/Data" ]]; then
    cp -a "${mirrored_script_dir}/Data" "${analysis_dir}/"
  fi
  for metric_file in analysis_time_us.txt case_time_us.txt case_error.txt; do
    if [[ -f "${mirrored_script_dir}/${metric_file}" ]]; then
      cp "${mirrored_script_dir}/${metric_file}" "${analysis_dir}/${metric_file}"
    fi
  done
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

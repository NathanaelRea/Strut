#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_source="${repo_root}/docs/agent-reference/OpenSees"
default_build="${repo_root}/.build/opensees-linux"
default_mumps_dir="${repo_root}/.build/mumps"
default_test_script="${default_source}/EXAMPLES/ExampleScripts/Example1.1.tcl"
default_mp_test_script="${default_source}/EXAMPLES/SmallMP/Example.tcl"

source_dir="${default_source}"
build_dir="${default_build}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
with_mp=0
openmpi=0
mumps_dir=""
scalapack_libs=""
test_script=""
mp_test_script=""
mp_test_procs=2
mpiexec_bin="${OPENSEESMP_MPIEXEC:-mpiexec}"
mpiexec_extra_args="${OPENSEESMP_MPIEXEC_ARGS:-}"
mpiexec_accelerator="${OPENSEESMP_MPI_ACCELERATOR:-null}"
declare -a cmake_extra_args=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- <extra cmake args>]

Build the vendored OpenSees reference tree for Linux.

Options:
  --source DIR            OpenSees source tree. Default: ${default_source}
  --build-dir DIR         CMake build directory. Default: ${default_build}
  --jobs N                Parallel build jobs. Default: ${jobs}
  --with-mp               Also build OpenSeesMP using Arch/OpenMPI defaults.
  --mumps-dir DIR         Built MUMPS directory. Default with --with-mp:
                          ${default_mumps_dir}
  --scalapack-libs LIST   ScaLAPACK link line. Default with --with-mp:
                          first matching /usr/lib/libscalapack*.so*
  --openmpi               Pass -DOPENMPI=TRUE to CMake.
  --test-script PATH      Run the built OpenSees binary on a Tcl script.
                          Default smoke test: ${default_test_script}
  --mp-test-script PATH   Run the built OpenSeesMP binary with mpiexec.
                          Default smoke test: ${default_mp_test_script}
  --mp-test-procs N       MPI ranks for --mp-test-script. Default: 2
  --help                  Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --test-script "${default_test_script}"
  $(basename "$0") --with-mp
  $(basename "$0") --with-mp --scalapack-libs "/usr/lib/libscalapack.so"
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    maybe_arch_hint
    die "required command '$cmd' not found on PATH"
  fi
}

maybe_arch_hint() {
  if [[ -f /etc/os-release ]] && grep -q '^ID=arch$' /etc/os-release; then
    cat >&2 <<'EOF'
hint: on Arch Linux, install the baseline prerequisites with:
  sudo pacman -S --needed cmake gcc-fortran tcl tk hdf5 eigen openmpi

Then bootstrap the OpenSeesMP extras with:
  scripts/build_opensees_archlinux_init.sh
EOF
  fi
}

detect_scalapack_libs() {
  local candidate=""
  local patterns=(
    "/usr/lib/libscalapack.so"
    "/usr/lib/libscalapack-openmpi.so"
    "/usr/lib/libscalapack-openmpi.so.*"
    "/usr/lib/libscalapack.so.*"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    local match=""
    for match in $pattern; do
      if [[ -e "${match}" ]]; then
        printf '%s\n' "${match}"
        return 0
      fi
    done
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || die "--source expects a directory"
      source_dir="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir expects a directory"
      build_dir="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs expects a number"
      jobs="$2"
      shift 2
      ;;
    --with-mp)
      with_mp=1
      shift
      ;;
    --mumps-dir)
      [[ $# -ge 2 ]] || die "--mumps-dir expects a directory"
      mumps_dir="$2"
      shift 2
      ;;
    --scalapack-libs)
      [[ $# -ge 2 ]] || die "--scalapack-libs expects a value"
      scalapack_libs="$2"
      shift 2
      ;;
    --openmpi)
      openmpi=1
      shift
      ;;
    --test-script)
      if [[ $# -ge 2 && "${2}" != --* ]]; then
        test_script="$2"
        shift 2
      else
        test_script="${default_test_script}"
        shift
      fi
      ;;
    --mp-test-script)
      if [[ $# -ge 2 && "${2}" != --* ]]; then
        mp_test_script="$2"
        shift 2
      else
        mp_test_script="${default_mp_test_script}"
        shift
      fi
      ;;
    --mp-test-procs)
      [[ $# -ge 2 ]] || die "--mp-test-procs expects a number"
      mp_test_procs="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      cmake_extra_args+=("$@")
      break
      ;;
    *)
      die "unknown argument '$1'"
      ;;
  esac
done

[[ -d "${source_dir}" ]] || die "OpenSees source directory not found: ${source_dir}"
[[ -f "${source_dir}/CMakeLists.txt" ]] || die "missing CMakeLists.txt in ${source_dir}"

need_command cmake
need_command c++
need_command gfortran
need_command tclsh

if [[ "${with_mp}" -eq 1 ]]; then
  need_command mpicxx
  need_command "${mpiexec_bin}"
  openmpi=1
  if [[ -z "${mumps_dir}" ]]; then
    mumps_dir="${default_mumps_dir}"
  fi
  [[ -d "${mumps_dir}" ]] || die "MUMPS directory not found: ${mumps_dir}"
  if [[ -z "${scalapack_libs}" ]]; then
    if scalapack_libs="$(detect_scalapack_libs)"; then
      :
    else
      die "could not detect a ScaLAPACK library under /usr/lib; pass --scalapack-libs explicitly or run scripts/build_opensees_archlinux_init.sh"
    fi
  fi
fi

mkdir -p "${build_dir}"

declare -a cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
)

if [[ "${with_mp}" -eq 1 ]]; then
  cmake_args+=("-DMUMPS_DIR=${mumps_dir}")
  cmake_args+=("-DSCALAPACK_LIBRARIES=${scalapack_libs}")
  if [[ "${openmpi}" -eq 1 ]]; then
    cmake_args+=("-DOPENMPI=TRUE")
  fi
fi

echo "Configuring OpenSees in ${build_dir}"
if ! cmake -S "${source_dir}" -B "${build_dir}" "${cmake_args[@]}" "${cmake_extra_args[@]}"; then
  maybe_arch_hint
  exit 1
fi

echo "Building OpenSees"
cmake --build "${build_dir}" --target OpenSees --parallel "${jobs}"

if [[ "${with_mp}" -eq 1 ]]; then
  echo "Building OpenSeesMP"
  cmake --build "${build_dir}" --target OpenSeesMP --parallel "${jobs}"
fi

if [[ -n "${test_script}" ]]; then
  [[ -f "${test_script}" ]] || die "test script not found: ${test_script}"
  echo "Running OpenSees test: ${test_script}"
  (
    cd "$(dirname "${test_script}")"
    "${build_dir}/OpenSees" "$(basename "${test_script}")"
  )
fi

if [[ -n "${mp_test_script}" ]]; then
  [[ "${with_mp}" -eq 1 ]] || die "--mp-test-script requires --with-mp"
  [[ -f "${mp_test_script}" ]] || die "MP test script not found: ${mp_test_script}"
  echo "Running OpenSeesMP test: ${mp_test_script}"
  declare -a mpiexec_cmd=("${mpiexec_bin}" "-n" "${mp_test_procs}")
  if [[ -n "${mpiexec_accelerator}" ]]; then
    mpiexec_cmd+=("--mca" "accelerator" "${mpiexec_accelerator}")
  fi
  if [[ -n "${mpiexec_extra_args}" ]]; then
    read -r -a mpiexec_extra_parts <<<"${mpiexec_extra_args}"
    mpiexec_cmd+=("${mpiexec_extra_parts[@]}")
  fi
  (
    cd "$(dirname "${mp_test_script}")"
    "${mpiexec_cmd[@]}" "${build_dir}/OpenSeesMP" "$(basename "${mp_test_script}")"
  )
fi

echo "Built binaries:"
echo "  ${build_dir}/OpenSees"
if [[ "${with_mp}" -eq 1 ]]; then
  echo "  ${build_dir}/OpenSeesMP"
  echo "Using MUMPS_DIR=${mumps_dir}"
  echo "Using SCALAPACK_LIBRARIES=${scalapack_libs}"
  if [[ -n "${mpiexec_accelerator}" ]]; then
    echo "Using OPENSEESMP_MPI_ACCELERATOR=${mpiexec_accelerator}"
  fi
fi

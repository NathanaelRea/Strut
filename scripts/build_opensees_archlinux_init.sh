#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_mumps_src="${repo_root}/.build/mumps-src"
default_mumps_build="${repo_root}/.build/mumps"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
mumps_repo="https://github.com/OpenSees/mumps.git"
scalapack_package="scalapack"
skip_packages=0
skip_scalapack=0
skip_mumps=0
force_mumps_clone=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Bootstrap an Arch Linux host for OpenSees/OpenSeesMP:
- installs baseline packages with pacman
- installs ScaLAPACK with yay
- clones and builds MUMPS into ${default_mumps_build}

Options:
  --jobs N              Parallel jobs for the MUMPS build. Default: ${jobs}
  --skip-packages       Skip the pacman baseline package install.
  --skip-scalapack      Skip the yay ScaLAPACK install.
  --skip-mumps          Skip the MUMPS clone/build step.
  --force-mumps-clone   Re-clone the MUMPS source tree.
  --help                Show this help.

After this finishes, the normal flow is:
  scripts/build_opensees_linux.sh
  scripts/build_opensees_linux.sh --with-mp
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "required command '$cmd' not found on PATH"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs expects a number"
      jobs="$2"
      shift 2
      ;;
    --skip-packages)
      skip_packages=1
      shift
      ;;
    --skip-scalapack)
      skip_scalapack=1
      shift
      ;;
    --skip-mumps)
      skip_mumps=1
      shift
      ;;
    --force-mumps-clone)
      force_mumps_clone=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1'"
      ;;
  esac
done

if [[ -f /etc/os-release ]] && ! grep -q '^ID=arch$' /etc/os-release; then
  die "this helper is intended for Arch Linux"
fi

need_command sudo
need_command git
need_command cmake

if [[ "${skip_packages}" -eq 0 ]]; then
  sudo pacman -S --needed --noconfirm \
    cmake gcc-fortran tcl tk hdf5 eigen openmpi
fi

if [[ "${skip_scalapack}" -eq 0 ]]; then
  need_command yay
  yay -S --needed --noconfirm "${scalapack_package}"
fi

if [[ "${skip_mumps}" -eq 0 ]]; then
  mkdir -p "${repo_root}/.build"
  if [[ "${force_mumps_clone}" -eq 1 ]]; then
    rm -rf "${default_mumps_src}"
  fi
  if [[ ! -d "${default_mumps_src}/.git" ]]; then
    rm -rf "${default_mumps_src}"
    git clone "${mumps_repo}" "${default_mumps_src}"
  fi
  cmake -S "${default_mumps_src}" -B "${default_mumps_build}" -Darith=d
  cmake --build "${default_mumps_build}" --parallel "${jobs}"
fi

echo "Arch OpenSees bootstrap complete."
echo "Default MUMPS build: ${default_mumps_build}"
echo "Next commands:"
echo "  ${repo_root}/scripts/build_opensees_linux.sh"
echo "  ${repo_root}/scripts/build_opensees_linux.sh --with-mp"

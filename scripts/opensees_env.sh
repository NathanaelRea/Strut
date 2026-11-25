# shellcheck shell=bash
# Sourced by the OpenSees run scripts. Resolves which OpenSees binary to use.
#
# Lookup order for a binary named NAME (OpenSees or OpenSeesMP):
#   1. The caller's explicit override (OPENSEES_BIN / OPENSEESMP_BIN).
#   2. A repo-local source build at .build/opensees-linux/NAME
#      (scripts/build_opensees_linux.sh).
#   3. The shared locked environment installed by scripts/setup_opensees.py at
#      <cache>/opensees/<lock-hash>/bin/NAME. <cache> is $STRUT_CACHE_DIR, else
#      $XDG_CACHE_HOME/strut, else ~/.cache/strut.

_strut_opensees_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_strut_opensees_lock="${_strut_opensees_repo_root}/scripts/opensees/opensees-linux-64.lock"

strut_cache_root() {
  if [[ -n "${STRUT_CACHE_DIR:-}" ]]; then
    printf '%s\n' "${STRUT_CACHE_DIR}"
  else
    printf '%s/strut\n' "${XDG_CACHE_HOME:-${HOME}/.cache}"
  fi
}

# Keep in sync with lock_hash() in scripts/setup_opensees.py.
strut_opensees_prefix() {
  [[ -f "${_strut_opensees_lock}" ]] || return 1
  local hash
  hash="$(sha256sum "${_strut_opensees_lock}" | cut -c1-16)"
  printf '%s/opensees/%s\n' "$(strut_cache_root)" "${hash}"
}

# Prints the locked environment prefix if it is fully installed.
strut_opensees_ready_prefix() {
  local prefix
  prefix="$(strut_opensees_prefix)" || return 1
  [[ -f "${prefix}/.strut-ready" ]] || return 1
  printf '%s\n' "${prefix}"
}

# strut_resolve_opensees_binary NAME OVERRIDE
# Prints the path for NAME. Falls back to the expected locked path (even if
# missing) so the caller's existence check reports a useful location.
strut_resolve_opensees_binary() {
  local name="$1"
  local override="${2:-}"
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
    return 0
  fi
  local local_build="${_strut_opensees_repo_root}/.build/opensees-linux/${name}"
  if [[ -x "${local_build}" ]]; then
    printf '%s\n' "${local_build}"
    return 0
  fi
  local prefix
  prefix="$(strut_opensees_prefix)" || prefix="$(strut_cache_root)/opensees/<missing-lock>"
  printf '%s/bin/%s\n' "${prefix}" "${name}"
}

strut_opensees_install_hint() {
  echo "hint: install the locked OpenSees binaries (shared by all worktrees) with:" >&2
  echo "  uv run scripts/setup_opensees.py" >&2
}

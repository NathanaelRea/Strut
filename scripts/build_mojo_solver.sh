#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${repo_root}/build/mojo"
out_bin="${out_dir}/strut"

mkdir -p "${out_dir}"
mojo build "${repo_root}/src/mojo/strut.mojo" -o "${out_bin}"
echo "Built ${out_bin}"

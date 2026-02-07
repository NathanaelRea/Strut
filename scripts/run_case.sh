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

enabled="$(python - <<'PY' "$tgt_json"
import json, sys
path = sys.argv[1]
data = json.loads(open(path).read())
print(str(data.get("enabled", True)).lower())
PY
)"
if [[ "${enabled}" != "true" && "${STRUT_FORCE_CASE:-}" != "1" ]]; then
  exit 0
fi

tcl_out="${case_root}/generated/model.tcl"
"${json_to_tcl}" "$tgt_json" "$tcl_out"

refresh_reference=0
if [[ "${STRUT_REFRESH_REFERENCE:-}" == "1" ]]; then
  refresh_reference=1
fi

ref_hash_file="${case_root}/reference/.ref_hash"
if [[ $refresh_reference -eq 0 ]]; then
  need_refresh="$(python - <<'PY' "$tgt_json" "$ref_hash_file"
import hashlib
import sys
from pathlib import Path

case_path = Path(sys.argv[1])
hash_path = Path(sys.argv[2])

data = case_path.read_bytes()
digest = hashlib.sha256(data).hexdigest()

if not hash_path.exists():
    print("true")
    raise SystemExit(0)

stored = hash_path.read_text().strip()
print("false" if stored == digest else "true")
PY
)"
  if [[ "$need_refresh" == "true" ]]; then
    refresh_reference=1
  fi
fi

if [[ $refresh_reference -eq 1 ]]; then
  "${runner}" --script "$tcl_out" --output "${case_root}/reference"
  python - <<'PY' "$tgt_json" "$ref_hash_file"
import hashlib
import sys
from pathlib import Path

case_path = Path(sys.argv[1])
hash_path = Path(sys.argv[2])
hash_path.parent.mkdir(parents=True, exist_ok=True)
digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
hash_path.write_text(digest + "\n")
PY
fi

"${repo_root}/scripts/run_mojo_case.py" --input "$tgt_json" --output "${case_root}/mojo"
"${repo_root}/scripts/compare_case.py" --case "$case_name"

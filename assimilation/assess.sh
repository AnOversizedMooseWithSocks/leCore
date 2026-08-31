#!/usr/bin/env bash
# assess.sh -- measure every model the pipeline produced.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
VPY="assimilation/.venv/bin/python"; [ -x "$VPY" ] || VPY="python3"
OUT="assimilation/assessments"; mkdir -p "$OUT"
if [ $# -gt 0 ]; then
  "$VPY" assimilation/galvatron.py "$1" --assess "$PWD/$OUT/$(basename "$1").npz"
  exit 0
fi
for d in assimilation/work/*/; do
  [ -f "$d/model.safetensors" ] || continue
  echo; echo "=== $(basename "$d") ==="
  "$VPY" assimilation/galvatron.py "${d%/}" --assess "$PWD/$OUT/$(basename "${d%/}").npz"
done
echo; echo "  All bundles in $OUT -- send the folder."

#!/usr/bin/env bash
# run_galvatron.sh -- run a Galvatron bundle you already built.
#   ./run_galvatron.sh              find a bundle, then chat
#   ./run_galvatron.sh info
#   ./run_galvatron.sh serve --port 5930
#   ./run_galvatron.sh --bundle work/other chat
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
VPY="assimilation/.venv/bin/python"
[ -x "$VPY" ] || VPY="python3"
BUNDLE=""
if [ "${1:-}" = "--bundle" ]; then BUNDLE="$2"; shift 2; fi
if [ -z "$BUNDLE" ]; then
  for d in assimilation/work/*/; do
    [ -f "$d/galvatron.py" ] && BUNDLE="${d%/}"
  done
fi
if [ -z "$BUNDLE" ] || [ ! -f "$BUNDLE/galvatron.py" ]; then
  echo "  [!] No Galvatron bundle under assimilation/work."
  echo "      Build one:  assimilation/assimilate.sh --ban \"words to forbid\""
  exit 1
fi
echo "  bundle: $BUNDLE"
exec "$VPY" "$BUNDLE/galvatron.py" "${@:-chat}"

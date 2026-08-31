#!/bin/sh
# Print the facts an install decision depends on. Run from assimilation/:
#   ./diagnose.sh                  looks at work/original
#   ./diagnose.sh work/galvatron   or any other model
export GALVATRON_CWD="$PWD"
cd "$(dirname "$0")/.." || exit 1
PYTHONHASHSEED=0 python3 tools/diagnose_install.py "${1:-work/original}"

#!/bin/sh
# Unicron makes Galvatron. One step, no pipeline.
#   ./install.sh                      work/original -> work/galvatron
#   ./install.sh MODEL_DIR            MODEL_DIR     -> work/galvatron
#   ./install.sh MODEL_DIR OUT_DIR    wherever you like
# Optional: --doc FILE  --registers N  --passages N
# keep the caller's directory so relative paths still mean what they say
export GALVATRON_CWD="$PWD"
cd "$(dirname "$0")/.." || exit 1
SRC="${1:-work/original}"
case "$SRC" in --*) SRC="work/original";; *) [ $# -gt 0 ] && shift;; esac
DST="${1:-work/galvatron}"
case "$DST" in --*) DST="work/galvatron";; *) [ $# -gt 0 ] && shift;; esac
if [ ! -d "$SRC" ]; then
  echo "  [!] no model found at $SRC"
  echo "      run assimilate.sh first, or pass the path: ./install.sh /path/to/model"
  exit 1
fi
echo "  $SRC  ->  $DST"
PYTHONHASHSEED=0 python3 assimilation/install.py "$SRC" "$DST" "$@"

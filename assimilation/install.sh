#!/bin/sh
# Unicron makes Galvatron. One step, no pipeline.
#   ./install.sh                      work/original -> work/galvatron
#   ./install.sh MODEL_DIR            MODEL_DIR     -> work/galvatron
#   ./install.sh MODEL_DIR OUT_DIR    wherever you like
# Optional: --doc FILE  --registers N  --passages N
# keep the caller's directory so relative paths still mean what they say
export GALVATRON_CWD="$PWD"
cd "$(dirname "$0")/.." || exit 1
SRC="${1:-}"
# THE PIPELINE'S POINT (cp95): no argument -> prefer the assimilated model.
if [ -z "$SRC" ]; then
    if [ -f "assimilation/work/assimilated/config.json" ] || [ -f "work/assimilated/config.json" ]; then
        SRC="work/assimilated"
        echo "  [pipeline] no model given -- using work/assimilated (assimilate -> install = Galvatron)"
    else
        SRC="work/original"
        echo "  [pipeline] work/assimilated not found -- using work/original"
    fi
fi
case "$SRC" in --*) SRC="work/original";; *) [ $# -gt 0 ] && shift;; esac
DST="${1:-work/galvatron}"
case "$DST" in --*) DST="work/galvatron";; *) [ $# -gt 0 ] && shift;; esac
if [ ! -d "$SRC" ]; then
  echo "  [!] no model found at $SRC"
  echo "      run assimilate.sh first, or pass the path: ./install.sh /path/to/model"
  exit 1
fi
echo "  $SRC  ->  $DST"
PYTHONHASHSEED=0
# GPU for the install pipeline (cp88): cupy engages the engine's gpu backend
# under --device auto; wheel bundles the CUDA runtime; cpu numpy is the fallback.
if command -v nvidia-smi >/dev/null 2>&1; then
    "${VPY:-python3}" -c "import cupy; cupy.asarray([1.,2.,3.])[cupy.asarray([0,2])]" >/dev/null 2>&1 || \
        "${VPY:-python3}" -m pip install --quiet "cupy-cuda12x[ctk]" 2>/dev/null || true
fi
python3 assimilation/install.py "$SRC" "$DST" "$@"

#!/usr/bin/env bash
# Galvatron -- run a REAL checkpoint inside leCore (no torch, no transformers).
# Text in, text out; reads the vocabulary from the model directory.
#   ./galvatron.sh work/assimilated --check-tokenizer
#   ./galvatron.sh work/original    --ppl "some text"      # and again on assimilated
#   ./galvatron.sh work/assimilated --report
#   ./galvatron.sh work/assimilated --chat                 # persists across runs
set -euo pipefail
export GALVATRON_CWD="$PWD"
cd "$(dirname "$0")/.."
if [ $# -lt 1 ]; then
  echo "usage: galvatron.sh MODEL_DIR [--chat | --ppl TEXT | --report | --demo | --leap | ...]"
  python3 assimilation/galvatron.py --help
  exit 1
fi
export PYTHONHASHSEED=0
VPY="assimilation/.venv/bin/python"
[ -x "$VPY" ] || VPY="python3"
exec "$VPY" assimilation/galvatron.py "$@"

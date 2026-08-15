#!/bin/sh
# Unicron makes Galvatron. One step, no pipeline.
#   ./install.sh --experimental                         use defaults
#   ./install.sh --experimental MODEL_DIR               choose source
#   ./install.sh --experimental MODEL_DIR OUT_DIR       choose both
# Optional: --doc FILE  --registers N  --passages N
# keep the caller's directory so relative paths still mean what they say
export GALVATRON_CWD="$PWD"
cd "$(dirname "$0")/.." || exit 1
# Do not pre-parse paths here. The launcher changes directory, while install.py
# deliberately resolves paths against GALVATRON_CWD first; checking them after
# cd was the Unix-only bug that rejected valid caller-relative model paths.
PYTHONHASHSEED=0 exec python3 assimilation/install.py "$@"

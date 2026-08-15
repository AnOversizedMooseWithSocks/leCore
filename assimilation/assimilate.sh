#!/bin/sh
# ============================================================================
#  assimilation/assimilate.sh -- download Qwen3.5-0.8B. Spectral filtering is
#  retained only as an explicit research control.
#
#      ./assimilation/assimilate.sh                 # download untouched weights
#      ./assimilation/assimilate.sh --research-spectral --eval
#      ./assimilation/assimilate.sh --model Qwen/Qwen3.5-2B    # other sizes
#
#  Everything installs into a private virtual environment at
#  assimilation/.venv (created on first run) -- your system Python is never
#  touched, and NO Hugging Face account or token is needed: the weights are
#  public and the download is anonymous by construction (token=False).
#
#  See assimilation/README.md before running either experimental path.
# ============================================================================
set -e
cd "$(dirname "$0")/.."                    # repo root, same convention as serve.sh
export PYTHONHASHSEED=0                    # the engine is deterministic and relies on this
export HF_HUB_DISABLE_TELEMETRY=1          # download only; report nothing anywhere

# --- find a Python 3 interpreter (same probe serve.sh uses) ------------------
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "  [!] Python 3 was not found on your PATH."
    echo "      Install it from https://www.python.org/downloads/ and re-run this."
    exit 1
fi

# --- private virtual environment (first run only) ----------------------------
VENV="assimilation/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "  Creating virtual environment at $VENV (first run only)..."
    "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"

# --- dependencies: numpy + huggingface_hub always; torch stack only for --eval
#     (torch is the caller-side measurement instrument, never an engine dep)
"$VPY" -c "import numpy, huggingface_hub" >/dev/null 2>&1 || {
    echo "  Installing numpy + huggingface_hub into the venv..."
    "$VPY" -m pip install --quiet --upgrade pip
    "$VPY" -m pip install --quiet numpy huggingface_hub
}
for arg in "$@"; do
    if [ "$arg" = "--eval" ]; then
        "$VPY" -c "import torch, transformers" >/dev/null 2>&1 || {
            echo "  Installing torch + transformers for --eval (one-time, large)..."
            "$VPY" -m pip install --quiet torch transformers
        }
    fi
done

# --- run ---------------------------------------------------------------------
exec "$VPY" assimilation/run.py --workdir assimilation/work "$@"

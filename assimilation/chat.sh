#!/bin/sh
# ============================================================================
#  assimilation/chat.sh -- talk to the model you just assimilated.
#
#      ./assimilation/chat.sh               # chat with the assimilated model
#      ./assimilation/chat.sh --original    # chat with the untouched original
#      ./assimilation/chat.sh --both        # same prompt to both, side by side
#
#  Uses the same private venv assimilate.sh created; installs the runtime
#  (torch + transformers) into it on first use. No accounts, no tokens.
# ============================================================================
set -e
cd "$(dirname "$0")/.."
VPY="assimilation/.venv/bin/python"
if [ ! -x "$VPY" ]; then
    echo "  [!] Run ./assimilation/assimilate.sh first (it creates the venv and the model)."
    exit 1
fi
"$VPY" -c "import torch, transformers" >/dev/null 2>&1 || {
    echo "  Installing torch + transformers into the venv (one-time, large)..."
    "$VPY" -m pip install --quiet torch transformers
}
exec "$VPY" assimilation/chat.py "$@"

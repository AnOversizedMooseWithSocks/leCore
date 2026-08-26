#!/bin/sh
# ============================================================================
#  serve.sh -- start the leCore standalone API service on Linux / macOS.
#
#  Talks to the engine over HTTP/JSON (see holographic_service.py for the API).
#  The service is STDLIB-ONLY except for numpy, so there is almost nothing to
#  install: this script finds Python 3, makes sure numpy is present, and starts
#  the server. Any extra arguments are passed straight through, e.g.:
#
#      ./serve.sh                       # local only, port 8080
#      ./serve.sh --port 9000           # a different port
#      ./serve.sh --host 0.0.0.0 --token secret   # expose on the network (behind
#                                                 # auth/TLS on a TRUSTED network only)
# ============================================================================
set -e
cd "$(dirname "$0")"                       # run from the repo folder regardless of where we were called
export PYTHONHASHSEED=0                    # the engine is deterministic and relies on this being fixed

# --- find a Python 3 interpreter --------------------------------------------
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

# --- make sure numpy (the only hard dependency) is available ----------------
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "  Installing numpy (first run only)..."
    if ! "$PY" -m pip install --user numpy; then
        echo "  [!] Could not install numpy automatically."
        echo "      Install it yourself with:  $PY -m pip install numpy"
        exit 1
    fi
fi

# --- launch (pass through any --host / --port / --token the user supplied) ---
echo "  Starting the leCore API service.  Press Ctrl+C here to stop."
exec "$PY" holographic_service.py "$@"

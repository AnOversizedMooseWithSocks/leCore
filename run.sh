#!/usr/bin/env bash
# leCore chat - unix launcher (cp63). Mirrors run.bat: local venv, deps, chat UI.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --disable-pip-version-check numpy flask matplotlib
export PYTHONHASHSEED=0
( sleep 2; xdg-open http://127.0.0.1:7860 2>/dev/null || open http://127.0.0.1:7860 2>/dev/null || true ) &
python chat_server.py

#!/usr/bin/env bash
# run_bench.sh -- the MINGW64 / Linux twin of run_bench.bat. Same contract: a throwaway venv in
# .venv-bench/ beside this script, nothing installed system-wide, results next to the script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-bench"
BENCH="$HERE/research/shader_retrieval/bench_gpu.py"

if [ "${1:-}" = "--clean" ]; then echo "[clean] removing $VENV"; rm -rf "$VENV"; shift; fi
[ -f "$BENCH" ] || { echo "ERROR: cannot find $BENCH -- run from the repo root"; exit 1; }

PY="$(command -v python3 || command -v python)" || { echo "ERROR: no Python found"; exit 1; }
echo "[python] using: $PY ($("$PY" --version 2>&1))"

if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/Scripts/python.exe" ]; then
  echo "[venv]   creating $VENV"; "$PY" -m venv "$VENV"
else
  echo "[venv]   reusing $VENV   (run with --clean to rebuild)"
fi
VPY="$VENV/bin/python"; [ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"

echo "[deps]   installing numpy, moderngl (quiet)"
"$VPY" -m pip install --upgrade pip --quiet --disable-pip-version-check
"$VPY" -m pip install --quiet --disable-pip-version-check "numpy>=1.24" "moderngl>=5.8" "glcontext>=2.5"

OUT="$HERE/results_$(hostname 2>/dev/null || echo machine).json"
echo "[run]    $BENCH"; echo
PYTHONPATH="$HERE" PYTHONHASHSEED=0 "$VPY" "$BENCH" --json "$OUT" "$@"
echo; echo "[done]   results written to $OUT"
echo "         The venv is at $VENV -- delete it whenever you like, nothing else was touched."

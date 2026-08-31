#!/usr/bin/env bash
# build_pages.sh -- the MINGW64 / Linux twin of build_pages.bat. Same contract: throwaway venv in
# .venv-bench/, pages rebuilt into pages/, nothing installed system-wide.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-bench"; GEN="$HERE/research/shader_retrieval"; OUT="$HERE/pages"
DOOPEN=1; DOBUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --clean) echo "[clean] removing $VENV"; rm -rf "$VENV";;
    --no-open) DOOPEN=0;;
    --open-only) DOBUILD=0;;
  esac; shift
done
mkdir -p "$OUT"
if [ "$DOBUILD" = "1" ]; then
  [ -f "$GEN/make_search_page.py" ] || { echo "ERROR: run from the repo root"; exit 1; }
  PY="$(command -v python3 || command -v python)" || { echo "ERROR: no Python"; exit 1; }
  echo "[python] using: $PY"
  if [ ! -x "$VENV/bin/python" ] && [ ! -x "$VENV/Scripts/python.exe" ]; then
    echo "[venv]   creating $VENV"; "$PY" -m venv "$VENV"
  else echo "[venv]   reusing $VENV   (--clean to rebuild)"; fi
  VPY="$VENV/bin/python"; [ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"
  # numpy for the maths, moderngl for three of the five generators (they import glsl_hier,
  # which builds a GL context). numpy alone made those three fail inside the venv while
  # passing on a host that happened to have moderngl -- a false green.
  echo "[deps]   installing numpy, moderngl (quiet)"
  "$VPY" -m pip install --upgrade pip --quiet --disable-pip-version-check
  "$VPY" -m pip install --quiet --disable-pip-version-check "numpy>=1.24" "moderngl>=5.8" "glcontext>=2.5"
  # Generators MUST run with the repo root as CWD -- they write into it, the corpus loader globs
  # relative to it, and several import engine modules by their FLAT name. Build there, then move.
  FAILED=""
  cd "$HERE"
  for g in make_search_page make_webgl_page make_webgl_full make_webgl_typed make_webgl_vsa; do
    echo "[build]  $g.py"
    PYTHONPATH="$HERE" PYTHONHASHSEED=0 "$VPY" "$GEN/$g.py" || FAILED="$FAILED $g"
  done
  mv -f "$HERE"/lecore_search_webgl2.html "$HERE"/lecore_webgl2*.html "$OUT"/ 2>/dev/null || true
  echo
  if [ -n "$FAILED" ]; then
    echo "[warn]   these generators failed:$FAILED"
    echo "         The checked-in pages in $OUT are still valid; only those are stale."
  else echo "[done]   all pages rebuilt in $OUT"; fi
  ls -1 "$OUT"/*.html
fi
if [ "$DOOPEN" = "1" ]; then
  echo; echo "[open]   $OUT/lecore_search_webgl2.html"
  (xdg-open "$OUT/lecore_search_webgl2.html" 2>/dev/null \
    || start "" "$OUT/lecore_search_webgl2.html" 2>/dev/null \
    || echo "         open it manually") &
fi

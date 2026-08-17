#!/bin/sh
# build_package.sh -- build an installable leOS-core wheel from the flat repo.
#
# The idea you described: copy ONLY what the package needs into a clean staging folder -- the
# holographic_*.py modules minus the tests, plus the packaging files -- then build the wheel from there.
# Building from a clean copy guarantees no test files, data, docs or demos leak into the distribution.
#
# Requires:  pip install build
# Usage:     sh build_package.sh
# Output:    dist/leos_core-<version>-py3-none-any.whl   and   dist/leos_core-<version>.tar.gz
#            (setuptools writes the distribution name "leos-core" as "leos_core" in the filenames; the
#             import name is still "lecore" -- that is the lecore.py shim, unaffected by the dist name.)
set -e

STAGE=build_pkg          # the clean staging folder we assemble the package in
DIST=dist                # where the finished wheel + sdist land

echo ">> cleaning any previous build"
rm -rf "$STAGE" "$DIST"
mkdir -p "$STAGE"

echo ">> copying the essential modules (the holographic/ package, plus the standalone service script) WITHOUT the tests"
cp -r holographic "$STAGE"/
# belt-and-suspenders: strip any stray test files that ended up inside the package tree
find "$STAGE"/holographic -name "test_*.py" -delete 2>/dev/null || true
[ -f holographic_service.py ] && cp holographic_service.py "$STAGE"/ || true

echo ">> copying the packaging files and the convenience shim"
cp setup.py pyproject.toml lecore.py "$STAGE"/
# VERSION is the single source of truth that setup.py reads for version=; without it in the stage the wheel would
# build as the 0.0.0 fallback. lecore.__version__ reads the recorded metadata once installed, so it does not need
# VERSION at runtime -- but the BUILD does.
cp VERSION "$STAGE"/
[ -f LICENSE ]   && cp LICENSE   "$STAGE"/ || true
[ -f README.md ] && cp README.md "$STAGE"/ || true
[ -f MANIFEST.in ] && cp MANIFEST.in "$STAGE"/ || true

echo ">> copying the runtime data package (lecore_data/: the dictionary + material JSON the engine needs at runtime)"
cp -r lecore_data "$STAGE"/

# capabilities.json IS THE ONE ARTIFACT WHOSE WHOLE POINT IS BEING READ WITHOUT
# IMPORTING THE ENGINE -- the machine-readable sibling of CAPABILITIES.md, for
# tools and apps that ingest the catalog. Leaving it out of the wheel means the
# audience it exists for is exactly the audience that cannot get it: a pip user
# has no repo to read it from and no capdoc.py to regenerate it with.
# It rides inside lecore_data/ rather than at the top level so it travels with
# the package_data rule that is already proven to work (the dictionary check in
# package.yml would have caught a data file that did not ship).
if [ -f capabilities.json ]; then
  cp capabilities.json "$STAGE"/lecore_data/
  echo ">> bundled capabilities.json ($(wc -c < capabilities.json) bytes) for "\
       "consumers that read the catalog without importing the engine"
fi
# keep the distribution clean: no compiled caches or stray pyc leak into the wheel
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true

echo ">> building the wheel and sdist from the clean copy"
cd "$STAGE"
python -m build --outdir "../$DIST"
cd ..

echo ">> done -- artifacts in $DIST/:"
ls -1 "$DIST"

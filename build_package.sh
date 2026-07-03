#!/bin/sh
# build_package.sh -- build an installable leCore wheel from the flat repo.
#
# The idea you described: copy ONLY what the package needs into a clean staging folder -- the
# holographic_*.py modules minus the tests, plus the packaging files -- then build the wheel from there.
# Building from a clean copy guarantees no test files, data, docs or demos leak into the distribution.
#
# Requires:  pip install build
# Usage:     sh build_package.sh
# Output:    dist/lecore-<version>-py3-none-any.whl   and   dist/lecore-<version>.tar.gz
set -e

STAGE=build_pkg          # the clean staging folder we assemble the package in
DIST=dist                # where the finished wheel + sdist land

echo ">> cleaning any previous build"
rm -rf "$STAGE" "$DIST"
mkdir -p "$STAGE"

echo ">> copying the essential modules (holographic_*.py) WITHOUT the tests"
for f in holographic_*.py; do
    case "$f" in
        test_*|*_test.py) continue ;;    # skip tests (glob already excludes test_holographic_*; this is safety)
    esac
    cp "$f" "$STAGE"/
done

echo ">> copying the packaging files and the convenience shim"
cp setup.py pyproject.toml lecore.py "$STAGE"/
[ -f LICENSE ]   && cp LICENSE   "$STAGE"/ || true
[ -f README.md ] && cp README.md "$STAGE"/ || true

echo ">> building the wheel and sdist from the clean copy"
cd "$STAGE"
python -m build --outdir "../$DIST"
cd ..

echo ">> done -- artifacts in $DIST/:"
ls -1 "$DIST"

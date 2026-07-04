#!/usr/bin/env python3
"""
check_version.py -- read the package version out of setup.py, and (optionally) assert it matches an expected value.

WHY THIS EXISTS. Releases are cut by pushing a git tag like `v0.2.0`, but the version that actually gets published is
the one written in setup.py. If you tag `v0.2.0` and forget to bump setup.py (still `0.1.0`), you'd publish the wrong
number -- or, since PyPI never lets you re-upload a version, hit a confusing failure. This tiny check closes that gap:
CI runs it on a tag push and fails the release if the tag and setup.py disagree, BEFORE anything is published.

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only. It reads setup.py with `ast` (it does NOT execute setup.py),
finds the setup(...) call, and pulls out the `version=` keyword -- so there is nothing to import and nothing to run.

    Print the version:   python tools/check_version.py
    Assert a match:      python tools/check_version.py --expect 0.2.0     # exit 1 if setup.py != 0.2.0
                         python tools/check_version.py --expect v0.2.0    # a leading 'v' (as in a git tag) is fine
"""
import ast
import os
import sys


def read_version(setup_path):
    """Return the string passed as version= to the setup(...) call in setup.py, or None if not found. AST only."""
    tree = ast.parse(open(setup_path, encoding="utf-8").read())
    for node in ast.walk(tree):
        # we want the call `setup(...)` -- match either `setup(` or `setuptools.setup(`
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name == "setup":
                for kw in node.keywords:
                    if kw.arg == "version" and isinstance(kw.value, ast.Constant):
                        return str(kw.value.value)
    return None


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tools/ is one level down)
    setup_path = os.path.join(root, "setup.py")
    version = read_version(setup_path)
    if version is None:
        print("could not find version= in setup.py")
        return 2

    expect = None
    if "--expect" in argv:
        expect = argv[argv.index("--expect") + 1]
        expect = expect[1:] if expect.startswith("v") else expect     # allow a git-tag-style leading 'v'

    if expect is None:
        print(version)                                                # just report it
        return 0

    if version != expect:
        print("VERSION MISMATCH: setup.py says %r but the tag/expected version is %r.\n"
              "Bump setup.py's version= to match the tag (or tag the version setup.py already has)." % (version, expect))
        return 1
    print("version OK: setup.py and the tag agree on %s" % version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

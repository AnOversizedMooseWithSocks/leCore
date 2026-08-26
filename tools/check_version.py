#!/usr/bin/env python3
"""
check_version.py -- print the package version, and (optionally) assert it matches an expected value.

WHY THIS EXISTS. The single source of truth for the version is the top-level VERSION file (setup.py, lecore and CI
all read it). This tiny helper prints that version for scripts/CI and can assert it equals an expected string --
useful as a sanity gate. Since CI now bumps VERSION automatically on each merge to main (tools/bump_version.py) and
publishes without any tag, there is no tag-vs-setup.py mismatch to guard anymore; this stays as a dependency-free
way to read and check the number.

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only. It reads the VERSION file directly (and falls back to parsing
setup.py with `ast`, not executing it, for older layouts) -- so there is nothing to import and nothing to run.

    Print the version:   python tools/check_version.py
    Assert a match:      python tools/check_version.py --expect 0.2.0     # exit 1 if VERSION != 0.2.0
                         python tools/check_version.py --expect v0.2.0    # a leading 'v' is tolerated
"""
import ast
import os
import sys


def read_version(setup_path):
    """Return the package version. The single source of truth is now the top-level VERSION file (setup.py reads
    it via read_version(), so it is no longer an AST-visible constant). Prefer VERSION; fall back to parsing a
    literal version= out of setup.py for older layouts. AST/text only -- nothing imported, nothing run."""
    root = os.path.dirname(os.path.abspath(setup_path))
    version_file = os.path.join(root, "VERSION")
    if os.path.exists(version_file):
        raw = open(version_file, encoding="utf-8").read().strip()
        if raw:
            return raw
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
        print("VERSION MISMATCH: the VERSION file says %r but the expected version is %r.\n"
              "Edit the VERSION file to match (or update what you are checking against)." % (version, expect))
        return 1
    print("version OK: %s" % version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""test_changed.py -- run only the tests affected by your CURRENT changes.

A developer convenience built on tools/select_tests.py. It asks git what you've touched (staged, unstaged, and new
untracked files), asks the import graph which test files those changes affect, and runs pytest on just those -- so a
one-line edit to a leaf module doesn't make you sit through the whole suite.

    python tools/test_changed.py               # affected tests vs your working tree
    python tools/test_changed.py --since main  # affected tests vs another branch/commit (e.g. before a PR)
    python tools/test_changed.py -- -x -q      # everything after `--` is passed straight to pytest

It errs toward SAFETY: if a change can't be reasoned about statically (a data file, a brand-new .py), it runs the
FULL suite rather than risk skipping something. If nothing is affected (e.g. you only edited a README), it says so
and runs nothing.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # so we can import the sibling select_tests
from select_tests import affected_tests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(args):
    out = subprocess.run(["git", "-C", REPO] + args, capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else []


def changed_files(since=None):
    """The files you've changed. Against `since` (a branch/commit) if given -- that's the PR view: everything your
    branch adds on top of it. Otherwise the working-tree view: tracked edits vs HEAD plus new untracked files."""
    if since:
        return _git(["diff", "--name-only", "%s...HEAD" % since])
    tracked = _git(["diff", "--name-only", "HEAD"])                # staged + unstaged edits to tracked files
    untracked = _git(["ls-files", "--others", "--exclude-standard"])  # brand-new files
    return sorted(set(tracked) | set(untracked))


def main(argv):
    since = None
    pytest_extra = []
    if "--" in argv:                                              # pass-through: everything after `--` goes to pytest
        i = argv.index("--")
        pytest_extra = argv[i + 1:]
        argv = argv[:i]
    if "--since" in argv:
        j = argv.index("--since")
        since = argv[j + 1]

    files = changed_files(since)
    if not files:
        print("No changes detected -- nothing to test.")
        return 0

    print("Changed files:")
    for f in files:
        print("  " + f)

    selected = affected_tests(files, root=REPO)
    if selected == "ALL":
        print("\nA change here can't be scoped statically -- running the FULL suite to be safe.")
        args = []
    elif not selected:
        print("\nNo test is affected by these changes (docs/config only). Nothing to run.")
        return 0
    else:
        print("\n%d affected test file(s):" % len(selected))
        for t in selected:
            print("  " + t)
        args = selected

    cmd = [sys.executable, "-m", "pytest", "-q"] + pytest_extra + args
    print("\n$ " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""bump_version.py -- increment the PATCH digit of the top-level VERSION file, in place.

THE VERSIONING CONTRACT (so a human and CI never fight over the number):
  * The VERSION file is the single source of truth. setup.py, lecore.__version__ and check_version.py all read it.
  * YOU own major.minor. To cut a 0.3 or 1.0 line, hand-edit VERSION to "0.3.0" / "1.0.0" and commit -- that's it.
  * CI owns patch. On each merge to main, the release workflow runs `bump_version.py`, which increments ONLY the
    third digit (0.2.0 -> 0.2.1 -> 0.2.2 ...), commits it back with [skip ci], and publishes that version.
  * So a hand-edit to 0.3.0 is immediately followed (next merge) by 0.3.1, 0.3.2, ... -- the auto-bumps resume
    from wherever you set the line. There is never a tag to manage.

WHY PATCH-ONLY AND NOT A RESET: bumping only the last digit is a pure, total function of the current string, so
two merges racing produce the same next number regardless of order of the DIGITS; the git push (fast-forward on
main) is what actually serialises them. Resetting patch to 0 on a minor change would need to know "did the
major.minor change since last publish?", which coples this tool to history -- kept out on purpose.

Usage:
    python tools/bump_version.py            # 0.2.0 -> 0.2.1, rewrites VERSION, prints the new version
    python tools/bump_version.py --print    # just print what the NEXT version WOULD be, write nothing
    python tools/bump_version.py --current  # print the current version, write nothing

Exit code 0 on success; 1 if VERSION is missing or malformed (so CI fails loudly rather than publishing garbage).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_FILE = os.path.join(ROOT, "VERSION")


def read_current():
    """Return (major, minor, patch) ints, or raise ValueError if VERSION is missing/malformed."""
    if not os.path.exists(VERSION_FILE):
        raise ValueError("VERSION file not found at %s" % VERSION_FILE)
    raw = open(VERSION_FILE, encoding="utf-8").read().strip()
    parts = raw.split(".")
    # We require exactly major.minor.patch, all non-negative ints -- anything else means someone hand-edited it
    # into a shape the auto-bumper can't reason about, and we must NOT guess and publish a wrong number.
    if len(parts) != 3:
        raise ValueError("VERSION must be 'major.minor.patch', got %r" % raw)
    try:
        major, minor, patch = (int(p) for p in parts)
    except ValueError:
        raise ValueError("VERSION components must be integers, got %r" % raw)
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError("VERSION components must be non-negative, got %r" % raw)
    return major, minor, patch


def next_version():
    major, minor, patch = read_current()
    return "%d.%d.%d" % (major, minor, patch + 1)      # ONLY the patch digit moves


def current_version():
    major, minor, patch = read_current()
    return "%d.%d.%d" % (major, minor, patch)


def main(argv):
    try:
        if "--current" in argv:
            print(current_version())
            return 0
        nxt = next_version()
        if "--print" in argv:
            print(nxt)                                  # dry run: don't touch the file
            return 0
        # write the bumped version back, keeping a single trailing newline (POSIX text file)
        with open(VERSION_FILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(nxt + "\n")
        print(nxt)
        return 0
    except ValueError as exc:
        print("bump_version: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

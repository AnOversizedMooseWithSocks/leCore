"""Tests for tools/bump_version.py -- the patch-only version bumper CI runs on each merge to main.

The load-bearing contracts, each pinned here because a wrong version is published to PyPI (append-only, no take-backs):
  * bumps ONLY the patch digit (0.2.0 -> 0.2.1, never touches major/minor),
  * hand-editing major.minor and letting the bump resume works (0.3.0 -> 0.3.1),
  * a malformed VERSION fails loudly (exit 1) instead of writing a garbage number,
  * dry-run flags (--print / --current) never modify the file.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUMP = os.path.join(ROOT, "tools", "bump_version.py")

# Each test bumps its OWN temporary VERSION file (the tool reads LECORE_VERSION_FILE). CI (2026-09-28): the tests used
# to rewrite the repo's real VERSION, and under `pytest -n auto` workers raced on it -- '--print' of 0.5.7 answered
# 0.5.9, and a malformed '1.2' test read another test's '1.4.101' and exited 0. A private file per test cannot race,
# and the tests no longer need the real VERSION at all (so they also run from a delivery archive, which omits it).
VERSION_FILE = None                                   # set per test by the fixture below


def _run(*args):
    env = dict(os.environ, LECORE_VERSION_FILE=VERSION_FILE)
    return subprocess.run([sys.executable, BUMP, *args], capture_output=True, text=True, env=env)


@pytest.fixture(autouse=True)
def restore_version(tmp_path):
    """A private VERSION file for this test (the name is kept from when this fixture restored the real one)."""
    global VERSION_FILE
    VERSION_FILE = str(tmp_path / "VERSION")
    yield
    VERSION_FILE = None


def _write(v):
    with open(VERSION_FILE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(v + "\n")


def test_bump_touches_only_the_patch_digit(restore_version):
    _write("0.2.0")
    r = _run()
    assert r.returncode == 0 and r.stdout.strip() == "0.2.1"
    assert open(VERSION_FILE, encoding="utf-8").read().strip() == "0.2.1"


def test_bump_resumes_after_a_hand_edited_minor(restore_version):
    """The whole point of the design: bump the major.minor by hand, and auto-bumps continue from there."""
    _write("0.3.0")                               # a human cutting a new minor line
    assert _run().stdout.strip() == "0.3.1"       # next merge auto-bumps the patch, not a reset or a minor jump
    assert _run().stdout.strip() == "0.3.2"


def test_bump_does_not_roll_into_minor(restore_version):
    _write("1.4.99")
    assert _run().stdout.strip() == "1.4.100"     # patch just increments; no carry into minor (semver patch is unbounded)


def test_print_and_current_are_dry_runs(restore_version):
    _write("0.5.7")
    assert _run("--print").stdout.strip() == "0.5.8"
    assert _run("--current").stdout.strip() == "0.5.7"
    assert open(VERSION_FILE, encoding="utf-8").read().strip() == "0.5.7", "dry runs must not write"


@pytest.mark.parametrize("bad", ["not.a.version", "1.2", "1.2.3.4", "1.x.0", ""])
def test_malformed_version_fails_loudly(restore_version, bad):
    _write(bad)
    r = _run()
    assert r.returncode == 1, "a malformed VERSION must exit 1, never publish a guessed number"
    # and it must not have overwritten the (bad) file with a guess
    assert open(VERSION_FILE, encoding="utf-8").read().strip() == bad.strip()

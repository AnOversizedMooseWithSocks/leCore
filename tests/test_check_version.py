"""Tests for tools/check_version.py -- the tag-vs-setup.py version guard used at release time."""
import os
import sys
import subprocess
import importlib.util


def _mod():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "check_version.py")
    spec = importlib.util.spec_from_file_location("check_version", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reads_the_version_from_setup():
    cv = _mod()
    v = cv.read_version(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"))
    assert v and all(part.isdigit() for part in v.split("."))     # looks like N.N.N


def _run(*args):
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "check_version.py")
    return subprocess.run([sys.executable, tool, *args], capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_matching_version_passes():
    cv = _mod()
    v = cv.read_version(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"))
    r = _run("--expect", v)
    assert r.returncode == 0 and "OK" in r.stdout


def test_leading_v_is_accepted():
    cv = _mod()
    v = cv.read_version(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"))
    r = _run("--expect", "v" + v)                                  # a git-tag-style 'v' prefix
    assert r.returncode == 0


def test_mismatch_fails_nonzero():
    r = _run("--expect", "999.999.999")
    assert r.returncode == 1 and "MISMATCH" in r.stdout


def test_bare_invocation_prints_version():
    r = _run()
    assert r.returncode == 0 and r.stdout.strip()

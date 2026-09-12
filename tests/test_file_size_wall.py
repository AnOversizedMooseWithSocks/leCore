"""THE SIZE WALL: working with files an agent cannot read whole (sweep 132).

WHY THIS FILE EXISTS. leCore's agent-facing read caps at 1 MB to stop a whole-file read flooding a
context window -- a deliberate guard, and it is NOT what this sweep changed. What it changed is the
ergonomics around it, which had quietly become worse than the problem: three of the repo's own most
important artifacts had crossed the cap (docs/NOTES_concepts.md at 6.19 MB, REFERENCE.md at 2.54 MB,
capabilities.json at 1.09 MB), and to append one line to the notebook over /invoke an agent had to
call file_read_lines over the WHOLE file -- 85,602 lines, 6,451,243 bytes -- purely to learn where
the end was. The cap on read() defeated through the back door, because the only faculty that yielded
a line count returned every line.

THE NON-GOAL IS PINNED AS HARD AS THE GOAL. `test_the_read_cap_still_refuses` exists because the
obvious fix -- raise the cap -- turns a loud refusal into a silent 6 MB context bomb. If a future
sweep makes that test pass by lifting the cap, the guard has been removed, not fixed.

The other trap here is `test_the_two_tail_paths_agree_exactly`: the tail has a fast seek-from-the-end
path and a slow read-the-whole-file path, and two implementations of one answer drift. They are
asserted equal on both line endings, at the file's edge, and past its start.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

import lecore
from holographic.io_and_interop.holographic_codeedit import Editor, EditError, _Truncate

CAP = 1_000_000
LINE = "a line of notebook prose about a sweep and what it measured\n"


@pytest.fixture
def ed(tmp_path):
    return Editor(str(tmp_path))


@pytest.fixture(scope="module")
def repo_editor():
    return Editor(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------------------
# 1. THE NON-GOAL. The cap is a guard, not a bug.
# --------------------------------------------------------------------------------------

def test_the_read_cap_still_refuses(ed):
    # THE TEST THAT MUST NOT BE MADE TO PASS BY LIFTING THE CAP. A loud refusal beats a silent
    # 6 MB context bomb; this sweep adds doors that do not need the whole file, it does not widen
    # the hole. If this ever fails, read the sentence above before "fixing" it.
    ed.write("big.py", "x = 0\n" * 200_000)
    assert ed.stat("big.py")["bytes"] > CAP
    with pytest.raises(EditError, match="max_bytes"):
        ed.read("big.py")
    # ...while every bounded-output door works on the very same file
    assert ed.read_lines("big.py", 5, 6) == ["x = 0", "x = 0"]
    assert ed.stat("big.py")["lines"] == 200_000
    assert ed.python_check("big.py") == {"ok": True, "error": None}


def test_the_engines_own_oversized_artifacts_are_reachable(repo_editor):
    # The regression this sweep exists for, asserted against the REAL files. If NOTES_concepts.md
    # ever becomes unreadable-by-any-door again, this fails loudly.
    for rel in ("docs/NOTES_concepts.md", "REFERENCE.md", "capabilities.json"):
        if not repo_editor.exists(rel):
            pytest.skip("%s absent from this checkout" % rel)
        st = repo_editor.stat(rel)
        assert st["over_read_cap"] is True, rel
        with pytest.raises(EditError):
            repo_editor.read(rel)
        assert len(repo_editor.read_lines(rel, 1, 3)) == 3, rel
        assert len(repo_editor.read_lines(rel, start=-5)) == 5, rel
        assert repo_editor.view(rel, -2).count("\n") == 1, rel
        assert st["lines"] > 0 and len(st["sha256"]) == 64


# --------------------------------------------------------------------------------------
# 2. APPEND: grows a file without reading it, and undoes for four bytes.
# --------------------------------------------------------------------------------------

def test_append_records_a_size_not_a_copy_of_the_file(ed):
    # THE PROOF THAT IT DOES NOT READ. Going through _atomic_write would snapshot the file's full
    # prior TEXT onto the undo stack -- 6.1 MB per append of the notebook, 100 deep, 610 MB of undo
    # to add one line. The record being an integer is what makes the cheap operation cheap.
    ed.write("big.md", LINE * 20_000)
    before = ed.stat("big.md")["bytes"]
    ed.append("big.md", "one more line\n")
    rec = ed._undo[-1][1]
    assert isinstance(rec, _Truncate), "append snapshotted the file instead of its size"
    assert rec.size == before
    assert not isinstance(rec, str)


def test_append_arithmetic_and_undo_are_exact(ed):
    ed.write("log.md", "line one\n")
    r = ed.append("log.md", "line two\n")
    assert (r["size_before"], r["size_after"], r["appended_bytes"]) == (9, 18, 9)
    assert r["created"] is False and r["separator_added"] is False
    before = ed.read("log.md")
    ed.append("log.md", "line three\n")
    ed.undo()
    assert ed.read("log.md") == before


def test_an_append_that_creates_the_file_is_undone_by_removing_it(ed):
    r = ed.append("fresh.md", "hello\n")
    assert r["created"] is True and r["size_before"] == 0
    assert ed.exists("fresh.md")
    ed.undo()
    assert not ed.exists("fresh.md")


def test_ensure_newline_stops_the_weld(ed):
    # The single way an append corrupts a document: the first appended line joining the last
    # existing one when the file did not end in a newline.
    ed.write("noeol.md", "tail without newline")
    r = ed.append("noeol.md", "next line\n")
    assert r["separator_added"] is True
    assert ed.read("noeol.md") == "tail without newline\nnext line\n"


def test_the_weld_can_be_opted_into(ed):
    ed.write("x.md", "abc")
    ed.append("x.md", "def", ensure_newline=False)
    assert ed.read("x.md") == "abcdef"


# --------------------------------------------------------------------------------------
# 3. STAT: the measurement that had no door.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("body", ["a\nb\nc\n", "a\nb\nc", "", "\n", "one line"])
def test_stat_counts_lines_exactly_as_splitlines_does(ed, body):
    # An off-by-one here is an off-by-one in somebody else's insert, so it is checked against the
    # definition an agent will actually compare against, on both endings and on the empty file.
    ed.write("s.txt", body)
    st = ed.stat("s.txt")
    assert st["lines"] == len(body.splitlines()), (body, st)
    assert st["bytes"] == len(body.encode())
    assert st["ends_with_newline"] is body.endswith("\n")


def test_stat_sha256_is_hashlib_stable_across_processes(ed):
    ed.write("sha.txt", "a")
    here = ed.stat("sha.txt")["sha256"]
    assert here.startswith("ca978112ca1bbdca")          # sha256('a'), fixed forever
    code = ("import sys; sys.path.insert(0, %r); "
            "from holographic.io_and_interop.holographic_codeedit import Editor; "
            "print(Editor(%r).stat('sha.txt')['sha256'])" % (_repo(), str(ed.root)))
    for seed in ("0", "1", "77777"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=_env(seed), check=True).stdout.strip()
        assert out == here, "stat sha256 moved with PYTHONHASHSEED=%s" % seed


def test_stat_answers_past_the_cap_where_read_refuses(ed):
    ed.write("big.py", "x = 0\n" * 200_000)
    st = ed.stat("big.py")
    assert st["over_read_cap"] is True and st["read_cap"] == CAP and st["lines"] == 200_000


def _repo():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(seed):
    e = dict(os.environ)
    e["PYTHONHASHSEED"] = seed
    e["PYTHONPATH"] = _repo() + os.pathsep + e.get("PYTHONPATH", "")
    return e


# --------------------------------------------------------------------------------------
# 4. THE TAIL: one answer, two paths, asserted identical.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("body", ["a\nb\nc\nd\ne\n", "a\nb\nc\nd\ne", LINE * 5000])
def test_the_two_tail_paths_agree_exactly(ed, body):
    # The fast path seeks from the end; the slow path reads everything. Two implementations of one
    # answer drift unless something pins them together, and the edges (n == len, n > len) are where
    # a block-reader gets it wrong.
    ed.write("t.txt", body)
    whole = body.splitlines()
    for n in (1, 2, len(whole), len(whole) + 7):
        assert ed.read_lines("t.txt", start=-n) == whole[-n:], n


def test_a_tail_view_keeps_absolute_line_numbers(ed):
    # view()'s job is producing coordinates an agent can then edit at. Numbering a tail from 1
    # would hand back numbers pointing at the wrong end of the file.
    ed.write("t.txt", "a\nb\nc\nd\n")
    v = ed.view("t.txt", -2)
    assert v == "3\tc\n4\td"


def test_a_positive_range_is_unchanged(ed):
    ed.write("t.txt", "a\nb\nc\nd\n")
    assert ed.read_lines("t.txt", 2, 3) == ["b", "c"]
    assert ed.view("t.txt", 2, 3) == "2\tb\n3\tc"


# --------------------------------------------------------------------------------------
# 5. THE MEASUREMENT, pinned.
# --------------------------------------------------------------------------------------

def test_append_costs_two_orders_of_magnitude_less_than_the_baseline(ed):
    # BASELINE = what an agent had to do before this sweep, /invoke only: read every line to learn
    # the count, then insert after it. Pinned as a RATIO so it survives a different fixture size.
    ed.write("n.md", LINE * 20_000)
    base = len(json.dumps(ed.read_lines("n.md")))
    now = len(json.dumps(ed.append("n.md", LINE)))
    assert base > 1_000_000 and now < 200
    assert base / now > 1000, "the wire-cost win collapsed: %d -> %d" % (base, now)


def test_the_tail_costs_a_window_not_a_file(ed):
    ed.write("n.md", LINE * 20_000)
    base = len(json.dumps(ed.read_lines("n.md")))
    now = len(json.dumps(ed.read_lines("n.md", start=-40)))
    assert now < 4000 and base / now > 100


# --------------------------------------------------------------------------------------
# 6. THE CANARY sees what it was built to see.
# --------------------------------------------------------------------------------------

def test_the_size_canary_walks_more_than_python_files():
    # It reported 0 for the whole of sweeps C7..131 while three artifacts sat past the cap, because
    # it walked holographic/**/*.py only. A size alarm blind to the files that crossed the threshold
    # is not an alarm.
    src = open(os.path.join(_repo(), "tools", "reachability_audit.py"), encoding="utf-8").read()
    canary = src[src.index("C7 canary"):src.index("DUPLICATE FACULTY DEFINITIONS")]
    assert 'fn.endswith(text_ext)' in canary, "the canary is filtering to one extension again"
    assert ".md" in canary and ".json" in canary
    # and it CLASSIFIES rather than crying wolf: a generated 2.5 MB reference is a fact, not a defect
    for kind in ("generated", "append-only", "source", "document"):
        assert '"%s"' % kind in canary or "'%s'" % kind in canary or kind in canary


def test_the_canary_reports_the_real_over_cap_artifacts():
    out = subprocess.run([sys.executable, "tools/reachability_audit.py"], cwd=_repo(),
                         capture_output=True, text=True, env=_env("0"))
    assert out.returncode == 0, out.stderr[-500:]
    line = [l for l in out.stdout.splitlines() if "SIZE CANARY" in l]
    assert line, out.stdout[-500:]
    assert "over the cap" in line[0]
    assert "docs/NOTES_concepts.md" in out.stdout and "[append-only]" in out.stdout
    assert "[generated]" in out.stdout, "generated artifacts must be classified, not alarmed about"
    assert "0 of them source" in line[0], "a SOURCE file over the cap is the actionable C7 shape"


# --------------------------------------------------------------------------------------
# 7. WIRING.
# --------------------------------------------------------------------------------------

def test_the_faculties_are_wired_to_the_mind_and_documented(tmp_path):
    m = lecore.UnifiedMind(dim=256, seed=0)
    for name in ("file_append", "file_stat"):
        fn = getattr(m, name, None)
        assert callable(fn), "%s is not wired to UnifiedMind" % name
        assert (fn.__doc__ or "").strip(), "%s has no docstring (undiscoverable)" % name
    m.set_file_root(str(tmp_path))
    m.file_write("n.md", "one\n")
    assert m.file_append("n.md", "two\n")["size_after"] == 8
    st = m.file_stat("n.md")
    assert st["lines"] == 2 and st["over_read_cap"] is False
    assert m.file_read_lines("n.md", start=-1) == ["two"]
    # JSON in, JSON out -- these are agent-facing by definition
    assert json.loads(json.dumps(st))["lines"] == 2

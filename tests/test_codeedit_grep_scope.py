"""file_grep with a FILE path must search that file, not its directory.

Found while auditing leCore with leCore. `grep(pattern, relpath="a/b/mod.py")`
did `walk_root = base if isdir(base) else dirname(base)` -- a file was WIDENED TO
ITS DIRECTORY. Scoping to one 31-match file returned 500 hits from 53 OTHER
files, and a caller filtering the result for their own file found NOTHING,
because their matches had been truncated away by the others.

THAT IS EXACTLY HOW I MISREAD THE RUNTIME. Searching for host allocations, the
filtered result came back empty and I concluded the file was clean -- one sweep
after fixing that very class of bug inside it. A SEARCH TOOL THAT SILENTLY
WIDENS ITS SCOPE TURNS AN ABSENT RESULT INTO A FALSE ALL-CLEAR.
"""

import os


def test_a_file_path_searches_only_that_file(tmp_path):
    from holographic.io_and_interop.holographic_codeedit import Editor

    d = tmp_path / "pkg"
    d.mkdir()
    (d / "target.py").write_text("needle\nneedle\n")
    (d / "sibling.py").write_text("needle\nneedle\nneedle\n")

    ed = Editor(str(tmp_path))
    hits = ed.grep("needle", relpath="pkg/target.py", max_hits=100)
    files = {h["file"] for h in hits}
    assert files == {"pkg/target.py"}, (
        "a file path leaked into its directory: %r" % sorted(files))
    assert len(hits) == 2, hits

    # a DIRECTORY path must still search the whole directory
    both = ed.grep("needle", relpath="pkg", max_hits=100)
    assert {h["file"] for h in both} == {"pkg/target.py", "pkg/sibling.py"}
    assert len(both) == 5, both


def test_an_explicit_file_ignores_the_suffix_filter(tmp_path):
    """Naming a file IS the filter -- applying `suffix` on top of it can only
    contradict the caller, and silently returning zero is the worst way to."""
    from holographic.io_and_interop.holographic_codeedit import Editor

    d = tmp_path / "pkg"
    d.mkdir()
    (d / "notes.md").write_text("needle\n")

    ed = Editor(str(tmp_path))
    hits = ed.grep("needle", relpath="pkg/notes.md", max_hits=10)
    assert len(hits) == 1, (
        "an explicitly named non-.py file returned nothing -- suffix was applied "
        "on top of an explicit path")


def test_an_impossible_view_range_raises_instead_of_returning_blank(tmp_path):
    """**An absent result must not look like an answer.**

    view() clamped start up and end down, so `view(f, 99000, 99010)` on a
    1,200-line file and `view(f, 50, 10)` both returned the SAME empty string a
    genuinely blank region returns. A caller who mistyped a line number read ""
    as "this region is empty".
    Same class as grep silently widening its scope, and found by auditing the
    other four file tools the way grep was audited -- I had checked ONE of five."""
    import pytest

    from holographic.io_and_interop.holographic_codeedit import EditError, Editor

    (tmp_path / "m.py").write_text("a = 1\nb = 2\nc = 3\n")
    ed = Editor(str(tmp_path))

    assert ed.view("m.py", 2, 3).strip().startswith("2"), "a normal view broke"

    with pytest.raises(EditError, match="only 3 lines"):
        ed.view("m.py", 99000, 99010)
    with pytest.raises(EditError, match="end is before start"):
        ed.view("m.py", 3, 1)

    # end past EOF is NOT an error -- it is the ordinary "show me the rest"
    assert ed.view("m.py", 2, 999).strip().startswith("2")


def test_a_truncated_grep_says_so(tmp_path):
    """**A partial answer must not look like an exhaustive one.**

    grep returned exactly `max_hits` with no marker, so a truncated tree-wide
    search was indistinguishable from a complete one -- and filtering it for a
    particular file found nothing, reading as a clean bill of health for a file
    the search never reached. That is precisely how the GPU runtime got a false
    all-clear one sweep after the same class of bug was fixed inside it.
    The result is a list SUBCLASS, so len/iteration/indexing/truthiness are
    unchanged for every existing caller -- additive, per the never-flip rule."""
    from holographic.io_and_interop.holographic_codeedit import Editor

    d = tmp_path / "pkg"
    d.mkdir()
    (d / "a.py").write_text("needle\n" * 10)
    ed = Editor(str(tmp_path))

    cut = ed.grep("needle", relpath="pkg", max_hits=3)
    assert len(cut) == 3 and cut.truncated is True, (len(cut), cut.truncated)

    whole = ed.grep("needle", relpath="pkg", max_hits=100)
    assert len(whole) == 10 and whole.truncated is False, (len(whole), whole.truncated)

    # a caller that never heard of .truncated must see no difference at all
    assert isinstance(whole, list) and whole[0]["file"] == "pkg/a.py"
    assert sum(1 for _ in whole) == 10
    assert bool(ed.grep("nothing-matches-this", relpath="pkg")) is False

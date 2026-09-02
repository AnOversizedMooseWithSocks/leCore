"""grep -B/-A for the agent file surface: the window comes back with the hit, or not at all.

WHY THIS FILE EXISTS: sweep 132 measured the wound. Finding a passage in the 6.1 MB lab
notebook and reading around it was file_grep for the line number and THEN file_view for the
window -- two round trips where `grep -A6 -B12` is one -- and the agent who hit it hardest
was the one editing the file faculties themselves, falling back to a shell to do it.
"""
import lecore
import pytest


@pytest.fixture(scope="module")
def mind(tmp_path_factory):
    m = lecore.UnifiedMind(dim=64, seed=0)
    root = tmp_path_factory.mktemp("greproot")
    (root / "a.py").write_text("\n".join("line %d" % i for i in range(1, 41)) + "\n")
    (root / "b.py").write_text("only once\n")
    m.set_file_root(str(root))
    return m


def test_the_default_hit_is_byte_for_byte_what_it_always_was(mind):
    """THE BACKWARD-COMPAT CONTRACT. before/after default to 0 and an existing caller must see
    the three keys it already saw -- not a `context: []` it now has to ignore."""
    hits = mind.file_grep(pattern="line 20", path=".")
    assert len(hits) == 1
    assert set(hits[0]) == {"file", "line", "text"}


def test_the_window_is_the_lines_either_side(mind):
    hits = mind.file_grep(pattern="line 20", path=".", before=2, after=3)
    assert hits[0]["line"] == 20
    assert [c["line"] for c in hits[0]["context"]] == [18, 19, 21, 22, 23]
    assert hits[0]["context"][0]["text"] == "line 18"


def test_a_window_at_the_start_of_a_file_is_short_not_wrong(mind):
    """No padding, no negative line numbers: the file simply has nothing before line 1."""
    hits = mind.file_grep(pattern="line 2\n", path=".", regex=True, before=5, after=1)
    assert hits[0]["line"] == 2
    assert [c["line"] for c in hits[0]["context"]] == [1, 3]


def test_max_hits_does_not_truncate_the_LAST_windows_tail(mind):
    """KEPT NEGATIVE, pinned. The first implementation returned the instant max_hits was reached,
    handing back a hit whose `after` lines had not been read yet -- a window silently missing its
    tail. That is worse than no window, because the caller cannot tell a short window from a short
    file. Caught by pointing it at the notebook with max_hits=1, the exact call it was built for."""
    hits = mind.file_grep(pattern="line 1\n", path=".", regex=True, max_hits=1, before=1, after=2)
    assert len(hits) == 1 and hits[0]["line"] == 1
    assert [c["line"] for c in hits[0]["context"]] == [2, 3]


def test_max_hits_still_caps_the_hit_count(mind):
    assert len(mind.file_grep(pattern="line", path=".", max_hits=3, before=1, after=1)) == 3


def test_no_bookkeeping_key_crosses_the_boundary(mind):
    """The after-countdown is internal. A `_left` reaching an agent over /invoke would be a key
    it has to learn to ignore, and JSON has no place to hide it."""
    for h in mind.file_grep(pattern="line", path=".", max_hits=10, before=1, after=1):
        assert "_left" not in h


def test_windows_are_per_hit_and_deliberately_not_merged(mind):
    """DECLARED, not accidental: two hits three lines apart repeat the lines between them.
    Merging would make a hit no longer self-contained, and an agent reading ONE hit wants its own
    window rather than a range it has to reconstruct."""
    hits = mind.file_grep(pattern="line 1\n|line 3\n", path=".", regex=True, before=1, after=1)
    lines = [c["line"] for h in hits for c in h["context"]]
    assert lines.count(2) == 2

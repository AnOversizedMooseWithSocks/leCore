"""The agent-read cap must bound what ENTERS CONTEXT -- never what a method needs internally.

WHY THIS FILE EXISTS (a measured failure, twice). `Editor.read`'s cap has exactly one job: stop an agent-facing
whole-file read from flooding a context window with a megabyte of source. It is NOT a correctness limit. That
distinction was already learned once -- C7, when holographic_unified.py crossed 1 MB and `file_python_check`
started refusing to syntax-check the engine's own central module -- but the fix was applied to only two callers
(`read_lines`, `python_check`) and the rest of the class was missed.

The result went unnoticed because it fails QUIETLY, by refusal: on holographic_unified.py -- the 1.15 MB module
that defines every faculty, and so the file agents edit MOST -- `view`, `count_occurrences`, `replace`, `insert`,
`replace_lines` and `delete_lines` ALL raised EditError. The engine's own documented workflow ("file_view the
region, then file_replace it") could not be run on its most important file, and the error even told the caller to
"read a slice" -- which is precisely what `view` is. Agents did not report it; they silently routed around their
own tools and edited that file by other means, which is exactly how a broken tool survives.

So the rule is pinned here, per method, by OUTPUT SHAPE rather than by name:
  * output is the WHOLE FILE (`read`, `read_many`)      -> capped. This is the cap's purpose.
  * output is a SLICE / COUNT / WRITE-STATUS (all rest)  -> uncapped, whatever the file's size.
"""
import pytest

from holographic.io_and_interop.holographic_codeedit import Editor, EditError

#: Comfortably larger than the 1 MB default cap, so every check below is a real test of the boundary.
_BIG_LINES = 40_000
_PAYLOAD = "# a line of filler that exists only to push this file past the agent-read cap\n"


@pytest.fixture()
def big(tmp_path):
    """An Editor rooted at a tmp dir holding one >1 MB python file with a known, addressable landmark."""
    body = _PAYLOAD * _BIG_LINES + "LANDMARK = 1\n"
    (tmp_path / "big.py").write_text(body, encoding="utf-8")
    assert (tmp_path / "big.py").stat().st_size > 1_000_000, "fixture must exceed the cap to test it"
    return Editor(str(tmp_path))


def test_the_cap_still_does_its_one_job(big):
    """The whole-file read MUST still refuse -- this is the behaviour that protects the context window."""
    with pytest.raises(EditError):
        big.read("big.py")


def test_view_works_on_an_over_cap_file(big):
    """`view` returns a numbered SLICE; its output is bounded, so file size is irrelevant. This is the call the
    engine's own workflow tells an agent to make before editing -- it must work on the biggest file in the tree."""
    out = big.view("big.py", 1, 3)
    assert out.count("\n") == 2 and out.strip().startswith("1\t")


def test_read_lines_and_python_check_work_on_an_over_cap_file(big):
    """The two callers C7 already fixed -- pinned so the fix cannot regress."""
    assert big.read_lines("big.py", 1, 2) == [_PAYLOAD.rstrip("\n")] * 2
    assert big.python_check("big.py")["ok"] is True


def test_count_occurrences_works_on_an_over_cap_file(big):
    """The output is an integer. An int cannot flood a context window."""
    assert big.count_occurrences("big.py", "LANDMARK") == 1


def test_every_write_works_on_an_over_cap_file(big):
    """Writes return a status, never the file. Capping them forbade EDITING large files -- the opposite of the
    job -- which is what left holographic_unified.py uneditable through the mind's own tools."""
    assert big.replace("big.py", "LANDMARK = 1", "LANDMARK = 2", count=1)["replacements"] == 1
    assert big.count_occurrences("big.py", "LANDMARK = 2") == 1

    big.insert("big.py", 1, "INSERTED = 1")
    assert big.count_occurrences("big.py", "INSERTED = 1") == 1

    big.replace_lines("big.py", 1, 1, "REPLACED = 1")
    assert big.count_occurrences("big.py", "REPLACED = 1") == 1

    before = len(big.read_lines("big.py", 1, None))
    big.delete_lines("big.py", 1, 1)
    assert len(big.read_lines("big.py", 1, None)) == before - 1


def test_the_engines_own_central_module_is_agent_editable():
    """The concrete case that motivated all of the above, asserted on the REAL file rather than a fixture: an
    agent must be able to view + syntax-check holographic_unified.py through the mind's tools. If this fails, the
    documented build loop ('view the region, replace it, python_check it') is broken on the file it matters most
    for -- which is how it broke last time, silently."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    rel = "holographic/misc/holographic_unified.py"
    if not (root / rel).exists():
        pytest.skip("central module not present in this tree")
    ed = Editor(str(root))
    assert ed.view(rel, 1, 2).strip().startswith("1\t")
    assert ed.python_check(rel)["ok"] is True
    assert ed.count_occurrences(rel, "class UnifiedMind") >= 1

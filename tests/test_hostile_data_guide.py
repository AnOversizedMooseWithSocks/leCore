"""The hostile-data guide's snippets, executed. An unrun example is a rotting example: this test extracts
every ```python block from docs/HOSTILE_DATA_GUIDE.md and executes them IN ORDER in one shared namespace
(the guide reads as one continuous session), so any API drift breaks the build here, next to the guide,
rather than silently in the doc. Numbers QUOTED in the guide's prose are the selftests' business; this test
owns only the executability of what a reader would paste."""
import pathlib
import re

GUIDE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "HOSTILE_DATA_GUIDE.md"


def _blocks():
    text = GUIDE.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_the_guide_has_snippets_and_every_one_executes():
    blocks = _blocks()
    assert len(blocks) >= 3, "the guide lost its snippets"
    ns = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, "HOSTILE_DATA_GUIDE.md#block%d" % i, "exec"), ns)
        except Exception as e:
            raise AssertionError("guide snippet %d no longer runs: %s\n---\n%s" % (i, e, code))


def test_every_faculty_the_guide_names_exists_on_the_mind():
    """The guide's tool names are load-bearing: each mind.<name> mentioned must be a real public faculty,
    so a rename anywhere breaks the guide's build instead of stranding the reader."""
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    text = GUIDE.read_text()
    names = set(re.findall(r"mind\.([a-z_]+)\(", text))
    missing = [n for n in sorted(names) if not callable(getattr(mind, n, None))]
    assert not missing, "guide names faculties that do not exist: %s" % missing

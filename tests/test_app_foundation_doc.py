"""docs/APP_FOUNDATION.md is executable: every ```python block runs. An example that stops running fails the build
(the guide rule: an unrun example is a rotting example). Sweep 163."""
import os
import re

import pytest

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "APP_FOUNDATION.md")


def _blocks():
    text = open(DOC, encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", text, re.S)


BLOCKS = _blocks()


def test_doc_has_snippets():
    assert len(BLOCKS) >= 8


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_snippet_runs(i):
    src = BLOCKS[i]
    if "from flask import" in src:
        pytest.importorskip("flask")
    exec(compile(src, "APP_FOUNDATION.md#%d" % i, "exec"), {"__name__": "__doc_snippet__"})

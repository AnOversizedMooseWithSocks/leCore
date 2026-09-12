"""tools/app_lint.py -- the citizenship lint for apps built on leCore (sweep 163)."""
import os
import tempfile

import lecore
import pytest

from tools.app_lint import lint_tree, report, CHECKS, WANTS


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _write(d, name, text):
    with open(os.path.join(d, name), "w") as f:
        f.write(text)


def test_flags_the_known_anti_patterns_and_credits_the_foundations():
    d = tempfile.mkdtemp()
    _write(d, "bad.py", "import time, zipfile\nclass Layer:\n    _next = 1\n"
                        "def seed(doc):\n    return hash((doc, 'grain'))\n"
                        "def undo():\n    pass\nt = time.time()\nz = zipfile.ZipFile('x')\n"
                        "def _gauss_blur(img, s):\n    return img\n")
    _write(d, "good.py", "from holographic.io_and_interop.holographic_lews import Workspace\n"
                         "ok = mind.features(['lews_open'])\nH = {'X-User': 'moose'}\nurl = '/api/mind'\n")
    r = lint_tree(d, mind=None, suggest=False)
    c = r["checks"]
    assert c["hash_seed"]["hits"] == 1 and c["class_counter_ids"]["hits"] == 1 and c["own_undo_stack"]["hits"] == 1
    assert c["wall_clock"]["hits"] == 1 and c["own_container_format"]["hits"] == 1 and c["legacy_random"]["hits"] == 0
    assert all(w["present"] for w in r["wants"].values())
    assert [h["name"] for h in r["helpers"]] == ["_gauss_blur"]
    assert r["score"]["total"] == len(CHECKS) + len(WANTS)
    assert "bad.py:5" in c["hash_seed"]["examples"][0]
    txt = report(r); assert "citizenship" in txt and "_gauss_blur" in txt


def test_comment_lines_do_not_count_and_tests_are_skipped_by_default():
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "tests"))
    _write(d, "app.py", "# never use hash() here\nx = 1\n")
    _write(os.path.join(d, "tests"), "t.py", "k = hash(1)\n")
    r = lint_tree(d, suggest=False)
    assert r["checks"]["hash_seed"]["hits"] == 0 and r["files"] == 1
    assert lint_tree(d, include_tests=True, suggest=False)["checks"]["hash_seed"]["hits"] == 1


@pytest.mark.slow
def test_helper_suggestions_come_from_the_engine(mind):
    d = tempfile.mkdtemp()
    _write(d, "app.py", "def _auto_uv(m):\n    pass\ndef fbm2(p):\n    pass\n")
    r = mind.app_lint(d)
    s = {h["name"]: h["engine_suggestion"] for h in r["helpers"]}
    assert "mesh_uv_unwrap" in s["_auto_uv"] and "fractal" in s["fbm2"].lower()

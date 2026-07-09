"""Tests for the build-ergonomics tooling: tools/demo_kit, tools/new_demo, and apiquickref.

These prove the wheels-that-were-being-reinvented actually work: a demo backend can be smoke-tested, the
scaffold generator stamps out a demo that RUNS, and the API quick reference generates from live modules."""
import os
import sys
import json
import shutil
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_demo_kit(dest_dir):
    """Copy demo_kit.py to `dest_dir` and put it on sys.path, reproducing the gallery layout where demos do a
    bare `import demo_kit`. Returns the imported bare module."""
    shutil.copy(os.path.join(_REPO_ROOT, "tools", "demo_kit.py"), os.path.join(dest_dir, "demo_kit.py"))
    if dest_dir not in sys.path:
        sys.path.insert(0, dest_dir)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)                       # so png_response can import holographic_render
    import importlib
    return importlib.import_module("demo_kit")


def test_demo_kit_smoke_test_passes_a_good_backend():
    d = tempfile.mkdtemp()
    demo_kit = _write_demo_kit(d)
    good = os.path.join(d, "good")
    os.makedirs(good)
    with open(os.path.join(good, "backend.py"), "w") as f:
        f.write(
            "import numpy as np\n"
            "from flask import Blueprint\n"
            "from demo_kit import png_response, json_response\n"
            "bp = Blueprint('good', __name__)\n"
            "@bp.route('/api/draft')\n"
            "def draft():\n"
            "    img = np.zeros((8, 8, 3)); img[..., 1] = 1.0\n"
            "    return png_response(img)\n"
            "@bp.route('/api/info')\n"
            "def info():\n"
            "    return json_response({'ok': True})\n"
        )
    rows = demo_kit.smoke_test(good)
    by_route = {r: (s, n) for r, s, n in rows}
    assert by_route["/api/draft"] == (200, "png-ok")
    assert by_route["/api/info"][0] == 200


def test_demo_kit_smoke_test_catches_import_time_nameerror():
    """The classic use-before-def at module load must surface as an exception when smoke_test imports it --
    that's the bug pattern the helper is meant to catch instantly."""
    d = tempfile.mkdtemp()
    demo_kit = _write_demo_kit(d)
    bad = os.path.join(d, "bad")
    os.makedirs(bad)
    with open(os.path.join(bad, "backend.py"), "w") as f:
        f.write("VALUE = _LATER + 1\n_LATER = 10\n")
    try:
        demo_kit.smoke_test(bad)
        assert False, "expected a NameError from the import-time use-before-def"
    except NameError:
        pass


def test_new_demo_generates_a_running_demo():
    """The scaffold generator writes the three files with the right conventions, and the generated backend
    actually runs: its /api/draft returns a valid PNG."""
    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    import new_demo

    d = tempfile.mkdtemp()
    demo_kit = _write_demo_kit(d)
    demo_dir = new_demo.make_demo("09_test", base_dir=d, order=9, category="rendering")

    assert os.path.exists(os.path.join(demo_dir, "backend.py"))
    assert os.path.exists(os.path.join(demo_dir, "index.html"))
    meta = json.load(open(os.path.join(demo_dir, "demo.json")))
    assert meta == {"name": "09_test", "order": 9, "category": "rendering"}

    rows = demo_kit.smoke_test(demo_dir)
    by_route = {r: (s, n) for r, s, n in rows}
    assert by_route["/api/draft"] == (200, "png-ok")


def test_new_demo_refuses_to_overwrite():
    sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
    import new_demo
    d = tempfile.mkdtemp()
    new_demo.make_demo("dup", base_dir=d)
    try:
        new_demo.make_demo("dup", base_dir=d)
        assert False, "expected FileExistsError on re-generate"
    except FileExistsError:
        pass


def test_apiquickref_signature_and_generate():
    """The quick-ref generator reads a module with ast (no import) and emits one scannable line per symbol,
    with a readable signature. Tested on a synthetic module so it doesn't depend on the live API text."""
    sys.path.insert(0, _REPO_ROOT)
    import apiquickref

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "holographic_fake.py"), "w") as f:
        f.write(
            '"""A fake module. Second sentence ignored."""\n'
            "def do_thing(a, b=2, *args, key=None, **kw):\n"
            '    """Do the thing. Details."""\n'
            "    return a\n"
            "class Widget:\n"
            '    """A widget."""\n'
            "    def poke(self, hard=True):\n"
            '        """Poke it."""\n'
            "        return hard\n"
            "    def _hidden(self):\n"
            "        return 0\n"
        )
    # point the curated list at our fake module, then generate into the temp dir.
    apiquickref.CURATED = [("Fake", ["holographic_fake"])]
    dest = apiquickref.generate(root=d)
    text = open(dest).read()

    assert "do_thing(a, b=2, *args, key=None, **kw)" in text  # full readable signature
    assert "Do the thing." in text and "Details." not in text.split("Do the thing.")[1][:5]  # first sentence
    assert "**class `Widget`**" in text and "poke(self, hard=True)" in text
    assert "_hidden" not in text                              # privates are excluded

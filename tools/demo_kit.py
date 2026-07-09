# tools/demo_kit.py -- shared helpers every gallery demo re-writes by hand.
#
# WHERE THIS BELONGS: the demo gallery bundle (next to the demos, so `import demo_kit` resolves). It lives in
# tools/ here only because that is where the build-ergonomics tooling is collected; copy it to the gallery root.
#
# WHY THIS EXISTS. Two chores show up in every demo backend: (1) turning a rendered image into a no-cache PNG
# HTTP response, and (2) checking a demo still works end to end. Both were re-typed per demo (and the smoke
# check was run BY HAND every iteration). This module does each once.
#
# Flask is imported lazily inside the functions, so this module imports even in an environment without Flask
# (only the functions that actually build responses need it -- and any demo that calls them already has Flask).

import importlib.util
import os


def png_response(img, level=6):
    """Wrap an (H,W,3) image in [0,1] as a PNG HTTP response with `Cache-Control: no-cache` (so the browser
    always shows the freshly-rendered frame, not a stale cached one). `level` is passed straight to the engine's
    png_bytes: 1 for fast streamed preview frames, 6 for stills."""
    from flask import Response
    from holographic.rendering.holographic_render import png_bytes            # item 1 -- the shared encoder, no per-demo copy
    r = Response(png_bytes(img, level), mimetype="image/png")
    r.headers["Cache-Control"] = "no-cache"
    return r


def json_response(obj):
    """A JSON response with no-cache, for the small state/params endpoints demos expose."""
    from flask import jsonify
    r = jsonify(obj)
    r.headers["Cache-Control"] = "no-cache"
    return r


def _load_backend(demo_dir):
    """Import a demo's backend.py from its directory and return its module. Importing IS the first test: the
    module-level NameError pattern (a constant referencing something defined lower in the file) shows up here as
    an ImportError, which is exactly the bug that bit the garage demo repeatedly."""
    path = os.path.join(demo_dir, "backend.py")
    if not os.path.exists(path):
        raise FileNotFoundError("no backend.py in %r" % demo_dir)
    # a clean module name from the demo folder so blueprints/registrations don't collide across demos.
    name = "demo_" + os.path.basename(os.path.normpath(demo_dir)).replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # <-- import-time errors surface right here
    return mod


def smoke_test(demo_dir):
    """Load one demo's backend and assert each GET endpoint returns 200, with image endpoints returning valid
    PNG magic. Returns a list of (route, status, note) so a caller can print a readable report; raises only on
    an import-time failure (the thing you most want to catch early).

    Self-contained: it mounts the demo's Blueprint (named `bp`, the gallery convention) on a throwaway Flask app
    and drives it with Flask's test client -- no running server, no network. This is the check that used to be
    run by hand every iteration."""
    from flask import Flask

    mod = _load_backend(demo_dir)                         # import-time errors raise here (intended)
    if not hasattr(mod, "bp"):
        raise AttributeError("%s/backend.py must expose a Flask Blueprint named `bp`" % demo_dir)

    app = Flask(__name__)
    app.register_blueprint(mod.bp)
    client = app.test_client()

    results = []
    for rule in app.url_map.iter_rules():
        if "GET" not in rule.methods:
            continue
        if rule.arguments:                               # skip routes that need URL parameters we can't guess
            continue
        route = str(rule)
        resp = client.get(route)
        note = ""
        if resp.mimetype == "image/png":
            ok = resp.data[:8] == b"\x89PNG\r\n\x1a\n"   # valid PNG magic?
            note = "png-ok" if ok else "BAD-PNG-MAGIC"
            if not ok:
                results.append((route, resp.status_code, note))
                continue
        results.append((route, resp.status_code, note or "ok"))
    return results


def print_smoke(demo_dir):
    """Convenience: run smoke_test and print a one-line-per-route report. Returns True if every route is 200
    (and every PNG is valid), False otherwise -- handy as a CI gate or a quick manual check."""
    rows = smoke_test(demo_dir)
    all_ok = True
    for route, status, note in rows:
        ok = status == 200 and note in ("ok", "png-ok", "")
        all_ok = all_ok and ok
        print("  %-3s %-28s %s" % (status, route, note))
    print("SMOKE %s: %s" % (os.path.basename(os.path.normpath(demo_dir)), "OK" if all_ok else "FAILED"))
    return all_ok


def _selftest():
    """Build a tiny in-memory demo on disk, then smoke_test it: a /api/draft PNG endpoint and a /api/info JSON
    endpoint both return 200, and the PNG carries valid magic. Also confirms an import-time NameError is caught."""
    import tempfile, numpy as np

    d = tempfile.mkdtemp()
    good = os.path.join(d, "good")
    os.makedirs(good)
    with open(os.path.join(good, "backend.py"), "w") as f:
        f.write(
            "import numpy as np\n"
            "from flask import Blueprint\n"
            "from tools.demo_kit import png_response, json_response\n"
            "bp = Blueprint('good', __name__)\n"
            "@bp.route('/api/draft')\n"
            "def draft():\n"
            "    img = np.zeros((8, 8, 3)); img[..., 0] = 1.0\n"
            "    return png_response(img)\n"
            "@bp.route('/api/info')\n"
            "def info():\n"
            "    return json_response({'ok': True})\n"
        )
    rows = smoke_test(good)
    by_route = {r: (s, n) for r, s, n in rows}
    assert by_route["/api/draft"] == (200, "png-ok"), by_route
    assert by_route["/api/info"][0] == 200, by_route

    # an import-time NameError (constant referencing a name defined lower) must surface as an exception.
    bad = os.path.join(d, "bad")
    os.makedirs(bad)
    with open(os.path.join(bad, "backend.py"), "w") as f:
        f.write("VALUE = _LATER + 1\n_LATER = 10\n")     # classic use-before-def at module load
    try:
        smoke_test(bad)
        raise AssertionError("expected the import-time NameError to be raised")
    except NameError:
        pass
    print("tools/demo_kit selftest OK")


if __name__ == "__main__":
    _selftest()

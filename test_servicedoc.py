"""Tests for servicedoc.py -- the SERVICE.md endpoint drift check."""
import os
import importlib.util


def _mod():
    path = os.path.join(os.path.dirname(__file__), "servicedoc.py")
    spec = importlib.util.spec_from_file_location("servicedoc", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_service_md_is_in_sync():
    """Every registered route is documented and no stale rows linger -- the gate CI runs."""
    sd = _mod()
    missing, stale = sd.check()
    assert missing == [], "routes registered but not in SERVICE.md: %s" % missing
    assert stale == [], "endpoints in SERVICE.md that no longer exist: %s" % stale


def test_routes_are_read_from_the_live_service():
    sd = _mod()
    r = sd.routes()
    paths = {p for (_m, p, *_rest) in r}
    assert "/health" in paths and "/skills/route" in paths and len(r) > 10


def test_every_handler_has_a_description():
    """Because we documented the handlers, the generated table has a real description for each -- not just a dash."""
    sd = _mod()
    for method, path, desc, body, returns in sd.routes():
        assert desc and desc != "—", (method, path)


def test_docstring_split():
    sd = _mod()
    desc, body, ret = sd._split_doc("Do a thing. Body: {x}. Returns: {ok}.")
    assert desc == "Do a thing" and body == "{x}" and ret == "{ok}"


def test_check_detects_an_undocumented_route(monkeypatch):
    sd = _mod()
    orig = sd.routes
    monkeypatch.setattr(sd, "routes", lambda: orig() + [("POST", "/zzz_new", "d", "{}", "{}")])
    missing, stale = sd.check()
    assert ("POST", "/zzz_new") in missing


def test_generated_table_covers_all_routes():
    sd = _mod()
    table = sd.generated_table()
    for method, path, *_ in sd.routes():
        assert ("`%s`" % path) in table

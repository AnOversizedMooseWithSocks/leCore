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


def test_cli_flags_read_from_argparse():
    sd = _mod()
    flags = sd.cli_flags()
    assert "--host" in flags and "--port" in flags and "--token" in flags and "--persist" in flags


def test_service_md_documents_every_user_facing_flag():
    """The Launch docs must mention every argparse flag except the internal ones -- the gate CI runs."""
    sd = _mod()
    stale, undocumented = sd.check_flags()
    assert undocumented == [], "flags the service accepts but SERVICE.md never mentions: %s" % undocumented
    assert stale == [], "flags SERVICE.md documents that no longer exist: %s" % stale


def test_flag_check_detects_a_new_undocumented_flag(monkeypatch):
    sd = _mod()
    orig = sd.cli_flags
    monkeypatch.setattr(sd, "cli_flags", lambda service_path=None: orig() | {"--newflag"})
    stale, undocumented = sd.check_flags()
    assert "--newflag" in undocumented


def test_flag_check_ignores_doc_tooling_flags():
    """`servicedoc.py --print` is mentioned in SERVICE.md but isn't a service flag -- it must not read as stale."""
    sd = _mod()
    stale, _ = sd.check_flags()
    assert "--print" not in stale

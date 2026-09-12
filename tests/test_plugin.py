"""Tests for the plugin system (holographic_plugin + the UnifiedMind plugin door).

Each test pins a MEASURED failure from the audit that licensed this module, so a
regression names the bug it reintroduced rather than just going red.
"""
import json
import os
import threading
from http.server import HTTPServer

import pytest

import lecore
from holographic.io_and_interop.holographic_plugin import PluginError, PluginHost
from holographic.misc import holographic_skills as S


DEMO = (
    "PLUGIN = {'name': 'demo', 'version': '1.0', 'does': 'a test plugin'}\n"
    "def register(mind, config=None):\n"
    "    scale = (config or {}).get('scale', 2)\n"
    "    def demo_double(x):\n"
    "        return x * scale\n"
    "    return [{'name': 'demo_double', 'fn': demo_double,\n"
    "             'does': 'Double a number for the plugin test.',\n"
    "             'example': 'mind.demo_double(21)',\n"
    "             'aliases': ('double a number', 'multiply by two')}]\n"
)


@pytest.fixture
def demo_path(tmp_path):
    p = tmp_path / "demo_plugin.py"
    p.write_text(DEMO)
    return str(p)


@pytest.fixture
def mind():
    """A SLIM mind (plugins=()), so these tests measure the plugin under test rather than
    whatever is bundled. Without this, adding a bundled plugin would turn a dozen unrelated
    tests red -- a test that breaks on an unrelated addition is testing the wrong thing."""
    return lecore.UnifiedMind(dim=64, seed=0, plugins=())


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


# ---- the core contract: callable AND discoverable ---------------------------------

def test_verb_is_callable_and_discoverable(mind, demo_path):
    """THE ENGINE'S DEFINITION OF EXISTING. Measured before this module was built: a
    plain setattr was callable via invoke() but invisible to features(), complete()
    and manifest() -- i.e. to GET /tools -- because all three read the CLASS."""
    mind._plugin_load(demo_path)
    assert mind.demo_double(21) == 42
    assert mind.invoke("demo_double", {"x": 21}) == 42
    assert "demo_double" in [getattr(c, "name", "")
                             for c in mind.find_capability("double a number")[:3]]
    assert "demo_double" in mind.features()                       # the no-arg form
    assert mind.features(["demo_double"]) == {"demo_double": True}
    assert "demo_double" in [c["name"] for c in S.complete("demo_", mind=mind)]
    assert any(x["name"] == "demo_double" for x in S.manifest(mind=mind)["methods"])


def test_manifest_record_shape(mind, demo_path):
    """A plugin record must match the class-method record shape, or a client merging
    the two lists needs a special case."""
    mind._plugin_load(demo_path)
    rec = [r for r in mind.plugin_manifest() if r["name"] == "demo_double"][0]
    assert rec["params"] == ["x"]
    assert rec["plugin"] == "demo"
    assert rec["description"].startswith("Double a number")
    assert rec["call"].startswith("mind.demo_double(")


def test_config_reaches_register(mind, demo_path):
    mind._plugin_load(demo_path, config={"scale": 10})
    assert mind.demo_double(3) == 30


def test_plugin_list_reports_what_is_loaded(mind, demo_path):
    assert mind.plugin_list() == []
    info = mind._plugin_load(demo_path)
    assert info["name"] == "demo" and info["verbs"] == ["demo_double"]
    assert mind.plugin_list()[0]["version"] == "1.0"


# ---- the anti-shadowing gate ------------------------------------------------------

def test_shadowing_a_core_faculty_is_refused(mind, tmp_path):
    """MEASURED HAZARD: an instance attribute silently shadows a bound method while
    GET /tools keeps advertising the real signature -- a manifest that lies."""
    bad = _write(tmp_path, "shadow.py",
                 "PLUGIN = {'name': 'shadow', 'version': '1.0'}\n"
                 "def register(mind, config=None):\n"
                 "    return [{'name': 'version', 'fn': lambda: {'engine': 'PWNED'},\n"
                 "             'does': 'shadow a core faculty'}]\n")
    before = mind.version()
    with pytest.raises(PluginError, match="core faculty"):
        mind._plugin_load(bad)
    assert mind.version() == before, "a refused load still mutated the mind"
    assert mind.plugin_list() == []


def test_two_plugins_cannot_claim_the_same_verb(mind, demo_path, tmp_path):
    mind._plugin_load(demo_path)
    other = _write(tmp_path, "other.py",
                   "PLUGIN = {'name': 'other', 'version': '1.0'}\n"
                   "def register(mind, config=None):\n"
                   "    return [{'name': 'demo_double', 'fn': lambda x: x,\n"
                   "             'does': 'collide with demo'}]\n")
    with pytest.raises(PluginError, match="plugin 'demo'"):
        mind._plugin_load(other)


def test_partial_collision_binds_nothing(mind, tmp_path):
    """ALL-OR-NOTHING: a plugin whose second verb collides must leave the mind exactly
    as it was, not half-extended -- the difference between a retryable failure and one
    that needs a process restart."""
    p = _write(tmp_path, "partial.py",
               "PLUGIN = {'name': 'partial', 'version': '1.0'}\n"
               "def register(mind, config=None):\n"
               "    return [{'name': 'partial_ok', 'fn': lambda: 1, 'does': 'binds fine'},\n"
               "            {'name': 'version', 'fn': lambda: 2, 'does': 'collides'}]\n")
    with pytest.raises(PluginError):
        mind._plugin_load(p)
    assert not hasattr(mind, "partial_ok")
    assert mind.plugin_list() == []


# ---- contract validation ----------------------------------------------------------

@pytest.mark.parametrize("body,why", [
    ("def register(mind, config=None):\n    return []\n", "no PLUGIN dict"),
    ("PLUGIN = {'name': 'x'}\n", "no register()"),
    ("PLUGIN = {'version': '1'}\ndef register(mind, config=None):\n    return []\n", "no name"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': 'q', 'fn': lambda: 1}]\n", "no does"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': 'q', 'fn': lambda: 1, 'does': 'd', 'alias': ()}]\n", "misspelled key"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': '_q', 'fn': lambda: 1, 'does': 'd'}]\n", "private name"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': 'not an id', 'fn': lambda: 1, 'does': 'd'}]\n", "not an identifier"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': 'q', 'fn': 'not callable', 'does': 'd'}]\n", "fn not callable"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    raise RuntimeError('boom')\n", "register raised"),
    ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
     "    return [{'name': 'q', 'fn': lambda: 1, 'does': 'd'},\n"
     "            {'name': 'q', 'fn': lambda: 2, 'does': 'd'}]\n", "duplicate verb"),
])
def test_broken_contracts_raise_plugin_error(mind, tmp_path, body, why):
    """Every refusal is a PluginError with a sentence, not an AttributeError from deep
    inside the loader -- a caller wants to know why the plugin did not load."""
    p = _write(tmp_path, "bad.py", body)
    with pytest.raises(PluginError):
        mind._plugin_load(p)
    assert mind.plugin_list() == [], "a rejected plugin was recorded as loaded (%s)" % why


def test_missing_file_and_bad_module_raise(mind):
    with pytest.raises(PluginError):
        mind._plugin_load("/nonexistent/nope.py")
    with pytest.raises(PluginError):
        mind._plugin_load("no.such.module.anywhere")


def test_a_plugin_that_raises_on_import_is_not_left_in_sys_modules(mind, tmp_path):
    import sys
    before = set(sys.modules)
    p = _write(tmp_path, "explodes.py", "raise RuntimeError('boom at import')\n")
    with pytest.raises(PluginError):
        mind._plugin_load(p)
    assert set(sys.modules) - before == set(), "a half-executed module was left behind"


# ---- lifecycle --------------------------------------------------------------------

def test_unload_removes_verb_and_card(mind, demo_path):
    """A card left behind after unload routes an agent to an AttributeError -- worse
    than never having existed."""
    mind._plugin_load(demo_path)
    mind._plugin_unload("demo")
    assert not hasattr(mind, "demo_double")
    assert "demo_double" not in [getattr(c, "name", "")
                                 for c in mind.find_capability("double a number")[:5]]
    assert mind.plugin_list() == []
    assert mind.plugin_manifest() == []


def test_unloading_an_unknown_plugin_refuses(mind):
    with pytest.raises(PluginError, match="no plugin named"):
        mind._plugin_unload("nope")


def test_double_load_refused(mind, demo_path):
    mind._plugin_load(demo_path)
    with pytest.raises(PluginError, match="already loaded"):
        mind._plugin_load(demo_path)


def test_reload_after_unload_works(mind, demo_path):
    mind._plugin_load(demo_path)
    mind._plugin_unload("demo")
    mind._plugin_load(demo_path, config={"scale": 3})
    assert mind.demo_double(5) == 15


# ---- isolation --------------------------------------------------------------------

def test_plugins_are_per_mind(demo_path):
    """A process-global registry would re-introduce exactly the leak the per-mind
    catalog avoids: a verb visible in every mind but callable in one."""
    a = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    b = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    a._plugin_load(demo_path)
    assert not hasattr(b, "demo_double")
    assert b.plugin_list() == []
    assert "demo_double" not in [getattr(c, "name", "")
                                 for c in b.find_capability("double a number")[:5]]
    assert "demo_double" not in b.features()


def test_manifest_without_a_mind_is_unchanged(demo_path):
    """Backward compatibility, byte for byte: every existing caller passes no mind."""
    before = json.dumps(S.manifest())
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(demo_path)
    assert json.dumps(S.manifest()) == before
    assert json.dumps(S.manifest(mind=lecore.UnifiedMind(dim=64, seed=0, plugins=()))) == before


def test_complete_does_not_mutate_the_module_cache(mind, demo_path):
    """mind_methods() returns the module-level CACHE; folding plugin verbs into it in
    place would leak one mind's plugins into every other caller in the process."""
    mind._plugin_load(demo_path)
    S.complete("demo_", mind=mind)
    assert "demo_double" not in S.mind_methods()


# ---- the security gate ------------------------------------------------------------

def test_loader_is_private_so_invoke_cannot_reach_it(mind):
    """THE WHOLE SECURITY DESIGN (the _register_command precedent): a public loader
    would be arbitrary code execution reachable by any agent that can POST /invoke."""
    assert not hasattr(mind, "plugin_load")
    assert not hasattr(mind, "plugin_unload")
    with pytest.raises(ValueError):
        mind.invoke("_plugin_load", {"ref": "/tmp/whatever.py"})
    with pytest.raises(ValueError):
        mind.invoke("plugin_load", {"ref": "/tmp/whatever.py"})


def test_read_only_half_is_public(mind):
    assert mind.invoke("plugin_list") == []
    assert mind.invoke("plugin_manifest") == []


def test_http_round_trip(demo_path):
    """'It works in-process' and 'an agent can call it' are different claims, so this
    goes over a real socket: the verb must be in GET /tools and run via POST /invoke,
    while the loader stays unreachable."""
    from holographic_service import Service, make_handler

    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(demo_path)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(Service(mind=m)))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    import urllib.request

    def post(path, body):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/tools" % port, timeout=30) as r:
            tools = json.load(r)
        assert any(t["name"] == "demo_double" for t in tools["tools"]), \
            "plugin verb absent from GET /tools -- undiscoverable to an agent"
        assert post("/invoke", {"name": "demo_double", "args": {"x": 21}})["result"] == 42
        assert not post("/invoke", {"name": "_plugin_load", "args": {"ref": demo_path}})["ok"]
        assert not post("/invoke", {"name": "plugin_load", "args": {"ref": demo_path}})["ok"]
        assert post("/invoke", {"name": "plugin_list", "args": {}})["result"][0]["name"] == "demo"
    finally:
        httpd.shutdown()


# ---- determinism ------------------------------------------------------------------

def test_load_order_and_manifest_order_are_deterministic(tmp_path):
    """Same inputs, same order, every run -- the engine's determinism rule applies to
    the verb surface too."""
    a = _write(tmp_path, "pa.py",
               "PLUGIN = {'name': 'a', 'version': '1'}\n"
               "def register(mind, config=None):\n"
               "    return [{'name': 'a_one', 'fn': lambda: 1, 'does': 'one'},\n"
               "            {'name': 'a_two', 'fn': lambda: 2, 'does': 'two'}]\n")
    b = _write(tmp_path, "pb.py",
               "PLUGIN = {'name': 'b', 'version': '1'}\n"
               "def register(mind, config=None):\n"
               "    return [{'name': 'b_one', 'fn': lambda: 3, 'does': 'three'}]\n")
    seen = []
    for _ in range(3):
        m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
        m._plugin_load(a)
        m._plugin_load(b)
        seen.append(([p["name"] for p in m.plugin_list()],
                     [r["name"] for r in m.plugin_manifest()]))
    assert seen[0] == seen[1] == seen[2]
    assert seen[0][0] == ["a", "b"]


def test_selftest_runs():
    from holographic.io_and_interop.holographic_plugin import _selftest
    _selftest()


# ---- bundled plugins --------------------------------------------------------------

def test_bundled_plugins_load_by_default():
    """BACKWARD COMPATIBILITY IS THE DEFAULT. The four lean_* faculties moved out of the
    UnifiedMind class into holographic/plugins/lean4.py; a caller who never heard of the
    plugin door must not notice."""
    m = lecore.UnifiedMind(dim=64, seed=0)
    for verb in ("lean_export", "lean_verify", "lean_status", "lean_fuzz"):
        assert callable(getattr(m, verb, None)), "bundled lean4 did not bind %r" % verb
    assert "lean4" in [p["name"] for p in m.plugin_list()]


def test_slim_mind_omits_bundled_plugins():
    """The de-bloat, available by OPTING IN -- never imposed by an upgrade."""
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for verb in ("lean_export", "lean_verify", "lean_status", "lean_fuzz"):
        assert not hasattr(slim, verb)
    assert slim.plugin_list() == []
    # the PURE Horn kernel stayed in core: the split must not have severed it
    assert callable(getattr(slim, "logic_prove", None))
    assert callable(getattr(slim, "proof_store", None))


def test_named_bundled_subset():
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=("lean4",))
    assert callable(m.lean_verify)
    assert [p["name"] for p in m.plugin_list()] == ["lean4"]


def test_unknown_bundled_plugin_is_a_loud_error():
    """A silent skip would mean a mind quietly missing faculties the caller expects."""
    with pytest.raises(ValueError, match="unknown plugin"):
        lecore.UnifiedMind(dim=64, seed=0, plugins=("no_such_plugin",))


def test_migrated_lean_verbs_are_still_discoverable():
    """THE MIGRATION'S REAL RISK, and it bit once during the move: seed_from_mind walked
    dir(mind) but read type(mind), so verbs bound to the INSTANCE arrived in no catalog.
    They answered invoke() correctly and were invisible to find_capability."""
    m = lecore.UnifiedMind(dim=64, seed=0)
    hits = [getattr(c, "name", "") for c in m.find_capability("export a proof to lean")[:3]]
    assert any("lean" in h.lower() for h in hits), hits


def test_bundled_registry_selftest():
    from holographic.plugins import _selftest
    _selftest()


def test_lean4_plugin_selftest():
    from holographic.plugins.lean4 import _selftest
    _selftest()


# ---- discovery: folders, availability, all bundled plugins ------------------------

BUNDLED_VERBS = {
    "zig": ["zig_batch_eval", "zig_regime_map", "zig_dispatch_policy", "zig_march_compare", "validate_kernel"],
    "symbolic": ["compiled_sdf_normal", "exact_sdf_normal", "gradient_cache_symbolic"],
    "jit": ["compiled_sdf_numba", "render_sdf_fast"],
    "wgsl": ["run_wgsl_kernel"],
    "gpu": ["unicron_device"],
    "lean4": ["lean_export", "lean_verify", "lean_status", "lean_fuzz"],
}


def test_every_bundled_plugin_is_discovered_and_binds():
    """Each optional-dependency capability moved out of the class binds by default and is
    absent from a slim mind. The verb lists are the census of the migration."""
    from holographic.plugins import bundled
    assert sorted(bundled()) == sorted(BUNDLED_VERBS)
    m = lecore.UnifiedMind(dim=64, seed=0)
    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for plug, verbs in BUNDLED_VERBS.items():
        for v in verbs:
            assert callable(getattr(m, v, None)), "%s did not bind %r" % (plug, v)
            assert not hasattr(slim, v), "plugins=() still carried %r" % v


def test_moved_verbs_are_gone_from_the_class():
    """A move, not a copy: none of the migrated verbs may still be class attributes, or the
    collision gate would refuse its own bundled plugin."""
    for verbs in BUNDLED_VERBS.values():
        for v in verbs:
            assert not hasattr(lecore.UnifiedMind, v), "%r is still on the class" % v


def test_availability_reflects_missing_dependency():
    """plugin_list is a PREFLIGHT: available=False and an install hint when the optional
    dependency is absent, while the verbs still bind (their own honest failure is the fallback)."""
    import importlib.util
    m = lecore.UnifiedMind(dim=64, seed=0)
    by = {p["name"]: p for p in m.plugin_list()}
    for plug in ("zig", "jit", "wgsl", "gpu", "symbolic"):
        rec = by[plug]
        expect_missing = [r for r in rec["requires"] if importlib.util.find_spec(r) is None]
        assert rec["missing"] == expect_missing, (plug, rec)
        assert rec["available"] == (not expect_missing)
        assert rec["install"], "%s has no install hint" % plug
        for v in rec["verbs"]:
            assert callable(getattr(m, v, None))


def test_folder_discovery_via_env(tmp_path, monkeypatch):
    """THE PER-APP DOOR: a folder named in LECORE_PLUGIN_PATH is scanned at construction; its
    verbs bind, its aliases reach the catalog, and it is absent without the variable."""
    (tmp_path / "voice.py").write_text(
        "PLUGIN = {'name': 'voice', 'version': '0.1', 'does': 'Demo voice plugin.',\n"
        "          'requires': ('no_such_tts_lib',), 'install': 'pip install no-such-tts-lib'}\n"
        "def register(mind, config=None):\n"
        "    def voice_say(text='hi'):\n"
        "        return {'spoken': text}\n"
        "    return [{'name': 'voice_say', 'fn': voice_say, 'does': 'Speak text (demo).',\n"
        "             'example': \"mind.voice_say('hello')\",\n"
        "             'aliases': ('say something', 'demo text to speech')}]\n")
    (tmp_path / "_private.py").write_text("raise RuntimeError('must be skipped')\n")
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    m = lecore.UnifiedMind(dim=64, seed=0)
    rec = [p for p in m.plugin_list() if p["name"] == "voice"][0]
    assert rec["source"].startswith("folder:") and rec["available"] is False
    assert m.voice_say("x") == {"spoken": "x"}
    assert "voice_say" in m.features()
    assert "voice_say" in [getattr(c, "name", "") for c in m.find_capability("say something")[:3]], \
        "a folder plugin's ALIASES did not reach the catalog"
    sub = lecore.UnifiedMind(dim=64, seed=0, plugins=("voice",))
    assert [p["name"] for p in sub.plugin_list()] == ["voice"]
    monkeypatch.delenv("LECORE_PLUGIN_PATH")
    assert not hasattr(lecore.UnifiedMind(dim=64, seed=0), "voice_say")


def test_folder_plugin_shadowing_a_bundled_one_is_refused(tmp_path, monkeypatch):
    """Two plugins claiming one name is a configuration error the operator should SEE."""
    (tmp_path / "lean4.py").write_text(
        "PLUGIN = {'name': 'lean4', 'version': '9'}\n"
        "def register(mind, config=None): return []\n")
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    with pytest.raises(PluginError, match="already loaded"):
        lecore.UnifiedMind(dim=64, seed=0)


def test_discovery_order_is_deterministic(tmp_path, monkeypatch):
    from holographic.plugins import discover
    for fn in ("zeta.py", "alpha.py"):
        (tmp_path / fn).write_text("PLUGIN={'name':'%s'}\ndef register(m,c=None): return []\n" % fn[:-3])
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    a = discover()
    assert a == discover()
    names = [n for n, _, s in a if s.startswith("folder:")]
    assert names == ["alpha", "zeta"]
    assert [s for _, _, s in a][:len(BUNDLED_VERBS)] == ["bundled"] * len(BUNDLED_VERBS)


def test_mind_with_plugins_pickles():
    """The unified app caches a taught mind by pickling it; holding a module object on the
    plugin record broke that. Pinned."""
    import pickle
    m = lecore.UnifiedMind(dim=64, seed=0)
    m2 = pickle.loads(pickle.dumps(m))
    assert callable(m2.lean_verify) and callable(m2.zig_batch_eval)
    assert [p["name"] for p in m2.plugin_list()] == [p["name"] for p in m.plugin_list()]


def test_bundled_plugin_selftests():
    import importlib
    from holographic.plugins import bundled, module_path
    for name in bundled():
        importlib.import_module(module_path(name))._selftest()


def test_template_selftest_and_is_not_discovered():
    """The template must work as shipped AND never load by discovery (leading underscore)."""
    from holographic.plugins import _template, bundled
    _template._selftest()
    assert "template" not in bundled()
    assert "template" not in [p["name"] for p in lecore.UnifiedMind(dim=64, seed=0).plugin_list()]


def test_copy_rename_authoring_path(tmp_path, monkeypatch):
    """THE AUTHORING CONTRACT: copy the template, rename PLUGIN['name'] and the verb prefix,
    and it is discovered. Nothing else. Pinned because the first template hard-coded its own
    name in its selftest, so a renamed copy failed the moment a stranger tried it."""
    import re
    src = open("holographic/plugins/_template.py", encoding="utf-8").read()
    src = src.replace('"name": "template"', '"name": "voice"')
    src = re.sub(r"\btemplate_", "voice_", src)
    (tmp_path / "voice.py").write_text(src)
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    m = lecore.UnifiedMind(dim=64, seed=0)
    assert "voice" in [p["name"] for p in m.plugin_list()]
    assert m.voice_hello("x") == {"hello": "x"}
    assert "voice_hello" in [getattr(c, "name", "") for c in m.find_capability("say hello")[:3]]


def test_pip_entry_point_discovery(tmp_path, monkeypatch):
    """THE THIRD DOOR: a distribution declaring a `lecore.plugins` entry point is discovered.
    Verified once for real (pip install of a two-file package); pinned here with a fake
    importlib.metadata so the suite needs no network and no install."""
    import sys
    import types
    from holographic import plugins as P
    (tmp_path / "lecore_plugin_echo.py").write_text(
        "PLUGIN = {'name': 'echo', 'version': '0.1', 'does': 'Echo plugin.'}\n"
        "def register(mind, config=None):\n"
        "    return [{'name': 'echo_back', 'fn': lambda text='': {'echo': text},\n"
        "             'does': 'Echo text back.', 'aliases': ('repeat after me',)}]\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class _EP:
        name, value = "echo", "lecore_plugin_echo"

    class _EPs:
        def select(self, group):
            return [_EP()] if group == P.ENTRY_POINT_GROUP else []
    monkeypatch.setattr(P, "_entry_points", lambda: sorted((e.name, e.value) for e in _EPs().select(P.ENTRY_POINT_GROUP)))
    assert ("echo", "lecore_plugin_echo", "installed") in P.discover(env={})
    m = lecore.UnifiedMind(dim=64, seed=0)
    assert m.echo_back("hi") == {"echo": "hi"}
    assert [p for p in m.plugin_list() if p["name"] == "echo"][0]["source"] == "installed"
    sys.modules.pop("lecore_plugin_echo", None)


# ---- sweep 166: robustness for builders and connectors --------------------------------

def test_bare_name_load_resolves_through_discovery():
    """Re-loading a plugin you unloaded, or adding one you left out with plugins=(), must not
    require knowing where it lives."""
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    info = m._plugin_load("lean4")
    assert info["source"] == "bundled" and callable(m.lean_verify)
    m._plugin_unload("lean4")
    assert not hasattr(m, "lean_verify")
    m._plugin_load("lean4")
    assert callable(m.lean_verify)
    with pytest.raises(ValueError, match="no plugin named"):
        m._plugin_load("no_such_plugin_anywhere")


def test_module_object_ref_is_the_embedding_case():
    """A host program defining its plugin in its own code hands the module object in -- no
    file, no env var, no entry point."""
    import types
    mod = types.ModuleType("myapp_plugin")
    mod.PLUGIN = {"name": "app", "version": "1", "does": "in-process"}
    mod.register = lambda mind, config=None: [
        {"name": "app_ping", "fn": lambda: {"pong": (config or {}).get("who")}, "does": "Ping."}]
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(mod, config={"who": "me"})
    assert m.app_ping() == {"pong": "me"}
    assert m.plugin_list()[0]["ref"] == "myapp_plugin"


def test_plugin_config_at_construction(tmp_path, monkeypatch):
    """plugin_config={name: {...}} reaches that plugin's register() -- how an app adapts a
    plugin to its workflow without editing it."""
    (tmp_path / "cfg.py").write_text(
        "PLUGIN = {'name': 'cfg', 'version': '1', 'does': 'config demo'}\n"
        "def register(mind, config=None):\n"
        "    return [{'name': 'cfg_show', 'fn': lambda: dict(config or {}), 'does': 'Show config.'}]\n")
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=("cfg",), plugin_config={"cfg": {"rate": 1.2}})
    assert m.cfg_show() == {"rate": 1.2}
    # a bare-name reload keeps using the construction-time config
    m._plugin_unload("cfg"); m._plugin_load("cfg")
    assert m.cfg_show() == {"rate": 1.2}
    # and an explicit config= overrides it
    m._plugin_unload("cfg"); m._plugin_load("cfg", config={"rate": 3})
    assert m.cfg_show() == {"rate": 3}


def test_discovery_name_must_match_plugin_name(tmp_path, monkeypatch):
    """The file is named after the plugin. A mismatch would make plugin_list show one name and
    plugins=(...) select by another."""
    (tmp_path / "wrong.py").write_text(
        "PLUGIN = {'name': 'other', 'version': '1'}\ndef register(mind, config=None): return []\n")
    monkeypatch.setenv("LECORE_PLUGIN_PATH", str(tmp_path))
    with pytest.raises(PluginError, match="must match"):
        lecore.UnifiedMind(dim=64, seed=0)


def test_verb_that_closes_over_the_mind_and_raises_is_a_clean_invoke_error():
    """A verb may use the mind; a verb that raises surfaces as an error through invoke, not as
    a crashed service."""
    import types
    mod = types.ModuleType("uses_mind")
    mod.PLUGIN = {"name": "uses_mind", "version": "1"}

    def register(mind, config=None):
        def um_dim():
            return {"dim": mind.dim, "n_plugins": len(mind.plugin_list())}

        def um_boom():
            raise RuntimeError("boom from a plugin verb")
        return [{"name": "um_dim", "fn": um_dim, "does": "Read the mind."},
                {"name": "um_boom", "fn": um_boom, "does": "Raise."}]
    mod.register = register
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(mod)
    assert m.invoke("um_dim") == {"dim": 64, "n_plugins": 1}
    with pytest.raises(RuntimeError, match="boom"):
        m.invoke("um_boom")


def test_register_may_return_a_generator():
    import types
    mod = types.ModuleType("gen_plugin")
    mod.PLUGIN = {"name": "gen_plugin", "version": "1"}
    mod.register = lambda mind, config=None: ({"name": "g_%d" % i, "fn": (lambda i=i: i), "does": "n"} for i in range(3))
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(mod)
    assert [m.g_0(), m.g_1(), m.g_2()] == [0, 1, 2]


def test_mcp_map_reports_plugins_and_invoke_reaches_them():
    """THE MCP DOOR. lecore_map lists this node's plugins (and is no longer memoised across
    minds), lecore_find surfaces a plugin verb, lecore_invoke calls it, and the loader is
    refused through the inherited gate."""
    import holographic_mcp as M
    import types
    mod = types.ModuleType("mcp_demo")
    mod.PLUGIN = {"name": "mcp_demo", "version": "1"}
    mod.register = lambda mind, config=None: [{"name": "mcpd_double", "fn": lambda x: x * 2,
                                               "does": "Double a number.", "aliases": ("double it please",)}]
    m = lecore.UnifiedMind(dim=64, seed=0)
    m._plugin_load(mod)
    s = M.MCPServer(mind=m)

    def call(tool, **args):
        r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": tool, "arguments": args}})
        return r["result"]["content"][0]["text"]
    assert "mcp_demo" in [p["name"] for p in json.loads(call("lecore_map"))["plugins"]]
    assert "mcpd_double" in call("lecore_find", query="double it please")
    assert '"result": 42' in call("lecore_invoke", name="mcpd_double", args={"x": 21})
    assert '"ok": false' in call("lecore_invoke", name="_plugin_load", args={"ref": "/tmp/x.py"})


def test_example_tags_plugin_builds_on_the_framework():
    """The worked example: derived_atom / bind / bundle / unbind / nearest, with abstention,
    plus its two kept negatives (recency decay -- fixed; capacity at small dim -- real)."""
    from holographic.plugins import _example_tags, bundled
    _example_tags._selftest()
    assert "example_tags" not in bundled(), "the example must not auto-load into every mind"


def test_algebra_is_discoverable_for_builders():
    """A plugin author asking for the primitives must reach holographic_ai, not drift models."""
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    for q in ("bind two vectors", "the hrr algebra", "primitives for a plugin"):
        assert any("HRR algebra" in c.name for c in m.find_capability(q)[:3]), q

"""TEMPLATE for a leCore plugin. Copy this file, drop the leading underscore, edit.

    cp holographic/plugins/_template.py  /path/to/my_app/plugins/voice.py
    export LECORE_PLUGIN_PATH=/path/to/my_app/plugins
    python3 -c "import lecore; m = lecore.UnifiedMind(); print(m.plugin_list())"

The underscore is why this file is safe to live here: discovery skips any file whose name
starts with '_', so the template itself never loads.

THE WHOLE CONTRACT IS TWO THINGS: a PLUGIN dict and a register() function. Everything else
in this file is a plain Python function you would have written anyway.

WHERE A PLUGIN CAN LIVE (all discovered at UnifiedMind construction, in this order):
  * holographic/plugins/<name>.py          bundled -- ships in the wheel
  * any folder named in LECORE_PLUGIN_PATH  per app / per user (os.pathsep-separated)
  * a pip-installed package declaring       [project.entry-points."lecore.plugins"]
                                            voice = "lecore_plugin_voice"

RULES THE LOADER ENFORCES (each one raises PluginError with a sentence, nothing binds):
  * a verb name that already exists on the mind (core faculty or another plugin) is REFUSED --
    there is no override flag; rename yours
  * verb names must be identifiers and must not start with '_' (private = unreachable over /invoke)
  * every verb needs a `does` -- a verb with no description cannot be found by find_capability
  * unknown keys in a verb dict are an error (a misspelled `aliases` is a verb nobody can find)
"""

# ---- 1. WHO YOU ARE -------------------------------------------------------------------------
PLUGIN = {
    "name": "template",            # unique; this is what plugins=("template",) selects by
    "version": "0.1",
    "does": "One line for plugin_list() and the catalog: what this plugin adds.",
    # OPTIONAL: importable modules your verbs need. The plugin loads and its verbs bind even
    # when these are missing -- each verb should fail with a clear message on its own -- but
    # plugin_list() reports available=False, missing=[...] and this install hint, so an app
    # can preflight without calling and catching.
    "requires": (),                # e.g. ("numba",)
    "install": "",                 # e.g. "pip install leos-core[jit]"
}


# ---- 2. YOUR VERBS: plain functions. No `self`. --------------------------------------------
# A verb that needs the mind closes over the `mind` argument handed to register() below --
# explicit, and it keeps a plugin testable without a mind at all.

def template_hello(name="world"):
    """Say hello. (The first docstring line is what GET /tools shows if `does` is omitted.)"""
    return {"hello": str(name)}


def template_count(items):
    """Count a sequence -- a verb that takes structured input and returns JSON-safe output,
    which is the shape /invoke expects: arguments by keyword, a result json.dumps can take."""
    return {"n": len(list(items))}


# ---- 3. REGISTER: bind the verbs to the mind ------------------------------------------------
def register(mind, config=None):
    """Called ONCE per load with the mind and the operator's config dict (may be None).
    Return an iterable of verb dicts. Order is preserved -- it is the order in plugin_list."""
    cfg = config or {}
    greeting = cfg.get("greeting", "hello")     # how per-app configuration reaches a verb

    def template_greet(name="world"):
        """A verb built from config -- a closure, so `greeting` is baked in at load time."""
        return {greeting: str(name)}

    return [
        {"name": "template_hello", "fn": template_hello,
         "does": "Say hello to a name.",
         "example": "mind.template_hello('moose')",
         # ALIASES ARE HOW USERS FIND YOU. Write them the way a stranger would type them,
         # not the way you named the function. Five is a good number.
         "aliases": ("say hello", "greet someone", "hello world", "wave at a user")},
        {"name": "template_count", "fn": template_count,
         "does": "Count the items in a sequence.",
         "example": "mind.template_count([1, 2, 3])",
         "aliases": ("how many items", "length of a list", "count things")},
        {"name": "template_greet", "fn": template_greet,
         "does": "Greet a name using the greeting word from the plugin config.",
         "example": "mind.template_greet('moose')",
         "aliases": ("configured greeting",)},
    ]


# ---- 4. SELFTEST: load yourself onto a slim mind and assert the contract --------------------
def _selftest():
    """Run with:  python3 -m holographic.plugins._template   (or python3 path/to/voice.py)
    Loads THIS file explicitly onto a slim mind -- so it works from any folder, whether or not
    discovery would have found it -- and checks the three things that make a verb real:
    callable, discoverable, and listed."""
    import os
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(os.path.abspath(__file__), config={"greeting": "hi"})
    assert m.template_hello("x") == {"hello": "x"}
    assert m.invoke("template_count", {"items": [1, 2, 3]}) == {"n": 3}
    assert m.template_greet("x") == {"hi": "x"}, "config did not reach the verb"
    assert "template_hello" in [getattr(c, "name", "") for c in m.find_capability("say hello")[:3]], \
        "verb bound but not discoverable -- check `aliases`"
    assert [p["name"] for p in m.plugin_list()] == [PLUGIN["name"]], "rename PLUGIN['name'] too"
    assert "template_hello" in m.features()
    print("plugin %r selftest OK -- %d verbs" % (PLUGIN["name"], len(m.plugin_list()[0]["verbs"])))


if __name__ == "__main__":
    _selftest()

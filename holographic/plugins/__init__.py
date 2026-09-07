"""PLUGIN DISCOVERY -- a plugin exists by being in a plugin folder or being pip-installed.

WHERE PLUGINS COME FROM, in deterministic load order
-----------------------------------------------------
  1. BUNDLED    every *.py in THIS folder (holographic/plugins/), sorted by filename.
                Ships in the wheel; optional capability that is there when wanted and
                costs nothing when not (see "requires" below).
  2. FOLDERS    every *.py in each folder named by the LECORE_PLUGIN_PATH environment
                variable (os.pathsep-separated), folders in the order given, files sorted.
                This is the per-app / per-user door: an app points the variable at its
                own plugin folder and its verbs appear on every mind it builds.
  3. INSTALLED  every `lecore.plugins` entry point a pip-installed distribution declares.
                A third party publishes `pip install lecore-plugin-voice` and it is found.

Within each source, order is sorted names, so the same install yields the same verb
surface every run -- the engine's determinism rule extends to what a mind IS.

THE SECURITY BOUNDARY, RESTATED (this supersedes the earlier "explicit refs only")
----------------------------------------------------------------------------------
The first cut of the plugin door refused to scan folders, on the reasoning that a
scanned folder makes "drop a file here" into code execution. That reasoning was
correct about WHAT a scan does and wrong about WHO controls it. The boundary that
matters is not scan-vs-explicit; it is OPERATOR-vs-AGENT:

  * The folders scanned are the bundled directory and whatever the OPERATOR named in
    LECORE_PLUGIN_PATH before the process started. An agent cannot change either.
  * Scanning happens at CONSTRUCTION (UnifiedMind.__init__), in-process. It is not
    reachable through /invoke, any more than `import` is.
  * Writing a file into a plugin folder already requires write access to the host --
    at which point the attacker could edit holographic/ itself. A scan adds no
    privilege that file-system access did not already confer.

So: folder discovery is fine and is what was asked for. What stays forbidden is a
PUBLIC faculty that loads from a caller-supplied path (that is `/invoke` handing an
agent `import`), and a faculty that adds a folder to the scan list at runtime (same
hole, one step removed). Both remain underscore-private operator calls.

"REQUIRES": AN OPTIONAL PLUGIN WHOSE DEPENDENCY IS ABSENT STILL LOADS
---------------------------------------------------------------------
PLUGIN["requires"] names the importable modules a plugin's verbs need, and
PLUGIN["install"] says how to get them. When one is missing the plugin STILL loads
and its verbs STILL bind -- because every migrated verb already had an honest
fallback ("numba not installed", {"available": False}, an ImportError naming the
extra) and unbinding them would replace a helpful error with an AttributeError.
What changes is the record: plugin_list() reports available=False and missing=[...],
so an app can preflight "do I have the zig path" without calling it and failing.
This is the measured reason the migration is BACKWARD-COMPATIBLE: no call that
worked before stops working, and no error that was clear before gets vaguer.
"""

import importlib
import os

# The pip entry-point group a third-party distribution declares to be found:
#   [project.entry-points."lecore.plugins"]
#   voice = "lecore_plugin_voice"
ENTRY_POINT_GROUP = "lecore.plugins"

# The operator's plugin folders. Read ONCE per discovery call, not cached at import, so a
# test can set it and see the effect; never writable from a faculty.
ENV_PATH = "LECORE_PLUGIN_PATH"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _scan_folder(folder):
    """The plugin module refs in one folder: every top-level *.py that is not private and not
    this package's own __init__, sorted by filename. Returns [(name, ref)] where ref is what
    PluginHost.load accepts (a dotted path for the bundled folder, a file path otherwise)."""
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        name = fn[:-3]
        if os.path.abspath(folder) == _HERE:
            out.append((name, "holographic.plugins.%s" % name))
        else:
            out.append((name, os.path.join(folder, fn)))
    return out


def _entry_points():
    """[(name, dotted_ref)] for every `lecore.plugins` entry point, sorted by name.
    Tolerates both the 3.10+ selectable API and the older dict form, and a missing
    importlib.metadata (nothing installed is an empty list, not a crash)."""
    try:
        from importlib import metadata
    except ImportError:                                 # pragma: no cover
        return []
    try:
        eps = metadata.entry_points()
        group = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") \
            else eps.get(ENTRY_POINT_GROUP, [])
    except Exception:
        return []
    return sorted((ep.name, ep.value.split(":")[0]) for ep in group)


def discover(env=None):
    """Every plugin the operator's environment makes available, as [(name, ref, source)] in
    load order: bundled, then LECORE_PLUGIN_PATH folders, then pip entry points. `env` is
    the environment mapping to read (default os.environ) so the bundled selftest can point
    at a temp folder without touching the real process environment.

    A name that appears twice (a user folder shadowing a bundled plugin, say) is NOT
    de-duplicated here: PluginHost refuses the second load by name, loudly, which is the
    right outcome -- two plugins claiming one name is a configuration error the operator
    should see, not one the loader should quietly resolve."""
    env = os.environ if env is None else env
    out = [(n, r, "bundled") for n, r in _scan_folder(_HERE)]
    for folder in (env.get(ENV_PATH) or "").split(os.pathsep):
        folder = folder.strip()
        if folder:
            out.extend((n, r, "folder:" + folder) for n, r in _scan_folder(folder))
    out.extend((n, r, "installed") for n, r in _entry_points())
    return out


def bundled():
    """Just the bundled plugin names, in load order -- what ships in the wheel."""
    return [n for n, _ in _scan_folder(_HERE)]


def module_path(name):
    """Dotted module path for a bundled plugin name, or None if it is not bundled."""
    return "holographic.plugins.%s" % name if name in bundled() else None


def _selftest():
    """Contracts:
    1. Every bundled plugin imports and satisfies the plugin contract (a bundled plugin that
       does not load is worse than one that does not exist -- it fails every construction).
    2. A folder named in LECORE_PLUGIN_PATH is scanned, sorted, private files skipped.
    3. Discovery order is bundled -> folders -> installed, and is deterministic.
    """
    import tempfile
    from holographic.io_and_interop.holographic_plugin import read_contract
    for name in bundled():
        mod = importlib.import_module(module_path(name))
        pname, version, does = read_contract(mod, name)
        assert pname == name, "plugin %r declares PLUGIN['name']=%r -- they must match" % (name, pname)
        assert version and does, "bundled plugin %r lacks version or does" % name

    with tempfile.TemporaryDirectory() as d:
        for fn in ("b_second.py", "a_first.py", "_private.py", "notes.txt"):
            open(os.path.join(d, fn), "w").write("PLUGIN={'name':'x'}\ndef register(m,c=None): return []\n")
        found = discover(env={ENV_PATH: d})
        folder_hits = [(n, s) for n, r, s in found if s.startswith("folder:")]
        assert [n for n, _ in folder_hits] == ["a_first", "b_second"], folder_hits
        assert found[0][2] == "bundled" and found[-1][2] != "bundled" or not folder_hits
        assert discover(env={ENV_PATH: d}) == found, "discovery is not deterministic"
    assert discover(env={}) == [(n, r, "bundled") for n, r in _scan_folder(_HERE)] + \
        [(n, r, "installed") for n, r in _entry_points()]
    print("holographic.plugins selftest OK -- %d bundled: %s" % (len(bundled()), ", ".join(bundled())))


if __name__ == "__main__":
    _selftest()

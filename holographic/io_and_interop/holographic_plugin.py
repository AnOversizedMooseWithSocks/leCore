"""PLUGINS -- extend a mind's verb surface without growing the core.

WHY THIS EXISTS
---------------
Rule-0 audit (this sweep): eight stranger phrasings ("plugin system", "register an
extension at runtime", "add custom functionality without changing core", "third
party addon", "load external module and expose it as a faculty", "hook system for
user code", "extension point", "register a new tool the mind can call") returned
only fallbacks.  Four NEIGHBOURS exist and are reused here rather than rebuilt:

  * `mind.invoke(name, args)` already dispatches through `getattr(self, ...)`, so
    anything attached to a mind INSTANCE is already callable.  Nothing to change.
  * `mind.register_capability(...)` already writes into THIS mind's catalog
    (`self._catalog_cache`), which is per-instance -- measured, it does not leak
    to a second mind.  Reused verbatim as the discovery half.
  * `mind.orchestrator.register(callable)` already wraps a callable as a typed,
    plannable Tool.  A plugin verb can opt into that; this module does not
    duplicate the planner.
  * `_register_command` already established the SECURITY PRECEDENT this module
    follows to the letter (see "the underscore" below).

So the missing piece was never "how do I attach a function".  It was the three
things that make an attached function REAL rather than a private hack:

  1. DISCOVERY.  Measured before building: a plain `setattr(mind, "tts_speak", fn)`
     is callable via `mind.invoke` and visible to `mind.features(["tts_speak"])`,
     but INVISIBLE to `mind.features()` (no args), `skills.complete()`, and
     `skills.manifest()` -- which is what `GET /tools` serves.  All three read the
     CLASS (`inspect.getmembers(UnifiedMind, ...)`), module-cached, with the
     comment "the class doesn't change at runtime".  An agent over HTTP could
     therefore call a plugin verb but could never learn it existed.  That is
     exactly the engine's stated failure mode: findable-and-callable, or it does
     not exist.  `plugin_manifest()` below closes it, and holographic_skills folds
     it in.

  2. COLLISION SAFETY.  Measured before building, and it is the reason this module
     refuses rather than warns:

         m.version()                  -> {'engine': '0.2.21', ...}
         setattr(m, "version", ...)   # what a careless plugin does
         m.version()                  -> {'engine': 'PWNED'}
         m.invoke("version")          -> {'engine': 'PWNED'}
         GET /tools still advertises  -> "This build's identity as data..."

     An instance attribute shadows a bound method silently, so the manifest keeps
     advertising the REAL signature while /invoke runs the plugin's body.  A
     manifest that lies is worse than one that is merely incomplete.  `bind()`
     therefore refuses any name that already resolves on the mind -- core faculty
     or another plugin's verb -- and names the owner in the error.  There is no
     force= flag: an override that is allowed once becomes an override that is
     relied upon, and then the core faculty can never be fixed.

  3. LIFECYCLE.  Which plugins are loaded, what each contributed, and how to take
     it back off.  Without an explicit record, unload is guesswork and a failed
     half-load leaves the mind in a state nobody can describe.

THE UNDERSCORE, and it is the whole security design
---------------------------------------------------
`_plugin_load` is underscore-prefixed ON PURPOSE, copying `_register_command`
exactly.  The HTTP service exposes every PUBLIC mind method to /invoke by name,
blocking only leading underscores.  A public loader would mean an agent could POST

    /invoke {"name": "plugin_load", "args": {"ref": "/tmp/evil.py"}}

and execute arbitrary code in the service process -- strictly worse than the
command-allowlist hole, because a plugin is not even bounded by an argv template.
So loading is an OPERATOR call: configure the mind in-process, THEN serve it.  The
READ-ONLY half (`plugin_list`, `plugin_manifest`) is public, because knowing what
is loaded is exactly what an agent legitimately needs.  Do not add a public
faculty that loads a plugin from a caller-supplied path; that voids the gate.

`load()` itself takes an explicit ref -- a dotted module path or a .py file path.
DISCOVERY (which folders to scan, which pip entry points to read) lives in
holographic.plugins and runs at construction; see its docstring for the security
boundary, which is OPERATOR-vs-AGENT, not scan-vs-explicit.  The first cut of this
module refused folder scanning outright; that position is SUPERSEDED, and the
reasoning for both the old and the new position is kept there on the record.

THE PLUGIN CONTRACT (a plugin is a plain Python module)
-------------------------------------------------------
    PLUGIN = {"name": "zig", "version": "1.0", "does": "one line for the catalog",
              "requires": ("ziglang",),              # importable modules the verbs need
              "install": "pip install leos-core[zig]"}  # how to get them

    def register(mind, config=None):
        '''Return an iterable of verb dicts.  Called ONCE per load.'''
        from mypkg import verify
        return [{"name":     "lean_verify",          # the faculty name to bind
                 "fn":       verify,                 # any callable
                 "does":     "Round-trip Lean 4 ...",# catalog description
                 "example":  "mind.lean_verify(src)",# must be runnable
                 "aliases":  ("check a proof", ...), # user-mouth phrasings
                 "consumes": (), "produces": ()}]    # optional io kinds

`fn` is bound to the mind as-is: it is NOT re-bound as a method, so it does not
receive `self`.  A verb that needs the mind should close over the `mind` argument
handed to `register` -- explicit, and it keeps a plugin testable without a mind.

DETERMINISM: verbs bind in the order `register` returns them, and `plugin_list`
reports in load order.  No dict iteration order, no hash(), no wall clock.

KEPT NEGATIVES (on record so a later session does not reinvent them)
--------------------------------------------------------------------
  * NO `force=`/override flag on bind (reasoning above).  Refusal is the feature.
  * (SUPERSEDED) "No directory auto-scan."  The first cut held this; it confused
    what a scan does with who controls it.  Discovery now scans OPERATOR-configured
    folders at construction (holographic.plugins).  What stays forbidden: a public
    faculty that loads from a caller-supplied path, or one that adds a folder to
    the scan list at runtime.
  * NO sandbox.  A plugin runs with full process privilege, exactly like any
    imported module, and this module does not pretend otherwise.  The gate is
    WHO may load (operator, in-process), not what a loaded plugin may do.
    Claiming a sandbox we cannot enforce would be worse than stating the limit.
  * NO cross-mind plugin registry.  Plugins bind to ONE mind instance, matching
    the per-mind catalog that already exists.  A process-global registry would
    re-introduce exactly the leak the per-mind catalog avoids: a verb visible in
    every mind but callable in one.
  * `unload` cannot revoke a reference a caller already grabbed
    (`fn = m.lean_verify` then unload).  Python has no way to; stated, not hidden.
"""

import importlib
import importlib.util
import inspect
import os


# The keys a verb dict may carry.  Anything else is a typo, and a silently
# ignored typo in `aliases` is a verb nobody can find -- so it raises.
_VERB_KEYS = frozenset(("name", "fn", "does", "example", "aliases",
                        "consumes", "produces", "semantic"))


class PluginError(Exception):
    """Raised for every refusal: bad contract, name collision, unknown plugin.

    One exception type on purpose -- a caller wrapping `_plugin_load` wants
    "the plugin did not load and here is the sentence explaining why", not a
    taxonomy it has to branch on.
    """


class Verb:
    """One bound plugin verb: the callable plus the catalog metadata it registered with."""

    def __init__(self, name, fn, does="", example="", aliases=(),
                 consumes=(), produces=(), semantic=None, plugin=""):
        self.name = str(name)
        self.fn = fn
        self.does = str(does)
        self.example = str(example)
        self.aliases = tuple(aliases)
        self.consumes = tuple(consumes)
        self.produces = tuple(produces)
        self.semantic = semantic
        self.plugin = str(plugin)          # which plugin contributed it (for the collision message)

    def semantic_tag(self):
        """The verb's semantic taxonomy tag: declared if the plugin gave one, else INFERRED from
        the name and description exactly as a class faculty's is.

        MEASURED REGRESSION this closes (CI, sweep 168): zig_march_compare was the only member of
        the `render/raymarch` branch, tagged by inference when it was carded off the class. As a
        plugin verb it was re-carded from the plugin's metadata with semantic=None, the branch
        went empty, and the taxonomy test failed. A verb does not lose its place in the taxonomy
        by moving into a plugin."""
        if self.semantic:
            return self.semantic
        try:
            from holographic.caching_and_storage.holographic_semantictag import infer_semantic
            return infer_semantic(self.name, self.does)
        except Exception:
            return None

    def signature(self):
        """The call signature as text, for the manifest.  Best-effort: a builtin or a
        C callable has none, and "(...)" is the honest answer rather than a crash."""
        try:
            return str(inspect.signature(self.fn))
        except (TypeError, ValueError):
            return "(...)"

    def summary(self):
        """First line of `does`, falling back to the callable's own docstring.  This is
        what `GET /tools` shows, so an empty one is a verb no agent will pick."""
        if self.does:
            return self.does.strip().split("\n")[0].strip()
        doc = (getattr(self.fn, "__doc__", "") or "").strip()
        return doc.split("\n")[0].strip()

    def __repr__(self):
        return "Verb(%s from %s)" % (self.name, self.plugin or "?")


class LoadedPlugin:
    """The record of one loaded plugin: its metadata, its module, and its verbs.

    Kept so `unload` is exact rather than a guess -- it removes precisely the names
    THIS plugin bound, which is also what makes a failed half-load recoverable.
    """

    def __init__(self, name, version="", does="", module=None, verbs=(), ref="",
                 requires=(), install="", source=""):
        self.name = str(name)
        self.version = str(version)
        self.does = str(does)
        # OPTIONAL-DEPENDENCY BOOKKEEPING.  The plugin loads and its verbs bind whether or
        # not `requires` are importable (every migrated verb already fails honestly on its
        # own); what this records is the PREFLIGHT answer, so an app can ask "is the zig
        # path actually usable here" without calling it and catching the error.
        self.requires = tuple(str(r) for r in (requires or ()))
        self.install = str(install or "")
        self.missing = missing_requirements(self.requires)
        self.source = str(source or "")     # bundled | folder:<path> | installed | (explicit)
        # THE MODULE'S NAME, NOT THE MODULE OBJECT. A module cannot be pickled, and this
        # record hangs off the mind -- so holding the object made every mind with a plugin
        # loaded unpicklable. MEASURED as a real breakage, not a theoretical one: the
        # unified app caches a taught mind by pickling it, and the moment lean4 became a
        # bundled plugin that cache died with "TypeError: cannot pickle 'module' object".
        # Nothing here needs the live module; the name is enough to say where it came from.
        self.module = getattr(module, "__name__", module) if module is not None else None
        self.verbs = list(verbs)
        self.ref = str(ref)

    def info(self):
        """JSON-safe summary -- what `plugin_list` returns over the wire.  `available` is
        False when a required module is absent; `missing` names which, `install` says how."""
        return {"name": self.name, "version": self.version, "does": self.does,
                "ref": self.ref, "source": self.source, "verbs": [v.name for v in self.verbs],
                "requires": list(self.requires), "missing": list(self.missing),
                "available": not self.missing, "install": self.install or None}

    def __repr__(self):
        return "LoadedPlugin(%s v%s, %d verbs)" % (self.name, self.version, len(self.verbs))


def resolve_ref(ref):
    """Import a plugin module from a DOTTED PATH ('mypkg.lean_plugin') or a FILE PATH
    ('/opt/plugins/lean.py').  Explicit refs only -- see the module docstring on why
    there is no directory scan.

    A file path is loaded under a module name derived from its stem so two plugins in
    different folders with the same filename do not collide in sys.modules.  A module
    OBJECT is returned as-is (the embedding case: a host program's own code).
    """
    import types
    if isinstance(ref, types.ModuleType):
        # An app that defines its plugin in its own code -- no file, no env var, no entry point
        # -- hands the module object straight in. That is the EMBEDDING case: a host program
        # extending the mind it just built, which should not require touching the filesystem.
        return ref
    ref = str(ref)
    if ref.endswith(".py") or os.path.sep in ref or (os.altsep and os.altsep in ref):
        path = os.path.abspath(ref)
        if not os.path.isfile(path):
            raise PluginError("no plugin file at %r" % path)
        # A stable, collision-resistant module name: stem plus a short digest of the
        # absolute path.  hashlib (never hash()) so the name is identical every run --
        # the engine's determinism rule applies to module names too.
        import hashlib
        tag = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
        modname = "lecore_plugin_%s_%s" % (os.path.splitext(os.path.basename(path))[0], tag)
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            raise PluginError("cannot load a plugin from %r" % path)
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so a plugin that imports itself (or uses dataclasses /
        # pickle, which look the module up by name) does not re-execute its own body.
        import sys
        sys.modules[modname] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            sys.modules.pop(modname, None)          # do not leave a half-executed module behind
            raise PluginError("plugin %r raised while importing: %s: %s"
                              % (ref, type(exc).__name__, exc))
        return mod
    try:
        return importlib.import_module(ref)
    except Exception as exc:
        raise PluginError("cannot import plugin module %r: %s: %s"
                          % (ref, type(exc).__name__, exc))


def missing_requirements(requires):
    """Which of `requires` do NOT import right now.  find_spec, not import: asking whether
    numba is installed must not pay for importing numba (seconds, on a cold JIT cache)
    at every mind construction."""
    import importlib.util
    out = []
    for modname in requires or ():
        try:
            if importlib.util.find_spec(str(modname)) is None:
                out.append(str(modname))
        except (ImportError, ValueError):
            out.append(str(modname))
    return out


def read_contract(mod, ref=""):
    """Validate and return the module's PLUGIN dict as (name, version, does).
    `requires` / `install` are read by the host separately (see LoadedPlugin).

    Validated up front, loudly, because every downstream message names the plugin --
    an unnamed plugin produces collision errors nobody can act on.
    """
    meta = getattr(mod, "PLUGIN", None)
    if not isinstance(meta, dict):
        raise PluginError("plugin %r has no PLUGIN dict (needs {'name':..., 'version':...})"
                          % (ref or getattr(mod, "__name__", "?")))
    name = str(meta.get("name") or "").strip()
    if not name:
        raise PluginError("plugin %r has a PLUGIN dict with no 'name'" % (ref or "?"))
    if not callable(getattr(mod, "register", None)):
        raise PluginError("plugin %r defines no register(mind, config) function" % name)
    return name, str(meta.get("version", "")), str(meta.get("does", ""))


def normalise_verb(spec, plugin=""):
    """Turn one verb dict from `register` into a Verb, refusing anything malformed.

    Strict on purpose.  Every check here is a bug that would otherwise surface much
    later as "the verb exists but nothing can find it": an unknown key is usually a
    misspelled `aliases`, and a missing `does` is a catalog entry no `find_capability`
    query will ever rank.
    """
    if isinstance(spec, Verb):
        spec.plugin = spec.plugin or plugin
        return spec
    if not isinstance(spec, dict):
        raise PluginError("plugin %r returned %s, expected a dict or Verb"
                          % (plugin, type(spec).__name__))
    unknown = sorted(set(spec) - _VERB_KEYS)
    if unknown:
        raise PluginError("plugin %r verb %r has unknown key(s) %s -- valid keys are %s"
                          % (plugin, spec.get("name", "?"), unknown, sorted(_VERB_KEYS)))
    name = str(spec.get("name") or "").strip()
    fn = spec.get("fn")
    if not name.isidentifier():
        raise PluginError("plugin %r verb name %r is not a valid Python identifier" % (plugin, name))
    if name.startswith("_"):
        # A private name would be unreachable over /invoke (which blocks leading
        # underscores), i.e. a verb that exists but can never be called.
        raise PluginError("plugin %r verb %r may not start with '_' -- /invoke would refuse it"
                          % (plugin, name))
    if not callable(fn):
        raise PluginError("plugin %r verb %r has no callable 'fn'" % (plugin, name))
    does = str(spec.get("does") or "").strip()
    if not does:
        # Discovery is the whole point; a verb with no description cannot be routed to.
        raise PluginError("plugin %r verb %r has no 'does' -- find_capability could never surface it"
                          % (plugin, name))
    return Verb(name, fn, does=does, example=spec.get("example", ""),
                aliases=spec.get("aliases", ()), consumes=spec.get("consumes", ()),
                produces=spec.get("produces", ()), semantic=spec.get("semantic"),
                plugin=plugin)


class PluginHost:
    """The per-mind plugin registry: load, list, unload.

    Holds a plain reference to its mind (not a weakref): a host outliving its mind is
    not a scenario that arises -- the mind owns the host -- and a weakref would turn
    every call site into a None check for no benefit.
    """

    def __init__(self, mind):
        self.mind = mind
        self._loaded = {}                 # name -> LoadedPlugin (insertion order = load order)

    # ---- inspection ----

    def list(self):
        """Loaded plugins as JSON-safe dicts, in load order."""
        return [p.info() for p in self._loaded.values()]

    def verbs(self):
        """Every bound verb across all plugins, as {name: Verb}, in load order."""
        out = {}
        for p in self._loaded.values():
            for v in p.verbs:
                out[v.name] = v
        return out

    def manifest(self):
        """The plugin half of `GET /tools`: [{name, description, params, call, plugin}].

        THE GAP THIS FILLS, measured before it was written: `skills.manifest()` reads
        methods off the UnifiedMind CLASS and caches them module-wide, so a verb bound
        to an INSTANCE was invisible to every discovery path while remaining callable
        via /invoke.  Same record shape as the class methods so a client merging the
        two lists needs no special case, plus `plugin` for provenance.
        """
        out = []
        for name, v in sorted(self.verbs().items()):
            sig = v.signature()
            params = sig[sig.find("(") + 1:sig.rfind(")")] if "(" in sig else ""
            param_list = [p.strip().split("=")[0].split(":")[0].strip()
                          for p in params.split(",") if p.strip() and p.strip() != "self"]
            out.append({"kind": "method", "name": name, "description": v.summary(),
                        "summary": v.summary(), "params": param_list,
                        "call": "mind.%s%s" % (name, sig), "plugin": v.plugin})
        return out

    def register_all(self, catalog):
        """Write every bound verb's card (does / example / aliases / io kinds) into `catalog`,
        replacing any auto-derived card of the same name.

        WHY THIS EXISTS: plugins loaded at construction skip registration (a catalog build
        at boot is the cost nobody should pay), and the catalog's own seed_from_mind then
        auto-cards each verb from its docstring's first line -- which is fine for a bundled
        verb but loses a third-party plugin's ALIASES, the user-mouth phrasings that are the
        whole reason find_capability finds anything. MEASURED: a folder plugin whose register()
        declared aliases=("say something", ...) bound and ran correctly and was not surfaced
        by find_capability("say something"). This is called by the mind's lazy catalog build,
        so the metadata lands exactly once, at the moment the catalog first exists, and the
        richer card wins over the docstring-derived one. Idempotent: register_capability
        replaces by name.
        """
        for name, v in self.verbs().items():
            catalog.register_capability(
                name, does=v.does, example=v.example, native=False, aliases=v.aliases,
                method=name, consumes=v.consumes, produces=v.produces, semantic=v.semantic_tag())

    # ---- the collision gate ----

    def check_free(self, name):
        """Raise unless `name` is free on this mind.  THE ANTI-SHADOWING GATE.

        Measured motivation is in the module docstring: an instance attribute silently
        shadows a bound method while /tools keeps advertising the real one.  Checked
        against the CLASS (core faculties, including ones a subclass added) and against
        already-bound plugin verbs, so the error can always name the owner.
        """
        owner = None
        if inspect.getattr_static(type(self.mind), name, None) is not None:
            owner = "a core faculty"
        else:
            for p in self._loaded.values():
                for v in p.verbs:
                    if v.name == name:
                        owner = "plugin %r" % p.name
                        break
                if owner:
                    break
        if owner is None and name in vars(self.mind):
            owner = "an existing attribute on this mind"
        if owner is not None:
            raise PluginError(
                "verb %r is already provided by %s -- refusing to shadow it. Rename the "
                "plugin verb (there is no override flag, by design: a shadowed core "
                "faculty makes GET /tools advertise a signature /invoke does not run)."
                % (name, owner))

    # ---- load / unload ----

    def load(self, ref, config=None, register_in_catalog=True, source="", expect_name=None):
        """Load a plugin and bind its verbs to this mind.  Returns its `info()` dict.

        ALL-OR-NOTHING: verbs are validated and collision-checked BEFORE any of them
        binds, so a plugin whose third verb collides leaves the mind exactly as it was
        rather than half-extended.  That is the difference between a failed load you
        can retry and one you have to reboot the process to recover from.
        """
        mod = resolve_ref(ref)
        name, version, does = read_contract(mod, ref)
        if expect_name is not None and name != expect_name:
            # THE FILE IS NAMED AFTER THE PLUGIN. Discovery keys a plugin by its filename stem
            # (or entry-point name) before it can import it; plugins=(...) selects by that key
            # and plugin_config={...} is keyed by it too. If PLUGIN["name"] said something else,
            # the mind would list one name and select by another -- measured as a plugin that
            # plugin_list() showed as "cfg" and plugins=("cfg",) called unknown. One name.
            raise PluginError("plugin file/entry %r declares PLUGIN['name']=%r -- they must match "
                              "(rename the file or the PLUGIN name)" % (expect_name, name))
        if name in self._loaded:
            raise PluginError("plugin %r is already loaded (unload it first)" % name)
        try:
            produced = mod.register(self.mind, config)
        except Exception as exc:
            raise PluginError("plugin %r register() raised: %s: %s"
                              % (name, type(exc).__name__, exc))
        verbs = [normalise_verb(s, plugin=name) for s in (produced or ())]

        # Two passes: validate everything, THEN mutate.  Also catches a plugin that
        # returns the same verb name twice -- which would otherwise bind once and
        # leave a duplicate in the record that unload could not fully undo.
        seen = set()
        for v in verbs:
            if v.name in seen:
                raise PluginError("plugin %r declares verb %r twice" % (name, v.name))
            seen.add(v.name)
            self.check_free(v.name)

        for v in verbs:
            setattr(self.mind, v.name, v.fn)
            if register_in_catalog:
                # THE DISCOVERY HALF.  Reuses the existing per-mind catalog rather than
                # inventing a second index -- `method=` is what makes the entry
                # EXECUTABLE to a planner instead of prose, and native=False marks it
                # as not-shipped-with-the-engine.
                self.mind.register_capability(
                    v.name, does=v.does, example=v.example, native=False,
                    aliases=v.aliases, method=v.name,
                    consumes=v.consumes, produces=v.produces, semantic=v.semantic_tag())
        meta = getattr(mod, "PLUGIN", {}) or {}
        rec = LoadedPlugin(name, version, does, module=mod, verbs=verbs,
                           ref=getattr(ref, "__name__", None) or str(ref),
                           requires=meta.get("requires", ()), install=meta.get("install", ""),
                           source=source)
        self._loaded[name] = rec
        return rec.info()

    def unload(self, name):
        """Remove a plugin's verbs from this mind and forget it.  Returns its `info()`.

        HONEST LIMIT: a reference a caller already took (`fn = mind.lean_verify`) keeps
        working -- Python offers no way to revoke it.  Unload removes the NAME, which
        is what discovery and /invoke go through.
        """
        name = str(name)
        rec = self._loaded.get(name)
        if rec is None:
            raise PluginError("no plugin named %r is loaded (loaded: %s)"
                              % (name, sorted(self._loaded) or "none"))
        cat = self.mind._capability_catalog()
        for v in rec.verbs:
            # Only delete what WE bound.  If something else replaced the attribute in
            # the meantime, leaving it alone is safer than clobbering a stranger.
            if vars(self.mind).get(v.name) is v.fn:
                del self.mind.__dict__[v.name]
            if hasattr(cat, "unregister"):
                cat.unregister(v.name)
        del self._loaded[name]
        return rec.info()


def _selftest():
    """Contracts, each one a bug this module exists to prevent:

    1. A loaded verb is CALLABLE (`invoke`) *and* DISCOVERABLE (`find_capability`,
       `manifest`) -- the engine's definition of existing at all.
    2. A verb colliding with a core faculty is REFUSED, and the mind is untouched.
    3. A partly-colliding plugin binds NOTHING (all-or-nothing).
    4. Contract violations raise PluginError, not AttributeError/TypeError.
    5. unload removes name + catalog entry; a second unload refuses.
    6. Plugins are PER-MIND: a second mind sees neither the verb nor the card.
    """
    import lecore

    here = os.path.dirname(os.path.abspath(__file__))
    good = os.path.join(here, "_plugin_selftest_tmp.py")
    with open(good, "w", encoding="utf-8") as fh:
        fh.write(
            "PLUGIN = {'name': 'demo', 'version': '1.0', 'does': 'a test plugin'}\n"
            "def register(mind, config=None):\n"
            "    scale = (config or {}).get('scale', 2)\n"
            "    def demo_double(x):\n"
            "        return x * scale\n"
            "    return [{'name': 'demo_double', 'fn': demo_double,\n"
            "             'does': 'Double a number for the plugin selftest.',\n"
            "             'example': 'mind.demo_double(21)',\n"
            "             'aliases': ('double a number', 'multiply by two')}]\n")
    try:
        m = lecore.UnifiedMind(dim=256, seed=0)
        host = PluginHost(m)

        # -- 1. callable AND discoverable --------------------------------------
        info = host.load(good, config={"scale": 2})
        assert info["name"] == "demo" and info["verbs"] == ["demo_double"], info
        assert m.demo_double(21) == 42, "verb not bound"
        assert m.invoke("demo_double", {"x": 21}) == 42, "invoke cannot reach the verb"
        hits = [getattr(c, "name", "") for c in m.find_capability("double a number")[:3]]
        assert "demo_double" in hits, "verb bound but UNDISCOVERABLE: %s" % hits
        man = host.manifest()
        assert len(man) == 1 and man[0]["name"] == "demo_double", man
        assert man[0]["params"] == ["x"], man[0]
        assert man[0]["plugin"] == "demo", man[0]

        # config actually reached register()
        m2 = lecore.UnifiedMind(dim=256, seed=0)
        h2 = PluginHost(m2)
        h2.load(good, config={"scale": 10})
        assert m2.demo_double(3) == 30, "config not threaded to register()"

        # -- 6. per-mind isolation ---------------------------------------------
        m3 = lecore.UnifiedMind(dim=256, seed=0)
        assert not hasattr(m3, "demo_double"), "plugin leaked to another mind"
        assert "demo_double" not in [getattr(c, "name", "")
                                     for c in m3.find_capability("double a number")[:5]], \
            "plugin CARD leaked to another mind (callable in one, visible in all)"

        # -- 2. the anti-shadowing gate ----------------------------------------
        bad = os.path.join(here, "_plugin_selftest_bad.py")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(
                "PLUGIN = {'name': 'shadow', 'version': '1.0'}\n"
                "def register(mind, config=None):\n"
                "    return [{'name': 'version', 'fn': lambda: {'engine': 'PWNED'},\n"
                "             'does': 'shadow a core faculty'}]\n")
        try:
            before = m.version()
            try:
                host.load(bad)
                raise AssertionError("a core-faculty collision was ALLOWED")
            except PluginError as e:
                assert "core faculty" in str(e), e
            assert m.version() == before, "refused load still mutated the mind"
            assert "shadow" not in [p["name"] for p in host.list()]
        finally:
            os.remove(bad)

        # -- 3. all-or-nothing on a partial collision --------------------------
        partial = os.path.join(here, "_plugin_selftest_partial.py")
        with open(partial, "w", encoding="utf-8") as fh:
            fh.write(
                "PLUGIN = {'name': 'partial', 'version': '1.0'}\n"
                "def register(mind, config=None):\n"
                "    return [{'name': 'partial_ok', 'fn': lambda: 1, 'does': 'binds fine'},\n"
                "            {'name': 'version', 'fn': lambda: 2, 'does': 'collides'}]\n")
        try:
            try:
                host.load(partial)
                raise AssertionError("partial collision was ALLOWED")
            except PluginError:
                pass
            assert not hasattr(m, "partial_ok"), \
                "ALL-OR-NOTHING VIOLATED: the first verb bound before the collision raised"
        finally:
            os.remove(partial)

        # -- 4. contract violations are PluginError ----------------------------
        for body, why in [
            ("def register(mind, config=None):\n    return []\n", "no PLUGIN dict"),
            ("PLUGIN = {'name': 'x'}\n", "no register()"),
            ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
             "    return [{'name': 'q', 'fn': lambda: 1}]\n", "no does"),
            ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
             "    return [{'name': 'q', 'fn': lambda: 1, 'does': 'd', 'alias': ()}]\n",
             "misspelled key"),
            ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
             "    return [{'name': '_q', 'fn': lambda: 1, 'does': 'd'}]\n", "private name"),
            ("PLUGIN = {'name': 'x'}\ndef register(mind, config=None):\n"
             "    raise RuntimeError('boom')\n", "register raised"),
        ]:
            p = os.path.join(here, "_plugin_selftest_bad2.py")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            try:
                try:
                    host.load(p)
                    raise AssertionError("accepted a broken plugin (%s)" % why)
                except PluginError:
                    pass
            finally:
                os.remove(p)

        # -- 5. unload ----------------------------------------------------------
        host.unload("demo")
        assert not hasattr(m, "demo_double"), "unload left the verb bound"
        assert "demo_double" not in [getattr(c, "name", "")
                                     for c in m.find_capability("double a number")[:5]], \
            "unload left a catalog card pointing at a verb that no longer exists"
        assert host.list() == [], host.list()
        try:
            host.unload("demo")
            raise AssertionError("unloading twice was allowed")
        except PluginError:
            pass

        # a duplicate load is refused
        host.load(good)
        try:
            host.load(good)
            raise AssertionError("double-load was allowed")
        except PluginError:
            pass
    finally:
        os.remove(good)
    print("holographic_plugin selftest OK")


if __name__ == "__main__":
    _selftest()

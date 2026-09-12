"""Part 26 of UnifiedMind's faculty surface -- THE PLUGIN DOOR.

Extending a mind without growing the core. The logic lives in
holographic.io_and_interop.holographic_plugin; every method here DELEGATES.

THE UNDERSCORES ARE THE SECURITY DESIGN, copied from `_register_command` (part 11).
The HTTP service exposes every PUBLIC mind method to /invoke by name, blocking only
leading underscores. So the two faculties that MUTATE the mind's verb surface --
`_plugin_load` and `_plugin_unload` -- are operator calls: configure the mind
in-process, THEN serve it. A public loader would be arbitrary code execution
reachable by any agent that can POST; a public unloader would let one agent strip
another's capabilities. The READ-ONLY half (`plugin_list`, `plugin_manifest`) is
public, because knowing what is loaded is exactly what an agent legitimately needs
and is what keeps a plugin verb from being callable-but-invisible.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone
uses. Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""

from holographic.unified import check_part


class _UnifiedPart26:

    @property
    def _plugins(self):
        """This mind's PluginHost, built lazily so a mind that loads no plugins pays
        nothing -- the same lazy-property pattern as `_editor` and `_commands`."""
        if getattr(self, "_plugin_host_obj", None) is None:
            from holographic.io_and_interop.holographic_plugin import PluginHost
            self._plugin_host_obj = PluginHost(self)
        return self._plugin_host_obj

    def _plugin_load(self, ref, config=None, register_in_catalog=True):
        """OPERATOR configuration -- load a plugin and bind its verbs to THIS mind.

        UNDERSCORE-PREFIXED ON PURPOSE (see the module docstring): a public loader would
        let an agent POST /invoke {"name":"plugin_load","args":{"ref":"/tmp/evil.py"}}
        and execute arbitrary code in the service process.

        `ref` is a dotted module path ('mypkg.lean_plugin') or a .py file path; there is
        deliberately no directory auto-scan. `config` is handed to the plugin's
        register(mind, config). Verbs are validated and collision-checked BEFORE any of
        them binds, so a plugin whose third verb collides leaves the mind untouched
        rather than half-extended. A verb that would shadow a core faculty is REFUSED --
        there is no override flag, because a shadowed faculty makes GET /tools advertise
        a signature /invoke does not run. Returns {name, version, does, ref, verbs}.
        See holographic_plugin.PluginHost.load."""
        # A BARE NAME ("zig", "voice") resolves through discovery, so re-loading a plugin you
        # unloaded -- or loading one you left out with plugins=() -- does not require knowing
        # where it lives. Anything with a dot, a slash or a .py suffix is a ref as before.
        source, expect = "", None
        if isinstance(ref, str) and ref.isidentifier():
            from holographic.plugins import discover
            found = {n: (r, s) for n, r, s in discover()}
            if ref not in found:
                raise ValueError("no plugin named %r discovered -- available: %s"
                                 % (ref, sorted(found)))
            expect = ref
            if config is None:
                config = (getattr(self, "_plugin_config", None) or {}).get(ref)
            ref, source = found[ref]
        return self._plugins.load(ref, config=config, register_in_catalog=register_in_catalog,
                                  source=source, expect_name=expect)

    def _plugin_unload(self, name):
        """OPERATOR configuration -- remove a plugin's verbs and its catalog cards from
        this mind. Private for the same reason as `_plugin_load`: over the wire it would
        let one agent strip capabilities another is relying on.

        HONEST LIMIT: a reference a caller already took (`fn = mind.lean_verify`) keeps
        working -- Python cannot revoke it. Unload removes the NAME, which is what
        discovery and /invoke go through. See holographic_plugin.PluginHost.unload."""
        return self._plugins.unload(name)

    def plugin_list(self):
        """Which plugins are loaded on this mind: [{name, version, does, ref, verbs}], in
        load order. Read-only and public -- the preflight an app or agent runs to find out
        whether the voice / solver / renderer verbs it wants are present, rather than
        sniffing for attributes and guessing. Empty list when none are loaded.
        See holographic_plugin.PluginHost.list."""
        return self._plugins.list()

    def plugin_manifest(self):
        """Every plugin-contributed verb as tool records: [{name, description, params,
        call, plugin}] -- the same shape GET /tools uses for core faculties, so a client
        merging the two lists needs no special case.

        WHY THIS IS NOT OPTIONAL: `skills.manifest()` reads methods off the UnifiedMind
        CLASS and caches them module-wide ('the class doesn't change at runtime'), so a
        verb bound to an INSTANCE was measurably callable via /invoke yet invisible to
        features(), complete() and /tools. This is the faculty that closes that gap.
        See holographic_plugin.PluginHost.manifest."""
        return self._plugins.manifest()


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p26_plugins", "_UnifiedPart26")
    print("holographic_unified_p26_plugins selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

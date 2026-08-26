"""flat_mount.py -- the supported way to mount leCore FLAT (client S-1).

THE DUALITY, stated: the engine's source of truth is the `holographic/` package, and flat modules
(`holographic_terrain.py` next to your code) internally perform PACKAGED imports
(`from holographic.mesh_and_geometry import ...`). An embedder who copies modules out flat hits
ModuleNotFoundError unless both directions resolve. Every embedder was rediscovering the shim;
this file IS the shim, shipped.

USAGE (one line, before any engine import):

    import flat_mount; flat_mount.install("/path/to/flat/modules")   # or the current dir

WHAT IT DOES, both directions:
  * packaged -> flat: synthesises the `holographic` package (and its family subpackages) in
    sys.modules, with a module finder that resolves `holographic.<family>.holographic_x` to the
    flat file `holographic_x.py` -- so the flat modules' own internal packaged imports work.
  * flat -> packaged: if the real package IS importable (repo mount), `import holographic_x`
    aliases to `holographic.<family>.holographic_x` instead -- so code written against flat
    names runs unchanged on a packaged install. Family membership comes from the package itself.

Deterministic, stdlib-only, no import side effects beyond the aliasing it exists to provide.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_FAMILIES = ("mesh_and_geometry", "rendering", "sampling_and_signal", "agents_and_reasoning",
             "caching_and_storage", "io_and_interop", "materials_and_texture", "misc", "unified",
             "geometry_and_fields")


class _FlatFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if parts[0] != "holographic":
            return None
        if len(parts) <= 2:                       # the package / a family: synthesise a namespace
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        flat = os.path.join(self.root, parts[-1] + ".py")
        if os.path.exists(flat):
            return importlib.util.spec_from_file_location(fullname, flat)
        return None


def install(flat_root="."):
    """Install the two-way shim. Safe to call twice."""
    flat_root = os.path.abspath(flat_root)
    try:
        import holographic                        # a real packaged install wins
        _alias_flat_to_packaged()
        return "packaged"
    except ImportError:
        pass
    if not any(isinstance(f, _FlatFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FlatFinder(flat_root))
    if flat_root not in sys.path:
        sys.path.insert(0, flat_root)
    return "flat"


def _alias_flat_to_packaged():
    """`import holographic_x` -> the packaged module, wherever its family home is."""
    import holographic

    class _Alias(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if "." in fullname or not fullname.startswith("holographic_"):
                return None
            for fam in _FAMILIES:
                try:
                    mod = importlib.import_module("holographic.%s.%s" % (fam, fullname))
                    sys.modules[fullname] = mod
                    return importlib.util.spec_from_loader(fullname, loader=None)
                except ImportError:
                    continue
            return None
    if not any(type(f).__name__ == "_Alias" for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())

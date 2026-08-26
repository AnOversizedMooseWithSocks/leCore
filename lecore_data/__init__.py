"""lecore_data -- the vendored data files leCore needs at RUNTIME, packaged so they ship in the wheel.

WHY THIS IS A PACKAGE (and not just a plain data/ folder). leCore installs its modules FLAT -- as top-level
`holographic_*.py` files (setup.py uses py_modules), exactly as they sit in the repo. A loose data/ directory sitting
beside flat modules is NOT carried into a wheel by that layout, so a `pip install` would be missing its data. Making the
runtime data a small IMPORTABLE package is the standard, reliable way to ship data files with Python: `import lecore_data`
resolves the same way from a source clone and from a pip-installed wheel, and file() hands back an absolute path to a
bundled file either way -- no sys.path tricks, no guessing.

Only RUNTIME data lives here:
  * knowledge/dictionary.json.gz  -- the WordNet dictionary the lookup faculty reads (holographic_dictionary).
  * definitions/.../*.json        -- material property data the heat model reads (holographic_heat), which also has
                                     built-in fallbacks, so it degrades gracefully if this is ever absent.

Demo / analysis datasets (market ticks in the repo's data/ folder, etc.) stay OUT of this package on purpose -- they
are for running the demos from a clone, not part of the installed library.
"""
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def file(*parts):
    """Absolute path to a bundled data file, e.g. file('knowledge', 'dictionary.json.gz'). Does not check existence --
    use exists() for that. Works identically from a clone and a pip-installed wheel."""
    return os.path.join(_ROOT, *parts)


def exists(*parts):
    """True if the bundled data file is present."""
    return os.path.exists(file(*parts))

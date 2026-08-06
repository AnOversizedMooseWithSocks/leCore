"""COMPATIBILITY SHIM -- this module moved to `holographic_supermemory`.

WHY THE RENAME (Rule-0 lesson, on record in NOTES): a week-old, DIFFERENT module
already lived at holographic/misc/holographic_superposed.py (leOS-ported "computing
in superposition"); this one's build audit queried capability phrasings but never
grepped the basename, so two unrelated modules shared a name across families -- a
discoverability tax caught by the fuzzy-ask demo answering 'misc' for this module's
name. The capacity-law memory now lives under its own name; this shim keeps every
existing import working forever (additive, backward-compatible only).
"""
from holographic.caching_and_storage.holographic_supermemory import *          # noqa: F401,F403
from holographic.caching_and_storage.holographic_supermemory import (          # noqa: F401
    _selftest as _supermemory_selftest,
)


def _selftest():
    """A SHIM'S CONTRACT IS THAT OLD IMPORTS STILL WORK -- so that is what this asserts.

    It previously re-exported `_selftest` from the new module instead of defining one, which is
    reasonable-looking and left the module counted as "has a __main__ but asserts nothing": the CI
    census matches the TEXT `def _selftest`, and an imported name is not a definition. The module was
    therefore never actually exercised, and the budget test went red the moment it landed.

    Re-exporting also tested the wrong thing. Running the new module's selftest through the old name
    proves the NEW module works; it does not prove the SHIM does. What can break here is the
    re-export itself -- a name dropped from `__all__`, or a rename on the far side -- so the check is
    that the public surface still resolves through this path and is the SAME object, not a copy.
    """
    import holographic.caching_and_storage.holographic_supermemory as _new
    import holographic.caching_and_storage.holographic_superposed as _old

    public = [n for n in dir(_new) if not n.startswith("_")]
    assert public, "the target module exposes nothing -- the shim would be re-exporting an empty API"
    missing = [n for n in public if not hasattr(_old, n)]
    assert not missing, "shim dropped re-exports (star-import or __all__ drift): %r" % missing[:8]
    # IDENTITY, not equality: the shim must forward to the same objects, so callers holding the old
    # name and the new name cannot diverge.
    for n in public:
        assert getattr(_old, n) is getattr(_new, n), "shim rebound %r to a different object" % n

    # And the thing it forwards to must itself still be green, so a broken target cannot hide behind
    # a healthy-looking shim.
    _supermemory_selftest()
    print("superposed shim selftest OK: %d public names re-export identically to supermemory, "
          "target selftest green" % len(public))


if __name__ == "__main__":
    _selftest()

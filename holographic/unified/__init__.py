"""The parts of UnifiedMind. Assembled by holographic.misc.holographic_unified -- import that.

Also the ONE HOME for the parts' shared self-check. Each part used to carry its own structurally identical
copy, which the duplication audit correctly flagged as 13 cross-module duplicates; its instruction in that
case is explicit -- "unify them (one home, an import) ... do not raise the budget to make the test pass" --
so the logic lives here and each part keeps a two-line wrapper.
"""


def check_part(module_name, class_name):
    """Assert the only contract a mixin part actually has: every member it defines reached the assembled
    UnifiedMind, and it is THIS part's body that won the MRO.

    The second assert is the whole point of the split's safety. It holds only while each name lives in
    exactly ONE part -- if a future edit defines a name in two of them, Python resolves the FIRST base
    silently and the other body becomes dead code with no error raised anywhere. Identity (`is`), not
    equality, so a same-named lookalike cannot pass.

    THE CLASS IS RE-IMPORTED BY ITS CANONICAL PATH rather than taken from the caller's globals, and that is
    not a nicety: `python -m holographic.unified.<part>` executes the part a SECOND time under the name
    `__main__`, producing a distinct class object whose functions fail an identity check against the ones
    UnifiedMind actually inherited. Written the obvious way, this reported all 97 members of part 1 as
    shadowed -- a false alarm about the code, caused by the test. Cheap by design: no mind is booted."""
    import importlib, inspect
    from holographic.misc.holographic_unified import UnifiedMind

    part = getattr(importlib.import_module(module_name), class_name)
    mine = {n: v for n, v in vars(part).items() if not n.startswith("__")}
    missing = sorted(n for n in mine if not hasattr(UnifiedMind, n))
    assert not missing, "defined in this part but absent from UnifiedMind: %s" % missing
    shadowed = sorted(n for n, v in mine.items() if inspect.getattr_static(UnifiedMind, n, None) is not v)
    assert not shadowed, ("UnifiedMind resolves %d of this part's names to a DIFFERENT body: %s -- a name must "
                          "live in exactly one part, or the MRO silently drops one of them"
                          % (len(shadowed), shadowed))
    return len(mine)

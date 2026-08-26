"""holographic_srcindex.py -- ONE content-addressed parse of the source tree, shared by every self-audit.

WHY, AND WHY THIS TIER SPECIFICALLY
------------------------------------
The self-audits re-parsed the whole tree on every call, and they did it separately from each other:
orphanaudit walks it for definitions and again for references; codehealth walks it again for complexity and
again for the delegation map. Roughly five full AST passes over ~600 files per report. MEASURED:

    orphan audit()   7,271 ms
    health_report() 10,358 ms      (of which the orphan audit is 7,271 -- it calls it)

That is not a cache-shaped problem invented to justify a cache. It is the exact `use_when` of the machine
model's L3 tier, quoted from its own spec sheet:

    t3_content_addressed   L3 -- shared, content-addressed, deduped
        use_when : the same spec is compiled from many call sites
        NOT when : the evaluator is not deterministic -- a cached nondeterministic evaluator is a bug

Both conditions hold and the second is worth checking rather than assuming: parsing Python source to an AST
is a pure function of the bytes. Same bytes, same tree, on any machine, under any PYTHONHASHSEED. So this
module routes the parse through `holographic_compile.CompileCache` -- the engine's own L3 unit -- instead of
hand-rolling a dict beside it.

THE KEY IS HASHED ONCE PER TREE, NOT ONCE PER LOOKUP
-----------------------------------------------------
This is the lesson from the SpectrumCache correction, applied before making the same mistake again: that
cache keyed on a sha256 of its operand computed on EVERY lookup, and hashing D floats cost MORE than the
transform it was avoiding (0.40x-0.82x -- a cache slower than no cache). The rule that fell out of it is
that a content key must be priced against the work it stands in for, and paid once per BLOCK.

So the digest here covers the whole tree in one pass -- ~600 files, ~18 MB, tens of milliseconds -- and is
compared against a 7-to-10 SECOND reparse. Ratio ~1:300. Within a single index build the per-file trees are
memoised by path so the five passes become one.

WHAT IS DELIBERATELY NOT DONE (kept negatives)
-----------------------------------------------
  * NO mtime/size KEYING, though it would be cheaper. mtime is not content: a fresh clone changes every
    mtime and would cold-miss the whole tree, and a touched-but-unmodified file would invalidate for
    nothing. Content addressing is the tier's name for a reason.
  * NO CACHING ACROSS PROCESSES. The artifact is ASTs and dicts, not bytes; serialising it would cost more
    than the reparse it saves, and a stale on-disk index of a source tree is a genuinely dangerous object.
    This is a within-process cache and says so.
  * NO INVALIDATION HOOK ON WRITE. The digest IS the invalidation -- an edited file changes the key and the
    next call rebuilds. Anything cleverer would be a second source of truth about what the tree contains.
"""
import ast
import glob
import hashlib
import os

from holographic.scene_and_pipeline.holographic_compile import CompileCache

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The L3 unit itself. Small: an index artifact is large, and holding more than a couple of distinct trees is
# not a workload anyone has -- a session audits one checkout.
_CACHE = CompileCache(maxsize=4)


def tracked_paths(root=None, patterns=("holographic/**/*.py", "lecore.py", "app.py", "holographic_service.py")):
    """Every source file the audits reason about, sorted so the digest is order-independent of the filesystem."""
    root = root or REPO
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(out))


def tree_digest(paths):
    """One sha256 over the whole tree: each file's repo-relative path and its bytes.

    Paths are folded in as well as contents so that RENAMING a file invalidates -- two trees with identical
    file contents under different names are different trees to an audit that reports locations."""
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.relpath(p, REPO).encode("utf-8"))
        h.update(b"\0")
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def _parse_all(paths):
    """path -> ast.Module for every readable, parseable file. Unparseable files are simply absent, which is
    the same thing every caller already handled -- none of the audits should fail because one file is mid-edit."""
    trees = {}
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                trees[p] = ast.parse(f.read(), filename=p)
        except (OSError, SyntaxError):
            continue
    return trees


def parsed_trees(root=None, cache=None):
    """{path: ast.Module} for the tree, parsed ONCE per distinct content and reused thereafter.

    The digest is computed on every call (it is the cheap half); the parse happens only when the digest is
    new. Pass your own CompileCache to isolate a test from the module-level one."""
    paths = tracked_paths(root)
    digest = tree_digest(paths)
    c = cache if cache is not None else _CACHE
    return c.get_or_compile(digest, lambda _d: _parse_all(paths), tag="srcindex")


def index_stats():
    """Hit/miss counters for the shared index, plus the hit rate. `compiles` is the number to watch: it should
    be 1 per distinct tree state, no matter how many audits run."""
    s = dict(_CACHE.stats)
    s["hit_rate"] = _CACHE.hit_rate()
    s["resident_trees"] = len(_CACHE)
    return s


def index_clear():
    """Drop the index (and its counters). Only needed if a file changed in a way the digest cannot see, which
    by construction it cannot -- so this is for tests."""
    _CACHE.clear()
    return True


def _selftest():
    """Asserts the two properties that make this safe: the artifact is IDENTICAL across calls (so audits
    reading it cannot disagree), and an edit INVALIDATES (so it can never serve a stale tree)."""
    import tempfile
    c = CompileCache(maxsize=4)
    a = parsed_trees(cache=c)
    b = parsed_trees(cache=c)
    assert a is b, "the second call rebuilt the index instead of hitting the cache"
    assert c.stats["compiles"] == 1, "expected exactly one compile, got %d" % c.stats["compiles"]
    assert len(a) > 400, "the index only found %d parseable files -- the scan missed the tree" % len(a)

    # an edit must invalidate: same tree plus one new file is a different digest
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, "a.py")
        with open(p1, "w") as f:
            f.write("def x():\n    pass\n")
        d1 = tree_digest([p1])
        with open(p1, "w") as f:
            f.write("def x():\n    return 1\n")
        d2 = tree_digest([p1])
        assert d1 != d2, "an edited file did not change the digest -- the cache would serve a stale tree"
        # and a rename must invalidate too, even with identical bytes
        p2 = os.path.join(d, "b.py")
        os.rename(p1, p2)
        assert tree_digest([p2]) != d2, "a renamed file kept its digest -- locations would be reported wrong"

    print("holographic_srcindex selftest OK -- %d files indexed, 1 compile, edit and rename both invalidate"
          % len(a))


if __name__ == "__main__":
    _selftest()

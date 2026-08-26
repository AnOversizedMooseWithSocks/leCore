"""holographic_capuri.py -- capability names as URIs: a branching namespace over every public function.

THE PROBLEM
-----------
The engine has ~1800 public functions across ~450 modules, and 42 of their SHORT names collide: `sphere` lives in
both mesh_and_geometry/sdf and misc/codegen; `resolve` in FOUR modules; `decompose` in two. Today the only way to
tell them apart is to know which module to import from. A bare name like "sphere" is ambiguous, and there is no
way to BROWSE the space the way you'd browse a context menu -- open a branch, see what's under it, drill in.

THE IDEA (S3-style, same as holographic_uri for scene items -- generalized on contact)
--------------------------------------------------------------------------------------
Don't nest folders; NAME things by their location so the name itself IS the hierarchy. Every public function has a
natural address computed from where it lives:

    family / module / function            e.g.  mesh_and_geometry/sdf/sphere
                                                 misc/codegen/sphere

The two spheres now have distinct URIs. A COLLISION is just a bare name that resolves to more than one URI, and you
disambiguate by supplying enough of the path to be unique -- exactly like a filesystem, or S3's flat keyspace with
'/' delimiters, or a context menu you address by the sequence of choices (File > Export > PNG).

This module reuses the SAME roll-up primitive as holographic_uri (common_prefixes -- S3's CommonPrefixes), so the
branching-menu behaviour is not reinvented: it is the scene-URI machinery pointed at the capability namespace.

WHAT YOU CAN DO
---------------
  * resolve_uri("sphere")            -> ['mesh_and_geometry/sdf/sphere', 'misc/codegen/sphere']  (all matches)
  * resolve_uri("sdf/sphere")        -> ['mesh_and_geometry/sdf/sphere']                          (disambiguated)
  * browse("mesh_and_geometry/") -> the next-level branches under it, with counts (open a submenu)
  * browse("")                   -> the top-level families (the root menu)
  * menu_path("mesh_and_geometry/sdf/sphere") -> ['mesh_and_geometry', 'sdf', 'sphere']       (the choices)

Deterministic (sorted, hashlib-free -- pure path arithmetic). NumPy/stdlib only. Additive: a new read-only view
over the existing tree; it changes NO existing name and moves NO code.
"""

import ast
import os

from holographic.io_and_interop.holographic_uri import common_prefixes


# ---------------------------------------------------------------------------------------------------------
# Build the namespace: walk family/module/function once, cache the URI list.
# ---------------------------------------------------------------------------------------------------------

def _repo_root():
    """The holographic/ package root, found relative to this file (family dir is our parent's parent)."""
    here = os.path.dirname(os.path.abspath(__file__))          # .../holographic/caching_and_storage
    return os.path.dirname(here)                               # .../holographic


def build_namespace(root=None):
    """Walk every holographic_*.py under the package and return the sorted list of capability URIs
    'family/module/function' for every PUBLIC (non-underscore) top-level function. The name IS the hierarchy --
    no hand-assignment, so the namespace can never drift from the code. Deterministic (sorted). Reads the source
    with AST (never imports the modules -- the same discipline docgen uses, so a heavy import can't slow or break
    the namespace)."""
    root = root or _repo_root()
    uris = []
    for dirpath, _dirs, files in os.walk(root):
        family = os.path.basename(dirpath)
        for fn in files:
            if not (fn.startswith("holographic_") and fn.endswith(".py")):
                continue
            module = fn[len("holographic_"):-len(".py")]
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read())
            except (SyntaxError, OSError):
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    uris.append("%s/%s/%s" % (family, module, node.name))
    return sorted(set(uris))


_NS_CACHE = None


def _namespace():
    """The cached capability URI list (built once per process). Cheap AST walk; cached so browse/resolve are O(1)
    amortised. WHY a module-level cache and not a class: the namespace is a pure function of the source tree, so a
    single process-wide copy is correct and the simplest thing that works."""
    global _NS_CACHE
    if _NS_CACHE is None:
        _NS_CACHE = build_namespace()
    return _NS_CACHE


# ---------------------------------------------------------------------------------------------------------
# The three operations: resolve a name/path, browse a branch, split a URI into its menu choices.
# ---------------------------------------------------------------------------------------------------------

def resolve_uri(query, namespace=None):
    """Resolve a bare name or a partial path to the FULL capability URI(s) that match, best (most specific) first.
    A bare colliding name returns every match ('sphere' -> two URIs); supplying more of the path narrows it
    ('sdf/sphere' -> one). Matching is by trailing path segments: a query matches a URI when the URI ends with
    '/query' or equals it, or (for a bare name) when the final segment equals the query. Deterministic (sorted).

    Examples:
        resolve_uri('sphere')     -> ['mesh_and_geometry/sdf/sphere', 'misc/codegen/sphere']
        resolve_uri('sdf/sphere') -> ['mesh_and_geometry/sdf/sphere']
        resolve_uri('resolve')    -> the four 'resolve' URIs, so a collision is VISIBLE, not silently picked."""
    ns = namespace if namespace is not None else _namespace()
    q = query.strip("/")
    exact = [u for u in ns if u == q]
    if exact:
        return exact
    # match on trailing segments: the query is a suffix of the URI path (segment-aligned).
    suffix = "/" + q
    hits = [u for u in ns if u.endswith(suffix)]
    if hits:
        return sorted(hits, key=lambda u: (u.count("/"), u))   # most specific (shortest path) first
    # last resort: the final segment equals the bare query (a name anywhere in the tree).
    return sorted(u for u in ns if u.rsplit("/", 1)[-1] == q)


def browse(prefix="", namespace=None):
    """Browse the capability namespace like a CONTEXT MENU: return the next-level branches under `prefix`, each
    with the count of leaves beneath it -- S3's CommonPrefixes roll-up. browse('') is the root menu (the
    families); browse('mesh_and_geometry/') opens that submenu (its modules); browse('mesh_and_geometry/sdf/')
    lists the leaf functions. Returns {branch_uri: leaf_count}, sorted. Delegates the roll-up to
    holographic_uri.common_prefixes -- the same primitive that rolls up scene keys, not a reinvention."""
    ns = namespace if namespace is not None else _namespace()
    return common_prefixes(ns, prefix=prefix, delim="/")


def semantic_namespace(catalog=None):
    """Build the SEMANTIC namespace: the sorted list of `semantic/path/name` URIs for every catalog capability that
    carries a `semantic=` tag (the File->Export->PNG verb tree). Untagged capabilities are omitted -- this view is
    the curated action hierarchy, not the whole namespace. Pairs with browse_semantic the way build_namespace pairs
    with browse. `catalog` defaults to the default_catalog()."""
    if catalog is None:
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        catalog = default_catalog()
    uris = []
    for cap in catalog.all():
        sem = getattr(cap, "semantic", None)
        if sem:
            uris.append(sem.rstrip("/") + "/" + cap.name)
    return sorted(set(uris))


def browse_semantic(prefix="", catalog=None):
    """Browse the SEMANTIC action tree like a context menu -- the File->Export->PNG verb hierarchy, grouped by what
    a user DOES rather than where the code lives. browse_semantic('') is the root verb menu (select/ transform/
    create/ ...); browse_semantic('transform/') opens that submenu (gizmo, snap, pivot, ...); a full path lists the
    capabilities under it. Returns {branch: leaf_count}, sorted. Complements browse() (which groups by physical
    location) -- same roll-up primitive, a different index over the same capabilities. See SEMANTIC_TAXONOMY.md."""
    ns = semantic_namespace(catalog)
    return common_prefixes(ns, prefix=prefix, delim="/")


def menu_path(uri):
    """Split a capability URI into its ordered menu CHOICES -- 'mesh_and_geometry/sdf/sphere' ->
    ['mesh_and_geometry', 'sdf', 'sphere']. The sequence of context-menu selections that addresses it (like
    File > Export > PNG). The inverse is '/'.join(...)."""
    return [seg for seg in uri.split("/") if seg]


def collisions(namespace=None, ignore_structural=False):
    """Every BARE function name that resolves to more than one URI -- the semantic collisions, now each with its
    disambiguating full paths. Returns {bare_name: [uri, ...]}, sorted. This is the name_collisions audit re-read
    through the URI lens: a collision is not a hazard to forbid but a name whose PATH you must supply. Useful for
    generating the disambiguation table and for confirming every collision is at least addressable.

    `ignore_structural=True` drops the structural names that every module may define (`main`, `demo`, `run`) -- the
    same set tools/name_collisions.py ignores -- so the two collision reports RECONCILE to the same count. Kept as a
    flag (default False) so the raw URI view still shows every multi-home name; the audit view passes True."""
    ns = namespace if namespace is not None else _namespace()
    by_name = {}
    for u in ns:
        name = u.rsplit("/", 1)[-1]
        by_name.setdefault(name, []).append(u)
    out = {n: sorted(us) for n, us in sorted(by_name.items()) if len(us) > 1}
    if ignore_structural:
        for n in _STRUCTURAL_NAMES:
            out.pop(n, None)
    return out


# structural names every module may define -- not capability collisions. Kept in ONE place and shared with
# tools/name_collisions.py (imported there) so the two audits can never disagree on what counts as a collision.
_STRUCTURAL_NAMES = frozenset({"main", "demo", "run"})


def _selftest():
    """Contracts:
    1. build_namespace finds a large, sorted, de-duplicated URI list.
    2. A known collision ('sphere') resolves to >=2 URIs; supplying the module segment narrows it to exactly one.
    3. browse('') returns the families; browsing into a family returns its modules; the counts are positive.
    4. menu_path round-trips with '/'.join.
    5. collisions() surfaces multi-URI names, each addressable (the URI lens on name_collisions).
    6. Everything is deterministic (same call twice -> identical result).
    """
    ns = build_namespace()
    assert len(ns) > 500, "expected a large capability namespace, got %d" % len(ns)
    assert ns == sorted(set(ns)), "namespace must be sorted and de-duplicated"

    # (2) a known collision resolves to several; the path narrows it.
    sph = resolve_uri("sphere", ns)
    assert len(sph) >= 2, ("sphere should collide across modules, got %r" % sph)
    narrowed = resolve_uri("sdf/sphere", ns)
    assert len(narrowed) == 1 and narrowed[0].endswith("/sdf/sphere"), narrowed

    # (3) browse behaves like a menu.
    root = browse("", ns)
    assert all(k.endswith("/") for k in root), "root menu should be families (all end in '/')"
    assert "mesh_and_geometry/" in root, root
    sub = browse("mesh_and_geometry/", ns)
    assert any(k.startswith("mesh_and_geometry/") for k in sub) and all(v > 0 for v in sub.values())

    # (4) menu_path round-trips.
    u = "mesh_and_geometry/sdf/sphere"
    assert menu_path(u) == ["mesh_and_geometry", "sdf", "sphere"]
    assert "/".join(menu_path(u)) == u

    # (5) collisions are surfaced and addressable.
    coll = collisions(ns)
    assert "sphere" in coll and len(coll["sphere"]) >= 2
    for name, uris in coll.items():
        for uri in uris:
            assert resolve_uri(uri, ns) == [uri], ("a full URI must resolve to exactly itself: %s" % uri)

    # (5b) RECONCILIATION with tools/name_collisions.py: ignore_structural drops exactly the shared structural
    # names, so the two audits agree on the count. A structural name in the raw set must vanish under the flag.
    raw_n = len(collisions(ns))
    filtered = collisions(ns, ignore_structural=True)
    assert all(s not in filtered for s in _STRUCTURAL_NAMES)
    assert len(filtered) <= raw_n and len(filtered) >= raw_n - len(_STRUCTURAL_NAMES)

    # (7) SEMANTIC browse (S2.4): the verb tree groups by action, not location. The root menu is verbs, and
    # browsing into one returns its sub-branches. Only tagged capabilities appear (a curated view).
    sem_root = browse_semantic("")
    if sem_root:                                               # tags exist -> the verb tree is populated
        assert all("/" in k for k in sem_root), sem_root       # root entries are verb branches
        # every semantic URI is addressable back to its capability name (the leaf).
        sns = semantic_namespace()
        assert all(u == u.strip("/") and "/" in u for u in sns)
        # browsing a known root returns sub-branches beneath it.
        first_root = sorted(sem_root)[0]
        assert browse_semantic(first_root), "a populated root must have sub-branches"

    # (6) determinism.
    assert build_namespace() == ns and browse("", ns) == browse("", ns)

    print("holographic_capuri selftest OK (%d capability URIs; 'sphere' collides %d ways and narrows to 1 by "
          "path; %d colliding names, each addressable; browse() branches like a menu; deterministic)"
          % (len(ns), len(sph), len(coll)))


if __name__ == "__main__":
    _selftest()

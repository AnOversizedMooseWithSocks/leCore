"""The DUPLICATION AUDIT, made executable -- so it cannot rot, and so the budget can only go down.

Run with the engine's own `holographic_pycontext.canonical_shape`, which erases identifiers AND constants and
leaves the structural fingerprint of a function body. Two functions with the same fingerprint in two modules are a
*hypothesis* of duplication. Every one below has been read, and the verdict recorded.

**Why not a regex.** The last audit's regex for `def _?wrap\\(` reported six homes; reading them showed
`conformal.wrap` is *conformal prediction* and `fuse.wrap` is unrelated -- exactly one was an angle wrap. And the
backlog's own rev. 1 "77 bind sites" had to be retracted to three. *A scan proposes; only reading disposes.* A
structural shape does better than a name, because it caught `_face_normal` and `_newell_normal` -- the same
algorithm under two names, which no name-based scan would ever pair.

    FIXED (one home now):
      _cell_centers   dirtyfield <- ndfield   (its own docstring said "matching holographic_ndfield's ordering")
      _rot            canonmesh, deltacache <- equivariance.sample_transform

    KNOWN, OPEN -- recorded with an owner, not quietly rewired:
      _reply          coordinator, distbus     identical
      damage_mask     archive, image           ADJUDICATED (D2): all three sites (Hologram,
                                               HolographicImage, HolographicArchive) now DELEGATE to
                                               holographic_ai.damage_mask -- the canonical home, since the
                                               mask is a property of a VECTOR, not of a storage class.
                                               Bit-identity measured on 48 (dim, fraction, seed) configs
                                               BEFORE the rewire and pinned by tests/test_damage_mask.py.
                                               The entry stays because the delegating shims are still
                                               textually identical bodies, by design -- same as _occlusion.
      _occlusion      cosamp, iht              ADJUDICATED: both now DELEGATE to occlusion_recall (measured
                                               bit-identical first) -- the entry stays because the two delegating
                                               shims are still textually identical bodies, by design
      _f1 / _f1f      cosamp, iht, occlusion   identical
      _face_normal    meshverbs == meshcurvature._newell_normal  (same Newell algorithm, two names)

    NOT A DUPLICATE, and the distinction matters:
      _face_normal    meshbridge is a DIFFERENT ALGORITHM under the SAME NAME -- see the test below.
"""

import ast
import collections
import pathlib

import numpy as np

from holographic.io_and_interop.holographic_pycontext import canonical_shape


_ROOT = pathlib.Path(__file__).resolve().parent.parent / "holographic"

#: The duplicate budget. Each entry is `(frozenset(function names), frozenset(modules))`. **This set may shrink and
#: must never grow.** A new entry means someone copied a body into a second module; go read it.
KNOWN_DUPLICATES = {
    (frozenset({"_reply"}), frozenset({"coordinator", "distbus"})),
    (frozenset({"damage_mask"}), frozenset({"archive", "image"})),
    (frozenset({"_occlusion"}), frozenset({"cosamp", "iht"})),
    (frozenset({"_f1", "_f1f"}), frozenset({"cosamp", "iht", "occlusion"})),
    (frozenset({"_face_normal", "_newell_normal"}), frozenset({"meshcurvature", "meshverbs"})),
    # READ, NOT A REAL DUPLICATE: filemap.FileEntry.__init__ (relpath/path/size/mtime/sha256/kind) vs
    # milkdrop.MilkPreset.__init__ (settings/init/frame/pixel/warp_shader/comp_shader). Two unrelated plain
    # record classes; canonical_shape erases identifiers, so ANY two six-field `self.x = x` constructors
    # collide. Boilerplate fingerprint, not copied logic -- nothing to rewire.
    (frozenset({"__init__"}), frozenset({"filemap", "milkdrop"})),
}


def _scan(min_statements=4):
    """Every non-trivial function body, grouped by canonical shape. Bodies under `min_statements` are excluded:
    a two-line delegating wrapper is not duplication, it is the point of a wrapper."""
    groups = collections.defaultdict(list)
    n = 0
    for path in sorted(_ROOT.rglob("holographic_*.py")):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = [s for s in node.body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if len(body) < min_statements:
                continue
            n += 1
            try:
                shape = canonical_shape(ast.get_source_segment(src, node))
            except Exception:
                continue
            groups[shape].append((path.name[len("holographic_"):-3], node.name))
    return groups, n


def _cross_module_duplicates():
    groups, _n = _scan()
    out = set()
    for _shape, entries in groups.items():
        mods = {m for m, _f in entries}
        if len(mods) > 1:
            out.add((frozenset({f for _m, f in entries}), frozenset(mods)))
    return out


def test_the_scan_sees_a_real_codebase():
    _groups, n = _scan()
    assert n > 2500, n                          # ~3,078 non-trivial functions across ~428 modules


def test_the_duplicate_budget_does_not_grow():
    """**A budget that may shrink and must never grow.**

    If this fails with a NEW entry, someone copied a function body into a second module. Read both, then either
    unify them (one home, an import) or -- if they are genuinely different algorithms that happen to share a shape
    -- add the entry here with the reasoning. *Do not raise the budget to make the test pass.*"""
    found = _cross_module_duplicates()
    new = found - KNOWN_DUPLICATES
    assert not new, "new cross-module duplicate(s): %s" % sorted((sorted(f), sorted(m)) for f, m in new)


def test_the_fixed_duplicates_stay_fixed():
    found = _cross_module_duplicates()
    for names in ({"_cell_centers"}, {"_rot"}):
        for fns, _mods in found:
            assert not (set(fns) & names), "%s came back" % (names,)


def test_cell_centers_has_exactly_one_home():
    # Its docstring used to promise it matched `ndfield`'s ordering. **A comment that promises two functions agree
    # is a promise nothing checks; an import is a promise the interpreter checks.**
    from holographic.misc.holographic_dirtyfield import _cell_centers as a
    from holographic.misc.holographic_ndfield import _cell_centers as b
    assert a is b


def test_the_engine_has_one_rodrigues_generator():
    from holographic.mesh_and_geometry.holographic_equivariance import sample_transform

    A, b = sample_transform("rotate", np.random.default_rng(0))
    assert A.shape == (3, 3) and np.allclose(A @ A.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(A) - 1.0) < 1e-12          # a rotation, not a reflection
    assert np.allclose(b, 0.0)


# ---------------------------------------------------------------------------------------------------------
# NOT a duplicate: the same NAME, a different ALGORITHM. This is worse than duplication, and it is pinned.
# ---------------------------------------------------------------------------------------------------------

def test_two_functions_named_face_normal_compute_different_normals():
    """`meshbridge._face_normal` takes the cross product of the FIRST THREE vertices. `meshverbs._face_normal` is
    Newell's method over ALL of them. On a planar triangle they agree; on a bent quad they do not.

    A structural scan pairs `meshverbs._face_normal` with `meshcurvature._newell_normal` -- correctly, they are the
    same algorithm. It does NOT pair either with `meshbridge`'s, also correctly. **The hazard is the name**: a
    reader who imports `_face_normal` from the nearer module gets a different answer on a non-planar face, and
    nothing warns them."""
    from holographic.mesh_and_geometry.holographic_meshbridge import _face_normal as first_three
    from holographic.mesh_and_geometry.holographic_meshcurvature import _newell_normal as newell_curv
    from holographic.mesh_and_geometry.holographic_meshverbs import _face_normal as newell_verbs

    V = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.4], [0, 1, 0]])   # a BENT quad

    tri = [0, 1, 2]
    assert np.allclose(first_three(V, tri), newell_verbs(V, tri), atol=1e-9)     # planar: all three agree
    assert np.allclose(newell_verbs(V, tri), newell_curv(V, tri), atol=1e-9)

    quad = [0, 1, 2, 3]
    assert np.allclose(newell_verbs(V, quad), newell_curv(V, quad), atol=1e-9)   # the two Newells agree
    assert not np.allclose(first_three(V, quad), newell_verbs(V, quad), atol=1e-2)   # ... and the third does not


def test_quat_from_axis_angle_has_one_home():
    """rev. 8 unified `cosserat.qmul` -> `transform.quat_mul` and missed the sibling CONSTRUCTOR. The shape scan
    could not have caught it: the two bodies build the array differently (`np.array([w, *v])` vs `np.concatenate`),
    so their canonical shapes differ while the math is bit-identical -- a blind spot the NAME-collision scan covers.
    Pinned the same way: identical on 200 random (axis, angle) draws AND on the degenerate zero axis."""
    from holographic.simulation_and_physics.holographic_cosserat import quat_from_axis_angle as qa
    from holographic.misc.holographic_transform import quat_from_axis_angle as qb

    rng = np.random.default_rng(0)
    for _ in range(200):
        axis, ang = rng.normal(size=3), float(rng.normal() * 10)
        assert np.array_equal(qa(axis, ang), qb(axis, ang))
    assert np.array_equal(qa([0, 0, 0], 1.0), qb([0, 0, 0], 1.0))


def test_two_functions_named_quat_rotate_agree_on_unit_but_diverge_off_it():
    """`cosserat.quat_rotate` is the sandwich product q*v*q_conj; `transform.quat_rotate` builds a matrix and
    multiplies. Same name, different algorithm -- the `_face_normal` hazard class again. On a UNIT quaternion they
    agree to floating point (2e-15 over 2000 draws); on a NON-UNIT quaternion they diverge COMPLETELY, because
    `quat_to_matrix` normalizes internally and the sandwich does not. A caller who passes an unnormalized
    quaternion gets a silently different answer depending on which module they imported from. NOT merged (they are
    not bit-identical even on unit q); pinned so the divergence -- and the normalization footgun -- is a stated
    fact. The name-collision scan (tools/name_collisions.py) records this pair in its reviewed budget."""
    from holographic.simulation_and_physics.holographic_cosserat import quat_rotate as sandwich
    from holographic.misc.holographic_transform import quat_rotate as matrix

    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4); q /= np.linalg.norm(q)          # UNIT quaternion
        v = rng.normal(size=3)
        assert np.allclose(sandwich(q, v), matrix(q, v), atol=1e-12)     # agree where both are valid

    q_bad = np.array([1.0, 2.0, 3.0, 4.0])                       # NOT normalized
    v = np.array([1.0, 0.0, 0.0])
    assert not np.allclose(sandwich(q_bad, v), matrix(q_bad, v), atol=1e-2)   # ... and diverge where they are not
    """`reproject.psnr` caps at 99.0 for mse < 1e-12 (its docstring says WHY: a fake-perfect `inf` was W4's bug).
    `splat.psnr` caps only at mse == 0.0. Same name, same formula, DIFFERENT tie contract -- the `_face_normal`
    hazard class again, in one comparison operator. They agree everywhere except the band 0 < mse < 1e-12, where
    reproject says 99.0 and splat reports a real (higher) value. NOT merged: making splat adopt reproject's cap
    would flip splat's answer in that band, and bit-identity is the merge gate. Pinned so the divergence is a
    stated fact rather than a surprise."""
    from holographic.rendering.holographic_reproject import psnr as psnr_r
    from holographic.rendering.holographic_splat import psnr as psnr_s

    a = np.zeros(16)
    assert psnr_r(a, a) == psnr_s(a, a) == 99.0                       # exact match: both cap
    b = a.copy(); b[0] = 1e-3
    assert abs(psnr_r(a, b) - psnr_s(a, b)) < 1e-12                   # ordinary mse: identical
    c = a.copy(); c[0] = 4e-7                                         # mse = 1e-14: inside the tie band
    assert psnr_r(a, c) == 99.0 and psnr_s(a, c) > 99.0               # ... and there they disagree


def test_the_deliberate_aliases_are_one_implementation_not_two():
    """`scatter_to_grid` calls `scatter`; `gather_from_grid` calls `gather`. Two names, one implementation -- a
    discoverability choice, not duplicated code. Recorded so nobody 'unifies' a delegation."""
    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(0)
    pts, vals = rng.uniform(0, 1, (30, 2)), rng.normal(size=30)
    assert np.array_equal(np.asarray(m.scatter(pts, vals, (8, 8))),
                          np.asarray(m.scatter_to_grid(pts, vals, (8, 8))))


# ---------------------------------------------------------------------------------------------------------
# The NAME-collision budget (the complement of the shape budget above). Same discipline: it may shrink and
# must never grow. A new public function name in a second module means someone added a collision without
# reading the other body -- CI stops them here until they do.
# ---------------------------------------------------------------------------------------------------------

def _collision_tool():
    import importlib.util
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "name_collisions.py"
    spec = importlib.util.spec_from_file_location("name_collisions", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_unreviewed_public_name_collisions():
    """Every public function name shared across modules is in the reviewed KNOWN_COLLISIONS budget. A NEW one is a
    collision nobody read -- read both bodies, then unify (delegate) or accept with a reason on the budget line."""
    nc = _collision_tool()
    found = nc.scan()
    unreviewed = {n: m for n, m in found.items() if frozenset(m) != nc.KNOWN_COLLISIONS.get(n)}
    assert not unreviewed, ("unreviewed public name collision(s) -- read both bodies, then add to "
                            "tools/name_collisions.KNOWN_COLLISIONS WITH A REASON: %s" % sorted(unreviewed))


def test_the_collision_budget_has_no_stale_entries():
    """The budget must not carry a name that no longer collides (or whose homes moved) -- a lint that lies about
    what exists is worse than no lint."""
    nc = _collision_tool()
    found = nc.scan()
    stale = {n: sorted(ms) for n, ms in nc.KNOWN_COLLISIONS.items() if frozenset(found.get(n, ())) != ms}
    assert not stale, "stale budget entries (collision changed or gone -- update the line): %s" % stale

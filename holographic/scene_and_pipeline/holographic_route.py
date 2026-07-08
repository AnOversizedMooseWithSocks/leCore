"""Representation routing (ARCH-7): route each operation to the representation that supports it -- the policy layer
on top of FWD-11's mesh<->SDF<->splat bridge.

WHY THIS MODULE EXISTS
----------------------
FWD-11 built the CONVERSIONS between the engine's geometry representations (mesh, implicit/SDF, splat). ARCH-7 is
the POLICY that uses them: different operations are natural in different representations, so the right move is to
convert to the representation that makes an operation easy, do it there, and convert back. The engine already does
this implicitly all over (the decode-vs-evaluate principle is the same idea for vectors -- index by the
representation the query actually resembles); ARCH-7 makes it an explicit, table-driven router for geometry.

THE FLAGSHIP: CSG (constructive solid geometry).
  Boolean union / intersection / difference of solids have NO direct implementation in the mesh kernel -- robust
  mesh booleans need surface-surface intersection, which the engine deliberately never built. But on a SIGNED
  DISTANCE FIELD they are trivial, exact field operations:
      union(A,B)        = min(d_A, d_B)        (the nearer surface wins)
      intersection(A,B) = max(d_A, d_B)
      difference(A,B)   = max(d_A, -d_B)       (inside A and outside B)
  So the router takes meshes, routes them to the SDF representation (mesh_to_sdf), combines the fields, and
  extracts the result back to a mesh (marching tetrahedra). Crucially this lets a boolean CHANGE TOPOLOGY -- two
  separate spheres become ONE blob when they overlap, or stay TWO when they don't -- which a mesh cannot do to
  itself. The field merges or keeps-separate automatically.

WHAT IT PROVIDES
  * REPRESENTATION_CAPABILITIES -- the routing table: which operations each representation supports natively.
  * representation_for(operation) -- the router: the representation whose capability set contains `operation`.
  * route_csg(operation, mesh_a, mesh_b, res, bounds) -- the flagship: route two meshes through the SDF to compute
    a boolean, returned as a mesh. `operation` in {"union", "intersection", "difference"}.
  * connected_components(mesh) / mesh_volume(mesh) -- the measurements the boolean's correctness is checked with.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * the table routes booleans to "sdf" (and "union" is explicitly NOT a mesh capability -- that is WHY we route),
    and boundary/render to "mesh".
  * union of two OVERLAPPING spheres -> ONE connected component, a closed manifold (the field merged the topology);
    union of two SEPARATE spheres -> TWO components (separation preserved).
  * the booleans are GEOMETRICALLY correct, not just topologically -- they satisfy inclusion-exclusion to a few
    percent: vol(A or B) = vol(A) + vol(B) - vol(A and B), and vol(A) = vol(A and B) + vol(A minus B).

DETERMINISM (per ISA.md)
  The grid, the field combine, and marching tetrahedra are all deterministic; same meshes + same res -> identical
  result (asserted).

KEPT NEGATIVES (loud)
  * Resolution is the grid's (FWD-11's negative inherited): a boolean's sharp intersection seam is rounded at the
    cell size; raising res sharpens it but costs res^3 field evals. The volumes converge to the true ones from
    below (the marching-tet discretization under-fills) -- that is why the inclusion-exclusion checks carry a few
    percent tolerance, not machine precision.
  * route_csg trusts mesh_to_sdf's sign, which (FWD-11) is reliable for convex-ish closed meshes but can mis-sign
    deep concavities in an INPUT mesh; the spheres here are convex so the combined field is exact. A non-convex
    input would need a winding-number sign (the FWD-11 fix, deferred).
  * The table is a small curated policy, not a learned cost model -- it encodes the published strengths (SDF for
    booleans/offsets, mesh for explicit boundary, splat for soft blends), nothing more.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf, sample_field, marching_tetrahedra


# The routing table: which operations each representation supports NATIVELY. The router converts to whichever
# representation owns an operation. (Curated from published strengths, not learned.)
REPRESENTATION_CAPABILITIES = {
    "mesh":  {"boundary", "render", "subdivide", "extrude", "deform", "unwrap"},
    "sdf":   {"union", "intersection", "difference", "inside_test", "offset"},
    "splat": {"blend", "scatter"},
}


def representation_for(operation):
    """The representation whose capability set contains `operation` -- the routing decision. Raises if no
    representation supports it."""
    for rep, caps in REPRESENTATION_CAPABILITIES.items():
        if operation in caps:
            return rep
    raise ValueError(f"no representation supports operation {operation!r}")


def _auto_bounds(meshes, pad=0.25):
    """A box enclosing all meshes with padding -- the field must be POSITIVE (outside) at the box wall so the
    isosurface closes, which the pad guarantees for the distance fields here."""
    allv = np.vstack([m.vertices for m in meshes])
    return (tuple(allv.min(axis=0) - pad), tuple(allv.max(axis=0) + pad))


def route_csg(operation, mesh_a, mesh_b, res=28, bounds=None):
    """Compute a boolean of two solids by ROUTING through the SDF representation (ARCH-7's flagship). `operation`
    is "union", "intersection", or "difference". Routes mesh_a, mesh_b -> SDF (mesh_to_sdf), combines the fields
    (min / max / max-with-negation), and extracts the result back to a mesh (marching tetrahedra). The result can
    change topology relative to the inputs. Returns a Mesh."""
    rep = representation_for(operation)
    assert rep == "sdf", f"route_csg handles SDF-routed booleans; {operation!r} routes to {rep}"
    if bounds is None:
        bounds = _auto_bounds([mesh_a, mesh_b])

    if operation == "union":
        field = lambda p: np.minimum(mesh_to_sdf(mesh_a, p), mesh_to_sdf(mesh_b, p))
    elif operation == "intersection":
        field = lambda p: np.maximum(mesh_to_sdf(mesh_a, p), mesh_to_sdf(mesh_b, p))
    else:  # difference: inside A and OUTSIDE B
        field = lambda p: np.maximum(mesh_to_sdf(mesh_a, p), -mesh_to_sdf(mesh_b, p))

    values, axes = sample_field(field, bounds, res)
    return marching_tetrahedra(values, axes, level=0.0)


def connected_components(mesh):
    """The number of connected components of a mesh (flood fill over edge adjacency). A boolean union of two
    overlapping solids has 1; of two separate solids, 2."""
    adj = {v: set() for v in range(mesh.n_vertices)}
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            adj[f[k]].add(f[(k + 1) % n])
            adj[f[(k + 1) % n]].add(f[k])
    seen = set()
    count = 0
    for start in range(mesh.n_vertices):
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return count


def mesh_volume(mesh):
    """The enclosed volume of a closed mesh (divergence theorem: sum of signed tetrahedra from the origin over
    outward-wound triangles). Used to check booleans are geometrically -- not just topologically -- correct."""
    V = mesh.vertices
    total = 0.0
    for f in mesh.faces:
        for k in range(1, len(f) - 1):                     # fan-triangulate an n-gon
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            total += np.dot(a, np.cross(b, c))
    return abs(total) / 6.0


def _translate(mesh, offset):
    return Mesh(mesh.vertices + np.asarray(offset, float), [tuple(f) for f in mesh.faces])


# =====================================================================================================
# Self-test -- CSG via SDF routing: topology merges/keeps-separate, and the booleans satisfy inclusion-exclusion.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    sph = _icosphere(2)                                    # unit sphere, V66 F128

    # --- the routing policy: booleans -> sdf (and union is NOT a mesh capability -- that is why we route) ---
    assert representation_for("union") == "sdf"
    assert representation_for("difference") == "sdf"
    assert representation_for("boundary") == "mesh"
    assert "union" not in REPRESENTATION_CAPABILITIES["mesh"], "the mesh can't do booleans -- the reason to route"

    # --- overlapping spheres: union MERGES topology (2 solids -> 1 component), a closed manifold ---
    A = _translate(sph, [-0.5, 0, 0]); B = _translate(sph, [0.5, 0, 0])
    uni = route_csg("union", A, B)
    assert connected_components(uni) == 1, "an overlapping union is ONE blob (the field merged the topology)"
    assert uni.is_closed() and uni.is_manifold()

    # --- separate spheres: union keeps them SEPARATE (2 components) ---
    A2 = _translate(sph, [-1.6, 0, 0]); B2 = _translate(sph, [1.6, 0, 0])
    assert connected_components(route_csg("union", A2, B2)) == 2, "a non-overlapping union stays TWO components"

    # --- geometric correctness: inclusion-exclusion to a few percent (not just topology) ---
    inter = route_csg("intersection", A, B)
    diff = route_csg("difference", A, B)
    vA, vB = mesh_volume(A), mesh_volume(B)
    v_uni, v_int, v_diff = mesh_volume(uni), mesh_volume(inter), mesh_volume(diff)
    assert v_int < vA and v_int < v_uni, "the intersection (a lens) is smaller than either input or the union"
    assert v_diff < vA, "the difference (a bite out of A) is smaller than A"
    assert abs(v_uni - (vA + vB - v_int)) / vA < 0.05, "vol(A or B) = vol(A)+vol(B)-vol(A and B) (inclusion-exclusion)"
    assert abs(vA - (v_int + v_diff)) / vA < 0.05, "vol(A) = vol(A and B) + vol(A minus B)"

    # --- determinism ---
    assert np.array_equal(route_csg("union", A, B).vertices, route_csg("union", A, B).vertices)

    print(f"holographic_route selftest: ok (routing table sends booleans->sdf, boundary->mesh; CSG via the SDF "
          f"bridge -- OVERLAPPING union merges to 1 component (closed manifold), SEPARATE union stays 2; "
          f"geometrically correct by inclusion-exclusion: vol(uni) {v_uni:.2f} ~ vA+vB-vInt "
          f"{vA + vB - v_int:.2f}, and vA {vA:.2f} ~ vInt+vDiff {v_int + v_diff:.2f}; deterministic)")


if __name__ == "__main__":
    _selftest()

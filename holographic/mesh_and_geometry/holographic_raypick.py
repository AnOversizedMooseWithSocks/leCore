"""holographic_raypick.py -- RAY QUERIES against real geometry, the layer that makes viewport picking hit a user's
actual mesh or SDF (not just the demo cage). This is the query half of the interactive edit spine: a screen ray
comes in, a hit record comes out {face/none, position, distance, ...}, and the selection/transform layers act on it.

TWO SURFACES, TWO METHODS
  * ray-vs-mesh  -- Moller-Trumbore ray/triangle intersection (the standard, branch-light, no precomputed plane),
                    with an axis-aligned bounding-box broad phase so a miss is rejected in O(1) before touching any
                    triangle. Quads are split into two triangles on the fly. Returns the NEAREST hit along the ray.
  * ray-vs-SDF   -- sphere tracing (raymarch): step by the signed distance until the surface is hit, then read the
                    normal from the SDF gradient. This is the native query for the field representation -- free
                    relative to the field, per the demoscene lineage.

WHY BOTH LIVE HERE: a modeling app switches between mesh objects and SDF/procedural objects, and picking must work
on either with the SAME hit-record shape so the front end handles one contract. NumPy/stdlib only, deterministic.
"""

import numpy as np


def _tris_from_faces(faces):
    """Fan-triangulate each face (triangle or quad or n-gon) into (i,j,k) triangles, remembering which ORIGINAL
    face each triangle came from -- so a hit reports the face the user sees, not the internal triangle."""
    tris, owner = [], []
    for fi, f in enumerate(faces):
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
            owner.append(fi)
    return tris, owner


def ray_mesh_intersect(mesh, origin, direction, cull_backface=False):
    """Cast a ray at a mesh and return the NEAREST hit, or None on a miss. `origin`/`direction` are 3-vectors
    (direction need not be normalized). Uses a bounding-box broad phase then Moller-Trumbore per triangle. Returns
    {face, position, distance, barycentric, triangle} where `face` is the ORIGINAL face index (quads/n-gons are fan
    triangulated internally). `cull_backface=True` ignores triangles facing away from the ray (a solid-object pick);
    the default hits either side (a surface pick).

    Moller-Trumbore is used because it needs no precomputed per-triangle plane (so it stays additive and cheap for a
    mesh that is being edited every frame) and its barycentric output is exactly what a UV / weight lookup needs."""
    V = np.asarray(mesh["vertices"], float)
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    # BROAD PHASE: if the ray misses the mesh AABB, there is no hit -- reject before any triangle work.
    lo, hi = V.min(axis=0), V.max(axis=0)
    if not _ray_aabb(o, d, lo, hi):
        return None
    tris, owner = _tris_from_faces(mesh.get("faces", []))
    best = None
    eps = 1e-9
    for ti, (a, b, c) in enumerate(tris):
        p0, p1, p2 = V[a], V[b], V[c]
        e1, e2 = p1 - p0, p2 - p0
        h = np.cross(d, e2)
        det = float(e1 @ h)
        if cull_backface and det < eps:                        # triangle faces away -> skip (solid pick)
            continue
        if abs(det) < eps:                                     # ray parallel to the triangle
            continue
        inv = 1.0 / det
        s = o - p0
        u = float(s @ h) * inv
        if u < -eps or u > 1 + eps:
            continue
        q = np.cross(s, e1)
        v = float(d @ q) * inv
        if v < -eps or u + v > 1 + eps:
            continue
        t = float(e2 @ q) * inv
        if t <= eps:                                           # behind the ray origin
            continue
        if best is None or t < best["distance"]:
            best = {"face": int(owner[ti]), "distance": t, "position": (o + t * d).tolist(),
                    "barycentric": [1.0 - u - v, u, v], "triangle": [int(a), int(b), int(c)]}
    return best


def _ray_aabb(o, d, lo, hi):
    """Slab test: does the ray (origin o, unit dir d) hit the axis-aligned box [lo,hi]? The O(1) broad phase that
    rejects a miss before any triangle is touched."""
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (lo - o) * inv
    t1 = (hi - o) * inv
    tmin = np.minimum(t0, t1).max()
    tmax = np.maximum(t0, t1).min()
    return tmax >= max(tmin, 0.0)


def ray_sdf_intersect(sdf_fn, origin, direction, max_dist=50.0, max_steps=128, eps=1e-3):
    """Sphere-trace a ray into an SDF and return the hit, or None on a miss. `sdf_fn(pt)->distance` is any signed
    distance function (a callable, or an SDF node). Returns {position, distance, normal, steps}. The native pick for
    the field/procedural half of a scene -- no triangulation, exact to the field.

    NOT A SECOND MARCHER: the sphere-tracing loop is the canonical, vectorised holographic_raymarch.sphere_trace
    (which drops inactive rays and supports over-relaxation). This wraps the single pick ray as a batch of one, calls
    that marcher, and adds only the two things a PICK wants that the batch marcher doesn't return: the surface NORMAL
    (SDF gradient by central differences) and the hit as a dict record. Every march step is the canonical loop's."""
    from holographic.rendering.holographic_raymarch import sphere_trace

    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)

    # sphere_trace is VECTORISED: it calls sdf(P) with P an (M,3) batch and expects (M,) distances. The pick API
    # takes a per-point sdf_fn(pt)->scalar, so adapt it to a batch here (evaluate each row). One adapter, and the
    # canonical marcher does the actual tracing -- no second march loop.
    def _batched(P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.array([float(sdf_fn(row)) for row in P])

    hit, t, pos = sphere_trace(_batched, o[None, :], d[None, :], max_steps=max_steps, max_dist=max_dist,
                               surf_eps=eps)
    if not bool(hit[0]):
        return None
    p = pos[0]
    h = 1e-3                                                    # outward normal from the field gradient
    n = np.array([float(sdf_fn(p + [h, 0, 0])) - float(sdf_fn(p - [h, 0, 0])),
                  float(sdf_fn(p + [0, h, 0])) - float(sdf_fn(p - [0, h, 0])),
                  float(sdf_fn(p + [0, 0, h])) - float(sdf_fn(p - [0, 0, h]))])
    n = n / (np.linalg.norm(n) + 1e-12)
    return {"position": p.tolist(), "distance": float(t[0]), "normal": n.tolist(), "steps": None}


def screen_ray(screen_u, screen_v, cam_eye=(0.0, 0.0, 3.0), cam_z=-1.6):
    """Build a world-space ray from a normalized screen coordinate (screen_u, screen_v in -1..1) for the demo
    camera (eye on +z looking toward -z). Returns (origin, direction). The bridge from a cursor position to a ray
    the intersect functions consume -- so 'the user clicked here' becomes a geometry query."""
    o = np.asarray(cam_eye, float)
    d = np.array([float(screen_u), float(screen_v), float(cam_z)])
    d = d / (np.linalg.norm(d) + 1e-12)
    return o.tolist(), d.tolist()


def pick_mesh(mesh, screen_u, screen_v, cam_eye=(0.0, 0.0, 3.0), cam_z=-1.6, want="face"):
    """VIEWPORT PICK on a real mesh: from a cursor (screen_u, screen_v in -1..1), build the ray and return what the
    user clicked -- the nearest 'face', or the nearest 'vertex' of the hit face (by barycentric), as a hit record.
    This is the B4 generalization: pick_element (framebudget) works on the demo wireframe cage; pick_mesh works on
    a USER's arbitrary mesh via the ray query, with the same {kind, index, position} shape. None on a miss.

    Composes screen_ray + ray_mesh_intersect so a front end has one call from 'clicked here' to 'selected this'."""
    o, d = screen_ray(screen_u, screen_v, cam_eye=cam_eye, cam_z=cam_z)
    hit = ray_mesh_intersect(mesh, o, d)
    if hit is None:
        return {"kind": want, "index": None}
    if want == "face":
        return {"kind": "face", "index": hit["face"], "position": hit["position"], "distance": hit["distance"]}
    if want == "vertex":
        # the vertex of the hit triangle with the largest barycentric weight is the nearest corner to the click.
        tri = hit["triangle"]; bary = hit["barycentric"]
        vi = tri[int(np.argmax(bary))]
        return {"kind": "vertex", "index": int(vi), "position": np.asarray(mesh["vertices"])[vi].tolist(),
                "distance": hit["distance"]}
    raise ValueError("want must be 'face' or 'vertex'; got %r" % want)


def _selftest():
    """Contracts:
    1. ray-vs-mesh hits a known triangle at the known distance; a miss returns None; the AABB broad phase rejects
       an off-target ray.
    2. ray-vs-SDF sphere-traces to the surface of a unit sphere at the expected distance, with an outward normal.
    3. screen_ray points from the eye through the cursor.
    """
    # a unit quad in the z=0 plane, facing +z.
    mesh = {"vertices": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], "faces": [[0, 1, 2, 3]]}
    hit = ray_mesh_intersect(mesh, [0, 0, 5], [0, 0, -1])       # straight down the z axis
    assert hit is not None and hit["face"] == 0
    assert abs(hit["distance"] - 5.0) < 1e-6, hit["distance"]
    assert abs(hit["position"][2]) < 1e-6                       # hits the z=0 plane
    # a ray that misses the quad entirely -> None (broad phase or triangle test rejects it).
    assert ray_mesh_intersect(mesh, [5, 5, 5], [0, 0, -1]) is None
    # a ray pointing away -> None.
    assert ray_mesh_intersect(mesh, [0, 0, 5], [0, 0, 1]) is None

    # ray-vs-SDF: a unit sphere at the origin, ray from z=3 toward -z hits at z=1 (distance 2).
    def sphere(p):
        return float(np.linalg.norm(np.asarray(p, float)) - 1.0)
    sh = ray_sdf_intersect(sphere, [0, 0, 3], [0, 0, -1])
    assert sh is not None and abs(sh["distance"] - 2.0) < 1e-2, sh
    assert sh["normal"][2] > 0.9                                # normal points back toward the camera (+z)
    assert ray_sdf_intersect(sphere, [0, 5, 3], [0, 0, -1]) is None  # misses the sphere

    # screen_ray: centre points straight ahead-ish, corner points off-axis.
    o, dctr = screen_ray(0.0, 0.0)
    o2, dcorner = screen_ray(0.5, 0.5)
    assert o == [0.0, 0.0, 3.0] and dctr[2] < 0                 # from the eye, looking -z
    assert dcorner[0] > 0 and dcorner[1] > 0                    # a +/+ cursor aims +/+ 

    # pick_mesh: click the centre of a quad facing the camera -> hits face 0; vertex mode returns a corner.
    quad = {"vertices": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], "faces": [[0, 1, 2, 3]]}
    pf = pick_mesh(quad, 0.0, 0.0)
    assert pf["kind"] == "face" and pf["index"] == 0
    pv = pick_mesh(quad, -0.5, -0.5, want="vertex")
    assert pv["kind"] == "vertex" and pv["index"] in (0, 1, 2, 3)
    assert pick_mesh(quad, 0.95, 0.95)["index"] is None or pick_mesh(quad, 5.0, 5.0)["index"] is None  # off-target miss

    print("holographic_raypick selftest OK (ray-vs-mesh Moller-Trumbore hits face 0 at dist 5, AABB broad phase "
          "rejects a miss, back-ray returns None; ray-vs-SDF sphere-traces a unit sphere to dist 2 with an outward "
          "+z normal; screen_ray maps a cursor to a world ray; pick_mesh composes them to click real geometry; "
          "deterministic)")


if __name__ == "__main__":
    _selftest()

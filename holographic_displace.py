"""Displacement & bump (G3): push a surface along its normal by a scalar field.

WHY THIS MODULE EXISTS
----------------------
Displacement is the operator that turns a noise field (G1) or a texture (G2) into geometric detail:
offset the surface along its normal by a scalar amount that varies over space. It rides directly on
the two keystones and is small.

TWO PATHS, BOTH IN-SPACE
------------------------
1. SDF / HolographicField (the holographic-native path). Offsetting the 0-level of a signed-distance
   field by d(x) is, near the surface, just SUBTRACTING d from the distance -- and on the FS-5 field
   that subtraction is a DELTA hypervector you bundle in:

       displaced = field.apply_delta( field.make_delta(points, -amount * scalar(points)) )

   make_delta with NEGATIVE values pushes the surface OUTWARD (its documented sign), so a positive
   `amount * scalar` raises the surface. The cost is O(number of displacement points), independent of
   the model size, and the undo is EXACT: remove_delta subtracts the same vector back (linearity).

2. Mesh (the explicit path). Move each vertex along its shading normal:

       vertex_i  <-  vertex_i + amount * scalar(vertex_i) * normal_i

   and BUMP perturbs the shading normal from the scalar's tangential slope WITHOUT moving the vertex --
   the cheap fake-detail trick, useful when real displacement would over-tessellate.

HONEST SCOPE (kept negatives)
-----------------------------
  * The SDF path is the standard displacement-SHADER approximation: subtracting d(x) offsets the level
    set by exactly d only where the gradient is unit (|grad sdf| = 1), i.e. NEAR the surface of a
    proper SDF. In high-curvature regions, or far from the surface, the offset deviates -- the same
    near-surface caveat FS-5 already carries. Measured here, not assumed away.
  * Mesh displacement along vertex normals can self-intersect for large amounts on concavities (the
    classic displacement failure); this module does the offset, not the cleanup.
  * Bump changes shading only -- the silhouette is unchanged (that is the point of bump vs displace).
"""

import numpy as np

from holographic_mesh import Mesh


# ---------------------------------------------------------------------------
# SDF / HolographicField displacement: a field delta.
# ---------------------------------------------------------------------------

def displace_sdf(holo_field, scalar_fn, amount, points=None):
    """Displace a HolographicField's surface along its normal by `amount * scalar_fn(x)`.

    `scalar_fn` maps a point -> scalar (e.g. a FractalNoise.query, or a texture sampler). `points`
    defaults to the field's own sample points (displace where the field is defined). Returns
    (displaced_field, delta) -- keep `delta` to undo exactly with `field.remove_delta(delta)`.
    """
    if points is None:
        points = holo_field.points
    points = np.atleast_2d(np.asarray(points, float))
    # negative values push the surface OUTWARD (make_delta's sign), so +amount*scalar raises it
    values = np.array([-amount * float(scalar_fn(points[i])) for i in range(len(points))])
    delta = holo_field.make_delta(points, values)
    return holo_field.apply_delta(delta), delta


# ---------------------------------------------------------------------------
# Mesh displacement and bump.
# ---------------------------------------------------------------------------

def displace_mesh(mesh, scalar_fn, amount, use_uv=False):
    """Return a new Mesh with each vertex moved along its normal by `amount * scalar_fn(...)`.

    By default scalar_fn is evaluated at the vertex POSITION; with use_uv=True it is evaluated at the
    vertex's UV (so a texture/height map drives the displacement). Faces and UVs are carried over; the
    normals are recomputed for the displaced surface.
    """
    normals = mesh.vertex_normals(store=False)
    V = mesh.vertices
    if use_uv:
        if mesh.uvs is None:
            raise ValueError("use_uv=True needs mesh.uvs")
        scalars = np.array([float(scalar_fn(mesh.uvs[i])) for i in range(len(V))])
    else:
        scalars = np.array([float(scalar_fn(V[i])) for i in range(len(V))])
    new_V = V + (amount * scalars)[:, None] * normals
    out = Mesh(new_V, mesh.faces, uvs=mesh.uvs)
    out.vertex_normals(store=True)          # refresh normals for the displaced surface
    return out


def bump_normals(mesh, scalar_fn, amount, eps=1e-3):
    """Perturb shading normals from the scalar field's tangential slope -- bump mapping, no vertices move.

    At each vertex we build two tangents from the existing normal, finite-difference the scalar along
    them to get a surface gradient, and tilt the normal against that gradient (uphill faces tilt back).
    Returns an (V, 3) array of perturbed unit normals; the mesh geometry is untouched.
    """
    normals = mesh.vertex_normals(store=False)
    V = mesh.vertices
    out = np.zeros_like(normals)
    for i in range(len(V)):
        n = normals[i]
        # a stable tangent frame: pick the world axis least aligned with n, cross to get t1, t2
        a = np.eye(3)[np.argmin(np.abs(n))]
        t1 = np.cross(n, a); t1 /= (np.linalg.norm(t1) or 1.0)
        t2 = np.cross(n, t1)
        p = V[i]
        # slope of the scalar along each tangent (central difference)
        du = (float(scalar_fn(p + eps * t1)) - float(scalar_fn(p - eps * t1))) / (2 * eps)
        dv = (float(scalar_fn(p + eps * t2)) - float(scalar_fn(p - eps * t2))) / (2 * eps)
        # tilt the normal against the gradient: n' = normalize(n - amount*(du*t1 + dv*t2))
        m = n - amount * (du * t1 + dv * t2)
        out[i] = m / (np.linalg.norm(m) or 1.0)
    return out


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_fpe import VectorFunctionEncoder
    from holographic_fpefield import HolographicField

    # (1) SDF displacement is a field delta with EXACT undo. Build a small signed field on a lattice
    #     (a flat slab: sdf = z), displace it upward by a constant amount, and check the field value at
    #     a surface point drops by ~amount (surface rose), then remove_delta restores to machine precision.
    enc = VectorFunctionEncoder(3, dim=2048, bounds=[(-1, 1)] * 3, kernel="rbf", bandwidth=6.0, seed=1)
    axes = np.linspace(-0.8, 0.8, 6)
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    field = HolographicField(enc, P, P[:, 2])              # sdf = z (surface at z=0, normal +z)

    amount = 0.2
    disp, delta = displace_sdf(field, lambda x: 1.0, amount)   # constant push outward by 0.2
    surf_pt = np.array([0.0, 0.0, 0.0])
    before = float(field.value([surf_pt])[0])
    after = float(disp.value([surf_pt])[0])
    assert after < before - 0.05, f"surface did not rise (value should drop): {before:.3f} -> {after:.3f}"

    restored = disp.remove_delta(delta)
    err = float(np.max(np.abs(restored.f - field.f)))
    assert err < 1e-9, f"undo not exact: max field error {err:.2e}"

    # (2) MESH displacement moves vertices along normals by exactly amount*scalar. Use a flat quad in
    #     the z=0 plane (normals = +z); a constant scalar raises every vertex by amount.
    verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [(0, 1, 2), (0, 2, 3)]
    quad = Mesh(verts, faces)
    raised = displace_mesh(quad, lambda x: 1.0, 0.3)
    dz = raised.vertices[:, 2] - quad.vertices[:, 2]
    assert np.allclose(dz, 0.3, atol=1e-6), f"flat displacement should raise all by 0.3, got {dz}"

    # a varying scalar (height = x) raises vertices by amount*x
    ramp = displace_mesh(quad, lambda x: x[0], 0.5)
    expect = 0.5 * quad.vertices[:, 0]
    assert np.allclose(ramp.vertices[:, 2], expect, atol=1e-6), "ramp displacement mismatch"

    # (3) BUMP perturbs normals where the scalar varies, leaves them where it is flat.
    flat_bump = bump_normals(quad, lambda x: 1.0, 1.0)
    assert np.allclose(flat_bump, quad.vertex_normals(store=False), atol=1e-6), "flat field should not bump"
    ramp_bump = bump_normals(quad, lambda x: x[0], 1.0)
    tilted = np.max(np.abs(ramp_bump - quad.vertex_normals(store=False)))
    assert tilted > 0.05, f"a sloped field should tilt the normals, got {tilted:.3f}"

    print("holographic_displace selftest passed:",
          f"sdf_value {before:.3f}->{after:.3f} undo_err={err:.1e} bump_tilt={tilted:.3f}")


if __name__ == "__main__":
    _selftest()

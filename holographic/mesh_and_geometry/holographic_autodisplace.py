"""holographic_autodisplace.py -- DISPLACEMENT FROM A CONFIDENT HEIGHT (inverse-rendering IR5).

A bump map (IR1) only tilts SHADING normals -- the silhouette and the grazing-angle profile stay flat. For a hero
surface you sometimes want REAL relief: actually move the geometry. IR5 promotes a HIGH-CONFIDENCE height map (from
IR1) from a bump to geometry, by pure reuse of the shipped `displace` operator -- offset each vertex along its
normal by amount * height(uv). The only genuinely-new code is the WIRING and the CONFIDENCE GATE.

The gate is the point. Real geometry is expensive and destructive, so a shaky height must NOT deform a mesh: IR5
only displaces when the IR1 bump-confidence clears a (stricter than the bump) threshold, and otherwise ABSTAINS,
returning the mesh untouched. So a flat/featureless image leaves the surface flat rather than crumpling it.

KEPT NEGATIVE (loud): displacement is only as good as the height estimate, so it inherits ALL of IR1's ambiguities
(a cast shadow becomes a real groove; a painted stripe becomes a real ridge). It also adds real geometry cost. It
is gated on confidence precisely because these failure modes are worse when they move vertices than when they only
tilt a normal. NumPy + stdlib only; deterministic.
"""
import numpy as np


def _bilinear(height_map, u, v):
    """Bilinearly sample a height map at uv in [0,1]^2 (u across the width, v down the height)."""
    H, W = height_map.shape[0], height_map.shape[1]
    x = np.clip(u * (W - 1), 0.0, W - 1)
    y = np.clip(v * (H - 1), 0.0, H - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    fx, fy = x - x0, y - y0
    return float((1 - fx) * (1 - fy) * height_map[y0, x0] + fx * (1 - fy) * height_map[y0, x1]
                 + (1 - fx) * fy * height_map[y1, x0] + fx * fy * height_map[y1, x1])


def _planar_uvs(mesh):
    """uv from the XY bounding box of the vertices (a planar projection) -- gives a flat surface without uvs a
    natural texture coordinate so the height map can drive it."""
    xy = np.asarray(mesh.vertices, float)[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return (xy - lo) / span


def displace_from_height(mesh, height_map, amount=0.1, confidence=None, min_confidence=0.02):
    """Promote a HEIGHT MAP to real geometry: move each vertex along its normal by amount * height(uv), via the
    shipped displace_mesh. GATED on `confidence`: if it is below `min_confidence`, ABSTAIN and return the mesh
    UNCHANGED (a shaky height must not deform a mesh). Uses the mesh's own uvs if present, else a planar XY
    projection. Returns (mesh, abstained)."""
    if confidence is not None and confidence < min_confidence:
        return mesh, True
    from holographic.misc.holographic_displace import displace_mesh
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    m = mesh
    if m.uvs is None:
        m = Mesh(m.vertices, m.faces, uvs=_planar_uvs(m))    # planar uvs for a flat surface
    sampler = lambda uv: _bilinear(height_map, float(uv[0]), float(uv[1]))
    return displace_mesh(m, sampler, amount, use_uv=True), False


def auto_displace(mesh, rgb, amount=0.1, sigma=4.0, min_confidence=0.02):
    """Full auto-displacement: auto-bump the image to a height + confidence (IR1), then displace the mesh by that
    height IF it is confident enough for geometry (a stricter gate than a shading bump), else return the mesh
    UNCHANGED. Returns (mesh, info) where info carries auto_bump's height/confidence/abstained plus `displaced`."""
    from holographic.mesh_and_geometry.holographic_autobump import auto_bump
    res = auto_bump(rgb, sigma=sigma)
    if res["abstained"] or res["confidence"] < min_confidence:
        return mesh, {**res, "displaced": False}
    out, abstained = displace_from_height(mesh, res["height"], amount=amount,
                                          confidence=res["confidence"], min_confidence=min_confidence)
    return out, {**res, "displaced": not abstained}


def field_displace(mesh, field, amount=0.1, weight=None, invert=False, bias=0.0):
    """Displace a mesh's vertices along their normals by a SCALAR FIELD or SDF sampled AT EACH VERTEX -- the general,
    field-driven modifier (auto_displace only reads an RGB image; this reads any field, so a FRACTAL can drive the
    displacement). `field` is anything with `.eval(P)` (an SDF like mandelbulb/fold_fractal) or a bare callable
    P(N,3)->values(N,). Each vertex v moves along its normal by `amount * (field(v) - bias) * weight[v]`.

    `weight` is the per-face / per-vertex MASK -- a per-vertex array (or a callable P->w) in [0,1], e.g. sampled from
    a texture map, so the fractal detail only grows WHERE THE MAP PAINTS IT (the 'per-face mandelbulb modifier from a
    texture' case). `invert` flips the field sign; `bias` recenters it (subtract the field's mean to displace both
    in and out). Returns a NEW Mesh; the input is unchanged.

    KEPT NEGATIVE: this displaces along EXISTING normals -- it adds surface relief, it does not re-topologize, so very
    high `amount` self-intersects (as any displacement modifier does); mesh finely first if you want deep fractal
    detail (the field is defined everywhere, the resolution is the base mesh's)."""
    import numpy as _np
    from holographic.mesh_and_geometry.holographic_mesh import Mesh

    V = _np.asarray(mesh.vertices, float)
    sample = (lambda P: field.eval(P)) if hasattr(field, "eval") else field
    h = _np.asarray(sample(V), float).reshape(-1)                # the field value at every vertex
    if invert:
        h = -h
    h = h - bias
    if weight is not None:
        w = weight(V) if callable(weight) else _np.asarray(weight, float).reshape(-1)
        h = h * w                                               # the texture / mask gates the displacement per vertex

    # per-vertex normals: average the incident triangle face normals (the direction each vertex moves).
    N = _np.zeros_like(V)
    for (i, j, k) in mesh.triangulate():
        fn = _np.cross(V[j] - V[i], V[k] - V[i])
        nrm = _np.linalg.norm(fn)
        if nrm > 1e-12:
            fn = fn / nrm
            N[i] += fn; N[j] += fn; N[k] += fn
    lens = _np.linalg.norm(N, axis=1, keepdims=True)
    N = _np.where(lens > 1e-12, N / _np.maximum(lens, 1e-12), 0.0)

    V2 = V + N * (amount * h)[:, None]
    return Mesh(V2, [tuple(f) for f in mesh.faces],
                uvs=mesh.uvs if getattr(mesh, "uvs", None) is not None else None)


def _selftest():
    """A confident bumpy height moves a flat grid's vertices in z (real relief, following the height); a low-
    confidence height ABSTAINS and leaves the mesh untouched; auto_displace applies on a bumpy image and abstains
    on a flat one; deterministic."""
    from holographic.mesh_and_geometry.holographic_mesh import grid

    N = 24
    plane = grid(nx=N, ny=N, width=1.0, height=1.0)          # a flat NxN grid in z=0, no uvs
    z0 = plane.vertices[:, 2].copy()
    assert np.allclose(z0, 0.0)                              # starts flat

    # a bumpy height map (a raised centre bump)
    yy, xx = np.mgrid[0:N + 1, 0:N + 1]
    hmap = np.exp(-(((xx - N / 2) / (N / 5)) ** 2 + ((yy - N / 2) / (N / 5)) ** 2))   # gaussian bump in [0,1]

    # (1) confident -> displace: the grid gains relief, the centre rises most
    disp, abstained = displace_from_height(plane, hmap, amount=0.3, confidence=0.2, min_confidence=0.02)
    assert not abstained
    z = disp.vertices[:, 2]
    assert z.max() > 0.2                                     # the surface actually rose
    # the centre vertex (highest height) rose more than a corner (near-zero height)
    center_i = int(np.argmin(np.linalg.norm(plane.vertices[:, :2], axis=1)))
    corner_i = int(np.argmax(np.linalg.norm(plane.vertices[:, :2], axis=1)))
    assert z[center_i] > z[corner_i] + 0.1                   # relief follows the height map

    # (2) low confidence -> ABSTAIN, mesh untouched
    same, abstained2 = displace_from_height(plane, hmap, amount=0.3, confidence=0.005, min_confidence=0.02)
    assert abstained2 and np.allclose(same.vertices[:, 2], 0.0)   # not deformed

    # (3) auto_displace: a bumpy image displaces; a flat image does not
    Ni = 48
    u = np.linspace(0, 6 * np.pi, Ni)
    bump = 0.5 + 0.4 * np.outer(np.sin(u), np.cos(u))
    bump_rgb = np.stack([bump, bump, bump], axis=-1)
    _, info = auto_displace(grid(nx=20, ny=20), bump_rgb, amount=0.2, min_confidence=0.02)
    assert info["displaced"]
    _, info_flat = auto_displace(grid(nx=20, ny=20), np.full((Ni, Ni, 3), 0.5), amount=0.2)
    assert not info_flat["displaced"]                        # flat image -> no geometry

    # (4) deterministic
    d1, _ = displace_from_height(plane, hmap, amount=0.3, confidence=0.2)
    d2, _ = displace_from_height(plane, hmap, amount=0.3, confidence=0.2)
    assert np.array_equal(d1.vertices, d2.vertices)

    # (5) FIELD_DISPLACE: a fractal SDF drives displacement, and a per-vertex mask gates it (per-face texture case).
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    sph_mesh = grid(nx=N, ny=N, width=2.0, height=2.0)
    fld = sphere(0.5)                                          # any .eval field; a sphere SDF is a clean test signal
    out = field_displace(sph_mesh, fld, amount=0.5, bias=0.0)
    moved = np.linalg.norm(out.vertices - sph_mesh.vertices, axis=1)
    assert moved.max() > 0.05, "field_displace moves vertices by the field value at each vertex"
    # a MASK that is 1 on the left half and 0 on the right must displace ONLY the left half (per-face texture gating)
    mask = (sph_mesh.vertices[:, 0] < 0.0).astype(float)
    masked = field_displace(sph_mesh, fld, amount=0.5, weight=mask)
    left = sph_mesh.vertices[:, 0] < 0.0
    dmoved = np.linalg.norm(masked.vertices - sph_mesh.vertices, axis=1)
    assert dmoved[left].max() > 0.01 and dmoved[~left].max() < 1e-9, \
        "a per-vertex mask gates the displacement (the fractal detail grows only where the map paints it)"
    assert np.array_equal(field_displace(sph_mesh, fld, amount=0.5).vertices,
                          field_displace(sph_mesh, fld, amount=0.5).vertices), "field_displace deterministic"

    print("holographic_autodisplace selftest OK: a confident bumpy height moves a flat grid's vertices into real "
          "relief (max z %.2f, centre rises more than a corner, following the height); a low-confidence height "
          "ABSTAINS and leaves the mesh flat; auto_displace applies on a bumpy image and abstains on a flat one; "
          "field_displace drives displacement from an SDF field and a per-vertex mask gates it (left half only); "
          "deterministic" % float(disp.vertices[:, 2].max()))


if __name__ == "__main__":
    _selftest()

"""Vertex / texel attribute channel (G6): carry per-vertex data as a FIELD, not a fixed array.

WHY THIS MODULE EXISTS
----------------------
A Mesh ships positions, normals, uvs, and colours, but no GENERAL per-vertex/per-texel attribute
channel (a scalar mask, a second colour set, a weight map, a custom signal). This is the plumbing that
adds one -- and adds it the holographic way, so the attribute is RESOLUTION-INDEPENDENT.

THE IDEA
--------
A per-vertex attribute is a function over the surface domain (UV, or position). Encoded as an FPE field
(exactly G2's `texture_field`), it is a continuous function you can sample at ANY resolution: bake it to
a coarse mesh's vertices, subdivide, and re-bake -- the new vertices get consistent values from the
SAME field, because the field never changed, only the sample points densified. A fixed per-vertex array
cannot do that; it has to be re-interpolated by hand every time the topology changes.

Two layers:
  * `attribute_field` / `sample_attribute` / `bake_to_vertices` -- the holographic, resolution-free path.
  * `attach_attribute` / `get_attribute` -- a light RASTER store (a `.data` dict on the mesh) for when a
    plain per-vertex array is what you want. Additive and backward-compatible: it never touches the Mesh
    constructor, just sets an attribute dict on demand.

HONEST SCOPE (kept negatives)
-----------------------------
  * The field path inherits G2's band-limit: smooth attributes interpolate beautifully; a hard-edged
    mask (0/1 boundary) comes back smoothed. Use the raster store for hard masks.
  * The raster store is exactly as resolution-bound as any vertex array -- it is the convenience path,
    not the holographic one. The two coexist on purpose.
"""

import numpy as np

from holographic_material import texture_field, sample_texture


# ---------------------------------------------------------------------------
# Holographic attribute: a resolution-independent field over the surface domain.
# ---------------------------------------------------------------------------

def attribute_field(encoder, points, values, weights=None):
    """A per-vertex/texel attribute as an FPE field over the surface domain (UV or position).

    Same construction as a texture (`encoder.bundle(points, values)`); named separately because the
    INTENT is a data channel rather than appearance. `encoder` matches the domain dimension (2 for UV,
    3 for position).
    """
    return texture_field(encoder, points, values, weights=weights)


def sample_attribute(encoder, field, point):
    """Read the attribute at any point in the domain -- the resolution-free sample."""
    return sample_texture(encoder, field, point)


def bake_to_vertices(encoder, field, sample_points):
    """Sample an attribute field at a set of points (e.g. mesh vertices or UVs) -> a value array.

    Because `field` is a continuous function, baking it at a COARSE point set and at a DENSER one are
    consistent: shared points get the same value either way (that is the resolution independence).
    """
    pts = np.atleast_2d(np.asarray(sample_points, float))
    return encoder.query_many(field, pts)


# ---------------------------------------------------------------------------
# Raster attribute store: a light additive .data dict on a mesh.
# ---------------------------------------------------------------------------

def attach_attribute(mesh, name, values):
    """Attach a named per-vertex raster attribute to a mesh (additive; creates mesh.data on first use)."""
    values = np.asarray(values, float)
    if values.shape[0] != mesh.n_vertices:
        raise ValueError(f"attribute '{name}' needs one value per vertex ({mesh.n_vertices}), got {values.shape[0]}")
    if not hasattr(mesh, "data") or mesh.data is None:
        mesh.data = {}
    mesh.data[name] = values
    return mesh


def get_attribute(mesh, name, default=None):
    """Fetch a named raster attribute, or `default` if absent."""
    data = getattr(mesh, "data", None) or {}
    return data.get(name, default)


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_fpe import VectorFunctionEncoder
    from holographic_mesh import box

    # (1) ATTRIBUTE field round-trip over UV: a ramp attribute reads back tracking u.
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    field = attribute_field(enc, grid, [u for (u, v) in grid])
    us = np.linspace(0.2, 0.8, 20)
    read = np.array([sample_attribute(enc, field, [u, 0.5]) for u in us])
    assert np.corrcoef(read, us)[0, 1] > 0.95, "attribute field should track its values"

    # (2) RESOLUTION INDEPENDENCE: sampling the SAME field at a coarse set and at a 2x-denser set agrees
    #     at the shared points -- the field is a function, so densifying the sample changes nothing there.
    coarse = np.array([[u, 0.5] for u in np.linspace(0.2, 0.8, 7)])
    dense = np.array([[u, 0.5] for u in np.linspace(0.2, 0.8, 13)])      # includes the coarse points
    cv = bake_to_vertices(enc, field, coarse)
    dv = bake_to_vertices(enc, field, dense)
    # the coarse points are dense[0,2,4,...]; their values must match the coarse bake exactly
    assert np.allclose(cv, dv[::2], atol=1e-9), "shared points must agree across resolutions"
    # and the in-between (midpoint) samples are smooth -- bounded between their neighbours, no blow-up
    mids = dv[1::2]
    lo = np.minimum(dv[:-1:2], dv[2::2]); hi = np.maximum(dv[:-1:2], dv[2::2])
    assert np.all(mids >= lo - 0.05) and np.all(mids <= hi + 0.05), "midpoints should interpolate smoothly"

    # (3) RASTER store: attach/get round-trips and rejects a wrong-length array.
    cube = box(1, 1, 1)
    attach_attribute(cube, "wear", np.linspace(0, 1, cube.n_vertices))
    got = get_attribute(cube, "wear")
    assert got is not None and len(got) == cube.n_vertices, "raster attribute did not round-trip"
    assert get_attribute(cube, "missing", default=-1) == -1, "missing attribute should return default"
    try:
        attach_attribute(cube, "bad", np.zeros(cube.n_vertices + 3))
        raised = False
    except ValueError:
        raised = True
    assert raised, "wrong-length attribute should raise"

    print("holographic_attributes selftest passed:",
          f"ramp_corr={np.corrcoef(read, us)[0,1]:.3f} res_indep_max_gap={np.max(np.abs(cv - dv[::2])):.2e} "
          f"raster_len={len(got)}")


if __name__ == "__main__":
    _selftest()

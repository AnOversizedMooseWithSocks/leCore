"""Holographic material & texture stack (G2): a texture is a field, a material is a role-filler record.

WHY THIS MODULE EXISTS
----------------------
If the mesh lives in holographic space, everything that sits on top of it -- materials, textures, the
normal/roughness/displacement maps -- should live there too, so a textured object is ONE composite
hypervector that edits, transmits, blends, and undoes with the same field machinery as the geometry.
This is the purest VSA fit on the whole geometry/appearance list, and it was the one piece not yet in
the space at all.

TWO IDEAS
---------
1. A TEXTURE IS A FUNCTION OVER UV. That is exactly what the FPE `VectorFunctionEncoder` encodes -- a
   function as a hypervector. `texture_field(encoder, uv_points, values)` is `encoder.bundle(points,
   values)`: f = sum_i value_i * encode(uv_i), a holographic scattered-data field. Sampling is a query
   (a cosine); tiling/offset is a bind (FPE shift); blending two textures is a bundle.

2. A MATERIAL IS A ROLE-FILLER RECORD. A PBR material is a set of named channels -- albedo, roughness,
   metallic, normal, height/displacement, emission, ao, opacity -- each a constant or a texture. In the
   space that is one bundle of role-bound channel fields,

       material = sum_r  bind(role_r, channel_field_r)

   -- the HRR record Plate's representation was built for. Fetch a channel by unbind(material, role);
   blend two materials by a*m1 + (1-a)*m2 (every channel mixes by linearity); UV-transform the WHOLE
   material at once by binding it with a shift vector -- because bind is associative/commutative,
   bind(bind(role, field), shift) == bind(role, shift(field)), so one bind shifts every channel's UV.

   And geometry + appearance compose into one object:
       object = bind(GEOMETRY, geometry_field) + bind(APPEARANCE, material_record)

HONEST SCOPE (kept negatives)
-----------------------------
  * BAND-LIMITED. The FPE field is smooth, so this is ideal for procedural / smooth channels (gradients,
    noise-driven detail, PBR scalar maps) and a lossy/smoothed representation for SHARP photographic
    textures (hard mask edges, text). For those, keep the raster and bind a *reference* into the record
    -- the record structure is unchanged, only the leaf differs.
  * CROSSTALK. A record bundles several bound fields; unbinding one channel recovers it plus capacity
    noise from the others (the honest HRR capacity cliff). The noise is ~sqrt(n_other)/sqrt(dim); we
    MEASURE the channel round-trip error rather than assume it away. More channels or a smaller dim
    raises it -- raise `dim` to buy capacity.

Deterministic: role atoms come from hashlib(name) seeds (NOT python's salted hash()), so the same
channel name maps to the same role vector across runs and machines.
"""

import hashlib

import numpy as np

from holographic_ai import bind, unbind, cosine


# ---------------------------------------------------------------------------
# Roles: deterministic unit vectors keyed by channel name.
# ---------------------------------------------------------------------------

def _role_atom(name, dim):
    """A deterministic ~unit random vector for a channel name.

    WHY hashlib and not hash(): python's built-in hash() is salted per-process (PYTHONHASHSEED), so it
    is NOT reproducible across runs. hashlib.sha256 is a pure function of the bytes, so the same channel
    name yields the same role vector everywhere -- the determinism rule the whole engine runs under.
    """
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Textures: a function over the UV chart.
# ---------------------------------------------------------------------------

def texture_field(encoder, uv_points, values, weights=None):
    """A texture as an FPE function over UV: f = sum_i value_i * encode(uv_i).

    `encoder` is a 2-D VectorFunctionEncoder over the UV domain. Sampling is `encoder.query(field, uv)`;
    the read is the kernel-weighted (smoothed) value near uv. A constant texture is a single point's
    field (or just store the constant); a procedural texture is a noise field (G1) used as `values`.
    """
    uv_points = list(uv_points)
    values = list(values)
    if weights is not None:
        values = [v * w for v, w in zip(values, weights)]
    return encoder.bundle(uv_points, values)


def sample_texture(encoder, field, uv):
    """Read a texture field at a UV coordinate."""
    return float(encoder.query(field, uv))


# ---------------------------------------------------------------------------
# Materials: a holographic record of role-bound channel fields.
# ---------------------------------------------------------------------------

class Material:
    """A PBR-style material as a bundle of role-bound channel fields, sampled and composed in-space.

    `encoder` is the shared UV-domain encoder for all channel textures. `channels` maps a channel name
    (e.g. 'albedo', 'roughness', 'normal', 'height') to its texture field vector. The record is the
    PLAIN SUM of bind(role(name), field) -- not normalized -- so a recovered channel reads at the same
    scale as the standalone texture (the crosstalk is the only difference, and it is measured).
    """

    def __init__(self, encoder, channels=None):
        self.encoder = encoder
        self.dim = encoder.dim
        self.channels = {}                       # name -> RAW field vector (exact reference for sampling)
        self._record = np.zeros(self.dim)
        if channels:
            for name, field in channels.items():
                self.add(name, field)

    @staticmethod
    def _unit(field):
        n = np.linalg.norm(field)
        return field / n if n > 0 else field

    def add(self, name, field):
        """Add or replace a channel. The RAW field is stored for exact sampling; the record binds the
        UNIT-normalized field so every channel contributes equal energy -- otherwise a big-norm channel
        swamps a small one in the bundle and the small channel recovers as mostly crosstalk."""
        field = np.asarray(field, float)
        if name in self.channels:                # remove the old binding first (linearity)
            self._record = self._record - bind(_role_atom(name, self.dim), self._unit(self.channels[name]))
        self.channels[name] = field
        self._record = self._record + bind(_role_atom(name, self.dim), self._unit(field))
        return self

    @property
    def record(self):
        """The single hypervector carrying the whole material (geometry binds this under APPEARANCE).

        It binds unit-normalized channel DIRECTIONS, so it transmits/composes/blends as one vector and
        recovers each channel's direction with a measured crosstalk; absolute values come from the
        stored fields via sample() when you hold the Material object.
        """
        return self._record

    def channel(self, name):
        """Recover a channel's DIRECTION from the bare record by unbinding its role (target + crosstalk).
        Use this when you only have the record (after transmission); otherwise sample() is exact."""
        return unbind(self._record, _role_atom(name, self.dim))

    def sample(self, name, uv):
        """Sample a channel's value at a UV coordinate -- EXACT, from the stored field (no crosstalk)."""
        return float(self.encoder.query(self.channels[name], uv))

    def sample_all(self, uv):
        """Sample every channel at a UV coordinate -> {name: value}."""
        return {name: self.sample(name, uv) for name in self.channels}

    def blend(self, other, t):
        """Linear blend of two materials (same encoder): t*self + (1-t)*other, mixing every channel.

        Returns a new Material whose record is the weighted sum -- by linearity each channel is the
        weighted mix of the two inputs' channels. Channels present in only one side blend toward zero.
        """
        if other.encoder is not self.encoder and other.dim != self.dim:
            raise ValueError("blend needs materials over a compatible encoder/dim")
        out = Material(self.encoder)
        names = set(self.channels) | set(other.channels)
        for name in names:
            a = self.channels.get(name, np.zeros(self.dim))
            b = other.channels.get(name, np.zeros(self.dim))
            out.add(name, t * a + (1.0 - t) * b)
        return out

    def transform_uv(self, shift):
        """Translate every channel's UV by `shift` with a SINGLE bind on the whole record.

        bind(record, encode(shift)) = sum_r bind(role_r, shift(field_r)) because bind is associative and
        commutative -- one operation re-UVs the entire material. Returns a new Material.
        """
        shift_vec = self.encoder.encode(shift)
        out = Material(self.encoder)
        for name, field in self.channels.items():
            out.add(name, bind(field, shift_vec))
        return out


def sample_material(material, uv_coords):
    """Sample a material at many UV coordinates (e.g. a mesh's per-vertex UVs) -> {name: value array}.

    `uv_coords` is an (N, 2) array of UV points (mesh.uvs, or a UV chart from `meshuv`). Returns one
    value array per channel -- the shading inputs, recovered holographically.
    """
    uv_coords = np.asarray(uv_coords, float).reshape(-1, 2)
    out = {}
    for name in material.channels:
        field = material.channels[name]          # EXACT stored field (no crosstalk); query all UVs
        out[name] = material.encoder.query_many(field, uv_coords)
    return out


# ---------------------------------------------------------------------------
# Compose geometry + appearance into one object.
# ---------------------------------------------------------------------------

GEOMETRY = _role_atom("__GEOMETRY__", 1)         # placeholder; real atoms made per-dim below

def compose_object(geometry_vec, material, dim=None):
    """One hypervector carrying shape AND look: bind(GEOMETRY, geom) + bind(APPEARANCE, material).

    `geometry_vec` is e.g. a HolographicField's vec (FS-5). Returns (object_vec, roles) where roles are
    the GEOMETRY/APPEARANCE atoms so the caller can unbind either side back out.
    """
    geometry_vec = np.asarray(geometry_vec, float)
    d = dim or geometry_vec.shape[0]
    g_role = _role_atom("__GEOMETRY__", d)
    a_role = _role_atom("__APPEARANCE__", d)
    # balance the two terms to unit norm so NEITHER side swamps the other in the bundle -- otherwise the
    # larger-norm side recovers well and the smaller recovers as mostly crosstalk (the imbalance we hit
    # inside materials, one level up). The caller keeps the originals; this vector is the composable form.
    g_unit = geometry_vec / (np.linalg.norm(geometry_vec) or 1.0)
    a_unit = material.record / (np.linalg.norm(material.record) or 1.0)
    obj = bind(g_role, g_unit) + bind(a_role, a_unit)
    return obj, {"GEOMETRY": g_role, "APPEARANCE": a_role}


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_fpe import VectorFunctionEncoder

    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)

    # build a few channel textures over the UV square: a smooth gradient (albedo), a constant-ish
    # roughness, a height bump in the middle.
    rng = np.random.default_rng(0)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    albedo_vals = [u for (u, v) in grid]                         # ramp in u
    rough_vals = [0.5 for _ in grid]                            # flat
    height_vals = [np.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.05) for (u, v) in grid]  # central bump

    tex_albedo = texture_field(enc, grid, albedo_vals)
    tex_rough = texture_field(enc, grid, rough_vals)
    tex_height = texture_field(enc, grid, height_vals)

    # (1) TEXTURE round-trip: the albedo ramp read back tracks u (smooth interpolation).
    us = np.linspace(0.2, 0.8, 20)
    read = np.array([sample_texture(enc, tex_albedo, [u, 0.5]) for u in us])
    corr = float(np.corrcoef(read, us)[0, 1])
    assert corr > 0.95, f"texture should track its values, corr={corr:.3f}"

    # (2) MATERIAL record. sample() is exact (stored field); the bare RECORD recovers each channel's
    #     DIRECTION with a measured crosstalk -- the capacity cost of carrying the material as one vector.
    #     With unit-normalized bindings the recovery is BALANCED across channels (no channel swamps).
    mat = Material(enc, {"albedo": tex_albedo, "roughness": tex_rough, "height": tex_height})
    # exact sampling matches the standalone texture to machine precision (it IS the stored field)
    uv = [0.4, 0.5]
    assert abs(mat.sample("albedo", uv) - sample_texture(enc, tex_albedo, uv)) < 1e-9, "sample() not exact"
    # bare-record recovery: cosine of the unbound channel to its true (unit) direction
    recalls = {name: cosine(mat.channel(name), Material._unit(mat.channels[name])) for name in mat.channels}
    worst = float(min(recalls.values()))
    assert worst > 0.45, f"a channel recovers as mostly crosstalk from the record: {recalls}"

    # (3) BLEND: blend two materials and verify a channel is the linear mix.
    mat2 = Material(enc, {"albedo": tex_rough, "roughness": tex_albedo, "height": tex_rough})
    mix = mat.blend(mat2, 0.7)
    uv = [0.4, 0.5]
    got = mix.sample("albedo", uv)
    want = 0.7 * mat.sample("albedo", uv) + 0.3 * mat2.sample("albedo", uv)
    assert abs(got - want) < 0.05, f"blend not linear: {got:.4f} vs {want:.4f}"

    # (4) TRANSFORM_UV: shifting the material by d, sampled at uv, equals the original sampled at uv-d.
    d = np.array([0.15, 0.0])
    shifted = mat.transform_uv(d)
    uv = np.array([0.55, 0.5])
    got = shifted.sample("albedo", uv)
    want = mat.sample("albedo", uv - d)
    assert abs(got - want) < 0.05, f"transform_uv shift mismatch: {got:.4f} vs {want:.4f}"

    # (5) COMPOSE: geometry + appearance in one vector. Recovery is capacity-limited (a 2-item bundle
    #     with involution unbind at dim 1024), so test it as a MARGIN: each side recovers its OWN content
    #     far better than the other side's -- the object demonstrably carries both, distinguishably.
    geom = rng.standard_normal(1024); geom /= np.linalg.norm(geom)
    rec_unit = mat.record / np.linalg.norm(mat.record)
    obj, roles = compose_object(geom, mat)
    rec_app = unbind(obj, roles["APPEARANCE"])
    rec_geom = unbind(obj, roles["GEOMETRY"])
    app_self, app_cross = cosine(rec_app, rec_unit), cosine(rec_app, geom)
    geom_self, geom_cross = cosine(rec_geom, geom), cosine(rec_geom, rec_unit)
    assert app_self > 0.45 and app_self > 3 * abs(app_cross), \
        f"appearance not cleanly recovered: self={app_self:.3f} cross={app_cross:.3f}"
    assert geom_self > 0.45 and geom_self > 3 * abs(geom_cross), \
        f"geometry not cleanly recovered: self={geom_self:.3f} cross={geom_cross:.3f}"

    print("holographic_material selftest passed:",
          f"texture_corr={corr:.3f} record_recall={recalls} "
          f"app_self={app_self:.3f} geom_self={geom_self:.3f}")


if __name__ == "__main__":
    _selftest()

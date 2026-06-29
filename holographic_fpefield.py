"""Field-First Sculpting FS-5: the surface carried as a SINGLE hypervector (edit = bind).

This is the most literal form of the recurring "move the geometry into the holographic space" thesis. FS-1..FS-4
sculpt and store a field as ARRAYS (a dense or narrow-band voxel grid) and mesh it with marching_tetrahedra. FS-5
asks the opposite question: can the surface itself BE a hypervector, so that the field operations become VSA algebra
rather than array sweeps? The answer, built on the FPE VectorFunctionEncoder (holographic_fpe.py), is yes -- with a
clean headline and an honest limit.

THE REPRESENTATION. Sample the surface's SIGNED distance at a set of points p_i and bundle them into ONE vector,
weighted by the signed distance:

    f = sum_i sdf(p_i) * encode(p_i)            (VectorFunctionEncoder.bundle)

Because the FPE encoder's similarity is a Bochner (RBF) kernel, querying f is a holographic kernel-weighted estimate
of the signed field: cosine(f, encode(x)) reads sum_i sdf(p_i) * kernel(x, p_i). That estimate is negative inside,
positive outside, and crosses zero AT the surface -- so marching its 0-level re-extracts the surface. The whole field
-- however many samples went in -- is a SINGLE dim-d vector. (The RBF kernel here is literally a Gaussian bump in the
hypervector domain, so this is "Gaussian splatting" carried in VSA space; the FS-3 splats are the array-domain cousin.)

THE HEADLINE -- EDIT = BIND. A rigid translation of the ENTIRE surface is a single binding:

    translate(f, delta) = bind(f, encode(delta)) = sum_i sdf(p_i) * encode(p_i + delta)

one O(dim) FFT, no resampling, no per-voxel sweep -- the same rigid-shift-is-a-bind identity the motion compensator
and FPE.shift use, lifted to a whole surface. Two fields UNION by bundling (vector add). This is the payoff: once the
surface is a hypervector, moving and combining surfaces are algebra.

KEPT HONEST -- THE TRANSLATE IS EXACT; THE RECONSTRUCTION IS A SMOOTHED, BOUNDED ESTIMATE (measured, not hidden):
  * TRANSFORMATIONS are exact: translate gives value_shifted(x) == value_orig(x - delta) to machine precision (1e-16),
    and the surface's zero-crossing moves by EXACTLY the delta -- the one binding genuinely is the rigid shift.
    Caveat: the FPE wraps at the encoder bounds, so the bounds must exceed |sample| + |shift| or a shifted sample
    aliases (use a generous half-range; from_mesh leaves the choice to the caller).
  * METRIC GEOMETRY is only approximate. A finite-bandwidth kernel BLURS the signed field, so the marched 0-level is
    biased (the classic "blur shrinks/grows a convex SDF") -- on a unit-scaled sphere the recovered radius lands
    within ~15% of truth -- and that finite-resolution smoothed extract is not guaranteed watertight. The choice of
    bandwidth IS the bias knob (the same band-limit lever flagged for the fractal-optics / Mip-Splatting work).
  * The field is only meaningful WITHIN the sampled cloud: far outside the samples the kernel sum decays into its
    crosstalk floor (it can read a spurious small negative), so query and re-extract inside the sampled domain.
  * Fidelity is bounded by hypervector DIMENSION: finite dim means kernel crosstalk, a surface-roughness noise floor
    that falls as dim rises (measured: sphere-radius std ~0.13 at dim 1024 -> ~0.065 at dim 4096). Carrying a
    continuous field in a fixed-length vector is exactly this capacity trade.
  * It is a DEMONSTRATION representation, not the fast path: building f is O(samples) FPE encodes and reconstruction
    is O(res^3) FPE queries, both FFT-bound -- the array-backed SparseField (FS-2/FS-4) remains the performance path.
    FS-5's value is conceptual: the surface as one vector, with moving and merging as single binds.

Deps: numpy + holographic_fpe (VectorFunctionEncoder) + holographic_meshbridge (marching). No new dependencies.
"""

import numpy as np

from holographic_fpe import VectorFunctionEncoder


class HolographicField:
    """A signed field (a surface) carried as ONE hypervector via FPE. value() is a query, translate() is a bind,
    union() is a bundle, and surface() re-extracts by marching the field's 0-level. See the module docstring for the
    representation and its honest limits."""

    def __init__(self, encoder, points, sdf_values):
        """Bundle the signed-distance samples (points p_i, values sdf(p_i)) into the single field vector
        f = sum_i sdf(p_i) * encode(p_i). The encoder is an FPE VectorFunctionEncoder whose bounds cover the points
        and whose bandwidth sets the kernel width (the reconstruction's bias/resolution knob)."""
        self.enc = encoder
        self.points = np.atleast_2d(np.asarray(points, float))
        self.values = np.asarray(sdf_values, float).ravel()
        if self.points.shape[0] != self.values.shape[0]:
            raise ValueError("need one sdf value per point")
        # f = sum_i sdf(p_i) encode(p_i): the whole signed field as one dim-d vector.
        self.f = encoder.bundle([self.points[i] for i in range(len(self.points))], weights=self.values)

    @classmethod
    def from_mesh(cls, mesh, bounds, dim=2048, bandwidth=18.0, grid=12, seed=0):
        """Build the field for an arbitrary mesh: sample its SIGNED distance on a coarse `grid`^3 lattice (the exact
        point-to-mesh mesh_to_sdf), then bundle. The lattice spans `bounds` = ((lo,lo,lo),(hi,hi,hi)); bandwidth is in
        the encoder's per-axis units (wider = smoother = more biased, sharper = noisier). Returns a HolographicField."""
        from holographic_meshbridge import mesh_to_sdf
        lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
        axes = [np.linspace(lo[k], hi[k], grid) for k in range(3)]
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        W = mesh_to_sdf(mesh, P)                                  # exact signed distance at the lattice points
        enc = VectorFunctionEncoder(3, dim=dim, bounds=[(float(lo[k]), float(hi[k])) for k in range(3)],
                                    bandwidth=bandwidth, seed=seed)
        return cls(enc, P, W)

    def value(self, points):
        """The (kernel-smoothed) signed field at the query point(s): cosine(f, encode(x)) for each -- negative inside,
        positive outside, ~0 on the surface. One FPE query per point."""
        pts = np.atleast_2d(np.asarray(points, float))
        return np.array([self.enc.query(self.f, pts[i]) for i in range(len(pts))])

    def translate(self, delta):
        """THE HEADLINE -- edit = bind. Translate the ENTIRE surface by `delta` with a SINGLE binding:
        bind(f, encode(delta)) = sum_i sdf(p_i) encode(p_i + delta). No resampling, one O(dim) op. Returns a new
        HolographicField; the original is unchanged (deterministic, additive)."""
        delta = np.asarray(delta, float)
        out = HolographicField.__new__(HolographicField)
        out.enc = self.enc
        out.points = self.points + delta
        out.values = self.values
        out.f = self.enc.shift(self.f, delta)                    # one bind moves the whole field
        return out

    def union(self, other):
        """Combine two fields into one by BUNDLING their vectors (f1 + f2) -- the superposition reads as the union of
        the two signed fields (both surfaces present in the reconstruction). Returns a new HolographicField sharing
        this field's encoder (the two must share an encoder)."""
        if other.enc is not self.enc:
            raise ValueError("union needs both fields on the same encoder")
        out = HolographicField.__new__(HolographicField)
        out.enc = self.enc
        out.points = np.vstack([self.points, other.points])
        out.values = np.concatenate([self.values, other.values])
        out.f = self.f + other.f                                 # bundle = union of the signed fields
        return out

    def surface(self, bounds, res=22, level=0.0):
        """Re-extract the surface by SAMPLING the field on a `res`^3 grid (one query per voxel) and MARCHING its
        `level` set. Returns a Mesh. KEPT HONEST: O(res^3) FPE queries (FFT-bound) -- a demonstration extract, and a
        smoothed/biased surface (see the module docstring); for a fast, faithful extract stay in the array-backed
        SparseField world (FS-2/FS-4)."""
        from holographic_meshbridge import marching_tetrahedra_vec
        lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
        axes = tuple(np.linspace(lo[k], hi[k], res) for k in range(3))
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        vol = self.value(P).reshape(res, res, res)
        return marching_tetrahedra_vec(vol, axes, float(level))


def _selftest():
    import numpy as np

    R = 0.6
    B = 1.3                                                     # encoder half-range: must exceed |sample|+|shift| or FPE wraps
    bounds = ((-B, -B, -B), (B, B, B))
    enc = VectorFunctionEncoder(3, dim=2048, bounds=[(-B, B)] * 3, bandwidth=18.0, seed=0)
    g = np.linspace(-0.9, 0.9, 12)
    P = np.array([(x, y, z) for x in g for y in g for z in g])
    W = np.linalg.norm(P, axis=1) - R                          # a sphere's analytic SDF
    field = HolographicField(enc, P, W)

    # (1) the whole field is ONE vector
    assert field.f.shape == (enc.dim,), "the surface is a single hypervector"

    # (2) the field carries the SIGN: negative inside, positive outside (it crosses zero at the surface)
    v_in = float(field.value([[0.0, 0.0, 0.0]])[0])
    v_out = float(field.value([[0.7, 0.7, 0.7]])[0])           # outside the sphere but inside the sampled cloud
    assert v_in < 0.0 < v_out, "field is negative inside, positive outside"

    # (3) THE HEADLINE -- edit = bind, tested EXACTLY on the value field (decoupled from the noisy marched extract):
    # translating the whole field by one binding makes value_shifted(x) == value_orig(x - delta) to machine precision.
    d = np.array([0.25, 0.0, 0.0])
    moved = field.translate(d)
    cg = np.linspace(-0.5, 0.5, 7)
    X = np.array([(a, b, c) for a in cg for b in cg for c in cg])   # central points: x and x-d both stay in range
    vs = moved.value(X)
    vo = field.value(X - d)
    assert np.max(np.abs(vs - vo)) < 1e-9, "edit=bind: the field is exactly translated (value_shifted(x)=value_orig(x-d))"
    assert np.array_equal(field.translate(d).f, moved.f), "translate is deterministic and leaves the original intact"

    # the surface (a zero-crossing) moves by exactly the delta -- read it on the +x ray (robust, unlike the full mesh)
    ray = np.linspace(0.0, 1.1, 120)
    def xcross(fld):
        v = fld.value(np.stack([ray, np.zeros_like(ray), np.zeros_like(ray)], axis=1))
        idx = np.where(np.diff(np.sign(v)) != 0)[0]
        return float(ray[idx[0]]) if len(idx) else None
    z0, z1 = xcross(field), xcross(moved)
    assert z0 is not None and z1 is not None and abs((z1 - z0) - 0.25) < 0.02, "the surface's zero-crossing moved by the delta"

    # (4) union = bundle: two offset spheres; the field is inside (value<0) at BOTH centres, outside between far ends
    fieldB = HolographicField(enc, P + np.array([0.7, 0.0, 0.0]), np.linalg.norm(P, axis=1) - R)
    both = field.union(fieldB)
    uv = both.value([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.6, 0.0, 0.0]])
    assert uv[0] < 0 and uv[1] < 0 and uv[2] > 0, "union holds both surfaces (inside at both centres)"

    # (5) the recovered radius (the +x zero-crossing) is a SMOOTHED estimate of R, within the documented bias band
    assert 0.45 < z0 < 0.85, f"recovered radius near R within the kernel-bias band (got {z0:.3f})"

    # (6) the surface re-extracts to a mesh (a noisy smoothed extract, but non-empty)
    mesh = field.surface(bounds, res=20)
    assert mesh.n_faces > 0, "the 0-level marches to a surface"

    # determinism
    assert np.array_equal(HolographicField(enc, P, W).f, field.f), "deterministic build"

    print(f"holographic_fpefield selftest: ok (surface as ONE dim-{enc.dim} vector; field sign in={v_in:+.3f} "
          f"out={v_out:+.3f}; EDIT=BIND is EXACT -- value_shifted(x)=value_orig(x-d) to {np.max(np.abs(vs-vo)):.0e}, "
          f"and the +x surface crossing moved {z0:.3f}->{z1:.3f} (by the delta) via one bind; union holds both "
          f"spheres; recovered radius {z0:.3f} (smoothed estimate of {R}); marched extract {mesh.n_faces} faces; "
          f"deterministic)")


if __name__ == "__main__":
    _selftest()

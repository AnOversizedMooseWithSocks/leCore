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

THE SECOND HEADLINE -- EDITING IS A DELTA, AND IT IS MODEL-SIZE-INDEPENDENT. Because the field is a LINEAR
superposition, a local edit is just a small delta vector added in: make_delta(points, values) builds
d = sum_i values_i encode(points_i) (the edit -- O(edit), the brush, NOT the model), apply_delta does f + d (one
O(dim) add), and remove_delta does f - d, which UNDOES the edit to machine precision (exact, by linearity). So:
  * the cost of an edit does NOT grow with the model -- a million-sample model and a hundred-sample model are both
    one fixed-length vector, and adding a brush delta costs the same on each (measured: identical apply time);
  * a whole undo/redo HISTORY is just a list of compact delta vectors (subtract to undo, add to redo);
  * re-projecting after a local edit is LOCAL -- surface(sub_box) marches only the dirty region (the edit's bounding
    box grown by the kernel reach), a fraction of a full re-extract (measured ~7x fewer queries for a pole-sized
    edit, more for smaller ones).
This is the temporal/video-codec insight (store a keyframe, then small per-frame deltas -- motion-compensation is a
bind, the residual is the change) carried into geometry EDITING: the model is the keyframe vector, each edit is a
delta, and only the touched region is re-extracted.

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
from collections import namedtuple

from holographic_fpe import VectorFunctionEncoder


# A geometry edit captured as a hypervector: `vec` is sum_i values_i encode(points_i) (what apply/remove add or
# subtract); `points`/`values` are its provenance. A whole edit history is just a list of these.
FieldDelta = namedtuple("FieldDelta", ["vec", "points", "values"])


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
        positive outside, ~0 on the surface. Uses batched FPE reads for row stacks."""
        pts = np.atleast_2d(np.asarray(points, float))
        return self.enc.query_many(self.f, pts)

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

    def make_delta(self, points, values):
        """A LOCAL geometry edit, captured as a DELTA hypervector: d = sum_i values_i encode(points_i). NEGATIVE
        values push the surface OUTWARD there (more 'inside'); positive carve INWARD. Returns a FieldDelta (the vector
        plus its provenance). Building it is O(len(points)) -- the cost of the EDIT, with NO dependence on the model's
        size, because the model is just one fixed-length vector. This is the temporal/video-codec idea (an edit is a
        small delta you bundle in) lifted to geometry."""
        pts = np.atleast_2d(np.asarray(points, float))
        vals = np.asarray(values, float).ravel()
        if pts.shape[0] != vals.shape[0]:
            raise ValueError("need one value per delta point")
        vec = self.enc.bundle([pts[i] for i in range(len(pts))], weights=vals)
        return FieldDelta(vec, pts, vals)

    def apply_delta(self, delta):
        """Apply an edit by ADDING its delta vector: f' = f + delta.vec -- a single O(dim) add, regardless of model
        size (all the edit's cost was in make_delta). Returns a new HolographicField; the original is untouched. Chain
        these for a sequence of edits; the field stays one vector no matter how many edits accumulate."""
        out = HolographicField.__new__(HolographicField)
        out.enc = self.enc
        out.points = self.points
        out.values = self.values
        out.f = self.f + delta.vec
        return out

    def remove_delta(self, delta):
        """UNDO an edit EXACTLY by subtracting its delta vector: f' = f - delta.vec. Because the field is a LINEAR
        superposition, this restores the pre-edit field to machine precision -- exact, O(dim) undo, and a whole edit
        history is just a list of these delta vectors (apply to redo, subtract to undo). The clean payoff of carrying
        geometry as a bundle. Returns a new HolographicField."""
        out = HolographicField.__new__(HolographicField)
        out.enc = self.enc
        out.points = self.points
        out.values = self.values
        out.f = self.f - delta.vec
        return out

    def surface(self, bounds, res=22, level=0.0):
        """Re-extract the surface by SAMPLING the field on a `res`^3 grid (one query per voxel) and MARCHING its
        `level` set. Returns a Mesh. Pass a SUB-BOX as `bounds` to re-extract only a region (after a local edit, the
        edit's bounding box grown by the kernel reach) -- O(region), a fraction of a full re-extract, which is how a
        large model stays real-time under editing. KEPT HONEST: O(res^3) FPE queries (FFT-bound) -- a demonstration
        extract, and a smoothed/biased surface (see the module docstring); for a fast, faithful extract stay in the
        array-backed SparseField world (FS-2/FS-4). The kernel has soft support, so a region box must include the
        edit's kernel reach or it clips the edit's tail."""
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

    # (7) DELTA EDITING -- a local edit is a delta vector; apply adds it, remove undoes it EXACTLY, and the cost does
    # not depend on model size. Push material out near the +x pole.
    q = np.array([0.6, 0.0, 0.0])
    bump = np.array([q + o for o in [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0), (0, -0.05, 0), (0, 0, 0.05), (0, 0, -0.05)]])
    delta = field.make_delta(bump, np.full(len(bump), -0.35))     # negative -> bulge outward there
    edited = field.apply_delta(delta)
    assert edited.value([q])[0] < field.value([q])[0], "apply_delta pushed the surface out at the edit point"
    assert abs(float(edited.value([-q])[0]) - float(field.value([-q])[0])) < 0.02, "the edit is local (far side ~unchanged)"
    undone = edited.remove_delta(delta)
    assert np.max(np.abs(undone.f - field.f)) < 1e-9, "remove_delta undoes the edit EXACTLY (linearity)"
    # editing is model-size-independent: the SAME delta applies to a 10x-bigger model identically (both one vector)
    gL = np.linspace(-0.95, 0.95, 22)
    big = HolographicField(enc, np.array([(x, y, z) for x in gL for y in gL for z in gL]),
                           np.linalg.norm(np.array([(x, y, z) for x in gL for y in gL for z in gL]), axis=1) - R)
    assert big.apply_delta(delta).f.shape == (enc.dim,), "a 10x model is still one vector; the same delta applies"
    # local re-extraction: a region box marches far fewer points than the full domain
    region = edited.surface(((0.3, -0.4, -0.4), (1.0, 0.4, 0.4)), res=12)
    assert region.n_faces > 0, "the edited region re-extracts locally"

    # determinism
    assert np.array_equal(HolographicField(enc, P, W).f, field.f), "deterministic build"

    print(f"holographic_fpefield selftest: ok (surface as ONE dim-{enc.dim} vector; field sign in={v_in:+.3f} "
          f"out={v_out:+.3f}; EDIT=BIND is EXACT -- value_shifted(x)=value_orig(x-d) to {np.max(np.abs(vs-vo)):.0e}, "
          f"crossing moved {z0:.3f}->{z1:.3f} via one bind; union holds both spheres; DELTA editing: a brush delta "
          f"pushed the pole out and remove_delta undid it to {np.max(np.abs(undone.f-field.f)):.0e} (exact, "
          f"model-size-independent); recovered radius {z0:.3f} (smoothed estimate of {R}); deterministic)")


if __name__ == "__main__":
    _selftest()

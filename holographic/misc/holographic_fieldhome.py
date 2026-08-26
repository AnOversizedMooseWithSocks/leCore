"""holographic_fieldhome.py -- the FIELD home (consolidation backlog R2): one interface over the engine's several
SPATIAL field representations, backend chosen by cost.

WHY THIS EXISTS
---------------
The engine grew several ways to hold a scalar/vector field over space, each good for a different cost regime:
a DENSE grid (holographic_fields, cheap random access, memory ~ volume), a NARROW-BAND sparse voxel field
(holographic_sparsefield, cost ~ surface area), a CALLABLE/analytic oracle (an SDF expression, an FPE surface),
and more (spectral, region, dirty). They all answer the SAME question -- "what is the field at these points?" --
but each with its own call. A caller that wants to sample a field shouldn't have to know which representation it
got.

`Field` is that one interface: `field.sample(points) -> values`, over R^D, no matter the backend. It is a thin
ADAPTER that ROUTES to each representation's OWN sampler -- it reimplements no field, and the reps stay distinct
(the consolidation golden rule: route, don't rewrite). Construct one with the backend-specific classmethods
(`Field.grid`, `Field.sparse`, `Field.callable`); consume it uniformly with `.sample`.

NOT to be confused with holographic_field.Field, which is a DIFFERENT abstraction: a compositional field over the
UNIT vector space (it normalises points, for value-functions / compasses / memory-density). This home is for
SPATIAL fields over ordinary R^D coordinates (grids, voxels, SDFs) -- the reps the render/sim stack sits on.
"""
import numpy as np


class Field:
    """One interface over a spatial field: `sample(points) -> values`. `kind` names the backend ('dense', 'sparse',
    'callable', ...); `backend` is the wrapped object (a grid array, a SparseField, a callable). Build with the
    classmethods below; each wires `sample` to the backend's own sampler."""

    def __init__(self, sampler, kind, backend=None):
        self._sampler = sampler                                   # callable: points (N,D) -> values (N,) or (N,C)
        self.kind = str(kind)
        self.backend = backend

    def sample(self, points):
        """Evaluate the field at `points` (N,D). Delegates to the backend's native sampler."""
        return self._sampler(np.asarray(points, float))

    def __repr__(self):
        return "Field(kind=%r)" % self.kind

    # --- backend constructors (ROUTE to each representation's own sampler) ---

    @classmethod
    def callable(cls, fn):
        """An analytic / oracle field: any function points (N,D) -> values. The most basic backend -- an SDF
        expression tree, an FPE surface read, or a closed-form field all fit here."""
        return cls(lambda P: np.asarray(fn(P), float), "callable", backend=fn)

    @classmethod
    def grid(cls, array, lo, hi):
        """A DENSE grid field (holographic_fields): a regular (Nx,Ny[,Nz]) array over the box [lo, hi], sampled by
        the engine's trilinear/bilinear periodic reader. Maps world points into grid-index space and delegates to
        holographic_fields.sample_field / sample_field_3d -- no new interpolation here."""
        arr = np.asarray(array, float)
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        span = np.where((hi - lo) == 0, 1.0, hi - lo)
        shape = np.asarray(arr.shape[:arr.ndim], float)
        from holographic.misc.holographic_fields import sample_field, sample_field_3d
        reader = sample_field_3d if arr.ndim == 3 else sample_field

        def s(points):
            P = np.asarray(points, float)
            idx = (P - lo) / span * shape                         # world -> grid index coordinates
            return reader(arr, idx)
        return cls(s, "dense", backend=arr)

    @classmethod
    def low_rank(cls, array, lo, hi, max_error=None, rank=None):
        """A 2-D dense grid held in FACTORED form (holographic_tucker.LowRankField): the grid is never materialised,
        and sampling reconstructs only the rows and columns a query touches. Backlog B2.

        `max_error` is an ABSOLUTE max-abs reconstruction budget and is the sizing you want -- `rank_gate`'s 99%
        ENERGY criterion can leave a 7-28% residual, which an SDF consumer cannot survive (a sphere trace overshoots
        a surface that is 7% wrong). MEASURED on real 128x128 fields at a 1%-of-amplitude budget:

            field                rank    factored bytes   vs dense (131,072)
            sphere SDF slice        4             8,224           16x   -- pays
            box SDF slice          12            24,672          5.3x   -- pays
            irradiance (1/r^2)      4             8,224           16x   -- pays
            fbm noise (4 oct)      50           102,800          1.27x  -- marginal, in practice no
            white noise           124                --           --    -- refused

        RAISES if `worth_factoring` says the factors would not be smaller than the dense array. That refusal is the
        gate, and it is why this is a constructor rather than a silent optimisation: a caller must ask for it and
        be told no.

        TWO CAVEATS, measured:
          * `sample` is NEAREST, not the bilinear reader `Field.grid` uses. At grid NODES it reproduces the rank-r
            reconstruction exactly (4.4e-16). Off-node, against an analytic sphere SDF, low_rank(nearest) is 3.39%
            of amplitude against grid(bilinear)'s 2.35% -- both dominated by the 0.031 grid spacing, and the extra
            1% is the SAMPLER, not the factorization.
          * This pays where a field is BAKED ONCE and QUERIED MANY TIMES. Factoring a STREAMED array costs an SVD:
            measured 67.5 ms per 128x128 frame against 1.26 ms for the FFT blur it would accelerate (53.7x), and
            91.7x at 256x256. A stored field, not a stream."""
        from holographic.caching_and_storage.holographic_tucker import LowRankField
        arr = np.asarray(array, float)
        if arr.ndim != 2:
            raise ValueError("Field.low_rank is 2-D; use Field.grid for 3-D (LowRankField is a matrix factorization)")
        ok, fb, db = LowRankField.worth_factoring(arr, max_error=max_error)
        if not ok:
            raise ValueError("low_rank refused: factors would take %d bytes against the dense array's %d. "
                             "This field is not low rank at the requested accuracy." % (fb, db))
        lf = LowRankField.from_dense(arr, rank=rank, max_error=max_error)
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        span = np.where((hi - lo) == 0, 1.0, hi - lo)
        shape = np.asarray(arr.shape, float)

        def s(points):
            # NEAREST sample from the factors: U[i] . (S * V[j]). Touches 3r numbers per query, never n*m.
            P = np.asarray(points, float)[:, :2]
            idx = np.rint((P - lo[:2]) / span[:2] * shape).astype(int)
            i = np.clip(idx[:, 0], 0, arr.shape[0] - 1)
            j = np.clip(idx[:, 1], 0, arr.shape[1] - 1)
            return np.einsum("nr,nr->n", lf.U[i], lf.S * lf.V[j])
        return cls(s, "low_rank", backend=lf)

    @classmethod
    def sparse(cls, sparse_field):
        """A NARROW-BAND field (holographic_sparsefield.SparseField): cost scales with surface area, not volume.
        Delegates straight to its own `.sample(points)` (values are the clamped signed distance inside the band,
        sign*band outside)."""
        return cls(lambda P: sparse_field.sample(P), "sparse", backend=sparse_field)


def field_backends():
    """The backend kinds this home can wrap today (route targets), for the catalog / discovery."""
    return ("callable", "dense", "sparse", "low_rank")


def _selftest():
    from holographic.misc.holographic_fields import scatter_to_field_3d           # to build a dense grid from an oracle
    # an oracle field: a smooth blob, f(x,y,z) = 1 - dist to a centre
    centre = np.array([0.0, 0.0, 0.0])

    def oracle(P):
        return 1.0 - np.linalg.norm(np.asarray(P, float) - centre, axis=1)

    lo = np.array([-1.0, -1.0, -1.0]); hi = np.array([1.0, 1.0, 1.0])
    N = 16
    # sample the oracle at NODE-aligned world points X_i = lo + i/N*(hi-lo), so array index i holds oracle(X_i) --
    # this matches how sample_field_3d reads the grid (integer index coord i == array node i, periodic).
    axis = [lo[d] + np.arange(N) / N * (hi[d] - lo[d]) for d in range(3)]
    gx, gy, gz = np.meshgrid(axis[0], axis[1], axis[2], indexing="ij")
    grid = oracle(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)).reshape(N, N, N)

    f_dense = Field.grid(grid, lo, hi)
    f_call = Field.callable(oracle)

    # TWO BACKENDS, IDENTICAL VALUES: at the grid-node world points, the dense reader returns the stored oracle
    # value (trilinear at a node = the node), so dense == callable there.
    probe = np.array([[axis[0][2], axis[1][5], axis[2][9]],
                      [axis[0][7], axis[1][1], axis[2][3]],
                      [axis[0][10], axis[1][12], axis[2][6]]])
    a = f_dense.sample(probe)
    b = f_call.sample(probe)
    assert np.allclose(a, b, atol=1e-9), (a, b)                    # identical at nodes -> two backends agree

    # the sample interface is uniform regardless of backend
    for f in (f_dense, f_call):
        v = f.sample(np.zeros((5, 3)))
        assert v.shape == (5,) and np.isfinite(v).all()
    print("OK: holographic_fieldhome self-test passed (dense & callable backends agree to %.1e at grid nodes; "
          "one .sample interface over %s)" % (float(np.max(np.abs(a - b))), ", ".join(field_backends())))


if __name__ == "__main__":
    _selftest()

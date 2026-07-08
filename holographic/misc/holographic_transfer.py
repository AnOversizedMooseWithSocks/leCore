"""holographic_transfer.py -- KERNEL SCATTER / GATHER: the ONE bundle/readout under every particle<->grid transfer.

The snow-MPM work surfaced that P2G (scatter particles onto a grid) IS bundling and G2P (gather back) IS the
readout. Probing the stack, that pattern is NOT MPM's alone -- the same local-kernel scatter/gather is written out
by hand in several modules, each with its own kernel:

  * holographic_fields:  scatter_to_field / sample_field  -- BILINEAR, for cloth<->fluid two-way coupling
                         (its own docstring already calls scatter_to_field "the ADJOINT of sample_field");
  * holographic_mpm:     P2G / G2P                          -- quadratic B-SPLINE, the material-point transfer;
  * holographic_splat / holographic_kde:                    -- the SAME bundle with a GLOBAL (Gaussian) kernel.

So this is the §5.1 "generalize on contact" case: the operation lives in >=3 places, so extract it ONCE and make
the call sites thin. The operation, stated in the engine's own terms:

  SCATTER (the bundle):  grid_node = SUM over points of  weight(node, point) * value(point)
      -- a superposition of kernel-weighted, position-bound contributions. Deposit == bundle.
  GATHER  (the readout): value(point) = SUM over nodes of  weight(node, point) * grid_node
      -- reading the bundle back at a position. Sample == readout.

They are ADJOINT (transposes of one linear operator: <scatter(v), f> = <v, gather(f)>), and a partition-of-unity
kernel (weights sum to 1) makes scatter preserve the total -- a NORMALIZED bundle, which is why P2G conserves
mass and a fluid deposit conserves momentum. _selftest VERIFIES that this primitive reproduces fields' bilinear
pair AND MPM's B-spline P2G to machine precision -- proof they are one operation, not three.

Kept honest: this unifies the LOCAL compact-support transfers (bilinear, B-spline). The Gaussian splat/KDE are the
same bundle at the GLOBAL end of the support spectrum (every node gets a contribution) -- recognised here, but left
in their own modules because a global kernel has a different cost structure than a 3-node stencil. NumPy + stdlib
only; deterministic (pure linear algebra, no RNG).
"""
import itertools

import numpy as np


def _bilinear(frac):
    """2-node linear weights for fractional position `frac` in [0,1): nodes {0, 1}. Returns (..., 2)."""
    return np.stack([1.0 - frac, frac], axis=-1)


def _bspline(frac):
    """3-node quadratic B-spline weights for `frac` in [0.5, 1.5): nodes {0, 1, 2}. The MPM kernel; a smooth
    partition-of-unity bump. Returns (..., 3)."""
    return np.stack([0.5 * (1.5 - frac) ** 2, 0.75 - (frac - 1.0) ** 2, 0.5 * (frac - 0.5) ** 2], axis=-1)


# each kernel: (weight_fn, n_nodes, base_shift) -- the stencil low corner is floor(point - base_shift)
_KERNELS = {
    "bilinear": (_bilinear, 2, 0.0),
    "bspline": (_bspline, 3, 0.5),
}


def _stencil(points, kernel):
    """The per-point stencil: the low-corner node `base` (N, D), the per-axis weights `w` (N, D, n_nodes), and the
    node count. Shared by scatter and gather so the SAME kernel is used both ways (the adjoint property needs it)."""
    wfn, nnode, shift = _KERNELS[kernel]
    base = np.floor(points - shift).astype(int)
    frac = points - base
    return base, wfn(frac), nnode


def scatter(points, values, shape, kernel="bilinear", periodic=False):
    """SCATTER = the BUNDLE: deposit each point's value onto the grid through the kernel. `points` (N, D) in
    grid-cell units; `values` (N,) or (N, C); returns a grid of `shape` (plus a trailing C for vector values).
    Because a partition-of-unity kernel's weights sum to 1, the total is preserved (a normalized bundle)."""
    points = np.atleast_2d(np.asarray(points, float))
    values = np.asarray(values, float)
    D = points.shape[1]
    base, w, nnode = _stencil(points, kernel)
    vec = values.ndim == 2
    grid = np.zeros(tuple(shape) + ((values.shape[1],) if vec else ()), float)
    for combo in itertools.product(range(nnode), repeat=D):       # every node of the (nnode)^D stencil
        weight = np.ones(len(points))
        idx = []
        for d in range(D):
            weight = weight * w[:, d, combo[d]]                   # the product kernel weight to this node
            i = base[:, d] + combo[d]
            idx.append(i % shape[d] if periodic else np.clip(i, 0, shape[d] - 1))
        contrib = weight[:, None] * values if vec else weight * values
        np.add.at(grid, tuple(idx), contrib)                     # accumulate = superpose (the bundle)
    return grid


def gather(field, points, kernel="bilinear", periodic=False):
    """GATHER = the READOUT: read the grid back at each point through the SAME kernel -- the adjoint of scatter.
    `field` of shape S (or S + (C,) for vector); returns (N,) or (N, C)."""
    points = np.atleast_2d(np.asarray(points, float))
    field = np.asarray(field, float)
    D = points.shape[1]
    gshape = field.shape[:D]
    base, w, nnode = _stencil(points, kernel)
    vec = field.ndim > D
    out = np.zeros((len(points), field.shape[-1]) if vec else len(points))
    for combo in itertools.product(range(nnode), repeat=D):
        weight = np.ones(len(points))
        idx = []
        for d in range(D):
            weight = weight * w[:, d, combo[d]]
            i = base[:, d] + combo[d]
            idx.append(i % gshape[d] if periodic else np.clip(i, 0, gshape[d] - 1))
        node = field[tuple(idx)]
        out += weight[:, None] * node if vec else weight * node  # read the bundle back
    return out


def _selftest():
    """scatter and gather are ADJOINT; a partition-of-unity kernel preserves the total (normalized bundle); and --
    the point of the module -- this ONE primitive reproduces fields' bilinear scatter/gather AND MPM's B-spline
    P2G to machine precision, proving they are one operation."""
    rng = np.random.default_rng(0)

    # (1) ADJOINTNESS: <scatter(points, v), f> == <v, gather(f, points)> for both kernels (the transpose property)
    for kern in ("bilinear", "bspline"):
        pts = rng.uniform(3, 13, size=(40, 2))
        v = rng.standard_normal(40)
        f = rng.standard_normal((16, 16))
        lhs = float(np.sum(scatter(pts, v, (16, 16), kernel=kern) * f))
        rhs = float(np.sum(v * gather(f, pts, kernel=kern)))
        assert abs(lhs - rhs) < 1e-9, (kern, lhs, rhs)

    # (2) PARTITION OF UNITY: scattering all-ones values deposits total weight N (a normalized bundle preserves mass)
    for kern in ("bilinear", "bspline"):
        pts = rng.uniform(3, 13, size=(50, 2))
        g = scatter(pts, np.ones(50), (16, 16), kernel=kern)
        assert abs(g.sum() - 50.0) < 1e-9, kern

    # (3) UNIFICATION with holographic_fields (BILINEAR): fields uses (x=col, y=row) -> grid[y, x], periodic. Pass
    # the coords swapped to (row, col) and this primitive reproduces scatter_to_field / sample_field EXACTLY.
    from holographic.misc.holographic_fields import scatter_to_field, sample_field
    pos = rng.uniform(0, 20, size=(30, 2))                        # fields positions are (x, y)
    vals = rng.standard_normal(30)
    mine = scatter(pos[:, ::-1], vals, (20, 20), kernel="bilinear", periodic=True)   # (y, x) order
    theirs = scatter_to_field((20, 20), pos, vals)
    assert np.allclose(mine, theirs, atol=1e-12), np.abs(mine - theirs).max()
    fld = rng.standard_normal((20, 20))
    mine_g = gather(fld, pos[:, ::-1], kernel="bilinear", periodic=True)
    theirs_g = sample_field(fld, pos)
    assert np.allclose(mine_g, theirs_g, atol=1e-12), np.abs(mine_g - theirs_g).max()

    # (4) UNIFICATION with holographic_mpm (B-SPLINE): the primitive reproduces MPM's P2G mass grid EXACTLY.
    from holographic.simulation_and_physics.holographic_mpm import MPMSnow
    m = MPMSnow(grid=32, seed=0).seed_block(cx=16, cy=16, w=8, h=8, n=200)
    mine_p2g = scatter(m.x * m.inv_dx, m.m, (32, 32), kernel="bspline", periodic=False)
    assert np.allclose(mine_p2g, m.p2g_mass_grid(), atol=1e-10), np.abs(mine_p2g - m.p2g_mass_grid()).max()

    # (5) VECTOR values (momentum): scatter/gather carry channels; round-trip recovers a constant field exactly
    pts = rng.uniform(3, 13, size=(60, 2))
    mom = rng.standard_normal((60, 2))
    vgrid = scatter(pts, mom, (16, 16), kernel="bspline")
    assert vgrid.shape == (16, 16, 2)
    const = np.ones((16, 16, 2)) * np.array([2.0, -1.0])
    back = gather(const, pts, kernel="bspline")
    assert np.allclose(back, np.array([2.0, -1.0]), atol=1e-9)   # gather of a constant is that constant (PoU)

    print("holographic_transfer selftest OK: scatter (bundle) and gather (readout) are adjoint for bilinear AND "
          "B-spline; a partition-of-unity kernel preserves the total (normalized bundle); and this ONE primitive "
          "reproduces holographic_fields' bilinear scatter/gather to 1e-12 AND holographic_mpm's B-spline P2G to "
          "1e-10 -- proof the fluid coupling and the material-point transfer are the same bundle/readout operation")


if __name__ == "__main__":
    _selftest()

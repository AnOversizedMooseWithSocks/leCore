"""VSA-native tiling -- domain repetition as bind + bundle, on FPE field hypervectors.

THE IDEA. A field encoded by FPE (holographic_fpe.VectorFunctionEncoder) is a HYPERVECTOR on which a rigid
shift is a SINGLE BIND (bind(f, encode(delta)) translates the whole field, exact to 1e-16). So tiling -- the
demoscene/Quilez move of repeating a motif across space -- is just BUNDLING shifted copies, each shift one
bind. The result is itself a hypervector: composable, storable in the archive, bindable to a role, queryable.
Nothing leaves VSA space; there is no Python loop over voxels, only binds and a sum.

WHY 3-D MATTERS HERE. FPE is n-DIMENSIONAL (a point in R^n is the bind of its per-axis encodings, and a shift
on ANY axis is a bind), so tiling generalises to 3-D for free: a 3-D motif, a 3-D period, a 3-D lattice of
binds. The same periodic structure the 3-D fluid grid lives on (a torus) is what FPE encodes -- tiling and the
3-D grid are the same periodicity seen two ways.

THE COMPOUNDING (recursion / inception / compression). tile_recursive TILES THE TILING: after L levels the
field contains count^L copies per axis, built from only L*prod(counts) binds -- exponential structure from
linear work, held in ONE fixed-size vector. A 64-tile field as a single hypervector is compression; a tile of
tiles of tiles is recursion; and because every level is the same bind+bundle, it is self-similar -- as above,
so below. The honest ceiling is VSA capacity: each tile is another pattern superposed into a fixed dim, so the
SNR falls as count^L grows (kept negative, measured).

THE BRIDGE (making the physics composable). grid_to_function / function_to_grid move a NumPy field (a fluid
density, a temperature field, an SDF slice) into an FPE hypervector and back. The heavy numerics still run on
grids (the FFT solve is fastest there -- doing it in hypervector space would be slow), but once a result is a
hypervector it can be TILED, BOUND, BUNDLED, and STORED like any other VSA object. That is the principled
answer to "VSA-native vs the Python boundary": simulate on the grid, then cross into VSA ONCE to compose.

Only NumPy and the engine's bind/bundle/cosine + the existing FPE encoder -- nothing new, nothing learned.
"""

import itertools

import numpy as np

from holographic_ai import bind, cosine


def _lattice(counts):
    """All integer offsets of an n-D lattice with `counts[k]` cells on axis k."""
    return itertools.product(*[range(int(c)) for c in counts])


def tile(enc, function, period, counts):
    """Tile a field hypervector over an n-D lattice: place a shifted copy at every lattice cell and SUM them.
    Each shift is one bind (enc.shift = bind with the encoded offset), so the whole operation is binds + a sum
    and the result is a hypervector. `period` and `counts` are per-axis (scalars are broadcast). VSA-native
    domain repetition -- Quilez's mod()-tiling, but composable."""
    period = np.atleast_1d(np.asarray(period, float))
    n = enc.n_dims
    if period.shape[0] == 1 and n > 1:
        period = np.repeat(period, n)
    counts = np.atleast_1d(np.asarray(counts, int))
    if counts.shape[0] == 1 and n > 1:
        counts = np.repeat(counts, n)
    out = None
    for k in _lattice(counts):
        shifted = enc.shift(function, np.asarray(k, float) * period)   # one bind per tile
        out = shifted if out is None else out + shifted               # bundle (FPE sum convention)
    return out


def tile_recursive(enc, function, period, counts, levels):
    """Inception: tile the tiling, `levels` deep. Each level tiles the previous block at the super-period
    (period * counts), so after L levels there are count^L copies per axis -- from only L*prod(counts) binds,
    in one fixed-size vector. Returns the hypervector. (Capacity caveat: count^L patterns share one dim, so the
    SNR falls as the tiling grows -- see the kept negative in the module docstring.)"""
    period = np.atleast_1d(np.asarray(period, float))
    n = enc.n_dims
    if period.shape[0] == 1 and n > 1:
        period = np.repeat(period, n)
    counts = np.atleast_1d(np.asarray(counts, int))
    if counts.shape[0] == 1 and n > 1:
        counts = np.repeat(counts, n)
    f = function
    p = period.astype(float)
    for _ in range(int(levels)):
        f = tile(enc, f, p, counts)        # tile the current block ...
        p = p * counts                     # ... then move up to the block's own period
    return f


def fractal_bands(enc, function, base_period, levels, count=3, decay=1.0):
    """A multi-scale (fBm-like) bundle: tile the motif at base_period, base_period/2, base_period/4, ...,
    summing the levels with an optional amplitude `decay` per octave. Richness at many scales from one motif
    and a handful of binds -- the demoscene 'fractal noise' move, done in VSA."""
    out = None
    amp = 1.0
    for L in range(int(levels)):
        p = np.asarray(base_period, float) / (2 ** L)
        band = tile(enc, function, p, [count] * enc.n_dims)
        term = amp * band
        out = term if out is None else out + term
        amp *= decay
    return out


# -- the grid <-> hypervector bridge (make a NumPy field composable) --------------------------------

def grid_to_function(enc, grid, coords, threshold=1e-3):
    """Encode a NumPy field `grid` as an FPE hypervector: bundle each cell's encoded coordinate weighted by its
    value (skipping near-zero cells). `coords` is a list of the axis coordinate arrays (one per axis) the grid
    is sampled on. The field becomes a hypervector you can tile / bind / bundle / store. (Cost is one encode per
    significant cell -- the ONE crossing from grid into VSA; do it once, then stay in VSA.)"""
    grid = np.asarray(grid, float)
    out = None
    mesh = np.meshgrid(*coords, indexing="ij")
    flat_vals = grid.ravel()
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    for val, p in zip(flat_vals, pts):
        if abs(val) < threshold:
            continue
        term = float(val) * enc.encode(p)
        out = term if out is None else out + term
    if out is None:
        out = enc.encode(pts[0]) * 0.0
    return out


def function_to_grid(enc, function, coords):
    """Read an FPE function back onto a grid: query the hypervector at each grid coordinate (cosine with the
    encoded point). The inverse of grid_to_function (a holographic KDE reconstruction)."""
    mesh = np.meshgrid(*coords, indexing="ij")
    shape = mesh[0].shape
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    out = np.array([enc.query(function, p) for p in pts])
    return out.reshape(shape)


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_fpe import VectorFunctionEncoder

    # 2-D tiling: a motif's copy in each cell reads the SAME as the original (tiling exact via bind), and the
    # gaps between cells are ~0 (the motif is localized).
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
    motif = enc.encode([5.0, 5.0])
    tiled = tile(enc, motif, period=10.0, counts=3)
    q00 = enc.query(tiled, [5, 5]); q22 = enc.query(tiled, [25, 25]); gap = enc.query(tiled, [10, 10])
    assert abs(q00 - q22) < 0.02 and q00 > 0.2 and gap < 0.1, (q00, q22, gap)

    # shift-is-bind exactness -- the property tiling rests on: a translated motif queried at the shifted point
    # equals the original at the base point, to machine precision.
    s = enc.shift(motif, np.array([10.0, 0.0]))
    assert abs(enc.query(s, [15, 5]) - enc.query(motif, [5, 5])) < 1e-9

    # 3-D tiling: a 3-D motif tiled over a 3-D lattice -- a bump in each cell, gaps empty.
    enc3 = VectorFunctionEncoder(3, dim=4096, bounds=[(0, 80)] * 3, bandwidth=40.0, seed=0)
    m3 = enc3.encode([5.0, 5.0, 5.0])
    t3 = tile(enc3, m3, period=10.0, counts=3)
    assert enc3.query(t3, [25, 25, 25]) > 0.15 and enc3.query(t3, [12, 12, 12]) < 0.12

    # recursion / inception: 3 levels of count-2 tiling -> 8 cells per axis from 3*4=12 binds; the far corner
    # cell (7*period) still carries the motif.
    rec = tile_recursive(enc, motif, period=10.0, counts=2, levels=3)
    corner = enc.query(rec, [75, 75]); rgap = enc.query(rec, [10, 10])
    assert corner > 0.05 and corner > rgap, (corner, rgap)

    # bridge: a small NumPy field -> hypervector -> queried back correlates with the original.
    xs = np.linspace(0, 20, 11)
    coords = [xs, xs]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    field = np.exp(-(((X - 10) ** 2 + (Y - 10) ** 2) / (2 * 3.0 ** 2)))
    encb = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 20), (0, 20)], bandwidth=12.0, seed=0)
    fv = grid_to_function(encb, field, coords)
    recon = function_to_grid(encb, fv, coords)
    corr = float(np.corrcoef(field.ravel(), recon.ravel())[0, 1])
    assert corr > 0.9, corr

    print(f"holographic_tiling selftest: ok (2-D TILE motif==copy {q00:.3f}=={q22:.3f}, gap {gap:.3f}; "
          f"shift-is-bind exact; 3-D TILE cell {enc3.query(t3,[25,25,25]):.3f} gap {enc3.query(t3,[12,12,12]):.3f}; "
          f"RECURSION 8x8 tiles from 12 binds, far corner {corner:.3f}; grid<->hypervector corr {corr:.3f})")


if __name__ == "__main__":
    _selftest()


def fractal_volume(enc, period, counts, levels, motif=None, beta=2.0, seed=0, motif_size=5,
                   threshold=1e-2, motif_grid=None, motif_coords=None):
    """Inception over ANY VSA object -> ONE hypervector, in a single call. The SEED of the self-similar volume
    can be:
      * motif=<hypervector> -- ANY VSA object: a smoke puff (a density field crossed into VSA), an SDF surface,
        a stored archive image, or the OUTPUT OF ANOTHER fractal_volume (inception over the engine itself).
        Used directly: tile_recursive replicates it count^levels deep.
      * motif_grid=<array>, motif_coords=... -- a NumPy field crossed into VSA ONCE via grid_to_function, then
        tiled (physics -> inception: simulate a puff on a grid, tile it self-similarly).
      * neither (default) -- synthesise a LOCALIZED fractal grain (a spectral_field patch under a Gaussian
        envelope, a positive bump modulated by 1/f^beta detail). The original behaviour, unchanged.

    Whatever the seed, tile_recursive gives count^levels self-similar copies per axis from only L*prod(counts)
    binds, in one fixed-size vector. HONEST framing: ANY hypervector yields a valid composable self-similar
    hypervector (it's all binds and a sum -- bind it to a role, bundle it, store it, clean it up). The SPATIAL
    read-back (enc.query at a copy) is meaningful only for FPE-FUNCTION motifs (the synthesized grain, a
    grid_to_function field, another fractal_volume's output); an arbitrary non-FPE plate still tiles into a
    valid bundle, just not a spatially queryable one. Works in 2-D and 3-D (enc.n_dims)."""
    if motif is not None:
        motif_fn = np.asarray(motif, dtype=float)                 # ANY VSA object, used directly as the seed
    elif motif_grid is not None:
        motif_fn = grid_to_function(enc, motif_grid, motif_coords, threshold=threshold)   # a field -> VSA once
    else:
        from holographic_fields import spectral_field
        n = enc.n_dims
        s = int(motif_size)
        grain = spectral_field((s,) * n, beta=beta, seed=seed)
        centered = [np.arange(s) - (s - 1) / 2.0 for _ in range(n)]
        mesh = np.meshgrid(*centered, indexing="ij")
        env = np.exp(-sum(m ** 2 for m in mesh) / (2.0 * (s / 4.0) ** 2))   # Gaussian envelope -> localized
        grain = env * (1.0 + 0.6 * grain)              # a POSITIVE localized bump, modulated by fractal detail
        coords = [np.arange(s, dtype=float) for _ in range(n)]
        motif_fn = grid_to_function(enc, grain, coords, threshold=threshold)
    return tile_recursive(enc, motif_fn, period, counts, levels)


def inception(enc, period, counts, depth, motif=None, beta=2.0, seed=0, motif_size=5):
    """One-parameter recursion DEPTH over fractal_volume, plus an honest capacity-ceiling MEASUREMENT.

    The volume is fractal_volume's own recursive tiling carried `depth` levels deep. (Probed and verified
    bit-for-bit identical to nesting fractal_volume on its own output: tile_recursive already feeds each
    level's output back in at a period grown by `counts`, so this is the EXISTING tiling exposed as a depth
    knob, not new tiling math -- the de-dup discipline kept honest.) Each extra level multiplies the motif
    instances by counts**n_dims but the vector size is FIXED, so the per-copy read (SNR) falls with depth.

    The genuinely-new part is the returned `profile`: at every depth 1..depth it reports copies-per-axis, the
    mean per-copy read (enc.query at the tiled instances), and the role-binding round-trip recovery -- so the
    capacity ceiling of nesting is a measured table, not a footnote. Returns (volume, profile): `volume` is one
    fixed-size hypervector, composable as any VSA object; `profile` is a list of dicts (one per depth). The
    seed `motif` is anything fractal_volume accepts (default: a synthesized fractal grain)."""
    from holographic_ai import bind, unbind, cosine, random_vector
    n = enc.n_dims
    p0 = float(np.atleast_1d(np.asarray(period, float))[0])     # base period: instances sit at k * p0 per axis
    role = random_vector(getattr(enc, "dim", None) or enc.encode([0.0] * n).shape[0],
                         np.random.default_rng(0))

    def _measure(vol, d):
        copies = int(counts) ** int(d)                         # motif instances per axis
        ks = range(min(copies, 8))                             # sample the first few along the diagonal
        reads = [float(enc.query(vol, [k * p0] * n)) for k in ks]
        recovery = float(cosine(unbind(bind(role, vol), role), vol))   # structured-vector round-trip
        return {"depth": int(d), "copies_per_axis": copies,
                "mean_read": float(np.mean(reads)), "recovery": recovery}

    vol = fractal_volume(enc, period, counts, 1, motif=motif, beta=beta, seed=seed, motif_size=motif_size)
    profile = [_measure(vol, 1)]
    p = p0
    for d in range(2, int(depth) + 1):
        p = p * int(counts)                                    # tile the whole previous block: copies nest
        vol = fractal_volume(enc, p, counts, 1, motif=vol)
        profile.append(_measure(vol, d))
    return vol, profile

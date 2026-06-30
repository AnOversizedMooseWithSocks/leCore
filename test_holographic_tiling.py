"""VSA-native tiling (holographic_tiling): domain repetition as bind+bundle on FPE field hypervectors. Tiling
is composable (returns a hypervector), n-D (2-D and 3-D), recursive (count^L tiles from linear binds), and
bridges NumPy grids <-> hypervectors so physics fields can be tiled/bound/bundled like any VSA object."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_tiling import tile, tile_recursive, fractal_bands, grid_to_function, function_to_grid


def test_tile_places_an_exact_copy_in_each_cell():
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
    motif = enc.encode([5.0, 5.0])
    tiled = tile(enc, motif, period=10.0, counts=3)
    q00 = enc.query(tiled, [5, 5]); q22 = enc.query(tiled, [25, 25])
    assert abs(q00 - q22) < 0.02 and q00 > 0.2            # every tile reads the same as the original
    assert enc.query(tiled, [10, 10]) < 0.1               # gaps between tiles are empty (localized motif)


def test_shift_is_bind_exact():
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
    motif = enc.encode([5.0, 5.0])
    s = enc.shift(motif, np.array([10.0, 0.0]))
    assert abs(enc.query(s, [15, 5]) - enc.query(motif, [5, 5])) < 1e-9


def test_3d_tiling():
    enc = VectorFunctionEncoder(3, dim=4096, bounds=[(0, 80)] * 3, bandwidth=40.0, seed=0)
    m = enc.encode([5.0, 5.0, 5.0])
    t = tile(enc, m, period=10.0, counts=3)
    assert enc.query(t, [25, 25, 25]) > 0.12              # a copy in a far 3-D cell
    assert enc.query(t, [12, 12, 12]) < enc.query(t, [25, 25, 25])   # gap weaker than a cell


def test_recursion_gives_exponential_tiles_from_linear_binds():
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
    motif = enc.encode([5.0, 5.0])
    rec = tile_recursive(enc, motif, period=10.0, counts=2, levels=3)   # 8x8 = 64 tiles, 12 binds
    corner = enc.query(rec, [75, 75])                     # the far corner cell still carries the motif
    assert corner > 0.05 and corner > enc.query(rec, [10, 10])


def test_fractal_bands_superpose_scales():
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 80), (0, 80)], bandwidth=40.0, seed=0)
    motif = enc.encode([5.0, 5.0])
    fb = fractal_bands(enc, motif, base_period=20.0, levels=3, count=3)
    assert enc.query(fb, [5, 5]) > 0.1                    # the motif is present (multi-scale bundle is non-trivial)


def test_grid_hypervector_bridge_round_trips():
    xs = np.linspace(0, 20, 11); coords = [xs, xs]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    field = np.exp(-(((X - 10) ** 2 + (Y - 10) ** 2) / (2 * 3.0 ** 2)))
    enc = VectorFunctionEncoder(2, dim=4096, bounds=[(0, 20), (0, 20)], bandwidth=12.0, seed=0)
    fv = grid_to_function(enc, field, coords)
    recon = function_to_grid(enc, fv, coords)
    assert np.corrcoef(field.ravel(), recon.ravel())[0, 1] > 0.9


def test_fractal_volume_recursively_tiles_a_fractal_grain():
    """fractal_volume: one call -> a localized fractal grain (spectral_field) crossed into VSA once and
    tile_recursive'd into count^levels self-similar copies, held in one fixed-size hypervector."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    fv = fractal_volume(enc, period=10.0, counts=2, levels=2, beta=2.0, seed=1, motif_size=5)
    assert fv.shape == (8192,)                                   # ONE fixed-size vector
    reads = [enc.query(fv, [2 + 10 * k, 2 + 10 * k]) for k in range(4)]   # 2^2 = 4 copies per axis
    assert all(r > 0.1 for r in reads)                           # every self-similar copy is present
    assert max(reads) - min(reads) < 0.1                        # and they read consistently


def test_fractal_volume_capacity_falls_off_with_more_copies():
    """Kept negative: more superposed copies share one fixed dim, so the per-copy read (SNR) falls."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    e4 = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    e9 = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 90), (0, 90)], bandwidth=20.0, seed=0)
    f4 = fractal_volume(e4, 10.0, 2, 2, beta=2.0, seed=1, motif_size=5)   # 4 copies/axis
    f9 = fractal_volume(e9, 10.0, 3, 2, beta=2.0, seed=1, motif_size=5)   # 9 copies/axis
    assert e4.query(f4, [2, 2]) > e9.query(f9, [2, 2])          # fewer copies -> stronger read


def test_fractal_volume_works_in_3d():
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    enc = VectorFunctionEncoder(3, dim=8192, bounds=[(0, 40)] * 3, bandwidth=16.0, seed=0)
    fv = fractal_volume(enc, period=10.0, counts=2, levels=1, beta=2.0, seed=1, motif_size=5)
    reads = [enc.query(fv, [2 + 10 * k, 2 + 10 * k, 2 + 10 * k]) for k in range(2)]
    assert all(r > 0.1 for r in reads) and fv.shape == (8192,)


def test_fractal_volume_accepts_a_motif_hypervector():
    """Generalized seed: ANY VSA object can seed the volume. Here an FPE point-encoding is the motif, used
    directly (no synthesis), and the recursively-tiled copies still read."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    motif = enc.encode([2.0, 2.0])                              # a localized feature, as a hypervector
    fv = fractal_volume(enc, 10.0, 2, 2, motif=motif)
    reads = [enc.query(fv, [2 + 10 * k, 2 + 10 * k]) for k in range(4)]
    assert all(r > 0.1 for r in reads) and fv.shape == (8192,)


def test_fractal_volume_tiles_a_physics_grid():
    """physics -> inception: a 'smoke puff' density field crossed into VSA once (motif_grid) then tiled
    self-similarly. The simulated feature becomes the seed of a self-similar volume."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    puff = np.zeros((5, 5)); puff[2, 2] = 1.0
    puff[1, 2] = puff[3, 2] = puff[2, 1] = puff[2, 3] = 0.5      # a little blob
    coords = [np.arange(5, dtype=float), np.arange(5, dtype=float)]
    fv = fractal_volume(enc, 10.0, 2, 2, motif_grid=puff, motif_coords=coords)
    reads = [enc.query(fv, [2 + 10 * k, 2 + 10 * k]) for k in range(4)]
    assert all(r > 0.1 for r in reads)


def test_fractal_volume_inception_over_engine_output():
    """Inception over the engine itself: a fractal_volume's OUTPUT is a hypervector, so feed it back in as the
    motif of another fractal_volume -> copies-of-copies, all from binds, in one fixed-size vector."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    inner = fractal_volume(enc, 10.0, 2, 1, beta=2.0, seed=1)   # 2 copies/axis
    nested = fractal_volume(enc, 20.0, 2, 1, motif=inner)       # tile that whole thing again
    reads = [enc.query(nested, [2 + 10 * k, 2 + 10 * k]) for k in range(4)]
    assert sum(r > 0.05 for r in reads) >= 3                    # copies-of-copies present


def test_fractal_volume_any_vsa_object_is_composable():
    """HONEST framing: an ARBITRARY (non-FPE) hypervector still tiles into a VALID composable hypervector --
    it's all binds and a sum. Not spatially queryable, but finite, non-degenerate, and round-trips through the
    VSA algebra like any other object."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import fractal_volume
    from holographic_ai import bind, unbind, cosine, random_vector
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 50), (0, 50)], bandwidth=20.0, seed=0)
    concept = random_vector(8192, np.random.default_rng(7))     # an arbitrary VSA object (not an FPE function)
    fv = fractal_volume(enc, 10.0, 2, 2, motif=concept)
    assert np.all(np.isfinite(fv)) and np.linalg.norm(fv) > 0   # a valid, non-degenerate hypervector
    role = random_vector(8192, np.random.default_rng(8))
    recovered = unbind(bind(role, fv), role)
    unrelated = random_vector(8192, np.random.default_rng(9))
    assert cosine(recovered, fv) > 0.5                          # composes through the algebra (structured vec)
    assert cosine(recovered, fv) > cosine(recovered, unrelated) + 0.3   # clearly not noise


def test_inception_returns_composable_volume_and_profile():
    """inception: one depth knob -> a composable volume plus a per-depth measurement table."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import inception
    from holographic_ai import bind, unbind, cosine, random_vector
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 200), (0, 200)], bandwidth=20.0, seed=0)
    vol, profile = inception(enc, 10.0, 2, 3, beta=2.0, seed=1)
    assert vol.shape == (8192,) and len(profile) == 3
    assert [r["copies_per_axis"] for r in profile] == [2, 4, 8]
    role = random_vector(8192, np.random.default_rng(5))
    assert cosine(unbind(bind(role, vol), role), vol) > 0.5    # the whole volume composes through the algebra


def test_inception_capacity_ceiling_read_falls_with_depth():
    """The honest ceiling: each level multiplies the motif instances but the dim is fixed, so the per-copy
    read (SNR) falls monotonically with depth. (Whole-vector recovery is a different quantity and need not.)"""
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import inception
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 200), (0, 200)], bandwidth=20.0, seed=0)
    _, profile = inception(enc, 10.0, 2, 3, beta=2.0, seed=1)
    reads = [r["mean_read"] for r in profile]
    assert reads[0] > reads[1] > reads[2]                      # 0.79 > 0.51 > 0.28


def test_inception_volume_is_fractal_volume_levels_exactly():
    """De-dup honesty: inception's volume at depth D is bit-for-bit fractal_volume(levels=D) -- the existing
    recursive tiling exposed as a depth knob, not new tiling math."""
    import numpy as np
    from holographic_fpe import VectorFunctionEncoder
    from holographic_tiling import inception, fractal_volume
    enc = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 200), (0, 200)], bandwidth=20.0, seed=0)
    vol, _ = inception(enc, 10.0, 2, 3, beta=2.0, seed=1)
    ref = fractal_volume(enc, 10.0, 2, 3, beta=2.0, seed=1)
    assert np.allclose(vol, ref, atol=1e-12)

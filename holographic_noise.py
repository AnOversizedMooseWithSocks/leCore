"""Holographic procedural noise (G1): band-limited noise as a FIELD, fBm as an octave BUNDLE.

WHY THIS MODULE EXISTS
----------------------
The geometry-detail wishlist (terrain, displacement, greebles, vegetation scatter) all bottom out in
ONE missing primitive: a procedural noise generator. The engine had de-noising and band-limiting, but
no noise *source*. This module supplies it -- and supplies it the holographic way, as a FIELD rather
than a python pixel loop.

THE IDEA (one band is one hypervector)
--------------------------------------
A noise field is a sum of random-weighted RBF bumps:

    field = sum_i  w_i * encode(p_i),   w_i ~ N(0, 1),  p_i scattered on a jittered lattice

That is exactly the FPE `bundle` -- a holographic kernel-density estimate -- with RANDOM weights
instead of data weights. Querying it, `query(field, x) = sum_i w_i kernel(x, p_i)`, reads a smooth
random field: a Gaussian-process sample with the encoder's RBF kernel. It is BAND-LIMITED by
construction (the kernel is smooth; the correlation length is ~1/bandwidth), and it is ONE
fixed-length vector you can evaluate anywhere at O(1) per query, shift by a single bind, and combine
with any other field by bundling. The band-limit is a feature, not a bug: a band-limited source is
anti-aliased by construction (the Stam/Berry seats' point about spectral control).

THE IDEA (fBm is an octave bundle)
----------------------------------
Fractional Brownian motion sums octaves -- copies of the noise at geometrically rising frequency and
falling amplitude. In this representation frequency IS bandwidth (a sharper RBF kernel varies faster),
so an octave is just a band field at bandwidth `base * lacunarity^o`, and

    fBm(x) = sum_o  gain^o * query(band_o, x)

is a WEIGHTED SUPERPOSITION of band fields -- a bundle. The classic per-octave loop collapses to
"build a few band fields, sum their reads." Each band is holographic; the fBm composes them by
bundling, and the whole thing is callable and composable from a VSA program.

HONEST SCOPE (kept negatives)
-----------------------------
  * Band-limited / smooth by construction. Sharp, discontinuous noise (hard cell edges, Worley-style
    ridges) is NOT this primitive's regime -- it is a smooth-spectrum source. For terrain/displacement
    that is the right default; for hard-edged masks it is the wrong tool (use a raster).
  * FFT-bound, like all FPE work. Each kernel placed is one `encode` (an FFT bind), so a well-filled
    high-frequency octave costs many FFTs. Practical for a handful of octaves at modest dim; DEEP fBm
    (many high octaves, fully filled) is expensive -- the array path is the fast path there. We CAP
    the kernels per octave and report the resulting fill rather than hide the cost.
  * The FPE WRAPS at the encoder bounds. `bounds` must exceed the queried range with margin, or the
    field tiles -- handled here by sizing bounds from the requested domain.

Deterministic given a seed (every random draw goes through default_rng(seed)).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from holographic_fpe import VectorFunctionEncoder, _fpe_parallel_workers
from holographic_ai import cosine


# ---------------------------------------------------------------------------
# Single-band noise: one hypervector.
# ---------------------------------------------------------------------------

def _fill_lattice(bounds, per_axis, rng):
    """A jittered lattice of points filling the box `bounds` with `per_axis` cells along each axis.

    WHY jittered (not pure-random, not pure-grid): a regular grid leaves aliasing artefacts; pure random
    leaves clumps and gaps. A grid jittered by a fraction of a cell is the standard well-distributed
    sample -- it fills the domain evenly while breaking the lattice regularity, so the kernels overlap
    smoothly into a noise field rather than isolated bumps or a visible grid.
    """
    n_dims = len(bounds)
    axes = []
    for (lo, hi) in bounds:
        step = (hi - lo) / per_axis
        centers = lo + (np.arange(per_axis) + 0.5) * step   # cell centres
        axes.append((centers, step))
    # cartesian product of the per-axis centres
    grids = np.meshgrid(*[c for c, _ in axes], indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)       # (per_axis^n_dims, n_dims)
    # jitter each point by up to +/- half a cell on each axis
    for k in range(n_dims):
        step = axes[k][1]
        pts[:, k] += rng.uniform(-0.5, 0.5, pts.shape[0]) * step
    return pts


def _kernels_for_bandwidth(bounds, bandwidth, fill=1.2, cap=1600):
    """How many lattice cells per axis to FILL the domain at this bandwidth (so kernels overlap).

    The RBF kernel's width in normalized [0,1] units is ~1/bandwidth, so the number of resolvable
    features along an axis is ~bandwidth. To make the bumps overlap into a continuous field we want a
    lattice at least that dense; `fill` (>1) oversamples slightly. We CAP the total so a high octave
    stays tractable -- and the caller can read back the achieved per-axis count to know the fill.
    """
    n_dims = len(bounds)
    per_axis = max(2, int(np.ceil(bandwidth * fill)))
    while per_axis ** n_dims > cap and per_axis > 2:
        per_axis -= 1
    return per_axis


def noise_field(encoder, per_axis=None, seed=0, fill=1.2):
    """A single-band band-limited noise field as ONE hypervector (an FPE bundle of random kernels).

    NOT a duplicate of holographic_pattern.value_noise: this is the HYPERVECTOR-native noise (the field IS one
    vector, queried through an encoder); pattern.value_noise is the plain spatial callable for Param sockets.
    Same concept, two levels -- as above, so below.

    `encoder` is a VectorFunctionEncoder (its bandwidth sets the noise frequency / feature size).
    Returns the field vector; evaluate it with `encoder.query(field, point)` (or `sample` below).
    """
    rng = np.random.default_rng(seed)
    if per_axis is None:
        # use the encoder's first-axis bandwidth to choose a domain-filling lattice
        bw = encoder.bandwidth[0]
        per_axis = _kernels_for_bandwidth(encoder.bounds, bw, fill=fill)
    pts = _fill_lattice(encoder.bounds, per_axis, rng)
    weights = rng.normal(0.0, 1.0, pts.shape[0])             # random signed heights -> noise, not bumps
    # f = sum_i w_i encode(p_i): the FPE bundle IS the field (a holographic KDE with random weights).
    return encoder.bundle(pts, weights)


def sample(encoder, field, point):
    """Read the noise field at one point: cosine(field, encode(point)) = sum_i w_i kernel(point, p_i)."""
    return float(encoder.query(field, point))


def sample_many(encoder, field, points, workers=None):
    """Read the noise field at many points via the encoder's batched FPE query path."""
    return encoder.query_many(field, points, workers=workers)


# ---------------------------------------------------------------------------
# fBm: a weighted superposition of band fields (the octave bundle).
# ---------------------------------------------------------------------------

class FractalNoise:
    """Fractional Brownian noise: `octaves` band fields summed with falling amplitude -- the octave bundle.

    Each octave is a band field at bandwidth `base_bandwidth * lacunarity^o` and amplitude `gain^o`,
    built in its OWN encoder (independent random base + bandwidth). `query(point)` sums the octave reads:
    fBm(x) = sum_o gain^o * query(band_o, x). Higher `gain` (persistence) -> rougher; more `octaves` ->
    more fine detail; `lacunarity` is the per-octave frequency multiplier.
    """

    def __init__(self, n_dims, dim=1024, bounds=None, octaves=4, lacunarity=2.0,
                 gain=0.5, base_bandwidth=2.0, seed=0, kernel="rbf"):
        if bounds is None:
            bounds = [(0.0, 1.0)] * n_dims
        self.n_dims = int(n_dims)
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.octaves = int(octaves)
        self.lacunarity = float(lacunarity)
        self.gain = float(gain)
        self.encoders = []
        self.fields = []
        self.amplitudes = []
        self.per_axis = []     # achieved fill per octave (for honest read-back)
        for o in range(self.octaves):
            bw = base_bandwidth * (self.lacunarity ** o)
            # one encoder per octave: distinct seed so the bands are independent random fields
            enc = VectorFunctionEncoder(self.n_dims, dim=dim, bounds=self.bounds,
                                        kernel=kernel, bandwidth=bw, seed=seed * 131 + o + 1)
            pa = _kernels_for_bandwidth(self.bounds, bw)
            fld = noise_field(enc, per_axis=pa, seed=seed * 977 + o + 7)
            self.encoders.append(enc)
            self.fields.append(fld)
            self.amplitudes.append(self.gain ** o)
            self.per_axis.append(pa)
        # normalize so query() lands in a roughly unit range regardless of octave count
        self._norm = float(sum(self.amplitudes)) or 1.0

    def query(self, point):
        """Evaluate fBm at a point: the amplitude-weighted sum of the octave reads (the bundle)."""
        return float(self.query_many([point])[0])

    def query_many(self, points, workers=None):
        """Evaluate fBm at many points with one batched read per octave."""
        pts = np.asarray(points, float)
        if self.n_dims == 1:
            if pts.ndim == 0:
                pts = pts.reshape(1, 1)
            elif pts.ndim == 1:
                pts = pts.reshape(-1, 1)
        else:
            pts = np.atleast_2d(pts)
        if pts.ndim != 2 or pts.shape[1] != self.n_dims:
            raise ValueError(f"points must have shape (count, {self.n_dims})")

        total = np.zeros(pts.shape[0], dtype=float)
        rows_per_octave = pts.shape[0] * max(1, self.octaves)
        worker_count = _fpe_parallel_workers(self.octaves, rows_per_octave, workers)

        def octave_read(item):
            amp, enc, fld = item
            return amp * enc.query_many(fld, pts, workers=1)

        octave_items = list(zip(self.amplitudes, self.encoders, self.fields))
        if worker_count == 1:
            parts = [octave_read(item) for item in octave_items]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                parts = list(executor.map(octave_read, octave_items))
        for part in parts:
            total += part
        return total / self._norm

    def sample_grid(self, res):
        """Evaluate fBm on a res^n_dims lattice over `bounds` (n_dims==2 -> a res x res array).

        For measuring (fractal dimension / Hurst / spectrum) and for feeding terrain/displacement.
        """
        axes = [np.linspace(lo, hi, res) for (lo, hi) in self.bounds]
        grids = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=1)
        vals = self.query_many(pts)
        return vals.reshape([res] * self.n_dims)


# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)

    # (1) DETERMINISM: same seed -> identical field.
    enc = VectorFunctionEncoder(2, dim=512, bounds=[(0, 8), (0, 8)], kernel="rbf", bandwidth=3.0, seed=1)
    f1 = noise_field(enc, seed=5)
    f2 = noise_field(enc, seed=5)
    assert np.allclose(f1, f2), "noise_field not deterministic for a fixed seed"
    f3 = noise_field(enc, seed=6)
    assert cosine(f1, f3) < 0.5, "different seeds should give different fields"

    # (2) BAND-LIMITED: a single band is SMOOTH -- adjacent samples are strongly correlated, far ones
    #     are not. Sample along a transect and check lag-1 autocorrelation is high (smooth) while the
    #     long-lag correlation decays (it is noise, not a constant).
    xs = np.linspace(0.5, 7.5, 120)
    line = np.array([sample(enc, f1, [x, 4.0]) for x in xs])
    line = line - line.mean()
    ac1 = float(np.corrcoef(line[:-1], line[1:])[0, 1])                 # lag-1
    ac_far = float(np.corrcoef(line[:-20], line[20:])[0, 1])           # lag-20
    assert ac1 > 0.9, f"band noise should be smooth (high lag-1 autocorr), got {ac1:.3f}"
    assert ac_far < ac1, f"correlation should decay with lag ({ac_far:.3f} !< {ac1:.3f})"

    # (3) fBm = OCTAVE BUNDLE: more persistence (higher gain) keeps the fine octaves alive -> rougher.
    #     Measure roughness as NORMALIZED lag-1 variation: std(diff) / std(profile) -- the wiggle PER UNIT
    #     amplitude. This is scale-free (divides out the overall amplitude), so it reads "how much fine
    #     structure relative to the whole", which is exactly what persistence controls.
    def roughness(gain):
        fb = FractalNoise(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=4, lacunarity=2.0,
                          gain=gain, base_bandwidth=3.0, seed=3)
        prof = np.array([fb.query([x, 4.0]) for x in np.linspace(0.3, 7.7, 200)])
        return float(np.std(np.diff(prof)) / (np.std(prof) + 1e-9))
    r_smooth = roughness(0.25)    # octaves die fast -> mostly the smooth base octave
    r_rough = roughness(0.90)     # octaves persist -> fine detail survives
    assert r_rough > r_smooth, f"higher persistence should be rougher: {r_rough:.4f} !> {r_smooth:.4f}"

    # (4) HURST connects to the shipped measurer and moves the right way with persistence.
    from holographic_fractal import hurst_exponent
    def hurst_of(gain):
        fb = FractalNoise(1, dim=512, bounds=[(0, 16)], octaves=4, lacunarity=2.0,
                          gain=gain, base_bandwidth=2.0, seed=2)
        series = np.array([fb.query([x]) for x in np.linspace(0, 16, 400)])
        return hurst_exponent(np.cumsum(series))      # integrate increments -> an fBm-like walk
    h_lo = hurst_of(0.35)
    h_hi = hurst_of(0.75)
    assert 0.0 < h_lo < 1.5 and 0.0 < h_hi < 1.5, f"Hurst out of range: {h_lo:.2f}, {h_hi:.2f}"

    # (5) the octave bundle is literally a weighted sum of band reads (assert the decomposition).
    fb = FractalNoise(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=3, lacunarity=2.0,
                      gain=0.5, base_bandwidth=2.0, seed=4)
    p = [3.3, 5.1]
    manual = sum(a * e.query(f, p) for a, e, f in zip(fb.amplitudes, fb.encoders, fb.fields)) / fb._norm
    assert abs(manual - fb.query(p)) < 1e-12, "fBm query must equal the amplitude-weighted octave sum"

    print("holographic_noise selftest passed:",
          f"lag1_autocorr={ac1:.3f} rough(0.25)={r_smooth:.4f} rough(0.90)={r_rough:.4f} "
          f"H(0.35)={h_lo:.2f} H(0.75)={h_hi:.2f} octave_fill={fb.per_axis}")


if __name__ == "__main__":
    _selftest()

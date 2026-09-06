"""Holographic caustics: the caustic as ONE hypervector, with wavelength as an axis (RENDER = QUERY).

WHERE THE RENDER PIPELINE LEFT ITS OWN PATH, AND WHY THIS MODULE EXISTS
------------------------------------------------------------------------
This engine's render lineage was holographic before it was anything else: PRT ("collapse, don't trace" -- relight
is a dot product), bake_scene/render_baked (bake once, every frame is a relight), holographic_volint (the whole
density field as one vector, the ray integral in closed form -- 90x over marching), holographic_radiance ("RENDER =
QUERY": radiance as FPE bundles, read at any point). Then, from sweep 138 -- "caustics + dispersion" -- the work went
conventional: Monte-Carlo path tracing with next-event estimation, hero-wavelength spectral sampling, manifold NEE,
photon-splat caustics into a histogram, a mesh baked to a voxel grid and sphere-traced. Every one of those is how
Cycles does it. Renders took 30-100 minutes and were hard to make look good, which is the tell: the pipeline had
stopped using the substrate. This module is the first step back.

THE OBSERVATION THAT MAKES IT HOLOGRAPHIC
-----------------------------------------
A caustic is two operations of different character. REFRACTION is Snell's law -- nonlinear, it does not superpose,
and it is cheap: one vectorised pass over the rays (holographic_globalillum.caustic_hits does it). ACCUMULATION of
where the rays land is a pure SUM -- a bundle -- and it is where every cost of the conventional version lives:
  * a histogram is fixed to ONE resolution; read it at 2x and you must re-shoot every ray;
  * a histogram is NOISY until the ray count scales with pixels^2;
  * colour means shooting the whole thing again per wavelength band -- the spectral render costs count-of-bands
    times the monochrome one (the shipped spectral_caustics traces mind.caustics 10 times).
The engine's own thesis (holographic_distribute, verbatim): "distribute the LINEAR accumulation by superposition,
keep the nonlinear part local." So: refract once, then BUNDLE the landings into a Fractional Power Encoding field --

    F = sum_i  w_i * encode(x_i, z_i, lambda_i)             (one vector; the whole spectral caustic)

-- and READ the image out of it: irradiance(x, z, lambda) ~ <F, encode(x, z, lambda)>, the FPE Bochner/RBF kernel-
density estimate. Wavelength is simply a THIRD AXIS of the encoder. That is the "just add more dimensions" lever:
the spectrum is not a loop over bands, it is one more coordinate of one vector.

THE COLOUR-MATCHING INTEGRAL FOLDS INTO THE QUERY -- THIS IS THE PART THAT IS GENUINELY THE SUBSTRATE
----------------------------------------------------------------------------------------------------
An RGB channel is the spectrum weighted by a colour-matching function and integrated over lambda:

    E_c(x, z) = integral  w_c(lambda) <F, encode(x, z, lambda)> d lambda
              = <F, integral w_c(lambda) encode(x, z, lambda) d lambda>            (linearity)
              = <F, bind(encode_xz(x, z), Q_c)>,   Q_c = sum_lambda w_c(lambda) encode_lambda(lambda)

because the FPE encode of a point is the BIND (spectral-domain product) of its per-axis encodes. And <F, bind(a, b)>
= <unbind(F, b), a>. So the entire spectral integral is THREE UNBINDS -- G_c = unbind(F, Q_c), one per channel --
after which each colour channel is a plain 2-D field read at any resolution. Bind a role (wavelength), bundle,
unbind to read: it is the identical mechanism the engine uses to store and recall a record, a scene, a sentence.
That is the correspondence Moose built the renderer to demonstrate, made literal.

WHAT IT BUYS (measured in the selftest and the sweep notes, not promised):
  * RESOLUTION-INDEPENDENT: one bundle, read at 128^2 or 1024^2 -- "project the film at any size".
  * SMOOTH BY CONSTRUCTION: the RBF kernel IS the density estimator; bandwidth is the one knob. No pixel grain.
  * ONE PASS FOR THE WHOLE SPECTRUM: each ray carries its own wavelength; N rays total, not N x bands.
  * COMPOSABLE: two objects' caustics ADD (bundle); a moved object is a BIND (the fpefield headline).
  * EMPTY SPACE IS KNOWN: where nothing landed, the read is ~0 -- a property of the field, not a discovery.

KEPT HONEST (loud):
  * The refraction is still one interface (entry only), same as caustics(): exit refraction is the documented next
    rung, unchanged by this module.
  * The kernel read has crosstalk: a finite-dim FPE field carries an interference floor that falls with dim (the
    capacity trade every holographic field in this engine states). Reads are clipped at 0; the floor is measured
    in the selftest against the histogram.
  * Bandwidth is set by CAPACITY, not by ray density: ~0.6*sqrt(dim) per spatial axis, the sharpest kernel the
    dimension can hold (measured: sharper than that, agreement with the histogram FALLS -- see default_bandwidth).
  * COLOUR SEPARATION IS CAPACITY-BOUNDED, and this is the loudest negative. Two single-wavelength reads of the
    SAME geometry correlate at only ~0.75 (dim 2048) / ~0.79 (dim 4096) with dispersion OFF -- each lambda slice
    has its own crosstalk realisation, and that is the lambda-axis noise floor. Physical dispersion through a
    sphere at scale 3 is a ~3%-of-variance effect (red vs blue landing histograms correlate at 0.972), so at
    dim <= 4096 the crosstalk EXCEEDS the effect; dispersion still measurably lowers the red-blue read
    correlation (0.009 at 2048, 0.024 at 4096, growing with dim), but a rainbow read out of a small field is
    mostly crosstalk wearing colour. Raise dim (the read is O(pixels x dim)) or TILE the receiver. The
    ~9,000-landing fixture in 2,048 dims is a COMPRESSED representation; the crosstalk is the compression loss.
  * Kept-negative metrics, so no one re-tries them: red-vs-blue CENTROID offset does not measure lens dispersion
    (a lens's bulk position is set by its central ray; the offset read 0.025 at every dispersion scale) --
    radial SPREAD does (blue focuses closer, lands wider: ratio 1.05 -> 0.82). Clipping the read at zero
    before measuring biases centroids by up to 4 pixels (raw=True exists for this). Padding the encoder bounds
    changes nothing: Gaussian-phase FPE does not wrap (that warning is for sinc encoders).
  * The read is O(pixels x dim) -- a couple of matmuls. That is the "computed, not shone" seam from the optical
    correspondence note: as parallel as optics, not free.

Basis: Frady, Kleyko & Sommer, "Computing on Functions Using Randomized Vector Representations" (VFA, 2021);
Plate (HRR); the engine's own holographic_volint and holographic_radiance, whose spectral-domain query this reuses
convention-for-convention (Theta = scale * phases per axis; encode_spec(p)_j = exp(i p . Theta_j)).
NumPy/stdlib only, deterministic.
"""
import numpy as np

from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


class HolographicCaustic:
    """The caustic as ONE FPE hypervector over (x, z, wavelength). `deposit` bundles landings in; `read` /
    `read_rgb` project the image out at any resolution. Build it once, read it as often and as large as you like.

    `bounds_xz` = ((x_lo, x_hi), (z_lo, z_hi)) is the receiver window; `lam_bounds` the wavelength range in nm.
    `bandwidth` is per-axis (x, z, lambda) -- larger is SHARPER in this encoder's convention (see fpe.py). `dim`
    is the hypervector length: more dim, lower crosstalk floor, more memory per read."""

    def __init__(self, bounds_xz, lam_bounds=(420.0, 680.0), dim=2048, bandwidth=(60.0, 60.0, 12.0), seed=0):
        bx, bz = bounds_xz
        self.bounds = [(float(bx[0]), float(bx[1])), (float(bz[0]), float(bz[1])),
                       (float(lam_bounds[0]), float(lam_bounds[1]))]
        self.dim = int(dim)
        self.enc = VectorFunctionEncoder(3, dim=self.dim, bounds=self.bounds, bandwidth=list(bandwidth), seed=seed)
        # Theta[k] = scale_k * phases_k -- the same basis volint and radiance read through, so the three
        # modules agree on what encode(p) means in the spectral domain.
        self.Theta = np.stack([ax.scale * ax.phases for ax in self.enc.axes], axis=0)      # (3, dim)
        self.F_spec = np.zeros(self.dim, dtype=complex)                                     # the field, spectral
        self.n_deposited = 0
        self.total_weight = 0.0

    # ------------------------------------------------------------------ write: bundle landings in
    def deposit(self, xz, lam, weights=None, chunk=4096):
        """Bundle landing points into the field: F += sum_i w_i encode(x_i, z_i, lambda_i). Pure superposition,
        so it may be called repeatedly -- more rays, a second object, a second light -- and the field simply
        accumulates. Chunked so the (chunk, dim) complex temporary stays small; at 360k rays x 2048 dim the
        unchunked form would be 11.8 GB, the exact class of allocation this session found OOM-killing bakes."""
        xz = np.atleast_2d(np.asarray(xz, float))
        lam = np.asarray(lam, float).reshape(-1)
        n = len(xz)
        if lam.shape[0] != n:
            raise ValueError("one wavelength per landing: got %d for %d landings" % (lam.shape[0], n))
        w = np.ones(n) if weights is None else np.asarray(weights, float).reshape(-1)
        P = np.column_stack([xz[:, 0], xz[:, 1], lam])                                      # (n, 3)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            phase = P[s:e] @ self.Theta                                                     # (c, dim)
            self.F_spec += (w[s:e, None] * np.exp(1j * phase)).sum(axis=0)
        self.n_deposited += n
        self.total_weight += float(w.sum())
        return self

    # ------------------------------------------------------------------ read: project the image out
    def _read_spec(self, G_spec, xs, zs):
        """Read a lambda-marginalised 2-D field G on the grid xs x zs -- as an OUTER PRODUCT, because the encoder
        is separable. encode(x, z)_j = encode_x(x)_j * encode_z(z)_j, so

            img[k, i] = Re sum_j G_j exp(-i z_k Th_z,j) exp(-i x_i Th_x,j) = Re ( E_z . diag(G) . E_x^T )[k, i]

        with E_x = exp(-i xs (x) Th_x) of shape (res_x, dim) and E_z likewise: two small exp matrices and ONE
        matmul, instead of a complex exponential per pixel per dimension. MEASURED: the per-pixel form took
        223.5s for a 256^2 RGB read at dim 16384 (3.2 billion complex exps); this form is a (256,16384)x(16384,256)
        product -- seconds. It is the optical-correspondence note's lesson applied literally: stay in the Fourier
        domain and let the structure (here, separability) do the work a lens would do passively. Same numbers to
        machine precision; resolution is whatever grid you pass."""
        xs = np.asarray(xs, float); zs = np.asarray(zs, float)
        Ex = np.exp(-1j * xs[:, None] * self.Theta[0][None, :])                             # (res_x, dim)
        Ez = np.exp(-1j * zs[:, None] * self.Theta[1][None, :])                             # (res_z, dim)
        return np.real((Ez * G_spec[None, :]) @ Ex.T)                                       # (res_z, res_x)

    def read(self, xs, zs, lam, raw=False):
        """Monochrome slice: the caustic at ONE wavelength, on the grid xs x zs. Fixing lambda is binding the
        wavelength axis to a single value and reading the rest.

        `raw=True` returns the read UNCLIPPED. The kernel read carries crosstalk with both signs (measured at
        dim=4096 with ~9k landings: 38% of pixels negative, amplitude ~11% of the peak), and clipping it at zero
        keeps the positive lobes while deleting the negative ones -- a net positive bias that pulled the centroid
        of two single-wavelength reads apart by 0.15 world units (4 pixels) where the unclipped reads differ by
        0.038. Clip for DISPLAY (negative irradiance is unphysical); MEASURE on the raw read."""
        q_lam = np.exp(-1j * float(lam) * self.Theta[2])                                    # conj(encode_lambda)
        out = self._read_spec(self.F_spec * q_lam, xs, zs)
        return out if raw else np.clip(out, 0.0, None)

    def channel_queries(self, cmf, n_lam=48):
        """The colour-matching integral as THREE QUERY VECTORS: Q_c = sum_lambda w_c(lambda) encode_lambda(lambda),
        sampled on n_lam wavelengths across the encoder's range. `cmf(lams) -> (n_lam, 3)` gives the RGB weight of
        each wavelength. Computed once per read; it depends only on the CMF and the encoder, not on the field."""
        lo, hi = self.bounds[2]
        lams = np.linspace(lo, hi, int(n_lam))
        W = np.asarray(cmf(lams), float)                                                    # (n_lam, 3)
        if W.shape != (len(lams), 3):
            raise ValueError("cmf must return (n_lam, 3) RGB weights, got %r" % (W.shape,))
        E = np.exp(1j * lams[:, None] * self.Theta[2][None, :])                            # (n_lam, dim)
        return (W.T @ E) * ((hi - lo) / max(len(lams) - 1, 1))                              # (3, dim), a Riemann sum

    def read_rgb(self, xs, zs, cmf, n_lam=48, normalise="mean", raw=False):
        """The RGB caustic image on the grid xs x zs: three UNBINDS then three 2-D reads.

            G_c = unbind(F, Q_c) = F_spec * conj(Q_c)          (the whole spectral integral, per channel)
            E_c(x, z) = Re sum_j G_c,j exp(-i (x, z) . Theta_j)

        No per-wavelength pass anywhere: the spectrum was one axis of the field and the CMF folded into the query.
        `normalise='mean'` scales so the image mean is 1 (the convention caustics() uses, so peaks read as
        focusing); 'max' scales the brightest channel to 1; None returns raw field units."""
        Q = self.channel_queries(cmf, n_lam=n_lam)                                          # (3, dim)
        img = np.stack([self._read_spec(self.F_spec * np.conj(Q[c]), xs, zs) for c in range(3)], axis=-1)
        if not raw:
            img = np.clip(img, 0.0, None)                       # display: negative irradiance is unphysical; see read()
        if normalise == "mean":
            img = img / (img.mean() + 1e-12)
        elif normalise == "max":
            img = img / (img.max() + 1e-12)
        return img

    # ------------------------------------------------------------------ algebra: the reasons it is a hypervector
    def translate(self, dx, dz):
        """Move the whole caustic by (dx, dz) with ONE bind -- no re-tracing, no resampling. The fpefield headline,
        on a caustic: encode(p + d) = encode(p) * encode(d) in the spectral domain."""
        self.F_spec = self.F_spec * np.exp(1j * (dx * self.Theta[0] + dz * self.Theta[1]))
        return self

    def add(self, other):
        """Two caustics -- a second object, a second light -- superpose by vector addition. Must share an encoder."""
        if other.Theta.shape != self.Theta.shape or not np.array_equal(other.Theta, self.Theta):
            raise ValueError("caustics must share an encoder (same bounds, dim, seed) to be added")
        self.F_spec = self.F_spec + other.F_spec
        self.n_deposited += other.n_deposited
        self.total_weight += other.total_weight
        return self


def default_bandwidth(dim, sharpness=1.0, lam_bands=10.0):
    """The kernel as sharp as the DIMENSION can support, and no sharper -- derived from a measurement, not a guess.

    The first draft sized the kernel from the ray density (a KDE's textbook bandwidth). It was wrong here, and the
    measurement that showed it is the most important fact about this representation: at dim=1024, agreement with
    the histogram FELL as the kernel sharpened (bandwidth 20 -> 0.81, 40 -> 0.74, 80 -> 0.45, 160 -> 0.12), the
    opposite of what a density estimate does. The cause is the FPE crosstalk floor: a hypervector of dim d
    resolves roughly sqrt(d) cells per axis in 2-D, and a kernel narrower than that cell is asking the vector to
    hold more distinct positions than it has dimensions for -- 9,322 landings as separate spikes in 1,024 dims.
    Raising dim confirmed it directly: at bandwidth 40, corr 0.744 (1024) -> 0.900 (4096) -> 0.953 (16384).

    So the spatial bandwidth is set to ~0.6 * sqrt(dim) -- 19 at 1024, 38 at 4096, 77 at 16384 -- which tracks
    the measured optimum at each dim. This is the capacity law every holographic field in the engine states,
    made into the default. Need more resolution than sqrt(dim) cells? Raise dim, or TILE (the radiance field's
    answer to the same wall). `lam_bands` sets how many resolvable colour bands span the wavelength range."""
    b = float(sharpness) * 0.6 * np.sqrt(max(int(dim), 1))
    return (b, b, float(lam_bands))


def spectral_landings(sdf, n_d, abbe, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, n_side=300,
                      dispersion_scale=1.0, lam_lo=420.0, lam_hi=680.0, seed=0, refracted_only=True,
                      emitter_center=None, aim=None, occluder=None, through=False, absorb_rgb=None, cmf=None,
                      max_internal=4):
    """ONE ray set carrying a CONTINUOUS spectrum: every ray gets its own wavelength (stratified, seeded), hence
    its own Cauchy index n(lambda) from (n_d, abbe), and all are refracted in a single call to caustic_hits.
    Returns (land_xz (M,2), lam (M,)) for the rays that count. This replaces 'trace the scene once per hero
    wavelength' with 'trace it once'."""
    from holographic.rendering.holographic_globalillum import caustic_hits
    from holographic.rendering.holographic_dispersion import cauchy_n, exaggerate
    n = int(n_side) ** 2
    rng = np.random.default_rng(seed)
    # stratified: one wavelength per ray, evenly covering the range, then shuffled so neighbouring rays differ
    lam = lam_lo + (np.arange(n) + rng.random(n)) / n * (lam_hi - lam_lo)
    rng.shuffle(lam)
    iors = np.asarray([float(cauchy_n(l, n_d, abbe)) for l in np.linspace(lam_lo, lam_hi, 64)])
    iors = np.asarray(exaggerate(iors, dispersion_scale), float)
    ior_per_ray = np.interp(lam, np.linspace(lam_lo, lam_hi, 64), iors)                    # smooth, per ray
    if not through:
        land, valid, hit = caustic_hits(sdf, light_dir=light_dir, receiver_y=receiver_y, extent=extent,
                                        ior=ior_per_ray, n_side=n_side, refracted_only=refracted_only,
                                        emitter_center=emitter_center, aim=aim, lam=lam, occluder=occluder)
        keep = valid
        return np.column_stack([land[keep, 0], land[keep, 2]]), lam[keep]
    # FULL TRANSIT + COLOUR: exit refraction sets where the light lands, and the interior path length lets each
    # landing carry its transmission exp(-sigma(lambda) L). sigma(lambda) is the material library's per-RGB absorption
    # spread over the spectrum by the colour-matching weights -- an amethyst's caustic is purple, its shadow too.
    land, valid, hit, plen = caustic_hits(sdf, light_dir=light_dir, receiver_y=receiver_y, extent=extent,
                                          ior=ior_per_ray, n_side=n_side, refracted_only=refracted_only,
                                          emitter_center=emitter_center, aim=aim, lam=lam, occluder=occluder,
                                          through=True, max_internal=max_internal)
    keep = valid
    w = np.ones(int(keep.sum()))
    if absorb_rgb is not None and cmf is not None:
        W = np.asarray(cmf(lam[keep]), float); W = W / (W.sum(axis=1, keepdims=True) + 1e-12)   # per ray, which channel
        sig = W @ np.asarray(absorb_rgb, float).reshape(3)                                        # sigma(lambda), per ray
        w = np.exp(-sig * plen[keep])
    return np.column_stack([land[keep, 0], land[keep, 2]]), lam[keep], w


def holographic_caustic(sdf, n_d=1.55, abbe=25.68, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, n_side=300,
                        window=None, center=(0.0, 0.0), dispersion_scale=1.0, dim=2048, bandwidth=None,
                        lam_lo=420.0, lam_hi=680.0, seed=0, aim=None, emitter_center=None):
    """Build the whole spectral caustic as ONE hypervector: trace once, bundle the landings. Returns the
    HolographicCaustic -- call .read_rgb(xs, zs, cmf) to project it at any resolution, .translate to move it,
    .add to superpose another. `window`/`center` frame the receiver exactly as caustics() does."""
    half = float(extent) if window is None else float(window)
    cx, cz = float(center[0]), float(center[1])
    bounds_xz = ((cx - half, cx + half), (cz - half, cz + half))
    xz, lam = spectral_landings(sdf, n_d, abbe, light_dir=light_dir, receiver_y=receiver_y, extent=extent,
                                n_side=n_side, dispersion_scale=dispersion_scale, lam_lo=lam_lo, lam_hi=lam_hi,
                                seed=seed, refracted_only=True, emitter_center=emitter_center, aim=aim)
    inb = ((xz[:, 0] >= bounds_xz[0][0]) & (xz[:, 0] <= bounds_xz[0][1]) &
           (xz[:, 1] >= bounds_xz[1][0]) & (xz[:, 1] <= bounds_xz[1][1]))
    xz, lam = xz[inb], lam[inb]
    bw = default_bandwidth(dim) if bandwidth is None else tuple(bandwidth)
    hc = HolographicCaustic(bounds_xz, (lam_lo, lam_hi), dim=dim, bandwidth=bw, seed=seed)
    hc.deposit(xz, lam)
    return hc


def caustic_concentration(xz, extent, n_side, light_dir, res=448, pct=99.0):
    """How many times brighter than the DIRECTLY LIT floor the caustic's hotspots are -- the physical `strength`
    for composite_caustic, measured from the landings instead of chosen. Each emitter ray owns a floor cell of
    area (2 extent / n_side)^2 / cos(theta) (theta = incidence on the floor); if the landings were spread evenly
    over the glass's shadow footprint every cell would hold one, so density x cell_area is the local concentration
    and its `pct`-th percentile is what composite's percentile normalisation maps `strength` onto. Typical faceted
    bodies measure 3-10x. WHY: with strength set to the lobe's bare Lambert term the caustic peaked DIMMER than
    the floor around it (0.149 vs 0.47 on the lounge map) -- a lens that does not concentrate is not a lens."""
    xz = np.asarray(xz, float)
    if len(xz) < 10:
        return 1.0
    d = np.asarray(light_dir, float); cos_t = max(abs(d[1] / (np.linalg.norm(d) + 1e-12)), 0.05)
    cell = (2.0 * float(extent) / float(n_side)) ** 2 / cos_t
    lo, hi = np.percentile(xz, 0.5, axis=0), np.percentile(xz, 99.5, axis=0)
    h, _, _ = np.histogram2d(xz[:, 0], xz[:, 1], bins=int(res), range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    bin_area = ((hi[0] - lo[0]) / res) * ((hi[1] - lo[1]) / res)
    dens = h[h > 0] / bin_area                                                        # landings per unit floor area
    return float(np.percentile(dens, pct) * cell)


def histogram_caustic_rgb(xz, lam, cmf, bounds_xz, res=512, blur_px=1.0, weights=None):
    """The BASELINE caustic image from spectral landings: bin each landing into an (res x res) grid weighted by
    its wavelength's colour-matching weights, then a small Gaussian blur of `blur_px` pixels (the emitter cell's
    footprint). Returns (res, res, 3), mean-normalised. This is the ground truth the holographic read is measured
    against, and at this sweep it is the render: MEASURED on an amethyst plate (25,232 landings), the tiled
    holographic read (grid 12, dim 2048) smeared the caustic filaments into blotches and showed its tile seams,
    while the histogram resolved the filaments with dispersion at their edges. The holographic caustic's capacity
    (~sqrt(dim) cells per axis per tile) is the documented limit; below it, bin."""
    xz = np.asarray(xz, float); lam = np.asarray(lam, float)
    (x0, x1), (z0, z1) = bounds_xz
    W = np.asarray(cmf(lam), float); W = W / (W.sum(axis=0, keepdims=True) + 1e-12) * (len(lam) / 3.0)
    if weights is not None:
        W = W * np.asarray(weights, float).reshape(-1, 1)                    # per-landing transmission (Beer-Lambert)
    H = np.zeros((int(res), int(res), 3))
    ix = np.clip(((xz[:, 0] - x0) / max(x1 - x0, 1e-9) * res).astype(int), 0, res - 1)
    iz = np.clip(((xz[:, 1] - z0) / max(z1 - z0, 1e-9) * res).astype(int), 0, res - 1)
    for c in range(3):
        np.add.at(H[:, :, c], (iz, ix), W[:, c])
    if blur_px > 0:
        k = int(np.ceil(3 * blur_px)); t = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (t / float(blur_px)) ** 2); g /= g.sum()
        for c in range(3):                                          # separable Gaussian, edge-clamped
            H[:, :, c] = np.apply_along_axis(lambda v: np.convolve(np.pad(v, k, mode="edge"), g, mode="valid"), 0, H[:, :, c])
            H[:, :, c] = np.apply_along_axis(lambda v: np.convolve(np.pad(v, k, mode="edge"), g, mode="valid"), 1, H[:, :, c])
    return H / max(float(H.mean()), 1e-12)


def gaussian_cmf(lams, centres=(610.0, 545.0, 465.0), width=45.0):
    """A stand-in colour-matching function for tests and for callers without the engine's spectrum_to_rgb: three
    Gaussian bumps at red/green/blue. The mind verb passes the engine's real CIE-derived wavelength_rgb instead."""
    lams = np.asarray(lams, float)
    return np.stack([np.exp(-0.5 * ((lams - c) / width) ** 2) for c in centres], axis=1)


def _selftest():
    """Pin what makes this holographic, not merely another caustic:
      (1) RESOLUTION INDEPENDENCE -- one bundle read at two resolutions agrees after downsampling;
      (2) AGREEMENT WITH THE HISTOGRAM -- the same landings, binned vs bundled, correlate strongly;
      (3) TRANSLATE IS A BIND -- moving the caustic by one bind equals re-depositing shifted landings;
      (4) DISPERSION LIVES IN THE FIELD -- with a dispersive index, red and blue centroids separate along the
          light's tangential direction and the separation scales with dispersion_scale; with none, they coincide.
    Each is stated as a number against a baseline. A glass sphere is the fixture: its caustic is a known focus."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.rendering.holographic_globalillum import caustics
    s = sphere(0.6)
    L = np.array([0.35, -1.0, 0.0]); L /= np.linalg.norm(L)
    recv = -1.2

    hc = holographic_caustic(s, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=160, window=1.2,
                             center=(0.0, 0.0), dispersion_scale=0.0, dim=1024, seed=0, aim=(0.0, 0.0, 0.0))
    assert hc.n_deposited > 2000, "the fixture must actually land rays: %d" % hc.n_deposited

    # (1) resolution independence: read at 64 and 128 from the SAME vector; block-average 128 -> 64 and compare
    xs64 = np.linspace(-1.2, 1.2, 64); xs128 = np.linspace(-1.2, 1.2, 128)
    a = hc.read_rgb(xs64, xs64, gaussian_cmf)
    b = hc.read_rgb(xs128, xs128, gaussian_cmf)
    b_down = b.reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
    ra, rb = a.mean(-1).ravel(), b_down.mean(-1).ravel()
    corr_res = float(np.corrcoef(ra, rb)[0, 1])
    assert corr_res > 0.97, "one bundle must read consistently at any resolution: corr %.3f" % corr_res

    # (2) agreement with the histogram on identical landings (monochrome, dispersion off)
    hist = caustics(s, light_dir=tuple(L), receiver_y=recv, extent=0.9, res=64, ior=1.55, n_side=160,
                    window=1.2, center=(0.0, 0.0), refracted_only=True, aim=(0.0, 0.0, 0.0))
    mono = hc.read(xs64, xs64, 550.0)
    # the histogram is grainy at 160^2 rays into 64^2 bins; compare after a matching 3x3 box smooth on the histogram
    hs = hist.copy()
    for _ in range(2):
        hs = (np.roll(hs, 1, 0) + hs + np.roll(hs, -1, 0)) / 3.0
        hs = (np.roll(hs, 1, 1) + hs + np.roll(hs, -1, 1)) / 3.0
    corr_hist = float(np.corrcoef(hs.ravel(), mono.ravel())[0, 1])
    assert corr_hist > 0.75, "bundled and binned landings must agree in where the light is: corr %.3f" % corr_hist
    # THE MECHANISM, pinned: the read is capacity-bounded, so agreement must RISE with dim at a fixed kernel.
    # (If a future change makes this flat or falling, the field has stopped behaving like a hypervector.)
    hc4 = holographic_caustic(s, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=160, window=1.2,
                              center=(0.0, 0.0), dispersion_scale=0.0, dim=4096, bandwidth=(40.0, 40.0, 10.0),
                              seed=0, aim=(0.0, 0.0, 0.0))
    hc1 = holographic_caustic(s, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=160, window=1.2,
                              center=(0.0, 0.0), dispersion_scale=0.0, dim=1024, bandwidth=(40.0, 40.0, 10.0),
                              seed=0, aim=(0.0, 0.0, 0.0))
    c4 = float(np.corrcoef(hs.ravel(), hc4.read(xs64, xs64, 550.0).ravel())[0, 1])
    c1 = float(np.corrcoef(hs.ravel(), hc1.read(xs64, xs64, 550.0).ravel())[0, 1])
    assert c4 > c1 + 0.05, "more dim must buy a better read at a fixed kernel (capacity law): %.3f -> %.3f" % (c1, c4)
    # and the bundle is SMOOTHER than the raw histogram at the same ray count -- the point of a kernel estimate
    def rough(img):
        return float(np.abs(img - (np.roll(img, 1, 0) + np.roll(img, -1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 4).mean()
                     / (img.mean() + 1e-12))
    assert rough(mono) < 0.5 * rough(hist), "the kernel read must be smoother than the histogram: %.3f vs %.3f" % (
        rough(mono), rough(hist))

    # (3) translate is a bind: shift the field by one bind, compare against re-depositing shifted landings
    xz, lam = spectral_landings(s, 1.55, 25.68, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=120,
                                dispersion_scale=0.0, seed=0, aim=(0.0, 0.0, 0.0))
    bw = default_bandwidth(1024)
    h1 = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=1024, bandwidth=bw, seed=0).deposit(xz, lam)
    h2 = HolographicCaustic(((-1.5, 1.5), (-1.5, 1.5)), dim=1024, bandwidth=bw, seed=0).deposit(xz + np.array([0.3, -0.2]), lam)
    h1.translate(0.3, -0.2)
    assert np.allclose(h1.F_spec, h2.F_spec, atol=1e-8 * np.abs(h2.F_spec).max()), \
        "translate must equal re-depositing shifted landings (a bind IS the rigid shift)"

    # (4) dispersion is IN the field, measured with the statistic a LENS actually changes. Blue refracts more,
    # so it focuses CLOSER and lands WIDER on a receiver below the focus: the radial spread of blue vs red grows
    # with dispersion. The centroid does NOT move much (a lens's bulk position is set by its central ray), which
    # is why a first draft of this test, built on centroids, read the same 0.025 at every dispersion -- the
    # sixth metric this arc has discarded for measuring something other than the effect. Raw landings first
    # (the physics, no encoding), then the same statistic on the FIELD's single-wavelength reads.
    def spread_ratio_raw(scale):
        xz_, lam_ = spectral_landings(s, 1.55, 25.68, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=160,
                                      dispersion_scale=scale, seed=0, aim=(0.0, 0.0, 0.0))
        def rms(mask):
            q = xz_[mask]; return float(np.sqrt(((q - q.mean(0)) ** 2).sum(1).mean()))
        return rms(lam_ > 620.0) / rms(lam_ < 480.0)                      # red spread / blue spread
    r0, r3 = spread_ratio_raw(0.0), spread_ratio_raw(3.0)
    assert abs(r0 - 1.0) < 0.10, "with no dispersion red and blue must spread alike: ratio %.3f" % r0
    assert r3 < r0 - 0.10, "with dispersion blue must land WIDER than red (it focuses closer): %.3f -> %.3f" % (r0, r3)

    # THE FIELD carries the wavelength, at the level the dimension supports -- stated as a CONTRAST against the
    # same field with dispersion off, because that is the honest form. Two single-wavelength reads of the same
    # geometry do NOT agree perfectly even with no dispersion: each lambda-slice read has its own crosstalk
    # realisation, and at dim=2048 they correlate at ~0.75 (0.79 at 4096). THAT NUMBER IS THE LAMBDA-AXIS NOISE
    # FLOOR, and the physical dispersion of this glass through a sphere at scale 3 is a ~3%-of-variance effect
    # (red vs blue landing histograms correlate at 0.972). So the field's colour separation is CAPACITY-BOUNDED
    # here: dispersion must lower corr(read 650, read 450) below the no-dispersion floor, and the drop grows with
    # dim (measured 0.009 at 2048, 0.024 at 4096). Raise dim or tile for more; do not read a subtle spectral
    # effect out of a small field and call the crosstalk colour.
    def rb_corr(scale, dim=2048):
        h = holographic_caustic(s, light_dir=tuple(L), receiver_y=recv, extent=0.9, n_side=160, window=1.2,
                                dispersion_scale=scale, dim=dim, seed=0, aim=(0.0, 0.0, 0.0))
        return float(np.corrcoef(h.read(xs64, xs64, 650.0, raw=True).ravel(),
                                 h.read(xs64, xs64, 450.0, raw=True).ravel())[0, 1])
    f0, f3 = rb_corr(0.0), rb_corr(3.0)
    assert f0 > 0.6, "two same-geometry reads must broadly agree (the lambda crosstalk floor): %.3f" % f0
    assert f3 < f0, "dispersion must pull the red and blue reads APART below the no-dispersion floor: %.4f vs %.4f" % (f3, f0)


    print("OK: holographic caustic -- one vector reads consistently at 64^2 and 128^2 (corr %.3f); agrees with the "
          "histogram (corr %.3f) and is %.1fx smoother at equal rays; translate==bind; red-blue separation "
          "raw red/blue spread ratio %.3f -> %.3f at dispersion 0 -> 3; in-field red-blue read correlation "
          "%.4f (none, = the lambda crosstalk floor) -> %.4f (scale 3)" % (
              corr_res, corr_hist, rough(hist) / max(rough(mono), 1e-12), r0, r3, f0, f3))


if __name__ == "__main__":
    _selftest()


class TiledHolographicCaustic:
    """The receiver split into a grid of tiles, each its own HolographicCaustic -- the capacity wall broken the way
    holographic_radiance.TiledRadianceField breaks it.

    WHY. A read of an FPE bundle is signal + crosstalk, and the two scale differently: the signal at a point is the
    landings inside the kernel there; the crosstalk is ~sqrt(N_total / dim), from EVERY landing in the vector.
    So SNR ~ n_local * sqrt(dim) / sqrt(N). Splitting the receiver into T tiles leaves n_local alone and divides N
    by T: SNR grows as sqrt(T), for the same total read cost. Raising dim instead buys only sqrt(dim) at linear
    cost. MEASURED on the 1-fold Mandelbox (39,736 landings, 5.76-wide window): one dim-16384 vector agreed with
    the histogram at 0.576 and showed the caustic arcs under a blotchy crosstalk floor; see the selftest for the
    tiled figure. Each tile's bandwidth is set from its OWN dim, and because the encoder's bandwidth is per unit
    of range, a tile 1/g the width has a kernel 1/g the world width for free -- finer where there are fewer
    landings to hold.

    Landings are deposited into the tile they fall in PLUS a halo of `halo` kernel widths, so a kernel tail that
    crosses a tile border is still summed by the neighbour; without the halo the read has seams at every tile
    edge (the usual tiling lesson, paid once here). Reads route each pixel to its tile."""

    def __init__(self, bounds_xz, grid=4, lam_bounds=(420.0, 680.0), dim=4096, bandwidth=None, halo=2.5, seed=0):
        (xl, xh), (zl, zh) = bounds_xz
        self.bounds = ((float(xl), float(xh)), (float(zl), float(zh)))
        self.grid = int(grid)
        self.lam_bounds = (float(lam_bounds[0]), float(lam_bounds[1]))
        self.dim = int(dim)
        self.bw = default_bandwidth(self.dim) if bandwidth is None else tuple(bandwidth)
        self.halo = float(halo)
        self.seed = int(seed)
        self.tx = np.linspace(xl, xh, self.grid + 1)
        self.tz = np.linspace(zl, zh, self.grid + 1)
        self.tiles = {}                                        # (ix, iz) -> HolographicCaustic, built on demand
        self.n_deposited = 0

    def _tile(self, ix, iz):
        key = (int(ix), int(iz))
        if key not in self.tiles:
            bx = (self.tx[ix], self.tx[ix + 1]); bz = (self.tz[iz], self.tz[iz + 1])
            # one seed per tile, derived deterministically, so tiles do not share a crosstalk realisation
            self.tiles[key] = HolographicCaustic((bx, bz), self.lam_bounds, dim=self.dim, bandwidth=self.bw,
                                                 seed=self.seed * 1009 + ix * 31 + iz)
        return self.tiles[key]

    def deposit(self, xz, lam, weights=None):
        """Route each landing to its tile and to any neighbour within the halo, then bundle."""
        xz = np.atleast_2d(np.asarray(xz, float)); lam = np.asarray(lam, float).reshape(-1)
        w = np.ones(len(xz)) if weights is None else np.asarray(weights, float).reshape(-1)
        wx = (self.tx[1] - self.tx[0]); wz = (self.tz[1] - self.tz[0])
        # kernel sigma in world units on this tile: range / bandwidth (see ScalarEncoder: exp(-bw^2 (scale dx)^2/2))
        hx = self.halo * wx / self.bw[0]; hz = self.halo * wz / self.bw[1]
        for ix in range(self.grid):
            for iz in range(self.grid):
                inx = (xz[:, 0] >= self.tx[ix] - hx) & (xz[:, 0] < self.tx[ix + 1] + hx)
                inz = (xz[:, 1] >= self.tz[iz] - hz) & (xz[:, 1] < self.tz[iz + 1] + hz)
                sel = inx & inz
                if sel.any():
                    self._tile(ix, iz).deposit(xz[sel], lam[sel], w[sel])
        self.n_deposited += len(xz)
        return self

    def _route_read(self, xs, zs, fn, floor_pct=None, blend=None):
        """Assemble the full image tile by tile -- WITH THE LEVER-6 CLEANUP BETWEEN LEVELS.

        The flat tiling (lever 5) read each tile on its own pixels and showed a PATCHWORK: each tile carries its own
        crosstalk realisation, so its floor -- the read where nothing landed -- sits at a different level, and the
        borders are discontinuous. hierarchical_pack's lesson is that a tile is a level, and levels need a cleanup
        step between them (its 'snap to the codebook' is the crosstalk reset). For a continuous field the analogue
        is two operations the coordinator can do because it sees all tiles:
          * FLOOR REMOVAL (`floor_pct`) -- KEPT NEGATIVE, default OFF. The idea: a tile's low percentile of its own
            read is its crosstalk floor; subtract it. MEASURED on the Mandelbox caustic, 16x16 tiles: agreement with
            the true caustic FELL 0.817 -> 0.638, because a caustic fills most of some tiles and the 15th percentile
            there is SIGNAL, not floor. A per-tile statistic cannot tell the two apart; the floor would have to come
            from a genuinely empty probe, which the coordinator does not have here. Left in as an explicit opt-in.
          * CROSS-FADE -- every tile reads a margin past its border (it holds the halo's landings), and neighbours
            are blended with a linear ramp across the overlap (`blend` = fraction of a tile's width). Kills seams.
        Cross-fade is ON by default (measured 0.831 -> 0.843 vs the true caustic and the blockiness gone -- see
        sweep 151); pass blend=0 for the flat lever-5 read. Cost: +10% read time."""
        xs = np.asarray(xs, float); zs = np.asarray(zs, float)
        fp = 0.0 if floor_pct is None else floor_pct       # OFF by default -- see the measurement below
        bl = 0.12 if blend is None else float(blend)
        wx = (self.tx[1] - self.tx[0]); wz = (self.tz[1] - self.tz[0])
        acc = None; wsum = np.zeros((len(zs), len(xs)))
        for (ix, iz), tile in self.tiles.items():
            x0, x1 = self.tx[ix] - bl * wx, self.tx[ix + 1] + bl * wx
            z0, z1 = self.tz[iz] - bl * wz, self.tz[iz + 1] + bl * wz
            cx = np.flatnonzero((xs >= x0) & (xs <= x1)); cz = np.flatnonzero((zs >= z0) & (zs <= z1))
            if not len(cx) or not len(cz):
                continue
            block = np.asarray(fn(tile, xs[cx], zs[cz]), float)
            if fp is not None and fp > 0:
                fl = np.percentile(block, fp, axis=(0, 1)) if block.ndim == 3 else np.percentile(block, fp)
                block = np.clip(block - fl, 0.0, None)                    # the crosstalk reset, per tile
            # separable linear ramps: 1 inside the tile, falling to 0 at the edge of the margin
            def ramp(v, lo, hi, m):
                return np.clip(np.minimum((v - lo) / max(m, 1e-12), (hi - v) / max(m, 1e-12)) + 1.0, 0.0, 1.0) if m > 0 \
                    else np.ones_like(v)
            rx = ramp(xs[cx], self.tx[ix], self.tx[ix + 1], bl * wx); rz = ramp(zs[cz], self.tz[iz], self.tz[iz + 1], bl * wz)
            w = rz[:, None] * rx[None, :]
            if acc is None:
                acc = np.zeros((len(zs), len(xs)) + block.shape[2:], float)
            acc[np.ix_(cz, cx)] += block * (w[..., None] if block.ndim == 3 else w)
            wsum[np.ix_(cz, cx)] += w
        if acc is None:
            return np.zeros((len(zs), len(xs)))
        wn = np.where(wsum > 0, wsum, 1.0)
        return acc / (wn[..., None] if acc.ndim == 3 else wn)

    def read(self, xs, zs, lam, raw=False):
        return self._route_read(xs, zs, lambda t, x, z: t.read(x, z, lam, raw=raw))

    def read_rgb(self, xs, zs, cmf, n_lam=48, normalise="mean", raw=False):
        img = self._route_read(xs, zs, lambda t, x, z: t.read_rgb(x, z, cmf, n_lam=n_lam, normalise=None, raw=raw))
        if normalise == "mean":
            img = img / (img.mean() + 1e-12)
        elif normalise == "max":
            img = img / (img.max() + 1e-12)
        return img


def holographic_caustic_tiled(sdf, n_d=1.55, abbe=25.68, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0,
                              n_side=300, window=None, center=(0.0, 0.0), dispersion_scale=1.0, grid=4, dim=4096,
                              bandwidth=None, lam_lo=420.0, lam_hi=680.0, seed=0, aim=None, emitter_center=None):
    """The tiled build: trace once, bundle into a grid x grid of tile vectors. Same call shape as
    holographic_caustic; returns a TiledHolographicCaustic. Use this when the receiver is wide or the landing
    count is large -- i.e. whenever a single vector's read shows a blotchy floor under the caustic."""
    half = float(extent) if window is None else float(window)
    cx, cz = float(center[0]), float(center[1])
    bounds_xz = ((cx - half, cx + half), (cz - half, cz + half))
    xz, lam = spectral_landings(sdf, n_d, abbe, light_dir=light_dir, receiver_y=receiver_y, extent=extent,
                                n_side=n_side, dispersion_scale=dispersion_scale, lam_lo=lam_lo, lam_hi=lam_hi,
                                seed=seed, refracted_only=True, emitter_center=emitter_center, aim=aim)
    inb = ((xz[:, 0] >= bounds_xz[0][0]) & (xz[:, 0] <= bounds_xz[0][1]) &
           (xz[:, 1] >= bounds_xz[1][0]) & (xz[:, 1] <= bounds_xz[1][1]))
    t = TiledHolographicCaustic(bounds_xz, grid=grid, lam_bounds=(lam_lo, lam_hi), dim=dim, bandwidth=bandwidth,
                                seed=seed)
    t.deposit(xz[inb], lam[inb])
    return t

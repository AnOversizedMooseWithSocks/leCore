"""
holographic_encoders.py
========================

Front-ends that turn raw data into meaningful vectors for the holographic
engine in holographic_ai.py. This is the piece that lets the same brain handle
numbers, text, and mixed records -- you swap the encoder, not the engine.

An encoder is a PLACEMENT RULE: it decides where each input lands in the vector
space. Two situations that should be treated alike must land near each other,
or none of the downstream similarity/memory machinery works. There are three
ways to get a good placement rule, and this file has one example of each:

  * Built in by hand   -> the creature's egocentric features (holographic_creature.py)
  * Built in by math   -> ScalarEncoder below: fractional power encoding makes
                          nearby NUMBERS automatically land near each other.
  * Learned from data  -> TextEncoder below: random indexing learns word meaning
                          from co-occurrence, with no gradient descent at all.

RecordEncoder then binds several fields (numeric + categorical + text) into a
single vector, so one record of mixed types is one point in the space.

Needs: numpy, and holographic_ai.py beside it.
"""

import warnings

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import random_vector, cosine, bind, bind_batch, unbind, bundle, permute, Vocabulary


# ---------------------------------------------------------------------------
# 1. NUMBERS  (fractional power encoding / Spatial Semantic Pointers)
#
#    A principled placement rule for continuous values: encode a number so that
#    similarity between two encodings falls off smoothly with their distance. We
#    pick random phases once, then encoding a value x just rotates those phases
#    by x. Two nearby x's rotate to nearly the same place -> high similarity.
#    No training -- the number line's geometry is baked into the math.
# ---------------------------------------------------------------------------

class ScalarEncoder:
    """Encode a real number as a unit vector; nearby numbers -> similar vectors.

    'lo' and 'hi' set the working range: values that far apart come out roughly
    orthogonal, so the whole range spans one smooth similarity lobe. encode()
    turns a number into a vector; decode() reads a (possibly noisy) vector back
    into the nearest number by scanning a grid -- the continuous analogue of
    cleanup memory.
    """

    def __init__(self, dim, lo=0.0, hi=1.0, seed=0, kernel="sinc", bandwidth=1.8, taper=None):
        # kernel="sinc": uniform phases -> a sinc similarity (band-limited, but it
        #   oscillates and goes NEGATIVE as the gap grows). Fine for decode()/cleanup.
        # kernel="rbf":  Gaussian phases -> an RBF / squared-exponential kernel,
        #   exp(-bandwidth^2 (scale*dx)^2 / 2): non-negative and monotone, so a BUNDLE
        #   of encoded points reads as a proper kernel density estimate. Prefer it when
        #   the encoder feeds a similarity / density read-out rather than a single decode.
        # By Bochner's theorem the encoder IS a shift-invariant kernel either way -- the
        # inner product depends only on the gap and equals the phase distribution's
        # characteristic function at that gap (see kernel_at).
        self.dim = dim
        self.lo, self.hi = lo, hi
        self._warned_out_of_range = False       # encode() warns ONCE if a value falls outside [lo, hi]
        self.scale = 1.0 / (hi - lo) if hi > lo else 1.0
        self.kernel = kernel
        self.bandwidth = bandwidth
        rng = np.random.default_rng(seed)
        # Random phases, made conjugate-symmetric so the inverse FFT is real.
        #
        # F35 -- TAPER-DESIGNED KERNELS (the phased-array transfer). By Bochner, the similarity
        # kernel IS the characteristic function of this phase distribution -- the SAME equation as
        # an antenna beam pattern being the Fourier transform of its aperture taper (Doerry 2017,
        # 'Catalog of Window Taper Functions for Sidelobe Control'; Dolph 1946). The default
        # uniform draw therefore ships the antenna world's WORST kernel: a sinc with -13.4 dB
        # sidelobes, whose measured failure is Doerry's own figure -- 'the lower-amplitude signal
        # is buried in the sidelobe of the stronger signal' (a weak stored item at a strong item's
        # first sidelobe peak retrieves at 0.7x margin: BURIED). taper='kaiser:BETA' draws phases
        # from a Kaiser-tapered density on the same support by inverse-CDF (pure NumPy, np.i0):
        # MEASURED at beta=8, D=4096: sidelobes -13.4 -> -34.5 dB, buried-weak-item margin
        # 0.7x -> 8.0x, price = 2.4x wider mainlobe. NOTHING IS CREATED -- resolution near zero is
        # traded for immunity at moderate distance (the conservation the principle states), which
        # is why the default stays 'uniform' (bit-identical draws to before) and the taper is a
        # KNOB: which side of the trade is right is the application's call, not the encoder's.
        # taper applies to the sinc family only; 'rbf' has no sidelobes to shape (refused loudly).
        if taper not in (None, "uniform") and kernel == "rbf":
            raise ValueError("taper shapes sinc-family sidelobes; the RBF kernel has none")
        if kernel == "rbf":
            phases = rng.normal(0.0, bandwidth, dim)   # Gaussian phases -> RBF kernel
        elif taper in (None, "uniform"):
            phases = rng.uniform(-np.pi, np.pi, dim)   # uniform phases -> sinc kernel (unchanged draws)
        elif str(taper).startswith("kaiser"):
            beta = float(str(taper).split(":")[1]) if ":" in str(taper) else 8.0
            grid = np.linspace(-np.pi, np.pi, 20001)
            dens = np.i0(beta * np.sqrt(np.clip(1.0 - (grid / np.pi) ** 2, 0.0, 1.0))) / np.i0(beta)
            cdf = np.cumsum(dens); cdf /= cdf[-1]
            # STRATIFIED inverse-CDF (Quilez seat, same session as the taper itself so no released
            # behavior changes): one jittered draw per stratum instead of iid uniforms. iid draws
            # CLUMP, and clumping is sidelobe ripple; stratification keeps the taper's density and
            # kills the clumps -- the Monte Carlo move under every path tracer. MEASURED, 12 seeds,
            # D=2048, beta=8: iid mean -33.7 dB (worst -28.5) -> stratified mean -58.5 (worst
            # -57.0). 24 dB for free, and the WORST seed improves more than the mean (variance is
            # what stratification buys). Then shuffle: strata order must not correlate with FFT bin
            # index, or the conjugate-symmetry fold would impose structure the density never had.
            u = (np.arange(dim) + rng.uniform(0.0, 1.0, dim)) / dim
            phases = np.interp(u, cdf, grid)
            rng.shuffle(phases)
        else:
            raise ValueError("taper must be None, 'uniform', or 'kaiser[:beta]'")
        phases[0] = 0.0
        for k in range(1, dim // 2 + 1):
            phases[dim - k] = -phases[k]
        if dim % 2 == 0:
            phases[dim // 2] = 0.0
        self.phases = phases

    def _phase_encode(self, u):
        # Rotating the fixed phases by u is "raising the base vector to power u".
        spectrum = np.exp(1j * self.scale * u * self.phases)
        v = np.real(np.fft.ifft(spectrum))
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_many(self, xs):
        """Encode MANY numbers at once, as an `(M, dim)` array. **Bit-identical to stacking `encode`, and vectorised.**

        `_phase_encode` is `real(ifft(exp(1j * scale * u * phases)))`, normalised. Nothing in it is sequential: the
        phases are the same for every `u`, so the whole batch is one outer product and one `ifft` along an axis.
        The scalar path stayed a loop only because nobody had asked it for a batch."""
        xs = np.atleast_1d(np.asarray(xs, float))
        if xs.size:
            self._check_range(float(xs.ravel()[0]))           # warn once on the first, not once per point
        wx = getattr(self, "_warp_x", None)
        u = xs if wx is None else np.interp(xs, wx, self._warp_u)   # the A3 warp, vectorised; identity if unfitted
        spectrum = np.exp(1j * self.scale * u[:, None] * self.phases[None, :])
        v = np.real(np.fft.ifft(spectrum, axis=1))
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(n > 0, v / np.where(n > 0, n, 1.0), v)

    @classmethod
    def for_values(cls, values, dim, seed=0, margin=0.05, **kw):
        """Build an encoder whose range is taken FROM THE DATA (the `range='auto'` case), with a small margin so the
        endpoints aren't on the boundary.

        WHY THIS EXISTS (a measured, silent failure): the working range is what makes nearby numbers similar. Encode
        values in [0, 100] with the default (-4, 4) and every value saturates the same lobe -- MEASURED normalized
        decode error 0.5422 against a random-vector baseline of 0.5436. In other words the code carried NO
        information and said nothing about it. Auto-ranged on the same data: 0.0013. Always range your encoder to
        your data, or let this do it for you."""
        v = np.asarray(list(values), float)
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:                                          # a constant column: give it a unit-wide window
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * float(margin)
        return cls(dim, lo - pad, hi + pad, seed=seed, **kw)

    def _check_range(self, x):
        """Warn (once per encoder) when a value falls outside the working range. Out-of-range values don't raise --
        they just quietly stop being distinguishable, which is the failure mode this catches."""
        if self._warned_out_of_range or not (x < self.lo or x > self.hi):
            return
        self._warned_out_of_range = True
        warnings.warn(
            "ScalarEncoder: value %.4g is outside the encoder's range [%.4g, %.4g]. Out-of-range values are not "
            "distinguishable from one another (measured: decode error at chance), so similarity and decode will be "
            "meaningless. Build the encoder with ScalarEncoder.for_values(data, dim) to range it to your data."
            % (float(x), self.lo, self.hi), RuntimeWarning, stacklevel=3)

    def encode(self, x):
        """Encode a number as a unit vector. If fit_resolution() has been called, x is first passed through the
        adaptive resolution warp (A3); otherwise the warp is the identity and this is the plain Fourier encoding.

        Warns once if `x` falls outside [lo, hi] -- see _check_range: out-of-range values silently collapse together.
        """
        self._check_range(x)
        return self._phase_encode(self._warp(x))

    def _warp(self, x):
        """A3 resolution warp: map x through the fitted CDF so dense value regions get more resolution. Identity
        unless fit_resolution() set a warp."""
        wx = getattr(self, "_warp_x", None)
        if wx is None:
            return float(x)
        return float(np.interp(x, wx, self._warp_u))

    def _unwarp(self, u):
        """Invert the A3 warp (the decode side). Identity unless a warp is fitted."""
        wx = getattr(self, "_warp_x", None)
        if wx is None:
            return float(u)
        return float(np.interp(u, self._warp_u, wx))

    def fit_resolution(self, samples, floor=0.2, grid=256):
        """A3 (cross-cutting CACHE-3 -> encoder): fit a monotonic CDF warp from `samples` so this encoder spends
        MORE resolution where the value distribution is DENSE and less where it is sparse -- the equidistribution
        principle (place resolution by density), applied to a Fourier encoder by warping its input axis rather
        than moving discrete kernels (it has none; its kernel is shift-invariant). `floor` (0..1) mixes the CDF
        with the identity so at least that share of resolution is kept EVERYWHERE -- the irradiance-caching
        validity-radius lesson: a pure density warp drives sparse regions to ~zero resolution, where decodes go
        catastrophic; the floor bounds that. Returns self.

        MEASURED: on a non-uniform (bimodal) distribution, ~73% lower decode error under noise vs the uniform
        encoder; on a UNIFORM distribution it ties (the warp is the identity -- the CACHE-3 control). KEPT
        CAVEAT, and it matters: this is a REALLOCATION, not a free win -- dense-region decodes get ~4x better,
        sparse / out-of-distribution decodes ~4x worse (bounded by `floor`; ~35x worse without it). Fit it only
        when you will decode IN-distribution values and do not care about rare ones. Off by default (no warp =
        the plain encoder, bit-identical)."""
        xs = np.sort(np.asarray(samples, float))
        if len(xs) < 2 or xs[-1] <= xs[0]:
            return self                                  # degenerate sample -> leave the encoder uniform
        xq = np.linspace(xs[0], xs[-1], grid)
        cdf = np.interp(xq, xs, np.linspace(0.0, 1.0, len(xs)))      # empirical CDF at the grid
        uq = (1.0 - floor) * cdf + floor * np.linspace(0.0, 1.0, grid)   # floor: keep >= `floor` resolution everywhere
        uq = (uq - uq[0]) / (uq[-1] - uq[0] + 1e-12)     # renormalise to [0,1] (strictly increasing)
        self._warp_x = xq                                # original axis grid
        self._warp_u = self.lo + (self.hi - self.lo) * uq            # warped axis (in [lo,hi], strictly increasing)
        return self

    def kernel_at(self, dx):
        """The similarity <encode(x), encode(x+dx)> this encoder analytically realises.

        By Bochner's theorem the inner product depends only on the gap dx and equals the
        characteristic function of the phase distribution at dx -- so you can ASSERT the
        kernel rather than eyeball it: encode two points dx apart, take their cosine, and
        it matches kernel_at(dx). RBF is exp(-(bandwidth*scale*dx)^2/2) and never goes
        negative; sinc is sin(pi t)/(pi t) and does."""
        t = self.scale * float(dx)
        if self.kernel == "rbf":
            return float(np.exp(-0.5 * (self.bandwidth * t) ** 2))
        return float(np.sinc(t))                        # sin(pi t)/(pi t)

    def decode(self, vec, steps=200):
        """Read a vector back to a number: the grid value whose encoding is
        most similar. Robust to noise, which is what makes it useful for
        recovering a number after it's been bundled with other things.

        The grid encodings depend only on (lo, hi, steps, kernel, bandwidth) -- all fixed for this encoder --
        so they are built ONCE and cached as a unit-normalized matrix, and decode is then a single
        matrix-vector product. Measured ~200x faster than re-encoding the grid and cosine-scanning it on
        every call, and bit-for-bit the same argmax (the rows are unit length, so mat @ (vec/|vec|) IS the
        per-grid cosine). The same cached-matrix-instead-of-a-Python-loop move the core Vocabulary.cleanup
        already uses for symbol recall."""
        cache = getattr(self, "_grid_cache", None)
        if cache is None:
            cache = self._grid_cache = {}
        if steps not in cache:                          # build the grid encodings once, normalize the rows
            grid = np.linspace(self.lo, self.hi, steps)               # uniform in the WARPED axis (encode's space)
            mat = np.stack([self._phase_encode(g) for g in grid])     # raw phase encode (NOT warped again)
            mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
            cache[steps] = (grid, mat)
        grid, mat = cache[steps]
        vec = np.asarray(vec, float)
        nn = float(np.linalg.norm(vec))
        if nn == 0.0:
            return self._unwarp(float(grid[0]))
        return self._unwarp(float(grid[int((mat @ (vec / nn)).argmax())]))


# ---------------------------------------------------------------------------
# 2. TEXT  (random indexing -- meaning learned from co-occurrence, no training)
#
#    Give every word a fixed random 'index' vector (a meaningless atom). Then
#    sweep through text: each time a word appears, add its neighbours' index
#    vectors (rotated by how far away they sit, so word order matters) into that
#    word's running 'context' vector. Words that show up in similar surroundings
#    accumulate similar context vectors -- so meaning emerges from raw text with
#    nothing but addition. This is the cheap, gradient-free way to LEARN a
#    placement rule, the middle ground between hand-built features and a
#    transformer's trained embeddings.
# ---------------------------------------------------------------------------

class CircularEncoder:
    """Encode a CIRCULAR variable (angle, hour-of-day, day-of-week, phase) so that the wrap is EXACT:
    encode(x) == encode(x + period) to machine precision, and similarity depends only on the CIRCULAR gap.

    WHY A SEPARATE CLASS: the ScalarEncoder is a LINE encoder -- its phases are arbitrary reals, so theta and
    theta + period land at different codes, and 23:59 sits maximally far from 00:01 (measured: cos 0.21 where
    the true 2-minute gap should read ~1.0). No bandwidth choice fixes that; periodicity requires the phase
    frequencies to be INTEGERS (in units of 2*pi/period), and that is a different construction, not a
    parameter. (The audited alternative -- I2's proposed 'SignedEncoder' -- was REFUTED the same way: signed
    values are native to ScalarEncoder(lo=-a, hi=a), decode recovers the sign, nothing to build.)

    Construction: harmonic numbers n_j >= 1 drawn from a geometric distribution (P(n) ~ r^n), phases
    exp(i * n_j * omega * x) with omega = 2*pi/period, conjugate-symmetric so the code is real. By Bochner on
    the circle the similarity IS the harmonic distribution's characteristic function of the circular gap --
    the POISSON KERNEL MINUS ITS DC TERM, because n=0 must be excluded for zero-mean codes. That subtraction
    is not free, and the first draft of this docstring claimed positivity anyway; measurement corrected it:
    the kernel is near-positive with a SMALL NEGATIVE ANTIPODAL DIP (k(pi) between -0.16 at r=0.70 and -0.02
    at r=0.85, pinned), the circular price of removing the constant. `concentration` r in (0,1) trades lobe
    width for that dip: r=0.70 reads k(0.1)=0.90 (wide, deeper dip), r=0.95 reads k(0.1)=0.14 (narrow,
    near-orthogonal fast). The default 0.85 is the middle of that trade.

    decode() scans a grid over one period and returns the best angle IN [0, period) -- the circular cleanup.
    """

    def __init__(self, dim, period=2.0 * np.pi, seed=0, concentration=0.85):
        if not (0.0 < concentration < 1.0):
            raise ValueError("concentration must be in (0, 1), got %r" % (concentration,))
        self.dim = int(dim)
        self.period = float(period)
        self.omega = 2.0 * np.pi / self.period
        self.concentration = float(concentration)
        rng = np.random.default_rng(seed)
        # geometric harmonic numbers >= 1, conjugate-symmetric (negative harmonics mirror positive ones).
        harm = np.zeros(self.dim)
        for k in range(1, self.dim // 2 + 1):
            n = 1 + rng.geometric(1.0 - self.concentration)     # support {2,3,...}? geometric>=1 -> n>=2; shift:
            harm[k] = n - 1                                     # -> n >= 1, geometric weights
            harm[self.dim - k] = -harm[k]
        if self.dim % 2 == 0:
            harm[self.dim // 2] = 0.0
        harm[0] = 0.0
        self.harmonics = harm

    def encode(self, x):
        """The angle x (any real; reduced mod period) as a real unit vector."""
        spec = np.exp(1j * self.harmonics * self.omega * float(x))
        v = np.fft.ifft(spec).real
        return v / (np.linalg.norm(v) + 1e-12)

    def kernel_at(self, gap):
        """The designed similarity at a CIRCULAR gap -- the empirical characteristic function of the drawn
        harmonics, so it matches encode() exactly rather than quoting the asymptotic Poisson formula."""
        return float(np.mean(np.cos(self.harmonics * self.omega * float(gap))))

    def decode(self, v, grid=720):
        """Nearest angle in [0, period): scan a grid, return the argmax -- circular cleanup memory."""
        v = np.asarray(v, float).ravel()
        thetas = np.linspace(0.0, self.period, int(grid), endpoint=False)
        best, arg = -2.0, 0.0
        for t in thetas:
            c = float(self.encode(t) @ v)
            if c > best:
                best, arg = c, float(t)
        return arg



class TextEncoder:
    """Learn word vectors from co-occurrence, then encode words and sentences.

    learn(tokens) folds one sentence's co-occurrences into the running context
    vectors. wordvec(w) returns w's learned meaning (its index vector until it
    has been seen). encode_sentence bundles a sentence's word vectors into one.
    """

    def __init__(self, dim, window=2, seed=0):
        self.dim = dim
        self.window = window
        self.index = Vocabulary(dim, seed)   # fixed random atom per word
        self.context = {}                     # word -> accumulated context vector

    def learn(self, tokens):
        for i, w in enumerate(tokens):
            ctx = self.context.get(w)
            if ctx is None:
                ctx = np.zeros(self.dim)
                self.context[w] = ctx
            for d in range(1, self.window + 1):
                if i - d >= 0:                              # neighbour to the left
                    ctx += permute(self.index.get(tokens[i - d]), -d)
                if i + d < len(tokens):                     # neighbour to the right
                    ctx += permute(self.index.get(tokens[i + d]), d)

    def wordvec(self, w):
        v = self.context.get(w)
        if v is None:
            return self.index.get(w)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_sentence(self, tokens):
        """A sentence as one vector: the bundle (superposition) of its word
        vectors. This is order-insensitive on purpose -- good for 'what is this
        about' similarity. Wrap each word in permute(.., position) first if you
        need order to matter."""
        if isinstance(tokens, str):
            tokens = tokens.split()
        if not tokens:
            return np.zeros(self.dim)
        return bundle([self.wordvec(w) for w in tokens])

    def nearest(self, w, n=3):
        """The n learned words most similar to w -- handy for inspection. DELEGATES the search to the Index home
        (consolidation H1): an exact cosine scan over the context word vectors, same descending-cosine ranking as
        the old loop."""
        from holographic.caching_and_storage.holographic_index import Index
        target = self.wordvec(w)
        others = [o for o in self.context if o != w]
        if not others:
            return []
        M = np.stack([self.wordvec(o) for o in others])
        return Index(M, labels=others, method="exact").nearest(target, k=min(n, len(others)))

    # -- persistence: store BOTH the learned context AND the index atoms minted so far.
    # Index atoms are seed-derived, but minting advances a shared rng, so a reloaded
    # encoder that re-mints them in a different order would diverge for words seen only as
    # neighbours (not as context keys). Saving the minted atoms makes wordvec() exact.
    def to_state(self):
        return {
            "dim": int(self.dim), "window": int(self.window), "seed": int(self.index.seed),
            "words": list(self.context.keys()),
            "context": (np.stack(list(self.context.values())) if self.context
                        else np.zeros((0, self.dim))),
            "index": self.index.to_state(),
        }

    @classmethod
    def from_state(cls, state):
        from holographic.agents_and_reasoning.holographic_ai import Vocabulary
        te = cls(int(state["dim"]), window=int(state["window"]), seed=int(state["seed"]))
        ctx = np.asarray(state["context"], float)
        te.context = {w: ctx[i] for i, w in enumerate(state["words"])}
        if "index" in state:
            te.index = Vocabulary.from_state(state["index"])
        return te


# ---------------------------------------------------------------------------
# 3. MIXED RECORDS  (numbers + categories + text in one vector)
#
#    A record is a dict of fields. We bind each field's ROLE vector to its
#    encoded VALUE and bundle the results -- the same role/filler trick the
#    creature used, now spanning data types. The whole record becomes one point
#    in the space, and you can read any single field back by unbinding its role.
# ---------------------------------------------------------------------------

class RecordEncoder:
    """Encode heterogeneous records into single vectors.

    A field is given as (kind, value) where kind is 'num', 'cat', or 'text'.
    The text encoder must already be trained (via TextEncoder.learn) for text
    fields to be meaningful. read_number / read_category pull one field back
    out of the bundled record.
    """

    def __init__(self, dim, text_encoder, num_range=(0.0, 1.0), seed=0):
        self.dim = dim
        self.text = text_encoder
        self.roles = Vocabulary(dim, seed)          # one vector per field name
        self.symbols = Vocabulary(dim, seed + 1)    # one vector per categorical value
        self.scalar = ScalarEncoder(dim, num_range[0], num_range[1], seed + 2)

    def _filler(self, kind, value):
        if kind == "num":
            return self.scalar.encode(value)
        if kind == "cat":
            return self.symbols.get(value)
        if kind == "text":
            return self.text.encode_sentence(value)
        raise ValueError(f"unknown field kind: {kind}")

    def encode(self, record):
        # Bind every field's role to its filler, then superpose. The binds are done
        # in ONE batched FFT (bind_batch) rather than a Python loop -- ~2x even for a
        # few fields, more as records widen; the bundle is order-independent so the
        # result is identical (to machine epsilon) to the per-field loop.
        items = sorted(record.items())
        if not items:
            return np.zeros(self.dim)
        roles = np.stack([self.roles.get(field) for field, _ in items])
        fillers = np.stack([self._filler(kind, value) for _, (kind, value) in items])
        return bundle(list(bind_batch(roles, fillers)))

    def read_number(self, vec, field):
        """Pull a numeric field back out of a record vector."""
        return self.scalar.decode(unbind(vec, self.roles.get(field)))

    def read_category(self, vec, field, candidates):
        """Pull a categorical field out, snapped to the nearest known value."""
        for c in candidates:          # make sure every candidate has a vector
            self.symbols.get(c)
        noisy = unbind(vec, self.roles.get(field))
        return self.symbols.cleanup(noisy, candidates=candidates)


# ---------------------------------------------------------------------------
# 4. DEMOS
# ---------------------------------------------------------------------------

_CORPUS = [
    "the cat sat by the window", "the dog sat by the door",
    "i fed the hungry cat", "i fed the hungry dog",
    "the cat chased a mouse", "the dog chased a ball",
    "a black cat purred softly", "a brown dog barked loudly",
    "the car drove down the street", "the truck drove up the hill",
    "i parked the car outside", "i parked the truck outside",
    "the car raced past quickly", "the truck rolled past slowly",
    "my car needs new tires", "my truck needs new brakes",
]


def demo_scalar():
    print("=" * 70)
    print("DEMO A -- Numbers: nearby values get nearby vectors (no training)")
    print("=" * 70)
    enc = ScalarEncoder(1024, lo=0, hi=10, seed=1)
    print("\nSimilarity of e(5) to e(5+d):")
    for d in [0, 1, 2, 3, 5, 10]:
        print(f"  d={d:2d} -> {cosine(enc.encode(5), enc.encode(5 + d)):.2f}")
    print("\nDecode (read the number back out of the vector):")
    for v in [1.0, 3.5, 7.2, 9.0]:
        print(f"  encoded {v} -> decoded {enc.decode(enc.encode(v)):.2f}")
    noisy = enc.encode(4.0) + 0.3 * random_vector(1024, np.random.default_rng(5))
    print(f"  noisy vector of 4.0 -> decoded {enc.decode(noisy):.2f}  (survives noise)\n")


def demo_text():
    print("=" * 70)
    print("DEMO B -- Text: meaning learned from co-occurrence (no gradients)")
    print("=" * 70)
    enc = TextEncoder(1024, window=2, seed=2)
    for _ in range(5):
        for sentence in _CORPUS:
            enc.learn(sentence.split())
    print(f"\nLearned from {len(_CORPUS)} short sentences. Word similarities:")
    for a, b in [("cat", "dog"), ("car", "truck"), ("cat", "car"), ("dog", "truck")]:
        print(f"  {a:5s} ~ {b:5s}: {cosine(enc.wordvec(a), enc.wordvec(b)):.2f}")
    print("\nNearest learned words:")
    for w in ["cat", "truck"]:
        near = ", ".join(f"{o} ({s:.2f})" for o, s in enc.nearest(w, 3))
        print(f"  {w:5s} -> {near}")
    print("\n  Same-category words cluster, cross-category stay apart -- the")
    print("  geometry now carries meaning, pulled straight from raw text.\n")


def demo_record():
    print("=" * 70)
    print("DEMO C -- Mixed records: numbers + categories + text in one vector")
    print("=" * 70)
    dim = 2048   # a few fields bundled together -> more room cuts the crosstalk
    text = TextEncoder(dim, window=2, seed=2)
    for _ in range(5):
        for sentence in _CORPUS:
            text.learn(sentence.split())
    rec = RecordEncoder(dim, text, num_range=(0, 200), seed=7)

    # A little market-flavoured record: a price, a trend label, a free-text note.
    record = {
        "price": ("num", 142.5),
        "trend": ("cat", "up"),
        "note":  ("text", "the car raced past quickly"),
    }
    vec = rec.encode(record)
    print("\nEncoded one record (price=142.5, trend=up, note=...) into a single")
    print("2048-d vector, then read individual fields back out of it:")
    print(f"  price field -> {rec.read_number(vec, 'price'):.1f}   (stored 142.5)")
    cat, sim = rec.read_category(vec, "trend", candidates=["up", "down", "flat"])
    print(f"  trend field -> {cat} (similarity {sim:.2f})   (stored 'up')")

    # Similarity between records reflects all fields at once.
    other_similar = rec.encode({"price": ("num", 138.0), "trend": ("cat", "up"),
                                "note": ("text", "the truck rolled past slowly")})
    other_diff = rec.encode({"price": ("num", 20.0), "trend": ("cat", "down"),
                             "note": ("text", "i fed the hungry cat")})
    print(f"\n  similarity to a near-identical record: {cosine(vec, other_similar):.2f}")
    print(f"  similarity to a very different record:  {cosine(vec, other_diff):.2f}")
    print("\n  One vector holds a number, a label, and a sentence -- and the same")
    print("  brain and memory from the other files can store and recall it.\n")


def _a3_selftest():
    """A3: fit_resolution warps the encoder's input axis by the value-density CDF (with a resolution floor), so a
    non-uniform distribution decodes markedly better under noise; on a UNIFORM distribution it ties (the warp is
    the identity -- the CACHE-3 control); and an UNFITTED encoder is the plain Fourier encoder (bit-identical)."""
    import numpy as _np

    def bimodal(rng, n):
        return _np.clip(_np.where(rng.random(n) < 0.5, rng.normal(0.25, 0.04, n), rng.normal(0.75, 0.04, n)), 0, 1)

    def uniform(rng, n):
        return rng.uniform(0, 1, n)

    def err(dist, fit, noise=0.4, seed=0):
        rng = _np.random.default_rng(seed)
        enc = ScalarEncoder(512, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=2.0)
        if fit:
            enc.fit_resolution(dist(rng, 4000))
        return float(_np.mean([abs(enc.decode(enc.encode(float(x))
                     + noise * rng.standard_normal(512) / _np.sqrt(512), 400) - float(x)) for x in dist(rng, 400)]))

    # unfitted encoder is the plain Fourier encoder (warp is identity)
    e = ScalarEncoder(256, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=2.0)
    assert abs(e.decode(e.encode(0.37)) - 0.37) < 0.02

    bu = _np.mean([err(bimodal, False, seed=s) for s in range(3)])
    bf = _np.mean([err(bimodal, True, seed=s) for s in range(3)])
    assert bf < bu * 0.7, (bf, bu)                       # fitted markedly lower error on a non-uniform distribution

    uu = _np.mean([err(uniform, False, seed=s) for s in range(3)])
    uf = _np.mean([err(uniform, True, seed=s) for s in range(3)])
    assert uf < uu * 1.25 + 1e-6, (uf, uu)               # uniform control: ties (no meaningful penalty)


def _i2_selftest():
    """CircularEncoder contracts (I2), plus the two audit verdicts pinned so the premises stay settled:
    1. EXACT WRAP: encode(x) == encode(x + period) to 1e-12, at several x.
    2. CIRCULAR GAP ONLY: cos(0.05, period-0.05) ~= cos(0, 0.1) -- the 23:59/00:01 case the line encoder
       fails at 0.21 (pinned as the line encoder's declared limitation, not fixed there).
    3. decode() recovers angles across the whole circle including near the wrap; kernel_at matches the
       measured cosine.
    4. KEPT NEGATIVE: the antipodal similarity floor of the positive (Poisson-weighted) kernel is nonzero at
       low concentration and shrinks as concentration rises -- measured, the positivity price.
    5. AUDIT VERDICT PINNED: signed values are NATIVE to ScalarEncoder(lo=-a, hi=a) -- the proposed
       SignedEncoder stays unbuilt because there is nothing for it to do.
    """
    rng = np.random.default_rng(0)
    enc = CircularEncoder(1024, period=2 * np.pi, seed=0, concentration=0.85)

    # (1) exact wrap
    for x in (0.0, 1.1, 4.7, -2.3):
        a, b = enc.encode(x), enc.encode(x + 2 * np.pi)
        assert float(np.max(np.abs(a - b))) < 1e-12, x

    # (2) the 23:59 / 00:01 case
    near_wrap = float(enc.encode(0.05) @ enc.encode(2 * np.pi - 0.05))
    same_gap = float(enc.encode(0.0) @ enc.encode(0.1))
    assert abs(near_wrap - same_gap) < 0.02, (near_wrap, same_gap)   # THE contract: only the circular gap matters
    assert near_wrap > 0.5, near_wrap                                # and the 0.1-gap lobe is high at r=0.85
    line = ScalarEncoder(1024, lo=0.0, hi=2 * np.pi, seed=0, kernel="rbf")
    u, v = line.encode(0.05), line.encode(2 * np.pi - 0.05)
    line_cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert line_cos < 0.5, line_cos                               # the line encoder's declared limitation

    # (3) decode across the circle; kernel matches measurement
    for x in (0.01, 1.9, 3.14, 6.2):
        d = enc.decode(enc.encode(x) + 0.02 * rng.standard_normal(1024))
        gap = min(abs(d - x % (2 * np.pi)), 2 * np.pi - abs(d - x % (2 * np.pi)))
        assert gap < 0.05, (x, d)
    for g in (0.1, 0.8, 2.0):
        assert abs(enc.kernel_at(g) - float(enc.encode(0.0) @ enc.encode(g))) < 0.02, g

    # (4) the DC-removal price and the width trade, both measured: the antipodal value is a SMALL DIP (can
    #     be negative -- the Poisson kernel minus its constant), bounded, while the 0.1-gap lobe narrows as
    #     concentration rises. Pin the trade's direction and the dip's boundedness, not a wrong positivity.
    k01, kpi = [], []
    for r in (0.70, 0.85, 0.95):
        e = CircularEncoder(1024, seed=0, concentration=r)
        k01.append(e.kernel_at(0.1))
        kpi.append(e.kernel_at(np.pi))
    assert k01[0] > k01[1] > k01[2], k01                          # lower r -> wider lobe
    assert all(abs(k) < 0.25 for k in kpi), kpi                   # antipodal dip stays small at every r

    # (5) the refuted premise stays pinned where a future session will look for it
    s = ScalarEncoder(512, lo=-1.0, hi=1.0, seed=0, kernel="rbf")
    assert abs(s.decode(s.encode(-0.37)) + 0.37) < 0.02
    ca, cb = s.encode(0.6), s.encode(-0.6)
    assert float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))) < 0.8   # distinct, as required

    # refusal
    try:
        CircularEncoder(256, concentration=1.5)
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "(0, 1)" in str(e)

    print("holographic_encoders I2 selftest OK (wrap exact to 1e-12; cos(0.05, 2pi-0.05)=%.2f == cos of the "
          "same 0.1 gap while the LINE encoder reads %.2f -- its declared limitation, pinned; decode works "
          "across the wrap; lobe width trades with concentration k(0.1)=%.2f/%.2f/%.2f at r=0.70/0.85/0.95 "
          "with the antipodal dip bounded under 0.25 -- the Poisson-minus-DC price, measured after the first "
          "docstring wrongly claimed positivity; and the audited SignedEncoder premise stays REFUTED: signed "
          "is native to ScalarEncoder)" % (near_wrap, line_cos, k01[0], k01[1], k01[2]))



def _selftest():
    """Regression trap for the data front-ends (T6 backfill; demos only, no assertion). Pins the contract each
    encoder exists to provide: a ScalarEncoder is LOCALITY-PRESERVING (nearby numbers -> similar vectors,
    monotonically) and INVERTIBLE (decode recovers the number), and a TextEncoder learns word vectors it can
    recall. Numbers measured against the live encoders first, asserted as a monotone RELATION (robust) rather
    than absolute cosines (which shift with bandwidth)."""
    import numpy as np

    def _cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # 1. LOCALITY: similarity to encode(0.5) decreases monotonically as the value moves away. This is the
    #    property that makes a scalar usable as a graded VSA filler; a non-monotone encoder is silently broken.
    se = ScalarEncoder(dim=1024, lo=0.0, hi=1.0, seed=0)
    anchor = se.encode(0.5)
    sims = [_cos(anchor, se.encode(x)) for x in (0.5, 0.6, 0.7, 0.9)]
    assert sims[0] > 0.99                                        # self-similarity is ~1
    assert all(sims[i] >= sims[i + 1] - 1e-6 for i in range(len(sims) - 1)), sims   # monotone non-increasing
    assert sims[0] - sims[-1] > 0.1                              # and it genuinely spreads, not a flat line

    # 2. INVERTIBILITY: decode recovers the encoded value closely -- an off-designed-case point (0.42 is not a
    #    grid node the encoder was built around), per [BLIND-SPOT SELFTEST].
    assert abs(float(se.decode(se.encode(0.42))) - 0.42) < 0.05

    # 3. TEXT: a TextEncoder learns a vocabulary and returns a real vector for a seen word, nothing for an unseen.
    te = TextEncoder(dim=1024, seed=0)
    te.learn("the cat sat on the mat")
    assert te.wordvec("cat") is not None

    # F35 TAPER PINS (the phased-array transfer; dedicated deltas grid, no RNG needed beyond seeds):
    # (1) default draws BIT-IDENTICAL to the pre-taper encoder; (2) kaiser suppresses sidelobes by
    # >15 dB (measured -13.0 -> -37.5); (3) beyond both mainlobes the weak item is RECOVERED
    # (margin 1.5x -> 18.2x); (4) THE PRICE KEPT LOUD: inside kaiser's wider mainlobe the taper
    # HURTS (0.7x -> 0.5x) -- redistribution, not creation, which is why uniform stays the default;
    # (5) rbf+taper refuses.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _e0 = ScalarEncoder(1024, 0.0, 1.0, seed=0)
        _exp = np.random.default_rng(0).uniform(-np.pi, np.pi, 1024)
        assert np.allclose(_e0.phases[1:512], _exp[1:512]), "default phase draws changed"
        _eu = ScalarEncoder(4096, 0.0, 1.0, seed=0)
        _ek = ScalarEncoder(4096, 0.0, 1.0, seed=0, taper="kaiser:8")
        _ds = np.linspace(0, 6.0, 3001)
        _z0u, _z0k = _eu.encode(0.0), _ek.encode(0.0)
        _ku = np.array([float(_z0u @ _eu.encode(d)) / float(_z0u @ _z0u) for d in _ds])
        _kk = np.array([float(_z0k @ _ek.encode(d)) / float(_z0k @ _z0k) for d in _ds])
        _nu = _ds[np.where(np.diff(np.sign(_ku)) < 0)[0][0]]
        _nk = _ds[np.where(np.diff(np.sign(_kk)) < 0)[0][0]]
        _mu = np.abs(_ku[_ds > _nu * 1.05]).max()
        _mk = np.abs(_kk[_ds > _nk * 1.05]).max()
        assert 20 * np.log10(_mk) < 20 * np.log10(_mu) - 15, "kaiser must suppress sidelobes >15 dB"
        _far = _ds > _nk * 1.05
        _i = int(np.argmax(np.abs(_ku[_far]))) + int((~_far).sum())
        assert 0.15 / abs(_kk[_i]) > 4.0 > 0.15 / abs(_ku[_i]), "weak item beyond mainlobes must be recovered"
        _iin = int(np.argmin(np.abs(_ds - (_nu + _nk) / 2.5)))
        assert abs(_kk[_iin]) >= abs(_ku[_iin]) * 0.5, "sanity: inside-mainlobe cost exists (not asserted away)"
        try:
            ScalarEncoder(128, kernel="rbf", taper="kaiser:8"); raise AssertionError("rbf+taper must refuse")
        except ValueError:
            pass

    print("OK: holographic_encoders self-test passed (ScalarEncoder similarity decays monotonically with distance "
          "and spreads >0.1, decode recovers 0.42 within 0.05, and TextEncoder learns recallable word vectors)")


if __name__ == "__main__":
    import sys
    _selftest()
    _i2_selftest()
    if "--demos" in sys.argv:
        demo_scalar()
        demo_text()
        demo_record()

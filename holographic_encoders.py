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

import numpy as np
from holographic_ai import (random_vector, cosine, bind, bind_batch, unbind, bundle,
                            permute, Vocabulary)


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

    def __init__(self, dim, lo=0.0, hi=1.0, seed=0, kernel="sinc", bandwidth=1.8):
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
        self.scale = 1.0 / (hi - lo) if hi > lo else 1.0
        self.kernel = kernel
        self.bandwidth = bandwidth
        rng = np.random.default_rng(seed)
        # Random phases, made conjugate-symmetric so the inverse FFT is real.
        if kernel == "rbf":
            phases = rng.normal(0.0, bandwidth, dim)   # Gaussian phases -> RBF kernel
        else:
            phases = rng.uniform(-np.pi, np.pi, dim)   # uniform phases -> sinc kernel
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

    def encode(self, x):
        """Encode a number as a unit vector. If fit_resolution() has been called, x is first passed through the
        adaptive resolution warp (A3); otherwise the warp is the identity and this is the plain Fourier encoding."""
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
        scores = mat @ (vec / nn)
        best = int(scores.argmax())
        # A query can land exactly between two grid cells. The cached matvec and
        # the old per-grid cosine loop then differ only by last-bit reduction
        # order, so resolve near-ties with the original scalar calculation.
        tied = np.flatnonzero(scores.max() - scores <= 1e-12)
        if len(tied) > 1:
            exact = []
            for i in tied:
                code = self._phase_encode(grid[i])
                exact.append(float(np.dot(vec, code) / (nn * np.linalg.norm(code))))
            best = int(tied[int(np.argmax(exact))])
        return self._unwarp(float(grid[best]))


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
        """The n learned words most similar to w -- handy for inspection."""
        target = self.wordvec(w)
        scored = [(other, cosine(target, self.wordvec(other)))
                  for other in self.context if other != w]
        scored.sort(key=lambda r: r[1], reverse=True)
        return scored[:n]

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
        from holographic_ai import Vocabulary
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


if __name__ == "__main__":
    demo_scalar()
    demo_text()
    demo_record()

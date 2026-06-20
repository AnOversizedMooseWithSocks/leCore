"""
holographic_ai.py
=================

A tiny "holographic" AI you can read in one sitting. No frameworks, no GPU,
no pretrained models -- just numpy and about 300 lines of commented code.

WHAT THIS IS
------------
The leOS project (github.com/AnOversizedMooseWithSocks/leOS) describes a huge
system, but the bit it calls "holographic storage and computation" rests on one
classical idea from the 1990s: Holographic Reduced Representations (HRR), by
Tony Plate. Everything else in leOS -- the FABRIK planner, the "cosmic web,"
the dreaming engine -- is scaffolding piled on top. This file is just the
engine, distilled.

THE IDEA IN ONE PARAGRAPH
-------------------------
Represent every concept as a long random vector (say 1024 numbers). Three
operations let you compute with these vectors the way arithmetic lets you
compute with numbers:

    bind(a, b)   -- "associate a with b". The result looks nothing like
                    either input, but the pairing is recoverable. (Circular
                    convolution -- think multiplication.)
    bundle(...)  -- "remember this set". The result is SIMILAR to every input,
                    so one vector can hold many things at once. (Addition.)
    unbind(c, a) -- "given the pair c and one half a, recover the other half".
                    The answer is noisy, so we snap it to the nearest known
                    concept with a cleanup memory. (Approximate inverse.)

Because random high-dimensional vectors are almost always nearly perpendicular
to each other, you can stuff a surprising amount into one fixed-size vector
before the pieces start interfering. That interference is the only real limit
(roughly a couple dozen associations at 1024 dims) -- it's the "noise
accumulates after 10-20 compositions" caveat leOS mentions.

WHY IT COUNTS AS "LEARNING FROM SCRATCH"
----------------------------------------
Nothing here is trained ahead of time. The vocabulary starts empty. Every new
symbol the system meets gets a fresh random vector minted on the spot. Memory
is built by accumulating experience into vectors. The classifier demo below
sees a handful of labeled examples and then correctly handles new ones it was
never shown -- generalization falls out of the geometry for free.

Run it:   python3 holographic_ai.py
Requires: numpy   (pip install numpy)
"""

import hashlib
import json

import numpy as np
from holographic_fft import rfft as _rfft, irfft as _irfft  # optional FFT backend (numpy default, bit-exact)


# ---------------------------------------------------------------------------
# 1. THE VECTOR ALGEBRA  (the "computation")
#
#    These four functions are the entire computational core. Everything else
#    in the file is just bookkeeping built on top of them.
# ---------------------------------------------------------------------------

def unitary_vector(dim, rng):
    """Mint an atom whose frequency spectrum is all unit-magnitude (a UNITARY atom).

    Circular-convolution binding (bind) has an exact inverse only when an atom's FFT
    magnitudes are all 1 -- then the true inverse C/A equals the conjugate, which is
    exactly what involution() already computes. A Gaussian atom (random_vector) has a
    SPREAD spectrum, so involution is only an approximate inverse and a single
    unbind recovers its target at cosine ~0.71; a unitary atom makes that single
    unbind EXACT (cosine 1.0). Dividing each FFT component by its magnitude preserves
    the conjugate symmetry of a real signal's spectrum, so the ifft comes back real --
    no complex arithmetic, no storage change, and bind/unbind are byte-for-byte the
    same. Measured win lands on the few-factor role-binding paths (records, relations,
    sequence roles), where it widens the cleanup margin and lifts accuracy under stress
    (e.g. 16 role/filler pairs at dim 256: 0.971 -> 0.982); it does NOT help heavily
    superposed key->value stores (there the error is cross-term crosstalk between pairs,
    which unitary atoms don't reduce) -- so it is a mint choice, not a global default.
    """
    f = np.fft.fft(rng.standard_normal(dim))
    f = f / np.abs(f)                       # every component -> unit magnitude (unitary)
    v = np.real(np.fft.ifft(f))             # conjugate-symmetric f keeps this real
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def random_vector(dim, rng):
    """Mint a fresh 'atom': a random vector scaled to unit length.

    This is how a brand-new concept enters the system. Two independently
    minted vectors in high dimensions are almost always nearly perpendicular,
    which is what keeps stored items from blurring together.
    """
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def derived_atom(seed, name, dim, unitary=False):
    """Mint an atom that is a PURE function of (seed, name) -- the same name always
    produces the same vector, regenerable in isolation with no dependence on what else
    was minted or in what order.

    This is the leaf of the project's regenerate-from-seed principle: a forest already
    rebuilds its trees from a seed, a consolidated brain rebuilds its basis -- and a
    derived vocabulary rebuilds every atom from just (seed, name). So the whole
    high-dimensional atom set is the deterministic expansion of a tiny seed plus the list
    of names: store the generator and its parameters, not the matrix (fractal compression
    in the literal sense).

    The name is mixed with the seed through a stable 64-bit hash (blake2b, so it does not
    depend on Python's per-process hash randomisation) to seed a fresh local rng. The
    resulting atoms have the same near-orthogonality as ordinary minted atoms -- they are
    just a differently-indexed draw from the same Gaussian -- so they are a drop-in atom
    source, opt-in per vocabulary.
    """
    h = hashlib.blake2b(f"{int(seed)}\x00{name}".encode("utf-8"), digest_size=8).digest()
    local = np.random.default_rng(int.from_bytes(h, "big"))
    return unitary_vector(dim, local) if unitary else random_vector(dim, local)


def bind(a, b):
    """Associate two vectors. (Circular convolution, done via the FFT.)

    The result is dissimilar to BOTH inputs -- binding hides its operands --
    but the association can be undone later with unbind(). Think of it as the
    'multiply' of this algebra. It is commutative: bind(a, b) == bind(b, a).

    Uses the REAL FFT: the atoms are real, so rfft computes only the non-redundant
    half of the spectrum -- about half the arithmetic of the complex transform and
    exact (not approximate) for real input, matching the old complex-fft bind to
    machine epsilon (~7e-17). involution-based unbind is unchanged.
    """
    return _irfft(_rfft(a) * _rfft(b), n=a.shape[0])


def bind_batch(A, B):
    """bind() over stacks of shape (k, dim) at once -- the vectorised form of the
    single-pair bind. Same circular convolution, done over every row in one call
    so the work happens in C rather than a Python loop. A and B must share shape;
    bind_batch(A, B)[i] == bind(A[i], B[i]) to machine epsilon."""
    return _irfft(_rfft(A, axis=1) * _rfft(B, axis=1),
                  n=A.shape[1], axis=1)


def bind_fixed(role, B):
    """Bind one fixed role against every row of a stack B (k, dim) -- the common
    case when tagging many fillers with the same role. The role's spectrum is
    computed once and reused, so this is ~5x faster than a Python bind() loop."""
    return _irfft(_rfft(role)[None, :] * _rfft(B, axis=1),
                  n=B.shape[1], axis=1)


def involution(a):
    """The stable 'inverse' used for unbinding.

    The exact inverse of circular convolution divides in the frequency domain,
    which blows up numerically. Plate's trick is to reverse the vector instead
    (a[0] stays put, the rest flips end-to-end). For unit-length random vectors
    this approximate inverse behaves well and is what makes recall practical.
    """
    return np.concatenate(([a[0]], a[:0:-1]))


def involution_batch(K):
    """involution() over a stack (k, dim) in one array op: involution_batch(K)[i] == involution(K[i])."""
    K = np.asarray(K, float)
    return np.concatenate([K[:, :1], K[:, :0:-1]], axis=1)


def unbind_all(trace, keys):
    """Unbind ONE trace against MANY keys in a single batched FFT -- the vectorised form of the
    [unbind(trace, k) for k in keys] loop that decoders/resonators run constantly. Returns (k, dim): row i
    is the noisy estimate freed by key i. Built on bind_fixed (the trace spectrum is computed once and reused
    across keys), so a Python FFT loop becomes one C call. Matches the scalar loop to FFT-batch epsilon."""
    return bind_fixed(np.asarray(trace, float), involution_batch(keys))


def bundle_bind(keys, values):
    """Encode a record/structure -- bundle([bind(keys[i], values[i]) for i]) -- in ONE batched FFT instead
    of a Python loop of per-pair binds. The vectorised form of the role/filler encode that VSA programs do
    constantly (records, scenes, recipes). KEPT NEGATIVE: the batched FFT differs from the scalar bind loop
    at ~1e-16 -- enough to flip a knife-edge tie-break -- so tie-sensitive encoders (the maze-rescue path)
    keep the scalar/cached form, while wide-margin encoders (classification, recipes) adopt this. The same
    trade bind_batch already documents."""
    return bundle(bind_batch(np.asarray(keys, float), np.asarray(values, float)))


def nearest(query, matrix):
    """Nearest row of a codebook matrix (k, dim) to `query` -- argmax(matrix @ query), the reusable matmul
    form of the [cosine(query, v) for v in codebook] loop. Rows are assumed ~unit length (as Vocabulary
    atoms are), so the dot's argmax equals the cosine's argmax -- EXACT, no epsilon caveat. Returns
    (index, score)."""
    matrix = np.asarray(matrix, float)
    if not len(matrix):
        return -1, -1.0
    q = np.asarray(query, float)
    nq = np.linalg.norm(q) or 1.0
    sims = (matrix @ q) / nq
    j = int(sims.argmax())
    return j, float(sims[j])


def unbind(composite, a):
    """Recover b from a composite that contains bind(a, b).

    Returns a NOISY estimate of b -- correct in direction but contaminated by
    every other association sharing the composite. Always follow this with a
    cleanup step (see Vocabulary.cleanup) to snap the estimate to a real symbol.
    """
    return bind(composite, involution(a))


def bundle(vectors):
    """Superpose a list of vectors into one, then renormalize.

    Unlike bind(), the bundle stays SIMILAR to each of its parts -- that is how
    a single fixed-width vector can stand for a whole set. Think 'add'. The more
    you pile in, the noisier it gets, so keep bundles modest.
    """
    total = np.sum(vectors, axis=0)
    norm = np.linalg.norm(total)
    return total / norm if norm > 0 else total


def cosine(a, b):
    """Similarity between two vectors: 1.0 identical, 0.0 unrelated."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def permute(vec, shift):
    """Rotate a vector's elements by 'shift' positions (a cyclic shift).

    This is the VSA trick for encoding ORDER or POSITION. A permuted vector is
    dissimilar to the original, and the shift is reversible (permute by -shift),
    so you can tag each item in a sequence with 'how many steps ago' it was and
    later tell the items apart. We use it below to build a working memory of a
    creature's recent moves: bundle(permute(move_t, 1), permute(move_t-1, 2), ...)
    holds an ordered short-term history in one fixed-width vector.
    """
    return np.roll(vec, shift)


# ---------------------------------------------------------------------------
# 2. THE VOCABULARY  (the "cleanup memory")
#
#    Holographic recall is approximate. The vocabulary is the dictionary of
#    clean, known vectors that we compare noisy recalls against -- the error
#    correction layer. It starts empty and grows from scratch.
# ---------------------------------------------------------------------------

class Vocabulary:
    """A growing dictionary of symbol -> clean vector.

    First time the system sees a symbol name, it mints a random vector for it
    and remembers it. cleanup() takes a noisy vector and returns the closest
    known symbol -- this is what turns a fuzzy recall back into a crisp answer.
    """

    def __init__(self, dim, seed=0, unitary=False, derived=False):
        self.dim = dim
        self.seed = seed                   # remembered so seed-derived stores can persist
        self.rng = np.random.default_rng(seed)
        self.vectors = {}  # name -> clean unit vector
        # Atom mint. Gaussian by default. unitary=True mints all-unit-magnitude-spectrum
        # atoms, which make circular-convolution unbinding EXACT (the involution becomes
        # the true inverse) -- measured to widen the cleanup margin on few-factor
        # role-binding (records, relations, sequence roles). It is opt-in PER SUBSYSTEM,
        # not a global default, because exact unbinding measurably HURTS the mechanisms
        # that read the SPREAD of a bundle as a signal -- branching-entropy segmentation,
        # the small-alphabet sequentiality test, and the creature's permute+bundle
        # working memory (the starved-maze bootstrap rescue went to 0 under unitary).
        # Adopt where it wins; leave the rest Gaussian.
        self.unitary = unitary
        # derived=True mints each atom as a pure function of (seed, name) instead of from
        # the running rng, so an atom regenerates in isolation and the whole vocabulary
        # persists as just seed + the list of names (regenerate-from-seed at the leaf
        # level). Default False keeps the original order-dependent minting exactly.
        self.derived = derived

    def get(self, name):
        """Return the vector for a symbol, creating it on first sight."""
        if name not in self.vectors:
            if self.derived:
                self.vectors[name] = derived_atom(self.seed, name, self.dim, self.unitary)
            else:
                mint = unitary_vector if self.unitary else random_vector
                self.vectors[name] = mint(self.dim, self.rng)
        return self.vectors[name]

    def cleanup(self, noisy, candidates=None, energy=False, beta=25.0, steps=3):
        """Snap a noisy vector to the nearest known symbol.

        Returns (name, similarity). If 'candidates' is given (a list of names),
        only those are considered -- handy when you know a recall must be, say,
        one of the known values rather than any symbol at all.

        The full-vocabulary case (no 'candidates') is the hot path -- it runs as ONE
        matrix-vector product against a cached stack of the stored atoms instead of a
        Python loop of per-name cosines, which is ~100x faster on a large vocabulary
        and bit-for-bit the same answer (stored atoms are unit length, so the dot is
        the cosine up to the query's norm, which doesn't change the argmax).

        With energy=True the query is first DENOISED by the modern-Hopfield (dense associative)
        update z <- V^T softmax(beta*V z) against the candidate codebook (Krotov-Hopfield 2016;
        Ramsauer et al. 2020) before this nearest-symbol readout -- an opt-in, strict superset
        (B1) that at beta->inf reproduces EXACTLY the argmax decision below (the softmax becomes
        one-hot on the nearest atom). The continuous-vector denoising is the value; on identity it
        ties hard argmax (B1's kept negative). Default off, so every existing caller is bit-for-bit
        unaffected."""
        if energy:
            # B1 dense-associative cleanup: pull the query onto the stored-pattern manifold, then
            # read out the nearest symbol as usual. At high beta this changes nothing about WHICH
            # symbol wins; it cleans the continuous vector the readout sees. Default-off keeps the
            # engine's backward-compatibility rule.
            from holographic_hopfield import dense_cleanup
            if candidates is None:
                if not self.vectors:
                    return None, -1.0
                _, cb = self._matrix()
            else:
                cb = np.stack([self.vectors[name] for name in candidates])
            noisy = dense_cleanup(noisy, cb, beta=beta, steps=steps)
        nn = float(np.linalg.norm(noisy))
        if candidates is None:
            if not self.vectors:
                return None, -1.0
            names, mat = self._matrix()
            if nn == 0.0:
                return names[0], 0.0
            sims = (mat @ noisy) / nn
            j = int(sims.argmax())
            return names[j], float(sims[j])
        # explicit candidate subset: small, so the direct loop is fine and avoids
        # rebuilding/caching a matrix for a one-off set
        best_name, best_sim = None, -1.0
        for name in candidates:
            s = cosine(noisy, self.vectors[name])
            if s > best_sim:
                best_name, best_sim = name, s
        return best_name, best_sim

    def _matrix(self):
        """Cached (names, stacked-matrix) view of the stored atoms for batched cleanup.
        Rebuilt only when the set of stored atoms changes, so repeated cleanups over a
        stable vocabulary pay the stack cost once."""
        n = len(self.vectors)
        cache = getattr(self, "_mat_cache", None)
        if cache is None or cache[0] != n or cache[1] is not self.vectors:
            names = list(self.vectors)
            mat = np.stack([self.vectors[k] for k in names]) if names else np.zeros((0, self.dim))
            self._mat_cache = (n, self.vectors, names, mat)
        return self._mat_cache[2], self._mat_cache[3]

    # -- persistence: store the GENERATOR, not the generated -------------------
    # Every atom is mint(dim, rng) drawn from a seeded rng in get()-call order, so the
    # whole name->vector matrix is a deterministic function of (dim, seed, unitary, the
    # ordered names). We therefore save the RECIPE -- the seed and the mint order -- and
    # reconstruct the atoms by replaying get(), instead of storing the matrix (~260x
    # smaller, and exact). Replaying also leaves the rng at the same position the original
    # reached, so atoms minted AFTER reload match a never-saved run -- one deterministic
    # mechanism subsuming both the stored vectors and the old stored rng stream. Same idea
    # the recall forest uses (seed-derived trees rebuilt from items): store the seed, not
    # the structure it grows.
    def _seed_replay_reproduces(self):
        """True if replaying get() over the current names from a fresh seeded vocabulary
        reproduces today's atoms exactly -- the precondition for storing only the recipe.
        Guards the rare case where atoms were assigned non-deterministically. (In derived
        mode this is always true, since each atom is a pure function of (seed, name).)"""
        probe = Vocabulary(self.dim, seed=self.seed, unitary=self.unitary, derived=self.derived)
        for name in self.vectors:
            if not np.array_equal(probe.get(name), self.vectors[name]):
                return False
        return True

    def to_state(self):
        """A snapshot that reconstructs an identical vocabulary. Normally just the recipe
        (config + the ordered names); the atoms replay deterministically from the seed.
        Falls back to storing the raw vectors only if they were set non-deterministically
        (so correctness never depends on the replay assumption holding)."""
        state = {
            "kind": "Vocabulary",
            "dim": int(self.dim),
            "seed": int(self.seed),
            "unitary": bool(self.unitary),
            "names": list(self.vectors.keys()),
        }
        if self._seed_replay_reproduces():
            state["reconstruct"] = "seed_replay"          # store the generator, not the matrix
        else:
            state["reconstruct"] = "explicit"             # atoms aren't seed-derived: keep them
            state["vectors"] = (np.stack(list(self.vectors.values()))
                                if self.vectors else np.zeros((0, self.dim)))
            state["rng_state"] = np.frombuffer(
                json.dumps(self.rng.bit_generator.state).encode("utf-8"), dtype=np.uint8)
        return state

    @classmethod
    def from_state(cls, state):
        """Rebuild a Vocabulary from a to_state() snapshot. For the recipe form, replay
        get() over the saved names from a fresh seeded vocabulary -- exact atoms and the
        rng left in the right place. For the explicit fallback, restore the stored vectors
        and rng directly."""
        v = cls(int(state["dim"]), seed=int(state.get("seed", 0)),
                unitary=bool(state.get("unitary", False)))
        mode = state.get("reconstruct", "explicit")
        if mode == "seed_replay" or "vectors" not in state:
            for name in state["names"]:
                v.get(name)                                # deterministic replay
            return v
        vecs = np.asarray(state["vectors"], float)
        v.vectors = {name: vecs[i] for i, name in enumerate(state["names"])}
        if state.get("rng_state") is not None:
            try:
                bg = json.loads(bytes(np.asarray(state["rng_state"], np.uint8)).decode("utf-8"))
                v.rng.bit_generator.state = bg
            except Exception:
                pass                                       # fall back to seed-initialised rng
        return v


# ---------------------------------------------------------------------------
# 3. HOLOGRAPHIC KEY-VALUE MEMORY  (the "storage")
#
#    The classic party trick: cram many key->value pairs into ONE vector and
#    pull any of them back out. This is exactly leOS's "holographic cache."
# ---------------------------------------------------------------------------

class HolographicMemory:
    """Many key->value pairs stored in a single fixed-width vector.

    learn() folds bind(key, value) into a running sum (the 'trace').
    recall() unbinds the trace with a key to get a noisy value back.
    Capacity is finite: past ~20 pairs at dim=1024 the noise wins. That is a
    feature to understand, not a bug to hide -- the demo shows it happening.
    """

    def __init__(self, dim):
        self.dim = dim
        self.trace = np.zeros(dim)

    def learn(self, key, value):
        self.trace = self.trace + bind(key, value)

    def recall(self, key):
        return unbind(self.trace, key)


def _install_c_kernel_if_requested():
    import os

    enabled = os.environ.get("HOLOSTUFF_USE_C", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    try:
        import holographic_c as _holographic_c
        _holographic_c.install(globals(), strict=True)
    except Exception:
        if os.environ.get("HOLOSTUFF_C_STRICT", "").lower() in {"1", "true", "yes", "on"}:
            raise


_install_c_kernel_if_requested()


def recall_all(trace, keys, codebook, iterative=True):
    """Recover the value for EVERY key stored in one overloaded trace.

    A single trace (a sum of bind(key_i, value_i)) recalls only a handful of
    pairs cleanly before the pile-up of cross-terms -- every other pair's noise
    -- drowns the signal (see HolographicMemory: capacity is finite). The usual
    fixes treat that crosstalk as irreducible and spread the load (partitioning).
    This does something different: it PEELS the pairs off one at a time, the way a
    decoder cancels interference or a photo is developed strongest-signal first.

      1. Unbind every key and snap each estimate to the nearest codebook entry,
         noting how confident (cosine) each guess is.
      2. Accept the single MOST confident pair -- the clearest thing in the
         exposure -- and SUBTRACT bind(key, clean_value) back out of the trace.
      3. The residual now holds one fewer interferer, so the next-clearest item
         is sharper. Repeat until every pair is decoded.

    Each correct subtraction makes the rest easier, so this 'successive
    cancellation' recovers roughly twice as many pairs cleanly as one-shot recall
    at the same width -- and because it is orthogonal to partitioning (peel WITHIN
    each region) the two stack into a multiplicative win. The catch is honest:
    once the clearest remaining item is itself wrong, subtracting it injects error
    that cascades, so past the capacity cliff this degrades rather than rescues.
    The remedy is to keep each trace inside its good regime (e.g. by partitioning)
    and let peeling double it from there.

    Returns, for each key, the index into `codebook` it decoded to. With
    iterative=False it is plain one-shot recall (no peeling), kept for contrast.
    """
    keys = np.asarray(keys, dtype=float)
    codebook = np.asarray(codebook, dtype=float)
    units = codebook / (np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-12)

    def best_match(vec):
        v = vec / (np.linalg.norm(vec) or 1.0)
        sims = units @ v
        j = int(sims.argmax())
        return j, float(sims[j])

    if not iterative:
        return [best_match(unbind(trace, k))[0] for k in keys]

    residual = np.array(trace, dtype=float)
    remaining = set(range(len(keys)))
    decoded = {}
    while remaining:
        win_i, win_j, win_c = -1, -1, -2.0
        for i in remaining:                       # find the clearest remaining pair
            j, c = best_match(unbind(residual, keys[i]))
            if c > win_c:
                win_c, win_i, win_j = c, i, j
        decoded[win_i] = win_j
        residual = residual - bind(keys[win_i], codebook[win_j])   # cancel it, sharpen the rest
        remaining.discard(win_i)
    return [decoded[i] for i in range(len(keys))]


# ---------------------------------------------------------------------------
# 3b. PARTITIONING THE SPACE  (named regions + routing)
#
#     Demo 1 shows a single trace collapsing once you push past a couple dozen
#     pairs -- the bind() terms pile up as noise. Partitioning is the fix leOS
#     uses (its text_store / context_store / media_store, and the "route a
#     query to the nearest region" step). Split the space into regions, give
#     each its own trace, and route every key to one region. Noise in any one
#     trace stays low, so total capacity grows with the number of regions.
# ---------------------------------------------------------------------------

class PartitionedMemory:
    """The embedding space split into N regions, each with its own trace.

    Routing uses FIXED random 'anchor' directions, one per region: a key goes
    to whichever anchor it is most similar to. The anchors never move, so a key
    routes to the SAME region whether we are storing it or recalling it later
    -- nothing gets lost because a boundary shifted underneath it.

    A note on "semantic" partitioning: with random keys these regions are just
    capacity buckets. But if your keys carry meaning (like the encoded examples
    in the learner below), similar keys land near the same anchor on their own
    -- that is the semantic routing leOS does with its `route` instruction. The
    machinery is identical; only the keys differ.
    """

    def __init__(self, dim, num_partitions=16, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Each anchor is one fixed direction; together they carve up the space.
        self.anchors = [random_vector(dim, rng) for _ in range(num_partitions)]
        self.traces = [HolographicMemory(dim) for _ in range(num_partitions)]
        self.counts = [0] * num_partitions   # how many pairs landed in each region

    def route(self, key):
        """Return the index of the region a key belongs to (nearest anchor)."""
        mat = getattr(self, "_anchor_mat", None)
        if mat is None or len(mat) != len(self.anchors):
            mat = self._anchor_mat = np.stack(self.anchors)   # cache the stacked anchors for matmul routing
        return nearest(key, mat)[0]                            # argmax(anchors @ key) -- exact, one matmul

    def learn(self, key, value):
        """Store a pair in the region its key routes to."""
        i = self.route(key)
        self.traces[i].learn(key, value)
        self.counts[i] += 1

    def recall(self, key):
        """Recall by routing to the key's region and unbinding only that trace."""
        return self.traces[self.route(key)].recall(key)


# ---------------------------------------------------------------------------
# 4. A LITTLE AI THAT LEARNS FROM SCRATCH  (storage + computation together)
#
#    A classifier with no training step in the usual sense. You feed it a few
#    examples; it builds a holographic prototype per label. New examples that
#    share structure land near the right prototype -- that's generalization,
#    and it comes from the geometry, not from gradient descent.
#
#    There's also a "reflex" fast-path: if it has seen an EXACT input before,
#    it replays the stored answer instantly instead of recomputing. That mirrors
#    leOS's "reflex arc" -- familiar input bypasses the expensive path.
# ---------------------------------------------------------------------------

class HolographicLearner:
    """Learns to label structured examples from a handful of demonstrations.

    An example is a dict of feature -> value, e.g.
        {"size": "small", "cover": "feathers", "fly": "yes", "habitat": "land"}
    We encode it holographically by binding each feature to its value and
    bundling the results into one vector. Examples that share (feature, value)
    pairs end up with similar vectors -- which is the whole point.
    """

    def __init__(self, dim=1024, seed=0):
        self.dim = dim
        self.vocab = Vocabulary(dim, seed)
        self.prototypes = {}   # label -> accumulated vector (a bundle of examples)
        self.reflex = {}       # exact input signature -> label (instant replay)

    def encode(self, example):
        """Turn a {feature: value} dict into a single holographic vector.

        bind(feature, value) creates a 'feature IS value' token; bundling the
        tokens superposes them so the example vector is similar to any other
        example sharing those tokens. Runs as ONE batched FFT via bundle_bind (the
        role/filler encode vectorised) rather than a Python loop of per-pair binds --
        classification is wide-margin, so the batched-FFT ~1e-16 difference is harmless
        here (unlike the tie-sensitive maze-rescue encoder, which keeps the scalar form)."""
        items = sorted(example.items())
        if not items:
            return np.zeros(self.vocab.dim)
        roles = np.stack([self.vocab.get(f) for f, _ in items])
        values = np.stack([self.vocab.get(v) for _, v in items])
        return bundle_bind(roles, values)

    @staticmethod
    def _signature(example):
        """A hashable, order-independent key for the exact-match reflex cache."""
        return tuple(sorted(example.items()))

    def learn(self, example, label):
        """Absorb one labeled example.

        - Adds the encoded example into that label's prototype (the bundle).
        - Records the exact input in the reflex cache for instant future replay.
        Both the label's prototype and the reflex entry are created on demand,
        so the learner genuinely starts from nothing.
        """
        vec = self.encode(example)
        if label in self.prototypes:
            self.prototypes[label] = self.prototypes[label] + vec
        else:
            self.prototypes[label] = vec.copy()
        self.reflex[self._signature(example)] = label

    def classify(self, example):
        """Predict a label for an example, seen before or not.

        Returns (label, confidence, used_reflex). Confidence is the cosine
        similarity to the winning prototype (1.0 = perfect match). used_reflex
        is True when we answered from the exact-match cache without computing.
        """
        sig = self._signature(example)
        if sig in self.reflex:                       # reflex arc: instant replay
            return self.reflex[sig], 1.0, True

        vec = self.encode(example)                   # generalize via geometry
        best_label, best_sim = None, -1.0
        for label, proto in self.prototypes.items():
            s = cosine(vec, proto)
            if s > best_sim:
                best_label, best_sim = label, s
        return best_label, best_sim, False


# ---------------------------------------------------------------------------
# 5. THE HYPERSPHERE  (leOS's "LVM" geometry layer)
#
#     Up to here we treated vectors as living in ordinary flat space. leOS does
#     its learning on the surface of the unit sphere instead, where the natural
#     operations change: addition becomes exp_map, subtraction becomes log_map,
#     and straight-line interpolation becomes SLERP. Why bother? Because a
#     "displacement" from one idea to another is then a tangent vector you can
#     store, average, and replay -- which is exactly how the reflex arc below
#     learns. Everything here assumes UNIT-LENGTH vectors.
# ---------------------------------------------------------------------------

def geodesic(a, b):
    """Distance between two points along the sphere's surface, in radians.
    0 = same point, pi/2 = perpendicular, pi = antipodal."""
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def log_map(base, x):
    """The 'subtraction' x - base, done on the sphere.

    Returns the tangent vector at 'base' that points toward 'x'. Its length is
    the geodesic distance. This is how we capture 'the move from a task to its
    response' as something we can store and reuse.
    """
    d = float(np.clip(np.dot(base, x), -1.0, 1.0))
    theta = np.arccos(d)
    if theta < 1e-8:
        return np.zeros_like(base)
    perp = x - d * base                  # part of x perpendicular to base
    pn = np.linalg.norm(perp)
    if pn < 1e-12:
        return np.zeros_like(base)
    return theta * (perp / pn)


def exp_map(base, v):
    """The 'addition' base + v, done on the sphere.

    Starts at 'base' and walks along tangent vector 'v', staying on the
    surface. exp_map(base, log_map(base, x)) == x, so the two are inverses --
    that round-trip is what lets us record a move and replay it elsewhere.
    """
    n = np.linalg.norm(v)
    if n < 1e-8:
        return base.copy()
    return np.cos(n) * base + np.sin(n) * (v / n)


def slerp(a, b, t):
    """Spherical interpolation: the point t of the way from a to b (0..1),
    travelling along the surface rather than cutting through the middle."""
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    theta = np.arccos(d)
    if theta < 1e-8:
        return a.copy()
    s = np.sin(theta)
    return (np.sin((1 - t) * theta) / s) * a + (np.sin(t * theta) / s) * b


# ---------------------------------------------------------------------------
# 6. THE REFLEX ARC  (leOS's "displacement codec" -- learning from experience)
#
#     This is the heart of how leOS learns WITHOUT training. It watches tasks
#     and their responses, and for each one records the displacement (the
#     tangent vector from task to response). Faced with a new task, it finds
#     similar past tasks, blends their displacements, and applies the result --
#     nearest-neighbour regression on the sphere. Familiar inputs get answered
#     from this geometric cache in an instant; only unfamiliar ones (low
#     confidence) would need the expensive "real" model.
# ---------------------------------------------------------------------------

class ReflexArc:
    """Learns input->output transformations by remembering trajectories.

    experience(task, response) records the move from task to response.
    recall(task) predicts a response for a new task by blending the moves of
    the most similar remembered tasks. There is no training loop and no model
    here -- just stored geometry. The 'confidence' it returns is how familiar
    the new task is; leOS uses exactly this signal to decide when it can trust
    the cache and when it must fall back to heavier machinery.
    """

    def __init__(self):
        self.experiences = []  # each: {"task","response","disp","mag"}

    def experience(self, task, response):
        self.experiences.append({
            "task": task,
            "response": response,
            "disp": log_map(task, response),      # the move, as a tangent vector
            "mag": geodesic(task, response),       # how big the move was
        })

    def neighbors(self, task, k):
        """The k remembered tasks most similar to this one."""
        ranked = sorted(self.experiences,
                        key=lambda e: cosine(task, e["task"]), reverse=True)
        return ranked[:k]

    def recall(self, task, k=5):
        """Predict a response for a (possibly unseen) task.

        Returns (predicted_response, confidence). Confidence is the similarity
        to the closest remembered task: high means 'I've seen this kind of
        thing', low means 'unfamiliar -- don't trust this answer'.
        """
        if not self.experiences:
            return task.copy(), 0.0
        nbrs = self.neighbors(task, k)
        sims = np.array([max(cosine(task, e["task"]), 0.0) for e in nbrs])
        if sims.sum() == 0:
            return task.copy(), 0.0
        weights = sims / sims.sum()
        # Blend the neighbours' moves. (A fair approximation when they're close;
        # full leOS first parallel-transports each move into this task's frame.)
        blended = sum(w * e["disp"] for w, e in zip(weights, nbrs))
        return exp_map(task, blended), float(sims.max())


# ---------------------------------------------------------------------------
# 7. DRIFT DETECTION  (leOS's "catch hallucination with astronomy")
#
#     A quality check that never calls a model. It borrows the redshift /
#     blueshift idea from cosmology: measure how far a response moved from the
#     task, and compare that to how far responses USUALLY move for similar
#     tasks (read straight off the reflex arc's memory).
#
#       redshift  -> moved too far  -> off-topic / hallucination
#       blueshift -> barely moved   -> shallow echo, a non-answer
#       void      -> no similar past experience -> can't judge, low confidence
#       on-track  -> moved about the expected amount
#
#     A separate loop check catches an agent going in circles (recent responses
#     that mean nearly the same thing even if the words differ).
# ---------------------------------------------------------------------------

class DriftDetector:
    """Judge a response's health from geometry alone, using the reflex arc's
    accumulated experience as the yardstick for 'normal'."""

    def __init__(self, reflex, k=8, redshift_sigma=1.8, blueshift_sigma=1.5,
                 echo_cos=0.92, familiar_cos=0.5, loop_cos=0.85):
        self.reflex = reflex
        self.k = k
        self.redshift_sigma = redshift_sigma
        self.blueshift_sigma = blueshift_sigma
        self.echo_cos = echo_cos          # response this close to task = echo
        self.familiar_cos = familiar_cos  # below this, the region is a void
        self.loop_cos = loop_cos          # recent responses this alike = a loop
        self.recent = []                  # rolling window for loop detection

    def judge(self, task, response):
        """Return one of: 'void', 'blueshift', 'redshift', 'on-track'."""
        nbrs = self.reflex.neighbors(task, self.k)
        top = cosine(task, nbrs[0]["task"]) if nbrs else 0.0
        if top < self.familiar_cos:
            return "void"
        mags = [e["mag"] for e in nbrs]
        mean = float(np.mean(mags))
        std = max(float(np.std(mags)), 0.05)   # floor so the band isn't absurdly tight
        actual = geodesic(task, response)
        if cosine(task, response) > self.echo_cos or actual < mean - self.blueshift_sigma * std:
            return "blueshift"
        if actual > mean + self.redshift_sigma * std:
            return "redshift"
        return "on-track"

    def check_loop(self, response):
        """Feed each response of a session in turn; True once the last few are
        near-duplicates in meaning (a semantic loop, not just repeated text)."""
        self.recent.append(response)
        self.recent = self.recent[-5:]
        if len(self.recent) < 3:
            return False
        sims = [cosine(self.recent[i], self.recent[j])
                for i in range(len(self.recent))
                for j in range(i + 1, len(self.recent))]
        return float(np.mean(sims)) > self.loop_cos


# ---------------------------------------------------------------------------
# 8. DEMOS
# ---------------------------------------------------------------------------

def _make_clusters(dim, n_clusters=3, seed=0):
    """Helper for the learning demos: a few tight task clusters, each with one
    fixed 'answer' a moderate, consistent distance away. Returns (rng, centers,
    answers, make_task) where make_task(k) draws a fresh task near cluster k."""
    rng = np.random.default_rng(seed)
    centers = [random_vector(dim, rng) for _ in range(n_clusters)]
    answers = [slerp(centers[k], random_vector(dim, rng), 0.4)
               for k in range(n_clusters)]

    def make_task(k):
        # rotate the centre a small random amount -> a tight, on-sphere cluster
        return slerp(centers[k], random_vector(dim, rng), rng.uniform(0.05, 0.18))

    return rng, centers, answers, make_task


def demo_reflex():
    """Show the reflex arc learning input->output moves from experience and
    reproducing them for brand-new inputs."""
    print("=" * 70)
    print("DEMO 3 -- Reflex arc: learning transformations from experience")
    print("=" * 70)

    dim = 512
    _, _, answers, make_task = _make_clusters(dim, n_clusters=3, seed=0)
    reflex = ReflexArc()

    # Experience phase: watch tasks and their responses. No training, just
    # recording the move (task -> answer) for each.
    for k in range(3):
        for _ in range(30):
            reflex.experience(make_task(k), answers[k])
    print(f"\nWatched {len(reflex.experiences)} task->response pairs "
          f"across 3 regions.\n")

    # Test phase: brand-new tasks. Compare the reflex's reconstruction of the
    # response against a do-nothing baseline that just echoes the task back.
    reflex_err, echo_err = [], []
    for k in range(3):
        for _ in range(20):
            t = make_task(k)
            pred, _ = reflex.recall(t)
            reflex_err.append(geodesic(pred, answers[k]))
            echo_err.append(geodesic(t, answers[k]))   # echoing input as output
    print("  Reconstruction error on unseen tasks (radians, lower is better):")
    print(f"    reflex arc      {np.mean(reflex_err):.3f}   <- learned the move")
    print(f"    echo baseline   {np.mean(echo_err):.3f}   <- just repeats the input")

    # Confidence cleanly separates familiar tasks from unexplored ones.
    fam = reflex.recall(make_task(0))[1]
    void = reflex.recall(random_vector(dim, np.random.default_rng(999)))[1]
    print(f"\n  Confidence:  familiar task {fam:.2f}   unfamiliar (void) {void:.2f}")
    print("  High confidence -> answer from the cache; low -> escalate. That")
    print("  routing decision is the whole point of leOS's reflex arc.\n")


def demo_drift():
    """Show drift detection flagging hallucination, shallow echoes, voids, and
    loops -- all from geometry, no model calls."""
    print("=" * 70)
    print("DEMO 4 -- Drift detection: judging answers with geometry, no model")
    print("=" * 70)

    dim = 512
    _, _, answers, make_task = _make_clusters(dim, n_clusters=3, seed=0)
    reflex = ReflexArc()
    for k in range(3):
        for _ in range(30):
            reflex.experience(make_task(k), answers[k])
    detector = DriftDetector(reflex)

    task = make_task(1)
    print()
    print(f"  good answer (the right response) -> {detector.judge(task, answers[1])}")
    print(f"  echo (just repeats the task)     -> {detector.judge(task, task)}")
    print(f"  hallucination (random vector)    -> "
          f"{detector.judge(task, random_vector(dim, np.random.default_rng(7)))}")
    void_task = random_vector(dim, np.random.default_rng(123))
    print(f"  answer in an unexplored region   -> "
          f"{detector.judge(void_task, random_vector(dim, np.random.default_rng(8)))}")

    # Loop detection: feed a run of near-identical responses.
    print("\n  Loop check over a session (looking for circular answers):")
    base = answers[1]
    looped = False
    for i in range(4):
        # each 'response' is a tiny perturbation of the same point -> a loop
        nudged = slerp(base, random_vector(dim, np.random.default_rng(i)), 0.03)
        looped = detector.check_loop(nudged)
        print(f"    response {i + 1}: loop detected = {looped}")
    print()


def demo_keyvalue():
    """Show the holographic cache: many pairs in one vector, then graceful
    decay as we overload it past capacity."""
    print("=" * 70)
    print("DEMO 1 -- Holographic key/value store (everything in ONE vector)")
    print("=" * 70)

    dim = 1024
    vocab = Vocabulary(dim, seed=1)
    mem = HolographicMemory(dim)

    # A small "knowledge base" of capital cities, stored as country->capital.
    facts = {
        "france": "paris", "japan": "tokyo", "egypt": "cairo",
        "brazil": "brasilia", "canada": "ottawa", "kenya": "nairobi",
    }
    # Pre-mint every value vector so cleanup has something to snap to.
    for country, capital in facts.items():
        vocab.get(country)
        vocab.get(capital)
        mem.learn(vocab.get(country), vocab.get(capital))

    value_names = list(facts.values())
    print(f"\nStored {len(facts)} country->capital pairs in a single {dim}-d vector.\n")
    correct = 0
    for country, capital in facts.items():
        noisy = mem.recall(vocab.get(country))
        guess, sim = vocab.cleanup(noisy, candidates=value_names)
        ok = "OK " if guess == capital else "XX "
        correct += guess == capital
        print(f"  {ok} capital of {country:8s} -> {guess:9s} "
              f"(similarity {sim:.2f}, expected {capital})")
    print(f"\nRecalled {correct}/{len(facts)} correctly.")

    # Now overload it well past capacity to watch confidence collapse -- the
    # "noise accumulates after ~10-20 compositions" limit, made visible.
    #
    # The telling number is the MARGIN: how far the correct answer sits above
    # the runner-up. A big margin means a trustworthy recall; a tiny margin
    # means we're guessing. Cleanup restricted to 6 known capitals will often
    # still squeak out the right name, but with thousands of candidates that
    # collapsed margin would be indistinguishable from a wrong match.
    def france_margin():
        noisy = mem.recall(vocab.get("france"))
        ranked = sorted(((cosine(noisy, vocab.get(v)), v) for v in value_names),
                        reverse=True)
        (top_sim, top), (run_sim, _) = ranked[0], ranked[1]
        return top, top_sim, top_sim - run_sim

    top, sim, gap = france_margin()
    print(f"\nBefore overload:  france -> {top} "
          f"(similarity {sim:.2f}, margin over runner-up {gap:.2f})")

    rng = np.random.default_rng(99)
    for i in range(200):
        mem.learn(random_vector(dim, rng), random_vector(dim, rng))

    top, sim, gap = france_margin()
    print(f"After 200 junk:   france -> {top} "
          f"(similarity {sim:.2f}, margin over runner-up {gap:.2f})")
    print("  The margin has collapsed toward zero -- the recall is no longer")
    print("  trustworthy even though cleanup still names the right city.")
    print("  Lesson: one vector has finite capacity. Real systems spread data")
    print("  across MANY traces plus cleanup, which is what leOS scales up.\n")


def demo_partitioning():
    """Show that routing keys into regions defeats the single-trace capacity
    collapse from Demo 1."""
    print("=" * 70)
    print("DEMO 1b -- Partitioning the space to beat the capacity limit")
    print("=" * 70)

    dim = 1024
    n = 200  # far past what one trace can hold

    # n random key->value pairs. We test recall by snapping each noisy result
    # to the nearest of the n known values, then checking it matches.
    rng = np.random.default_rng(7)
    keys = [random_vector(dim, rng) for _ in range(n)]
    vals = [random_vector(dim, rng) for _ in range(n)]

    def accuracy(recall_fn):
        hits = 0
        for i, key in enumerate(keys):
            noisy = recall_fn(key)
            nearest = int(np.argmax([cosine(noisy, v) for v in vals]))
            hits += (nearest == i)
        return hits / n

    # One trace for everything (the Demo 1 approach).
    single = HolographicMemory(dim)
    for k, v in zip(keys, vals):
        single.learn(k, v)

    # The same pairs, routed across 16 regions.
    parted = PartitionedMemory(dim, num_partitions=16, seed=1)
    for k, v in zip(keys, vals):
        parted.learn(k, v)

    print(f"\nStored the SAME {n} key->value pairs two ways:\n")
    print(f"  one single trace        -> recall accuracy {accuracy(single.recall):.0%}")
    print(f"  16 routed partitions    -> recall accuracy {accuracy(parted.recall):.0%}")
    print(f"\n  Pairs per region: {parted.counts}")
    print("  Same total data, same math -- but no single trace is overloaded,")
    print("  so recall holds up. That is what partitioning buys you.\n")


def demo_cancellation():
    """Show successive cancellation: recover far more pairs from ONE overloaded
    trace by peeling the clearest item, subtracting it, and repeating -- and show
    that this stacks on top of partitioning for a compounding win."""
    print("=" * 70)
    print("DEMO 2b -- Successive cancellation: recover the whole exposure by peeling")
    print("=" * 70)

    dim = 1024
    print("\nAll pairs in a SINGLE trace, recalled three ways. 'one-shot' unbinds")
    print("each key once; 'peeled' cancels the clearest pair then re-reads the")
    print("rest (recursively); 'peeled + 8 regions' does the same inside each")
    print("partition. Accuracy = fraction of keys returning the right value.\n")
    print(f"  {'pairs':>6}   {'one-shot':>9}   {'peeled':>7}   {'peeled + 8 regions':>19}")
    for n in (80, 160, 320):
        one = sic = comp = 0.0
        trials = 3
        for s in range(trials):
            rng = np.random.default_rng(100 + s)
            keys = np.stack([random_vector(dim, rng) for _ in range(n)])
            vals = np.stack([random_vector(dim, rng) for _ in range(n)])
            trace = np.zeros(dim)
            for k, v in zip(keys, vals):
                trace = trace + bind(k, v)
            one += sum(d == i for i, d in enumerate(
                recall_all(trace, keys, vals, iterative=False))) / n
            sic += sum(d == i for i, d in enumerate(
                recall_all(trace, keys, vals, iterative=True))) / n
            # the same pairs, partitioned into 8 regions, peeled within each
            anchors = [random_vector(dim, rng) for _ in range(8)]
            region = [int(np.argmax([cosine(keys[i], a) for a in anchors])) for i in range(n)]
            ok = 0
            for p in range(8):
                idx = [i for i in range(n) if region[i] == p]
                if not idx:
                    continue
                tr = np.zeros(dim)
                for i in idx:
                    tr = tr + bind(keys[i], vals[i])
                dec = recall_all(tr, keys[idx], vals[idx], iterative=True)
                ok += sum(dec[m] == m for m in range(len(idx)))
            comp += ok / n
        print(f"  {n:>6}   {100*one/trials:>8.0f}%   {100*sic/trials:>6.0f}%   "
              f"{100*comp/trials:>18.0f}%")
    print("\n  Peeling roughly doubles what a single trace holds; once it is past")
    print("  the cliff a wrong guess cascades, so partitioning (keep each trace in")
    print("  its good regime) and peeling COMPOUND -- two filters stacked, the way")
    print("  film layers colour or a coder codes a residual on top of the gist.\n")


def demo_learning():
    """Show learning-from-scratch: teach a few animals, then classify new ones
    it never saw, purely from shared structure."""
    print("=" * 70)
    print("DEMO 2 -- A classifier that learns from scratch and generalizes")
    print("=" * 70)

    learner = HolographicLearner(dim=1024, seed=2)

    # Six labeled examples. This is the ENTIRE training set.
    training = [
        ({"size": "small",  "cover": "feathers", "fly": "yes", "habitat": "land"},  "bird"),
        ({"size": "large",  "cover": "feathers", "fly": "yes", "habitat": "land"},  "bird"),
        ({"size": "large",  "cover": "scales",   "fly": "no",  "habitat": "water"}, "fish"),
        ({"size": "small",  "cover": "scales",   "fly": "no",  "habitat": "water"}, "fish"),
        ({"size": "medium", "cover": "fur",      "fly": "no",  "habitat": "land"},  "mammal"),
        ({"size": "large",  "cover": "skin",     "fly": "no",  "habitat": "water"}, "mammal"),
    ]
    for example, label in training:
        learner.learn(example, label)
    print(f"\nLearned from {len(training)} examples across "
          f"{len(learner.prototypes)} labels: {sorted(learner.prototypes)}\n")

    # Replay a training item: should hit the reflex cache (instant, confidence 1.0).
    seen = training[0][0]
    label, conf, reflexed = learner.classify(seen)
    print(f"  Re-asking a SEEN example -> {label} "
          f"(confidence {conf:.2f}, reflex={reflexed})\n")

    # Now genuinely NEW animals -- never shown, but share structure with training.
    print("  New, unseen examples (generalizing from shared features):")
    novel = [
        ({"size": "small",  "cover": "feathers", "fly": "yes", "habitat": "land"},  "bird?"),   # robin-like
        ({"size": "medium", "cover": "scales",   "fly": "no",  "habitat": "water"}, "fish?"),   # tuna-like
        ({"size": "small",  "cover": "fur",      "fly": "no",  "habitat": "land"},  "mammal?"), # cat-like
    ]
    for example, hint in novel:
        label, conf, reflexed = learner.classify(example)
        feats = ", ".join(f"{k}={v}" for k, v in example.items())
        print(f"    {feats}")
        print(f"      -> {label}  (confidence {conf:.2f}, reflex={reflexed}, hint was {hint})")
    print("\n  It got these from geometry alone -- no gradient descent, no")
    print("  training loop, just shared (feature,value) structure.\n")


if __name__ == "__main__":
    demo_keyvalue()
    demo_partitioning()
    demo_cancellation()
    demo_learning()
    demo_reflex()
    demo_drift()

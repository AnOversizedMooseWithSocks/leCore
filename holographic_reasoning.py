"""
holographic_reasoning.py
========================

Three more ideas lifted from leOS and boiled down to pure vector math (numpy
only, no models, no Ollama). Each extends the simple engine in a different
direction we kept bumping into:

  * ResonatorNetwork  -- take apart a vector that is several atoms bound
                         together. This is the principled cleanup for deep
                         (multi-factor) bindings, and it unlocks structured
                         representations: store a fact as subject(x)relation(x)object
                         and read all three back.
  * SemanticCompass   -- a learned DIRECTION toward good outcomes. Remembers
                         where successes and failures happened and, for any
                         point, returns "which way is better." The
                         direction-of-improvement signal, from geometry alone.
  * ConformalPredictor-- calibrated error bars / confidence. Turns a raw score
                         into "I'm 90% sure" with a distribution-free guarantee,
                         so a fast cached answer knows when to defer.
  * EpistemicMap      -- two models that learned differently disagree where the
                         knowledge is contested (leOS's "purple channel"), and
                         that disagreement, paired with void/density detection,
                         maps the frontier of what the system reliably knows.

Needs: numpy, and holographic_ai.py beside it.
"""

import numpy as np
from holographic_ai import random_vector, cosine, bind, unbind, bundle, Vocabulary


# ---------------------------------------------------------------------------
# 1. RESONATOR NETWORK  (take a bound product back apart)
#
#    bind() is reversible for two factors: unbind(bind(a, b), b) ~ a. But once
#    you bind three or more atoms together, a single unbind leaves too much
#    noise to read off a clean answer. The resonator recovers all the factors
#    at once WITHOUT knowing any of them -- only the codebooks of possible atoms.
#    The trick: hold each factor as a SUPERPOSITION of its entire codebook, then
#    repeatedly clean each estimate against the others. Like coupled tuned
#    circuits, the estimates ring until only the one consistent combination
#    survives. (Frady, Kent, Olshausen & Sommer, 2020.)
# ---------------------------------------------------------------------------

class ResonatorNetwork:
    """Factor a composite vector (several atoms bound together) into its parts.

    codebooks is a list of (n_f x D) arrays -- one codebook per factor slot,
    each row a unit atom vector. factor() returns the chosen index in each
    codebook. Capacity is finite: more factors or bigger codebooks need more
    dimensions, and past capacity recovery degrades gracefully.
    """

    def __init__(self, codebooks):
        self.codebooks = [np.asarray(cb) for cb in codebooks]

    def factor(self, composite, iters=100, init=None, rng=None):
        # Start every factor as the superposition of its whole codebook -- the
        # estimate explores all possibilities at once instead of guessing. With
        # init='random' it starts from random unit estimates instead, which lets
        # several restarts escape a bad fixed point (see multi-object factoring).
        if init == "random":
            r = rng or np.random.default_rng()
            est = []
            for cb in self.codebooks:
                v = r.standard_normal(cb.shape[1]); est.append(v / np.linalg.norm(v))
        else:
            est = [bundle(list(cb)) for cb in self.codebooks]
        F = len(self.codebooks)
        for _ in range(iters):
            for f in range(F):                       # update in place (Gauss-Seidel)
                probe = composite
                for g in range(F):                   # unbind by the OTHER estimates
                    if g != f:
                        probe = unbind(probe, est[g])
                sims = self.codebooks[f] @ probe     # similarity to each atom
                cleaned = self.codebooks[f].T @ sims  # project back onto the codebook
                n = np.linalg.norm(cleaned)
                est[f] = cleaned / n if n > 0 else cleaned
        return [int(np.argmax(self.codebooks[f] @ est[f])) for f in range(F)]


# ---------------------------------------------------------------------------
# 2. SEMANTIC COMPASS  (a learned direction toward success)
#
#    Every value-based learner we built answers "how good is this?" The compass
#    answers the harder question "which way is better?" It remembers where
#    things went well and badly, and for any point returns the local direction
#    pointing away from nearby failures and toward nearby successes. That is a
#    gradient read straight out of memory -- the direction-of-improvement signal
#    that pure instance memory lacks, with no training loop.
# ---------------------------------------------------------------------------

class SemanticCompass:
    """Records labelled outcomes in vector space and reports the local
    'toward success' direction."""

    def __init__(self):
        self.successes = []
        self.failures = []

    def record(self, vec, success):
        (self.successes if success else self.failures).append(np.asarray(vec))

    def _local_centroid(self, pool, query, k):
        if not pool:
            return None
        nearest = sorted(pool, key=lambda v: cosine(query, v), reverse=True)[:k]
        m = np.mean(nearest, axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    def direction(self, query, k=8):
        """Unit vector pointing from the local failure region toward the local
        success region. Zero vector if there's nothing nearby to learn from."""
        s = self._local_centroid(self.successes, query, k)
        f = self._local_centroid(self.failures, query, k)
        if s is None and f is None:
            return np.zeros_like(query)
        if f is None:
            return s
        if s is None:
            return -f
        d = s - f
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    def steer(self, query, step=0.3, k=8):
        """Nudge a point along the compass and renormalize -- a cheap way to
        bias a candidate toward what has worked nearby."""
        moved = query + step * self.direction(query, k)
        n = np.linalg.norm(moved)
        return moved / n if n > 0 else moved


# ---------------------------------------------------------------------------
# 3. CONFORMAL PREDICTOR  (calibrated confidence, distribution-free)
#
#    Every fast/cheap answer we make needs a trustworthy "how sure am I?" so it
#    knows when to defer to something slower. Conformal prediction gives exactly
#    that with a guarantee and no assumptions about the error distribution: feed
#    it the errors a predictor made on held-out data, and it returns the error
#    bar that will contain the truth at least (1 - alpha) of the time on NEW
#    data. leOS uses this to gate its reflex arc.
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """Split-conformal error bars for any numeric predictor."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha          # 0.1 -> aim for 90% coverage
        self.q = None               # calibrated half-width

    def calibrate(self, residuals):
        """residuals = |prediction - truth| on a calibration set the predictor
        did NOT learn from. Picks the conformal quantile."""
        residuals = np.abs(np.asarray(residuals, dtype=float))
        n = len(residuals)
        # the (n+1)(1-alpha) order statistic gives the finite-sample guarantee
        rank = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q = float(np.sort(residuals)[min(rank - 1, n - 1)])
        return self.q

    def interval(self, prediction):
        """Return (low, high) around a prediction with the calibrated coverage."""
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        return prediction - self.q, prediction + self.q


# ---------------------------------------------------------------------------
# 4. EPISTEMIC MAP  (disagreement + void: the frontier of what we know)
#
#    leOS's "purple channel" says: when two models embed the same thing and
#    disagree, the disagreement is itself information -- a signal that lives in
#    neither model alone. That only works for models that learned DIFFERENTLY;
#    two random models disagree about everything and it means nothing. So here
#    the two models share one set of random atoms (the same substrate) but are
#    trained on different data -- now their learned vectors are comparable, and
#    where they differ is a genuine boundary in the data.
#
#    The deeper point, and where void detection comes in: disagreement and void
#    are TWO DIFFERENT kinds of "I don't know," and each is blind where the
#    other sees.
#       * VOID      = little or no data here. Two models that are both ignorant
#                     can AGREE by accident (they fall back to the same shared
#                     atom), so disagreement alone reports false confidence --
#                     only density catches it.
#       * BOUNDARY  = plenty of data, but the models conflict (e.g. a word used
#                     in two senses). It looks well-covered, so density alone
#                     reports false confidence -- only disagreement catches it.
#    Used together they cover each other's blind spots and mark exactly where
#    the system's knowledge runs out or splits.
#
#    This is also the missing piece for "two models that update each other":
#    disagreement is the signal worth training on (active learning -- spend
#    effort where the models conflict), AND a collapse alarm (if disagreement
#    falls to zero everywhere, the two have merged into one and the loop has
#    stopped learning; healthy systems keep a live disagreement frontier).
# ---------------------------------------------------------------------------

def vector_disagreement(vec_a, vec_b):
    """How much two models disagree about one item: 0 = identical, ~1 = unrelated.
    Only meaningful when the two models share a basis (same atoms) but learned
    from different data."""
    return 1.0 - cosine(vec_a, vec_b)


class EpistemicMap:
    """Classify how well-known an item is by combining two uncertainties.

    classify() takes how much data each model has on the item (density_a,
    density_b) and how much the models disagree, and returns one of:
        'confident' -- enough data, models agree
        'boundary'  -- enough data, models conflict (the contested/purple signal)
        'void'      -- too little data for either model to be trusted
    """

    def __init__(self, density_threshold=2, disagree_threshold=0.15):
        self.density_threshold = density_threshold
        self.disagree_threshold = disagree_threshold

    def classify(self, density_a, density_b, disagreement):
        if min(density_a, density_b) < self.density_threshold:
            return "void"                       # density catches mutual ignorance
        if disagreement > self.disagree_threshold:
            return "boundary"                   # disagreement catches contested data
        return "confident"


# ---------------------------------------------------------------------------
# 5. DEMOS
# ---------------------------------------------------------------------------

def demo_resonator():
    print("=" * 70)
    print("DEMO X -- Resonator: reading a bound-together fact back apart")
    print("=" * 70)
    dim = 2048
    vocab = Vocabulary(dim, seed=1)
    subjects = ["alice", "bob", "carol", "dave"]
    relations = ["likes", "knows", "avoids", "trusts"]
    objects = ["coffee", "jazz", "rain", "python"]

    def codebook(words):
        return np.array([vocab.get(w) for w in words])

    resonator = ResonatorNetwork([codebook(subjects), codebook(relations),
                                  codebook(objects)])

    # Encode a fact as one vector: subject (x) relation (x) object.
    rng = np.random.default_rng(0)
    print("\nEncoding facts as subject(x)relation(x)object in ONE vector, then")
    print("recovering all three parts knowing only the vocabularies:\n")
    correct = 0
    for _ in range(6):
        s, r, o = (int(rng.integers(4)) for _ in range(3))
        fact = bind(bind(vocab.get(subjects[s]), vocab.get(relations[r])),
                    vocab.get(objects[o]))
        si, ri, oi = resonator.factor(fact)
        got = (si == s and ri == r and oi == o)
        correct += got
        mark = "OK " if got else "XX "
        print(f"  {mark}{subjects[s]} {relations[r]} {objects[o]:7s} -> recovered "
              f"{subjects[si]} {relations[ri]} {objects[oi]}")
    print(f"\nRecovered {correct}/6 facts exactly. A single unbind can't do this --")
    print("with three factors bound together, each part is buried under the")
    print("others until the resonator settles them out together.\n")


def demo_compass():
    print("=" * 70)
    print("DEMO Y -- Semantic compass: a learned direction toward success")
    print("=" * 70)
    dim = 1024
    rng = np.random.default_rng(2)
    good_region = random_vector(dim, rng)     # where things tend to work
    bad_region = random_vector(dim, rng)      # where they tend to fail

    compass = SemanticCompass()
    for _ in range(50):
        compass.record(bundle([good_region, 0.4 * random_vector(dim, rng)]), True)
        compass.record(bundle([bad_region, 0.4 * random_vector(dim, rng)]), False)

    query = bundle([good_region, bad_region])  # a point sitting between the two
    d = compass.direction(query)
    print("\nFrom a point between the success and failure regions, the compass")
    print("points toward success and away from failure:")
    print(f"  similarity of direction to success region: {cosine(d, good_region):+.2f}")
    print(f"  similarity of direction to failure region: {cosine(d, bad_region):+.2f}")
    before = cosine(query, good_region)
    after = cosine(compass.steer(query, step=0.4), good_region)
    print("\nSteering the point along the compass moves it toward success:")
    print(f"  similarity to success region: {before:+.2f} -> {after:+.2f}\n")


def demo_conformal():
    print("=" * 70)
    print("DEMO Z -- Conformal: calibrated error bars on a holographic readout")
    print("=" * 70)
    from holographic_encoders import ScalarEncoder

    dim = 1024
    enc = ScalarEncoder(dim, lo=0, hi=100, seed=3)
    rng = np.random.default_rng(4)

    # A noisy numeric readout: encode a value, corrupt the vector, decode it.
    def noisy_readout(value):
        v = enc.encode(value) + 0.4 * random_vector(dim, rng)
        return enc.decode(v)

    # Calibration split: measure the readout's errors on known values.
    cal_values = rng.uniform(0, 100, 300)
    cal_resid = [noisy_readout(v) - v for v in cal_values]
    conf = ConformalPredictor(alpha=0.1)       # want 90% coverage
    half = conf.calibrate(cal_resid)

    # Test split: check the guarantee holds on fresh values.
    test_values = rng.uniform(0, 100, 300)
    covered = 0
    for v in test_values:
        pred = noisy_readout(v)
        lo, hi = conf.interval(pred)
        covered += (lo <= v <= hi)
    print(f"\nCalibrated a +/-{half:.1f} error bar for 90% target coverage.")
    print(f"Empirical coverage on fresh values: {covered / len(test_values) * 100:.0f}%")
    print("\nSo every decoded number can carry an honest interval: 'about 47,")
    print(f"and I'm 90% sure it's within +/-{half:.1f}' -- no assumption about the")
    print("noise, just a guarantee from the calibration data.\n")


def demo_epistemic():
    print("=" * 70)
    print("DEMO W -- Epistemic map: disagreement + void mark the frontier")
    print("=" * 70)
    from holographic_encoders import TextEncoder

    dim = 1024
    # Two models, ONE shared set of atoms, trained on different data. 'cat' gets
    # the same contexts in both (a control); 'bank' is used in a water sense for
    # model A and a money sense for model B (the same word, two meanings).
    shared_cat = ["the cat sat on the mat", "the cat slept all day",
                  "i saw the cat outside"]
    corpus_a = shared_cat + ["fish swim near the bank", "the river flows past the bank",
                             "water covers the muddy bank", "the boat reached the bank"]
    corpus_b = shared_cat + ["i went to the bank today", "the bank gave a loan",
                             "deposit money at the bank", "the bank holds my account"]

    atoms = Vocabulary(dim, seed=0)             # the shared substrate
    model_a = TextEncoder(dim, window=2); model_a.index = atoms
    model_b = TextEncoder(dim, window=2); model_b.index = atoms
    for _ in range(6):
        for s in corpus_a:
            model_a.learn(s.split())
        for s in corpus_b:
            model_b.learn(s.split())

    def density(corpus, w):
        return sum(s.split().count(w) for s in corpus)

    emap = EpistemicMap(density_threshold=2, disagree_threshold=0.15)
    print("\n  word        data A/B   disagreement   verdict")
    for w in ["cat", "bank", "river", "loan", "xylophone"]:
        da, db = density(corpus_a, w), density(corpus_b, w)
        dis = vector_disagreement(model_a.wordvec(w), model_b.wordvec(w))
        print(f"  {w:10s}   {da}/{db}        {dis:.2f}          {emap.classify(da, db, dis)}")

    print("\n  'bank' -- what each model learned it means (the disagreement IS the")
    print("  discovery of two senses, present in neither model on its own):")
    print(f"    model A near: {[x for x, _ in model_a.nearest('bank', 4)]}")
    print(f"    model B near: {[x for x, _ in model_b.nearest('bank', 4)]}")
    print("\n  The two blind spots each signal covers for the other:")
    print("   - 'xylophone' is unseen by both, so they AGREE (disagreement 0.00) --")
    print("     a false confidence that only the data/void check catches.")
    print("   - 'bank' has plenty of data, so it looks well-covered -- only the")
    print("     disagreement reveals it's a contested boundary (two meanings).\n")


if __name__ == "__main__":
    demo_resonator()
    demo_compass()
    demo_conformal()
    demo_epistemic()

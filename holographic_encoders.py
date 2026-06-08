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
from holographic_ai import (random_vector, cosine, bind, unbind, bundle,
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

    def __init__(self, dim, lo=0.0, hi=1.0, seed=0):
        self.dim = dim
        self.lo, self.hi = lo, hi
        self.scale = 1.0 / (hi - lo) if hi > lo else 1.0
        rng = np.random.default_rng(seed)
        # Random phases, made conjugate-symmetric so the inverse FFT is real.
        phases = rng.uniform(-np.pi, np.pi, dim)
        phases[0] = 0.0
        for k in range(1, dim // 2 + 1):
            phases[dim - k] = -phases[k]
        if dim % 2 == 0:
            phases[dim // 2] = 0.0
        self.phases = phases

    def encode(self, x):
        # Rotating the fixed phases by x is "raising the base vector to power x".
        spectrum = np.exp(1j * self.scale * x * self.phases)
        v = np.real(np.fft.ifft(spectrum))
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def decode(self, vec, steps=200):
        """Read a vector back to a number: the grid value whose encoding is
        most similar. Robust to noise, which is what makes it useful for
        recovering a number after it's been bundled with other things."""
        grid = np.linspace(self.lo, self.hi, steps)
        sims = [cosine(vec, self.encode(g)) for g in grid]
        return float(grid[int(np.argmax(sims))])


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
        tokens = [bind(self.roles.get(field), self._filler(kind, value))
                  for field, (kind, value) in sorted(record.items())]
        return bundle(tokens)

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


if __name__ == "__main__":
    demo_scalar()
    demo_text()
    demo_record()

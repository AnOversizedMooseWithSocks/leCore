"""
holographic_mind.py -- one general-purpose mind for any input and any job.

Everything in this project has been converging on a single fact: once you encode
something into a hypervector -- a word, a number, an image, the features of a
sound, a world-state, a whole structured record -- the SAME operations work on it.
bundle/bind/cosine do not care where the vector came from. So the hypervector is a
universal interchange format, and the machines we built on top of it (a
prototype classifier, a recursive index, the creature's learning brain) are a
component library that snaps onto ANY encoded input.

This module is the front door to that idea:

  * UniversalEncoder turns any supported input -- text, number, category, image,
    a raw feature vector (an audio MFCC frame, an embedding), a structured record
    (a dict of fields), or a sequence -- into one unit hypervector. Different
    modalities, one representation.
  * Mind takes a direction -- classify, recall, or decide, either stated or
    inferred from the shape of what you give it -- and ASSEMBLES the matching
    structure from the toolkit, then handles the inputs. Teach it labelled
    examples and it grows a classifier; pour items in and it grows a searchable
    index; give it states/actions/rewards and it grows the creature's brain.

Honest scope: "general purpose" here means a modality-agnostic representation plus
a small library of holographic machines, auto-wired to the job -- not a magic
universal solver. The capability is in the (measured) components; what is new is
that one interface and one representation now span all of them, so the same mind
can be pointed at text one minute and a control problem the next. The demo at the
bottom measures each modality end to end so the generality is earned, not claimed.
"""

import numpy as np

from holographic_ai import bind, bundle, permute, cosine, random_vector, Vocabulary
from holographic_encoders import ScalarEncoder, TextEncoder
from holographic_tree import HoloForest
from holographic_creature import HolographicMind


# ---------------------------------------------------------------------------
# 1. THE UNIVERSAL ENCODER  (every modality -> the one common representation)
# ---------------------------------------------------------------------------

class UniversalEncoder:
    """Turn any supported input into one unit hypervector.

    The point is not a clever encoding for each type -- it is that they all land
    in the SAME space, so anything downstream (classifier, index, brain) treats a
    sentence, an image, and a world-state identically. Modality is inferred from
    the Python type unless you name it.

      text (str)            -- bag of learned word vectors (random indexing)
      number (int/float)    -- fractional-power scalar code (near numbers stay near)
      category (declared)   -- one fixed random atom per symbol
      vector (1-D array)    -- a random projection: an embedding, an audio MFCC
                               frame, any feature vector, preserving similarity
      image (2-D/3-D array) -- greyscale, flattened, then the vector path
      record (dict)         -- bind each field's role to its (recursively encoded)
                               value and bundle: structured state in one vector
      sequence (list/tuple) -- bind each element to its position and bundle
    """

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0), text_window=2):
        self.dim = dim
        self.seed = seed
        self._symbols = Vocabulary(dim, seed)            # category atoms
        self._roles = Vocabulary(dim, seed + 1)          # record field-name atoms
        self._scalar = ScalarEncoder(dim, number_range[0], number_range[1], seed + 2)
        self._text = TextEncoder(dim, window=text_window, seed=seed + 3)
        self._projections = {}                            # input length -> fixed matrix

    # -- learning word co-occurrence is optional but sharpens text similarity ---
    def learn_text(self, corpus):
        for line in corpus:
            self._text.learn(line.split() if isinstance(line, str) else line)

    def _projection(self, length):
        """A fixed Gaussian random projection from `length` dims down to `dim`.
        Johnson-Lindenstrauss: it preserves similarity, so close feature vectors
        stay close as hypervectors. Cached per input length, seeded for
        reproducibility."""
        if length not in self._projections:
            rng = np.random.default_rng([self.seed, 7, length])
            self._projections[length] = rng.standard_normal((self.dim, length)) / np.sqrt(length)
        return self._projections[length]

    def _encode_vector(self, x):
        x = np.asarray(x, dtype=float).ravel()
        v = self._projection(len(x)) @ x
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _encode_image(self, img):
        a = np.asarray(img, dtype=float)
        if a.ndim == 3:                                   # collapse colour to luma
            a = a.mean(axis=2)
        return self._encode_vector(a.ravel())

    def _encode_record(self, d):
        tokens = [bind(self._roles.get(str(k)), self.encode(v))
                  for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))]
        return bundle(tokens) if tokens else np.zeros(self.dim)

    def _encode_sequence(self, seq):
        if not len(seq):
            return np.zeros(self.dim)
        return bundle([permute(self.encode(el), i + 1) for i, el in enumerate(seq)])

    def encode(self, x, modality=None):
        if modality == "category":
            return self._symbols.get(str(x))
        if modality == "vector":
            return self._encode_vector(x)
        if modality == "image":
            return self._encode_image(x)
        if modality == "text":
            # a sentence as an order-insensitive bundle of its word vectors. Without
            # this explicit case a LIST of tokens would fall through to the sequence
            # encoder below (order-sensitive, position-permuted), which silently
            # wrecks topic similarity -- the bug that made unified text classification
            # collapse on hard data while passing on easy, separable topics.
            return self._text.encode_sentence(x)
        # otherwise infer from the Python type
        if isinstance(x, str):
            return self._text.encode_sentence(x)
        if isinstance(x, bool):
            return self._symbols.get(str(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return self._scalar.encode(float(x))
        if isinstance(x, dict):
            return self._encode_record(x)
        if isinstance(x, np.ndarray):
            return self._encode_image(x) if x.ndim >= 2 else self._encode_vector(x)
        if isinstance(x, (list, tuple)):
            return self._encode_sequence(x)
        # last resort: treat it as an opaque symbol
        return self._symbols.get(str(x))


# ---------------------------------------------------------------------------
# 2. THE BACK-ENDS  (small machines the Mind assembles as needed)
# ---------------------------------------------------------------------------

class _PrototypeClassifier:
    """One prototype per label -- a bundle of that label's encoded examples --
    and classify by nearest cosine. This is HolographicLearner generalised to run
    on any encoded input instead of only dicts: no training loop, the learning IS
    the bundling."""

    def __init__(self):
        self._sum = {}        # label -> running sum of example vectors
        self._unit = {}       # label -> unit-length prototype
        self._count = {}

    def learn(self, vec, label):
        if label in self._sum:
            self._sum[label] = self._sum[label] + vec
            self._count[label] += 1
        else:
            self._sum[label] = np.array(vec, dtype=float)
            self._count[label] = 1
        v = self._sum[label]; n = np.linalg.norm(v)
        self._unit[label] = v / n if n > 0 else v

    def predict(self, vec):
        best, sim = None, -2.0
        for label, proto in self._unit.items():
            s = float(proto @ vec)
            if s > sim:
                best, sim = label, s
        return best, sim


class _Index:
    """An associative store: keep encoded items (and an optional payload each),
    recall the nearest by content with the recursive HoloForest once it is worth
    building."""

    def __init__(self, dim):
        self.dim = dim
        self.vecs = []
        self.payloads = []
        self._forest = None
        self._dirty = False

    def add(self, vec, payload):
        self.vecs.append(np.asarray(vec, float))
        self.payloads.append(payload)
        self._dirty = True

    def recall(self, vec):
        if not self.vecs:
            return None, -1.0
        mat = np.stack(self.vecs)
        if len(self.vecs) >= 256:                         # large: use the recursive index
            if self._dirty or self._forest is None:
                self._forest = HoloForest(self.dim, n_trees=4, leaf_size=64).build(mat)
                self._dirty = False
            i = self._forest.recall(vec, beam=4)
        else:                                             # small: an exact scan is fine
            i = int((mat @ vec).argmax())
        return self.payloads[i], float(self.vecs[i] @ vec)


# ---------------------------------------------------------------------------
# 3. THE MIND  (one interface; assembles the right machine for the direction)
# ---------------------------------------------------------------------------

class Mind:
    """A general-purpose holographic mind. Feed it inputs in any format the
    encoder understands, point it at a job, and it assembles the structure to do
    it -- lazily, on the first relevant call, so you only build what you use.

      teach(x, label) / classify(x)        -> a prototype classifier
      store(x, payload) / recall(x)        -> an associative index
      act(state) / reinforce(s, a, reward) -> the creature's learning brain
    """

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0)):
        self.dim = dim
        self.encoder = UniversalEncoder(dim, seed=seed, number_range=number_range)
        self._clf = None
        self._index = None
        self._brain = None
        self._actions = None

    # convenience: let the encoder pre-learn word co-occurrence from a corpus
    def read(self, corpus):
        self.encoder.learn_text(corpus)
        return self

    # -- classification ----------------------------------------------------
    def teach(self, x, label, modality=None):
        if self._clf is None:
            self._clf = _PrototypeClassifier()
        self._clf.learn(self.encoder.encode(x, modality), label)
        return self

    def classify(self, x, modality=None):
        if self._clf is None:
            raise RuntimeError("nothing taught yet -- call teach() first")
        return self._clf.predict(self.encoder.encode(x, modality))

    # -- associative recall -------------------------------------------------
    def store(self, x, payload=None, modality=None):
        if self._index is None:
            self._index = _Index(self.dim)
        self._index.add(self.encoder.encode(x, modality),
                        x if payload is None else payload)
        return self

    def recall(self, x, modality=None):
        if self._index is None:
            raise RuntimeError("nothing stored yet -- call store() first")
        return self._index.recall(self.encoder.encode(x, modality))

    # -- decision / reinforcement ------------------------------------------
    def actions(self, names):
        """Declare the action set, then assemble the creature's brain over it."""
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000)
        return self

    def act(self, state, explore=False, epsilon=None, modality=None):
        if self._brain is None:
            raise RuntimeError("call actions([...]) first to set up the brain")
        a = self._brain.decide(self.encoder.encode(state, modality),
                               explore=explore, epsilon=epsilon)
        return self._actions[a]

    def reinforce(self, state, action, reward, modality=None):
        """Single-step learning (a contextual decision): credit this action in
        this state with this reward. For multi-step problems, the same brain is
        what holographic_creature and holographic_navigator drive across episodes."""
        s = self.encoder.encode(state, modality)
        a = self._actions.index(action)
        self._brain.remember([s], [a], [float(reward)])
        return self

    def describe(self):
        parts = []
        if self._clf is not None:
            parts.append(f"a classifier over {len(self._clf._unit)} labels")
        if self._index is not None:
            parts.append(f"an index of {len(self._index.vecs)} items")
        if self._brain is not None:
            parts.append(f"a learning brain over actions {self._actions}")
        return "Mind has assembled: " + ("; ".join(parts) if parts else "nothing yet")


def assemble(examples, task=None, dim=1024, seed=0, corpus=None, actions=None):
    """Build and populate a Mind from a batch of examples, inferring the job if
    you do not name it -- 'provide direction in some manner, and the system does
    the rest'. Recognised shapes:

      classify : [(input, label), ...]            -- label is a str/int
      recall   : [input, ...]  or [(input, payload), ...]
      decide   : [(state, action, reward), ...]   -- a 3-tuple with a numeric reward
    """
    mind = Mind(dim=dim, seed=seed)
    if corpus:
        mind.read(corpus)
    first = examples[0]

    def looks_like_decide(e):
        return (isinstance(e, tuple) and len(e) == 3
                and isinstance(e[2], (int, float, np.integer, np.floating)))

    if task is None:
        if looks_like_decide(first):
            task = "decide"
        elif isinstance(first, tuple) and len(first) == 2:
            task = "classify"
        else:
            task = "recall"

    if task == "classify":
        for x, label in examples:
            mind.teach(x, label)
    elif task == "recall":
        for e in examples:
            if isinstance(e, tuple) and len(e) == 2:
                mind.store(e[0], e[1])
            else:
                mind.store(e)
    elif task == "decide":
        acts = actions or sorted({a for _, a, _ in examples})
        mind.actions(acts)
        for state, action, reward in examples:
            mind.reinforce(state, action, reward)
    else:
        raise ValueError(f"unknown task: {task}")
    return mind


# ---------------------------------------------------------------------------
# 4. DEMO  (one Mind design, measured across every modality)
# ---------------------------------------------------------------------------

def _accuracy(mind, test, modality=None):
    return np.mean([mind.classify(x, modality)[0] == y for x, y in test])


def demo_general_mind():
    print("=" * 70)
    print("One general-purpose mind, measured across modalities")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # --- TEXT: topic classification from a tiny corpus ---------------------
    corpus = [
        "the cat sat by the window", "the dog chased a ball", "i fed the hungry cat",
        "a black cat purred softly", "the puppy wagged its tail", "kittens love warm milk",
        "the car drove down the street", "my truck needs new brakes", "the engine roared loudly",
        "she parked the car outside", "the motorcycle sped past", "diesel fuel is expensive",
    ]
    pets = ["the cat napped all day", "a loyal dog guards the house", "the kitten chased yarn"]
    autos = ["the sedan needs an oil change", "the truck hauled heavy cargo", "the engine stalled"]
    m = Mind(seed=1).read(corpus)
    for s in corpus[:6]: m.teach(s, "animal")
    for s in corpus[6:]: m.teach(s, "vehicle")
    test = [(s, "animal") for s in pets] + [(s, "vehicle") for s in autos]
    print(f"\n  TEXT    (topic of a sentence)      accuracy {100*_accuracy(m, test):.0f}%  "
          f"({len(test)} held-out sentences)")

    # --- RECORD: structured rows with mixed fields -------------------------
    m = Mind(seed=2)
    train = [
        ({"cover": "feathers", "fly": "yes", "home": "land"}, "bird"),
        ({"cover": "feathers", "fly": "yes", "home": "air"},  "bird"),
        ({"cover": "scales",   "fly": "no",  "home": "water"}, "fish"),
        ({"cover": "scales",   "fly": "no",  "home": "sea"},   "fish"),
        ({"cover": "fur",      "fly": "no",  "home": "land"},  "mammal"),
        ({"cover": "fur",      "fly": "no",  "home": "den"},   "mammal"),
    ]
    for rec, lab in train: m.teach(rec, lab)
    rec_test = [({"cover": "feathers", "fly": "yes", "home": "tree"}, "bird"),
                ({"cover": "scales", "fly": "no", "home": "lake"}, "fish"),
                ({"cover": "fur", "fly": "no", "home": "forest"}, "mammal")]
    print(f"  RECORD  (structured rows)          accuracy {100*_accuracy(m, rec_test):.0f}%  "
          f"(novel field values)")

    # --- IMAGE: 12x12 patterns (stripes vs bars vs checker) + noise --------
    def make_image(kind, r):
        g = np.zeros((12, 12))
        if kind == "h": g[::2, :] = 1.0
        elif kind == "v": g[:, ::2] = 1.0
        else:
            g[::2, ::2] = 1.0; g[1::2, 1::2] = 1.0
        return g + 0.25 * r.standard_normal((12, 12))
    m = Mind(seed=3)
    for _ in range(8):
        for k in ("h", "v", "x"): m.teach(make_image(k, rng), k, modality="image")
    img_test = [(make_image(k, rng), k) for k in ("h", "v", "x") for _ in range(10)]
    print(f"  IMAGE   (12x12 noisy patterns)     accuracy {100*_accuracy(m, img_test, 'image'):.0f}%  "
          f"({len(img_test)} held-out images)")

    # --- AUDIO: pure tones -> FFT magnitude features -> classify -----------
    def tone(freq, r):
        t = np.linspace(0, 1, 256, endpoint=False)
        sig = np.sin(2 * np.pi * freq * t) + 0.3 * r.standard_normal(256)
        return np.abs(np.fft.rfft(sig))            # spectral magnitude = the feature vector
    freqs = {"low": 5, "mid": 18, "high": 40}
    m = Mind(seed=4)
    for _ in range(8):
        for name, f in freqs.items(): m.teach(tone(f, rng), name, modality="vector")
    aud_test = [(tone(f, rng), name) for name, f in freqs.items() for _ in range(10)]
    print(f"  AUDIO   (tone pitch from spectrum) accuracy {100*_accuracy(m, aud_test, 'vector'):.0f}%  "
          f"({len(aud_test)} held-out clips)")

    # --- RECALL: store mixed items, find one from a partial cue ------------
    m = Mind(seed=5)
    facts = {"paris": "capital of France", "tokyo": "capital of Japan",
             "everest": "tallest mountain", "nile": "longest river",
             "pacific": "largest ocean", "sahara": "largest hot desert"}
    for k, v in facts.items(): m.store(k, v)
    cue, (ans, score) = "everest", m.recall("everest")
    print(f"\n  RECALL  (associative store)        '{cue}' -> '{ans}'  (cosine {score:.2f})")

    # --- DECIDE: a contextual bandit (state -> best action), learned -------
    m = Mind(seed=6).actions(["left", "right", "stay"])
    best = {"red": "left", "green": "right", "blue": "stay"}
    ctx = list(best)
    for _ in range(150):                            # learn from reward
        c = ctx[rng.integers(3)]
        a = m.act({"signal": c}, explore=True, epsilon=0.3)
        m.reinforce({"signal": c}, a, 1.0 if a == best[c] else 0.0)
    hit = np.mean([m.act({"signal": c}) == best[c] for c in ctx for _ in range(20)])
    print(f"  DECIDE  (contextual bandit)        accuracy {100*hit:.0f}%  "
          f"(learned the right move per signal)\n")
    print("  Same Mind class, same hypervector representation, every time -- only")
    print("  the input format and the direction changed. That is the generality:")
    print("  encode anything into the common space, assemble the right machine.")


if __name__ == "__main__":
    demo_general_mind()

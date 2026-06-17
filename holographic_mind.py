"""
holographic_mind.py -- the one shared encoder, and the small machines built on it.

Everything in this project has been converging on a single fact: once you encode
something into a hypervector -- a word, a number, an image, the features of a
sound, a world-state, a whole structured record -- the SAME operations work on it.
bundle/bind/cosine do not care where the vector came from. So the hypervector is a
universal interchange format, and the machines built on top of it (a prototype
classifier, a recursive index, the creature's learning brain) are a component
library that snaps onto ANY encoded input.

This module holds the shared pieces:

  * UniversalEncoder turns any supported input -- text, number, category, image,
    a raw feature vector (an audio MFCC frame, an embedding), a structured record
    (a dict of fields), or a sequence -- into one unit hypervector. Different
    modalities, one representation. It can also NAME the modality it would use
    (`infer`), which is what lets the top-level mind discover an input's kind
    instead of being told.
  * _PrototypeClassifier and _Index are the small back-ends the rest of the
    project composes.

HISTORY: this file used to also hold a `Mind` facade (teach/store/act, plus an
`assemble()` that guessed the task from data shape). It was retired: it
re-implemented thin versions of machinery that exists for real elsewhere --
exactly the failing `UnifiedMind` (holographic_unified.py) was built to fix.
The one good idea in `assemble()`, building a working mind straight from a pile
of examples, lives on as `UnifiedMind.absorb()`, on the real self-organizing
memory instead of a toy one.
"""

import numpy as np

from holographic_ai import bind, bundle, permute, Vocabulary
from holographic_encoders import ScalarEncoder, TextEncoder
from holographic_tree import HoloForest, ReflexCache


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

    def infer(self, x):
        """Name the modality this input would encode as -- the SELF-DISCOVERY step.

        This is the single source of truth for type dispatch: `encode` calls it
        when no modality is declared, and the top-level mind calls it to tag and
        route inputs it was never told about. Keeping both behind one function
        means the tag used for routing can never disagree with the encoding
        actually used -- the divergence that caused the original cross-modal bug.

        One rule needs a word of defence: a list/tuple of strings infers as TEXT
        (an order-insensitive bag of word vectors), not as a generic sequence.
        Naive type dispatch sent token lists to the order-sensitive sequence
        encoder, and that was measured to silently wreck topic similarity -- the
        bug that made unified text classification collapse on hard data while
        passing on easy, separable topics. Re-measured when this method was
        added: with the sequence rule, routing-by-inference scored 93.8% on the
        mixed-modality demo; with the text rule it scores 97.5%, exactly
        matching caller-declared tags. A genuinely ORDERED list of symbols is
        still available by declaring modality="sequence" explicitly.
        """
        if isinstance(x, str):
            return "text"
        if isinstance(x, bool):                       # bool before int: True is an int
            return "category"
        if isinstance(x, (int, float, np.integer, np.floating)):
            return "number"
        if isinstance(x, dict):
            return "record"
        if isinstance(x, np.ndarray):
            return "image" if x.ndim >= 2 else "vector"
        if isinstance(x, (list, tuple)):
            if len(x) and all(isinstance(t, str) for t in x):
                return "text"                          # token list = bag of words (see above)
            return "sequence"
        return "category"                              # last resort: an opaque symbol

    def encode(self, x, modality=None):
        if modality is None:
            modality = self.infer(x)
        if modality == "category":
            return self._symbols.get(str(x))
        if modality == "vector":
            return self._encode_vector(x)
        if modality == "image":
            return self._encode_image(x)
        if modality == "number":
            return self._scalar.encode(float(x))
        if modality in ("text", "code"):
            # a sentence (or token list) as an order-insensitive bundle of its word
            # vectors -- see infer() for why token lists must land here and not in
            # the sequence encoder. "code" ENCODES exactly like text (its tokens
            # learn co-occurrence the same way); the distinct name exists for
            # ROUTING, so code labels and prose labels can live in one memory
            # without competing. Before this case existed, a declared "code" fell
            # to the opaque-symbol path below and two nearly identical snippets
            # encoded as orthogonal -- measured, cosine 0.04 -- a silent foot-gun.
            return self._text.encode_sentence(x)
        if modality == "record":
            return self._encode_record(x)
        if modality == "sequence":
            return self._encode_sequence(x)
        # an unrecognised modality name: treat the input as an opaque symbol
        return self._symbols.get(str(x))

    # -- persistence: everything but the learned word co-occurrence is seed-derived, so a
    # round-trip stores config + the text encoder's learned context and rebuilds the rest.
    def to_state(self):
        return {
            "dim": int(self.dim), "seed": int(self.seed),
            "number_range": [float(self._scalar.lo), float(self._scalar.hi)],
            "text_window": int(self._text.window),
            "text": self._text.to_state(),
        }

    @classmethod
    def from_state(cls, state):
        enc = cls(int(state["dim"]), seed=int(state["seed"]),
                  number_range=tuple(state["number_range"]),
                  text_window=int(state["text_window"]))
        enc._text = TextEncoder.from_state(state["text"])
        return enc


# ---------------------------------------------------------------------------
# 2. THE BACK-ENDS  (small machines the rest of the project composes)
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
    recall the nearest by content -- an exact scan until the store is genuinely
    big, then the recursive HoloForest fronted by a slime-mould ReflexCache."""

    def __init__(self, dim):
        self.dim = dim
        self.vecs = []
        self.payloads = []
        self._forest = None
        self._reflex = None
        self._mat = None          # cached stack of vecs (re-stacking 16k x 1024
        self._dirty = False       # vectors cost a measured 54 ms PER CALL)

    def add(self, vec, payload):
        self.vecs.append(np.asarray(vec, float))
        self.payloads.append(payload)
        self._dirty = True

    def recall(self, vec):
        if not self.vecs:
            return None, -1.0
        if self._dirty or self._mat is None:
            self._mat = np.stack(self.vecs)
        mat = self._mat
        # The switch-over point is MEASURED, not assumed: a single numpy matmul
        # scan is exact AND faster than the tree's Python-level routing until
        # roughly 4,000 items at dim 1024 (scan 1.9ms vs forest 1.7ms there; at
        # 256 items the scan is ~7x faster and the forest already costs recall).
        # Below the crossover the forest would pay MORE wall-clock for LESS
        # accuracy, so the exact scan keeps the job until the data is genuinely
        # big; past it the forest's O(leaf.logN) wins and keeps winning (2.5ms vs
        # 46ms at 64k items).
        if len(self.vecs) >= 4096:                        # big: the recursive index
            if self._dirty or self._forest is None:
                self._forest = HoloForest(self.dim, n_trees=4, leaf_size=64).build(mat)
                self._reflex = ReflexCache(len(self.vecs), hot_size=48)
                self._dirty = False
            # SLIME-MOULD FAST PATH: check the most-recalled items first (veins
            # thicken with use). Measured at N=16k on a Zipf workload: the reflex
            # answers 70% of queries, recall@1 RISES 96.8% -> 99.0% (a popular
            # noisy cue snaps to the right hot item where the beam sometimes
            # misses), and cost drops 1.67 -> 0.52 ms/query; under a popularity
            # SHIFT it re-adapts within the rebuild period (98.5% at 0.66 ms);
            # on a uniform stream the flux guard deactivates it and the cost is
            # a wash (1.60 vs 1.63 ms) -- the habit never costs more than it saves.
            i, _ = self._reflex.consider(vec, mat)
            if i is None:
                i = self._forest.recall(vec, beam=4)
                self._reflex.reinforce(i, False, mat)
            else:
                self._reflex.reinforce(i, True, mat)
        else:                                             # small/medium: exact scan
            self._dirty = False
            i = int((mat @ vec).argmax())
        return self.payloads[i], float(self.vecs[i] @ vec)

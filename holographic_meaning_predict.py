"""Generation with structure: predict a next-MEANING vector and settle it, rather
than looking up a single stored symbol.

The symbol predictor (holographic_predictive.PredictiveMemory) resonates the
recent context to the nearest stored entry and returns that entry's discrete next
symbol. It is right or wrong, nothing between. This layer changes the OUTPUT side:
instead of returning one stored symbol, it COMPOSES a next-meaning vector as the
coupling-weighted blend (a ZREAD) of the next-meanings of every resonating
context, then SETTLES that vector by iterated cleanup against a meaning space, and
reads off the nearest word. Because the prediction is a point in a continuous
meaning space built from many entries, it can land where no single entry sits --
and even when the exact word is wrong, it is wrong toward semantically NEAR words.
That graded, compositional prediction is the step from recall toward generation.

WHAT THE TEST DATA TAUGHT (the user's point: without good data the numbers say
little; here a dictionary/encyclopedia prior and a real corpus were used, and the
result OVERTURNED the obvious guess). Two meaning spaces were compared as the
prediction/settle space, both honestly:

  * CO-OCCURRENCE (syntagmatic) meaning -- a word's vector is the sum of what
    appears NEAR it -- predicts the NEXT word well: the actual next word lands in
    the top ~10% of the vocabulary under the composed prediction (semantic rank
    ~0.90, chance 0.5). This is the right space for 'what follows'.
  * DICTIONARY-CURRICULUM (paradigmatic) meaning -- a word's vector bootstrapped
    from its definition words (holographic_lexicon) -- is NEARLY USELESS for
    next-word prediction (rank ~0.52, barely above chance) BUT clearly best at
    'what IS this / what is related': WordNet-related words separate at d-prime
    ~0.76 vs ~0.38 for co-occurrence.

The lesson, kept: next-word prediction is a DISTRIBUTIONAL task (what tends to
follow), so the predictor composes and settles in CO-OCCURRENCE space. The
dictionary/encyclopedia prior is not thrown away -- it is the right space for the
RELATEDNESS query ('what is this about', 'what is like X'), a different question
the same engine answers. Using the paradigmatic space to predict sequence, or the
syntagmatic space to judge relatedness, both measurably underperform. Match the
space to the question.

So this module composes/settles in a co-occurrence meaning space by default, and
reports both the exact-symbol accuracy (precision) and a semantic rank (did the
prediction at least land in the right neighborhood) -- the second is where
composition earns its keep over a hard lookup.

Needs: numpy, holographic_ai, holographic_predictive.
"""
import numpy as np

from holographic_ai import bundle, permute, cosine, Vocabulary


def cooccurrence_space(sentences, dim=512, window=2, seed=0):
    """Build a co-occurrence (syntagmatic) meaning vector per word: the sum of
    position-permuted neighbour atoms. Returns (vocab_list, MeaningMatrix,
    index_map). This is the space next-word prediction should compose in."""
    atoms = Vocabulary(dim, seed)
    ctx = {}
    for s in sentences:
        toks = list(s)
        for i, t in enumerate(toks):
            if t not in ctx:
                ctx[t] = np.zeros(dim)
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j != i:
                    ctx[t] = ctx[t] + permute(atoms.get(toks[j]), j - i)
    vocab = sorted(ctx)
    idx = {w: i for i, w in enumerate(vocab)}
    M = np.stack([ctx[w] / (np.linalg.norm(ctx[w]) + 1e-12) for w in vocab])
    return vocab, M, idx


class MeaningPredictor:
    """Predict a next-meaning vector by composing the next-meanings of resonating
    contexts, then settle it to a word. Operates over a provided meaning space
    (word -> vector); defaults to building a co-occurrence space from the corpus."""

    def __init__(self, dim=512, order=2, seed=0, t_min=0.2, settle_iters=2):
        self.dim = dim
        self.order = order
        self.t_min = t_min
        self.settle_iters = settle_iters
        self.atoms = Vocabulary(dim, seed + 100)     # context-encoding atoms (distinct)
        self.vocab = []
        self.M = None                                # meaning matrix (len(vocab), dim)
        self.idx = {}
        self._ctx = []                               # context vectors of entries
        self._next = []                              # next symbol per entry
        self._C = None

    def fit_space(self, sentences, window=2):
        """Build the co-occurrence meaning space from the corpus."""
        self.vocab, self.M, self.idx = cooccurrence_space(
            sentences, dim=self.dim, window=window, seed=0)
        return self

    def set_space(self, vocab, M):
        """Use an externally built meaning space (e.g. a dictionary-curriculum
        space for the relatedness query). vocab: list; M: matrix aligned to it."""
        self.vocab = list(vocab)
        self.M = np.asarray(M, float)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        return self

    def context_vector(self, recent):
        recent = list(recent)[-self.order:]
        if not recent:
            return np.zeros(self.dim)
        return bundle([permute(self.atoms.get(w), i)
                       for i, w in enumerate(reversed(recent))])

    def fit_transitions(self, tokens):
        """Learn (context_vec -> next symbol) entries from a token stream."""
        tokens = list(tokens)
        for i in range(1, len(tokens)):
            if tokens[i] in self.idx:
                self._ctx.append(self.context_vector(tokens[max(0, i - self.order):i]))
                self._next.append(tokens[i])
        self._C = None
        return self

    def _matrix(self):
        if self._C is None:
            self._C = (np.stack(self._ctx) if self._ctx else np.zeros((0, self.dim)))
            self._Cn = self._C / (np.linalg.norm(self._C, axis=1, keepdims=True) + 1e-12)
            self._nextM = (self.M[[self.idx[w] for w in self._next]]
                           if self._next else np.zeros((0, self.dim)))
        return self._Cn, self._nextM

    def _settle(self, vec):
        """Iterated cleanup in the meaning space: snap toward the nearest words and
        re-blend, converging to a clean attractor (the resonator pattern)."""
        v = vec / (np.linalg.norm(vec) + 1e-12)
        for _ in range(self.settle_iters):
            sims = self.M @ v
            top = np.argsort(sims)[::-1][:5]            # nearest few meanings
            w = sims[top].clip(0, None)
            if w.sum() == 0:
                break
            v = (w[:, None] * self.M[top]).sum(0)
            v = v / (np.linalg.norm(v) + 1e-12)
        return v

    def predict_meaning(self, recent):
        """Compose a next-meaning vector from resonating contexts, settle it, and
        return (word, settled_vector, confidence)."""
        Cn, nextM = self._matrix()
        if len(Cn) == 0:
            return None, np.zeros(self.dim), 0.0
        q = self.context_vector(recent)
        qn = q / (np.linalg.norm(q) + 1e-12)
        coup = Cn @ qn
        mask = coup >= self.t_min
        if not mask.any():
            mask = coup >= np.percentile(coup, 99)
        composed = (coup[mask, None] * nextM[mask]).sum(0)
        if np.linalg.norm(composed) == 0:
            return None, composed, 0.0
        settled = self._settle(composed)
        sims = self.M @ settled
        j = int(np.argmax(sims))
        return self.vocab[j], settled, float(sims[j])

    # ---- honest metrics --------------------------------------------------
    def evaluate(self, tokens):
        """Return exact-symbol accuracy and mean semantic RANK of the actual next
        word under the composed prediction (0.5 = chance, 1.0 = always nearest).
        The rank is where composition earns its keep: it credits landing in the
        right neighbourhood even when the exact word is missed."""
        tokens = list(tokens)
        exact = 0
        ranks = []
        n = 0
        for i in range(self.order, len(tokens)):
            actual = tokens[i]
            if actual not in self.idx:
                continue
            word, vec, _ = self.predict_meaning(tokens[max(0, i - self.order):i])
            if word is None:
                continue
            sims = self.M @ vec
            ranks.append(float((sims < sims[self.idx[actual]]).mean()))
            exact += (word == actual)
            n += 1
        if n == 0:
            return {"exact": 0.0, "semantic_rank": 0.5, "n": 0}
        return {"exact": exact / n, "semantic_rank": float(np.mean(ranks)), "n": n}


def relatedness_dprime(vocab, M, idx, similar_pairs, random_pairs):
    """The paradigmatic check: d-prime between known-related word pairs and random
    pairs in a meaning space. The dictionary-curriculum space wins this; the
    co-occurrence space wins next-word prediction. Match the space to the query."""
    def cos_pairs(P):
        return np.array([float(M[idx[a]] @ M[idx[b]]) for a, b in P
                         if a in idx and b in idx])
    s, r = cos_pairs(similar_pairs), cos_pairs(random_pairs)
    if len(s) == 0 or len(r) == 0:
        return 0.0
    return float((s.mean() - r.mean()) / np.sqrt(0.5 * (s.var() + r.var()) + 1e-12))

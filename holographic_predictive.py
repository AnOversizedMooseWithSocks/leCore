"""A predictive loop on the holographic substrate: turn a passive associative
store into an active model that ANTICIPATES its next input, measures how
surprised it was, and corrects itself in proportion to that surprise.

This is predictive coding (Rao & Ballard, 1999) built from bind/bundle/permute
instead of gradient descent. The living cycle, one step at a time:

    predict   -- from the recent context, resonate the memory to a next symbol
    measure   -- surprise = 1 - cosine(predicted, actual); 0 means it knew
    correct   -- reinforce if it was right, nudge or create an entry if wrong,
                 the size of the update SCALED BY the surprise (error-gated:
                 familiar input barely moves the model, novel input moves it)
    report    -- self_free_energy (how self-consistent the model is) and
                 valence (whether this step improved that consistency)

WHAT IS GENUINELY NEW HERE for this engine (it stored and retrieved but did not
ACT on what it held):
  * PREDICTION BY RESONANCE, NOT EXACT MATCH. Each entry is a context VECTOR
    (order-aware, via permute) paired with a next symbol. Prediction resonates
    the query context against all stored contexts by cosine and reads back the
    best (hard) or a coupling-weighted blend (soft). A context never seen
    exactly still predicts sensibly if a SIMILAR one was seen -- generalization
    an exact n-gram backoff cannot do. Confidence (the resonance score) cleanly
    separates a known continuation (~1.0) from a generalized guess (~0.5).
  * SURPRISE AS A FIRST-CLASS SIGNAL. Every step yields surprise in [0, 1]. The
    model learns error-gated: it spends effort where it was wrong, not uniformly.
  * SELF-CONSISTENCY READOUTS. self_free_energy = distance between the model's
    running state and what it predicts for itself; it falls toward 0 as the model
    becomes a fixed point of its own field. valence = the per-step change in that
    consistency -- a signed 'did this help' that turns positive once a pattern is
    integrated.
  * GENERATION BY ANTICIPATION. Feed a seed, predict the next symbol, append,
    repeat -- the same predictor used for learning runs forward to produce a
    sequence.

These are the active verbs the substrate was missing. The memory now anticipates,
notices when it is wrong, and changes itself accordingly.

Needs: numpy, holographic_ai.
"""
from dataclasses import dataclass

import numpy as np

from holographic_ai import Vocabulary, bundle, permute, cosine


@dataclass
class Step:
    """Everything observable about one predictive step."""
    predicted: object        # the symbol the model expected
    actual: object           # what actually came
    surprise: float          # 1 - cosine(predicted_vec, actual_vec), in [0, 2]
    confidence: float        # resonance score of the prediction
    hit: bool                # predicted == actual
    action: str              # 'reinforce' | 'correct' | 'create'
    self_free_energy: float  # model's distance from its own prediction
    valence: float           # prev_sfe - sfe (>0 = became more self-consistent)


def zread(query, contexts, values, t_min=0.5, ordered=True):
    """Soft, coupling-weighted read (the 'population' read): blend the `values`
    whose `contexts` resonate with `query`, each weighted by max(0, cosine)
    above the participation gate t_min. Entries below t_min contribute nothing.
    Returns the blended value vector (unnormalised bundle). If `ordered`, the
    blend respects entry order by folding in a small positional permutation --
    a path-aware read rather than a commutative sum."""
    if len(contexts) == 0:
        return np.zeros_like(query)
    q = query / (np.linalg.norm(query) + 1e-12)
    C = contexts / (np.linalg.norm(contexts, axis=1, keepdims=True) + 1e-12)
    couplings = C @ q
    parts = []
    for i, (t, v) in enumerate(zip(couplings, values)):
        if t >= t_min:
            vv = permute(v, i % 7) if ordered else v
            parts.append(t * vv)
    if not parts:
        return np.zeros_like(query)
    return np.sum(parts, axis=0)


class PredictiveMemory:
    """Predict the next symbol from recent context, measure surprise, learn
    error-gated, and report self-consistency. Built on the shared VSA primitives."""

    def __init__(self, dim=2048, order=2, seed=0,
                 reinforce_threshold=0.15, novelty_threshold=0.55):
        self.dim = dim
        self.order = order
        self.symbols = Vocabulary(dim, seed)
        # parallel arrays: each entry is (context_vec, next_symbol, support, next_vec)
        self._ctx = []          # list of context vectors
        self._next_sym = []     # next symbol per entry
        self._next_vec = []     # next symbol's atom (cached)
        self._support = []      # reinforcement count
        self._C = None          # cached stacked context matrix (built lazily)
        self.reinforce_threshold = reinforce_threshold
        self.novelty_threshold = novelty_threshold
        self.prev_sfe = None

    # ---- context encoding ------------------------------------------------
    def context_vector(self, recent):
        """Order-aware context: most-recent symbol at position 0, bundled. Stable
        across context lengths so partial contexts still resonate."""
        recent = list(recent)[-self.order:]
        if not recent:
            return np.zeros(self.dim)
        return bundle([permute(self.symbols.get(w), i)
                       for i, w in enumerate(reversed(recent))])

    def _matrix(self):
        if self._C is None or len(self._C) != len(self._ctx):
            self._C = np.stack(self._ctx) if self._ctx else np.zeros((0, self.dim))
        return self._C

    # ---- prediction ------------------------------------------------------
    def predict(self, recent, soft=False):
        """Predict the next symbol. Hard (default): nearest stored context, return
        its next symbol. Soft: ZREAD-blend the next-vectors, clean up to a symbol.
        Returns (symbol, confidence)."""
        if not self._ctx:
            return None, 0.0
        q = self.context_vector(recent)
        qn = q / (np.linalg.norm(q) + 1e-12)
        C = self._matrix()
        sims = C @ qn
        if soft:
            blend = zread(q, C, self._next_vec, t_min=self.reinforce_threshold)
            return self._cleanup(blend)
        j = int(np.argmax(sims))
        return self._next_sym[j], float(sims[j])

    def _cleanup(self, vec):
        """Snap a noisy next-vector estimate to the nearest known symbol."""
        if np.linalg.norm(vec) == 0 or not self.symbols.vectors:
            return None, 0.0
        names = list(self.symbols.vectors)
        M = np.stack([self.symbols.vectors[n] for n in names])
        v = vec / (np.linalg.norm(vec) + 1e-12)
        sims = M @ v
        j = int(np.argmax(sims))
        return names[j], float(sims[j])

    # ---- the living step -------------------------------------------------
    def step(self, recent, actual, learn=True):
        """One predict -> measure -> correct cycle. Returns a Step with all
        observable signals."""
        ctx = self.context_vector(recent)
        actual_vec = self.symbols.get(actual)
        pred_sym, conf = self.predict(recent)
        pred_vec = self.symbols.get(pred_sym) if pred_sym is not None else np.zeros(self.dim)
        surprise = 1.0 - cosine(pred_vec, actual_vec) if pred_sym is not None else 1.0
        action = "none"
        if learn:
            action = self._correct(ctx, actual, actual_vec, surprise)
        # FREE ENERGY = the model's smoothed running prediction error. It falls
        # toward 0 as the model learns to anticipate its input -- the model
        # becoming a fixed point of the stream it sees (predictive coding's free
        # energy is exactly expected surprise). A low-pass of per-step surprise.
        s = max(0.0, min(1.0, surprise))
        sfe = s if self.prev_sfe is None else 0.9 * self.prev_sfe + 0.1 * s
        valence = (self.prev_sfe - sfe) if self.prev_sfe is not None else 0.0
        self.prev_sfe = sfe
        return Step(predicted=pred_sym, actual=actual, surprise=float(surprise),
                    confidence=float(conf), hit=(pred_sym == actual), action=action,
                    self_free_energy=float(sfe), valence=float(valence))

    def _correct(self, ctx, actual, actual_vec, surprise):
        """Error-gated write. Low surprise: reinforce the matching entry (no
        geometric change). Medium: nudge the nearest matching context toward this
        one (slerp-like), scaled by surprise. High: create a new entry. Mirrors a
        three-layer ingest routing, on this substrate."""
        ctxn = ctx / (np.linalg.norm(ctx) + 1e-12)
        # find the nearest entry that already predicts `actual`
        best_j, best_s = -1, -1.0
        for j, (c, sym) in enumerate(zip(self._ctx, self._next_sym)):
            if sym == actual:
                s = float(c @ ctxn / (np.linalg.norm(c) + 1e-12))
                if s > best_s:
                    best_j, best_s = j, s
        if best_j >= 0 and surprise <= self.reinforce_threshold:
            self._support[best_j] += 1
            return "reinforce"
        if best_j >= 0 and surprise <= self.novelty_threshold:
            # move that entry's context toward the new one, by the surprise
            a = min(0.5, surprise)
            self._ctx[best_j] = bundle([(1 - a) * self._ctx[best_j], a * ctx])
            self._support[best_j] += 1
            self._C = None
            return "correct"
        # create
        self._ctx.append(ctx)
        self._next_sym.append(actual)
        self._next_vec.append(actual_vec)
        self._support.append(1)
        self._C = None
        return "create"

    # ---- sequence-level verbs -------------------------------------------
    def learn_sequence(self, tokens, learn=True):
        """Run the loop over a token sequence; return the list of Steps (the
        surprise/valence trace). With learn=False this is pure evaluation."""
        tokens = list(tokens)
        steps = []
        for i in range(1, len(tokens)):
            steps.append(self.step(tokens[max(0, i - self.order):i], tokens[i], learn=learn))
        return steps

    def generate(self, seed, length=30, soft=False):
        """Generate by anticipation: predict the next symbol, append, repeat."""
        out = list(seed)
        for _ in range(length):
            sym, conf = self.predict(out[-self.order:], soft=soft)
            if sym is None:
                break
            out.append(sym)
        return out[len(seed):]

    def predict_accuracy(self, tokens):
        """Top-1 next-symbol accuracy on a token sequence WITHOUT learning -- how
        often the model's best guess is right. The headline measurement."""
        steps = self.learn_sequence(tokens, learn=False)
        if not steps:
            return 0.0
        return sum(s.hit for s in steps) / len(steps)

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

from holographic.agents_and_reasoning.holographic_ai import Vocabulary, bundle, permute, cosine


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


def zread(query, contexts, values, t_min=0.5, ordered=True, weights=None):
    """Soft, coupling-weighted read (the 'population' read): blend the `values`
    whose `contexts` resonate with `query`, each weighted by max(0, cosine)
    above the participation gate t_min. Entries below t_min contribute nothing.
    Returns the blended value vector (unnormalised bundle). If `ordered`, the
    blend respects entry order by folding in a small positional permutation --
    a path-aware read rather than a commutative sum.

    `weights` (optional, one per entry) multiplies each entry's coupling -- pass
    the per-entry SUPPORT (reinforcement count) to make the blend FREQUENCY-WEIGHTED
    rather than relevance-only. This is what makes a soft next-symbol read MAP-correct
    when one context has several successors at different rates: without it, two equally-
    resonant entries (cosine 1.0) contribute equally regardless of how often each was
    seen, so a 70/30 successor split blends 50/50 and decodes to the wrong symbol."""
    if len(contexts) == 0:
        return np.zeros_like(query)
    q = query / (np.linalg.norm(query) + 1e-12)
    C = contexts / (np.linalg.norm(contexts, axis=1, keepdims=True) + 1e-12)
    couplings = C @ q
    parts = []
    for i, (t, v) in enumerate(zip(couplings, values)):
        if t >= t_min:
            vv = permute(v, i % 7) if ordered else v
            wt = t if weights is None else t * max(float(weights[i]), 0.0)
            parts.append(wt * vv)
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
            # frequency-weighted (support) blend, no storage-order permutation: a soft read of the
            # NEXT symbol must weight candidates by how often each was seen, not by resonance alone,
            # or a context with several successors decodes to the minority (measured: a 70/30 split
            # blended 50/50 and returned the 30% symbol). Support-weighting makes it MAP-correct.
            blend = zread(q, C, self._next_vec, t_min=self.reinforce_threshold,
                          weights=self._support, ordered=False)
            return self._cleanup(blend)
        j = int(np.argmax(sims))
        return self._next_sym[j], float(sims[j])

    def next_distribution(self, recent):
        """The distribution over next symbols at this context, as {symbol: weight}.

        WHY this exists: `predict` collapses the evidence to a single symbol (argmax or a soft blend),
        which is right for PREDICTION but LIMIT-CYCLES when looped as a GENERATOR (measured: a greedy
        recipe generator gave MMD2 0.599 and 15x the real verbatim-copy rate, looping on the top
        continuation). A generator must SAMPLE, and to sample you need the whole distribution, not the
        winner. This exposes it, built from the SAME similarities `predict` already computes, so nothing
        is recomputed or reinvented -- the weight of each candidate next-symbol is its stored context's
        resonance with the query, times its support (how often that continuation was seen). Support
        weighting is the same MAP-correctness fix `predict(soft=True)` documents: a 70/30 successor split
        must read 70/30, not 50/50.
        """
        if not self._ctx:
            return {}
        q = self.context_vector(recent)
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = self._matrix() @ qn
        # accumulate resonance*support per candidate next-symbol (a context may recur with the same next)
        dist = {}
        for sym, s, sup in zip(self._next_sym, sims, self._support):
            w = max(float(s), 0.0) * float(sup)      # only positive resonance votes; support weights it
            if w > 0:
                dist[sym] = dist.get(sym, 0.0) + w
        return dist

    def sample(self, recent, temperature=1.0, top_p=1.0, rng=None):
        """Sample the next symbol (the GENERATION dual of `predict`). Delegates the temperature+nucleus
        draw to holographic_tokensample.sample_from_distribution -- the same primitive the character
        generator uses -- over this memory's `next_distribution`. Returns (symbol, weight) or (None, 0.0).

        This is the fix for the deterministic limit-cycle: `predict` for the single best guess, `sample`
        for a diverse, distribution-faithful continuation.
        """
        from holographic.agents_and_reasoning.holographic_tokensample import sample_from_distribution
        dist = self.next_distribution(recent)
        sym = sample_from_distribution(dist, temperature=temperature, top_p=top_p, rng=rng)
        return (sym, dist.get(sym, 0.0)) if sym is not None else (None, 0.0)

    def generate_sampled(self, seed, length=30, temperature=1.0, top_p=1.0, seed_rng=0):
        """Generate by SAMPLING (not argmax): the non-limit-cycling generator. Mirrors `generate` but
        draws each next symbol from the distribution, so it does not lock onto one continuation and loop."""
        rng = np.random.default_rng(seed_rng)
        out = list(seed)
        for _ in range(length):
            sym, _ = self.sample(out[-self.order:], temperature=temperature, top_p=top_p, rng=rng)
            if sym is None:
                break
            out.append(sym)
        return out[len(seed):]

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
        if best_j >= 0 and surprise <= self.reinforce_threshold and best_s >= self.novelty_threshold:
            self._support[best_j] += 1
            return "reinforce"
        if best_j >= 0 and surprise <= self.novelty_threshold and best_s >= self.novelty_threshold:
            # move that entry's context toward the new one, by the surprise
            a = min(0.5, surprise)
            self._ctx[best_j] = bundle([(1 - a) * self._ctx[best_j], a * ctx])
            self._support[best_j] += 1
            self._C = None
            return "correct"
        # create. WHY the best_s floor above: reinforce/nudge used to gate on SURPRISE alone, so a
        # prediction that happened to be right (argmax over near-zero sims) reinforced the nearest
        # SAME-SUCCESSOR entry even at similarity ~0 -- the transition's mass landed on an UNRELATED
        # context (measured: a (ctx -> next) pair identical to a stored one read back as an EMPTY
        # distribution because its support had been absorbed by the zero-context row). "Nearest
        # matching entry" must actually MATCH: below the novelty similarity line, this context is a
        # new home, not a reinforcement of an old one. Pinned by
        # test_repeated_recipe_context_is_stored_not_absorbed.
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

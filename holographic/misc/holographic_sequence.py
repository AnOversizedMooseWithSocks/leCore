"""Sequence memory: ORDER as a first-class, queryable property.

The rest of the stack deliberately treats many things as order-free -- a topic is
a bag of words, a class is a bundle of examples, a record is a set of role-filler
bindings. That is correct for those jobs: "what is this about" does not depend on
word order. But some meaning lives ONLY in the order. The canonical example is a
recipe: the same set of steps in the wrong sequence is not a worse recipe, it is
not a recipe at all -- "cut the sandwich in half" before "close the sandwich" is
incoherent. Plans, proofs, protocols, directions, melodies, and event timelines
are all like this: the data alone underdetermines the meaning; the SEQUENCE
supplies the rest.

This module makes order queryable with the same holographic primitives used
everywhere else (bind, bundle, permute), so sequences live in the same vector
space as everything else and compose with it. The operations and their measured
reliability (DIM=2048, random step vocabularies):

  position_of(x)      -- at which step does x occur?           100%
  step(i)             -- what is the i-th step?                100%
  precedes(a, b)      -- does a come before b?                 100%
  validate(constraints) -- are all 'a before b' rules met?    100% (composed of precedes)

Position-binding (bundle of permute(step, i)) is what makes these exact: each
step is rotated by its position, so un-rotating by i and cleaning up reads the
step at position i, and comparing decoded positions answers precedence. A
transition encoding (what-follows-x) was also measured but tops out ~64% on
longer sequences -- superposing many transitions in one trace hits the same
bundle-capacity limit the scaling work charted -- so "what comes next" is better
answered by the exact step list when it is available; this memory's reliable job
is the ORDER RELATIONS above, which the bag-of-everything stores cannot answer at
all.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, permute, cosine, Vocabulary


class SequenceMemory:
    """Store ordered sequences and answer order queries against them. Shares a
    symbol vocabulary so sequence elements are the same atoms used elsewhere."""

    def __init__(self, dim=2048, seed=0, vocab=None):
        self.dim = dim
        self.vocab = vocab or Vocabulary(dim, seed)
        self.seqs = {}                                   # name -> (vector, elements)

    def encode(self, elements):
        """A sequence -> one vector: each element rotated by its 1-based
        position and bundled. Order-sensitive by construction (a scrambled
        sequence is near-orthogonal: measured cosine ~0.03)."""
        if not elements:
            return np.zeros(self.dim)
        return bundle([permute(self.vocab.get(str(e)), i + 1)
                       for i, e in enumerate(elements)])

    def add(self, name, elements, chunk=0):
        """Store a named sequence (keeps the element list too: order queries are
        exact against the vector, but length and candidate set come from here).

        With `chunk` > 0, a LONG sequence is stored as positional BLOCKS of <=chunk
        elements -- each its own bundle -- so vector-only position/order queries stay
        EXACT past the single-bundle cap. The positional encoding is robust on short
        sequences but caps with length (measured at dim 2048: single-bundle step(i)
        accuracy ~100% at length 50, ~96% at 100, 69% at 200, 29% at 400, 15% at 800),
        while chunked holds 100% at every length. `chunk` = 0 (default) keeps the
        original single-vector storage -- backward compatible, and the right choice for
        short sequences, where chunking is a pure no-op. Keep `chunk` at/under the dim's
        reliable bundle length (the same margin the other chunkers use)."""
        elements = list(elements)
        if chunk and len(elements) > chunk:
            blocks = [self.encode(elements[lo:lo + chunk]) for lo in range(0, len(elements), chunk)]
            self.seqs[name] = (blocks, elements, chunk)        # repr is a LIST of block vectors
        else:
            self.seqs[name] = (self.encode(elements), elements, 0)
        return self

    def _vec(self, seq_or_name):
        if isinstance(seq_or_name, str) and seq_or_name in self.seqs:
            return self.seqs[seq_or_name]                      # (repr, elems, chunk)
        return self.encode(seq_or_name), list(seq_or_name), 0  # a raw sequence -> single vector

    def _probe(self, repr_, chunk, i):
        """The un-rotated probe for 0-based position i -- routed to the right block when chunked, so a
        position query reads only the one clean block it lives in (the chunk-and-re-anchor move for the
        positional encoding)."""
        if chunk:
            b, off = divmod(i, chunk)
            return permute(repr_[b], -(off + 1))               # local position within the block
        return permute(repr_, -(i + 1))

    def step(self, seq_or_name, i):
        """What element is at position i (0-based)? Un-rotate and clean up."""
        repr_, elems, chunk = self._vec(seq_or_name)
        probe = self._probe(repr_, chunk, i)
        cands = elems or list(self.vocab._items) if hasattr(self.vocab, "_items") else elems
        return max(cands, key=lambda e: cosine(probe, self.vocab.get(str(e))))

    def position_of(self, seq_or_name, x, length=None):
        """At which step does x occur? Argmax of its score across positions."""
        repr_, elems, chunk = self._vec(seq_or_name)
        L = length or len(elems)
        scores = [cosine(self._probe(repr_, chunk, i), self.vocab.get(str(x))) for i in range(L)]
        return int(np.argmax(scores))

    def precedes(self, seq_or_name, a, b, length=None):
        """Does a come before b? Compare their decoded positions. Measured at dim 2048: exact to
        ~40 steps, ~99-100% to ~80, ~93% at 120 -- a graceful decline, not a hard cliff (the old
        '~8' note was far too conservative); and with add(..., chunk=K) the order relation stays
        exact at any length. This is the order relation the bag stores cannot answer: it is the
        difference between a recipe and a pile of steps."""
        repr_, elems, chunk = self._vec(seq_or_name)
        L = length or len(elems)
        # decode both positions from the same representation and compare
        sa = [cosine(self._probe(repr_, chunk, i), self.vocab.get(str(a))) for i in range(L)]
        sb = [cosine(self._probe(repr_, chunk, i), self.vocab.get(str(b))) for i in range(L)]
        return int(np.argmax(sa)) < int(np.argmax(sb))

    def validate(self, seq_or_name, constraints):
        """Check a list of (before, after) ordering rules against the sequence.
        Returns (ok, violations) where violations are the rules broken -- the
        PB&J check: 'apply_jelly before cut', 'close before cut', etc. A plan
        satisfies its constraints or it names exactly which step is out of
        order."""
        repr_, elems, chunk = self._vec(seq_or_name)
        L = len(elems)
        violations = []
        for a, b in constraints:
            if not self.precedes(seq_or_name, a, b, L):
                violations.append((a, b))
        return (not violations), violations


def sequentiality_z(members, vocab, n_shuffle=15, seed=0):
    """DISCOVER whether a set of member-sequences is genuinely ordered -- without
    a magic threshold, by a permutation test against the data's OWN shuffle.

    The honest question: does the real order of these members predict the next
    element better than the SAME members with their order destroyed? A transition
    model (bundle of bind(a, next=b) over adjacent pairs) is built leave-one-out;
    its held-out next-element accuracy is compared to the mean accuracy over
    `n_shuffle` order-scrambled copies. The shuffles are the class's own null
    hypothesis, so nothing external is assumed.

    Returns a z-score: how many null standard deviations the real order's
    predictive accuracy sits above the shuffled baseline. z is a continuous
    measure of order signal, not a brittle flag -- but the standard statistical
    bar z>2 (signal exceeds two sigma of the null) is the natural place to call a
    class sequential, and it is a STATEMENT not a tuned constant. Measured (score-margin statistic): ~+16 for genuinely sequential
    classes, ~0 for order-free bags (real order
    indistinguishable from shuffled), degrading gracefully through partial noise
    (still +5 at 30% scrambled, ~+2 at 50%, ~0 once order is mostly gone)."""
    rng = np.random.default_rng(seed)
    members = [list(m) for m in members if len(m) >= 2]
    if len(members) < 3:
        return 0.0                                   # too few to test honestly
    cands = sorted({str(e) for m in members for e in m})

    # This transition test mints its OWN Gaussian atoms, deterministically per symbol,
    # rather than using whatever atoms the passed `vocab` holds. Measured negative:
    # exact-unbinding (unitary) atoms BREAK this test on tiny alphabets -- on real SOL
    # tick signs (a 2-symbol U/D series with a true +0.20 lag-1 sign autocorrelation)
    # unitary atoms report z=-4.77 (no order) while Gaussian atoms correctly report
    # z=+44. With only two heavily-repeated symbols the permute + exact-unbind
    # transition model goes degenerate; Gaussian atoms' mild spectral noise is what
    # lets the score-margin statistic track real order. Deterministic per-symbol
    # minting also makes the verdict independent of the caller's vocab and call order.
    # `vocab` is now used only for its .dim. A recorded place exact unbind is a net
    # negative, so Gaussian is kept on purpose.
    dim = vocab.dim
    _atoms = {}

    def _atom(sym):
        sym = str(sym)
        if sym not in _atoms:
            # seed each atom from the symbol name so the same symbol always maps to the
            # same vector within this call, independent of insertion order
            r = np.random.default_rng(abs(hash((seed, sym))) % (2**32))
            v = r.standard_normal(dim)
            _atoms[sym] = v / (np.linalg.norm(v) or 1.0)
        return _atoms[sym]

    def tmodel(ms):
        toks = [bind(_atom(a), permute(_atom(b), 1))
                for m in ms for a, b in zip(m, m[1:])]
        return bundle(toks) if toks else np.zeros(dim)

    def next_margin(model, ms):
        # SCORE MARGIN, not argmax accuracy: how much higher the TRUE next
        # element scores than the mean of the others. Argmax accuracy saturates
        # when a class has few distinct elements (random guesses are often right
        # by luck, washing out the signal); the margin is a graded measure that
        # keeps the real-vs-shuffled difference visible even for a 6-step recipe.
        margins = []
        for m in ms:
            for a, b in zip(m, m[1:]):
                probe = permute(unbind(model, _atom(a)), -1)
                sc = {s: cosine(probe, _atom(s)) for s in cands}
                others = np.mean([sc[s] for s in cands if s != str(b)]) if len(cands) > 1 else 0.0
                margins.append(sc[str(b)] - others)
        return float(np.mean(margins)) if margins else 0.0

    def loo(ms):
        return np.mean([next_margin(tmodel(ms[:h] + ms[h + 1:]), [ms[h]])
                        for h in range(len(ms))])

    real = loo(members)
    null = []
    for _ in range(n_shuffle):
        shuf = [list(rng.permutation(m)) for m in members]
        null.append(loo(shuf))
    null = np.array(null)
    return float((real - null.mean()) / (null.std() + 1e-9))

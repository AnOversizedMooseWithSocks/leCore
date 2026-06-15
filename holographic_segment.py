"""Self-discovery of structure: find the units in a stream with no labels, by
listening to where the predictor's certainty breaks down.

Strip the spaces out of text and a person can still read it -- the boundaries are
recoverable from the statistics alone. The signal (Harris 1955; Saffran 1996) is
branching: inside a unit the next symbol is tightly constrained, at the end of a
unit many symbols can follow. So uncertainty about the next symbol peaks at unit
boundaries. This module reads that peak off the holographic substrate and uses it
to segment a stream into its own units -- decomposition the system discovers rather
than is told.

HOW IT IS DONE ON THIS SUBSTRATE (and a kept negative that picked the method):
  * For each exact context (the last K symbols) accumulate a BUNDLE of the symbol
    atoms that followed it. Project that bundle onto every symbol atom -- the
    similarities are the next-symbol readout -- and take its ENTROPY. High entropy
    means many different symbols follow this context: a boundary. Low entropy means
    one symbol dominates: mid-unit. Boundaries are the local peaks above a
    percentile threshold.
  * A KEPT NEGATIVE: doing the readout via RESONANCE (blending over SIMILAR
    contexts, the ZREAD that helps prediction) DESTROYS this signal -- it smears the
    next-symbol distribution across neighbours and the boundary peaks wash out (F1
    fell to ~0.26, barely above the ~0.21 random baseline). Boundary discovery needs
    the EXACT context's successor diversity, not a generalised one. Generalisation
    and segmentation want opposite things; this module uses exact contexts.

WHAT WAS MEASURED (Brown news, spaces removed, recover the word boundaries):
  * The branching-entropy boundaries hit F1 ~0.61 against the true word boundaries,
    versus ~0.21 for a random cut at the same rate -- the words are genuinely
    self-discovered from an unsegmented stream.
  * BETTER STRUCTURE -> BETTER COMPRESSION, again: coding the stream as the
    discovered chunks costs ~2.1 bits/char (unigram over chunks) versus ~4.2
    bits/char over single characters. Finding the right decomposition roughly
    halves the description length -- the same principle as the predictive compressor,
    now reached by discovering the units instead of being given them.

This is the decomposition rung: the system finds its own units, which compose
upward (a discovered chunk can become a symbol for a higher layer) and compress
downward (the units are where the code resets).

Needs: numpy, holographic_ai.
"""
from collections import defaultdict

import numpy as np

from holographic_ai import Vocabulary, bundle, permute


class Segmenter:
    """Discover the units in a symbol stream by branching entropy on the holographic
    substrate, then cut the stream into chunks at the boundaries."""

    def __init__(self, dim=512, order=4, seed=0):
        self.dim = dim
        self.order = order
        # Gaussian atoms on purpose. Branching entropy here reads the SPREAD of a
        # next-symbol bundle as uncertainty; that estimate relies on Gaussian atom
        # statistics. Measured negative: exact-unbind (unitary) atoms degrade the
        # boundary signal (word-boundary F1 falls below the random-cut baseline on the
        # app's spaceless-text probe), the same way they break the small-alphabet
        # transition test in holographic_sequence. Unitary helps clean role-unbinding,
        # not entropy-over-bundle -- so this stays Gaussian.
        self.atoms = Vocabulary(dim, seed, unitary=False)
        self._next = defaultdict(lambda: np.zeros(dim))   # exact context -> next-atom bundle
        self._charset = []
        self._Cmat = None

    def fit(self, stream):
        """Accumulate, per exact context, the bundle of symbols that followed it."""
        stream = list(stream)
        self._charset = sorted(set(stream))
        self._Cmat = np.stack([self.atoms.get(c) for c in self._charset]) if self._charset \
            else np.zeros((0, self.dim))
        K = self.order
        for i in range(1, len(stream)):
            ctx = tuple(stream[max(0, i - K):i])
            self._next[ctx] = self._next[ctx] + self.atoms.get(stream[i])
        return self

    def branching_entropy(self, stream):
        """Per-position branching entropy: the entropy of the next-symbol readout
        (the context's next-atom bundle projected onto every symbol atom). High at
        unit ends, low inside a unit."""
        stream = list(stream)
        K = self.order
        out = []
        for i in range(len(stream)):
            ctx = tuple(stream[max(0, i - K + 1):i + 1])
            v = self._next.get(ctx)
            if v is None or np.linalg.norm(v) == 0:
                out.append(0.0)
                continue
            sims = np.clip(self._Cmat @ (v / np.linalg.norm(v)), 0, None)
            if sims.sum() == 0:
                out.append(0.0)
                continue
            p = sims / sims.sum()
            p = p[p > 0]
            out.append(float(-(p * np.log2(p)).sum()))
        return np.array(out)

    def boundaries(self, stream, percentile=70):
        """Discovered boundary positions: local peaks of branching entropy above the
        given percentile. A boundary at index i means a unit ends after stream[i]."""
        H = self.branching_entropy(stream)
        if len(H) < 3:
            return set()
        thr = np.percentile(H, percentile)
        return set(i for i in range(1, len(H) - 1)
                   if H[i] >= thr and H[i] >= H[i - 1] and H[i] >= H[i + 1])

    def segment(self, stream, percentile=70):
        """Cut the stream into discovered chunks at the boundaries."""
        stream = list(stream)
        bounds = self.boundaries(stream, percentile)
        chunks, cur = [], []
        for i, s in enumerate(stream):
            cur.append(s)
            if i in bounds:
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        return chunks


def boundary_f1(predicted, truth):
    """Precision / recall / F1 of discovered boundaries against known ones."""
    tp = len(predicted & truth)
    p = tp / max(1, len(predicted))
    r = tp / max(1, len(truth))
    f = 2 * p * r / max(1e-9, p + r)
    return {"precision": p, "recall": r, "f1": f}


def chunk_compression(stream, chunks):
    """Bits per symbol of a unigram code over the discovered chunks vs over single
    symbols -- the MDL payoff of finding the right decomposition. Returns
    (chunk_bits_per_symbol, symbol_bits_per_symbol)."""
    from collections import Counter
    stream = list(stream)
    wc = Counter(tuple(c) for c in chunks)
    tot = sum(wc.values())
    chunk_bits = sum(n * -np.log2(n / tot) for n in wc.values())
    cc = Counter(stream)
    ctot = len(stream)
    sym_bits = sum(n * -np.log2(n / ctot) for n in cc.values())
    return chunk_bits / len(stream), sym_bits / len(stream)

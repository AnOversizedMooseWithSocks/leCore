"""form_prior.py -- PREDICT A WORD'S MEANING FROM ITS FORM (cp113).

Built from cp111/cp112. Given a vocabulary and its vectors -- distributional vectors, or a
model's own embedding matrix -- this fits a map from a word's SHAPE to its meaning, so a
word the table has never seen (or has seen too rarely to have estimated well) can be given
a vector for free.

WHAT IT IS FOR, AND WHAT IT IS NOT FOR. Measured on 5,064,037 tokens:

    it identifies WHICH word          rank@1 0.255-0.273, median rank 19-23 of ~840
                                      against 0.001 chance and 0.003 for a
                                      shuffled-letter null that keeps every letter
                                      and destroys only their order

    it does NOT reproduce WHERE       scale-corrected variance explained -0.110.
    the vector sits                   A form-only prediction is WORSE than predicting
                                      the centroid in squared error.

Both of those are true at once, and the dissociation is the whole design constraint: this
is a DISCRIMINATIVE PRIOR, useful for retrieval, for tying rare words to their relatives,
and as an INITIALISATION for embeddings that are otherwise estimated from too few
occurrences. It is NOT a compression scheme and it cannot replace an embedding table.
`evaluate()` reports both numbers every time so the second one cannot be quietly dropped.

FEATURES. Two kinds, doing two different jobs (cp112 E5):

    affix parse only    387 features   mean cos 0.464 (BEST)   rank@1 0.078 (WORST)
    char n-grams      7,392 features   mean cos 0.367          rank@1 0.273 (BEST)
    hybrid            7,779 features   mean cos 0.363          median rank 18 (BEST)

A morphological parse points in the right semantic direction but cannot separate words
sharing a parse; overlapping substrings supply the identity. Morphology gives the field,
substrings give the address -- which is the same conclusion MorphBPE (Asgari et al., 2025)
reached from the other direction, and why the default here is the hybrid.

Usage:
    from tools.form_prior import FormPrior
    fp = FormPrior().fit(vocab, vectors)
    v  = fp.predict_one("unseenword")
    print(fp.evaluate(vocab, vectors))

    python3 tools/form_prior.py --selfcheck
"""
import sys
import collections

import numpy as np

PREFIXES = ["un", "re", "in", "im", "dis", "non", "pre", "pro", "con", "com", "de", "ex",
            "sub", "inter", "trans", "super", "anti", "multi", "over", "under", "mis",
            "fore", "semi", "auto", "co"]
SUFFIXES = ["ation", "ition", "ment", "ness", "ity", "ties", "ive", "ous", "ance",
            "ence", "able", "ible", "less", "ful", "ize", "ise", "ing", "ed", "er",
            "or", "ly", "al", "ic", "ism", "ist", "es", "ry", "y", "s"]


def affix_parse(word):
    """Crude morphological parse: strip one known prefix and one known suffix, keep the stem.

    Deliberately crude. cp112 measured that this alone gives the best average direction and
    the worst discrimination, which is exactly what a coarse parse should do.
    """
    out = []
    s = word
    for p in sorted(PREFIXES, key=len, reverse=True):
        if s.startswith(p) and len(s) - len(p) >= 3:
            out.append("PRE:" + p)
            s = s[len(p):]
            break
    for q in sorted(SUFFIXES, key=len, reverse=True):
        if s.endswith(q) and len(s) - len(q) >= 3:
            out.append("SUF:" + q)
            s = s[:-len(q)]
            break
    out.append("STEM:" + s)
    return out


def char_ngrams(word, sizes=(3, 4, 5)):
    t = "<" + word + ">"
    return [t[i:i + n] for n in sizes for i in range(len(t) - n + 1)]


def hybrid_features(word):
    return char_ngrams(word) + affix_parse(word)


class FormPrior(object):
    """Ridge map from form features to meaning vectors."""

    def __init__(self, featurizer=hybrid_features, min_count=3, ridge=1e-2):
        self.featurizer = featurizer
        self.min_count = int(min_count)
        self.ridge = float(ridge)
        self.features = None
        self.index = None
        self.B = None

    def _matrix(self, words):
        M = np.zeros((len(words), len(self.features)), dtype=np.float32)
        for r, w in enumerate(words):
            for g in self.featurizer(w):
                j = self.index.get(g)
                if j is not None:
                    M[r, j] = 1.0
        return M

    def fit(self, words, vectors):
        counts = collections.Counter()
        for w in words:
            counts.update(set(self.featurizer(w)))
        self.features = [g for g, c in counts.items() if c >= self.min_count]
        if not self.features:
            raise ValueError("no features survived min_count=%d" % self.min_count)
        self.index = {g: j for j, g in enumerate(self.features)}
        X = self._matrix(words)
        Y = np.asarray(vectors, np.float32)
        A = X.T @ X
        lam = self.ridge * float(np.trace(A)) / len(self.features)
        self.B = np.linalg.solve(A + lam * np.eye(len(self.features), dtype=np.float32), X.T @ Y)
        return self

    def predict(self, words, normalize=True):
        P = self._matrix(words) @ self.B
        if normalize:
            n = np.linalg.norm(P, axis=1, keepdims=True)
            P = P / (n + 1e-12)
        return P

    def predict_one(self, word, normalize=True):
        return self.predict([word], normalize=normalize)[0]

    def predict_via_donors(self, words, donor_words, donor_vectors, k=64, temperature=1.0):
        """THE METHOD THAT FIXES THE 'WHERE' PROBLEM (cp114).

        `predict` returns the raw form vector, which identifies WHICH word well and
        reproduces WHERE the vector sits badly (variance explained -0.112). FOCUS
        (Dobler & de Melo, EMNLP 2023) does not use the form vector as the embedding at
        all: it uses similarity to SELECT donors and then sets the value as a sparse
        convex combination of REAL in-vocabulary vectors, which already lie on the
        manifold. Measured here on 845 held-out words against 4,783 donors:

            Hewitt mean-of-all baseline        variance explained  0.000   cos 0.458
            raw form vector as embedding                          -0.112   cos 0.360
            donors, k=8                                           +0.029   cos 0.481
            donors, k=64 (best)                                   +0.063   cos 0.509
            donors, k=256                                         +0.056   cos 0.503

        and the selection is doing the work, not the averaging:

            random donors, k=64                                   -0.015   cos 0.445
            shuffled-letter selection, k=64                       -0.084   cos 0.381

        So k is bounded (best near 64, declining after), form-selected donors beat random
        donors at every k tested, and destroying letter order collapses the result below
        random. Use this, not `predict`, whenever you need an actual embedding value.
        """
        D = np.asarray(donor_vectors, np.float64)
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        sims = self.predict(words, normalize=True) @ Dn.T
        k = int(min(k, D.shape[0]))
        idx = np.argsort(sims, axis=1)[:, -k:]
        w = np.take_along_axis(sims, idx, axis=1)
        if temperature != 1.0:
            w = w / float(temperature)
        w = np.maximum(w, 0.0) + 1e-9
        w = w / w.sum(axis=1, keepdims=True)
        return np.einsum("ij,ijk->ik", w, D[idx])

    def evaluate(self, words, vectors, train_mean=None):
        """Report BOTH halves of the dissociation. Never report one without the other."""
        Yh_raw = self._matrix(words) @ self.B
        Y = np.asarray(vectors, np.float64)
        mu = np.asarray(train_mean if train_mean is not None else Y.mean(0), np.float64)
        alpha = float(np.sum(Yh_raw * Y) / max(np.sum(Yh_raw * Yh_raw), 1e-12))
        r2 = 1.0 - float(np.sum((alpha * Yh_raw - Y) ** 2) / max(np.sum((Y - mu) ** 2), 1e-12))
        keep = np.linalg.norm(Yh_raw, axis=1) > 1e-9
        P = Yh_raw[keep] / (np.linalg.norm(Yh_raw[keep], axis=1, keepdims=True) + 1e-12)
        T = Y[keep] / (np.linalg.norm(Y[keep], axis=1, keepdims=True) + 1e-12)
        sims = P @ T.T
        d = np.diag(sims)
        ranks = ((sims > d[:, None]).sum(1) + 1
                 + 0.5 * ((np.abs(sims - d[:, None]) < 1e-12).sum(1) - 1))
        n = max(len(ranks), 1)
        return {
            "n": int(n),
            "rank@1": float((ranks <= 1).mean()),
            "rank@10": float((ranks <= 10).mean()),
            "median_rank": float(np.median(ranks)),
            "chance_rank@1": 1.0 / n,
            "lift_over_chance": float((ranks <= 1).mean() * n),
            "mean_cosine": float(np.mean(d)),
            "variance_explained": r2,
            "shrinkage_alpha": alpha,
            "reading": ("identifies WHICH word at %.0fx chance; does NOT reproduce WHERE "
                        "the vector sits (variance explained %.3f). Use as a discriminative "
                        "prior or initialisation, not as a table replacement."
                        % ((ranks <= 1).mean() * n, r2)),
        }


def _selfcheck():
    """Reproduce the cp111/cp112 shape on a synthetic vocabulary with known morphology."""
    rng = np.random.default_rng(0)
    roots = ["port", "struct", "form", "duc", "spect", "scrib", "ject", "tend", "vert", "cred"]
    pres = ["", "re", "in", "ex", "trans", "pre", "de", "con"]
    sufs = ["", "ion", "ive", "ing", "ed", "or", "ure"]
    dim = 100
    rv = {r: rng.standard_normal(dim) for r in roots}
    pv = {p: rng.standard_normal(dim) * 0.6 for p in pres}
    sv = {s: rng.standard_normal(dim) * 0.6 for s in sufs}
    words, vecs = [], []
    for r in roots:
        for p in pres:
            for s in sufs:
                w = p + r + s
                if len(w) < 4:
                    continue
                words.append(w)
                v = rv[r] + pv[p] + sv[s] + 0.35 * rng.standard_normal(dim)
                vecs.append(v / np.linalg.norm(v))
    V = np.stack(vecs)
    o = rng.permutation(len(words))
    k = int(0.85 * len(o))
    tr, te = o[:k], o[k:]
    fp = FormPrior().fit([words[i] for i in tr], V[tr])
    res = fp.evaluate([words[i] for i in te], V[te], train_mean=V[tr].mean(0))
    print("form_prior selfcheck (synthetic vocabulary, morphology IS the generator)")
    print("  words %d, held out %d, features %d" % (len(words), len(te), len(fp.features)))
    for k_ in ("rank@1", "rank@10", "median_rank", "mean_cosine", "variance_explained"):
        print("  %-20s %s" % (k_, round(res[k_], 4)))
    print("  lift over chance     %.0fx" % res["lift_over_chance"])
    ok = res["rank@1"] > 0.5
    print("SELFCHECK %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck() if "--selfcheck" in sys.argv else (print(__doc__) or 0))

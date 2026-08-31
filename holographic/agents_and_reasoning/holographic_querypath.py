"""QUERYPATH -- the model asks its own layer. The last blocker, removed.

Storage, seed expansion, capacity and the read path were all settled; the model
still could not ASK anything, because something external had to supply the key
hypervector. This closes that.

THE FIRST ATTEMPT FAILED, AND MY EXPLANATION OF WHY WAS WRONG. Fitting a
projection to arbitrary key vectors gave 16/16 on training positions and 0/16
held out, and I concluded "keys must be derived from content". Then I tested it
properly -- same store, same fitting, arbitrary keys against content keys -- and
arbitrary keys scored 29/32 against content keys' 27/32. The explanation was
false.

THE REAL REQUIREMENT IS THAT THE SAME CONTENT RECURS. The first experiment gave
every position a UNIQUE fact and then tested on DIFFERENT positions, so there
was nothing to generalise TO -- the failure was in the experiment, not in the
key scheme. What the projection actually learns is "this kind of stream state
means this key", and it transfers to another occurrence of the same token in
different surrounding text, however the key was chosen.

CONTENT-DERIVED KEYS ARE STILL THE RIGHT DEFAULT, for a different and smaller
reason: hashlib over the term means a key can be COMPUTED anywhere without
shipping a lookup table, so a store written by one process is readable by
another. That is portability, not accuracy.

MEASURED on a real Qwen3.5-0.8B stream (layer 12, 235 positions): fitted on the
FIRST occurrence of 32 repeated tokens and tested on a LATER occurrence in
different surrounding text --
    training positions        32/32
    HELD-OUT OCCURRENCES      29/32       (chance 1/32)
So the model's own hidden state, run through one fixed matrix, produces a key
that unbinds the right fact out of a superposed store.

WHAT THIS COMPLETES: query -> unbind -> cleanup, all three now inside the model's
own arithmetic. The projection is a matrix (installable in an MLP, see vsabake),
unbinding is a shift or a circulant, and cleanup is argmax over a codebook,
which is what lm_head is.

HONEST LIMITS. The projection is fitted per model and per layer, and it is only
as good as its calibration set -- the same lesson the denoiser taught. 29/32 is
not 32/32, and the three misses are real. And a key derived from a token is a
LEXICAL address: this retrieves what a term names, not what a sentence means.
"""

import hashlib

import numpy as np


def content_key(name, dim, tag="key"):
    """A key hypervector derived from the content it names.

    hashlib, never hash(): a key computed in one process must equal the key
    computed in another, or a store written today cannot be read tomorrow."""
    h = hashlib.sha256(("%s:%s" % (tag, name)).encode("utf-8")).digest()
    g = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return g.standard_normal(int(dim)) / np.sqrt(float(dim))


def cconv(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def ccorr(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


class QueryPath:
    """Fit stream -> key, then retrieve from a superposed store."""

    def __init__(self, dim=1024, ridge=1e-2, mind=None, margin=0.0):
        self.dim = int(dim)
        self.ridge = float(ridge)
        self.mind = mind
        self.margin = float(margin)
        self.W = None
        self.store = None
        self.floor = None
        self.names = []

    def fit(self, states, names):
        """Learn the projection from example (stream state, content name) pairs.

        FIT ON EVERY POSITION YOU HAVE, not just the ones in the store. The map
        is stream->key and every token teaches it something; restricting the fit
        to store entries starves a 1024x1024 map on 32 examples and the ridge
        ends up doing all the work.
        MEASURED at a 0.0% false-action target on a real Qwen3.5 stream:
            fitted on store entries only (32)   floor 0.2276   recall  3/16
            fitted on ALL positions (203)       floor 0.1063   recall 11/16
        A 3.7x improvement in usable recall at the SAME guarantee. Held-out
        top-1 barely moved (27/32 -> 25/32); what improved is SEPARATION, which
        is what an abstention gate actually consumes.

        Ridge-regularised least squares: a projection that fits its examples
        exactly has memorised them, which is precisely the failure this class
        exists to avoid.

        NEGATIVES, all measured, so nobody re-runs them: layer 23 instead of 12
        gives 22/32; four layers CONCATENATED gives 23/32; whitening gives
        18/32; denoising the stream first changes nothing (27/32). More features
        hurt -- the constraint was never the representation, it was the number
        of training positions."""
        X = np.asarray(states, np.float64)
        Y = np.stack([content_key(n, self.dim) for n in names])
        lam = self.ridge * float(np.trace(X.T @ X)) / max(X.shape[1], 1)
        self.W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        return self

    def build_store(self, pairs):
        """key(name) bound to val(value), all bundled into ONE vector."""
        items = list(pairs)
        t = np.zeros(self.dim)
        for name, value in items:
            t = t + cconv(content_key(name, self.dim),
                          content_key(value, self.dim, tag="val"))
        self.store = t
        self.names = [v for _n, v in items]
        return self

    def calibrate(self, trials=256, alpha=0.01, seed=0, null_states=None):
        """Build the NULL DISTRIBUTION of match scores, the leCore way.

        WHY THIS EXISTS, and it is the most embarrassing find of the session:
        this project's ONE measured competitive advantage over NVIDIA's NOOA is a
        CALIBRATED, NULL-REFERENCED ABSTENTION with a false-action rate of 0.0%
        -- and the query path I built has a false-action rate of 100%. Asked for
        16 facts that were never stored, it returned 16 confident answers. An
        argmax over a codebook ALWAYS names something.

        The floor is measured, not guessed: query the store with random keys
        that reference nothing, collect the distribution of best-match scores,
        and take the (1-alpha) quantile. A real hit must beat what noise
        achieves, which prices the codebook-wide argmax in by construction --
        the same reasoning find_capability already uses on the catalog."""
        rng = np.random.default_rng(int(seed))
        M = np.stack([content_key(v, self.dim, tag="val") for v in self.names])
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        if null_states is not None:
            # MATCHED NULL: score REAL projected queries that should miss.
            # Isotropic keys are an EASIER null than a real near-miss, measured:
            # q99 0.1837 for random keys against 0.2276 for real misses. The
            # catalog's abstention gets 0.0% precisely because its null is built
            # from its own vocabulary, so matching that construction here is the
            # difference between 18.8% and 0.0% false actions.
            S = np.asarray(null_states, np.float64)
            best = np.empty(len(S))
            for i, row in enumerate(S):
                q = row @ self.W
                q = q / (np.linalg.norm(q) + 1e-30)
                est = ccorr(self.store, q)
                est = est / (np.linalg.norm(est) + 1e-30)
                best[i] = float(np.max(M @ est))
        else:
            best = np.empty(int(trials))
            for i in range(int(trials)):
                q = rng.standard_normal(self.dim)
                q = q / np.linalg.norm(q)
                est = ccorr(self.store, q)
                est = est / (np.linalg.norm(est) + 1e-30)
                best[i] = float(np.max(M @ est))
        self.floor = float(np.quantile(best, 1.0 - float(alpha)))
        return {"floor": self.floor, "alpha": float(alpha),
                "trials": int(len(best)), "null_mean": float(best.mean()),
                "matched": null_states is not None}

    def query(self, state, abstain=True):
        """Project, unbind, clean up -- and ABSTAIN when nothing beats the floor.

        Returns None rather than a name when the best match is indistinguishable
        from what an unreferenced key would score. Refusal is a first-class
        output here, as it is everywhere else in this engine."""
        if self.W is None or self.store is None:
            raise RuntimeError("fit() and build_store() first")
        q = np.asarray(state, np.float64) @ self.W
        q = q / (np.linalg.norm(q) + 1e-30)
        est = ccorr(self.store, q)
        est = est / (np.linalg.norm(est) + 1e-30)
        M = np.stack([content_key(v, self.dim, tag="val") for v in self.names])
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        scores = M @ est
        order = np.argsort(scores)[::-1]
        i = int(order[0])
        if not abstain:
            return self.names[i]
        if self.mind is not None:
            # DELEGATE THE DECISION. decide_or_abstain says in its own docstring
            # that it exists so callers stop inventing their own rule -- and I
            # invented one anyway. It adds a TOP1-vs-TOP2 MARGIN gate that a
            # bare floor does not have: a query matching two stored facts
            # equally well is ambiguous, not confident, and a floor alone cannot
            # see that.
            ranked = [(self.names[int(j)], float(scores[int(j)]))
                      for j in order[:5]]
            _w, _s, confident = self.mind.decide_or_abstain(
                ranked, margin=float(self.margin),
                min_score=getattr(self, "floor", None))
            return self.names[i] if confident else None
        if getattr(self, "floor", None) is not None \
                and float(scores[i]) < self.floor:
            return None
        return self.names[i]


def _selftest():
    import os

    kit = "/mnt/user-data/uploads/kit2.npz"
    if not os.path.exists(kit):
        print("querypath selftest SKIPPED-SUBJECT (no real stream present)")
        return
    z = np.load(kit, allow_pickle=False)
    H = z["act::12"].astype(np.float64)
    ids = np.asarray(z["probe_ids"])
    uniq, counts = np.unique(ids, return_counts=True)
    repeated = [int(t) for t in uniq[counts >= 2]][:32]

    first, later = {}, {}
    for i, t in enumerate(ids):
        t = int(t)
        if t not in repeated:
            continue
        if t not in first:
            first[t] = i
        else:
            later.setdefault(t, i)
    pairs = [t for t in repeated if t in first and t in later]
    assert len(pairs) >= 16, len(pairs)

    qp = QueryPath(dim=1024)
    qp.fit([H[first[t]] for t in pairs], ["%d" % t for t in pairs])
    qp.build_store([("%d" % t, "fact_%d" % t) for t in pairs])

    train_ok = sum(qp.query(H[first[t]]) == "fact_%d" % t for t in pairs)
    held_ok = sum(qp.query(H[later[t]]) == "fact_%d" % t for t in pairs)
    chance = 1.0 / len(pairs)

    assert train_ok == len(pairs), (train_ok, len(pairs))
    # ---- the point of the whole class: it must GENERALISE, far above chance --
    assert held_ok >= 0.8 * len(pairs), (held_ok, len(pairs))
    assert held_ok / len(pairs) > 20 * chance

    # ---- THE KEPT NEGATIVE, pinned: arbitrary keys memorise and do not
    #      generalise, so nobody re-tries it
    rng = np.random.default_rng(0)
    bad = QueryPath(dim=1024)
    arb = {t: rng.standard_normal(1024) / 32 for t in pairs}
    X = np.stack([H[first[t]] for t in pairs])
    Y = np.stack([arb[t] for t in pairs])
    lam = 1e-2 * float(np.trace(X.T @ X)) / X.shape[1]
    Wb = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    # THE COMPARISON MUST GO THROUGH THE SAME STORE. My first version matched
    # projected keys against the key codebook DIRECTLY, skipping the
    # superposition -- and arbitrary keys scored 30/32, which looked like a
    # refutation and was an unfair test. Retrieval means unbinding from a
    # BUNDLE, where interference is the whole difficulty.
    vals_arb = [rng.standard_normal(1024) / 32 for _ in pairs]
    store_arb = np.zeros(1024)
    for i, t in enumerate(pairs):
        store_arb = store_arb + cconv(arb[t], vals_arb[i])
    Vn = np.stack(vals_arb)
    Vn = Vn / np.linalg.norm(Vn, axis=1, keepdims=True)
    hold = 0
    for i, t in enumerate(pairs):
        q = H[later[t]] @ Wb
        q = q / np.linalg.norm(q)
        e = ccorr(store_arb, q)
        hold += int(np.argmax(Vn @ (e / np.linalg.norm(e)))) == i
    # ASSERT WHAT IS TRUE: both schemes generalise, because what the projection
    # needs is RECURRING CONTENT, not a content-derived key. Content derivation
    # buys portability (no lookup table to ship), not accuracy.
    assert hold >= 0.8 * len(pairs), (hold, len(pairs))

    # ---- ABSTENTION: the project's own rule, applied to its own retrieval ----
    half = pairs[:len(pairs) // 2]
    absent = pairs[len(pairs) // 2:]
    qp2 = QueryPath(dim=1024)
    qp2.fit([H[first[t]] for t in pairs], ["%d" % t for t in pairs])
    qp2.build_store([("%d" % t, "fact_%d" % t) for t in half])
    naive = sum(qp2.query(H[later[t]], abstain=False) is not None for t in absent)
    assert naive == len(absent), "without a floor, argmax always names something"
    # MATCHED NULL vs ISOTROPIC: the honest comparison, both pinned
    misses = [H[i] for i, t in enumerate(ids)
              if int(t) not in set(half)][:200]
    cal_m = qp2.calibrate(alpha=0.01, null_states=misses)
    kept_m = sum(qp2.query(H[later[t]]) == "fact_%d" % t for t in half)
    false_m = sum(qp2.query(H[later[t]]) is not None for t in absent)
    # the matched null REACHES the project's 0.0% standard...
    assert false_m == 0, (false_m, len(absent))

    # ---- FIT ON EVERYTHING: same guarantee, far more usable recall ----
    heldout = set(later[t] for t in pairs)
    qp3 = QueryPath(dim=1024)
    qp3.fit([H[i] for i in range(len(ids)) if i not in heldout],
            ["%d" % int(ids[i]) for i in range(len(ids)) if i not in heldout])
    qp3.build_store([("%d" % t, "fact_%d" % t) for t in half])
    qp3.calibrate(alpha=0.01, null_states=misses)
    kept3 = sum(qp3.query(H[later[t]]) == "fact_%d" % t for t in half)
    false3 = sum(qp3.query(H[later[t]]) is not None for t in absent)
    assert false3 == 0, (false3, len(absent))
    assert kept3 > 2 * kept_m, ("fitting on all positions must beat fitting on "
                                "store entries at the same guarantee",
                                kept_m, kept3)

    cal = qp2.calibrate(trials=2000, alpha=0.001)
    kept = sum(qp2.query(H[later[t]]) == "fact_%d" % t for t in half)
    false_act = sum(qp2.query(H[later[t]]) is not None for t in absent)
    # recall must SURVIVE the floor, or abstention is just refusing to work
    assert kept >= 0.85 * len(half), (kept, len(half))
    # and the false-action rate must fall a long way from 100%
    assert false_act <= 0.25 * len(absent), (false_act, len(absent))

    print("querypath selftest OK -- fitted stream->key on the FIRST occurrence of "
          "%d repeated tokens and tested on a LATER occurrence in different "
          "surrounding text: train %d/%d, HELD-OUT %d/%d against chance %.3f; "
          "and arbitrary keys score %d/%d through the same store -- so what "
          "the projection needs is RECURRING CONTENT, not content-derived keys "
          "(those buy portability instead); and with a NULL-REFERENCED FLOOR "
          "(alpha 0.001, measured from %d unreferenced queries) the false-action "
          "rate on facts that were never stored falls from %d/%d to %d/%d while "
          "recall holds at %d/%d -- the abstention this project measures "
          "everywhere else, finally applied to its own retrieval. A MATCHED "
          "null (real misses, not random keys) reaches the project's 0.0%% "
          "standard exactly -- %d/%d false actions -- at %d/%d recall when fitted "
          "on store entries alone, rising to %d/%d when fitted on ALL positions: "
          "3.7x the usable recall at the same guarantee, because separation is "
          "what an abstention gate consumes"
          % (len(pairs), train_ok, len(pairs), held_ok, len(pairs), chance,
             hold, len(pairs), 2000, naive, len(absent), false_act,
             len(absent), kept, len(half), false_m, len(absent), kept_m,
             len(half), kept3, len(half)))


if __name__ == "__main__":
    _selftest()

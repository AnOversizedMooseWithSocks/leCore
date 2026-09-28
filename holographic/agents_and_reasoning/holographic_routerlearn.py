"""holographic_routerlearn.py -- the capability ROUTER learns on the shared contrastive rule (CLM backlog E4.1).

WHY THIS EXISTS
---------------
The catalog router (holographic_catalog.Catalog._score_all) ranks ~4,000 capability cards by content-word overlap.
It never learns: a wording that routed wrong routes wrong forever. The CLM panel (docs/research/evidence/
clm_panel_20260926/w3-reflex/) measured it on held-out aliases -- an alias removed from the index is a question the
router has never seen, and the card that carried it is the truth, so the labels are FREE and NOISELESS:

    find_capability top-1 0.402, top-5 0.638 (3,954 held-out aliases, 3,952 cards); route_tiered answers 37.1% at
    0.604 accuracy. 44.8% of queries have an exact score TIE at the top, and the truth sits inside that tie group
    51% of the time -- so the tie-break alone (today: alphabetical by card name) is worth up to 11 points.

This module is the router's learner, on the ONE rule every door shares (holographic_protostore.ProtoStore, InfoNCE on
per-option prototypes): a ProtoStore("route") whose options are cards, in a hashed feature space of the card's own
words, trained on alias -> card pairs and on every reported route outcome. It does NOT replace the lexical scorer --
it RERANKS the lexical top-K:

    fused(card) = lexical(card) + lam * cos(q, P_card) + mu * log(1 + verdicts(card))

    lexical   Catalog._score_all's score (word overlap + 0.5 per name hit + the exact-alias bonus), unchanged
    cos       the card's learned prototype against the query (a card never labelled scores against its TEXT seed,
              so being in the store is never itself an advantage)
    verdicts  the store's per-option count of labelled verdicts: how often this card WAS the answer. A usage prior,
              learned from the same labels -- MEASURED to carry most of the gain (tools/bench_router.py).

lam, mu and K are chosen on a hash-carved VAL split of the training aliases and reported on the panel's held-out
test aliases (docs/research/evidence/bench_router.json).

MEASURED (PYTHONHASHSEED=0 python tools/bench_router.py --data <dir>; 3,995 cards, pretrained on 4,422 train aliases with
leave-one-out rivals, val and test aliases both removed from the index -- so the lexical baseline here is 0.3968, a
little under the panel's 0.4016, whose index still held the val aliases):
    arm (test, 3,954 held-out aliases)        top-1    top-5    vs lexical, paired bootstrap 95% CI
    lexical (the shipped scorer)              0.3968   0.6353   --
    LEARNED lam 2.0 mu 0.45 (val choice)      0.4120   0.6646   +1.52 [+0.45, +2.50]  -> beats the panel's 0.402
    prior only (lam 0)                        0.4082   0.6459   +1.14 [+0.20, +2.05]
    prototype only (mu 0)       KEPT NEG      0.3561   0.6254   -4.07 [-5.18, -3.01]
    untrained store (seeds)     KEPT NEG      0.3483   0.6163   -4.86
So the gain is real but SMALL, and most of it is the learned usage prior; the prototype term helps only on top of it
(alone it reorders ties toward cards whose text resembles the query, which on this data are often the auto-generated
method/module twins of the curated card that carries the alias). lam sits at the edge of the grid searched (2.0): a
larger lam was not tried. The feature space, measured before choosing
(scratch arms, val split, 3,982 cards):
    * char 3..5-grams of the whole text, signed-hashed, raw counts (the panel's default space): prototype-only
      top-1 0.15, and fusing it HURT at every lam (0.343 vs lexical 0.354) -- common n-grams (' th', 'ing') swamp it;
    * the systemone gaussian-atom encoder (hashed_ngram_encode): same space, but its per-n-gram atom cache is
      16 KB per distinct n-gram at d=2048 -- the 4,000 card texts hold ~10^5 of them, i.e. >1.5 GB: not viable here;
    * CHOSEN: content tokens (the catalog's own _tokens: stop words dropped) as whole-word features plus the
      char 3..5-grams INSIDE each token at half weight (morphology: render/rendering), IDF-weighted over the card
      corpus, signed-hashed (blake2b, deterministic) into `dim` buckets. Cheap (a token's features are cached),
      bounded memory, and it is the catalog's own notion of a word.
Kept negatives (measured, recorded so nobody re-runs them): prototypes seeded from name+does+aliases as separate
examples (0.08-0.15 prototype-only); alias-to-alias nearest neighbour (0.126: a card's aliases share few words);
hard negatives from LEAVE-ONE-OUT lexical errors pushed with ProtoStore.negative (train errors 2,423 -> 2,401 over two
passes, val 0.371 -> 0.365: the prototype term can only reorder inside a ~0.5-point lexical band, so pushing the
wrong card harder does not move the argmax).

numpy + stdlib only, deterministic (hashlib, stable sorts, catalog order breaks ties), and every learned array
round-trips through state() / from_state() so it lives in the memory partition.
"""
import hashlib
import math

import numpy as np

from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
from holographic.caching_and_storage.holographic_catalog import _alias_tokens, _tokens

# ---------------------------------------------------------------------------------------------------------------------
# The measured operating point (tools/bench_router.py -> docs/research/evidence/bench_router.json, CHOSEN ON VAL, 2026-09-26,
# 3,995 cards; train 4,422 / val 1,157 / test 3,954 aliases). Change them only with a new bench run.
ROUTER_DIM = 1024            # hashed feature buckets. test top-1 0.4120 at 1024 vs 0.4107 at 2048: half the memory
ROUTER_K = 30                # the lexical candidates the learner may reorder (the truth is in the top 30 ~80% of the time)
ROUTER_LAM = 2.0             # weight of the prototype cosine in the fused score  } val grid 5 x 8 = 40 configurations;
ROUTER_MU = 0.45             # weight of log(1 + verdicts): the usage prior       } best val top-1 0.3803 (lexical 0.3535)
NGRAM_WEIGHT = 0.5           # a token's char n-grams weigh half its whole-word feature
# The learned mode's ANSWER gate, re-derived with the E0.4 protocol (bench_contrastive.gate_protocol: calibrated on the val
# aliases + CLINC150 oos_test half A, precision under the 4,500 : 1,000 prior; reported on the test aliases + half B).
# Signal chosen by val AURC among six (lexical z / margin / z+margin, fused z / margin / z+margin): fused z + margin
# (val AURC 0.501 vs the shipped lexical z's 0.559; test 0.429 vs 0.474). Threshold = the P 0.70 calibration: on test it
# answers 12.5% of held-out aliases at 0.822 accuracy (the shipped z tiers: 36.9% at 0.596) with 0.0% of oos answered
# (0.6%), the same refusals (in-scope 30.8%, oos 92.0%), and a menu that holds the truth 73.2% (66.0%).
ROUTER_GATE_SIGNAL = "z+margin"
ROUTER_GATE_ANSWER = 1.6101


class RouteFeatures:
    """The router's feature space: IDF-weighted hashed features of a text's content tokens.

    Per token w: one whole-word feature 'w:<w>' (weight 1) and its char 3..5-grams inside ' w ' ('g:<gram>', weight
    NGRAM_WEIGHT). Each feature hashes (blake2b, 8 bytes) to a bucket in [0, dim) and a sign; the value is
    count * idf(feature). IDF is computed over the catalog's CARD documents (name + does + aliases), smoothed
    log((N+1)/(df+1)) + 1, so a word every card uses weighs little and a rare one a lot -- the lexical scorer counts
    them all as 1, which is exactly the information this adds. Unit-normalised output (a zero vector stays zero)."""

    def __init__(self, catalog, dim=ROUTER_DIM):
        self.dim = int(dim)
        self._tok = {}                      # token -> [(feature key, weight)] (tokens repeat: cache)
        self._hix = {}                      # feature key -> (bucket, sign)
        docs = [self.raw_features(self.card_text(c, with_aliases=True)) for c in catalog.all()]
        df = {}
        for d in docs:
            for k in d:
                df[k] = df.get(k, 0) + 1
        n = len(docs)
        self.n_docs = n
        self.idf = {k: math.log((n + 1.0) / (v + 1.0)) + 1.0 for k, v in df.items()}
        self.idf_unseen = math.log(n + 1.0) + 1.0   # a feature no card has: maximally specific
        # a digest of the IDF table: a learned store is only meaningful in the feature space it was trained in
        h = hashlib.sha256(("%d|%d" % (self.dim, n)).encode())
        for k in sorted(self.idf):
            h.update(("%s=%.6f;" % (k, self.idf[k])).encode())
        self.digest = h.hexdigest()[:16]

    @staticmethod
    def card_text(cap, with_aliases=False):
        """The text a card is known by: name + does (+ its aliases -- used for the IDF corpus only; a card's SEED
        prototype is name + does, and aliases arrive as labelled verdicts, like any other question)."""
        t = cap.name + " " + (cap.does or "")
        if with_aliases:
            t += " " + " ".join(cap.aliases or ())
        return t

    def _token_features(self, w):
        f = self._tok.get(w)
        if f is None:
            f = [("w:" + w, 1.0)]
            t = " " + w + " "
            for n in (3, 4, 5):
                for i in range(len(t) - n + 1):
                    f.append(("g:" + t[i:i + n], NGRAM_WEIGHT))
            if len(self._tok) < 200000:      # bounded: a long-running service sees unbounded text
                self._tok[w] = f
        return f

    def raw_features(self, text):
        """{feature key: weighted count} for a text (before IDF and hashing)."""
        out = {}
        for w in _tokens(text):
            for k, wt in self._token_features(w):
                out[k] = out.get(k, 0.0) + wt
        return out

    def _bucket(self, key):
        h = self._hix.get(key)
        if h is None:
            x = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")
            h = (x % self.dim, 1.0 if (x >> 63) else -1.0)
            if len(self._hix) < 500000:      # bounded, like the token cache
                self._hix[key] = h
        return h

    def encode(self, text):
        """Unit feature vector of a text (float64, length dim)."""
        v = np.zeros(self.dim)
        for k, c in self.raw_features(text).items():
            i, s = self._bucket(k)
            v[i] += s * c * self.idf.get(k, self.idf_unseen)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


_FEATURES = {}                  # (catalog content hash, dim) -> RouteFeatures, shared by every mind in the process


def shared_features(catalog, dim=ROUTER_DIM):
    """The RouteFeatures for a catalog's CURRENT content, built once per process and shared. Why: building the IDF
    table walks ~4,000 card texts (measured 2.5 s under load), and every UnifiedMind builds an identical catalog --
    a test suite that reports route outcomes on fresh minds would pay it per mind. The features are a pure function
    of the catalog content (keyed by the catalog's own content hash, _fc_hash: names + does + aliases), read-only
    once built, so sharing them is exact. Bounded to the 4 most recent catalogs."""
    catalog._ensure_fc_baked()
    key = (getattr(catalog, "_fc_hash", None), int(dim))
    f = _FEATURES.get(key) if key[0] else None
    if f is None:
        f = RouteFeatures(catalog, dim=dim)
        if key[0]:
            if len(_FEATURES) >= 4:
                _FEATURES.pop(next(iter(_FEATURES)))
            _FEATURES[key] = f
    return f


class RouteLearner:
    """The router's contrastive learner: a ProtoStore("route") over cards + the fusion rule. See the module docstring.

    The store holds only cards that have been LABELLED or been a labelled question's rival (added lazily, seeded
    from the card's text), so a mind that has learned three routes carries three-ish rows, not 4,000. A card not in
    the store is scored against its text seed (cached), which is exactly what its row would be on arrival."""

    def __init__(self, catalog, dim=ROUTER_DIM, lam=ROUTER_LAM, mu=ROUTER_MU, k=ROUTER_K, features=None):
        self.catalog = catalog
        self.features = features or shared_features(catalog, dim=dim)
        self.dim = self.features.dim
        self.lam, self.mu, self.k = float(lam), float(mu), int(k)
        self.store = ProtoStore(self.dim, tau=0.05, lr=0.3, topk=64, name="route", mine=False)
        self._seed = {}                     # card name -> its text seed vector (cache)
        self.n_learned = 0
        self.guard_ok = True                # every question this store learned from passed the learning guard
        self.salted = False                 # ... and none was a session-salted question (never shareable then)

    # ------------------------------------------------------------------ the store
    def seed_vec(self, name):
        """The text seed of a card (name + does), cached. Unknown card -> zero vector."""
        v = self._seed.get(name)
        if v is None:
            cap = self.catalog.get(name)
            v = self.features.encode(RouteFeatures.card_text(cap)) if cap is not None else np.zeros(self.dim)
            self._seed[name] = v
        return v

    def _ensure(self, names):
        """Add every missing card to the store in ONE rebuild, seeded from its text. (ProtoStore.add_option copies
        the whole matrix per call -- O(n^2) for thousands of cards -- so batches go through state/from_state.)"""
        missing = [n for n in dict.fromkeys(names) if n not in self.store and self.catalog.get(n) is not None]
        if not missing:
            return 0
        meta, arr = self.store.state()
        A = np.asarray(arr["A"], np.float64).reshape(len(self.store.labels), self.dim)
        new = np.stack([self.seed_vec(n) for n in missing])
        meta["labels"] = list(meta["labels"]) + missing
        meta["count"] = list(meta["count"]) + [0] * len(missing)
        old = self.store
        self.store = ProtoStore.from_state(meta, {"A": np.vstack([A, new]) if len(A) else new})
        # from_state goes through float32 for the EXISTING rows (the persisted dtype); keep the live float64 rows
        if len(A):
            self.store.A[:len(A)] = old.A
            self.store.P[:len(A)] = old.P
        return len(missing)

    def learn(self, problem, truth, rivals=()):
        """One labelled verdict: `problem` routes to card `truth`. The truth and its rivals (the lexical candidates
        that competed -- the hard negatives already in the record) join the store, then the InfoNCE rule moves every
        candidate row: pull the truth, push whatever the question looked like instead. -> the rule's report."""
        if self.catalog.get(truth) is None:
            return {"learned": False, "why": "unknown card %r" % (truth,)}
        self._ensure([truth] + [r for r in rivals if r != truth])
        rep = self.store.update(self.features.encode(problem), truth)
        self.n_learned += 1
        rep["learned"] = True
        return rep

    def learn_many(self, items):
        """Many verdicts at once -- [(problem, truth, rivals)] in order (the pretraining pass): every card any of them
        needs joins the store in ONE rebuild first (one-at-a-time _ensure copies the whole matrix per new card), then
        the verdicts apply in order, exactly as learn() would. -> number learned."""
        items = [(str(q), t, list(r or ())) for q, t, r in items if self.catalog.get(t) is not None]
        need = []
        for _, t, r in items:
            need.append(t)
            need.extend(x for x in r if x != t)
        self._ensure(need)
        for q, t, _ in items:
            self.store.update(self.features.encode(q), t)
            self.n_learned += 1
        return len(items)

    # ------------------------------------------------------------------ the fused ranking
    def cosines(self, q, names):
        """cos(q, prototype) for each card name: the learned row when the card is in the store, else its seed."""
        out = np.empty(len(names))
        for j, n in enumerate(names):
            if n in self.store:
                out[j] = float(self.store.P[self.store.index(n)] @ q)
            else:
                out[j] = float(self.seed_vec(n) @ q)
        return out

    def verdicts(self, names):
        """The store's labelled-verdict count per card (0 for a card never labelled)."""
        return np.array([self.store.count[self.store.index(n)] if n in self.store else 0 for n in names], float)

    def rerank(self, problem, ranked):
        """ranked = [(capability, lexical score)] best first (the lexical top-K). Returns the same capabilities
        reordered by the fused score, as [(capability, fused, lexical, cosine, verdicts)] best first. Ties keep the
        LEXICAL order (a stable sort on -fused), so lam = mu = 0 is exactly the lexical ranking."""
        if not ranked:
            return []
        q = self.features.encode(problem)
        names = [c.name for c, _ in ranked]
        lex = np.array([float(s) for _, s in ranked])
        cos = self.cosines(q, names)
        cnt = self.verdicts(names)
        fused = lex + self.lam * cos + self.mu * np.log1p(cnt)
        order = np.argsort(-fused, kind="stable")
        return [(ranked[int(j)][0], float(fused[j]), float(lex[j]), float(cos[j]), float(cnt[j])) for j in order]

    # ------------------------------------------------------------------ persistence
    def state(self):
        """(meta, arrays): the store's state + the fusion settings + the feature-space digest + the guard flags."""
        meta, arr = self.store.state()
        return ({"store": meta, "lam": self.lam, "mu": self.mu, "k": self.k, "dim": self.dim,
                 "features": self.features.digest, "n_learned": self.n_learned,
                 "guard_ok": bool(self.guard_ok), "salted": bool(self.salted)},
                {"A": arr["A"]})

    @classmethod
    def from_state(cls, catalog, meta, arrays):
        """Rebuild against a catalog. Cards that no longer exist are dropped (their rows mean nothing now); a changed
        IDF table (the catalog grew) keeps the learned rows -- the buckets are stable, only the weights moved --
        and says so in `features_changed`."""
        lr = cls(catalog, dim=int(meta.get("dim", ROUTER_DIM)), lam=meta.get("lam", ROUTER_LAM),
                 mu=meta.get("mu", ROUTER_MU), k=meta.get("k", ROUTER_K))
        st = ProtoStore.from_state(meta["store"], arrays)
        keep = [i for i, n in enumerate(st.labels) if catalog.get(n) is not None]
        if len(keep) < len(st.labels):
            m2, a2 = st.state()
            m2["labels"] = [st.labels[i] for i in keep]
            m2["count"] = [st.count[i] for i in keep]
            st = ProtoStore.from_state(m2, {"A": np.asarray(a2["A"])[keep]})
        lr.store = st
        lr.n_learned = int(meta.get("n_learned", 0))
        lr.guard_ok = bool(meta.get("guard_ok", True))
        lr.salted = bool(meta.get("salted", False))
        lr.features_changed = meta.get("features") != lr.features.digest
        return lr


def loo_candidates(catalog, alias, k=ROUTER_K):
    """The lexical top-k for `alias` AS IF IT WERE NOT IN THE INDEX: the alias leaves every card that carries it
    (tokens and the exact-alias bonus), the catalog is scored, and every card is restored. These are the rivals an
    unseen wording of that card meets -- the hard negatives a pretraining pass should push (an alias scored with
    itself in the index gets the +5 exact bonus and has no rivals worth the name). -> [(capability, score)]."""
    catalog._ensure_fc_baked()
    carriers = [c for c in catalog.all() if alias in (c.aliases or ())]
    saved = []
    for c in carriers:
        saved.append((c, c._hay, c._al))
        rest = tuple(a for a in c.aliases if a != alias)
        c._hay = c._nw | set(_tokens(c.does)) | _alias_tokens(rest)
        c._al = tuple(a.lower() for a in rest)
    try:
        q = set(_tokens(alias))
        scored = catalog._score_all(alias, q) if q else []
    finally:
        for c, h, al in saved:
            c._hay, c._al = h, al
    return [(cap, float(s)) for s, _, cap in scored[:int(k)]]


def _selftest():
    """Contracts on a hand-made catalog (the levels live in tools/bench_router.py)."""
    from holographic.caching_and_storage.holographic_catalog import Catalog
    cat = Catalog()
    cat.register_capability("Smooth a mesh", "smooth a bumpy mesh surface", aliases=("denoise mesh",))
    cat.register_capability("Sharpen a mesh", "sharpen mesh creases and edges", aliases=("crisp edges",))
    cat.register_capability("Render a scene", "path trace a scene to an image", aliases=("make a picture",))
    lr = RouteLearner(cat, dim=256)
    ranked = cat.find_scored("mesh surface", k=3)
    # 1) lam = mu = 0 is exactly the lexical order
    lr0 = RouteLearner(cat, dim=256, lam=0.0, mu=0.0, features=lr.features)
    assert [c.name for c, *_ in lr0.rerank("mesh surface", ranked)] == [c.name for c, _ in ranked]
    # 2) learning adds the truth + rivals only, and a verdict raises the truth's prior
    rep = lr.learn("fix a lumpy mesh", "Smooth a mesh", rivals=["Sharpen a mesh"])
    assert rep["learned"] and set(lr.store.labels) == {"Smooth a mesh", "Sharpen a mesh"}
    assert lr.verdicts(["Smooth a mesh", "Sharpen a mesh"]).tolist() == [1.0, 0.0]
    # 3) state round trip gives the same fused ranking
    meta, arr = lr.state()
    lr2 = RouteLearner.from_state(cat, meta, arr)
    a = [(c.name, round(f, 6)) for c, f, *_ in lr.rerank("mesh", cat.find_scored("mesh", k=3))]
    b = [(c.name, round(f, 6)) for c, f, *_ in lr2.rerank("mesh", cat.find_scored("mesh", k=3))]
    assert a == b and not lr2.features_changed
    # 4) LOO candidates: the alias's own +5 bonus is gone, the card is restored afterwards
    before = [c._al for c in cat.all()]
    loo = loo_candidates(cat, "denoise mesh")
    assert all(s < 5.0 for _, s in loo) and [c._al for c in cat.all()] == before
    return "ok"


if __name__ == "__main__":
    print(_selftest())

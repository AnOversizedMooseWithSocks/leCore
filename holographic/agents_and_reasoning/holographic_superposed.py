"""holographic_superposed.py -- SUPERPOSED FROZEN ENCODERS: several text channels in one hypervector, plus a
handful of learned per-channel relevances (the CLM backlog, item E6.1).

THE IDEA, IN ONE PARAGRAPH
--------------------------
leCore already has more than one frozen way to turn text into a vector: character n-grams (SystemOne's
`hashed_ngram_encode`), word hashing (random indexing's index vectors), and a vendored WordNet-derived dictionary
that knows synonyms and "is a kind of" parents. Each sees something the others miss -- n-grams survive typos and
morphology, words keep identity, synonyms bridge "refund" and "repayment". This module puts all of them into ONE
vector the way the rest of the engine composes things: each channel's vector is BOUND under its own unitary ROLE
key, and the bound channels are SUMMED:

    H(x) = sum_c  a_c * (R_c (*) x_c)          (*) = circular convolution (holographic_ai.bind)

A unitary role (every Fourier magnitude exactly 1) makes binding an ISOMETRY: <R (*) x, R (*) y> = <x, y> exactly,
and two different random roles make their channels quasi-orthogonal (cross-talk ~ 1/sqrt(dim)). So

    <H(q), H(p)>  ~=  sum_c  a_c^2 <q_c, p_c>   +  cross-talk

i.e. the superposition is a weighted sum of per-channel cosines, packed into one dim-sized vector instead of a
C*dim concatenation. The weights are the only thing learned, and they are learned GRLVQ-style (Hammer & Villmann
2002, "generalized relevance LVQ"): a handful of non-negative scalars, one per channel, summing to 1, moved by
every labelled verdict. NOT a matrix -- the panel settled that (Q2: per-row prototypes, no projection matrix;
the ridge-W probe lost).

WHERE THE RELEVANCE IS APPLIED, AND WHY THERE
---------------------------------------------
On the QUERY side only. The prototypes live in ONE ProtoStore built from the uniform superposition, and a query
is re-weighted as  H_lam(q) = sum_c lam_c * (R_c (*) q_c).  Then <H_lam(q), P_k> = sum_c lam_c * s_kc, where
s_kc = <R_c (*) q_c, P_k> is channel c's own score for option k -- an exact linear readout, no unbinding pass.
Two reasons: (1) changing lam never invalidates a stored prototype (nothing to re-encode, the store keeps
learning by the ProtoStore rule unchanged); (2) the per-channel scores the GRLVQ step needs fall out of the same
dot product.

WHAT WAS MEASURED -- KEPT NEGATIVE (tools/bench_superposed.py -> docs/research/evidence/bench_superposed.json)
-------------------------------------------------------------------------------------------------------------
dim 2048, the ProtoStore rule (InfoNCE, tau 0.05, lr 0.3), K=5 init + one pass over every other training wording,
3 seeds, top-1 on the test split with a paired bootstrap CI pooled over seeds (2026-09-26, 47 min on a shared
2-core box):

                        CLINC150 (150 intents)          Banking77 (77 intents)
    ngram   (alone)     0.9029 [0.8948, 0.9110]         0.8813 [0.8710, 0.8915]    <- THE WINNER, both datasets
    word    (alone)     0.8699                          0.8466
    synonym (alone)     0.8592                          0.8169
    superposed          0.8999  (-0.30 [-0.76, +0.19])  0.8761  (-0.52 [-1.14, +0.09])   tie
    superposed + lam    0.9021  (-0.07 [-0.53, +0.43])  0.8746  (-0.67 [-1.29, -0.08])   tie / LOSS
    fusion_exact (diag) 0.9072  (+0.43 [+0.01, +0.95])  0.8794  (-0.18 [-0.77, +0.37])   win / tie

AURC (absolute top cosine as the confidence) ties everywhere: CLINC 0.0272 ngram vs 0.0273 superposed vs 0.0272
with relevances; with out-of-scope rows at the 4,500:1,000 prior, 0.0630 vs 0.0640 vs 0.0636 (all CIs span 0).

What that says, plainly:
  * The superposition does NOT beat its best channel. The acceptance's other branch applies: the character n-gram
    channel (= hashed_ngram_encode) won on both datasets. Keep using it alone for text doors.
  * The channels ARE slightly complementary on CLINC -- exact score fusion (three stores, 3x the memory, no
    cross-talk) is +0.43 above n-grams -- but binding them into ONE 2048-d vector costs ~0.7 points of cross-talk
    (0.9072 -> 0.8999) and eats the gain. On Banking77 even exact fusion does not beat n-grams.
  * The GRLVQ relevances learned a stable mix (CLINC ngram 0.47 / word 0.17 / synonym 0.36; Banking77 0.40 / 0.25
    / 0.35, identical to 3 decimals across seeds) and recovered +0.22 of the superposition's loss on CLINC; on
    Banking77 they lost 0.15 more. Larger relevance steps drove the word channel to 0 and scored WORSE on the
    validation split (CLINC 0.8887 at lr 0.001 vs 0.8857 at 0.01 and 0.05; Banking77 0.8671 vs 0.8585 / 0.8595),
    so the chosen step is the smallest in the grid. The GRLVQ margin cost is not top-1, and the prototypes were
    trained on the UNIFORM superposition, so a re-weighted query scores against rows it did not shape.
  * The synonym channel is the weakest alone (one WordNet sense per word, as warned below), yet the relevance
    learner kept it ABOVE the word channel -- it carries something the other two do not, just not enough.
Not tried (the obvious next measurements, not claims): a wider superposition (dim 4096+) to cut the cross-talk;
prototypes trained on the relevance-weighted query; a random-indexing CONTEXT channel fitted on unlabelled text.

WHAT THIS MODULE DOES NOT DO
----------------------------
  * No trained weights. Every channel is FROZEN: its atoms are sha256-seeded, its roles are derived from names.
    The only learned state is the relevance vector (C scalars) and whatever ProtoStore the caller owns.
  * No nltk. The synonym channel reads leCore's vendored dictionary (holographic_dictionary, Princeton WordNet
    3.0 extracted once at build time). If that file is absent the channel is SKIPPED with the reason recorded in
    `encoder.skipped` -- never silently, never a crash.
  * KNOWN WEAKNESS of the synonym channel, stated before measuring: the vendored dictionary stores ONE sense per
    word (the primary one), so "pin" is-a "jewelry" and "cancel" has the synonym "natural" (the musical sign).
    Word-sense collisions bite exactly as holographic_word_index's docstring warns.

numpy + stdlib only; deterministic (hashlib seeds, never hash(); stable sorts; insertion-order ties).
"""
import hashlib
import os
import re
from collections import OrderedDict

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind_batch, derived_atom, involution
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode

# The channels this module knows by name, in their default order (the order is also the tie order).
CHANNELS = ("ngram", "word", "synonym")


# ---------------------------------------------------------------------------------------------------------------
# 1. ATOMS -- one frozen random index vector per feature string
# ---------------------------------------------------------------------------------------------------------------
def _atom_seed(feature):
    """THE seed rule of hashed_ngram_encode, reproduced exactly: first 8 bytes of sha256(feature) mod 2**32.
    Using the identical rule is what lets the n-gram channel equal SystemOne's encoder (pinned in the tests)."""
    return int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big") % (2 ** 32)


def feature_atom(feature, dim):
    """The frozen index vector of one feature string: a standard-normal draw seeded by sha256(feature).
    Not unit-normalised, on purpose -- hashed_ngram_encode does not normalise its atoms either, and the
    channel vector is unit-normalised once at the end."""
    return np.random.default_rng(_atom_seed(feature)).standard_normal(int(dim))


# ---------------------------------------------------------------------------------------------------------------
# 2. FEATURE BAGS -- what each channel "sees" in a text. A bag is {feature_string: weight}.
#    Every channel is the same machine (a weighted sum of index vectors) with a different feature extractor, so
#    one batch encoder serves all three and there is exactly one place the vector arithmetic happens.
# ---------------------------------------------------------------------------------------------------------------
def ngram_bag(text, lo=3, hi=5):
    """Character lo..hi-grams of ' ' + text.lower() + ' ', counted -- EXACTLY the grams hashed_ngram_encode sums.
    The feature string is the bare gram (no channel prefix) so the atoms are hashed_ngram_encode's atoms."""
    t = " " + str(text).lower() + " "
    bag = {}
    for n in range(int(lo), int(hi) + 1):
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            bag[g] = bag.get(g, 0.0) + 1.0
    return bag


_WORD_RE = re.compile(r"[a-z0-9]+")


def word_tokens(text):
    """Lower-case alphanumeric word tokens, in order (apostrophes split: "don't" -> "don", "t")."""
    return _WORD_RE.findall(str(text).lower())


def word_bag(text):
    """WORD HASHING (random indexing's index vectors): every word token and every adjacent word pair is one
    feature. The pair features add a little word ORDER, which a pure bag of words cannot see ("transfer to
    savings" vs "savings to transfer"). Prefixed "w\\x00" / "b\\x00" so no word atom can coincide with an
    n-gram atom -- the channels are independent carriers even before their roles separate them."""
    toks = word_tokens(text)
    bag = {}
    for t in toks:
        k = "w\x00" + t
        bag[k] = bag.get(k, 0.0) + 1.0
    for a, b in zip(toks, toks[1:]):
        k = "b\x00" + a + " " + b
        bag[k] = bag.get(k, 0.0) + 1.0
    return bag


def synonym_available():
    """(ok, why): is the vendored dictionary present? Checks the FILE, never loads it -- loading costs ~0.5 s
    and ~150 MB resident (measured on this box; the catalog card's "~22 MB" understates the Python-dict cost),
    so availability must be answerable without paying that."""
    try:
        import holographic.misc.holographic_dictionary as hd
    except Exception as exc:                                   # the module itself is core; this is paranoia
        return False, "holographic_dictionary did not import: %s" % exc
    path = getattr(hd, "_DATA_PATH", None)
    if not path or not os.path.exists(path):
        return False, "vendored dictionary not found at %s (lecore_data/knowledge/ missing)" % path
    return True, None


def _lemma(hd, w):
    """The dictionary key for a token: the token itself, else the token minus a plural 's' when THAT is a
    headword. Deliberately the crudest stemmer that helps ("cards" -> "card"); anything smarter is a
    morphology model, which is out of scope for a frozen channel."""
    if hd.has(w):
        return w
    if len(w) > 3 and w.endswith("s") and hd.has(w[:-1]):
        return w[:-1]
    return None


def synonym_bag(text):
    """WORDNET CHANNEL: each content word w expands to S(w) = {w} + its dictionary synonyms + its is_a parent,
    and each member of S(w) carries weight 1/|S(w)|, so every word contributes the same total mass whether the
    dictionary knows 0 or 6 synonyms for it (otherwise a word with many synonyms would dominate the vector).
    Words the dictionary does not know contribute only themselves. Stop words are dropped with the SAME stop
    list the semantic word index uses (holographic_word_index), so there is one definition of "content word".
    Prefixed "s\\x00". Loads the dictionary on first call (~0.5 s, ~150 MB)."""
    import holographic.misc.holographic_dictionary as hd
    from holographic.caching_and_storage.holographic_word_index import _content_tokens
    bag = {}
    for w in _content_tokens(text):
        members = [w]
        lem = _lemma(hd, w)
        if lem is not None:
            if lem != w:
                members.append(lem)
            for s in hd.synonyms(lem) or []:
                s = str(s).lower().strip()
                if s and s not in members:
                    members.append(s)
            parent = hd.is_a(lem)
            if parent:
                parent = str(parent).lower().strip()
                if parent not in members:
                    members.append(parent)
        wt = 1.0 / len(members)
        for m in members:
            k = "s\x00" + m
            bag[k] = bag.get(k, 0.0) + wt
    return bag


_BAGS = {"ngram": ngram_bag, "word": word_bag, "synonym": synonym_bag}


def _bag_block(bags, dim):
    """float64 unit rows for one block of bags, FEATURE-MAJOR: each distinct feature's atom is generated once for
    the block, scattered into the rows that contain it, and dropped (see bag_matrix for why)."""
    from array import array
    fid, feat_of = {}, []
    # Three flat typed arrays, not a list of tuples: a tuple per (text, feature) entry costs ~100 bytes of Python
    # objects, these cost 20 bytes an entry.
    col_f, col_i, col_w = array("q"), array("q"), array("d")
    for i, bag in enumerate(bags):
        for f, w in bag.items():
            j = fid.get(f)
            if j is None:
                j = fid[f] = len(feat_of)
                feat_of.append(f)
            col_f.append(j)
            col_i.append(i)
            col_w.append(w)
    H = np.zeros((len(bags), int(dim)), dtype=np.float64)
    if len(col_f):
        arr_f = np.frombuffer(col_f, dtype=np.int64)
        arr_i = np.frombuffer(col_i, dtype=np.int64)
        arr_w = np.frombuffer(col_w, dtype=np.float64)
        order = np.argsort(arr_f, kind="stable")              # group by feature; stable keeps text order inside
        arr_f, arr_i, arr_w = arr_f[order], arr_i[order], arr_w[order]
        cuts = np.flatnonzero(np.diff(arr_f)) + 1
        for seg_i, seg_w, f in zip(np.split(arr_i, cuts), np.split(arr_w, cuts), arr_f[np.r_[0, cuts]]):
            # seg_i has unique rows (one bag entry per text per feature), so fancy-index += is exact here
            H[seg_i] += seg_w[:, None] * feature_atom(feat_of[int(f)], dim)[None, :]
    nrm = np.linalg.norm(H, axis=1, keepdims=True)
    return np.where(nrm > 0, H / np.maximum(nrm, 1e-12), 0.0)


def bag_matrix(texts, bag_fn, dim, dtype=np.float32, chunk=4096):
    """BATCH ENCODER shared by every channel: row i = unit(sum_f bag_fn(texts[i])[f] * atom(f)).

    FEATURE-MAJOR, IN ROW CHUNKS -- both choices are memory, measured on CLINC150 (19,500 texts, dim 2048):
      * the obvious per-text loop needs every atom it has seen in a cache, and all ~150k distinct n-grams at
        dim 2048 is ~2.4 GB -- past this box's memory. Feature-major generates each atom once per chunk;
      * building every bag first and accumulating all rows in float64 peaked at 1,415 MB resident (n-gram
        channel). Chunks of `chunk` texts keep one block of bags and one float64 block alive at a time; the
        cost is re-generating a common feature's atom once per chunk (a few seconds on CLINC150).
    Sums in a different order than a per-text loop, so it matches hashed_ngram_encode to float rounding
    (allclose), not bit-for-bit -- pinned in tests/test_superposed.py. Empty bags give zero rows (never NaN)."""
    texts = list(texts)
    out = np.zeros((len(texts), int(dim)), dtype=dtype)
    for a in range(0, len(texts), int(chunk)):
        blk = texts[a:a + int(chunk)]
        out[a:a + len(blk)] = _bag_block([bag_fn(t) for t in blk], dim)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. THE ENCODER -- channels bound under unitary roles and summed
# ---------------------------------------------------------------------------------------------------------------
class SuperposedEncoder:
    """Several frozen text channels superposed into one hypervector under unitary role keys.

    channels : names from CHANNELS and/or (name, fn) pairs where fn(text) -> vector of length dim (a custom
               frozen channel; it is unit-normalised here like the built-in ones)
    dim      : the hypervector width (every channel and the superposition share it)
    seed     : seeds the ROLE keys (derived_atom(seed, 'superposed-role:<name>', unitary=True)). Atoms are
               seeded by feature string alone, so the n-gram channel stays equal to hashed_ngram_encode.
    cache    : bound on the per-text atom LRU (atoms are dim float64 each; 2048 atoms = 32 MB at dim 2048).
               float64, not float32, so the n-gram channel equals hashed_ngram_encode to float rounding
               (a float32 cache moved it by ~1e-7). SystemOne's hashed_ngram_encode used to cache without bound;
               since 2026-09-27 it shares one LRU per dimension capped by NGRAM_CACHE_MB (256 MB default), and this
               encoder keeps its own smaller per-text LRU on top.
               A cache miss costs one sha256 + one dim-length normal draw (~20 us at dim 2048).

    `skipped` maps a requested channel that could not be built to the reason (the synonym channel when the
    vendored dictionary is missing). It is never silent: a skipped channel is in `skipped`, not in `channels`."""

    def __init__(self, channels=CHANNELS, dim=2048, seed=0, cache=2048):
        self.dim = int(dim)
        self.seed = int(seed)
        self.channels = []
        self.skipped = {}
        self._custom = {}
        for ch in channels:
            if isinstance(ch, (tuple, list)) and len(ch) == 2 and callable(ch[1]):
                name, fn = str(ch[0]), ch[1]
                self._custom[name] = fn
            elif ch in _BAGS:
                name = ch
                if name == "synonym":
                    ok, why = synonym_available()
                    if not ok:
                        self.skipped[name] = why
                        continue
            else:
                raise ValueError("unknown channel %r -- use one of %s or a (name, fn) pair" % (ch, CHANNELS))
            if name in self.channels:
                raise ValueError("channel %r listed twice" % name)
            self.channels.append(name)
        if not self.channels:
            raise ValueError("no channel could be built: %s" % self.skipped)
        # One UNITARY role per channel: binding by it is an exact isometry (inner products preserved), and its
        # inverse is exactly involution(role) -- so unbinding a channel back out is exact up to cross-talk.
        self.roles = {c: derived_atom(self.seed, "superposed-role:" + c, self.dim, unitary=True)
                      for c in self.channels}
        self._R = np.stack([self.roles[c] for c in self.channels])
        self._cache_max = int(cache)
        self._atoms = OrderedDict()

    # ---- per-text path ------------------------------------------------------------------------------------
    def _atom(self, feature):
        """feature_atom with a bounded LRU (least recently used atom evicted first)."""
        v = self._atoms.get(feature)
        if v is None:
            v = feature_atom(feature, self.dim)
            self._atoms[feature] = v
            if len(self._atoms) > self._cache_max:
                self._atoms.popitem(last=False)
        else:
            self._atoms.move_to_end(feature)
        return v

    def channel_vector(self, name, text):
        """One channel's UNBOUND unit vector for a text (zeros when the channel sees nothing, e.g. a text of
        only stop words in the synonym channel)."""
        if name in self._custom:
            v = np.asarray(self._custom[name](text), dtype=np.float64).reshape(-1)
            if v.shape[0] != self.dim:
                raise ValueError("custom channel %r returned dim %d, encoder dim is %d" % (name, v.shape[0], self.dim))
        else:
            v = np.zeros(self.dim)
            for f, w in _BAGS[name](text).items():
                v += w * self._atom(f)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def channel_vectors(self, text):
        """(C, dim): every channel's unbound unit vector, rows in self.channels order."""
        return np.stack([self.channel_vector(c, text) for c in self.channels])

    def bound(self, text):
        """(C, dim): each channel's vector bound under its role -- the terms the superposition sums."""
        return bind_batch(self._R, self.channel_vectors(text))

    def encode(self, text, weights=None):
        """THE superposed hypervector (unit): sum_c w_c * (R_c (*) x_c). weights=None is uniform. `weights` is a
        sequence in channel order or a {name: w} dict (a missing name weighs 0)."""
        return self.combine(self.bound(text), weights)

    def combine(self, B, weights=None):
        """Superpose already-bound channel rows B (C, dim) with weights -> unit vector."""
        w = self._weights(weights)
        v = w @ np.asarray(B, dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _weights(self, weights):
        if weights is None:
            return np.ones(len(self.channels))
        if isinstance(weights, dict):
            return np.array([float(weights.get(c, 0.0)) for c in self.channels])
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != len(self.channels):
            raise ValueError("need %d weights (one per channel), got %d" % (len(self.channels), w.shape[0]))
        return w

    def unbind(self, vec, name):
        """Recover channel `name`'s vector from a superposition: R_c^-1 (*) H. For a unitary role the inverse is
        exactly the involution, so the estimate is the channel vector plus the other channels' cross-talk."""
        r = self.roles[name]
        return bind_batch(np.asarray(vec, dtype=np.float64)[None, :], involution(r)[None, :])[0]

    def channel_scores(self, text_or_bound, P):
        """(C, K) per-channel scores s_kc = <R_c (*) q_c, P_k> of a text (or its bound rows) against prototype
        rows P (K, dim). With the relevance vector lam, the superposed score is lam @ this."""
        B = self.bound(text_or_bound) if isinstance(text_or_bound, str) else np.asarray(text_or_bound, float)
        return B @ np.asarray(P, dtype=np.float64).T

    # ---- batch path (fitting a store, benches) ------------------------------------------------------------
    def matrix(self, name, texts, dtype=np.float32):
        """(N, dim) unbound unit rows of one channel for many texts, feature-major (see bag_matrix)."""
        if name in self._custom:
            return np.stack([self.channel_vector(name, t) for t in texts]).astype(dtype)
        return bag_matrix(texts, _BAGS[name], self.dim, dtype=dtype)

    def bind_rows(self, name, M, chunk=2048, dtype=np.float32):
        """Bind every row of an (N, dim) channel matrix under that channel's role, in chunks (bounded memory)."""
        M = np.asarray(M)
        out = np.empty(M.shape, dtype=dtype)
        r = self.roles[name][None, :]
        for a in range(0, M.shape[0], int(chunk)):
            blk = M[a:a + int(chunk)].astype(np.float64)
            out[a:a + int(chunk)] = bind_batch(np.repeat(r, blk.shape[0], axis=0), blk)
        return out

    def describe(self):
        """JSON-able description: the encoder is frozen, so its config IS its state."""
        return {"channels": list(self.channels), "dim": self.dim, "seed": self.seed, "skipped": dict(self.skipped)}


# ---------------------------------------------------------------------------------------------------------------
# 4. THE LEARNED PART -- one relevance scalar per channel, GRLVQ-style
# ---------------------------------------------------------------------------------------------------------------
class ChannelRelevance:
    """Per-channel relevances lam (non-negative, summing to 1), learned from labelled verdicts GRLVQ-style.

    GRLVQ (Hammer & Villmann 2002) learns one relevance per input dimension by descending the relative-distance
    cost mu = (d+ - d-) / (d+ + d-), where d+ is the relevance-weighted distance to the nearest correct prototype
    and d- to the nearest wrong one. Here the "dimensions" are whole CHANNELS -- the grouped form -- and each
    channel's distance to a prototype is d_c = 1 - s_kc (s_kc the channel's score, see
    SuperposedEncoder.channel_scores), so d_lam = sum_c lam_c d_c = 1 - lam @ s. The gradient is

        d mu / d lam_c = 2 (d- * d_c+  -  d+ * d_c-) / (d+ + d-)^2

    so a channel whose own evidence favours the truth over the rival MORE than the current mix does gains
    relevance. After each step lam is clipped at `floor` and renormalised to sum 1 (the GRLVQ convention; only
    the ratios matter for ranking). f(mu) is the identity -- the plain GRLVQ step, one hyperparameter (lr).

    That is the whole learned state: C scalars. The panel ruled out a projection matrix (Q2); this is the
    smallest learnable thing that can still prefer one frozen encoder over another."""

    def __init__(self, channels, lr=0.05, floor=0.0):
        self.channels = list(channels)
        if not self.channels:
            raise ValueError("ChannelRelevance needs at least one channel")
        self.lr = float(lr)
        self.floor = float(floor)
        self.lam = np.full(len(self.channels), 1.0 / len(self.channels))
        self.n = 0

    @property
    def weights(self):
        """{channel: relevance}, in channel order."""
        return {c: float(v) for c, v in zip(self.channels, self.lam)}

    def score(self, S):
        """Relevance-weighted scores: lam @ S for S (C, K) -> (K,)."""
        return self.lam @ np.asarray(S, dtype=np.float64)

    def observe(self, S, truth):
        """One labelled verdict. S: (C, K) per-channel scores of the query against every option; truth: the true
        option's column index. Returns {pred, rival, correct, mu, lam}. With a single option there is no rival
        and nothing to learn (returned as rival None)."""
        S = np.asarray(S, dtype=np.float64)
        y = int(truth)
        tot = self.lam @ S
        pred = int(np.argmax(tot))                    # argmax: first index wins ties (insertion order)
        if S.shape[1] < 2:
            return {"pred": pred, "rival": None, "correct": pred == y, "mu": None, "lam": self.weights}
        masked = tot.copy()
        masked[y] = -np.inf
        rival = int(np.argmax(masked))
        dp = 1.0 - S[:, y]                            # per-channel distance to the truth
        dm = 1.0 - S[:, rival]                        # ... and to the nearest wrong option
        Dp, Dm = float(self.lam @ dp), float(self.lam @ dm)
        den = (Dp + Dm) ** 2
        mu = (Dp - Dm) / (Dp + Dm) if (Dp + Dm) > 0 else 0.0
        if den > 1e-12:
            grad = 2.0 * (Dm * dp - Dp * dm) / den
            lam = np.maximum(self.lam - self.lr * grad, self.floor)
            s = lam.sum()
            if s > 0:
                self.lam = lam / s
        self.n += 1
        return {"pred": pred, "rival": rival, "correct": pred == y, "mu": float(mu), "lam": self.weights}

    def state(self):
        """JSON-able: the relevances ARE the learned state (C floats)."""
        return {"channels": list(self.channels), "lr": self.lr, "floor": self.floor,
                "lam": [float(v) for v in self.lam], "n": self.n}

    @classmethod
    def from_state(cls, st):
        """Rebuild from state()."""
        r = cls(st["channels"], lr=st.get("lr", 0.05), floor=st.get("floor", 0.0))
        r.lam = np.asarray(st["lam"], dtype=np.float64)
        r.n = int(st.get("n", 0))
        return r


# ---------------------------------------------------------------------------------------------------------------
# 5. A DOOR -- encoder + the shared ProtoStore rule + relevances, the exact arm the bench measures
# ---------------------------------------------------------------------------------------------------------------
class SuperposedDoor:
    """A nearest-prototype decision over the superposed encoder, learning from verdicts two ways at once:

      * the prototypes move by the ONE shared contrastive rule (holographic_protostore.ProtoStore.update), fed
        the UNIFORM superposition -- so they never depend on the current relevances;
      * the relevances move by ChannelRelevance.observe, fed the per-channel scores of the same verdict.

    rank() scores the relevance-weighted query when use_relevance is True, the uniform one otherwise -- the two
    arms of the bench share this one prototype trajectory, so their comparison is paired exactly."""

    def __init__(self, encoder, store=None, relevance=None, use_relevance=True, name="superposed"):
        from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
        self.encoder = encoder
        self.store = store if store is not None else ProtoStore(encoder.dim, name=name)
        if self.store.dim != encoder.dim:
            raise ValueError("store dim %d != encoder dim %d" % (self.store.dim, encoder.dim))
        self.relevance = relevance if relevance is not None else ChannelRelevance(encoder.channels)
        self.use_relevance = bool(use_relevance)

    def add_option(self, label, texts=None):
        """Seed an option's prototype at the unit mean of its example texts' uniform superpositions."""
        vecs = [self.encoder.encode(t) for t in (texts or [])]
        return self.store.add_option(label, vecs if vecs else None)

    def _query(self, B):
        w = self.relevance.lam if self.use_relevance else None
        return self.encoder.combine(B, w)

    def rank(self, text, k=None):
        """[(label, cosine)] best first for a text; ties keep insertion order."""
        return self.store.rank(self._query(self.encoder.bound(text)), k=k)

    def learn(self, text, truth):
        """One labelled verdict: decide first (the honest prequential test), then move the relevances and the
        prototypes. Returns {pred, correct, relevance, store}."""
        B = self.encoder.bound(text)
        pred = self.store.rank(self._query(B), k=1)
        pred = pred[0][0] if pred else None
        rel = None
        if truth in self.store and len(self.store) >= 2:
            S = self.encoder.channel_scores(B, self.store.P)
            rel = self.relevance.observe(S, self.store.index(truth))
        up = self.store.update(self.encoder.combine(B, None), truth)
        return {"pred": pred, "correct": pred == truth, "relevance": rel, "store": up}


def _selftest():
    """Pins the algebra this module rests on, with synthetic data only (no datasets, no dictionary load)."""
    dim = 512
    # 1. the n-gram channel IS hashed_ngram_encode (same atoms, same sum), to float rounding
    ref = hashed_ngram_encode(dim=dim)
    enc = SuperposedEncoder(channels=("ngram", "word"), dim=dim, seed=0)
    for t in ("where is my card", "I was charged twice!"):
        a = ref(t)
        a = a / np.linalg.norm(a)
        assert np.allclose(a, enc.channel_vector("ngram", t), atol=1e-9)
    # 2. the batch path matches the per-text path
    M = enc.matrix("word", ["transfer money now", "", "money money"])
    assert np.allclose(M[0], enc.channel_vector("word", "transfer money now"), atol=1e-5)
    assert not M[1].any(), "an empty text must give a zero row, not NaN"
    # 3. a unitary role is an isometry: bound inner products equal unbound ones
    X = enc.channel_vectors("pay my bill"), enc.channel_vectors("pay the bill please")
    B0, B1 = enc.bound("pay my bill"), enc.bound("pay the bill please")
    assert np.allclose((X[0] * X[1]).sum(1), (B0 * B1).sum(1), atol=1e-9)
    # 4. unbinding a channel out of the superposition recovers it above the cross-talk
    H = enc.encode("pay my bill")
    est = enc.unbind(H, "word")
    c = float(est @ X[0][1] / (np.linalg.norm(est) + 1e-12))
    assert c > 0.5, "unbind recovered the word channel at cosine %.3f" % c
    # 5. GRLVQ prefers an informative channel over a pure-noise one
    rng = np.random.default_rng(0)
    cls_atoms = rng.standard_normal((4, dim))

    def signal(t):
        k = int(t.split()[0][1:])
        return cls_atoms[k] + 0.8 * feature_atom("sig\x00" + t, dim)

    def noise(t):
        return feature_atom("noise\x00" + t, dim)

    enc2 = SuperposedEncoder(channels=(("signal", signal), ("noise", noise)), dim=dim, seed=0)
    door = SuperposedDoor(enc2)
    for k in range(4):
        door.add_option(k, ["c%d seed%d" % (k, j) for j in range(3)])
    for i in range(200):
        k = i % 4
        door.learn("c%d item%d" % (k, i), k)
    w = door.relevance.weights
    assert w["signal"] > w["noise"], w
    print("holographic_superposed selftest OK -- relevances on a signal/noise pair: %s" %
          {k: round(v, 3) for k, v in w.items()})


if __name__ == "__main__":
    _selftest()

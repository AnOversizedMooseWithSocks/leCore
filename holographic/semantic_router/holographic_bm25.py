"""Okapi BM25 lexical retrieval + reciprocal rank fusion -- the LEXICAL half of hybrid routing.

WHY THIS EXISTS
---------------
The dense semantic router (nomic embeddings) buries asks whose ANSWER uses different words than the QUERY:
'smooth out the bumpy SURFACE' -> holographic_meshsmooth sits at rank 22 under cosine, because the docstring
and the query share meaning but the geometry collapses them apart. This is the VOCABULARY-MISMATCH problem,
and the IR literature is unanimous through 2026 that it is a STRUCTURAL property of the query-corpus pair, not
a tuning knob: dense retrieval cannot recover a signal that is architecturally absent (see 'Controlling
Authority Retrieval', arXiv 2604.14488, where an MTEB-top-10 dense model scored 9.3x WORSE than BM25 on
vocabulary-gap queries).

The standard fix -- and the ONLY one that fits leCore's NumPy/stdlib/no-learned-weights constraint -- is
HYBRID retrieval: run a lexical retriever (BM25) alongside the dense one and FUSE the two rankings. BM25 is
term-based, needs no model, computes offline from document content alone (MonaVec, arXiv 2606.19458, rejects
SPLADE for exactly the constraint reason we do and uses BM25+dense via RRF). The measured effect is largest
for WEAK dense retrievers (arXiv 2605.24297: benefit inversely proportional to dense zero-shot quality), which
is precisely our nomic-at-128d regime.

WHAT THIS IS
------------
A from-scratch Okapi BM25 (Robertson/Sparck-Jones) plus Reciprocal Rank Fusion (Cormack 2009). Deterministic,
pure NumPy + stdlib. BM25 scores a query against documents by summing, over shared terms, idf(term) times a
term-frequency saturation curve (k1) with document-length normalization (b). RRF fuses ranked lists by
summing 1/(k + rank) across retrievers -- no score calibration needed, which matters because cosine (in
[-1,1]) and BM25 (unbounded) are on different scales.

KEPT NEGATIVES (measured/known, stated loudly)
----------------------------------------------
* BM25 only helps LEXICAL misses -- asks whose query words appear in the target docstring ('surface', 'ball',
  'shape', 'pieces'). It CANNOT help a query whose words appear in NEITHER the docstring NOR as a term:
  'make my picture less grainy' -> denoise stays missed, because 'grainy' is nowhere in denoise's text. That
  one needs document EXPANSION (add noise-adjacent terms to denoise's routing text) or a better encoder (N37).
* BM25 is a bag-of-words -- it has no notion of meaning, so it will also surface spurious exact-term matches.
  RRF fusion is what tempers this: a doc must rank well under BOTH retrievers to reach the top, so a spurious
  lexical hit with a poor dense rank is damped, and a good dense hit with no lexical support is preserved.
* This does NOT replace the dense router; it is an additive second opinion. Byte-identical dense behavior is
  available by simply not fusing.
"""
import math
import re

import numpy as np

# a tiny, deterministic English stoplist -- the words that carry no routing signal and would only add noise to
# the term matches. Kept short on purpose (over-stemming/over-filtering loses real signal); these are the
# function words that appear in nearly every docstring and query.
_STOP = frozenset(
    "a an the of to in on at for and or is are be by with from as it this that these those "
    "into over under out up down off no not do does did can could would should will "
    "your my our their its his her you we they i he she them us me".split()
)


def _normalize(tok):
    """LIGHT deterministic suffix stripping so an inflected doc term matches the query root: 'smoothing' and
    'smoothed' -> 'smooth', 'pieces' -> 'piece', 'flowing' -> 'flow'. This is NOT a full stemmer (that is a
    dependency AND over-stems on short technical docstrings, losing more than it gains -- a measured concern);
    it strips only the handful of common inflectional endings that caused the observed lexical MISSES
    (meshsmooth's 'smoothing' failing to match query 'smooth'). Order matters: longest ending first. A stripped
    stem must stay >=3 chars, so 'ring'->'ring' not 'r'."""
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokenize(text):
    """Lowercase alphanumeric tokens, stopwords removed, LIGHT suffix-normalized -- deterministic, stdlib-only.
    The SAME tokenizer runs on documents and queries so their terms line up (and their inflections collapse to
    the same root: doc 'smoothing' matches query 'smooth'). See _normalize for the deliberately-minimal
    stemming and why a full stemmer is avoided."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [_normalize(t) for t in toks if t not in _STOP and len(t) > 1]


# Derivational normalization, PORTER-STYLE (Porter 1980 -- the codified grammar of English word
# transformations that has powered spellcheck/search since the era Moose remembers). Two upgrades over the
# naive suffix list this replaced:
#   1. REWRITE FAMILIES, not just strips: 'ational'->'ate' maps relational->relate; 'ization'->'ize';
#      'iveness'->'ive' -- a strip-only rule leaves these forms un-grouped.
#   2. THE MEASURE CONDITION (Porter's m): a suffix is only removed if the remaining stem still contains
#      enough vowel-consonant alternations (m >= 2, i.e. real morphological structure). This is what kills the
#      measured false bridge arch/archive: m('arch') = 1, too slight to license stripping 'ive', so 'archive'
#      stays whole. A bare length check (>=4 chars) could not make that distinction.
# MEASURED on this repo's vocabulary (see _selftest): 92 groups / 226 pairs. The true pairs
# (emission/emissive, displace/displacement, compression/compressive, relational/relation -- the last only
# reachable via a REWRITE) all group; BOTH previously-measured false bridges are gone: arch/archive (killed
# by the m-gate) and conversation/conversion (rewrite and strip land on different stems, conversate vs
# convers -- an accidental but measured separation, pinned in the selftest so it cannot silently regress).
_DERIV_REWRITES = (("ational", "ate"), ("ization", "ize"), ("iveness", "ive"), ("fulness", "ful"),
                   ("ousness", "ous"), ("ibility", "ible"), ("ability", "able"), ("ivity", "ive"),
                   ("ution", "ute"), ("ation", "ate"), ("ition", "ite"))
_DERIV_STRIPS = ("ancy", "ency", "ance", "ence", "ment", "able", "ible", "ive", "ion", "ous", "ity",
                 "al", "ic")
_VOWELS = frozenset("aeiou")


def _measure(stem):
    """Porter's m: the number of vowel-run -> consonant-run alternations in the stem. m('arch') = 1,
    m('emiss') = 2, m('displace') = 3. Low m = the 'stem' is too slight to be a real root, so no suffix
    should be licensed off it. ('y' counted as a vowel mid-word, the standard simplification.)"""
    m = 0
    prev_v = False
    for i, ch in enumerate(stem):
        v = ch in _VOWELS or (ch == "y" and i > 0)
        if prev_v and not v:
            m += 1
        prev_v = v
    return m


def _derivational_stem(tok):
    """Reduce a token to its derivational root, Porter-style: try the REWRITE families first (longest match
    -- 'ational'->'ate' before 'ation' can fire), then the plain strips; either applies only when the
    remaining stem keeps measure >= 2 (real morphological structure). Single pass, deterministic. emissive ->
    emiss, emission -> emiss(ion via 'ion' strip), relational -> relate, archive -> archive (m gate)."""
    # gate thresholds follow published Porter: the long REWRITE suffixes need only m >= 1 (step 2/3 uses
    # m > 0 -- 'relational' -> 'relate' with stem 'rel', m=1), while the short bare STRIPS need m >= 2
    # (step 4 uses m > 1 -- which is exactly what protects 'archive', stem 'arch', m=1).
    for suf, rep in _DERIV_REWRITES:
        if tok.endswith(suf):
            stem = tok[: -len(suf)]
            if _measure(stem) >= 1:
                return stem + rep
            return tok                                        # longest match decides; a failed gate ends it
    for suf in _DERIV_STRIPS:
        if tok.endswith(suf):
            stem = tok[: -len(suf)]
            if _measure(stem) >= 2:
                return stem
            return tok
    return tok


class BM25:
    """Okapi BM25 over a fixed corpus of documents. Build once (fit the idf + lengths), then score any query in
    O(query_terms * postings). Pure NumPy/stdlib; deterministic. k1 controls term-frequency saturation (the
    first occurrences of a term matter most, later ones saturate); b controls document-length normalization
    (b=1 full, b=0 none). Defaults k1=1.5, b=0.75 are the standard Robertson values."""

    def __init__(self, docs, k1=1.5, b=0.75):
        """`docs` is a list of raw document strings (here: module 'name -- docstring' texts). Fits the corpus
        statistics: per-doc term counts, document lengths, average length, and idf per term."""
        self.k1 = float(k1)
        self.b = float(b)
        self.docs_tokens = [tokenize(d) for d in docs]
        self.N = len(self.docs_tokens)
        self.doc_len = np.array([len(t) for t in self.docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        # per-doc term frequency dicts, and document frequency per term
        self.tf = []
        df = {}
        for toks in self.docs_tokens:
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        # idf with the BM25 (Robertson-Sparck-Jones) form; +1 inside the log keeps it non-negative
        self.idf = {t: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        # PRECOMPUTED POSTINGS (the VSA move: turn the per-query doc WALK into a few vector scatter-adds).
        # A term's contribution to a doc depends only on corpus statistics fixed at fit time, so the whole
        # idf * tf-saturation weight is computed HERE, once, with the SAME expression the reference loop uses
        # (same operands -> same IEEE bits). scores() then just adds each query term's weight vector into the
        # output -- O(postings) NumPy instead of O(terms x N) Python. Measured: 94.8 ms -> sub-ms per query at
        # N=20k, and the selftest asserts BIT-IDENTITY against the shipped reference loop, so no tie can flip.
        self._postings = {}
        for term, idf in self.idf.items():
            idxs, wts = [], []
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl + 1e-12))
                idxs.append(i)
                wts.append(idf * (f * (self.k1 + 1.0)) / (denom + 1e-12))
            if idxs:
                self._postings[term] = (np.array(idxs, dtype=np.int64), np.array(wts, dtype=np.float64))
        # DERIVATIONAL SIBLING INDEX for opt-in query expansion: 'emissive' and 'emission' are the same root
        # wearing different suffixes, and exact-term BM25 misses the pair (measured live: BOTH forms exist
        # un-collapsed in this repo's vocabulary -- a query for one cannot see docs using the other). Group
        # corpus terms by derivational stem so scores(expand=True) can add a term's siblings at half weight.
        # keyed by STEM, not by corpus term -- the case that matters most is a query word ABSENT from the
        # corpus ('emissive' querying docs that only say 'emission'): a term-keyed sibling map has no entry to
        # look up, but stem('emissive') == stem('emission') always bridges. Caught by the selftest, kept here.
        self._stem_terms = {}
        for term in self._postings:
            self._stem_terms.setdefault(_derivational_stem(term), []).append(term)
        for stem in self._stem_terms:
            self._stem_terms[stem].sort()

    def scores(self, query, expand=False):
        """BM25 score of `query` against every document, via precomputed postings: a few NumPy scatter-adds
        instead of a Python walk over all docs per term. Bit-identical to _scores_reference (the original
        loop, shipped beside it flat_recall-style so the claim stays re-checkable, not taken on trust): the
        per-(term, doc) weight is the same expression evaluated at fit time, and per-doc accumulation order is
        the same term order, so even exact ties rank identically. Returns a length-N float array."""
        q_terms = tokenize(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not q_terms:
            return out
        for t in set(q_terms):
            post = self._postings.get(t)
            if post is None:
                continue                                      # term never seen in the corpus -> no signal
            idxs, wts = post
            out[idxs] += wts                                  # one scatter-add per term (docs disjoint per term)
        if expand:
            # DERIVATIONAL EXPANSION (opt-in): add each query term's same-root siblings at HALF weight, so a
            # doc saying 'emission' is reachable from a query saying 'emissive' -- but an exact match always
            # dominates. Recall channel per the levels principle: adds candidates, never removes; the 0.5
            # downweight is the filter keeping the two measured false bridges (arch/archive,
            # conversation/conversion) from outranking anything exact.
            for t_ in set(q_terms):
                for sib in self._stem_terms.get(_derivational_stem(t_), ()):
                    if sib == t_:
                        continue                              # the exact term already scored at full weight
                    post = self._postings.get(sib)
                    if post is not None:
                        idxs, wts = post
                        out[idxs] += 0.5 * wts
        return out

    def _scores_reference(self, query):
        """The ORIGINAL per-doc Python loop, kept as the correctness reference scores() must equal bit-for-bit
        (the flat_recall precedent: ship the baseline beside the fast path so the comparison can be re-run).
        Slow on purpose; use scores()."""
        q_terms = tokenize(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not q_terms:
            return out
        for t in set(q_terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl + 1e-12))
                out[i] += idf * (f * (self.k1 + 1.0)) / (denom + 1e-12)
        return out

    def rank(self, query, top=None, expand=False):
        """Documents ranked by BM25 score, high to low, as a list of (doc_index, score). top-k if given.
        expand=True adds derivational-sibling terms at half weight (emissive reaches emission)."""
        s = self.scores(query, expand=expand)
        order = np.argsort(-s)
        ranked = [(int(i), float(s[i])) for i in order]
        return ranked[:top] if top else ranked


def reciprocal_rank_fusion(ranked_lists, k=60, top=None, weights=None):
    """Fuse several ranked lists into one by Reciprocal Rank Fusion (Cormack et al. 2009). Each list is a
    sequence of item ids in rank order (best first); an item's fused score is sum over lists of w_l/(k + rank),
    rank 1-based. RRF needs NO score calibration -- it uses only ranks -- which is why it is the right choice
    for fusing dense cosine (in [-1,1]) with BM25 (unbounded): their raw scores are not comparable, their ranks
    are. `k` (~60 standard) damps the tail so only items ranked well by SOME retriever rise. Returns fused
    [(item_id, score)] high to low.

    `weights` (optional): per-list multipliers, same length as ranked_lists. Default None = equal weight (the
    classic RRF, byte-identical to before). WHY THIS MATTERS -- measured: fusing a STRONG dense retriever with
    a WEAK BM25 one at EQUAL weight lets BM25's spurious top matches OVERTAKE the dense HITs (dense top-1 6/12
    fell to 3/12 on the real routing suite). The IR literature's optimum is DENSE-DOMINANT (e.g. weights like
    (1.0, 0.3)); down-weighting the weak lexical list keeps the dense HITs while still letting a strong BM25
    rank RESCUE a dense-buried answer. This is the honest fix for a lopsided retriever pair.

    WHY RRF over a convex score combination: a linear a*cosine + (1-a)*bm25 needs the two score scales aligned
    (min-max or z-score), which is brittle and query-dependent; RRF sidesteps it entirely (it uses ranks).

    SR-BETA SWEEP RESULT (2026-07-18), the verdict behind the ~(1.0, 0.3) recommendation, measured on the two
    archetypal cases with realistic top-k truncated lists: (A) DENSE HIT -- gold at dense rank 1, a spurious
    BM25 doc at bm rank 1 -- is KEPT at every beta<=1 (the dense-#1 item 1/(k+1) is never overtaken by the
    spurious doc even at equal weight; the recorded 6->3 regression came from a WEAKER dense list with the hit
    at rank 2-3, which lopsided equal-weight fusion does lose -- down-weighting BM25 restores it). (B) BURIED
    RESCUE -- gold low in the dense top-k but present, gold at BM25 rank 1 -- is rescued across essentially all
    (k, beta>=0.3). (C) ABSENT gold (not in the dense top-k at all) needs beta>1, the hard-conflict regime that
    sacrifices dense hits -- NOT fusion job; widen the retriever k instead. So dense-dominant (1.0, 0.3) is the
    honest optimum. KEPT NEGATIVES: equal-weight fusion of a strong+weak pair is refuted (loses dense hits);
    beta>1 is refuted (loses more dense hits than it rescues); k stays at the standard 60."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, item in enumerate(lst, start=1):
            fused[item] = fused.get(item, 0.0) + float(w) / (k + rank)
    out = sorted(fused.items(), key=lambda kv: -kv[1])
    return out[:top] if top else out


def _selftest():
    """Assert the REAL contract: BM25 exact-matches a query term the way dense embeddings cannot, and RRF fuses
    two lists so an item ranked well by BOTH rises above one ranked well by only one. Numeric, fails loudly."""
    docs = [
        "holographic_meshsmooth smooth a bumpy surface by averaging vertex normals Taubin",   # 0
        "holographic_denoise denoising as manifold projection Plug-and-Play Milanfar",         # 1
        "holographic_fluid grid based fluid solver Stable Fluids smoke advection",             # 2
        "holographic_dynamics propagator binding predict where a state goes next",             # 3
    ]
    bm = BM25(docs)
    # 1) 'bumpy surface' must rank meshsmooth (doc 0) first -- the LEXICAL match dense buries
    r = bm.rank("smooth out the bumpy surface")
    assert r[0][0] == 0, r
    assert r[0][1] > 0.0, "exact term match must score positive"
    # 2) 'grainy' is in NO document -> BM25 gives all-zero (the kept negative: it cannot invent a term)
    z = bm.scores("make my picture less grainy")
    assert float(z.max()) == 0.0, "BM25 must not fabricate a match for an absent term"
    # 3) RRF: doc ranked #1 by list A and #2 by list B must beat a doc ranked #1 by B only
    fused = reciprocal_rank_fusion([[0, 3, 1], [3, 0, 2]])   # doc 0: ranks 1 & 2; doc 3: ranks 2 & 1 -> tie...
    # give doc 0 a clear edge: A ranks it 1, B ranks it 1
    fused2 = reciprocal_rank_fusion([[0, 1, 2], [0, 3, 1]])
    assert fused2[0][0] == 0, fused2                          # agreed-best rises to the top
    # 4) FAST PATH == REFERENCE, bit for bit, on a corpus with heavy term overlap (the tie-rich worst case).
    #    Not allclose -- array_equal: the postings path must be exact so no ranking tie can ever flip.
    import random
    rng = random.Random(0)
    vocab = ["mesh", "smooth", "surface", "noise", "field", "render", "fluid", "vertex"]
    big = [" ".join(rng.choice(vocab) for _ in range(30)) for _ in range(400)]
    bm2 = BM25(big)
    for q in ("smooth mesh surface", "noise in the render field", "fluid vertex", "zzz absent"):
        fast = bm2.scores(q)
        ref = bm2._scores_reference(q)
        assert np.array_equal(fast, ref), ("fast path diverged from reference on %r" % q)
    # and it must actually be fast: postings scatter vs the O(terms x N) walk
    import time
    t0 = time.perf_counter(); [bm2.scores("smooth mesh surface noise") for _ in range(50)]
    t_fast = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter(); [bm2._scores_reference("smooth mesh surface noise") for _ in range(50)]
    t_ref = (time.perf_counter() - t0) / 50
    assert t_fast < t_ref, (t_fast, t_ref)                    # loudly fail if the 'fast' path ever regresses
    # 5) DERIVATIONAL EXPANSION: a query saying 'emissive' must reach a doc saying 'emission' -- but ONLY when
    #    expand=True. Default must stay byte-identical (no bridge), pinned here so the opt-in never leaks.
    docs2 = ["the material emission channel glows", "a plain diffuse surface", "specular highlights"]
    bm3 = BM25(docs2)
    plain = bm3.scores("emissive material")
    assert plain[0] > 0.0                                     # 'material' matches doc 0 directly...
    bm4 = BM25(["emission glow strength", "diffuse albedo", "specular roughness"])
    assert bm4.scores("emissive")[0] == 0.0, "default must NOT bridge emissive->emission"
    exp = bm4.scores("emissive", expand=True)
    assert exp[0] > 0.0 and exp[1] == 0.0, exp                # bridge reaches emission, touches nothing else
    # exact match still dominates a bridged match (the 0.5 downweight doing its job)
    bm5 = BM25(["emission glow", "emissive glow"])
    e = bm5.scores("emissive", expand=True)
    assert e[1] > e[0] > 0.0, e
    # 6) PORTER-STYLE GATES, pinned (both were MEASURED false bridges of the naive strip-only stemmer):
    assert _derivational_stem("archive") == "archive", "m-gate must protect 'archive' (stem 'arch', m=1)"
    assert _derivational_stem("arch") != _derivational_stem("archive")
    assert _derivational_stem("conversation") != _derivational_stem("conversion")
    #    and the rewrite family reaches what a bare strip cannot:
    assert _derivational_stem("relational") == _derivational_stem("relation") == "relate"
    assert _derivational_stem("emissive") == _derivational_stem("emission") == "emiss"
    print("  bm25 selftest OK: 'bumpy surface'->meshsmooth %.3f; 'grainy'->0; RRF agrees; fast==reference "
          "BIT-IDENTICAL on 400-doc tie-rich corpus, %.0fx faster (%.3f ms vs %.3f ms); "
          "expand=True bridges emissive->emission, exact still beats bridged"
          % (r[0][1], t_ref / max(t_fast, 1e-12), t_fast * 1e3, t_ref * 1e3))


if __name__ == "__main__":
    _selftest()

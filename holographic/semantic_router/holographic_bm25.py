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
from collections import Counter
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


def tokenize_once(x):
    """Normalise a string, or pass an ALREADY-NORMALISED token list through untouched.

    WHY THIS EXISTS. `tokenize` is deliberately not idempotent -- 'settings' -> 'setting' -> 'sett',
    'classes' -> 'class' -> 'clas' -- and 2.8% of a real vocabulary changes under a second pass.
    Callers holding tokens used to write `" ".join(toks)` to satisfy a string-only API, which
    re-normalised them and silently over-stemmed the index or the query. That produced a shipped
    page that disagreed with the faculty on 8 of 60 queries, and a benchmark harness whose
    "relative error" was 0.208 when the true error was 1.5e-07.

    Passing tokens is now the supported path, so the join-and-re-tokenise workaround has no reason
    to exist. Anything list-like is taken as final; only a string is normalised.
    """
    if isinstance(x, str):
        return tokenize(x)
    return list(x)


class BM25:
    """Okapi BM25 over a fixed corpus of documents. Build once (fit the idf + lengths), then score any query in
    O(query_terms * postings). Pure NumPy/stdlib; deterministic. k1 controls term-frequency saturation (the
    first occurrences of a term matter most, later ones saturate); b controls document-length normalization
    (b=1 full, b=0 none). Defaults k1=1.5, b=0.75 are the standard Robertson values."""

    def __init__(self, docs, k1=1.5, b=0.75, slim=False, stats=None):
        """`docs` is a list of raw document strings (here: module 'name -- docstring' texts). Fits the corpus
        statistics: per-doc term counts, document lengths, average length, and idf per term."""
        self.k1 = float(k1)
        self.b = float(b)
        # docs may be raw strings OR already-normalised token lists -- see tokenize_once.
        self.docs_tokens = [tokenize_once(d) for d in docs]
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
        # SHARDING SEAM. The per-(term, doc) weight below is baked from idf and avgdl AT FIT TIME,
        # so a shard fitted on its own slice bakes LOCAL statistics and its scores are not
        # comparable with another shard's. Measured on 6,000 documents: naive sharding reached
        # max relative error 0.31 and top-1 agreement 0.76 against a single index, and patching
        # .idf/.avgdl afterwards changed NOTHING because the weights were already baked.
        # `stats` lets a caller fit a shard with the whole corpus's statistics:
        #     stats = {"N": total_docs, "avgdl": corpus_avgdl, "idf": {term: idf}}
        # Absent, behaviour is unchanged. corpus_stats() below produces the dict.
        if stats:
            if "avgdl" in stats:
                self.avgdl = float(stats["avgdl"])
            if "idf" in stats:
                self.idf = dict(stats["idf"])
            self._corpus_N = int(stats.get("N", self.N))
        # PRECOMPUTED POSTINGS (the VSA move: turn the per-query doc WALK into a few vector scatter-adds).
        # A term's contribution to a doc depends only on corpus statistics fixed at fit time, so the whole
        # idf * tf-saturation weight is computed HERE, once, with the SAME expression the reference loop uses
        # (same operands -> same IEEE bits). scores() then just adds each query term's weight vector into the
        # output -- O(postings) NumPy instead of O(terms x N) Python. Measured: 94.8 ms -> sub-ms per query at
        # N=20k, and the selftest asserts BIT-IDENTITY against the shipped reference loop, so no tie can flip.
        #
        # BUILT DOC-MAJOR: one pass over each doc's term counts, appending to per-term lists, O(total tokens).
        # The previous term-major loop (`for term in idf: for i in range(N): tf[i].get(term, 0)`) probed every
        # (term, doc) pair whether or not the term occurs in the doc -- O(vocab x N) -- and on real prose vocab
        # grows with N, so the build was effectively superlinear in corpus size. Measured at BEIR NQ scale
        # (2,681,468 docs, vocab 821,276 under this file's own tokenize): 2.2e12 probes, build did not complete;
        # the doc-major reorder finished in 309.8 s including tokenization. The postings are IDENTICAL by
        # construction: same idf, same weight expression with the same operands (so the same IEEE bits), and
        # ascending doc order per term either way -- asserted bit-for-bit in tests/test_bm25_docmajor_build.py.
        post = {}
        for i in range(self.N):
            dl = self.doc_len[i]
            for term, f in self.tf[i].items():
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl + 1e-12))
                lists = post.get(term)
                if lists is None:
                    lists = post[term] = ([], [])
                lists[0].append(i)
                lists[1].append(self.idf[term] * (f * (self.k1 + 1.0)) / (denom + 1e-12))
        self._postings = {term: (np.array(idxs, dtype=np.int64), np.array(wts, dtype=np.float64))
                          for term, (idxs, wts) in post.items()}
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
        # SLIM MODE (default off; stacc's PR #32 note: retaining docs_tokens + tf cost ~15 GB at 2.7M docs).
        # THE LEVER, not the axe: his XL build DROPPED them and lost _scores_reference -- the bit-identity
        # oracle this file's whole verification story rests on. leCore already has the escape: cold_store
        # (the tiered-memory spill move) PARKS them zlib-compressed and inflates on demand, so scoring pays
        # nothing and the reference loop still runs, just slower on first touch. MEASURED on 9,723 real-prose
        # paragraphs from this repo's own docs (not synthetic): 9.34 MB live -> parked, ~3.6x smaller at
        # rest; scores() untouched (never reads either); _scores_reference inflates transparently and stays
        # bit-identical (asserted in the selftest ON slim mode, real prose).
        self._cold = None
        if slim:
            from holographic.caching_and_storage.holographic_coldstore import ColdStore
            self._cold = ColdStore(keep_warm=0, codec="zlib")
            self._cold.put("tf", self.tf)
            self._cold.put("docs_tokens", self.docs_tokens)
            self.tf = None
            self.docs_tokens = None

    def _corpus_stats(self):
        """(tf, docs_tokens), inflating from cold storage in slim mode. The reference loop's door."""
        if self._cold is None:
            return self.tf, self.docs_tokens
        return self._cold.get("tf"), self._cold.get("docs_tokens")

    def corpus_stats(self):
        """The statistics a SHARD must be fitted with to stay comparable: N, avgdl and idf.

        Fit the shards with this and their scores live on one scale, so merging their top-k lists
        is exact -- which is what T4 (tiled_max_eq_global) already promises for the merge itself.
        Without it the merge silently ranks by which shard a document happened to land in.
        """
        return {"N": self.N, "avgdl": self.avgdl, "idf": dict(self.idf)}

    def scores(self, query, expand=False):
        # `query` may be a string or an already-normalised token list; see tokenize_once.
        """BM25 score of `query` against every document, via precomputed postings: a few NumPy scatter-adds
        instead of a Python walk over all docs per term. Bit-identical to _scores_reference (the original
        loop, shipped beside it flat_recall-style so the claim stays re-checkable, not taken on trust): the
        per-(term, doc) weight is the same expression evaluated at fit time, and per-doc accumulation order is
        the same term order, so even exact ties rank identically. Returns a length-N float array.
        QUERY-SIDE TERM FREQUENCY (PR #33, stacc): a term occurring c times in the query contributes c x its
        per-doc weight -- Okapi's qtf factor with k3 -> inf, what a reference loop over the raw token list
        computes. Deduping is invisible on keyword queries (six BEIR tasks, repeat rates 0.003-0.028, every
        delta < 0.002) but on ArguAna's passage queries (121.6 mean tokens, 0.230 repeat rate) it discards
        real signal: +5.7 nDCG@10. Counter is insertion-ordered, so accumulation order is also deterministic
        without a hashseed pin -- set() iteration was not, a latent determinism hole this closes."""
        q_terms = tokenize_once(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not q_terms:
            return out
        for t, c in Counter(q_terms).items():
            post = self._postings.get(t)
            if post is None:
                continue                                      # term never seen in the corpus -> no signal
            idxs, wts = post
            out[idxs] += float(c) * wts                       # one scatter-add per term (docs disjoint per term)
        if expand:
            # DERIVATIONAL EXPANSION (opt-in): add each query term's same-root siblings at HALF weight, so a
            # doc saying 'emission' is reachable from a query saying 'emissive' -- but an exact match always
            # dominates. Recall channel per the levels principle: adds candidates, never removes; the 0.5
            # downweight is the filter keeping the two measured false bridges (arch/archive,
            # conversation/conversion) from outranking anything exact.
            for t_, c in Counter(q_terms).items():
                for sib in self._stem_terms.get(_derivational_stem(t_), ()):
                    if sib == t_:
                        continue                              # the exact term already scored at full weight
                    post = self._postings.get(sib)
                    if post is not None:
                        idxs, wts = post
                        out[idxs] += 0.5 * float(c) * wts
        return out

    def _scores_reference(self, query):
        """The ORIGINAL per-doc Python loop, kept as the correctness reference scores() must equal bit-for-bit
        (the flat_recall precedent: ship the baseline beside the fast path so the comparison can be re-run).
        Slow on purpose; use scores(). Counts query terms (Counter) to match scores() -- see its docstring."""
        q_terms = tokenize_once(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not q_terms:
            return out
        tf, _ = self._corpus_stats()                          # inflates from cold storage in slim mode
        for t, c in Counter(q_terms).items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in range(self.N):
                f = tf[i].get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl + 1e-12))
                out[i] += float(c) * (idf * (f * (self.k1 + 1.0)) / (denom + 1e-12))
        return out

    def rank(self, query, top=None, expand=False):
        """Documents ranked by BM25 score, high to low, as a list of (doc_index, score). top-k if given.
        expand=True adds derivational-sibling terms at half weight (emissive reaches emission).
        TIES: ordered by ascending doc index, deterministically. The previous np.argsort used an UNSTABLE
        quicksort, so equal-score order was unspecified (numpy-version dependent) -- the same latent
        determinism hole class as scores()'s old set() iteration, closed the same release."""
        s = self.scores(query, expand=expand)
        # TOP-K SHORTLIST (stacc's PR #32 note: full argsort at 2.7M docs cost ~0.3s/query): with `top`
        # given, argpartition shortlists in O(N) and only the shortlist is sorted. Tie-break matches the
        # full sort exactly -- both order by (-score, ascending index) -- pinned in the selftest against
        # the full argsort kept as reference. top=None still returns the complete ranking, full sort.
        if top and top < len(s):
            # KEPT NEGATIVE (caught live on discrete scores): a k+1 shortlist is WRONG under ties AT the
            # k-th score -- argpartition guarantees the top-k VALUES, but which tied items fill the
            # boundary slots is arbitrary, so the ascending-index tie contract silently broke (measured:
            # 10 docs tied at rank 10; the shortlist kept index 133435 and dropped 19999). The exact rule:
            # include EVERYTHING >= the k-th value, then stable-sort that shortlist. Ties are bounded in
            # practice, so this stays ~O(N + t log t).
            # DELEGATED (F17): the boundary rule above now lives ONCE in
            # holographic_determinism.topk_det -- bit-identical, pinned by the planted-tie test below.
            from holographic.misc.holographic_determinism import topk_det
            order = topk_det(s, top)
        else:
            order = np.lexsort((np.arange(len(s)), -s))[:top] if top else np.lexsort((np.arange(len(s)), -s))
        ranked = [(int(i), float(s[i])) for i in order]
        return ranked


def prf_expand(query, docs, top_f=3, top_t=8, k1=1.5, b=0.75, bm=None):
    """Harvest expansion terms by PSEUDO-RELEVANCE FEEDBACK (Rocchio 1971 / RM3): run BM25 once,
    treat the top `top_f` docs AS IF relevant, and pick their best `top_t` terms by
    count-in-feedback x idf -- query terms excluded, because re-adding what the query already said
    only re-weights it (the benchmark's phase-8 rule, lifted verbatim). Deterministic term order:
    ties break alphabetically, the same (-weight, term) sort the measured 0.3442 nDCG run used.
    Returns {"terms": [...], "feedback": [doc indices], "ranked": bm25 first-pass ranking} so the
    caller can rescore without a second fit. `bm` lets a caller reuse a fitted BM25 (a corpus fit
    is the expensive half); passing docs alone fits one here.
    KEPT NEG: PRF cannot rescue gold OUTSIDE the first pass -- feedback docs are the horizon."""
    bm = bm if bm is not None else BM25(docs, k1=k1, b=b)
    ranked = bm.rank(query, top=None)
    # A doc scoring ZERO shares no term with the query: the first pass says nothing about it,
    # so "pseudo-relevant" cannot stretch to cover it. On the measured benchmark shape (200+
    # docs, small F) this filter is a no-op -- it exists for small corpora, where top_f would
    # otherwise sweep off-topic docs into feedback and amplify them.
    fb = [i for i, sc in ranked[:int(top_f)] if sc > 0.0]
    q_terms = set(tokenize_once(query))
    cnt = {}
    for i in fb:
        for t in bm.docs_tokens[i]:
            if t not in q_terms:
                cnt[t] = cnt.get(t, 0) + 1
    idf = getattr(bm, "idf", {})
    terms = sorted(cnt, key=lambda t: (-cnt[t] * float(idf.get(t, 0.0)), t))[:int(top_t)]
    return {"terms": terms, "feedback": fb, "ranked": ranked}


def prf_rank(query, docs, alpha=0.3, top_f=3, top_t=8, k1=1.5, b=0.75, top=None):
    """PSEUDO-RELEVANCE FEEDBACK re-ranking (Rocchio 1971 / RM3): a second bounce where the first
    pass's top docs relight the query. Pure counting, zero learned weights, zero model calls.
    MEASURED (benchmarks/beir phase 8, test-once): NFCorpus nDCG@10 0.3371 -> 0.3442.

    ALPHA=0 IS BIT-IDENTICAL to BM25(docs).rank(query) BY CONSTRUCTION: the second pass is never
    run, the first-pass ranking is returned unchanged -- opt-in, not a silent default shift.
    alpha>0 min-normalizes both passes and interpolates (1-alpha)*first + alpha*second, the exact
    phase-8 formula, with the deterministic topk tie rule riding through BM25.rank.

    Returns {"ranked": [(doc_index, score)...], "expansion": [terms], "alpha", "feedback"}.
    KEPT NEG: cannot rescue gold OUTSIDE the first pass; a corpus where the top-F docs are all
    off-topic makes the expansion off-topic too -- PRF amplifies the first pass, right or wrong."""
    a = float(alpha)
    bm = BM25(docs, k1=k1, b=b)
    if a <= 0.0:
        # The identity contract: no second scoring pass exists to perturb ties or floats.
        return {"ranked": bm.rank(query, top=top), "expansion": [], "alpha": 0.0, "feedback": []}
    ex = prf_expand(query, docs, top_f=top_f, top_t=top_t, k1=k1, b=b, bm=bm)
    s1 = np.array([sc for _, sc in sorted(ex["ranked"], key=lambda p: p[0])], dtype=np.float64)
    # Second pass: the expanded query is original terms + harvested terms (token-level, so the
    # non-idempotent tokenize trap pinned in _selftest cannot bite -- we never re-join to a string).
    q2 = tokenize_once(query) + list(ex["terms"])
    s2 = np.asarray(bm.scores(q2), dtype=np.float64)
    r1 = float(s1.max() - s1.min()); r2 = float(s2.max() - s2.min())
    s = (1.0 - a) * (s1 - s1.min()) / (r1 if r1 > 0 else 1.0) \
        + a * (s2 - s2.min()) / (r2 if r2 > 0 else 1.0)
    from holographic.misc.holographic_determinism import topk_det
    order = topk_det(s, top) if top else np.lexsort((np.arange(len(s)), -s))
    ranked = [(int(i), float(s[i])) for i in order]
    return {"ranked": ranked, "expansion": ex["terms"], "alpha": a, "feedback": ex["feedback"]}


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

    # PRF, three planted truths.
    _pd = ["laplacian mesh smoothing averages vertex positions",
           "taubin smoothing for meshes without shrinkage",
           "fluid solver pressure projection on a grid",
           "smoothing kernels and mesh fairing energies"]
    # 1) ALPHA=0 IDENTITY, bit-for-bit: no second pass may exist to perturb a float or a tie.
    _r0 = prf_rank("mesh smoothing", _pd, alpha=0.0)
    assert _r0["ranked"] == BM25(_pd).rank("mesh smoothing") and _r0["expansion"] == [], \
        "alpha=0 must be BIT-IDENTICAL to bm25 rank -- PRF is opt-in by construction"
    # 2) Expansion harvests from FEEDBACK docs only, zero-score docs excluded: the off-topic
    #    fluid doc scores 0.0 on this query, so its vocabulary must not leak into the expansion.
    _rx = prf_rank("mesh smoothing", _pd, alpha=0.5, top_f=3)
    assert _rx["expansion"] and all(t not in ("fluid", "solver", "pressure") for t in _rx["expansion"]), \
        "zero-score docs leaked into feedback: %r" % (_rx["expansion"],)
    # 3) KEPT NEGATIVE pinned: gold OUTSIDE the first pass is not rescued. A doc sharing no term
    #    with query OR expansion stays at score 0 whatever alpha does.
    assert all(sc == 0.0 for i, sc in _rx["ranked"] if _pd[i].startswith("fluid")), \
        "PRF must not invent relevance for a doc the first pass never touched"

    # SHARDING, pinned in both directions. Shards fitted WITH the corpus statistics must be
    # BIT-IDENTICAL to a single index -- that is what makes merging their top-k lists exact, which
    # T4 already promises for the merge itself. Shards fitted WITHOUT must still be visibly wrong,
    # or this seam is doing nothing: measured on 6,000 documents, naive sharding reached max
    # relative error 0.31-0.51 and top-1 agreement 0.76-0.84.
    _sd = [tokenize(t) for t in
           ["smooth a bumpy surface", "a fluid solver on a torus", "holographic memory recall",
            "bumpy surface normals", "recall from a noisy cue", "torus fluid pressure"]]
    _one = BM25(_sd)
    _st = _one.corpus_stats()
    _q = tokenize("bumpy surface recall")
    _ref = _one.scores(_q)
    _with = np.concatenate([BM25(_sd[:3], stats=_st).scores(_q), BM25(_sd[3:], stats=_st).scores(_q)])
    _without = np.concatenate([BM25(_sd[:3]).scores(_q), BM25(_sd[3:]).scores(_q)])
    assert np.array_equal(_with, _ref), "shards fitted with corpus stats must be BIT-IDENTICAL"
    assert not np.allclose(_without, _ref), (
        "shards fitted WITHOUT corpus stats must still differ -- if they stopped differing, the "
        "weights are no longer baked at fit time and this seam needs re-measuring")

    # DOUBLE-TOKENISATION TRAP, pinned in both directions. tokenize is NOT idempotent, so a caller
    # that holds tokens and joins them back into a string gets a DIFFERENT index than one that
    # passes the tokens. That has now cost this project three separate bugs, so both the trap and
    # the fix are asserted here rather than described in a comment somewhere.
    _toks = [tokenize("the settings of these classes"), tokenize("a process for meshing surfaces")]
    _joined = BM25([" ".join(t) for t in _toks])
    _direct = BM25(_toks)
    assert _direct.docs_tokens == _toks, "passing tokens must not re-normalise them"
    assert _joined.docs_tokens != _toks, (
        "tokenize became idempotent -- that is a BEHAVIOUR CHANGE to be re-measured, not a bug "
        "fix; see the P1.11 measurement before adopting it")
    _q = tokenize("classes setting")
    assert list(_direct.scores(_q)) == list(_direct.scores(_q)), "scores must be deterministic"
    assert not np.allclose(_direct.scores(_q), _joined.scores(" ".join(_q))), (
        "the join-and-re-tokenise path must remain visibly DIFFERENT, or this pin is asleep")
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
    # 7) SLIM MODE on REAL PROSE (this repo's own docs -- the corpus register BM25 actually serves;
    #    per the test-data rule, a compression/retention claim is only meaningful on genuine text):
    #    slim scores == full scores bitwise, the reference oracle SURVIVES parking (inflates from cold
    #    storage), and the parked stats are measurably smaller than live.
    import pickle, zlib as _z
    real = [p_.strip() for p_ in open("docs/NOTES_concepts.md").read().split("\n\n") if len(p_.strip()) > 80][:1500]
    full_b, slim_b = BM25(real), BM25(real, slim=True)
    for q_ in ("kept negative measured baseline", "capability catalog aliases", "forest recall regression"):
        assert np.array_equal(full_b.scores(q_), slim_b.scores(q_)), "slim changed scores"
        assert np.array_equal(slim_b.scores(q_), slim_b._scores_reference(q_)), "reference broken in slim mode"
    live = len(pickle.dumps(full_b.tf)) + len(pickle.dumps(full_b.docs_tokens))
    parked = sum(len(_z.compress(pickle.dumps(v))) for v in
                 (slim_b._cold.get("tf"), slim_b._cold.get("docs_tokens")))
    assert slim_b.tf is None and slim_b.docs_tokens is None
    assert parked < live * 0.5, f"parking must at least halve the stats ({parked} vs {live})"

    # 7b) RANK TOP-K SHORTLIST == deterministic full ranking prefix, on the tie-rich corpus (400 docs, many
    #    exact score ties): the argpartition path must reproduce the (-score, ascending index) full order
    #    exactly, for several k. The full lexsort is the in-test reference (flat_recall pattern again).
    for q_ in ("alpha common", "beta", "alpha alpha beta"):
        s_ = bm.scores(q_)
        full = list(np.lexsort((np.arange(len(s_)), -s_)))
        for k_ in (1, 5, 37):
            assert [i for i, _ in bm.rank(q_, top=k_)] == [int(j) for j in full[:k_]], (q_, k_)

    print("  bm25 selftest OK: 'bumpy surface'->meshsmooth %.3f; 'grainy'->0; RRF agrees; fast==reference "
          "BIT-IDENTICAL on 400-doc tie-rich corpus, %.0fx faster (%.3f ms vs %.3f ms); "
          "expand=True bridges emissive->emission, exact still beats bridged"
          % (r[0][1], t_ref / max(t_fast, 1e-12), t_fast * 1e3, t_ref * 1e3))


if __name__ == "__main__":
    _selftest()

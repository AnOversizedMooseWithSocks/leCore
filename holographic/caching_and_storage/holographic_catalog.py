"""holographic_catalog.py -- the capability CATALOG (consolidation backlog C1): "search before you build".

WHY THIS EXISTS
---------------
The engine has grown to ~340 modules, and the recurring cost is DUPLICATION: the same job (search a pile of
vectors, bake a slow factor and look it up, represent a field) is often already solved somewhere, but a new
session -- human or AI -- can't easily find it and builds a fourth copy. This catalog is the index of what exists.
Describe a problem in plain English and it points you at the home that already does it.

Each entry says, in plain English, what it DOES, gives a copy-paste EXAMPLE, and flags whether it is NATIVE
(stays in the batched / fusable vector domain) or hops to Python. `find_capability` runs a small, READABLE
token-overlap match over the entries -- no training, fully deterministic -- so "search a big pile of vectors"
finds the search index without anyone having to know its module name.

This PROMOTES two shipped things into one home: the auto-listing of the mind's public faculties
(holographic_query.capability_registry, which builds a SQL-queryable table) and query-by-description. The catalog
is the richer home (does/example/native + find); `seed_from_mind` reuses the faculty walk so every faculty is
findable, and `to_rows` can still hand the entries to the SQL table path. NumPy-free, stdlib only.
"""
import re

# small stop-word list so a problem sentence matches on its CONTENT words, not "how do I ..." scaffolding
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "how", "do", "i", "my", "is", "are",
    "it", "that", "this", "at", "by", "as", "be", "can", "want", "need", "get", "from", "into", "over", "some",
    "you", "me", "we", "our", "your", "using", "use", "make", "build", "create", "given", "when", "where", "what",
}


#: Leading conversational scaffolding a person types before the actual request. Stripped from the QUERY
#: PHRASE before the exact-alias check -- see _strip_filler for why the stop-word list is not enough.
_FILLER_PREFIXES = (
    "what's the best way to", "whats the best way to", "what is the best way to",
    "can you help me", "could you help me", "can you", "could you",
    "how do i", "how do you", "how can i", "how would i",
    "i want to", "i need to", "i would like to", "i'd like to",
    "please", "help me",
)


def _strip_filler(text):
    """Remove leading conversational prefixes from a query, repeatedly, until none matches.

    WHY THIS IS NOT REDUNDANT WITH _STOP, which is the obvious objection: the stop-word list ALREADY drops
    'how', 'do', 'i', so `_tokens('how do i smooth a bumpy mesh')` and `_tokens('smooth a bumpy mesh')` are
    ALREADY identical. Stripping fillers to help the TOKEN match would therefore be pure ceremony.

    What actually breaks is the EXACT-ALIAS BONUS below, which is a whole-string equality test worth +5.0 --
    by far the largest term in the score. Any prefix at all, even one whose every token is a stop word, makes
    `q_phrase != alias` and silently forfeits it. MEASURED over 150 author aliases (>=4 words): exact phrasing
    top-1 99.3%, and EVERY fully-stopped filler lands on exactly 91.3% -- identical, because they all destroy
    the same bonus and nothing else. ("what's the best way to" is worse at 81.3%: it also leaks the content
    words 'best' and 'way', which add spurious overlap elsewhere.)

    So the fix has to reach the PHRASE, not the tokens. A change that only touched tokenization would ship
    green and move nothing.

    Returns the normalised, stripped phrase. If stripping would consume the whole query, the original is kept
    -- 'please' alone is a query about nothing, but it is not improved by becoming the empty string."""
    s = " ".join((text or "").lower().split())
    changed = True
    while changed:
        changed = False
        for p in _FILLER_PREFIXES:
            if s.startswith(p + " "):
                rest = s[len(p) + 1:].strip()
                if rest:                       # never strip away the entire query
                    s, changed = rest, True
                    break
    return s


def _tokens(text):
    """Lower-cased content words of `text` (drop stop-words and 1-char tokens). Readable and deterministic."""
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP and len(w) > 1]


class Capability:
    """One catalog entry: a named home, what it DOES (plain English), a copy-paste EXAMPLE, whether it is NATIVE
    (True = batched / fusable / stays in the vector domain; False = crosses to Python), search `aliases`
    (extra words a problem might use for it -- e.g. 'knn', 'lookup' for the search index), and an optional
    `semantic` action path (the File->Export->PNG verb hierarchy -- e.g. 'transform/rotate', 'select/loop'). The
    semantic path is orthogonal to the physical location URI: it groups a capability by what a USER does, not by
    which module the code lives in. Default None -> the capability falls back to its location URI for grouping."""

    def __init__(self, name, does, example="", native=True, aliases=(), semantic=None, consumes=(), produces=(),
                 module=None, method=None, polymorphic=False):
        self.name = str(name)
        # C6: SAME KIND IN -> SAME KIND OUT. `consumes`/`produces` are read as a CROSS PRODUCT by the edge
        # builder, which is right for a CONJUNCTIVE capability (transform_selection needs mesh AND selection AND
        # transform, so "I hold a selection, can I reach a mesh?" is a real question), but WRONG for a
        # POLYMORPHIC one: denoise_tensor takes an image OR a field and hands back THE SAME KIND, so the cross
        # product invents `image->field` and `field->image` edges it can never perform. That fake hop is exactly
        # how a downstream audit's `suggest_pipeline("image","mesh")` escaped into field-space and returned a
        # tensor denoiser followed by an Aharonov-Bohm ring. Setting this keeps only the DIAGONAL edges.
        # Default False = the old cross-product behaviour, so no existing edge moves unless a tag opts in.
        self.polymorphic = bool(polymorphic)
        # C7: the CALLABLE name as structured data -- `mind.<method>(...)` -- or None when this capability is
        # import-only (needs a direct class import; e.g. holographic_spatial.knn). A downstream node pack was
        # REGEXING `m.foo(` out of the example string to recover it, "exactly as fragile as it sounds" (4 entries
        # in their EXCLUDED.md prove it). Deriving it ONCE, here, kills that regex for every client at once --
        # and the None case IS the honest answer to "is this thing callable?" (their item 8), so one field
        # answers both questions instead of two fields that could disagree.
        self.method = str(method) if method else None
        self.does = str(does)
        self.example = str(example)
        self.native = bool(native)
        self.aliases = tuple(aliases)
        # BAKED AT BIRTH. The lazy bake left a staleness window: a capability
        # registered AFTER _ensure_fc_baked ran had no token sets and crashed the
        # scorer (caught by this module's own selftest registering 'MyThing' late).
        # Token sets are pure functions of the constructor arguments, so the only
        # correct home is construction itself -- no window, no flag, no ordering.
        self._nw = set(_tokens(self.name))
        self._hay = self._nw | set(_tokens(self.does)) | _alias_tokens(self.aliases)
        self._al = tuple(a.lower() for a in self.aliases)
        self.semantic = str(semantic) if semantic else None
        self.module = str(module) if module else None
        # io-shape kinds (S3): what datatype(s) this capability takes / returns, from the closed IO_KINDS vocabulary.
        # Validated so a typo is caught here, not silently at pipeline time. Empty = unspecified ('always shown').
        from holographic.caching_and_storage.holographic_iokinds import validate_kinds
        self.consumes = validate_kinds(consumes, where="consumes of %r" % name)
        self.produces = validate_kinds(produces, where="produces of %r" % name)

    def resolved_module(self):
        """The code module this capability lives in -- explicit `module=` if set, else DERIVED from the
        `holographic_X` reference in its does/example text. WHY: only ~18 of ~390 registrations set module=
        by hand, but almost all NAME their module in the docstring ('...holographic_meshsmooth...'). Deriving
        it deterministically completes the module->capability join that pipeline_map and the io-graph need,
        WITHOUT hand-editing hundreds of calls (unreviewable, error-prone). Picks the most-referenced module
        name in the text (a capability may mention several; its OWN module is the one it names most). Returns
        the bare stem (e.g. 'meshsmooth') or None if nothing is referenced -- an honest None, never a guess."""
        if self.module:
            return self.module
        import re as _re, collections as _collections
        refs = _re.findall(r"holographic_([a-z0-9_]+)", (self.does or "") + " " + (self.example or ""))
        if not refs:
            return None
        return _collections.Counter(refs).most_common(1)[0][0]

    def __repr__(self):
        return "Capability(%r, %s)" % (self.name, "native" if self.native else "python")


def _alias_tokens(aliases):
    """Every alias contributes BOTH its whole lowercased phrase (so an exact single-token alias still matches)
    and its individual content words. Without the tokenization a multi-word alias is dead weight -- see
    Catalog.find_capability's docstring for the measurement (55.5% of this repo's aliases were inert)."""
    out = set()
    for a in aliases:
        a = a.lower()
        out.add(a)
        out.update(_tokens(a))
    return out


class Catalog:
    """The registry + the search. `register_capability` adds/updates an entry; `find_capability` ranks entries
    against a problem description by how many content words they share (favouring entries that cover the query)."""

    def __init__(self):
        self._by_name = {}                                           # name -> Capability (insertion order kept)

    def register_capability(self, name, does, example="", native=True, aliases=(), semantic=None,
                            consumes=(), produces=(), module=None, method=None, polymorphic=False):
        """Add (or replace) a capability. Returns the entry. Additive -- registering the same name again updates it.
        `semantic` (optional) is the File->Export->PNG verb path, e.g. 'transform/rotate'. `consumes`/`produces`
        (optional, S3) are tuples of io kinds (holographic_iokinds) declaring the datatype(s) this takes/returns --
        validated against the closed vocabulary, empty = unspecified. All default off -> byte-identical old entries.
        `method` (optional, C7) is the callable faculty name; when omitted it is DERIVED from the name or the
        example, and stays None for import-only capabilities."""
        cap = Capability(name, does, example, native, aliases, semantic=semantic, consumes=consumes,
                         produces=produces, module=module, method=method or _derive_method(name, example),
                         polymorphic=polymorphic)
        self._by_name[name] = cap
        return cap

    def get(self, name):
        return self._by_name.get(name)

    def all(self):
        return list(self._by_name.values())

    def __len__(self):
        return len(self._by_name)


    def _ensure_fc_baked(self):
        """Shared bake for BOTH scoring paths. The audit that forced this: the bake
        lived inline in find_capability only, so route_or_abstain's 64 null queries
        went through find_scored at unbaked ~55ms each -- 3.5s per first routing call,
        invisible to every find_capability benchmark. One bake, every path."""
        if hasattr(self, "_fc_baked"):
            return
        for cap in self._by_name.values():
            cap._nw = set(_tokens(cap.name))
            cap._hay = cap._nw | set(_tokens(cap.does)) | _alias_tokens(cap.aliases)
            cap._al = tuple(a.lower() for a in cap.aliases)
        import hashlib as _hl
        blob = "\x1f".join(n + c.does + "\x1e".join(c.aliases)
                           for n, c in sorted(self._by_name.items()))
        self._fc_hash = _hl.sha256(blob.encode()).hexdigest()[:16]
        import json as _js, os as _os, tempfile as _tf
        self._fc_memo_path = _os.path.join(_tf.gettempdir(),
                                           "lecore_fc_%s.json" % self._fc_hash)
        try:
            self._fc_memo = _js.load(open(self._fc_memo_path))
        except Exception:
            self._fc_memo = {}
        # catalog vocab for the null router, tokenised ONCE per catalog state
        self._fc_vocab = sorted(set(t for cap in self._by_name.values()
                                    for t in cap._hay))
        # Disk-warmed null floors. The NULL ARRAY IS PERSISTED TOO, so a warm boot keeps the empirical p.
        #
        # BUG THIS FIXES: the memo used to store [mu, sd] only, on the reasoning that an unavailable p is
        # more honest than a fabricated one. True as far as it went -- but it made `p` depend on whether a
        # /tmp file happened to exist, so the SAME query returned p=0.0154 cold and p=None warm. Under
        # pytest-xdist the workers share /tmp, so whichever ran second saw None and six tests failed in CI
        # while passing locally. Persisting the null array is not fabrication: it IS the measured data, and
        # it is 64 floats. Honesty was never the thing forcing p to vanish; an incomplete cache format was.
        # A 2-list is still tolerated so an older memo file cannot raise -- it just yields p=None as before.
        for mk, v in list(self._fc_memo.items()):
            if mk.startswith("~null|") and isinstance(v, list) and len(v) in (2, 3):
                self._null_floor_cache = getattr(self, "_null_floor_cache", {})
                tc, nn, sd_ = mk.split("|")[1:4]
                if len(v) == 2:
                    entry = (v[0], v[1])
                else:
                    import numpy as _np
                    entry = (v[0], v[1], _np.asarray(v[2], float))
                self._null_floor_cache[(int(tc), int(nn), int(sd_), len(self._by_name))] = entry
        self._fc_baked = True

    def find_capability(self, problem, k=3, accepts=None, produces=None):
        """Return up to `k` capabilities whose description best matches `problem`, best first. The score is the
        number of shared content words, normalised by the query length so a short query isn't swamped -- plus a
        small bonus when a query word appears in the entry's NAME (a strong signal). Deterministic ties break by
        name so the result is stable run-to-run (the engine's determinism rule).

        S3 io-shape filter: `accepts` (a kind) keeps only capabilities that CONSUME that kind -- "what can I run on
        this mesh?" -> accepts='mesh'. `produces` keeps only those that PRODUCE that kind. A capability with NO
        consumes/produces tag is unspecified and is NEVER filtered out (tagging is additive; untagged stays shown).

        Aliases are TOKENIZED into the haystack, not matched as whole strings. That was a real bug, measured: the
        haystack used to take each alias as one lowercased phrase, so a multi-word alias could only ever match a
        query that was that exact single token -- i.e. never. 827 of the repo's 1,489 aliases (55.5%) contained a
        space and were therefore INERT, including every alias written the way the build loop asks for them ("a
        phrase a stranger would type"). The whole phrase is still kept as a token, so exact-phrase hits are
        unaffected; tokens are added, never removed, so no previously-findable capability became unfindable."""
        q = set(_tokens(problem))
        if not q:
            return []
        # BAKED HAYSTACKS + CROSS-SESSION MEMO. Measured before building: 50ms/query
        # was 164 capabilities x re-tokenising name/does/aliases on EVERY call; the
        # token sets are pure functions of registration-time strings, so they are
        # computed once and stored on the instance. The memo persists find_capability
        # results across sessions keyed by a hash of the catalog CONTENT (names+does+
        # aliases), because Rule-0 bootstrap rituals re-ask the same phrasings every
        # session; a catalog edit changes the hash and silently retires stale entries.
        self._ensure_fc_baked()
        if False and not hasattr(self, "_fc_baked"):
            for cap in self._by_name.values():
                cap._nw = set(_tokens(cap.name))
                cap._hay = cap._nw | set(_tokens(cap.does)) | _alias_tokens(cap.aliases)
                cap._al = tuple(a.lower() for a in cap.aliases)
            import hashlib as _hl
            blob = "\x1f".join(n + c.does + "\x1e".join(c.aliases)
                               for n, c in sorted(self._by_name.items()))
            self._fc_hash = _hl.sha256(blob.encode()).hexdigest()[:16]
            import json as _js, os as _os, tempfile as _tf
            self._fc_memo_path = _os.path.join(_tf.gettempdir(),
                                               "lecore_fc_%s.json" % self._fc_hash)
            try:
                self._fc_memo = _js.load(open(self._fc_memo_path))
            except Exception:
                self._fc_memo = {}
            self._fc_baked = True
        mk = "%s|%d|%s|%s" % (problem, k, accepts, produces)
        if mk in self._fc_memo:
            hit = [self._by_name[n] for n in self._fc_memo[mk] if n in self._by_name]
            if len(hit) == len(self._fc_memo[mk]):
                return hit
        q_phrase = " ".join((problem or "").lower().split())        # normalised whole query, for exact-alias hits
        # FILLER-STRIPPED FORM, restored: the split kept q_phrase but dropped this and the alias comparison
        # that used it, silently deleting the fix. BOTH forms are tested, never just the stripped one --
        # four shipped aliases THEMSELVES begin with a filler ("how do I get from points to a mesh"), so
        # stripping the query alone would destroy the very +5.0 bonus this exists to protect. Testing both
        # can only ADD a match, never remove one.
        q_stripped = _strip_filler(problem)
        scored = []
        for cap in self._by_name.values():
            # S3 pre-filter: skip a capability whose declared shape is incompatible. Untagged (empty) = always shown.
            if accepts is not None and cap.consumes and accepts not in cap.consumes:
                continue
            if produces is not None and cap.produces and produces not in cap.produces:
                continue
            name_words = cap._nw
            hay = cap._hay
            overlap = len(q & hay)
            if overlap == 0:
                continue
            score = overlap + 0.5 * len(q & name_words)              # a name-word hit counts extra
            # EXACT-ALIAS bonus: if the whole query IS one of this cap's aliases, that is the strongest possible
            # signal -- a stranger typed the exact phrase we anticipated. Without this, a cap with the exact alias
            # ties with siblings that merely scatter the same words across their prose, and the alphabetical
            # tie-break can bury it below k (measured: ascii_view, exact alias 'render image to terminal', lost to
            # ascii_animate/field/sdf all tied at 3.0). Additive -- only raises exact matches, never demotes.
            if any(q_phrase == a or q_stripped == a for a in cap._al):
                score += 5.0
            scored.append((score, cap.name, cap))
        scored.sort(key=lambda s: (-s[0], s[1]))                     # best score first, then name (stable)
        out = [cap for _, _, cap in scored[:k]]
        self._fc_memo[mk] = [c.name for c in out]
        if len(self._fc_memo) % 32 == 1:                             # amortised, atomic-enough
            try:
                import json as _js
                _js.dump(self._fc_memo, open(self._fc_memo_path, "w"))
            except Exception:
                pass
        return out

    def route_or_abstain(self, problem, k=3, n_null=64, z_min=0.8, seed=0):
        """J1 -- NULL-REFERENCED ROUTING: find_capability that can say "no capability matches" instead of
        returning its argmax on noise. The router failure this closes, from the campaign's own logs:
        'counter traders' confidently routed to dialect emitters and 'purple monkey dishwasher' to an
        opponent-agreement tool -- the renko lesson (machinery manufactures confident-looking output on
        nothing) applied to retrieval.

        THE NULL: n_null scrambled queries with the SAME token count as the real one, each token drawn
        uniformly from the CATALOG'S OWN vocabulary. In-domain word salad is exactly what a misroutable
        query is, so the null must be built from in-domain words -- a null of out-of-vocabulary gibberish
        ('flurb granp') scores 0 by construction and gates nothing (the null that cannot fire is
        decoration; measured before choosing this one). The top-1 match score becomes a z against that
        null; below z_min the router ABSTAINS with the z on record. Token COUNT is matched, so query
        length is controlled for by construction.

        z_min=0.8 was CHOSEN FROM THE MEASURED SEPARATION, not principle: the two logged misroutes score
        z=-0.9 and z=-1.5 (below the null MEAN -- word salad matches scattered prose better than a wrong
        real query does), while the weakest genuine queries tested ('screen a battery of detectors',
        'non-manifold repair') sit at z=+1.02; a first draft at z_min=2.0 abstained on them outright, and a
        line AT 1.0 clipped them (rounding is not a margin). 0.8 leaves the weakest-true population 0.2
        above the line and the misroutes 1.7 sigma below it; recalibration is owed whenever the catalog's
        vocabulary shifts substantially (kept negative 1 below).

        Returns {"abstain": bool, "z": float, "score": float, "null_mean": float, "null_std": float,
        "hits": [(capability, score), ...] or [], "reason": str}. Deterministic given seed.

        KEPT NEGATIVES: (1) abstention is calibrated to the catalog's CURRENT vocabulary -- a new family of
        capabilities shifts the null, so z values are not comparable across catalog versions; (2) a genuine
        query phrased entirely in words the catalog never uses will abstain -- that abstention is CORRECT
        behaviour (the catalog truly has no purchase on it), but it reads as a false negative to a user who
        knows a capability exists under other words; the fix is aliases, not a lower z_min."""
        real = self.find_scored(problem, k=k)
        top = real[0][1] if real else 0.0
        q_tokens = _tokens(problem)
        if not q_tokens:
            return {"abstain": True, "z": 0.0, "score": 0.0, "null_mean": 0.0, "null_std": 0.0,
                    "hits": [], "reason": "empty query"}
        # The null depends only on (token count, n_null, seed) -- not on the query's words -- so it is
        # memoised per catalog state: without this, a 7-query battery re-ran 64*7 full-catalog searches and
        # blew the test-time budget (measured 26s -> the cache takes repeats to ~0). The cache is keyed on
        # the capability COUNT as a cheap staleness proxy; register_capability growth invalidates it.
        np = __import__("numpy")
        key = (len(q_tokens), int(n_null), int(seed), len(self._by_name))
        cache = getattr(self, "_null_floor_cache", None)
        if cache is None:
            cache = self._null_floor_cache = {}
        if key in cache:
            # 3-tuple since the empirical p landed; tolerate a 2-tuple so a cache warmed by older code in
            # the same process cannot raise. p is simply unavailable in that case, never fabricated.
            entry = cache[key]
            mu, sd = entry[0], entry[1]
        else:
            self._ensure_fc_baked()
            vocab = self._fc_vocab
            rng = np.random.default_rng(seed)
            null = np.empty(int(n_null))
            for i in range(int(n_null)):
                fake = " ".join(vocab[int(j)] for j in rng.integers(0, len(vocab), len(q_tokens)))
                fs = self.find_scored(fake, k=1)
                null[i] = fs[0][1] if fs else 0.0
            mu, sd = float(null.mean()), float(null.std()) or 1.0
            cache[key] = (mu, sd, null)
            # persist the null array as well, so the empirical p survives a warm boot (see _ensure_fc_baked)
            self._fc_memo["~null|%d|%d|%d" % (len(q_tokens), int(n_null), int(seed))] = [
                mu, sd, [float(x) for x in null]]
            try:
                import json as _js
                _js.dump(self._fc_memo, open(self._fc_memo_path, "w"))
            except Exception:
                pass
        z = (top - mu) / sd
        # EMPIRICAL p, additive and exact. The z above is fine as a floor test, but it is NOT a p-value:
        # this null is a distribution of MAXIMA (each draw takes find_scored(fake, k=1)), which is
        # right-skewed, so a normal approximation 1-Phi(z) understates the tail and yields an
        # ANTI-CONSERVATIVE p -- the wrong direction for anything feeding an FDR correction. Counting
        # directly is non-parametric and correct for any null shape. The +1 plug matches permutation_null,
        # so p is never exactly 0 and never claims more evidence than n_null draws can support.
        _n = cache[key][2] if len(cache[key]) > 2 else None
        p = float((int((_n >= top).sum()) + 1) / (len(_n) + 1)) if _n is not None else None
        if z < float(z_min):
            return {"abstain": True, "z": float(z), "score": float(top), "null_mean": mu, "null_std": sd,
                    "p": p,
                    "hits": [], "reason": "top score %.2f does not clear the in-vocabulary noise floor "
                                          "(null %.2f +/- %.2f, z=%.1f < %.1f) -- no capability matches"
                                          % (top, mu, sd, z, z_min)}
        return {"abstain": False, "z": float(z), "score": float(top), "null_mean": mu, "null_std": sd,
                "p": p,
                "hits": real, "reason": "top score clears the noise floor (z=%.1f)" % z}


    def find_scored(self, problem, k=3):
        """Like find_capability, but returns [(capability, score)] best-first -- so an agentic layer can turn the raw
        match scores into a CONFIDENCE (how dominant the top hit is). Same scoring, same deterministic tie-break."""
        self._ensure_fc_baked()
        q = set(_tokens(problem))
        if not q:
            return []
        q_phrase = " ".join((problem or "").lower().split())        # normalised whole query, for exact-alias hits
        # FILLER-STRIPPED FORM, restored: the split kept q_phrase but dropped this and the alias comparison
        # that used it, silently deleting the fix. BOTH forms are tested, never just the stripped one --
        # four shipped aliases THEMSELVES begin with a filler ("how do I get from points to a mesh"), so
        # stripping the query alone would destroy the very +5.0 bonus this exists to protect. Testing both
        # can only ADD a match, never remove one.
        q_stripped = _strip_filler(problem)
        scored = []
        for cap in self._by_name.values():
            name_words = cap._nw
            hay = cap._hay
            overlap = len(q & hay)
            if overlap == 0:
                continue
            score = overlap + 0.5 * len(q & name_words)
            if any(q_phrase == a.lower() or q_stripped == a.lower() for a in cap.aliases):  # see find_capability
                score += 5.0
            scored.append((score, cap.name, cap))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(cap, float(sc)) for sc, _, cap in scored[:k]]

    def suggest_pipeline(self, start_kind, goal_kind, max_len=4, require_step=False):
        """S3.4 -- propose a PIPELINE from `start_kind` to `goal_kind` by chaining capabilities whose `produces` feeds
        the next's `consumes`. Returns the shortest chain as a list of {name, consumes, produces} steps (fewest
        steps; deterministic tie-break by name), or None if no chain within `max_len`. This is the render-graph idea
        (nodes typed by what they consume/produce) applied to the whole catalog: instead of a single capability, the
        catalog proposes a *route* -- e.g. points -> mesh might be one step (points_to_mesh) or several.

        When `start_kind == goal_kind`: by default returns [] (the empty pipeline -- you already have that kind). Set
        `require_step=True` to instead demand at least one TRANSFORMING edge of that kind -- 'mesh -> mesh' then
        returns e.g. mesh_smooth / mesh_subdivide, the answer a user asking "refine this mesh" actually wants. This
        is the difference between 'are these the same type?' (default) and 'what can I DO to a mesh?' (require_step).

        Only capabilities that declare BOTH a consumes and a produces participate (an untagged capability has no
        typed edge to route through). A capability is a directed edge kind_in -> kind_out; this is breadth-first
        over those edges, so the first chain found is a shortest one. `start_kind`/`goal_kind` must be io kinds."""
        from holographic.caching_and_storage.holographic_iokinds import is_kind
        if not is_kind(start_kind) or not is_kind(goal_kind):
            raise ValueError("start_kind and goal_kind must be io kinds; got %r -> %r" % (start_kind, goal_kind))
        if start_kind == goal_kind and not require_step:
            return []                                            # already there; the empty pipeline
        # build the typed edges: each tagged capability is an edge from every consumed kind to every produced kind.
        # sorted by name so the BFS is deterministic (an earlier-named capability wins an equal-length race).
        # C6: honour `polymorphic` -- same kind in, same kind out. The cross product below is right for a
        # CONJUNCTIVE capability (transform_selection needs mesh AND selection AND transform, so reaching a mesh
        # from a selection is a real question) but wrong for a POLYMORPHIC one: denoise_tensor takes an image OR a
        # field and returns THE SAME KIND, so image->field is a conversion it cannot perform. Routing over that
        # fake hop is how suggest_pipeline("image","mesh") answered with a tensor denoiser and an Aharonov-Bohm
        # ring in a downstream audit -- a wrong tag does not merely fail to help, it actively misleads.
        # DUPLICATE, now reconciled: pipelinemap._edges built this same edge set a second time and its docstring
        # claimed to match "EXACTLY" -- they had already diverged (the polymorphic fix landed there first and the
        # nonsense route SURVIVED here). Both now apply the same rule; tests/test_pipeline_edges.py pins that
        # they agree edge-for-edge, so the claim is checked rather than asserted.
        edges = []                                               # (consume_kind, produce_kind, cap)
        for cap in sorted(self._by_name.values(), key=lambda c: c.name):
            if not cap.consumes or not cap.produces:
                continue
            if getattr(cap, "polymorphic", False):
                for k in cap.consumes:
                    if k in cap.produces:
                        edges.append((k, k, cap))
                continue
            for ci in cap.consumes:
                for po in cap.produces:
                    edges.append((ci, po, cap))
        # BFS from start_kind; frontier holds (current_kind, path_of_caps). Visited kinds prevent cycles.
        # require_step subtlety: when start==goal we must NOT accept the empty path, so we don't mark start visited
        # up front (we let the search take a real edge first); a self-edge (mesh->mesh) then satisfies the goal.
        from collections import deque
        frontier = deque([(start_kind, [])])
        visited = set() if (start_kind == goal_kind and require_step) else {start_kind}
        while frontier:
            kind, path = frontier.popleft()
            if len(path) >= max_len:
                continue
            for ci, po, cap in edges:
                if ci != kind:
                    continue
                if po == goal_kind:
                    # reached the goal kind by taking this edge -> a non-empty chain (>=1 step). When require_step and
                    # start==goal, the empty path was never accepted because start wasn't pre-marked visited, so the
                    # first self-edge (e.g. mesh->mesh via mesh_smooth) is the shortest valid answer.
                    step = path + [cap]
                    return [{"name": c.name, "consumes": list(c.consumes), "produces": list(c.produces)}
                            for c in step]
                if po not in visited:
                    visited.add(po)
                    frontier.append((po, path + [cap]))
        return None                                              # no route within max_len

    def find_capability_uris(self, problem, k=3):
        """Like find_capability, but each result is annotated with its disambiguating capability URI(s) so a caller
        NEVER gets a bare ambiguous name. Returns [{name, does, example, uris}] where `uris` is the list of full
        paths the name resolves to (holographic_capuri) -- a single path for a unique name, several for a colliding
        one (e.g. 'rotation' -> both meshskin and scenegraph). This is the collision fix at the discovery layer: the
        agent sees the path to supply, not just the name. Falls back to the bare name if the URI index is
        unavailable, so it degrades gracefully."""
        try:
            from holographic.caching_and_storage.holographic_capuri import resolve_uri
        except Exception:
            resolve_uri = None
        out = []
        for cap in self.find_capability(problem, k=k):
            uris = []
            if resolve_uri is not None:
                try:
                    uris = resolve_uri(cap.name)
                except Exception:
                    uris = []
            out.append({"name": cap.name, "does": cap.does, "example": cap.example, "uris": uris or [cap.name]})
        return out

    def to_rows(self):
        """Export entries as plain dict rows (name/does/native) -- e.g. to hand to the SQL capability table."""
        return [{"name": c.name, "does": c.does, "native": c.native} for c in self._by_name.values()]


#: IO-SHAPE MAP for faculty-derived capabilities (S3). seed_from_mind auto-registers every public mind method from
#: its docstring but can't read consumes/produces from prose, so the shapes for the high-value CONVERSION faculties
#: (the pipeline edges) are declared here, in one readable table, and applied at seed time. Grounded in each
#: faculty's real signature. Only conversion/geometry edges that a pipeline would route through are listed -- a
#: faculty absent here is simply unspecified (always shown), which is the correct default. Keep coarse (IO_KINDS).
#: ALIAS MAP for faculty-derived capabilities. seed_from_mind auto-registers every public mind method from its
#: docstring but assigns NO aliases, so bare method-names (agent, scene, query, material, ...) get shadowed in
#: find_capability by a descriptively-titled sibling and become UNDISCOVERABLE. These are stranger-phrasings a
#: user would actually type, so each method surfaces for its own concept. (Discoverability audit D1.)
_METHOD_ALIASES = {
    # D1 REGRESSION, SECOND WAVE (post-merge). Ranking is GLOBAL: ~57 capabilities arrived with two merges and
    # pushed these 25 bare method-names out of the top-15 for their OWN name, which is the audit's definition
    # of dark. Nothing about them changed -- their neighbours did. Fixed the documented way: stranger
    # phrasings, written from a user's mouth rather than the implementer's.
    # D1 REGRESSION, THIRD WAVE. Same mechanism again: the creature/anatomy merge added capabilities and
    # pushed ten more bare names out of the top-15 for their own name. Nothing about them changed. Written
    # from a user's mouth, as the entries below were.
    # D1, FOURTH WAVE, and I caused this one: registering "Explain a stream (...)" pushed the bare
    # method-name `explain` (why are two RECORDS similar) out of the top-15 for its own name. The
    # descriptive title outranks the generic verb, exactly as the earlier waves did. Aliases written
    # from what a caller comparing two records would actually type.
    # D1, FIFTH WAVE, and this session caused it: the science-instrument + media-drift merges
    # (~15 new descriptively-titled entries full of "generate"/"train"/"drift" language) pushed
    # four more bare names out of the top-15 for their own name. Same mechanism as every wave:
    # ranking is global, the neighbours changed, the methods did not. Aliases from the caller's
    # mouth, per the standing rule.
    # D1, SIXTH WAVE, caused by the compression arc (C-1..C-3): three codec entries dense with
    # "train"/"drift"/"code"/"model" language re-darkened drift_train / drift_generate /
    # train_model despite their wave-5 aliases. Same mechanism; the fix is MORE aliases from
    # the caller's mouth (the wave-5 sets stay -- they route, they just no longer outrank).
    # D1, SEVENTH WAVE, caused by merging the codec-arc aliases WITH the unicron
    # aliases: the union of two dense blocks ("save a checkpoint", "compress",
    # "store") re-darkened the bare names place / save. Same mechanism as waves
    # 5 and 6; same fix -- caller's-mouth aliases so the short names outrank.
    "place": ("move an object in the scene", "rotate an object", "scale an object",
              "position an item at coordinates", "transform an object in the world"),
    "save": ("save my mind to disk", "persist the mind", "write the mind to a file",
             "store this session's memory", "save state"),
    "unicron_runtime": ("run an llm forward pass", "numpy transformer inference",
                        "generate text from a checkpoint", "evaluate perplexity of a model",
                        "own the forward pass", "run a qwen model in lecore",
                        "execute a gdn hybrid model", "fast generation with a state cache",
                        "snapshot the model's state", "rewind a conversation",
                        "branch a conversation into two futures", "fork the model's timeline",
                        "carry state between generation calls"),
    "unicron_resident_memory": ("be inside the model", "install a memory expert in a model",
                                "holographic rag inside an llm", "steer a model with lecore memory",
                                "resident expert in the residual stream",
                                "give the model perfect recall"),
    "unicron_bundle": ("bundle the engine with the model", "self contained model",
                       "ship a model that carries lecore", "model that runs anywhere without install",
                       "package everything into one folder", "make the model include the engine"),
    "unicron_capability_tools": ("advertise the model's features as tools",
                                 "openai tool schemas for every capability",
                                 "what can this model do", "function calling schema for lecore"),
    "unicron_grounded_generate": ("pick the most grounded answer",
                                  "generate several answers and choose the best",
                                  "best of n with a verifier", "reduce hallucination by branching",
                                  "let the model think before answering"),
    "unicron_retarget": ("give the model new abilities",
                         "transform qwen into a galvatron",
                         "improve a model where it actually needs it",
                         "plan what to change in a checkpoint",
                         "which layers should i modify"),
    "unicron_autoscale_memory": ("bigger context window", "make the model handle long documents",
                                 "scale memory to a target length",
                                 "multi timescale memory", "beat the context limit"),
    "unicron_hrnn_grow": ("add a memory channel to the model",
                          "give the model long memory for free",
                          "grow a new head instead of reusing one",
                          "extend memory without hurting quality"),
    "unicron_hrnn_bake": ("give the model a longer memory",
                          "make a head remember for longer",
                          "hrnn inside the weights", "retune the decay gates",
                          "why does the model forget so fast"),
    "unicron_load_factors": ("make the model run faster", "speed up inference",
                             "use the factored weights at runtime",
                             "faster forward pass", "why is my smaller model not faster"),
    "unicron_gather_attention": ("make attention actually faster",
                                "skip the keys we already ruled out",
                                "sparse attention that is really sparse",
                                "stop scoring keys we do not use"),
    "unicron_kv_compress": ("much longer context", "shrink the kv cache",
                           "fit more tokens in the same memory",
                           "compress attention cache", "run out of memory on long prompts"),
    "unicron_fold_correction": ("bake the correction into the model",
                               "make a fix part of the weights",
                               "turn a low rank patch into neurons"),
    "unicron_residual_correction": ("undo quantization damage",
                                   "recover accuracy after compressing",
                                   "predict the error and subtract it",
                                   "cheaper than adding another bit"),
    "unicron_requantize": ("shrink the model properly", "quantize by measurement",
                           "pick bit width per tensor", "compress a heavy tailed model",
                           "make the checkpoint smaller for gguf"),
    "unicron_refactor": ("make the model smaller without breaking it",
                         "decompose and rebuild a model", "factor the weights",
                         "shrink a checkpoint with a measured budget",
                         "rebuild a model holographically"),
    "unicron_progbake": ("store a shader inside the model",
                         "put code into the weights", "programs as model data",
                         "hide data in unused vocabulary",
                         "project a program out of a hypervector"),
    "unicron_harden": ("prove the model really has lecore",
                       "test the install under abuse",
                       "does the layer still work after quantizing",
                       "end to end check of an installed model"),
    "unicron_evolve": ("train without gradients", "evolution strategies",
                       "optimize something that has no derivative",
                       "population search over model weights",
                       "train the lecore additions into a model"),
    "unicron_assess": ("measure a model so someone else can judge it",
                       "compare two assimilation runs",
                       "which step actually helped", "export a model report"),
    "unicron_deployable": ("can this model run in ollama",
                           "is the galvatron actually usable",
                           "check it converts to gguf",
                           "is it as good as the original",
                           "will this work outside lecore"),
    "unicron_model_store": ("store the model in our own format",
                            "compatibility wrapper for a model",
                            "keep weights compressed on disk",
                            "produce a normal checkpoint on demand",
                            "smaller model file that still loads"),
    "unicron_tensor_map": ("how are the tensors in this model related",
                          "encode a weight matrix as a hypervector",
                          "which tensors look like each other",
                          "find a tensor that does not fit",
                          "structure of a safetensors file"),
    "unicron_measure": ("is this difference real or noise",
                        "perplexity with error bars",
                        "did that change actually help",
                        "how many tokens do i need to measure this",
                        "compare two models honestly"),
    "unicron_sidecar": ("add lecore without touching the model",
                        "wrapper in front of a model",
                        "keep the base file untouched",
                        "adapter file next to a checkpoint",
                        "turn lecore on and off"),
    "unicron_install_facts": ("teach the model something new in the weights",
                             "make it answer a question it could not",
                             "store a fact the model will recall",
                             "can this model even hold facts",
                             "edit what a model knows"),
    "unicron_vsa_run": ("make the model do vsa algebra itself",
                       "install unbind as neurons",
                       "run holographic memory in the forward pass",
                       "lecore computing inside the weights"),
    "unicron_memory_search": ("search memory from inside the model",
                              "let the model look things up itself",
                              "give the model a searchable index",
                              "retrieve a passage from a partial cue",
                              "expand what the model can remember"),
    "unicron_router": ("let the model decide when to use a capability",
                       "gate a circuit on the prompt",
                       "first layers decide what later layers do",
                       "only search when the prompt asks",
                       "decision inside the forward pass"),
    "unicron_prepend_layers": ("add a lecore layer to any model",
                               "give a model extra layers up front",
                               "bios layer before the model starts",
                               "work with models we know nothing about",
                               "empty layer that changes nothing"),
    "unicron_prefix_cache": ("speed up a long conversation",
                             "stop re-reading the whole chat history",
                             "reuse work from earlier turns",
                             "cache prompt prefixes"),
    "unicron_state_io": ("save what the model has accumulated",
                         "persist lecore memory between sessions",
                         "what does the harness need to store",
                         "restore a conversation's memory",
                         "export the recurrent state"),
    "unicron_reserve_keys": ("make a memory that never gets overwritten",
                             "why did the model forget that",
                             "protect a slot in the recurrent state",
                             "permanent memory across a long conversation"),
    "unicron_install_lecore": ("install lecore into a model",
                               "put the whole engine in the weights",
                               "build a model with lecore inside",
                               "give this model memory and routing"),
    "unicron_write_policy": ("decide what is worth remembering",
                             "what should go in memory",
                             "pick the important parts of a passage",
                             "which tokens surprised the model"),
    "unicron_early_exit": ("skip layers when the answer is already decided",
                           "make the model faster without changing it",
                           "shortcut through the layers",
                           "which tokens need the whole model"),
    "unicron_adapt": ("work out what kind of model this is",
                      "read a checkpoint with no config",
                      "install into a model we have never seen",
                      "infer hidden size and layer count from tensors"),
    "unicron_self_write": ("let the model store things on its own",
                           "decide what to remember without being told",
                           "the model writes to its own memory",
                           "automatic storage of surprising input"),
    "unicron_sequence": ("store a sequence so order matters",
                        "remember which came first",
                        "encode position in a hypervector",
                        "hierarchy that a bundle cannot express"),
    "unicron_hlb": ("cheaper binding operator",
                    "bind without storing a matrix",
                    "hadamard binding instead of convolution",
                    "make an installed circuit smaller"),
    "unicron_model_vault": ("save a trained model and run it later",
                            "store a drift model holographically",
                            "recall a model from storage",
                            "keep a trained thing without keeping its encoder"),
    "unicron_program_library": ("find the right vsa program for a situation",
                                "run a stored holographic program",
                                "programs that discover themselves",
                                "composable vsa procedures"),
    "unicron_device": ("use the gpu if there is one",
                       "run the model on a graphics card",
                       "check cpu or gpu is being used",
                       "make sure it still works without a gpu"),
    "unicron_vm_install": ("put the virtual gpu inside the model",
                          "which cache tiers can live in weights",
                          "install a gather unit into a layer",
                          "can the memory hierarchy be baked in"),
    "unicron_install_plan": ("how should i install this operator",
                             "can a whole pipeline fit in one layer",
                             "install an iterative solver into weights",
                             "fuse a chain of transforms"),
    "unicron_install_order": ("what order should i install things in",
                              "do these two steps interfere",
                              "which step has to go last",
                              "did this step change what it said it would"),
    "unicron_long_context": ("context of a billion tokens",
                             "remember across an unbounded stream",
                             "how often must memory be refreshed",
                             "what limits how far back the model can see"),
    "unicron_self_heal": ("repair memory without a backup copy",
                          "fix a register that has drifted",
                          "is the stored value still trustworthy",
                          "clean up a corrupted memory slot"),
    "unicron_actr": ("rank memories by recency and frequency",
                     "which memory should i retrieve",
                     "forget what stopped being useful",
                     "activation ranking for stored items"),
    "unicron_nullspace": ("install without disturbing what the model knows",
                          "make an edit that preserves existing behaviour",
                          "project a weight change onto unused directions",
                          "reduce the cost of installing a circuit"),
    "unicron_state_track": ("keep count across a long sequence",
                            "track a state machine while reading",
                            "something attention cannot compute",
                            "remember a running total"),
    "unicron_hybrid": ("combine the language model with exact memory",
                       "decide when to generate and when to recall",
                       "which tokens should memory handle",
                       "use both the model and the store together"),
    "unicron_runtime": ("run a model using its installed leCore parts",
                        "the loop that uses memory and the model together",
                        "serve a model with exact recall",
                        "actually use what was installed"),
    "unicron_ref": ("get a handle for an object over http",
                    "return something json cannot carry",
                    "pass a live object between invoke calls",
                    "why did this capability return a memory address"),
    "unicron_turn_memory": ("keep every turn instead of forgetting old ones",
                            "a separate memory per conversation turn",
                            "stop the register file filling up",
                            "hold more facts than slots"),
    "unicron_recipe": ("ship the install without shipping the model",
                       "why is the installed model so much bigger",
                       "store the rule instead of the bytes",
                       "send someone a leCore install"),
    "unicron_vm_install": ("put a vsa program into the model weights",
                           "install an opcode sequence",
                           "make the model run a holographic program",
                           "fuse several operations into one"),
    "unicron_bios": ("what kind of model is this",
                     "probe a checkpoint before touching it",
                     "will this fit in my model", "enumerate a model's layout",
                     "is lecore already installed here"),
    "unicron_install": ("install lecore into a model",
                        "check that the install actually worked",
                        "audit a model for lecore", "set up the layer in a checkpoint",
                        "did the install take"),
    "unicron_query_path": ("let the model look things up itself",
                          "turn the stream into a lookup key",
                          "retrieve a stored fact from a prompt",
                          "the model asks its own memory"),
    "unicron_seeded_channel": ("read hidden data with only a seed",
                              "self describing storage in weights",
                              "payload that needs no original file"),
    "unicron_quantsafe": ("storage that survives gguf",
                         "hide data that quantization cannot erase",
                         "keep a payload through q4 conversion",
                         "write bits into the rounding"),
    "unicron_store_program": ("put code inside the model",
                             "run lecore programs from the weights",
                             "store a program in a checkpoint",
                             "execute instructions stored in weights"),
    "unicron_fountain": ("recover data from any subset of pieces",
                         "erasure codes", "survive losing part of the data",
                         "rateless codes", "luby transform droplets"),
    "unicron_store_route": ("store a generator instead of the data",
                            "decide how to store this payload",
                            "is this data compressible at all",
                            "keep learning after the model ships",
                            "store a rule not the output"),
    "unicron_resilient_store": ("storage that survives damage",
                                "payload that tolerates a dead channel",
                                "keep data even if part of the model is rewritten"),
    "unicron_substrate": ("hide data inside the weights",
                          "use the model as a disk", "storage capacity of a checkpoint",
                          "write files into a model", "how much can i hide in the weights"),
    "unicron_boot": ("boot lecore inside the model", "an operating system in the weights",
                     "regenerate the whole layer from a seed",
                     "store a boot record in a checkpoint",
                     "make the model carry lecore itself"),
    "unicron_call_tokens": ("let the model call capabilities itself",
                            "tool calling baked into the weights",
                            "the model decides to run a function",
                            "capability tokens in unused vocabulary",
                            "model asks for a tool without being told"),
    "unicron_swarm_bake": ("swarm inside the model",
                          "experts that route by content",
                          "run specialists in one forward pass",
                          "mixture of experts in the weights",
                          "inject capability without a prompt"),
    "unicron_vsa_roles": ("structured memory inside the model",
                          "bind subject and object into one vector",
                          "role filler slots for a model", "store relations in a vector",
                          "give the model somewhere to put structure"),
    "unicron_vsabake": ("run vsa inside the model", "hypervector algebra in the weights",
                        "holographic computing space inside a model",
                        "bake bind and unbind into a checkpoint",
                        "make the model do vsa by itself"),
    "unicron_distill": ("train the abilities into the model",
                        "make residents permanent", "teach the weights what the residents do",
                        "absorb runtime behaviour into weights",
                        "distill a galvatron into a plain checkpoint"),
    "unicron_bake": ("bake abilities into the weights",
                     "make the ban survive gguf conversion",
                     "put a memory into the weights themselves",
                     "edit weights instead of hooking the runtime",
                     "keep capabilities after export"),
    "unicron_port": ("run it in ollama", "use the model with llama.cpp",
                     "export for gguf", "make it work in a normal runtime",
                     "what survives outside lecore"),
    "unicron_cache": ("stop redoing the same work", "cache the model's internal work",
                      "speed up repeated lookups", "memoize routing and retrieval",
                      "make the model faster without changing answers"),
    "unicron_toolbelt": ("give the model all the capabilities",
                         "let the model call any tool", "model can run physics and math",
                         "all of lecore inside the model", "capability router for the model"),
    "unicron_memory": ("obsidian alternative", "notes with backlinks",
                       "a second brain for the model", "knowledge graph of my notes",
                       "store notes and query them", "memory the model can write to"),
    "unicron_vault": ("import an obsidian vault", "read a folder of markdown notes",
                      "convert markdown notes into memory"),
    "unicron_knowledge": ("start a conversation with a clean slate",
                          "stop this chat from seeing old conversations",
                          "delete old conversations and notes",
                          "prune what the model remembers",
                          "private session that references nothing",
                          "remember everything i tell the model",
                          "search what the model has been told",
                          "store documents the model can cite later",
                          "make conversation history searchable",
                          "catalog of what the model knows",
                          "reference information from an old conversation"),
    "unicron_scribe": ("let the swarm write its own notes",
                       "internal experts keep reference documents",
                       "agent notes that are searchable later",
                       "partitioned notes from the model's own reasoning"),
    "unicron_sessions": ("keep a conversation going for days",
                         "save and restore the model's context",
                         "multiple conversations at once",
                         "swap contexts in and out", "persistent context store",
                         "manage many chats with one model",
                         "continue where we left off after restarting"),
    "unicron_imbue": ("imbue a model", "imbue a checkpoint",
                      "imbued galvatron", "make an imbued model",
                      "give a checkpoint its residents",
                      "make an imbued model", "build an imbued galvatron",
                      "turn weights into a galvatron"),
    "unicron_maximal_specs": ("give the model every capability we have",
                              "maximal galvatron", "wire all the experts at once",
                              "full resident stack for a model",
                              "put as much of lecore in the model as possible"),
    "unicron_best_portable": ("best plain checkpoint we can make",
                              "optimize a model for normal harnesses",
                              "export the strongest compatible model",
                              "measured retention export"),
    "unicron_save_pack": ("package a model with its scaffolding", "ship a galvatron",
                          "save a model plus its residents", "bundle model and experts",
                          "export a model that needs lecore"),
    "unicron_load_pack": ("load a packaged model", "restore a model and its residents",
                          "open a galvatron package", "run a packaged model without lecore"),
    "unicron_serve_openai": ("serve my model with an openai compatible api",
                             "make my model look like a normal api",
                             "chat completions endpoint for my model",
                             "point lm studio at my model", "wrap the model as a standard server"),
    "unicron_hf_wrapper": ("make it work like a transformers model",
                           "drop in replacement for a huggingface model",
                           "generate like a normal model object"),
    "unicron_lazy_weights": ("keep the model compressed in memory",
                             "decompress weights on demand", "run a model with less ram",
                             "compression inside the model", "lazy weight loading",
                             "stream weights as the model needs them"),
    "unicron_export_portable": ("export a model that runs in ollama",
                                "make it work in llama.cpp", "convert to a normal checkpoint",
                                "run our model on huggingface", "portable model export",
                                "ship the model to a standard harness"),
    "unicron_middleout": ("middle out compression", "progressive weight code",
                          "compress a model so i can decode it at any size",
                          "one file many fidelity levels", "truncatable model artifact",
                          "coarse to fine weight encoding", "scalable model storage"),
    "unicron_middleout_decode": ("decode a progressive weight stream at a budget",
                                 "load a model at lower fidelity", "truncate a weight stream",
                                 "read fewer refinement layers"),
    "unicron_capability_resident": ("let the model call a simulation",
                                    "give a model access to physics",
                                    "tool use inside the forward pass",
                                    "model calls lecore capabilities",
                                    "inject a computed answer into the model",
                                    "can the model run a fluid sim",
                                    "give an llm exact math and simulation"),
    "unicron_salience_trigger": ("let the model decide when to search",
                                 "detect when the model is uncertain",
                                 "trigger retrieval on hesitation",
                                 "model asks for help by itself",
                                 "fire a tool only when the model needs it",
                                 "uncertainty detection from hidden states"),
    "unicron_corpus_resident": ("rag inside the model", "give the model a document corpus",
                                "retrieve passages during generation",
                                "search my documents from inside the forward pass",
                                "unlimited knowledge without context window",
                                "ground the model in my own documents"),
    "unicron_hrnn_resident": ("run hrnn on the model's hidden states",
                              "analyze the model's trajectory",
                              "sequence analysis of what the model is doing",
                              "let lecore watch the model think"),
    "unicron_manifold_voids": ("where has the model never been",
                               "find holes in the activation space",
                               "regions the model never visits",
                               "gaps in what a model represents",
                               "void exploration of a model"),
    "unicron_void_probe": ("decode an unvisited state", "what would the model say there",
                           "explore a novel activation", "read out a void"),
    "unicron_carrier": ("write structured data into the model's activations",
                        "side channel inside the residual stream",
                        "exact symbolic state alongside the model's thinking",
                        "use the unused dimensions of the hidden state",
                        "carry key value pairs through the layers",
                        "read back what i wrote into the stream"),
    "unicron_forward_embeds": ("run the model from hidden states",
                               "feed embeddings instead of tokens",
                               "superpose inputs into the model",
                               "run a model on a vector not a token"),
    "unicron_layer_schedule": ("run layers twice", "depth upscaling without retraining",
                               "frankenmerge a model", "make the model deeper with the same weights",
                               "layer recursion", "prune layers at inference"),
    "unicron_screen_routing": ("skip attention work with a summary index",
                               "route attention through block summaries",
                               "cheaper long context without changing the answer",
                               "read the boundary instead of the whole context",
                               "sparse attention that finds the right keys"),
    "unicron_capacity_report": ("how much context does this model really use",
                                "boundary versus volume in a model",
                                "is the state or the kv cache doing the work",
                                "information capacity of a model's state",
                                "holographic capacity audit"),
    "unicron_memory_horizon": ("how far back does the model remember",
                               "real memory length of a recurrent model",
                               "when does the model forget a token",
                               "causal memory horizon"),
    "unicron_attention_waste": ("how much attention compute is wasted",
                                "how many keys actually matter",
                                "sparse attention radius", "is the model doing useless work",
                                "measure redundancy in attention"),
    "unicron_leap": ("generate tokens faster", "speed up my model's output",
                     "speculative decoding with a learned drafter",
                     "make the llm faster without changing what it says",
                     "cache the routes the model takes", "faster inference same output"),
    "unicron_verified_generate": ("fact check the output before it is emitted",
                                  "stop the model from making things up",
                                  "verify claims against sources during generation",
                                  "grounded generation with a checker",
                                  "internal critique and revise loop",
                                  "agent loop without token round trips"),
    "unicron_evidence": ("build an evidence store", "allowed claims for the model",
                         "source spans the model may assert"),
    "unicron_swarm": ("inner monologue for the model", "swarm of agents inside the model",
                      "subconscious deliberation", "many agents thinking between tokens",
                      "nested agent swarm", "internal committee for a model"),
    "unicron_swarm_mind": ("generate with an inner monologue",
                           "run the model with a subconscious",
                           "let inner agents vote on the next token",
                           "orchestrate a swarm and merge the result"),
    "unicron_galvatron": ("rebuild a model with resident experts", "put lecore inside the model",
                          "model with a memory expert and a guard", "make a galvatron",
                          "run a model with residents in its forward pass",
                          "hard ban tokens during generation", "repair the model's thoughts"),
    "unicron_council": ("deliberate over alternate continuations", "branch futures and pick the best",
                        "self consistency without a second model", "model council",
                        "compare steered and unsteered generations"),
    "unicron_generator_audit": ("can this data be regenerated from a seed",
                                "does a generator exist for this tensor",
                                "is this compressible to a formula",
                                "check if weights have hidden structure"),
    "unicron_archive": ("archive a fleet of models", "store only the difference from a reference",
                        "deduplicate shared tensors across checkpoints", "store a model as delta",
                        "seed instead of data", "recipe instead of data", "model version control"),
    "unicron_restore": ("restore a model from the archive", "rebuild a checkpoint from deltas",
                        "regenerate a tensor from its seed"),
    "unicron_shelve": ("remember this model", "add a model to the library",
                       "register a checkpoint in memory"),
    "unicron_identify": ("which stored model is this", "identify a mystery checkpoint",
                         "what lineage is this model from", "recognize a model by content"),
    "unicron_report": ("tell me everything about this model", "what should i do with this checkpoint",
                       "full analysis of a model file", "can this model be compressed",
                       "audit a checkpoint", "what are my options for this llm",
                       "one call model diagnosis"),
    "unicron_lineage": ("which model is this based on", "find the base model of a fine tune",
                        "detect model lineage from weights", "who is this checkpoint's parent",
                        "pair a fine tune with its base without metadata"),
    "unicron_delta_store": ("store only what the fine tune changed",
                            "delta storage for models", "ship many fine tunes of one base",
                            "compress the difference between two checkpoints",
                            "lora style storage after the fact", "save a model as a diff"),
    "unicron_delta_apply": ("rebuild a fine tune from a diff", "apply a stored model delta",
                            "interpolate between base and fine tune"),
    "unicron_taskvector": ("extract a capability from a fine tune", "task vector arithmetic",
                           "difference between two checkpoints as a skill", "what did fine tuning add",
                           "pull the learning out of a fine tune"),
    "unicron_imbue": ("add a skill to a model", "inject an expert into a model",
                      "transplant a capability", "graft knowledge into weights",
                      "combine two fine tunes", "give a model new powers",
                      "merge a fine tune into another model"),
    "unicron_heads": ("how many attention heads does this matrix have", "find the head structure",
                      "discover heads in a projection", "dissect a weight matrix into heads",
                      "recover the head count blind"),
    "unicron_depthshare": ("shared structure across many matrices", "do the layers repeat themselves",
                           "how redundant is model depth", "cross layer shared subspace",
                           "is the model one matrix wearing costumes", "depth redundancy of a model"),
    # SWEEP-7 routing fixes (post-merge battery misses, phrasings verbatim):
    "unicron_analyze": ("analyze a trained neural network model", "inspect an llm",
                        "weight matrix analysis", "how well trained is this model",
                        "spectral analysis of weights", "marchenko pastur on my model",
                        "what is inside this llm file", "look inside a checkpoint"),
    "unicron_transform": ("transform a model into a smaller one", "compress a whole checkpoint",
                          "shrink a neural network", "upgrade a trained model", "rewrite model weights",
                          "low rank factorize every layer", "model surgery", "make a model smaller",
                          "make my model smaller without breaking it", "safely compress a model"),
    # UNICRON (part 16): aliases from the caller's mouth -- someone holding a
    # checkpoint file, not someone who knows the module name.
    "unicron_load": ("read model weights", "load a safetensors file", "open an llm checkpoint",
                     "load trained model weights", "parse a model file", "safetensors",
                     "load a gguf file", "llama.cpp model", "dequantize model weights"),
    "unicron_fingerprint": ("hypervector for a whole model", "model fingerprint",
                            "embed a model as a vector", "model signature",
                            "represent a checkpoint holographically"),
    "unicron_subspace": ("principal angles between subspaces", "subspace overlap of two matrices",
                         "compare singular vector spaces", "do two layers point the same way",
                         "grassmann distance between weight matrices"),
    "unicron_assimilate": ("assimilate a model end to end", "one call model pipeline",
                           "defrag a model", "clean and re-export a checkpoint",
                           "optimize a whole llm checkpoint", "reorganize model weights",
                           "process a qwen or llama checkpoint", "full model upgrade pass"),
    "unicron_reconstruct": ("expand a factored model back to dense", "undo model compression",
                            "multiply the u v factors back", "rebuild dense weights"),
    "unicron_retention": ("did the transform keep accuracy", "measure accuracy before and after",
                          "functional retention of a model", "prove the compression is safe",
                          "capability check after surgery"),
    "unicron_localize": ("where is the learned information in the weights", "singular vector localization",
                         "porter thomas test on weights", "which coordinates does this layer use",
                         "localized singular vectors"),
    "unicron_filter": ("denoise model weights", "filter noise out of a weight matrix",
                       "strip the random part of a trained layer", "rmt weight filtering",
                       "compress a checkpoint by keeping outliers", "clean up trained weights"),
    "unicron_trajectory": ("track a model across training checkpoints", "training trajectory of a model",
                           "how did my model change during training", "checkpoint time series",
                           "watch training move the weights", "spectral dynamics of a run"),
    "unicron_compare": ("compare two trained models", "teacher vs student weights",
                        "did distillation work", "diff two checkpoints",
                        "compare model checkpoints"),
    "generate": ("continue this text", "next tokens from the model", "text continuation",
                 "sample from the sequence model", "autocomplete from schema"),
    "train_model": ("train a classifier on sequences", "fit a trajectory classifier",
                    "label sequences and learn", "one call training front door"),
    "drift_train": ("train a generative model on points", "learn a distribution from samples",
                    "fit a drift model", "moments from my data", "point cloud generative model"),
    "drift_generate": ("sample new points from a drift model", "generate from the moments",
                       "draw samples like my data", "run the drift sampler"),
    "explain": ("why are these two records similar", "compare two records field by field",
                "explain a match", "why did these match", "per-role decode of two records"),
    "build_creature": ("make a creature", "generate a creature", "build a whole creature",
                       "create an animal", "one call creature"),
    "build_part": ("make a body part", "build a horn", "generate a foot", "make an eye",
                   "create a claw"),
    "creature_parts": ("body part library", "what parts can i use", "part palette",
                       "list of body parts", "available parts"),
    "make_mixture": ("mix two materials", "smoke and dye", "multi channel matter",
                     "blend substances", "milk in water"),
    "make_surrogate": ("cheap stand-in for a slow function", "approximate an expensive call",
                       "learned shortcut", "cache a function as a model"),
    "rig": ("creature skeleton", "one rig type", "bones and joints", "skeleton for any body",
            "humanoid and creature rig", "articulation"),
    "skin_mesh": ("deform a mesh with bones", "linear blend skinning", "attach a mesh to a skeleton",
                  "bind mesh to bones", "move the mesh with the rig"),
    "terrain": ("make a landscape", "generate ground", "heightfield", "hills and valleys",
                "procedural terrain"),
    "tissue_at": ("what tissue is here", "bone or muscle at this point", "what is inside the body",
                  "sample the anatomy", "is this point bone"),
    "what_is_at": ("what part is on this socket", "which part did i attach", "read back an attachment",
                   "query a socket"),
    "bake_texture": ("bake a texture", "bake lighting to a texture", "bake to uv", "texture baking"),
    "forecast": ("predict the next value", "forecast a series", "time series prediction", "what comes next"),
    "forward_forward": ("forward-forward learning", "train without backprop", "local learning rule"),
    "from_file": ("load from a file", "read a saved mind", "restore from disk"),
    "from_state": ("restore from a state dict", "rebuild from saved state", "resume a mind"),
    "gather": ("gather values by index", "collect entries", "index into a table"),
    "invite": ("invite a collaborator", "share access", "add a participant"),
    "invoke": ("call a faculty by name", "dispatch a method by name", "call a tool by name"),
    "is_manifold": ("is the mesh manifold", "check mesh manifoldness", "watertight check"),
    "load": ("load a saved mind", "open a saved session", "read a mind from disk"),
    "make_water": ("make water", "create a water surface", "water material"),
    "materials": ("list the materials", "what materials are available", "material library"),
    "mesh_repair": ("repair a broken mesh", "fix mesh errors", "clean up a mesh"),
    "mesh_report": ("mesh statistics", "report on a mesh", "mesh health check"),
    "mesh_to_field": ("turn a mesh into a field", "mesh as a field", "sample a mesh as a function"),
    "mesh_to_sdf": ("mesh to sdf", "convert a mesh to a distance field", "signed distance from a mesh"),
    "node_graph": ("build a node graph", "node based pipeline", "wire nodes together"),
    "reproject": ("reproject to a new view", "warp between viewpoints", "reprojection"),
    "scan": ("scan an object", "capture a scan", "photogrammetry scan"),
    "scene_from_image": ("build a scene from a photo", "image to scene", "reconstruct a scene from a picture"),
    "sculpt": ("sculpt a mesh", "push and pull geometry", "digital sculpting"),
    "semantic_scene": ("describe the scene semantically", "scene meaning", "what is in this scene"),
    "shape": ("make a shape by name", "build a primitive shape", "named shape"),
    "use_gpu": ("turn the gpu on", "enable gpu acceleration", "switch to the device"),
    "weld_mesh": ("weld duplicate vertices", "merge coincident vertices", "stitch a mesh"),
    # C2/D1: these constructors ALWAYS existed and worked -- m.render_mesh(m.mesh_box(), m.camera(...)) renders
    # today with no class imports. A downstream audit still reported them "absent", because it searched for
    # make_box / box_mesh / cube / primitive / make_camera and find_capability answered "Catmull-Clark
    # subdivision". A capability find_capability cannot surface does not exist, so the bug was the vocabulary,
    # not the code. Aliases written from the CLIENT's mouth, verbatim where they used them.
    "mesh_box": ("make a box", "box mesh", "cube", "make_box", "box_mesh", "primitive", "create a box",
                 "build a cube", "box primitive", "make a cube mesh", "new box"),
    "mesh_grid": ("make a grid", "plane mesh", "make_plane", "ground plane", "quad grid", "make a plane"),
    "camera": ("make a camera", "make_camera", "create a camera", "new camera", "viewpoint", "eye and target",
               "set up a camera", "camera from eye and target", "pinhole camera"),
    "camera_controller": ("orbit camera", "viewport camera", "pan dolly zoom", "turntable", "orbit the view"),
    "render_mesh": ("mesh to image", "render a mesh", "rasterize a mesh", "picture of a mesh", "mesh render",
                    "draw a mesh", "3d to image"),
    "agent": ('reinforcement learning agent', 'creature that learns from reward', 'action library agent', 'npc brain', 'agent with reward and pain', 'learning game character'),
    "bake": ('bake a slow function into a cache', 'precompute and look up', 'consolidation cache', 'cache an expensive evaluator', 'bake once sample many'),
    "blend": ('blend learned classes', 'mix two concepts', 'interpolate between categories', 'combine learned prototypes', 'projection to create new things'),
    "build_index": ('build a nearest neighbour index', 'index vectors for search', 'make a search index', 'knn index over vectors', 'build an ann index'),
    "build_scene": ('describe a scene in words and build it', 'text to scene', 'make a scene from a description', 'natural language scene builder', 'scene from prompt'),
    "bus": ('shared message bus', 'event bus for the mind', 'publish subscribe channel', 'inter-agent messaging', 'one shared bus'),
    "database": ('a database you own', 'user namespaces over records', 'owned query database', 'personal database', 'namespaced data store'),
    "denoise": ('clean a noisy signal', 'remove noise from a vector', 'denoise by projecting onto a manifold', 'plug and play denoiser', 'restore a corrupted signal'),
    "encyclopedia_is_a": ('parent in the taxonomy', 'one hop up the is-a tree', 'get the category of a concept', 'taxonomic parent', 'what is this a kind of'),
    "find": ('which record holds this binding', 'find the record with a role filler', 'locate a bound pair', 'search absorbed records', 'find bind role filler'),
    "game_world": ('massive sharded game world', 'open world of game shards', 'lazy grid game world', 'streaming game world', 'huge multiplayer world'),
    "hypervector": ('encode as a first-class hypervector', 'raw vector plus metadata', 'hypervector object', 'typed hypervector', 'encode to a hypervector wrapper'),
    "is_a": ('is this a kind of that', 'taxonomic membership', 'does concept belong to category', 'check is-a relationship', 'subtype check'),
    "is_deterministic": ('is this function deterministic', 'does it return the same result every time', 'check determinism', 'bit-identical repeatability', 'is the output reproducible'),
    "light": ('add a light to a scene', 'directional or point light', 'sun light source', 'scene lighting object', 'place a light'),
    "make_cloud": ('render a volumetric cloud', 'make a cloud image', 'volumetric cloud in one call', 'draw a cloud', 'cloud render'),
    "make_observer": ('custom sensor from sensitivity curves', 'build an observer', 'spectral sensor with channels', 'define a camera response', 'per-channel sensor'),
    "make_scene": ('build a scene from nested objects', 'assemble a scene from a list', 'scene from objects', 'construct a nested scene', 'scene builder from parts'),
    "make_table": ('ingest tabular data', 'load a table of records', 'rows into a vsa table', 'tabular data to vectors', 'make a table from dicts'),
    "material": ('a pbr material', 'physically based material', 'material with textures', 'role-filler material record', 'surface material'),
    "measure": ('variance harness', 'mean spread and confidence interval', 'measure with error bars', 'bootstrap a number honestly', 'report a metric with variance'),
    "plan": ('bake a route to the next decision', 'short executable plan', 'corridor to the next waypoint', 'plan a path to a goal', 'route to decision point'),
    "query": ('run a sql query', 'select from where', 'query records with sql', 'small sql over a table', 'sql subset query'),
    "read": ('pre-learn word co-occurrence', 'learn word meanings from text', 'read a corpus', 'warm up text meaning', 'learn from reading'),
    "recall": ('nearest stored item', 'recall the closest memory', 'find the nearest stored individual', 'exact nearest neighbour recall', 'retrieve closest stored vector'),
    "refine": ('improve a result with a critic', 'iterate against a scorer', 'refine with feedback', 'critic-guided improvement', 'produce then critique then redo'),
    "route": ('decide which skill to use', 'agent decision node', 'should i act or choose', 'route a task to a skill', 'pick the winning skill'),
    "sample_from": ('draw a symbol from a distribution', 'weighted random pick', 'sample with temperature', 'nucleus sampling', 'sample from a weighted dict'),
    "sampler": ('placeable read probe', 'read a field at a point', 'read-dual of a field effect', 'probe a field value', 'sample a field with a probe'),
    "scatter": ('scatter points onto a grid', 'deposit values with a kernel', 'splat onto a grid', 'scatter is the bundle', 'rasterize points to a field'),
    "scene": ('compose visual attribute scenes', 'scene coder', 'encode and decode a scene', 'compose decompose a scene', 'visual scene algebra'),
    "scene_to_render": ('flatten a scene for the path tracer', 'scene to sdf and material', 'prepare a scene for rendering', 'brain-to-muscle scene handoff', 'bake a scene to renderable'),
    "sdf_render": ('march an sdf to a mesh', 'sdf tree to triangles', 'turn an sdf into a mesh', 'marching cubes on an sdf', 'sdf to drawable mesh'),
    # C1: PBR Neutral tonemap (a postfx stage) -- make the tonemapper discoverable via the chain faculties.
    "post_process": ('apply a post processing chain', 'tonemap a rendered frame', 'pbr neutral tonemap',
                     'khronos neutral tone mapping', 'tonemap that holds hue better than aces', 'colour grade a frame',
                     'aces or pbr neutral tonemapping', 'gltf standard tonemapper', 'run a postfx chain on a frame'),
    "postfx_chain": ('build a post processing chain', 'exposure bloom tonemap vignette chain', 'pbr neutral or aces stage',
                     'assemble a colour grade pipeline', 'tonemapping and grading program'),
    # ITEM 9: PostChain.to_glsl / mind.postfx_to_glsl compiles the pointwise colour pipeline to a fragment shader.
    "postfx_to_glsl": ('emit glsl for a post processing chain', 'fragment shader for a color grading pipeline',
                       'postfx to webgl', 'run the colour grade on the gpu', 'compile a postfx chain to a shader',
                       'post processing chain to glsl', 'gpu color pipeline shader', 'live video grading shader',
                       'export a colour pipeline as glsl', 'tone map and grade in a fragment shader'),
    # C5: closed-form patterns emit to WGSL (WebGPU) too, not just GLSL.
    "pattern_to_wgsl": ('emit a pattern as wgsl', 'webgpu shader for a pattern', 'checker or dots as wgsl',
                        'procedural background for webgpu', 'wgsl fragment for a pattern', 'pattern field as wgsl'),
    # C4: bloom as a multi-pass GPU DAG (single-pass postfx_to_glsl must refuse it -- no intermediate texture).
    "bloom_passes": ('multi pass bloom shader', 'emit bloom as several gpu passes', 'render target ping pong bloom',
                     'separable blur passes for glow', 'bloom pass dag with wiring', 'gpu bloom the honest way'),
    # C2: pattern_to_glsl now emits the GPU-reproducible noise32/fbm32 (int64 noise/fbm still refuse -- see module note).
    "pattern_to_glsl": ('emit a pattern as glsl', 'gpu reproducible noise', 'value noise that runs on the gpu',
                        'fractal noise shader that matches cpu', 'emit noise to glsl', 'noise32 or fbm32 shader',
                        'pattern field as a fragment shader', 'checker or stripes or noise as glsl'),
    # ITEM 6: inpaint/harmonic_fill now accept (H,W,C) -- an RGB image fills in one call. The image-oriented phrasings
    # ("inpaint an rgb image", "fill a masked area of a photo") did not rank on the field-oriented docstring alone.
    "inpaint": ('fill the gaps in a field', 'impute missing values', 'inpaint an rgb image', 'fill missing regions in a color image',
                'fill a masked area of a photo', 'inpaint all channels at once', 'fill holes in an image', 'fill in erased pixels',
                'harmonic fill a field', 'label propagation into holes'),
    "harmonic_fill": ('laplace fill a continuous field', 'harmonic inpaint an image', 'fill holes in an rgb image',
                      'smooth interpolation into holes', 'minimal energy fill', 'inpaint a colour image per channel'),
    # ITEM 5: segment_image gained max_dim (bound the pixel-scaled sweep, upsample masks back to full size). These
    # latency-oriented phrasings did not rank on the docstring alone -- aliased from the interactive caller's mouth.
    "segment_image": ('segment a photo into regions', 'demux an image by colour', 'segment a large image quickly',
                      'bound the segmentation resolution', 'downsample before segmenting', 'fast segmentation of a big photo',
                      'limit segmentation to a max dimension', 'segment at a capped resolution', 'object regions from a photo'),
    # ITEM 8: the emitted shader gained a camera="uniforms" mode (orbit camera driven by host-bound uAngle/uHeight/
    # uDist), so a WebGL2 host can spin/zoom without string-splicing the source. These abstract phrasings ("spin and
    # zoom", "controllable camera") did not rank on the faculty docstring alone -- aliased from the host's mouth.
    "to_shadertoy": ('export an sdf to shadertoy glsl', 'get the shadertoy code for a shape', 'sdf to glsl shader',
                     'shadertoy shader with orbit camera', 'glsl with a controllable camera', 'movable camera shader',
                     'spin and zoom the shadertoy scene', 'camera uniforms in the emitted glsl',
                     'orbit camera uniforms uAngle uHeight uDist'),
    "sdf_shader": ('emit a glsl fragment shader for an sdf', 'sdf to shadertoy shader', 'raymarch shader from an sdf',
                   'glsl shader with orbit camera uniforms', 'controllable camera in the emitted glsl',
                   'movable camera raymarcher', 'spin and zoom the emitted shader'),
    "selection": ('select objects in a scene', 'selection set', 'query objects into a selection', 'scene selection helper', 'pick a set of objects'),
    "simulation": ('wrap a solver in a step loop', 'generic simulation scaffold', 'step any time-stepped solver', 'simulation loop wrapper', 'run any solver step by step'),
    "step_at": ('the i-th step of a plan', 'get a plan step by index', 'which step is next', 'read a stored plan step', 'plan step at index'),
    "surface_mesh": ('re-extract a mesh from a field', 'field to drawable mesh', 'sculpt re-extract step', 'turn any field into a mesh', 'remesh from a field'),
    "to_state": ('snapshot the mind for saving', 'serialize the learned mind', 'save state with quantization', "export the mind's state", 'dump the brain to state'),
    "trace": ('provenance of a result', 'style and material trace', 'where did this come from', 'full provenance answer', 'explain the origin'),
}

#: Hand-set semantic tags for faculties whose NAME lies about their verb. infer_semantic is a stem table, and a
#: stem is a strong signal that is sometimes simply wrong: `resolve_swept_collision` is "positional CCD for a
#: PBD-style solver" -- physics -- but the stem "resolve" routes it to analyze/pipeline with the capability
#: browser. The cure is not a cleverer table (each new special case there costs every OTHER name accuracy); it is
#: a small explicit override list, the same shape as _METHOD_ALIASES. Overrides win; inference fills the rest.
#: Keep this SHORT: if it grows past a few dozen, the table is wrong, not the names.
_SEMANTIC_OVERRIDES = {
    # convert/isosurface was a DOCUMENTED-BUT-EMPTY branch -- and the doc's own rule is that an empty branch is
    # the discovery equivalent of a dark module. It was empty because its members were untagged, not because it
    # was aspirational: these three are exactly "points<->mesh, sdf<->mesh" as the doc describes it.
    "occupancy_to_mesh": "convert/isosurface",
    "mesh_from_sdf": "convert/isosurface",
    "points_to_mesh": "convert/isosurface",
    # simulate/pbd, same story: this is a PBD solver's collision step, not a capability lookup.
    "resolve_swept_collision": "simulate/pbd",
}

_IO_SHAPES = {
    # points/sdf/mesh interconversions -- the core geometry pipeline edges
    "points_to_mesh":    (("points",), ("mesh",)),
    "mesh_from_sdf":     (("sdf",), ("mesh",)),
    "voxelize_mesh":     (("mesh",), ("field",)),
    "mesh_sample_field": (("mesh", "points"), ("scalar",)),
    "mesh_uv_unwrap":    (("mesh",), ("mesh",)),
    # ---- S7 io-kind drive, batch 2. Same discipline: signature + first docstring line READ for every entry.
    "advect_field":        (("field",), ("field",)),       # transport a field along a velocity field
    "diffuse_field":       (("field",), ("field",)),       # heat-kernel diffuse; field in, field out
    "domain_twist":        (("sdf",), ("sdf",)),           # iq's opTwist -- an SDF domain operator
    "domain_bend":         (("sdf",), ("sdf",)),
    "domain_repeat":       (("sdf",), ("sdf",)),
    "domain_fold":         (("sdf",), ("sdf",)),
    "bake_sdf":            (("sdf",), ("field",)),         # PRECOMPUTE an sdf onto a grid = a sampled field
    "collide_sdf":         (("points", "sdf"), ("points",)),   # push points out of an sdf; needs BOTH (conjunctive)
    "emit_from_surface":   (("sdf",), ("points",)),        # sample an sdf's zero level-set -> particles
    "ambient_occlusion":   (("sdf", "points"), ("scalar",)),   # conjunctive: needs the field AND the query points
    # ABSTAINED, verified (a wrong tag actively misleads; a missing one merely waits):
    #   ascii_field / ascii_sdf -- produce TEXT. There is no `text` io kind and minting one to fit two faculties
    #     is a taxonomy decision, not a tagging one (SEMANTIC_TAXONOMY.md: a new root is a design conversation).
    #   curve_resample_arc_length -- signature is (points, n); the NAME says curve, the code takes points. This is
    #     the exact trap batch 1 recorded: the name lies, the signature does not.
    #   bump / apply_mueller / classify_transform -- their docstrings NAME several kinds while consuming one
    #     compound argument; "mentions two kinds" is not "converts between them".
    # ---- S7 io-kind drive, batch 1. EVERY entry below was verified by READING the signature and the first
    # docstring line, not inferred from the name. That discipline is not ceremony: the `X_to_Y` convention looks
    # like a free tag source but LIES about types -- mesh_to_stl produces a string, mesh_to_softbody a SoftBody,
    # procedure_to_recipe a recipe. None are io kinds, and tagging them from the name would manufacture exactly
    # the fake edges tag_lint exists to catch. A tag is a claim a router will ACT on.
    "image_to_3d":         (("image",), ("mesh",)),        # END-TO-END photo -> mesh; the client's flagship
    "grid_to_hypervector": (("field",), ("hypervector",)), # a NumPy field -> an FPE hypervector
    "hypervector_to_grid": (("hypervector",), ("field",)), # the stated inverse of the above
    "occupancy_to_mesh":   (("field",), ("mesh",)),        # occupancy grid -> surface mesh (surface_nets)
    "near_surface_to_sdf": (("field",), ("sdf",)),         # thin-band signed field -> a full SDF
    "skin_skeleton":       (("skeleton",), ("mesh",)),     # stick figure -> wrapped mesh; makes `skeleton` routable
    "record_physics_trace": ((), ("timeseries",)),         # a recorded trace IS the timeseries producer
    "audio_param_bus":     ((), ("timeseries",)),          # per-frame band-energy envelopes over time
    "phase_fold":          (("timeseries",), ("timeseries",)),
    # ABSTAINED (verified, deliberately untagged -- a wrong tag actively misleads, a missing one merely waits):
    #   field_to_splats     -- takes `centers` (points), not a field; the NAME says field. Read the signature.
    #   dynamics_to_mesh    -- `source` is a dynamics OBJECT, not an io kind; produces mesh from nothing tagged.
    #   mesh_to_softbody    -- produces a SoftBody; there is no such io kind and inventing one is a design call.
    #   curve_bspline       -- CONSUMES control points and produces samples; it does not produce a `curve`.
    # image -> geometry: the RETURN leg, and the reason a downstream audit saw a nonsense route. These three
    # faculties say "image -> MESH" in their own first docstring line and were UNTAGGED, so the router knew of NO
    # typed image->mesh edge -- and reached the goal through a FAKE cross-product hop instead (see `polymorphic`).
    # Two bugs compounding: an invented edge the router could use, and the real route invisible. This is S7's
    # thesis in one line -- the coverage gap is not cosmetic, it actively produces wrong routes.
    "image_to_mesh":     (("image",), ("mesh",)),
    "depth_to_mesh":     (("image",), ("mesh",)),
    "photo_to_3d":       (("image",), ("mesh",)),
    # rendering edges -- geometry -> image
    "render_mesh":       (("mesh",), ("image",)),
    "render_scene":      (("sdf_scene",), ("image",)),
    # field sampling
    "sample_field":      (("field", "points"), ("scalar",)),
    # field hole-filling (inpaint) -- field -> field, so a holed field can be repaired mid-pipeline
    "inpaint":           (("field",), ("field",)),
    "harmonic_fill":     (("field",), ("field",)),
    "majority_fill":     (("field",), ("field",)),
    # mesh <-> field, and mesh refinement (mesh -> mesh) so a raw mesh can be cleaned before use
    "mesh_to_field":     (("mesh",), ("field",)),
    "mesh_subdivide":    (("mesh",), ("mesh",)),
    "mesh_smooth":       (("mesh",), ("mesh",)),
    "mesh_poke":         (("mesh",), ("mesh",)),         # fan a face to triangles -- a mesh->mesh retopology edge
    # generative edges: a profile/curve swept into a mesh; a mesh deformed by a skeleton
    "sweep_tube":        (("curve",), ("mesh",)),
    "skin_mesh":         (("mesh", "skeleton"), ("mesh",)),
}


def _derive_method(name, example):
    """The callable faculty name behind a capability, or None if it has none (import-only).

    Two honest sources, in order: the NAME itself when it is already a bare identifier (every auto-registered
    faculty), else the first `m.foo(` / `mind.foo(` in the runnable example (how the curated prose-titled entries
    -- 'Mesh repair (weld + split non-manifold + fill + compact)' -- name their callable). This is the SAME regex a
    downstream client was running on our example strings; doing it here means one implementation to fix instead of
    one per client, and `seed_from_mind` then VERIFIES the answer against a live mind and nulls the liars, which
    no client could do for itself.

    Returns None rather than a guess: a wrong method name is worse than an absent one -- it makes a broken call
    look like a supported one, which is exactly what the client's EXCLUDED.md was working around.
    """
    import re
    if re.fullmatch(r"holographic_[a-z0-9_]+", name or ""):
        return None                                          # a MODULE, not a faculty -- reach it via `import`
    if re.fullmatch(r"[a-z_][a-z0-9_]*", name or ""):
        return name                                          # a bare method name IS the callable
    hit = re.search(r"\bm(?:ind)?\.([a-z_][a-z0-9_]*)\s*\(", example or "")
    return hit.group(1) if hit else None


def seed_from_mind(catalog, mind):
    """Reuse the faculty walk (as holographic_query.capability_registry does) to auto-register every public method
    of `mind` as a catalog entry, using its one-line docstring as `does`. Curated entries (registered explicitly)
    win, because they carry better `does`/`example`/`native` -- we don't overwrite them here. The io-shape map
    (_IO_SHAPES) is applied to both auto-registered AND curated entries, so the conversion faculties become pipeline
    edges (S3) even though their consumes/produces can't be read from prose."""
    import inspect
    # C7 VERIFICATION -- the step no client can do for itself, and the reason a `method` field beats a regex.
    # _derive_method is a GUESS from text: a module-name entry like "holographic_rayindex" is a bare identifier,
    # so the guess claims it, but `mind.holographic_rayindex` does not exist. Measured: 511 of 2,040 guesses were
    # wrong (508 module names + 3 strays). Here we hold a LIVE mind, so every claim is checked and the liars are
    # nulled to None -- which is not a loss, it is the honest "import-only" answer (their item 8). A wrong method
    # name makes a broken call look supported; that is what the client's EXCLUDED.md was working around.
    for cap in catalog._by_name.values():
        if cap.method and not callable(getattr(mind, cap.method, None)):
            cap.method = None
    for name in dir(mind):
        if name.startswith("_"):
            continue
        cons, prod = _IO_SHAPES.get(name, ((), ()))
        if name in catalog._by_name:
            # curated entry already exists -- don't overwrite it, but DO stamp its io-shape if we know it and it
            # doesn't already carry one (so a curated conversion becomes a pipeline edge too).
            existing = catalog._by_name[name]
            if (cons or prod) and not existing.consumes and not existing.produces:
                from holographic.caching_and_storage.holographic_iokinds import validate_kinds
                existing.consumes = validate_kinds(cons, where="_IO_SHAPES of %r" % name)
                existing.produces = validate_kinds(prod, where="_IO_SHAPES of %r" % name)
            continue
        attr = getattr(type(mind), name, None)
        if not callable(attr):
            continue
        doc = (inspect.getdoc(attr) or "").strip().split("\n")[0][:160]
        # S4.2 SEMANTIC TAG, DERIVED. browse_semantic() omits untagged capabilities, and every one of these
        # auto-registered faculties -- the engine's actual verb surface, the things a user DOES -- arrived with
        # semantic=None. Coverage was 108/2095 (5.2%): the action menu could not see 95% of the engine. Tagging
        # 1,200 methods by hand is not a fix (1,200 chances to rot, and the next faculty lands untagged anyway),
        # so the tag is INFERRED from the verb in the name, at registration, exactly as _METHOD_ALIASES fixed the
        # dark method names (D1). infer_semantic ABSTAINS rather than guess -- a wrong branch files a capability
        # under a verb nobody will look for and, unlike a missing one, looks done.
        from holographic.caching_and_storage.holographic_semantictag import infer_semantic
        catalog.register_capability(name, doc or name, example="mind.%s(...)" % name, native=True,
                                    consumes=cons, produces=prod,
                                    semantic=_SEMANTIC_OVERRIDES.get(name) or infer_semantic(name, doc),
                                    aliases=_METHOD_ALIASES.get(name, ()))   # D1: bare method-names were dark w/o these
    return catalog


def default_catalog():
    """A catalog seeded with the CONSOLIDATION HOMES and the key shipped modules the audits named -- the search
    indices, the caches / bakes, and the field types -- so they are findable TODAY. As each consolidation home
    (Index, Cache, Field, ...) lands, register it here with native=True."""
    c = Catalog()
    # Each part's entry point is register_pNN, not a shared `register`: six modules exporting the same
    # public name is a name-collision the budget must not grow to absorb, and the unified split set the
    # precedent with distinct _UnifiedPartNN classes. Distinct names also make a traceback name its part.
    # THE REGISTRY LIVES IN PARTS (see holographic_catalog_p01). Called in ORDER: find_capability
    # ranks by score and ties break by registration order, so the sequence is part of the contract.
    # IMPORT STYLE IS LOAD-BEARING: `from PKG import MODULE` is INVISIBLE to tools/wiring_report, which
    # then reports all six parts as dark modules with no caller (a CI gate). `import PKG.MODULE as X`
    # is the form the audit can see. Same lesson previously cost a 43-import sweep elsewhere.
    import holographic.caching_and_storage.holographic_catalog_p01 as holographic_catalog_p01
    import holographic.caching_and_storage.holographic_catalog_p02 as holographic_catalog_p02
    import holographic.caching_and_storage.holographic_catalog_p03 as holographic_catalog_p03
    import holographic.caching_and_storage.holographic_catalog_p04 as holographic_catalog_p04
    import holographic.caching_and_storage.holographic_catalog_p05 as holographic_catalog_p05
    import holographic.caching_and_storage.holographic_catalog_p06 as holographic_catalog_p06
    holographic_catalog_p01.register_p01(c)
    holographic_catalog_p02.register_p02(c)
    holographic_catalog_p03.register_p03(c)
    holographic_catalog_p04.register_p04(c)
    holographic_catalog_p05.register_p05(c)
    holographic_catalog_p06.register_p06(c)
    return c


# module-name keyword -> the domain words to tag it with, so a problem description can reach a whole family. Broad on
# purpose (the module's own docstring is the real match; these just add domain synonyms).
_FAMILY_KEYWORDS = [
    (("light", "lamp", "dome", "ies", "spot", "spharm", "prt", "radiance", "illum", "occlusion", "shadow"),
     ("lighting", "light")),
    (("mat", "brdf", "shade", "thinfilm", "iridesc", "clearcoat"), ("material", "shading")),
    (("tex", "noise", "weather", "burn", "oxid", "cellular", "inclusion", "curl", "tiling", "displace"),
     ("texture", "procedural")),
    (("field", "ndfield", "voxel", "regionfield", "dirtyfield", "sparsefield", "spectralfield", "fpefield"),
     ("field",)),
    (("mesh", "splat", "gltf", "svg", "subdiv", "sculpt", "poly", "geodes", "uv", "sdf", "octree", "terrain",
      "meshverbs", "deform", "skin", "blendpose", "curvature"), ("geometry", "mesh")),
    (("cache", "bake", "lut", "compile", "residency", "anim", "multires", "mipmap"), ("cache", "bake")),
    (("fluid", "smoke", "fire", "combust", "softbody", "cosserat", "groom", "mpm", "collide", "automaton",
      "cloth", "eulerops", "fields"), ("simulation", "physics")),
    (("matter", "mixture", "diffus", "equilibrium", "chem", "emitter", "scatter", "phasemorph", "physics"),
     ("physics", "chemistry", "matter")),
    (("tree", "pivot", "rayindex", "archive", "forest", "organizer", "navigator", "spatial", "pack", "vault"),
     ("search", "index", "recall")),
    (("sampl", "discrepancy", "mis", "accumulate", "traverse", "lowdiscrepancy"), ("sampling",)),
    (("denoise", "svgf", "sharpen", "downscale", "diffuse", "modulate", "diffusion"), ("denoise",)),
    (("render", "path", "ray", "raster", "postfx", "lens", "volint", "camera", "gbuffer", "brdf", "globalillum",
      "pathtrace", "radiance"), ("render", "rendering")),
    (("query", "sql", "table", "knowledge", "relation", "encyclopedia", "workspace"), ("query", "database")),
    (("distribute", "nystrom", "spectral", "graph", "transport", "topology", "cosmic", "market", "kde", "chart"),
     ("scale", "analysis")),
    (("text", "lexicon", "encyclopedia", "intent", "answer", "respond", "deliberate", "grammar", "segment",
      "meaning", "lang", "vision", "reasoning"), ("language", "text")),
    (("creature", "agent", "value", "drive", "classifier", "kan", "moe", "forward", "recurrent", "reservoir",
      "predictive", "dream", "partition", "orchestrator", "voidsynth"), ("learning", "agent")),
    (("honesty", "measure", "ablate", "structure", "protocol", "flatness", "benchmark", "stress"),
     ("honesty", "measurement")),
]


def _family_of(name):
    """The domain words a module belongs to (from its name), for catalog search. Empty -> ('engine',)."""
    n = name.replace("holographic_", "")
    fams = set()
    for keys, fam in _FAMILY_KEYWORDS:
        if any(k in n for k in keys):
            fams.update(fam)
    return tuple(sorted(fams)) or ("engine",)


def seed_from_modules(catalog, module_dir=None):
    """Register EVERY engine module as a findable capability, so nothing built stays buried. AST-reads each
    holographic_*.py module's DOCSTRING without importing it (the docgen discipline -- safe, side-effect-free) and
    adds it with its one-line summary as `does`, tagged with its domain family. Curated homes and mind faculties
    already registered are NOT overwritten (they carry better descriptions). This is what makes the catalog COMPLETE:
    a plain-English problem can surface any module, home, or faculty in the engine.

    The engine modules live under the `holographic` package (holographic/<family>/holographic_*.py). We walk that
    whole tree -- not just one directory -- and register each module under its DOTTED import path (e.g.
    `holographic.rendering.holographic_render`), so the `import ...` example the catalog serves actually works."""
    import ast
    import os
    import glob
    if module_dir is None:
        # this file lives in holographic/caching_and_storage/, so the package root is two levels up
        here = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(here)                       # .../holographic
        repo_root = os.path.dirname(pkg_root)                  # repo root (parent of holographic/)
    else:
        # backward compatible: if a caller passes an explicit dir, treat it as the package root's parent
        pkg_root = module_dir
        repo_root = os.path.dirname(os.path.abspath(module_dir))
    for path in sorted(glob.glob(os.path.join(pkg_root, "**", "holographic_*.py"), recursive=True)):
        base = os.path.basename(path)[:-3]
        if base.startswith("test_"):
            continue
        if base in catalog._by_name:
            continue
        # dotted module path relative to the repo root: holographic/rendering/holographic_render.py
        #   -> holographic.rendering.holographic_render
        rel = os.path.relpath(path, repo_root)
        dotted = rel[:-3].replace(os.sep, ".")
        # DOCSTRING BAKE. Profiled before building: 581 ast.parse+compile calls were
        # 2.1s of the 2.7s catalog cold start, re-extracting one docstring line per
        # module on EVERY first find_capability of every session. The summary is a
        # pure function of file bytes, so it is cached keyed by (size, mtime_ns) --
        # a fresh checkout pays one rebuild, then every boot is warm. The cache
        # lives in tempdir; a stale or missing entry falls through to the parse.
        st = os.stat(path)
        ck = "%s|%d|%d" % (rel, st.st_size, st.st_mtime_ns)
        if "_docbake" not in globals():
            import json as _js
            import tempfile as _tf
            globals()["_docbake_path"] = os.path.join(_tf.gettempdir(),
                                                      "lecore_docbake.json")
            try:
                globals()["_docbake"] = _js.load(open(_docbake_path))
            except Exception:
                globals()["_docbake"] = {}
            globals()["_docbake_dirty"] = 0
        if ck in _docbake:
            summary = _docbake[ck]
        else:
            try:
                tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            except (SyntaxError, OSError):
                continue
            doc = ast.get_docstring(tree) or ""
            summary = doc.strip().replace("\n", " ").split("  ")[0][:200] if doc else base
            _docbake[ck] = summary
            globals()["_docbake_dirty"] += 1
        cap = catalog.register_capability(base, summary or base, example="import %s" % dotted, native=True,
                                          aliases=_family_of(base))
        # C7: a MODULE is not a callable faculty. `base` looks like a bare identifier, so _derive_method would
        # happily claim `mind.holographic_rayindex` -- which does not exist. These entries are registered AFTER
        # seed_from_mind's verification pass, so they would sail past it: 494 of the 511 measured liars were
        # exactly this. Null it at the source. None is the honest answer and IS the import-only flag (their
        # item 8) -- you reach a module through `import`, which is what the example already says.
        cap.method = None
    if globals().get("_docbake_dirty"):
        try:
            import json as _js
            _js.dump(_docbake, open(_docbake_path, "w"))
            globals()["_docbake_dirty"] = 0
        except Exception:
            pass
    return catalog


def check_catalog_part(part_module, register_fn):
    """The contract EVERY catalog part must satisfy, in ONE home -- the mirror of holographic.unified.check_part.

    A part must register onto a fresh Catalog, register a NON-EMPTY set, and not collide with itself. Those are
    the failures a mechanical split actually threatens: a chunk boundary that swallowed a registration or
    repeated one. WHY THIS IS SHARED RATHER THAN COPIED SIX TIMES: six byte-identical `_selftest` bodies are a
    real cross-module duplicate and the duplication budget caught them -- correctly. Budgeting them would have
    silenced a working alarm; unifying keeps the assertion AND removes the duplicate, which is what the budget
    exists to push you toward."""
    c = Catalog()
    register_fn(c)
    caps = c.all()
    assert caps, "%s registered NOTHING -- a part that registers nothing is a silently missing chunk" % part_module
    names = [x.name for x in caps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, "%s registers the same name twice: %s" % (part_module, dupes)
    return len(caps)


def _selftest():
    c = default_catalog()
    # the headline: describe a problem, get the right home
    hits = c.find_capability("search a big pile of vectors")
    assert hits and "Index" in hits[0].name, [h.name for h in hits]
    assert any("Cache" in h.name or "bake" in h.does for h in c.find_capability("precompute a slow thing and look it up"))
    assert any("Field" in h.name for h in c.find_capability("represent a density volume over space"))
    assert any("light" in (h.name + h.does).lower() for h in c.find_capability("my placed light has speckle noise"))
    # register + find a fresh capability
    c.register_capability("MyThing", "does a special new job with widgets", aliases=("widget",))
    assert c.find_capability("widget job")[0].name == "MyThing"
    # native flag is carried
    assert c.get("holographic_catalog").native is False and c.get("Index (search)").native is True
    # NO ACCIDENTAL DOUBLE-REGISTRATION: a capability name registered twice in the source silently overwrites the
    # first (register updates by name), leaving dead prose and a confusing diff. Scan the source for duplicate
    # register_capability("name", ...) calls -- there must be none. (Found two this way: snap_to_grid x3, rayindex.)
    import re as _re, collections as _co, pathlib as _pl
    _src = _pl.Path(__file__).read_text()
    _names = _re.findall(r'register_capability\(\s*"([^"]+)"', _src)
    _dupes = [n for n, k in _co.Counter(_names).items() if k > 1]
    assert not _dupes, "capability name(s) registered more than once (delete the redundant block): %s" % _dupes
    # S4.1 SEMANTIC DRIFT GATE: every semantic= tag's ROOT verb must be one of the taxonomy roots (SEMANTIC_
    # TAXONOMY.md). A tag under an unknown root is a menu branch nobody documented -- the discovery equivalent of a
    # dark module. Roots are kept here in sync with the doc; adding a root is a deliberate edit in both places.
    _ROOTS = {"create", "select", "transform", "modify", "measure", "convert", "render", "simulate",
              "animate", "analyze", "io"}
    _bad = [(cap.name, cap.semantic) for cap in c.all()
            if getattr(cap, "semantic", None) and cap.semantic.split("/")[0] not in _ROOTS]
    assert not _bad, "semantic tag(s) under an unknown root (add the root to SEMANTIC_TAXONOMY.md or fix): %s" % _bad
    # S3 io-shape filter: accepts= keeps only mesh-consumers; an untagged capability is unspecified (still shown).
    mesh_hits = c.find_capability("selection", accepts="mesh")
    assert all((not h.consumes) or ("mesh" in h.consumes) for h in mesh_hits), [(h.name, h.consumes) for h in mesh_hits]
    # S3.4 pipeline: a tagged edge chains produces->consumes; a known 1-step and a known multi-step both resolve.
    one = c.suggest_pipeline("mesh", "selection")
    # more than one capability now produces a selection from a mesh (mesh_selection, mesh_auto_seam); the BFS returns
    # ONE valid 1-step edge -- assert it is one of the known mesh->selection producers, not a specific one.
    assert one and len(one) == 1 and one[0]["name"] in ("mesh_selection", "mesh_auto_seam"), one
    multi = c.suggest_pipeline("transform", "selection")               # transform->mesh->selection, 2 steps
    assert multi and len(multi) == 2, multi
    assert c.suggest_pipeline("mesh", "mesh") == []                    # already there -> empty pipeline
    # NOTE: the conversion edges (points_to_mesh, render_mesh, ...) are faculty-derived and only get their io-shape
    # via seed_from_mind(_IO_SHAPES); this bare default_catalog() has only the directly-registered tags, so we pin
    # a route through THOSE (mesh -> selection -> mesh via transform_selection) rather than a seeded conversion.
    round_trip = c.suggest_pipeline("selection", "mesh")
    assert round_trip and round_trip[0]["name"] == "transform_selection", round_trip
    # require_step turns a same-kind query into a transforming edge instead of the empty 'already there' pipeline.
    assert c.suggest_pipeline("mesh", "mesh") == []                    # default: already there
    same = c.suggest_pipeline("mesh", "mesh", require_step=True)       # forced: a real mesh->mesh edge
    assert same and len(same) == 1 and "mesh" in c.get(same[0]["name"]).produces, same
    print("OK: holographic_catalog self-test passed (%d capabilities; 'search a big pile of vectors' -> %s; "
          "no double-registrations; every semantic tag under a known verb root; io-filter + pipeline route)"
          % (len(c), c.find_capability("search a big pile of vectors")[0].name))


if __name__ == "__main__":
    _selftest()

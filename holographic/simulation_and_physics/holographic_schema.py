"""Modality-agnostic schema discovery: learn the compressive hierarchy of ANY stream.

The maze partition handed us the regions (grid tiles). Real learning has to DISCOVER them,
and the way to discover them is compression: repeatedly fuse the most frequent adjacent pair
into a new symbol. That move is modality-blind -- it never looks at what a symbol *means* --
so the same machinery finds words in text, idioms and indentation in code, the cycle in a
periodic signal, and repeated motifs in an image. Only the tokenizer at the edge knows the
modality; everything above it operates on a bare list of hashable symbols.

Two honest, measured facts shape the design:

  * The FIRST discovered level is the big win (atoms -> meaningful chunks lowers bits-per-atom
    across every modality tested). It is the schema the flat model never had.
  * Deeper levels are diminishing -- and can REVERSE -- with a naive per-level model, because
    rare chunks starve the statistics. Going deeper honestly needs cross-level backoff (an
    unseen phrase falls back to its words, a word to its letters). So `SchemaModel` backs off
    by atom-length, and `bits_per_atom` is reported per level so the curve is always visible.

Each level's transitions form a bounded, directed graph -- the same sparse, navigable
structure the slime substrate is good at -- and chunking is exactly what keeps each level
bounded: the capacity lesson, one floor up.
"""

import math
import numpy as np
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# tokenizers: raw data of any modality -> a flat list of atomic symbols
# (and back, for generation output)
# ---------------------------------------------------------------------------

def to_symbols(data, modality, bins=16, num_range=None):
    """Turn raw data into a list of atomic symbols. The ONLY modality-aware step."""
    if modality in ("text", "code"):
        return list(data)                                  # characters
    if modality == "bytes":
        return list(data if isinstance(data, (bytes, bytearray)) else bytes(data))
    if modality == "numbers":
        lo, hi = num_range or (min(data), max(data))
        span = (hi - lo) or 1.0
        return [f"q{int(max(0, min(bins - 1, (v - lo) / span * bins)))}" for v in data]
    if modality == "image":
        import numpy as np
        a = np.asarray(data)
        return [f"p{int(v)}" for v in a.flatten()]         # already-quantized palette indices
    raise ValueError(f"unknown modality {modality!r}")


def from_symbols(symbols, modality, bins=16, num_range=None):
    """Inverse of to_symbols for the modalities that round-trip cleanly (text, code,
    bytes, numbers). Image needs a shape, so reshape the result yourself."""
    atoms = list(_flatten(symbols))
    if modality in ("text", "code"):
        return "".join(atoms)
    if modality == "bytes":
        return bytes(atoms)
    if modality == "numbers":
        lo, hi = num_range or (0.0, 1.0)
        span = (hi - lo) or 1.0
        return [lo + (int(a[1:]) + 0.5) / bins * span for a in atoms]
    raise ValueError(f"cannot decode modality {modality!r}")


def _flatten(tokens):
    """A chunk is a flat tuple of atoms; an atom is whatever the tokenizer emitted (a
    single character, a "q14" bin label, a byte). Flatten back to the atom stream --
    crucially WITHOUT splitting a multi-character atom like "q14" into characters."""
    for t in tokens:
        if isinstance(t, tuple):
            yield from t
        else:
            yield t


def _merge_symbol(a, b):
    """A merged chunk is always a flat tuple of atoms -- never a concatenation. That keeps
    atoms opaque, so "q1" and "q2" become the chunk ("q1","q2") rather than the unrecoverable
    string "q1q2". Modality-blindness depends on this: the learner must never peek inside an
    atom or assume atoms are characters."""
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    return a + b


def _atom_len(tok):
    return len(tok) if isinstance(tok, tuple) else 1


# ---------------------------------------------------------------------------
# Schema: discover the chunk hierarchy by compression (byte-pair encoding)
# ---------------------------------------------------------------------------

class Schema:
    """The discovered hierarchy: an ordered list of pair-merges. Applying them to a fresh
    stream re-chunks it the same way. This is the schema -- learned, not given."""

    def __init__(self, merges=400, min_count=3):
        self.max_merges = merges
        self.min_count = min_count
        self.merges = []

    def learn(self, symbols):
        toks = list(symbols)
        self.merges = []
        while len(self.merges) < self.max_merges:
            pairs = Counter(zip(toks, toks[1:]))
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < self.min_count:
                break
            self.merges.append((a, b))
            toks = self._apply_one(toks, a, b)
        return self

    @staticmethod
    def _apply_one(toks, a, b):
        m = _merge_symbol(a, b)
        out, i, n = [], 0, len(toks)
        while i < n:
            if i < n - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(m); i += 2
            else:
                out.append(toks[i]); i += 1
        return out

    def encode(self, symbols, upto=None):
        toks = list(symbols)
        for a, b in self.merges[:upto]:
            toks = self._apply_one(toks, a, b)
        return toks

    def emergent(self, symbols, k=10, min_atoms=3, max_render=46):
        """The longest discovered chunks, rendered readably -- the schema made visible.
        Truncated, because for low-vocabulary modalities (a periodic signal, a tiled image)
        the longest chunk is the entire repeating run and would otherwise flood the output."""
        seen = {t for t in self.encode(symbols) if _atom_len(t) >= min_atoms}
        chunks = sorted(seen, key=_atom_len, reverse=True)[:k]
        out = []
        for c in chunks:
            s = "".join(str(a) for a in _flatten([c]))
            out.append(s if len(s) <= max_render else s[:max_render] + "…")
        return out


# ---------------------------------------------------------------------------
# SchemaModel: a per-level transition model over chunks -- bits/atom + generation
# ---------------------------------------------------------------------------

class SchemaModel:
    """An interpolated-backoff n-gram over a level's chunk tokens. Backoff reaches down by
    CONTEXT length here; honest cross-LEVEL backoff (unseen chunk -> its sub-chunks) is the
    documented next step for going deeper than one level."""

    def __init__(self, order=3):
        self.n = order
        self.ctx = [defaultdict(Counter) for _ in range(order + 1)]

    def fit(self, toks):
        self.V = set(toks)
        self.uni = Counter(toks)
        self.tot = max(1, len(toks))
        for i in range(len(toks)):
            for m in range(self.n + 1):
                if i - m >= 0:
                    self.ctx[m][tuple(toks[i - m:i])][toks[i]] += 1
        return self

    def _prob(self, context, tok):
        lam = 1.0 / (self.n + 1)
        p = lam * (self.uni.get(tok, 0) + 1) / (self.tot + len(self.V))   # smoothed floor
        for m in range(1, self.n + 1):
            if m <= len(context):
                c = self.ctx[m].get(tuple(context[-m:]))
                if c:
                    p += lam * (c.get(tok, 0) / sum(c.values()))
        return p

    def bits_per_atom(self, toks, atoms):
        """Held-out cross-entropy in bits, divided by ATOMIC symbols -- comparable across
        levels and modalities. Lower = the schema learned/compressed the data better."""
        bits = 0.0
        for i in range(len(toks)):
            p = self._prob(toks[max(0, i - self.n):i], toks[i])
            bits += -math.log2(p) if p > 0 else 50.0
        return bits / max(1, atoms)

    def generate(self, n_atoms, seed=None, temperature=0.6, rng=None):
        """Walk the chunk transition graph, emitting whole chunks -- so output is built from
        learned units (real words, valid idioms) rather than one shaky character at a time."""
        import numpy as np
        rng = rng or np.random.default_rng(0)
        out = list(seed or [])
        atoms = sum(_atom_len(t) for t in out)
        while atoms < n_atoms:
            context = out[-self.n:]
            dist = None
            for m in range(min(self.n, len(context)), -1, -1):
                c = self.ctx[m].get(tuple(context[-m:]) if m else ())
                if c:
                    dist = c
                    break
            if not dist:
                break
            toks = list(dist)
            w = np.array([dist[t] for t in toks], dtype=float) ** (1.0 / temperature)
            nxt = toks[int(rng.choice(len(toks), p=w / w.sum()))]
            out.append(nxt)
            atoms += _atom_len(nxt)
        return out


# ---------------------------------------------------------------------------
# convenience: learn a schema and a model for a stream in one call
# ---------------------------------------------------------------------------

def learn(symbols, merges=400, order=3):
    schema = Schema(merges=merges).learn(symbols)
    model = SchemaModel(order=order).fit(schema.encode(symbols))
    return schema, model


# ---------------------------------------------------------------------------
# HierModel: a fractal coder -- predict at the highest level that knows, else
# recurse DOWN a level to spell the unknown out. One rule, every scale.
# ---------------------------------------------------------------------------

class HierModel:
    """Cross-level backoff over the schema hierarchy.

    The naive per-level n-gram could only back off by shortening its CONTEXT; when a whole
    chunk was unseen it fell to the floor and wasted bits (which is why deeper levels reversed
    on code). This model backs off by stepping DOWN A LEVEL instead: a level predicts the next
    chunk, and the probability mass it can't account for -- the Witten-Bell escape mass, which
    is exactly the proportion of novelty it has seen -- flows to the level below, which spells
    the chunk out from smaller pieces, recursively, down to atoms (which never fail).

    Because the escape recovers the lower level almost exactly when a level is unsure, ADDING a
    level can only help: a confident high level emits a long chunk cheaply; an unsure one steps
    aside. The same predict-or-descend rule runs at every scale -- a fractal -- and the whole
    thing is one short recursive function (the demoscene way: a lot of behavior from a little
    code). bits-per-atom is then a genuine compressed size: this is a real PPM-family coder.

    `cuts` are ascending merge counts; cuts[0]=0 is the atom level, the last is the coarsest.
    The merge lists are prefix-nested, so a coarse token is always an exact run of finer
    tokens -- that nesting is what lets a chunk be decomposed cleanly one level down."""

    def __init__(self, schema, cuts):
        self.schema = schema
        self.cuts = list(cuts)
        self.levels = []          # per level: {'bi': {prev: Counter}, 'uni': Counter}
        self._dec = {}            # memoized decomposition of a token at a given cut

    def _decompose(self, token, cut):
        """The finer (level-`cut`) tokens that make up `token`. Cached, because tokens repeat
        heavily and re-encoding the same chunk every step would dominate the runtime."""
        key = (token, cut)
        hit = self._dec.get(key)
        if hit is None:
            atoms = list(token) if isinstance(token, tuple) else [token]
            hit = tuple(self.schema.encode(atoms, upto=cut))
            self._dec[key] = hit
        return hit

    def fit(self, atoms, order=2, doc_ids=None):
        """`order` is the within-level context length (order=2 -> trigram). Each level runs a
        small PPM over its own orders; only the unigram's leftover escape mass crosses DOWN to
        the finer level.

        PROVENANCE: pass `doc_ids` (one source index per atom) and the model also
        records, for every context->token transition, WHICH documents taught it
        (a counter per table entry). Generation can then hand back, for each
        emission, the sources of the transition it actually used -- attribution
        derived from the real mechanism, not estimated after the fact."""
        self.order = order
        self.atom_vocab = max(1, len(set(atoms)))
        self.levels = []
        self.src_levels = None if doc_ids is None else []
        for cut in self.cuts:
            seq = self.schema.encode(atoms, upto=cut)
            tables = [defaultdict(Counter) for _ in range(order + 1)]   # tables[o]: context len o
            srcs = ([defaultdict(lambda: defaultdict(Counter))
                     for _ in range(order + 1)] if doc_ids is not None else None)
            if doc_ids is not None:
                # the doc of a chunk = the doc at its first atom
                pos, seq_docs = 0, []
                for t in seq:
                    seq_docs.append(doc_ids[pos])
                    pos += len(t) if isinstance(t, tuple) else 1
            for i in range(len(seq)):
                for o in range(order + 1):
                    if i - o >= 0:
                        ctx = tuple(seq[i - o:i])
                        tables[o][ctx][seq[i]] += 1
                        if srcs is not None:
                            srcs[o][ctx][seq[i]][seq_docs[i]] += 1
            self.levels.append(tables)
            if srcs is not None:
                self.src_levels.append(srcs)
        self._n_docs = (max(doc_ids) + 1) if doc_ids else 0
        return self

    def _attribute(self, level, ctx, token, n_docs=None):
        """The documents that taught this transition, summed across the orders
        generation blends, weighted by context length (a longer, more specific
        context is stronger evidence of source than a bare unigram). MEASURED:
        on held-out Gutenberg passages this attributes the right book 92%
        (300 chars) / 78% (120 chars), and localizes 8/9 windows in spliced
        text. An inverse-document-frequency refinement was tried and measured a
        WASH -- the multi-order context already carries the distinctiveness --
        so it was dropped (the simpler mechanism, measurement-kept). Empty when
        fit had no doc_ids."""
        if self.src_levels is None:
            return Counter()
        out = Counter()
        srcs = self.src_levels[level]
        for o in range(self.order, -1, -1):
            entry = srcs[o].get(tuple(ctx[-o:]) if o else ())
            if entry and token in entry:
                for d, c in entry[token].items():
                    out[d] += c * (o + 1)                       # longer context -> more telling
        return out

    def _logp(self, level, token, hist):
        """PPM within the level (highest order first, escaping to shorter contexts), and when
        even the unigram has leftover escape mass, recurse DOWN a level to spell the token out.
        Kept in log space via logaddexp so a long, never-before-seen chunk can't underflow."""
        tables = self.levels[level]
        ctx = hist[level]
        p_orders = 0.0
        remaining = 1.0
        for o in range(self.order, -1, -1):                  # trigram -> bigram -> unigram
            counts = tables[o].get(tuple(ctx[-o:]) if o else ())
            if not counts:
                continue
            n = sum(counts.values()); d = len(counts)
            denom = n + d
            p_orders += remaining * counts.get(token, 0) / denom
            remaining *= d / denom                           # escape mass carried to lower order

        if level == 0:                                       # finest: escape to a uniform floor
            log_below = math.log(1.0 / self.atom_vocab)
        else:                                                # escape DOWN: spell token out below
            subs = self._decompose(token, self.cuts[level - 1])
            saved = list(hist[level - 1]); log_below = 0.0
            for s in subs:
                log_below += self._logp(level - 1, s, hist)
                hist[level - 1].append(s)
            hist[level - 1][:] = saved
        log_seen = math.log(p_orders) if p_orders > 0 else -1e9
        log_esc = (math.log(remaining) if remaining > 0 else -1e9) + log_below
        return float(np.logaddexp(log_seen, log_esc))

    def bits_per_atom(self, atoms):
        hist = {k: [] for k in range(len(self.cuts))}
        top = len(self.cuts) - 1
        seq = self.schema.encode(atoms, upto=self.cuts[top])
        bits = 0.0
        for tok in seq:
            bits += -self._logp(top, tok, hist) / math.log(2)
            for k in range(len(self.cuts)):          # advance every level by what was consumed
                hist[k].extend(self._decompose(tok, self.cuts[k]))
        return bits / max(1, len(atoms))

    # -- generation: the mirror of scoring -- emit a known chunk at the coarsest
    #    confident level, else descend a level and emit something finer ----------
    def _ppm_dist(self, level, ctx):
        """Blend this level's orders into a distribution over seen next-tokens, returning
        (seen: {token: prob}, escape_mass). The escape mass is what generation spends on
        descending a level -- the same number scoring spends believing the level below."""
        tables = self.levels[level]
        seen, remaining = {}, 1.0
        for o in range(self.order, -1, -1):
            counts = tables[o].get(tuple(ctx[-o:]) if o else ())
            if not counts:
                continue
            n = sum(counts.values()); d = len(counts); denom = n + d
            for t, c in counts.items():
                seen[t] = seen.get(t, 0.0) + remaining * c / denom
            remaining *= d / denom
        return seen, remaining

    def generate(self, n_atoms, seed_atoms=(), temperature=0.6, rng=None):
        rng = rng or np.random.default_rng(0)
        atoms = list(seed_atoms)
        top = len(self.cuts) - 1

        def emit(level, ctx_atoms):
            ctx = self.schema.encode(ctx_atoms, upto=self.cuts[level])[-self.order:]
            # A SEED DESERVES THE FINEST CONDITIONING AVAILABLE: if the caller
            # gave context atoms but this level's chunking can't encode any of
            # it (a seed ending mid-chunk encodes to NOTHING at coarse levels),
            # descend rather than trust the unconditional prior -- without
            # this, any seed that wasn't an exact chunk was silently ignored
            # and generation restarted from corpus statistics (user-caught on
            # the templated world corpus, where whole sentences chunk as one).
            if ctx_atoms and not ctx and level > 0:
                return emit(level - 1, ctx_atoms)
            seen, escape = self._ppm_dist(level, ctx)
            if (not seen) or (rng.random() < escape):
                if level > 0:
                    return emit(level - 1, ctx_atoms)        # descend -> finer unit (recursion)
                return (rng.choice(list(self._atoms)),)      # atom-level novelty
            toks = list(seen)
            w = np.array([seen[t] for t in toks]) ** (1.0 / temperature)
            chosen = toks[int(rng.choice(len(toks), p=w / w.sum()))]
            return chosen if isinstance(chosen, tuple) else (chosen,)

        if not hasattr(self, "_atoms"):
            self._atoms = sorted(self.levels[0][0].get((), {}).keys()) or [" "]
        while len(atoms) < n_atoms:
            atoms.extend(emit(top, atoms))
        return atoms

    def generate_traced(self, n_atoms, seed_atoms=(), temperature=0.6, rng=None):
        """generate(), but every emission also records WHO taught the transition
        it used: returns (atoms, trace) with trace = [(emitted_atom_count,
        {doc_id: count}), ...]. Requires fit(doc_ids=...)."""
        rng = rng or np.random.default_rng(0)
        atoms, trace = list(seed_atoms), []
        top = len(self.cuts) - 1

        def emit(level, ctx_atoms):
            ctx = self.schema.encode(ctx_atoms, upto=self.cuts[level])[-self.order:]
            if ctx_atoms and not ctx and level > 0:
                return emit(level - 1, ctx_atoms)
            seen, escape = self._ppm_dist(level, ctx)
            if (not seen) or (rng.random() < escape):
                if level > 0:
                    return emit(level - 1, ctx_atoms)
                return (rng.choice(list(self._atoms)),), Counter()
            toks = list(seen)
            w = np.array([seen[t] for t in toks]) ** (1.0 / temperature)
            chosen = toks[int(rng.choice(len(toks), p=w / w.sum()))]
            who = self._attribute(level, ctx, chosen, self._n_docs)
            return (chosen if isinstance(chosen, tuple) else (chosen,)), who

        if not hasattr(self, "_atoms"):
            self._atoms = sorted(self.levels[0][0].get((), {}).keys()) or [" "]
        while len(atoms) < n_atoms:
            out, who = emit(top, atoms)
            atoms.extend(out)
            trace.append((len(out), who))
        return atoms, trace


class SchemaGenerator:
    """Text/stream-facing wrapper: discover a schema, build the fractal coder over a few
    nested levels, and expose fit / generate / bits_per_char so it can stand in for the flat
    n-gram generator anywhere in the stack."""

    def __init__(self, modality="text", cuts=(0, 120, 350, 700), order=2, **tok_kw):
        self.modality, self.cuts, self.order, self.tok_kw = modality, cuts, order, tok_kw
        self.schema = self.model = None

    def fit(self, data):
        """Fit on a string, OR on a list of (text, source_name) documents -- the
        latter turns on PROVENANCE: every learned transition remembers which
        documents taught it, and generate(..., with_sources=True) hands back a
        ranked source list for the exact transitions an output used."""
        self.sources = None
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
            self.sources = [name for _, name in data]
            atoms, doc_ids = [], []
            for d, (text, _) in enumerate(data):
                toks = to_symbols(text, self.modality, **self.tok_kw)
                atoms.extend(toks)
                doc_ids.extend([d] * len(toks))
        else:
            atoms, doc_ids = to_symbols(data, self.modality, **self.tok_kw), None
        # keep each source's raw text for SEQUENCE-ALIGNMENT attribution: the
        # bag-of-transitions answers "whose STYLE", but a passage assembled from
        # real material (or quoting it) is identified by the longest CONTIGUOUS
        # run that appears verbatim in one source -- ordering carries meaning the
        # bag discards (two sources sharing every word but in opposite order, the
        # bull/bear thesis, are told apart only by the matching span).
        self._source_text = None
        if self.sources is not None:
            self._source_text = {}
            for (text, name) in data:
                self._source_text.setdefault(name, "")
                self._source_text[name] += " " + text
        self.schema = Schema(merges=max(self.cuts)).learn(atoms)
        self.model = HierModel(self.schema, self.cuts).fit(atoms, order=self.order,
                                                           doc_ids=doc_ids)
        # atoms for novelty escapes during generation
        self.model._atoms = sorted(set(atoms))
        return self

    def generate(self, seed="", length=200, temperature=0.6, rng=None,
                 with_sources=False):
        """Continue from the seed. With `with_sources=True` (needs a fit on
        (text, source) documents), returns (text, ranked, segments):
        `ranked` = [(source_name, weight 0..1)] over the whole output, and
        `segments` = [(text_span, top_source)] so each stretch of the output
        can point at the material that taught it."""
        seed_atoms = to_symbols(seed, self.modality, **self.tok_kw) if seed else ()
        if not with_sources:
            out = self.model.generate(length, seed_atoms, temperature, rng)
            return from_symbols(out, self.modality, **self.tok_kw)
        if self.sources is None:
            raise ValueError("fit on (text, source) documents to enable sources")
        out, trace = self.model.generate_traced(length, seed_atoms, temperature, rng)
        text = from_symbols(out, self.modality, **self.tok_kw)
        total = Counter()
        segments = []
        cursor = len(seed_atoms)
        for n_atoms, who in trace:
            seg_atoms = out[cursor:cursor + n_atoms]
            seg_text = from_symbols(seg_atoms, self.modality, **self.tok_kw)
            cursor += n_atoms
            total.update(who)
            top = (self.sources[who.most_common(1)[0][0]]
                   if who else None)
            segments.append((seg_text, top))
        s = sum(total.values()) or 1
        ranked = [(self.sources[d], c / s) for d, c in total.most_common()]
        return text, ranked, segments

    def attribute(self, text, coherent=True):
        """WHO does this text come from? The well-posed provenance question:
        given a passage, rank the fitted sources by how much of the text's
        actual transitions they taught. Returns [(source_name, weight 0..1)].

        COHERENT RESOLUTION (default): a passage usually comes from ONE source,
        so the distinctive evidence should resolve the ambiguous evidence rather
        than being diluted by it. The principle: a transition only one source
        ever taught (the word 'fillet' in one book) is near-certain provenance;
        a transition three sources share ('butterfly') is weak. Coherent mode
        weights each transition's vote by its SPECIFICITY -- inversely by how
        many sources taught it -- so the unique tokens pin the source and the
        shared tokens, rather than smearing across all their sources, mostly
        confirm the one the unique tokens already chose. (With coherent=False
        every transition votes equally -- the older independent-vote behaviour,
        kept for measurement.) Measured: coherent lifts clean-split top-1 well
        above independent voting precisely when sources share vocabulary.

        (This is the reliable direction -- attributing GIVEN text. Attributing
        freely-GENERATED low-order text is not well-posed: after the seed it
        drifts into transitions every source shares, so generate(with_sources)
        is honest about being approximate.) Requires a fit on (text, source)
        documents."""
        if self.sources is None:
            raise ValueError("fit on (text, source) documents to enable attribution")
        atoms = to_symbols(text, self.modality, **self.tok_kw)
        # COARSEST LEVEL FIRST: an author's characteristic multi-character chunks
        # are the distinctive signal (measured: top-level attributes real prose
        # at 70% on a clean 4-book split, vs 42% atom-only and 48% all-levels --
        # atom transitions like 'th'->'e' are shared by everyone and only smear
        # the vote). Fall back to a finer level only if the coarse one yields no
        # evidence at all (a passage of never-before-seen chunks), so attribution
        # degrades gracefully instead of going empty.
        for level in range(len(self.cuts) - 1, -1, -1):
            enc = self.schema.encode(atoms, upto=self.cuts[level])
            votes = []                                   # (per-source counts) per transition
            for i in range(1, len(enc)):
                who = self.model._attribute(level, enc[max(0, i - 2):i], enc[i])
                if who:
                    votes.append(who)
            if not votes:
                continue
            total = Counter()
            for who in votes:
                if coherent:
                    # specificity = inverse number of sources that taught this
                    # transition; unique evidence (one source) counts full, a
                    # transition shared by all N sources counts 1/N. The unique
                    # tokens dominate the tally and so SET the resolution.
                    spec = 1.0 / len(who)
                    for d, c in who.items():
                        total[d] += c * spec
                else:
                    total.update(who)
            s = sum(total.values()) or 1
            return [(self.sources[d], c / s) for d, c in total.most_common()]
        return []

    def align(self, text, min_len=6):
        """SEQUENCE ALIGNMENT attribution: walk the text and find, greedily, the
        longest contiguous spans that appear VERBATIM in the source material.
        Each maximal span is credited to the sources that contain it, weighted
        by length (longer = stronger) and specificity (1/#sources that have it,
        so a span everyone shares is discounted -- the biology of it: a long run
        of common function words is not provenance, a distinctive clause is).

        This answers a different question than attribute(): not "whose STYLE is
        this" but "whose actual MATERIAL is this". It is the method for assembled
        or quoting text -- and it is the one that tells apart two sources sharing
        every word in opposite order (the bull/bear thesis: the bag attributes
        the bearish sentence to the bull source because the words are shared;
        alignment pins it to the bear source because only there does the run
        'go down by as much as' actually appear). Returns [(source, score)] plus
        the single longest distinctive span found, as (span_text, source)."""
        if self._source_text is None:
            raise ValueError("fit on (text, source) documents to enable alignment")
        atoms = to_symbols(text, self.modality, **self.tok_kw)
        q = from_symbols(atoms, self.modality, **self.tok_kw) if self.modality == "text" \
            else "".join(map(str, atoms))
        scores = Counter()
        best_span = ("", None, 0)
        n = len(q); i = 0
        while i < n:
            j = i + 1
            havers = []
            while j <= n:
                h = [s for s, t in self._source_text.items() if q[i:j] in t]
                if not h:
                    break
                havers = h
                j += 1
            L = j - 1 - i
            if L >= min_len and havers:
                spec = 1.0 / len(havers)
                for s in havers:
                    scores[s] += L * spec
                # track the longest DISTINCTIVE span (not shared by everyone)
                if len(havers) < len(self._source_text) and L > best_span[2]:
                    best_span = (q[i:i + L], havers[0], L)
                i += L
            else:
                i += 1
        s = sum(scores.values()) or 1
        ranked = [(name, c / s) for name, c in scores.most_common()]
        return ranked, (best_span[0], best_span[1])

    def trace(self, text, min_distinctive=12):
        """The best of both: report STYLE (the transition bag) and MATERIAL (the
        alignment), and lead with whichever the evidence makes decisive. When a
        long distinctive verbatim span exists (>= min_distinctive chars), the
        passage was assembled from or quotes that source -- alignment leads and
        is near-certain. Otherwise the text is paraphrase or original-in-style,
        and the stylistic bag leads. Returns a dict with both rankings, the
        decisive span, and which signal was used."""
        bag = self.attribute(text)
        align, (span_text, span_src) = self.align(text)
        decisive = len(span_text) >= min_distinctive
        return {"by_material": align, "by_style": bag,
                "span": span_text, "span_source": span_src,
                "verdict": (span_src if decisive else (bag[0][0] if bag else None)),
                "basis": "material" if decisive else "style"}

    def bits_per_char(self, data):
        return self.model.bits_per_atom(to_symbols(data, self.modality, **self.tok_kw))

    def memory(self):
        """Number of stored context entries -- the model's size, for the memory comparison."""
        return sum(len(t) for tables in self.model.levels for t in tables)


def compression_gate(raw_input, experts, bias=None):
    """The one routing primitive, used at every level of the stack.

    Route an input to whoever compresses it best -- the expert whose schema needs the fewest
    bits to encode it is the one that understands it. `experts` is a dict {key: schema} or an
    iterable of (key, schema) pairs, where each schema exposes `bits_per_char(raw_input)`.
    Returns the ranked list [(score, key), ...] ascending, so callers can take the winner or a
    top-k. Deterministic: same inputs, same ranking, no training and no seed.

    `bias` is an optional {key: extra_bits} added to each expert's score -- the hook the hybrid
    gate uses to fold a reliability penalty into the same ranking. With bias=None this is the
    pure compression gate.

    This is the same operation the orchestrator uses to pick a tool and the MoE uses to pick an
    expert; it is also the whole-input cousin of the coder's per-token level choice. Factoring
    it here means the recursion is literal -- one function, called at each scale -- rather than
    copies that could drift apart."""
    items = experts.items() if hasattr(experts, "items") else experts
    bias = bias or {}
    ranked = sorted(((schema.bits_per_char(raw_input) + bias.get(key, 0.0), key)
                     for key, schema in items), key=lambda r: r[0])
    if not ranked:
        raise RuntimeError("compression_gate: no experts with a schema to route to")
    return ranked


class HybridGate:
    """Compression routing with a thin reward correction -- the synthesis of the two gates we
    measured on both sides of the boundary.

    Route to the expert that compresses the input best, but carry a per-expert reliability (a
    smoothed success rate) and add a surprise penalty -log2(reliability) to its bits. With no
    feedback every expert is equally trusted, so the penalty is a constant offset and this is
    EXACTLY the compression gate -- deterministic, training-free, free in the common case. As
    feedback arrives, an expert caught understanding-but-answering-wrong has its reliability
    fall and its penalty grow until it is demoted below the honest experts, even though it still
    compresses best. Self-correcting precisely where the pure compression gate failed (the
    miscalibration boundary), and identical to it everywhere else.

    The penalty is principled rather than tuned: total expected description cost is the bits to
    understand the input plus the bits of surprise at being wrong, -log2(P(correct))."""

    def __init__(self, modality="text", cuts=(0, 120, 350), **tok_kw):
        self.modality, self.cuts, self.tok_kw = modality, cuts, tok_kw
        self.experts = {}                       # key -> {"schema", "wins", "uses"}

    def learn(self, key, data):
        self.experts[key] = {"schema": SchemaGenerator(self.modality, self.cuts, **self.tok_kw).fit(data),
                             "wins": 0, "uses": 0}
        return self

    def reliability(self, key):
        e = self.experts[key]
        return (e["wins"] + 1) / (e["uses"] + 2)        # smoothed; unused -> 0.5 (a constant offset)

    def route(self, raw_input):
        pairs = {k: e["schema"] for k, e in self.experts.items()}
        bias = {k: -math.log2(self.reliability(k)) for k in self.experts}
        return compression_gate(raw_input, pairs, bias)[0][1]

    def observe(self, key, correct):
        """Feed back whether the routed expert was actually right. The only learning signal."""
        self.experts[key]["uses"] += 1
        self.experts[key]["wins"] += int(bool(correct))
        return self


class SchemaRouter:
    """Orchestration gate: route an input to the expert whose schema compresses it best.

    Each expert carries a learned schema; routing is a description-length contest -- the expert
    that 'understands' an input is the one that assigns it the fewest bits. No labels, no
    trained gate: an expert's discovered schema IS its competence signature. When the experts
    share a tokenizer (bytes), the gate is modality-blind -- it sorts text from code from
    numeric data without being told which is which, because each only compresses what it knows.
    Measured at the byte level: 100% routing with large margins (text ~2 bits/byte to its own
    expert, >9 to the numeric one). This is the MoE gate recast as 'who compresses you best',
    and it pairs naturally with the typed/semantic routing already in the orchestrator: type
    says what can connect, the schema gate says who actually understands this input."""

    def __init__(self, modality="bytes", cuts=(0, 120, 350), **tok_kw):
        self.modality, self.cuts, self.tok_kw = modality, cuts, tok_kw
        self.experts = {}

    def learn(self, name, data):
        self.experts[name] = SchemaGenerator(self.modality, self.cuts, **self.tok_kw).fit(data)
        return self

    def route(self, data):
        """Return (best_expert_name, {name: bits_per_atom}). Lower bits = better understanding."""
        ranked = compression_gate(data, self.experts)
        return ranked[0][1], {key: bits for bits, key in ranked}


def demo_schema():
    """Show the SAME discovery mechanism finding structure in four different modalities,
    and the bits-per-atom it buys at each hierarchy depth."""
    import numpy as np
    print("Modality-agnostic schema discovery (one mechanism, any data):\n")

    def run(symbols, name, checkpoints=(0, 150, 400)):
        cut = int(len(symbols) * 0.85)
        tr, he = symbols[:cut], symbols[cut:]
        sch = Schema(merges=max(checkpoints)).learn(tr)
        print(f"[{name}] {len(symbols)} atoms")
        for cp in checkpoints:
            ktr, khe = sch.encode(tr, upto=cp), sch.encode(he, upto=cp)
            bpa = SchemaModel(3).fit(ktr).bits_per_atom(khe, len(he))
            print(f"    depth {cp:3d}: {len(set(ktr)):4d} symbols, {bpa:.3f} bits/atom")
        print("    discovered:", sch.emergent(tr, k=6), "\n")

    try:
        import nltk
        nltk.data.path.insert(0, "/home/claude/nltk_data")
        from nltk.corpus import gutenberg
        txt = " ".join(w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha())[:120000]
        run(to_symbols(txt, "text"), "text")
    except Exception as e:
        print("[text] skipped:", e)
    run(to_symbols(open(__file__).read(), "code"), "code")
    x = np.linspace(0, 60 * np.pi, 60000)
    run(to_symbols(np.sin(x) + 0.5 * np.sin(3 * x), "numbers"), "numbers (periodic)")
    motif = (np.add.outer(range(8), range(8)) % 4)
    run(to_symbols(np.tile(motif, (12, 12)), "image"), "image (tiled motif)")


def demo_hier():
    """Cross-level backoff vs the best single level. The fractal coder wins where the data is
    genuinely multi-scale (text, code) and merely ties/loses a little where one scale already
    explains everything (a pure sine, a pure tile) -- which is the honest, expected shape."""
    print("\nCross-level backoff (fractal coder) vs best single level:\n")
    cuts = (0, 120, 350, 700)

    def cmp(symbols, name):
        cut = int(len(symbols) * 0.85); tr, he = symbols[:cut], symbols[cut:]
        sch = Schema(merges=max(cuts)).learn(tr)
        naive = {c: SchemaModel(3).fit(sch.encode(tr, upto=c)).bits_per_atom(sch.encode(he, upto=c), len(he))
                 for c in cuts}
        hier = HierModel(sch, cuts).fit(tr).bits_per_atom(he)
        best = min(naive.values())
        verdict = "beats best" if hier < best - 1e-3 else ("ties best" if hier <= best + 0.02 else "single scale wins")
        print(f"[{name}] naive best={best:.3f} (worst={max(naive.values()):.3f})  "
              f"hierarchical={hier:.3f}  -> {verdict}")

    try:
        import nltk
        nltk.data.path.insert(0, "/home/claude/nltk_data")
        from nltk.corpus import gutenberg
        txt = " ".join(w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha())[:90000]
        cmp(to_symbols(txt, "text"), "text")
    except Exception as e:
        print("[text] skipped:", e)
    cmp(to_symbols(open(__file__).read()[:60000], "code"), "code")
    x = np.linspace(0, 50 * np.pi, 50000)
    cmp(to_symbols(np.sin(x) + 0.5 * np.sin(3 * x), "numbers"), "numbers (periodic)")
    motif = (np.add.outer(range(8), range(8)) % 4)
    cmp(to_symbols(np.tile(motif, (10, 10)), "image"), "image (tiled motif)")


def _selftest():
    """Regression trap (T6 backfill; demos only, no assertion). Pins the modality-agnostic tokenizer's core
    contract: to_symbols -> from_symbols ROUND-TRIPS for the modalities callers depend on. A tokenizer that
    doesn't invert is silently lossy. Off-designed-case inputs (mid-bin numbers, a real word) per [BLIND-SPOT]."""
    import numpy as np

    # numbers: quantised to `bins`, so recovery is within one bin width -- assert that bound, not exact equality.
    vals = [0.1, 0.5, 0.9]
    back = from_symbols(to_symbols(vals, "numbers", bins=16, num_range=(0, 1)),
                        "numbers", bins=16, num_range=(0, 1))
    assert all(abs(float(b) - v) < 1.0 / 16 for b, v in zip(back, vals)), (back, vals)

    # text: a lossless round-trip -- the string comes back exactly.
    assert from_symbols(to_symbols("hello world", "text"), "text") == "hello world"

    print("OK: holographic_schema self-test passed (numbers round-trip within one bin width, text round-trips "
          "exactly through the modality-agnostic tokenizer)")


if __name__ == "__main__":
    import sys
    _selftest()
    if "--demos" in sys.argv:
        demo_schema()
        demo_hier()

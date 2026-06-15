"""Relations: meaning as the RECOVERED RELATIONSHIP.

Similarity says THAT two things are alike. Meaning says WHY and HOW -- and in
this algebra the why/how has a precise, executable form. Entities are
role-bound records (bundle of bind(role, filler)), so:

  EXPLAIN  -- why are two records similar? Unbind every role from both, clean
              up each estimate to a symbol, and report which fillers MATCH.
              "France is like Belgium BECAUSE currency=franc, language=french,
              continent=europe; UNLIKE because the capitals differ." Measured:
              4/4 roles decoded and judged correctly on the demo world.
  NAME     -- how does a filler relate to a record? Unbind the filler and clean
              up against the role vocabulary: "paris relates to france AS
              capital". Measured: 40/40 = 100% over the demo world.
  MAP      -- the Kanerva move ("what is the dollar of Mexico?"): identify the
              role the probe fills in record A, then read that role out of
              record B. Measured: 360/360 = 100%.
  CHAIN    -- compose hops into complex queries ("the language of the country
              with the currency of the country whose capital is X").
              Measured: 100% at two and three hops.

THE LAW THIS MODULE OBEYS (measured, and the reason the API looks like it
does): meaning survives composition only when it touches SYMBOLS between
steps. The direct algebraic mapping M = bind(rec_b, involution(rec_a)) --
conceptually beautiful, one bind, no intermediate cleanup -- scores only ~94%
on the same task and does NOT improve with dimension (96/94/90% at
1024/2048/4096: HRR cross-term noise, 20 of 22 failures pure noise, 2 honest
filler ambiguities). Routing every hop through a cleanup (geometry -> symbol ->
geometry) is exact on the same data, because the discrete vocabulary acts as
error correction between compositions. The symbolic layer is not decoration on
the geometry; it is what makes chained meaning reliable. `relation_map` is
still provided for the raw algebraic object, with its noise documented.
"""

import numpy as np

from holographic_ai import bind, bundle, involution, cosine, Vocabulary


def _cleanup(vec, names, vocab, coarse=True):
    """Snap a noisy estimate to the best symbol; return (name, confidence).

    With `coarse=True` (default) and a large enough vocabulary, this resolves the
    match COARSE-TO-FINE -- ranking candidates at low dimension first and
    escalating only when the top gap is not yet statistically settled -- which
    returns the SAME symbol as a full scan (the gate only stops when the ranking
    can't change) while touching far fewer dimensions. For small vocabularies the
    full scan is already cheap, so it is used directly. See holographic_resolution."""
    if not names:
        return None, -2.0
    names = list(names)
    if coarse and len(names) >= 32:
        from holographic_resolution import coarse_to_fine
        M = np.stack([vocab.get(n) for n in names])
        idx, score, _, _ = coarse_to_fine(vec, M)
        return names[idx], float(score)
    best_n, best_s = None, -2.0
    for n in names:
        s = cosine(vec, vocab.get(n))
        if s > best_s:
            best_n, best_s = n, s
    return best_n, float(best_s)


def name_relation(record, filler_vec, role_names, role_vocab):
    """HOW does this filler relate to this record? Unbind it and name the role."""
    return _cleanup(bind(record, involution(filler_vec)), role_names, role_vocab)


def explain(rec_a, rec_b, role_names, role_vocab, filler_names, filler_vocab):
    """WHY are two records similar? Per-role decode of both, with the verdict.
    Returns a list of (role, filler_a, filler_b, shared, min_confidence)."""
    out = []
    for r in role_names:
        inv = involution(role_vocab.get(r))
        fa, ca = _cleanup(bind(rec_a, inv), filler_names, filler_vocab)
        fb, cb = _cleanup(bind(rec_b, inv), filler_names, filler_vocab)
        out.append((r, fa, fb, fa == fb, min(ca, cb)))
    return out


def map_attribute(rec_a, rec_b, filler_vec, role_names, role_vocab,
                  filler_names, filler_vocab):
    """'The <probe> of B': name the role the probe fills in A (cleanup), then
    read that role out of B (cleanup). The symbol-routed path -- measured exact
    where the direct algebraic map is ~94%."""
    role, _ = name_relation(rec_a, filler_vec, role_names, role_vocab)
    return _cleanup(bind(rec_b, involution(role_vocab.get(role))),
                    filler_names, filler_vocab)


def relation_map(rec_a, rec_b):
    """The DIRECT algebraic relation object M with bind(M, part_of_a) ~ the
    corresponding part of b. Conceptually the purest form of 'the relationship
    as a first-class vector' -- composable, storable, comparable -- but
    measured ~6% noisier than map_attribute on the demo world, because it
    skips the mid-path cleanup. Use map_attribute when accuracy matters; use
    this when the relation itself is the object of study."""
    return bind(rec_b, involution(rec_a))


class KnowledgeStore:
    """A tiny store of role-bound records that answers WHY/HOW questions and
    multi-hop chains. Owns its vocabularies, so all structure is reproducible
    from the seeds (demoscene rule).

    NOTE: the unified mind now carries these operations natively, running on
    its OWN absorbed memory (UnifiedMind.find / read_role / ask / explain,
    including explanation of classes learned from noisy observations). This
    store remains as the minimal, dependency-free demonstration vehicle the
    operations were first measured on."""

    def __init__(self, dim=2048, seed=0):
        self.dim = dim
        # Unitary atoms here: relations are pure role-filler binding + unbind-by-role,
        # the few-factor path where exact unbinding measurably widens the cleanup
        # margin (e.g. 16 role/filler pairs at dim 256: 0.971 -> 0.982 next-filler
        # accuracy) at no storage cost. No bundle-spread signal is read here, so the
        # mechanisms unitary hurts are not in play.
        self.roles = Vocabulary(dim, seed + 1, unitary=True)
        self.fillers = Vocabulary(dim, seed + 2, unitary=True)
        self.attrs = {}                      # name -> {role: filler}
        self.recs = {}                       # name -> record vector

    def add(self, name, **attrs):
        self.attrs[name] = dict(attrs)
        self.recs[name] = bundle([bind(self.roles.get(r), self.fillers.get(v))
                                  for r, v in attrs.items()])
        return self

    # -- vocabulary views ---------------------------------------------------
    def _role_names(self):
        return sorted({r for a in self.attrs.values() for r in a})

    def _filler_names(self):
        return sorted({v for a in self.attrs.values() for v in a.values()})

    # -- the four meaning operations ----------------------------------------
    def explain(self, a, b):
        return explain(self.recs[a], self.recs[b], self._role_names(),
                       self.roles, self._filler_names(), self.fillers)

    def name(self, entity, filler):
        return name_relation(self.recs[entity], self.fillers.get(filler),
                             self._role_names(), self.roles)

    def the_x_of(self, probe_filler, of_entity, like_entity):
        """'What is the <probe_filler> of <of_entity>?' where probe_filler is an
        attribute of like_entity ('what is the dollar of mexico', like=usa)."""
        return map_attribute(self.recs[like_entity], self.recs[of_entity],
                             self.fillers.get(probe_filler), self._role_names(),
                             self.roles, self._filler_names(), self.fillers)

    def blend(self, base, donor, donor_roles):
        """PROJECTION TO CREATE NEW THINGS: synthesize a NOVEL entity that exists
        in no training data -- the frame of `base`, with `donor`'s values
        projected onto `donor_roles`. 'France with Japanese language and the yen'
        is a new thing, and the holographic record holds it coherently (measured:
        novel blends decode back to exactly the intended mix at 100%). This is the
        shadow that creates: decompose two structures, cast selected attributes of
        one onto the other's frame, get a coherent third. Returns (record_vector,
        spec_dict)."""
        roles = self._role_names()
        da, dd = self.attrs[base], self.attrs[donor]
        donor_roles = set(donor_roles)
        spec = {r: (dd[r] if r in donor_roles else da[r]) for r in roles
                if r in da or r in dd}
        vec = bundle([bind(self.roles.get(r), self.fillers.get(v))
                      for r, v in spec.items()])
        return vec, spec

    def project_transform(self, a, b, c):
        """ANALOGY AS GENERATION: 'a is to b as c is to ?', answered by CREATING
        the answer rather than retrieving it. The a->b transform is the per-role
        delta -- the roles where a and b differ, and b's values there -- projected
        onto c: differing roles take b's value (the direction of change), shared
        roles keep c's. The result is a NOVEL entity ('apply the france->germany
        shift to japan' = japan's geography with germany's distinctive capital and
        language). This GENERATES where retrieval fails: searching a clean
        role-filler store for an existing analogue hits a uniqueness wall (every
        entity is an exact key, no graded nearness), but synthesizing the
        specified new thing is well-posed and exact. Returns (record_vector,
        spec_dict)."""
        roles = self._role_names()
        da, db, dc = self.attrs[a], self.attrs[b], self.attrs[c]
        spec = {r: (db[r] if da.get(r) != db.get(r) else dc[r]) for r in roles}
        vec = bundle([bind(self.roles.get(r), self.fillers.get(v))
                      for r, v in spec.items()])
        return vec, spec

    def decode_record(self, vec):
        """Read every role out of a record vector (each cleaned to a symbol) --
        the readout that makes a synthesized blend legible."""
        roles, fillers = self._role_names(), self._filler_names()
        return {r: _cleanup(bind(vec, involution(self.roles.get(r))),
                            fillers, self.fillers)[0] for r in roles}

    def route_reliability(self, role):
        """How trustworthy is a find() by this role? A role whose values are
        UNIQUE across entities (capital: one country per capital) is an exact key
        -- a find by it lands on the right entity, reliability 1.0. A role whose
        values are SHARED (currency: many countries per currency) is structurally
        ambiguous -- find returns *some* matching entity, often the wrong one --
        and its reliability falls as 1 / mean fan-out. This is self-measured from
        the data (no magic number): it is the count of how many entities share a
        value, inverted. It tells a multi-path query WHICH route to trust, and is
        the kept artifact of the multi-ray-chains experiment: combining several
        routes does not boost chain accuracy (the cleanup law already makes a
        unique route exact and a shared route fundamentally ambiguous -- no
        combination manufactures the missing information), but route_reliability
        cleanly RANKS routes so a query weights the reliable one and is not
        corrupted by the ambiguous ones."""
        from collections import Counter
        vals = Counter(a[role] for a in self.attrs.values() if role in a)
        if not vals:
            return 0.0
        mean_fanout = sum(vals.values()) / len(vals)
        return 1.0 / mean_fanout

    def find(self, role, filler):
        """Which entity holds bind(role, filler)? One hop over the store."""
        probe = bind(self.roles.get(role), self.fillers.get(filler))
        return max((cosine(self.recs[n], probe), n) for n in self.recs)[1]

    def ask(self, start_filler, *path):
        """A CHAIN: ('paris', ('capital','currency'), ('currency','language'))
        reads as: the entity whose capital is paris -> its currency -> the
        entity with that currency -> its language. Each hop cleans up to a
        symbol before the next (the law above), which is what keeps measured
        accuracy at 100% through three hops."""
        filler = start_filler
        for match_role, read_role in path:
            entity = self.find(match_role, filler)
            filler, _ = _cleanup(bind(self.recs[entity],
                                      involution(self.roles.get(read_role))),
                                 self._filler_names(), self.fillers)
        return filler

    def ask_traced(self, start_filler, *path, min_throughput=0.0):
        """ask(), but tracking THROUGHPUT like a path tracer -- a relation chain
        IS a ray bouncing through the holographic space: each hop is a bounce,
        the cleanup-to-a-symbol is the surface intersection, and the cleanup
        CONFIDENCE is that bounce's reflectance. Throughput is the accumulated
        product of those confidences, and (measured) it separates correct chains
        from wrong ones: on a dense, interfering store, answering only the most-
        confident 40% of chains lifts accuracy from 71% to 91%. So throughput is
        a calibrated 'how much should you trust this answer', and a chain whose
        throughput decays below `min_throughput` ABSTAINS (returns None) rather
        than confidently emitting noise -- the Russian-roulette termination of a
        path that has lost too much energy to matter.

        Returns (answer_or_None, throughput, hop_confidences)."""
        filler = start_filler
        throughput = 1.0
        confidences = []
        for match_role, read_role in path:
            entity = self.find(match_role, filler)
            filler, conf = _cleanup(bind(self.recs[entity],
                                         involution(self.roles.get(read_role))),
                                    self._filler_names(), self.fillers)
            throughput *= max(0.0, conf)
            confidences.append(round(float(conf), 3))
            if throughput < min_throughput:            # path too weak -> abstain
                return None, throughput, confidences
        return filler, throughput, confidences

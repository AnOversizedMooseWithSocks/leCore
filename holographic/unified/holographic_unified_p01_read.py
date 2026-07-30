"""Part 01 of UnifiedMind's faculty surface -- 97 methods, read .. assemble_pipeline.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart01:

    # -- perception (the single front door) --------------------------------
    def read(self, corpus):
        """Pre-learn word co-occurrence so text perceptions carry meaning."""
        self.encoder.learn_text(corpus)
        return self

    def learn_dictionary(self, definitions, iters=3, alpha=0.7):
        """LANGUAGE CURRICULUM, layer 1 -- learn word MEANING from a dictionary,
        natively, into the mind's own text encoder. A word's meaning is the bundle
        of its definition words' meanings; a dictionary is self-referential, so
        this is a fixed-point iteration on the definition graph (the resonator
        dynamic applied to a lexicon). Measured separately to peak around three
        passes before over-diffusing -- so the default is three. After this the
        encoder's word vectors carry definitional meaning, which every downstream
        text perception then inherits.

        definitions: {word: [words in its definition]}. Returns self."""
        from holographic.agents_and_reasoning.holographic_ai import random_vector
        rng = np.random.default_rng(0)
        words = sorted(definitions)
        wset = set(words)
        defs = {w: [d for d in definitions[w] if d in wset and d != w] for w in words}
        base = {w: random_vector(self.dim, rng) for w in words}     # atomic ids
        meaning = dict(base)
        for _ in range(max(1, iters)):                              # the recursion
            nxt = {}
            for w in words:
                if defs[w]:
                    v = np.sum([meaning[d] for d in defs[w]], axis=0)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    v = alpha * v + (1 - alpha) * base[w]           # damp toward identity
                    nxt[w] = v / (np.linalg.norm(v) + 1e-12)
                else:
                    nxt[w] = meaning[w]
            meaning = nxt
        # write the bootstrapped meaning into the encoder's word-vector store, so
        # the brain's perception of these words now carries definitional meaning
        for w, v in meaning.items():
            self.encoder._text.context[w] = v.copy()
        self._lexicon_words = wset
        return self

    def define(self, word, k=5):
        """The nearest words by learned meaning -- 'what is this word like?',
        answered from the dictionary-bootstrapped vectors. Returns [(word, sim)].
        Empty if the word was never in the learned dictionary (so an unknown word
        yields no spurious neighbours)."""
        lex = getattr(self, "_lexicon_words", set())
        if word not in lex:
            return []
        wv = self.encoder._text.wordvec(word)
        if wv is None:
            return []
        out = []
        for w in lex:
            if w == word:
                continue
            ov = self.encoder._text.wordvec(w)
            if ov is not None:
                out.append((w, float(wv @ ov / ((np.linalg.norm(wv) * np.linalg.norm(ov)) + 1e-12))))
        return sorted(out, key=lambda t: -t[1])[:k]

    def lookup(self, word):
        """Look a word up in the VENDORED DICTIONARY (~144k English words): its definition, part of speech, synonyms,
        an example, and its 'is a kind of' parent. This is real world-knowledge the engine carries with it -- distinct
        from define(), which returns nearest words by LEARNED meaning. Returns None for an unknown word.
        See holographic_dictionary."""
        import holographic.misc.holographic_dictionary as _dict
        return _dict.entry(word)

    def word_taxonomy(self, word):
        """The 'what kind of thing is this?' chain from the vendored dictionary's taxonomy: e.g. 'dog' -> ['domestic
        animal', 'animal', 'organism', ... 'entity']. Contextual grounding straight from the dictionary's is_a
        hierarchy. See holographic_dictionary.hypernym_chain."""
        import holographic.misc.holographic_dictionary as _dict
        return _dict.hypernym_chain(word)

    def dictionary_size(self):
        """How many words the vendored dictionary holds (and its source/license via .manifest())."""
        import holographic.misc.holographic_dictionary as _dict
        return _dict.size()

    def learn_vocabulary(self, vocab, iters=3, alpha=0.7):
        """Bootstrap the mind's word meanings from the VENDORED DICTIONARY over a given vocabulary: builds the
        {word: [definition words]} map from real definitions and runs learn_dictionary on it. So instead of supplying
        your own dictionary, the mind can learn meaning from the batteries-included one. Returns self."""
        import holographic.misc.holographic_dictionary as _dict
        return self.learn_dictionary(_dict.definition_map(vocab), iters=iters, alpha=alpha)

    def learn_encyclopedia(self, facts, maintain=True):
        """LANGUAGE CURRICULUM, layer 3 -- learn RELATIONAL knowledge (an
        encyclopedia) natively, by absorbing each concept as a role-bound record
        into the mind's OWN memory. `facts` is {concept: {role: filler}}, e.g.
        {'dog': {'is_a': 'canine'}, ...}. After this the mind can climb is_a
        chains, test taxonomic membership, and find structural relatedness using
        the SAME find/ask machinery it uses for every other record -- the
        encyclopedia is not a side table, it is in the brain. Returns self."""
        self._encyclopedia = dict(facts)
        for concept, rel in facts.items():
            self.learn(dict(rel), concept, modality="record")
        if maintain:
            self.maintain_now()
        return self

    def climb(self, concept, role="is_a", hops=99, min_throughput=0.0, hop_discount=0.9):
        """Walk a relation chain (default is_a) up through the absorbed
        encyclopedia, as a path-traced ray over the mind's own memory. Returns
        (chain, throughput); a chain whose throughput would fall below
        min_throughput stops rather than emitting a low-confidence deeper hop.

        Each hop applies an explicit `hop_discount` (<1): a deduction reached through
        more inference steps is less certain. With exact (unitary-atom) unbinding each
        hop is near-lossless, so this depth penalty is stated deliberately rather than
        emerging from unbinding noise -- the 'how far has this traveled' signal is
        intended. hop_discount=1.0 disables it."""
        chain = [concept]
        cur = concept
        throughput = 1.0
        for _ in range(hops):
            filler, conf = self.read_role(cur, role) if cur in self._class_labels() else (None, 0.0)
            if filler is None:
                break
            t = throughput * max(0.0, float(conf)) * hop_discount   # explicit depth penalty
            if t < min_throughput:
                break
            throughput = t
            chain.append(filler)
            cur = filler
        return chain, throughput

    def is_a(self, concept, ancestor, role="is_a"):
        """Taxonomic membership over the absorbed encyclopedia: does `concept`
        reach `ancestor` by following `role`? Returns (reached, hops, throughput)."""
        chain, tp = self.climb(concept, role=role)
        if ancestor in chain:
            return True, chain.index(ancestor), tp
        return False, -1, tp

    def _class_labels(self):
        return set(self.memory.live.labels())

    def answer(self, question):
        """A QUESTION ROUTER -- the honest middle ground between 'completes your
        sentence' and 'is a chatbot'. This mind is NOT a language model and does
        not converse; but it holds real knowledge, and most questions have a
        SHAPE that maps to one of its actual operations. This recognizes a handful
        of question forms by template (keyword matching, not natural-language
        understanding -- it says so), pulls out the argument, and answers from the
        brain's own knowledge:

          'what is a dog?' / 'define dog' / 'what is dog like?'
                -> define()  (nearest words by learned meaning)
                   + climb() (its is_a chain, if an encyclopedia was learned)
          'is a dog an animal?'      -> is_a()   (taxonomic membership)
          'what is the capital of france?' / 'capital of france'
                -> read_role()        (a role of a known concept)
          'what is this: <text>' / 'classify <text>' / 'what kind of text is ...'
                -> classify()         (nearest learned category)
          'what is like <text>'      -> recall() (nearest individual memory)

        Anything it cannot map falls through to generation (sentence completion),
        clearly LABELLED as a completion rather than an answer, so the system is
        never pretending to answer when it is really just continuing text.

        Returns {kind, ...} describing which operation answered and the result."""
        import re
        q = (question or "").strip()
        ql = q.lower().rstrip("?.! ").strip()
        if not ql:
            return {"kind": "none", "text": "ask me something"}

        # -- 'is X a Y?' / 'is X an Y?' -> taxonomic membership ----------------
        m = re.match(r"^(?:is|are)\s+(?:a|an|the)?\s*(.+?)\s+(?:a|an|the)?\s*(\S+)$", ql)
        # The guard asks "was a curriculum encyclopedia ever TAUGHT?" -- without one there is no taxonomy to walk,
        # and answering `is_a: False` from an empty memory is a fabrication, not an answer. Written as an explicit
        # getattr against the curriculum DICT rather than hasattr(): a lazily-instantiating property named
        # `_encyclopedia` once shadowed this attribute and pinned the flag permanently True (see the encyclopedia
        # faculty below). An existence check that a mere name collision can satisfy is not a check.
        if m and getattr(self, "_encyclopedia", None) is not None:
            x, y = m.group(1).strip().split()[-1], m.group(2).strip()
            reached, hops, tp = self.is_a(x, y)
            return {"kind": "is_a", "subject": x, "ancestor": y,
                    "answer": bool(reached), "hops": hops, "throughput": round(float(tp), 3),
                    "chain": self.climb(x)[0]}

        # -- 'capital of france' / 'what is the <role> of <concept>' ----------
        m = (re.match(r"^what\s+is\s+the\s+(\w+)\s+of\s+(.+)$", ql)
             or re.match(r"^(\w+)\s+of\s+(.+)$", ql))
        if m:
            role, concept = m.group(1).strip(), m.group(2).strip().split()[-1]
            if concept in self._class_labels() and role in getattr(self, "_fillers", {}):
                val, conf = self.read_role(concept, role)
                if val is not None:
                    return {"kind": "role", "concept": concept, "role": role,
                            "value": val, "confidence": round(float(conf), 3)}

        # -- 'what is like <text>' -> recall nearest individual ---------------
        m = re.match(r"^what(?:'s| is)?\s+like\s+(.+)$", ql)
        if m:
            (lab, _), score = self.recall(m.group(1).strip())
            return {"kind": "recall", "label": lab, "score": round(float(score), 3)}

        # -- 'what is X' / 'define X' / 'what is X like' -> meaning + is_a -----
        m = (re.match(r"^define\s+(.+)$", ql)
             or re.match(r"^what\s+is\s+(?:a|an|the)?\s*(.+?)(?:\s+like)?$", ql)
             or re.match(r"^what\s+(?:is|are)\s+(.+)$", ql))
        if m:
            word = m.group(1).strip().split()[-1]
            near = self.define(word, 5) if hasattr(self, "define") else []
            chain = self.climb(word)[0] if hasattr(self, "climb") else [word]
            if near or len(chain) > 1:
                return {"kind": "define", "word": word,
                        "meaning": [(w, round(s, 3)) for w, s in near],
                        "is_a_chain": chain}

        # -- 'classify <text>' / 'what kind of text is <text>' ----------------
        m = (re.match(r"^classify[:\s]+(.+)$", ql)
             or re.match(r"^what\s+(?:kind|category|genre|type)\s+(?:of\s+\w+\s+)?is[:\s]+(.+)$", ql))
        if m:
            label, score = self.classify(m.group(1).strip())
            return {"kind": "classify", "label": label, "score": round(float(score), 3)}

        # -- natural / verbose phrasing the templates missed: the VSA-native router (a blend of the
        #    question's word meanings -> intent, then a known-concept scan WITH ORDER for the args).
        #    Tried only AFTER the exact templates, so anything they matched is byte-for-byte unchanged.
        try:
            from holographic.agents_and_reasoning.holographic_intent import route_question as _vsa_route
            _vsa = _vsa_route(self, q)
        except Exception:
            _vsa = None
        if _vsa is not None:
            return _vsa

        # -- nothing matched: this is the sentence-completion path, labelled ---
        try:
            text = self.generate(q if q.endswith(" ") else q + " ", length=80)
        except Exception:
            text = None
        if text:
            return {"kind": "completion",
                    "note": "I don't recognize this as a question I can answer from "
                            "knowledge, so I'm continuing the text instead (this is "
                            "generation, not an answer).",
                    "text": text}
        return {"kind": "unknown",
                "note": "I can't map this to something I know. Try 'what is a dog?', "
                        "'is a dog an animal?', 'define wolf', or 'what is the capital "
                        "of france?' -- or load a corpus and I can complete text."}

    def answer_text(self, question):
        """Answer a question as a short, CONSTRUCTED sentence -- the surface-realization layer the
        engine lacked. Delegates retrieval to answer() (which routes the question to the brain's
        real operations: is_a chains, role lookups, learned-meaning similarity, classification),
        then realizes the result into one coherent sentence (holographic_answer.realize_answer).

        The three properties it holds, honestly: ACCURATE (content is retrieved, never invented),
        CONTEXTUAL (the form matches the question shape), and NOT VERBATIM (the sentence is built
        from the retrieved structure, so it is a new sentence -- verbatim only where the answer
        simply IS a stored value, e.g. a capital). When the mind does not know -- an unknown
        concept, a low-confidence recall/classify, or a question that falls through to the
        generation path -- it ABSTAINS with an honest 'I don't know' rather than fabricating.

        This deliberately uses the parts of the engine that WORK (relational retrieval + learned
        distributed meaning + calibrated abstention) and NOT the free n-gram generator, which the
        text-generation review measured to be locally fluent but globally incoherent. Returns a
        string."""
        from holographic.agents_and_reasoning.holographic_answer import realize_answer
        return realize_answer(self.answer(question))

    def perceive(self, x, modality=None):
        """Any input -> one vector in the shared space. This is the only encoder in the
        system; the memory and the brain never encode anything themselves."""
        return self.encoder.encode(x, modality)

    def hypervector(self, x, modality=None, tag=None):
        """The same encode, but returned as a first-class Hypervector (consolidation D1): the raw array plus its
        dim / encoder / tag, with the five verbs (bind/unbind/bundle/cleanup/permute) as methods. The encoder is the
        'constructor'; the raw array stays one attribute away (.array / np.asarray(hv)). See holographic_hypervector.
        """
        from holographic.sampling_and_signal.holographic_hypervector import Hypervector
        return Hypervector(self.encoder.encode(x, modality), encoder=self.encoder,
                           tag=tag if tag is not None else repr(x)[:40])

    def adaptive_record(self, expected_pairs, exact=False, max_numbers=None, seed=0):
        """A role->filler memory whose representation is GATED BY LOAD and FIDELITY NEED (the FHRR/tensor re-enables):
        cheap real-HRR at low load, FHRR phasors past the capacity knee, or tensor-product binding for EXACT recall
        (perfect to M~dim, at dim*dim storage) when `exact=True` and the D*D budget fits `max_numbers`. Uniform
        add/recall. The deciders are exact integers/flags, and there is no harm mode on recall (FHRR >= real-HRR,
        tensor is exact in-regime), so the gate just avoids paying for capacity until it's worth it. See
        holographic_loadmemory."""
        from holographic.simulation_and_physics.holographic_loadmemory import AdaptiveRoleFillerMemory
        return AdaptiveRoleFillerMemory(self.encoder.dim, expected_pairs, exact=exact, max_numbers=max_numbers, seed=seed)

    # -- axial perception (orientation-like values; holographic_mobius via the encoder) ----
    # An AXIAL value is one where theta and theta+pi mean the SAME thing -- the orientation of an
    # unoriented line, a director field, a crystal axis. modality="axial" encodes it on the Mobius
    # base (the double-angle map), so learn / classify / recall over orientations no longer treat a
    # value and its pi-flip as different. It is OPT-IN: declare modality="axial" (a bare float still
    # infers as "number" -- the scalar encoder, which has no notion that theta and theta+pi are the
    # same orientation and simply encodes them as two unrelated values).
    def axial_similarity(self, a, b):
        """Cosine similarity of two axial values (radians): ~+1 when they are the same orientation,
        INCLUDING the theta vs theta+pi case (a pi flip is invisible). Contrast self.perceive on the
        plain 'number' modality, which encodes theta and theta+pi as two unrelated values and so does
        NOT recognize a pi flip as the same orientation."""
        from holographic.agents_and_reasoning.holographic_ai import cosine
        return cosine(self.perceive(a, "axial"), self.perceive(b, "axial"))

    def damage_mask(self, destroy_fraction, seed=0, dim=None):
        """GRACEFUL-DEGRADATION PROBE: a keep-mask that zeroes a random `destroy_fraction` of a vector's slots.
        Multiply a stored hypervector by it to simulate real damage -- a scratched plate, a dropped shard, a lossy
        channel -- then measure what recall survives. This is how a caller PROVES holography's headline claim on
        their own data instead of taking it on faith: because a record is spread across every slot rather than
        filed in one, recall should degrade SMOOTHLY as slots die, not fall off a cliff.
        `dim` defaults to this mind's dimension. Returns (dim,) of 1.0=keep / 0.0=destroyed, with exactly
        int(dim*destroy_fraction) slots zeroed. DETERMINISTIC in (dim, fraction, seed), so a degradation curve is
        reproducible and can sit in a regression test.
        Delegates to holographic_ai.damage_mask -- the D2 consolidation: this exact body was written three times
        byte-identically on Hologram / HolographicImage / HolographicArchive, which all now delegate here too."""
        from holographic.agents_and_reasoning.holographic_ai import damage_mask as _dm
        return _dm(self.dim if dim is None else dim, destroy_fraction, seed=seed)

    def decode_axial(self, vec):
        """Recover the axial value in [0, pi) from an axial hypervector -- the inverse of
        perceive(theta, 'axial'). Lets a recalled/blended orientation be read back as an angle."""
        return self.encoder.decode_axial(vec)

    # -- one memory: classification + organization -------------------------
    def learn(self, x, label, modality=None):
        """Learn one labelled example NATIVELY -- the base learning verb the whole curriculum is built on. Perceives
        `x` into a hypervector and folds it into memory under `label` two ways: as a self-organized PROTOTYPE (for
        classification) and as an individual kept for exact RECALL. `modality` ('text' / 'image' / 'record' / ...) is
        INFERRED from `x` when left None, so the tag recorded always matches the encoding actually used. For a `record`
        (a {role: filler} dict) it also registers the fillers seen per role -- the cleanup vocabulary that lets the
        mind later read roles back out by unbinding. Returns self (chainable). Specializations: learn_text,
        learn_dictionary, learn_sequence, learn_encyclopedia.
        """
        # SELF-DISCOVERY: if the caller does not name the modality, the encoder
        # infers it from the input itself (encoder.infer is the single source of
        # truth, so the tag recorded here always matches the encoding used).
        # Without this, untagged learning stored modality=None and the routing
        # safeguard in classify() silently vanished for those labels.
        if modality is None:
            modality = self.encoder.infer(x)
        v = self.perceive(x, modality)
        self.memory.observe_vector(v, label)        # aggregate into self-organized prototypes
        self._index(v, (label, x))                  # AND keep the individual for recall
        if modality == "record" and isinstance(x, dict):
            # register the fillers seen per role: this becomes the cleanup
            # vocabulary that lets the mind READ roles back out of its own
            # memory (geometry -> symbol needs candidates, and the honest
            # candidates are the values experience actually contained)
            for k, val in x.items():
                if isinstance(val, (str, int, float, bool)):
                    self._fillers.setdefault(str(k), set()).add(val)
        self._label_modality[label] = modality      # remember which modality this label is
        if modality in self._TEXT_LIKE:
            # keep a bounded sample of each text-like sub-format's raw characters --
            # the corpus the classify-time compression gate is fitted on. Capped so
            # the gate's schema fit stays a few seconds, never grows with the mind.
            cur = self._format_corpus.get(modality, "")
            if len(cur) < self._FORMAT_CORPUS_CAP:
                raw = x if isinstance(x, str) else " ".join(str(t) for t in x)
                self._format_corpus[modality] = (cur + " " + raw)[:self._FORMAT_CORPUS_CAP]
        self._taught += 1
        if self.maintain == 'auto':
            if self.coherence_floor is None:
                if self._taught % self.check_every == 0:
                    self._reorganize_and_narrate()          # original FIXED SCHEDULE (default, unchanged)
            # coherence-GATED: checked EVERY observation over a RESPONSIVE window (so a distribution
            # shift actually moves the signal -- the default window=400 is too smooth to see one), and
            # reorganize only when the store is incoherent, with a cooldown to avoid thrashing. Skips the
            # passes a coherent store does not need. (Window/cooldown scale with check_every; measured to
            # match the best fixed schedule's accuracy at a fraction of its passes.)
            elif self.coherence_floor == 'auto':
                # AUTO floor: no hand-set coherence LEVEL. Track recent coherence and reorganize when it
                # drops below ~90% of its own recent PEAK -- a RELATIVE retention that transfers across data
                # scales (dim, structure) where an absolute 0.65 does not, rather than a constant. (Honestly:
                # this trades an absolute parameter for a relative one, not for nothing -- but the relative
                # one needs no per-dataset retuning.) Cooldown and warm-up as for the fixed floor; the
                # baseline resets after a reorganize because the store has changed.
                coh = self.memory.coherence(window=self.check_every)
                self._coh_hist.append(coh)
                if len(self._coh_hist) > 40:
                    self._coh_hist.pop(0)
                if len(self._coh_hist) >= 8 and (self._taught - self._last_reorg) >= max(20, self.check_every // 2) \
                        and coh < 0.90 * max(self._coh_hist):
                    self._reorganize_and_narrate()
                    self._last_reorg = self._taught
                    self._coh_hist = [coh]                   # store changed: rebuild the baseline
            elif (self._taught - self._last_reorg) >= max(20, self.check_every // 2) and \
                    self.memory.coherence(window=self.check_every) < self.coherence_floor:
                self._reorganize_and_narrate()
                self._last_reorg = self._taught
        return self

    def classify(self, x, modality=None, route=True, abstain=None):
        """Nearest self-organized prototype. If `route` is on, the query competes only
        against its own modality's concepts -- a cheap router that removes the
        cross-modal interference a single flat store can otherwise suffer (a text
        query mistaken for an image). The modality may be declared or, when it is
        not, DISCOVERED from the input -- in two stages:

        * TYPE inference (`encoder.infer`): measured to score identically to
          caller-declared tags on the mixed-modality demo (97.5% both ways).
        * CONTENT inference, only where type goes blind: code and prose are both
          `str`, so when the mind holds text-like sub-formats a string query is
          resolved by the compression gate fitted on the mind's own learned
          samples. This is a CORRECTNESS fix, not a booster -- measured on a
          docs-vs-code set with heavy shared vocabulary, plain type inference
          routed every code query into a pool that EXCLUDED the code labels
          (24% accuracy, 66% cross-pool leakage, worse than no routing at all),
          while the gate identified the sub-format on 100% of held-out queries
          and recovered declared-tag accuracy (61%) exactly. Routing's GAIN over
          a flat scan stayed zero on that data (the bag-of-token vectors already
          separate docs from code) -- the safeguard story again, now one level
          down.

        With `abstain` set (a false-alarm level alpha), the label is returned only if it is
        calibrated-significant -- p <= alpha against the mind's own noise floor (recognize) -- and None
        otherwise (an honest 'I don't recognise this'). Default None preserves the original
        always-name-a-nearest-label behaviour exactly."""
        if abstain is not None:
            label, sim, p = self.recognize(x, modality=modality, route=route)
            # keep classify's (label, score) shape; None label is its existing 'no match' convention
            return (label, sim) if (p == p and p <= abstain) else (None, sim)
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items() if m == modality}
            among = among or None
        return self.memory.classify_vector(self.perceive(x, modality), among=among)

    def _resolve_text_like(self, x):
        """Which text-like sub-format is this string? Type inference can only say
        'text'; if the mind has learned other text-like sub-formats (code), decide by
        the compression gate over schemas fitted on the mind's OWN learned samples --
        whoever compresses the query best understands it."""
        present = {m for m in self._label_modality.values() if m in self._TEXT_LIKE}
        if not present or present == {"text"}:
            return "text"                       # nothing to disambiguate
        if len(present) == 1:
            return next(iter(present))          # only code was learned: a string means code
        gens = self._format_schemas(present)
        if not gens or len(gens) < 2:
            return "text"                       # no corpus to gate with -- fall back safely
        from holographic.simulation_and_physics.holographic_schema import compression_gate
        raw = x if isinstance(x, str) else " ".join(str(t) for t in x)
        return compression_gate(raw, gens)[0][1]

    def _format_schemas(self, modalities):
        """Fit (and cache) one small schema per text-like sub-format from the raw
        samples learn() accumulated. Refit only when a corpus has grown by more than
        a third since its schema was fitted, so steady-state classify pays nothing."""
        from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
        if self._format_gate is None:
            self._format_gate = {}
        for m in modalities:
            corpus = self._format_corpus.get(m, "")
            if len(corpus) < 200:                # too little to characterise a format
                continue
            fitted_at = self._format_fitted_at.get(m, 0)
            if m not in self._format_gate or len(corpus) > 1.34 * fitted_at:
                self._format_gate[m] = SchemaGenerator(m if m == "code" else "text",
                                                       cuts=(0, 60, 150)).fit(corpus)
                self._format_fitted_at[m] = len(corpus)
        return {m: g for m, g in self._format_gate.items() if m in modalities}

    # -- self-assembly: a working mind straight from a pile of examples -----
    def absorb(self, examples, maintain=True, sequences=False):
        """SELF-ASSEMBLY: hand the mind a pile of `(input, label)` or
        `(input, label, modality)` examples and it builds itself -- discovers each
        item's modality, pre-reads whatever text it sees (so word vectors carry
        co-occurrence meaning BEFORE any text is filed; learning text into the
        memory with cold word vectors throws information away), learns everything
        into the one memory, and runs one maintenance pass.

        With `sequences=True` the assembly is COMPLETE: the mind also fits one
        named sequence schema per text-like sub-format it discovered, from the
        same accumulated samples the classify gate uses -- so the one call returns
        a mind that classifies, recalls, AND generates, with unnamed generation
        routed by the compression gate. Off by default only because the schema
        fits cost a few seconds each.

        This is the one good idea of the retired `assemble()` facade, done on the
        real self-organizing machinery instead of a toy reimplementation. It is
        sugar over read()/learn()/maintain_now()/learn_sequence() -- deliberately,
        so there is nothing here to drift out of sync with the long-hand path."""
        examples = [(e if len(e) == 3 else (e[0], e[1], None)) for e in examples]
        examples = [(x, lab, m if m is not None else self.encoder.infer(x))
                    for x, lab, m in examples]
        # first pass: read everything text-LIKE so co-occurrence is learned before
        # filing -- code included, since code encodes through the same word-vector
        # path and its tokens (self, def, bind...) carry co-occurrence meaning too
        text = [x for x, _, m in examples if m in self._TEXT_LIKE]
        if text:
            self.read(text)
        # second pass: file everything into the one memory
        for x, lab, m in examples:
            self.learn(x, lab, m)
        if maintain:
            self.maintain_now()
        # ORDER DISCOVERY as part of self-assembly: if any examples are ordered
        # lists (steps of a plan, not bag-of-words text), the mind tests each
        # label for genuine sequential structure (the permutation test against
        # its own shuffle), proves the winners executable, and registers them --
        # so order becomes a discovered property of the absorbed data, not a
        # separate manual step. Bag-shaped classes are silently left alone.
        list_examples = [(x, lab) for x, lab, m in examples
                         if isinstance(x, (list, tuple)) and len(x) >= 2
                         and not isinstance(x[0], (list, tuple))]
        if list_examples:
            if not hasattr(self, "_seq_members"):
                self._seq_members = {}
            for x, lab in list_examples:
                self._seq_members.setdefault(lab, []).append(list(x))
            self.discover_sequential()
        if sequences:
            # third pass: one sequence schema per discovered text-like sub-format,
            # fitted on the same capped samples learn() accumulated for the gate
            for m, corpus in self._format_corpus.items():
                if len(corpus) >= 200:
                    self.learn_sequence(corpus, modality=("code" if m == "code" else "text"),
                                        name=m)
        return self

    # -- the same data, a recall view (nearest individual) -----------------
    def _index(self, v, payload):
        if self._recall is None:
            self._recall = _Index(self.dim)
        self._recall.add(v, payload)

    # -- relations over the mind's OWN memory --------------------------------
    # The relations operations (explain/name/map/chain) were first measured on a
    # standalone KnowledgeStore; these fold them into the unified mind, running
    # on the records absorb() already stored and the filler vocabulary learn()
    # already registered. The law from the measurements governs every method
    # here: each hop cleans up to a SYMBOL before the next (the symbol-routed
    # path measured 360/360 where the direct algebraic map was ~94% and
    # dimension did not save it).

    def _record_items(self):
        """(vector, label, dict) for every absorbed record in the recall index."""
        if self._recall is None:
            return []
        return [(v, lab, x) for v, (lab, x) in
                zip(self._recall.vecs, self._recall.payloads, strict=True)
                if isinstance(x, dict)]

    def _class_vec(self, label):
        """A learned class's vector: the count-weighted bundle of its
        sub-prototypes in the live memory (one observation -> the record itself;
        many -> their superposition, which is what makes prototype-level
        explanation a real question)."""
        total = None
        for lab, s, _, _ in self.memory.live._p:
            if lab == label:
                total = s if total is None else total + s
        if total is None:
            raise KeyError(f"unknown label: {label!r}")
        n = np.linalg.norm(total)
        return total / n if n else total

    def _clean_filler(self, vec, role):
        """Snap a noisy role-readout to the best filler EXPERIENCE registered
        for that role (falling back to every registered value)."""
        from holographic.agents_and_reasoning.holographic_ai import cosine
        cands = self._fillers.get(str(role)) or {v for s in self._fillers.values()
                                                 for v in s}
        best, score = None, -2.0
        for val in cands:
            s = cosine(vec, self.encoder.encode(val))
            if s > score:
                best, score = val, s
        return best, float(score)

    def find(self, role, filler):
        """Which absorbed record holds bind(role, filler)? One hop over the
        mind's own recall store (the same stored vectors recall() scans),
        restricted to record items. Returns (label, score).

        Over a large record store this resolves COARSE-TO-FINE -- ranking at low
        dimension first and escalating only when the top match is not yet settled
        -- returning the same record as a full scan for far less work (see
        holographic_resolution)."""
        from holographic.agents_and_reasoning.holographic_ai import bind
        probe = bind(self.encoder._roles.get(str(role)), self.encoder.encode(filler))
        items = self._record_items()
        if not items:
            return None, -1.0
        if len(items) >= 32:
            from holographic.misc.holographic_resolution import coarse_to_fine
            M = np.stack([it[0] for it in items])
            idx, score, _, _ = coarse_to_fine(probe, M)
            return items[idx][1], float(score)
        best = max(items, key=lambda it: float(it[0] @ probe))
        return best[1], float(best[0] @ probe)

    def fractal_dimension(self, x, modality=None):
        """The fractal (box-counting) dimension of an input's structure -- a
        perceptual roughness/complexity descriptor the mind can read directly
        from the data. For an image it is the edge map's dimension (natural
        scenes ~1.4-1.6, smooth synthetic shapes ~1.0); for a 1-D series it is
        the self-affinity expressed as a dimension (2 - Hurst). Returns a float."""
        from holographic.misc.holographic_fractal import image_fractal_dimension, hurst_exponent
        m = modality or self.encoder.infer(x)
        arr = np.asarray(x)
        if m == "image" or (arr.ndim >= 2 and arr.dtype != object):
            return float(image_fractal_dimension(arr))
        seq = np.asarray(x, float).ravel()
        return float(2.0 - hurst_exponent(seq))     # self-affinity as a dimension

    def self_affinity(self, series):
        """Hurst exponent of a 1-D series read by the mind: 0.5 random walk,
        <0.5 mean-reverting, >0.5 trending. The fractal lens on a time series."""
        from holographic.misc.holographic_fractal import hurst_exponent
        return float(hurst_exponent(np.asarray(series, float).ravel()))

    def spectral_bandwidth(self, x, energy_fraction=0.95):
        """The BANDWIDTH a signal occupies (holographic_bandwidth) -- the fraction of Nyquist (in [0,1]) holding
        `energy_fraction` of the spectral energy: small for band-limited content, near 1 for broadband noise. The
        number that drives a band-limited encoder's bandwidth knob. Complements the shipped fractal_dimension (which
        says how rough) with how much spectrum to keep. Kept negative: it is an ENERGY rolloff, so a fractal's
        front-loaded 1/f^b energy can read a small bandwidth even though its self-similar detail extends higher --
        honest about fidelity-for-a-budget, not lossless bandwidth."""
        from holographic.misc.holographic_bandwidth import spectral_bandwidth
        return spectral_bandwidth(x, energy_fraction=energy_fraction)

    def mutual_information(self, x, y, bins=16):
        """Mutual information I(X;Y) in BITS between two equal-length signals (discrete or continuous, continuous
        quantile-binned). Zero iff independent; higher = more shared information. This is the RAW estimate --
        biased upward by finite samples, so for a significance-aware number use mutual_information_vs_null. See
        holographic_mutualinfo.mutual_information."""
        from holographic.sampling_and_signal.holographic_mutualinfo import mutual_information
        return mutual_information(x, y, bins=bins)

    def mutual_information_vs_null(self, x, y, bins=16, n_shuffle=64, seed=0):
        """Mutual information ABOVE its SHUFFLE NULL -- the honest dependence measure. Computes raw MI, then a null
        of MI with `y` shuffled (real dependence destroyed, only finite-sample bias left), and reports the excess
        as a z-score. A dependence counts as REAL only when z clears a few sigma; raw MI without its null is a
        Rorschach test. Returns {mi, null_mean, null_std, excess, z}. The gate the pipeline assembler needs. See
        holographic_mutualinfo.mutual_information_vs_null."""
        from holographic.sampling_and_signal.holographic_mutualinfo import mutual_information_vs_null
        return mutual_information_vs_null(x, y, bins=bins, n_shuffle=n_shuffle, seed=seed)

    def permutation_null(self, observed, score_fn, resample_fn, n_null=1000, seed=0, alpha=0.05, side="greater"):
        """The shuffled-null test as ONE composable primitive: score your real datum, then re-run the IDENTICAL
        scoring on `n_null` resamples that destroy the structure (resample_fn(rng)), and report whether the real
        score stands out. This is the discipline radio-SETI (Tarter) and particle physics (Cranmer) live by --
        "score it, then prove it isn't an artifact of your own pipeline" -- lifted out of the engine's five
        procedure-matched private nulls so ANY capability, including one built on the engine, can call it. Returns
        {p, null_mean, null_std, null_ci, observed, collapsed, n_null}: p is the false-alarm probability (with the
        +1 plug so it is never exactly 0), collapsed is True when p<=alpha (the real score stood out). `side` is
        'greater' (a match/recall similarity), 'less', or 'two-sided'. Deterministic given deterministic
        score_fn/resample_fn + seed. KEPT NEGATIVE: a WRONG resample_fn (one that does not destroy the structure
        the score keys on) gives a mis-calibrated null -- the procedure-match is the caller's job. See
        holographic_honesty.permutation_null."""
        from holographic.agents_and_reasoning.holographic_honesty import permutation_null
        return permutation_null(observed, score_fn, resample_fn, n_null=n_null, seed=seed, alpha=alpha, side=side)

    def measure(self, run_once, seeds=range(0, 10), n_boot=2000, boot_seed=0):
        """The VARIANCE HARNESS (holographic_measure) -- every headline number gets a mean, a spread, and a
        confidence interval, not a lucky-seed point estimate. Runs `run_once(seed)` (a callable returning a scalar
        score) across `seeds` and returns {mean, std, ci (95% bootstrap), n, scores}. The constitution's honest-
        measurement discipline made invocable: a claim without this is not a result. See holographic_measure.measure."""
        from holographic.misc.holographic_measure import measure
        return measure(run_once, seeds=seeds, n_boot=n_boot, boot_seed=boot_seed)

    def measure_report(self, name, stats, floor=None):
        """Format a stats dict from `measure` as 'name: mean +/- std (95% CI [lo, hi], n)', flagging FRAGILE when
        the spread is large relative to the margin above `floor`. The honest one-line summary of a measured claim.
        See holographic_measure.report."""
        from holographic.misc.holographic_measure import report
        return report(name, stats, floor=floor)

    def assert_robust(self, stats, floor):
        """Pass only if the LOWER CI bound of a `measure` result clears `floor` -- not just the mean. This is what
        stops a lucky-seed point estimate from passing as a real result. Returns None (raises on failure). The
        no-win-without-a-baseline gate made invocable. See holographic_measure.assert_robust."""
        from holographic.misc.holographic_measure import assert_robust
        return assert_robust(stats, floor)

    def is_fragile(self, stats, margin_floor):
        """Is a measured claim FRAGILE? True if its spread is large relative to how far its mean sits above the
        floor it must clear (std >= half the margin -- a couple of unlucky seeds could sink it). Flags a result
        that looks fine on the mean but rests on luck. See holographic_measure.is_fragile."""
        from holographic.misc.holographic_measure import is_fragile
        return is_fragile(stats, margin_floor)

    def regime_gate(self, name, detect, threshold, superior, fallback, above=True):
        """Build a REGIME GATE (holographic_regimegate) -- route to a superior-but-NICHE method only when a cheap
        detector says you are in its regime, and to a safe fallback everywhere else. Returns a RegimeGate; call
        `.apply(x, *a, **k)` to get (result, info) where info records the score, threshold, and which path ran.
        `detect(x)->score`, `above=True` uses `superior` when score>=threshold. The honest way to re-enable a
        shelved 'only good in a niche' method: the fallback stays the safe default, so a misfire costs at most the
        default. The adaptive-dispatch pattern as a reusable object. See holographic_regimegate.RegimeGate."""
        from holographic.misc.holographic_regimegate import RegimeGate
        return RegimeGate(name, detect, threshold, superior, fallback, above=above)

    def route_or_abstain(self, problem, k=3, n_null=64, z_min=0.8, seed=0):
        """NULL-REFERENCED ROUTING (J1): find_capability that can say "no capability matches" instead of
        returning its argmax on noise. The top-1 score is judged against a null of scrambled queries built
        from the CATALOG'S OWN vocabulary at matched token count (out-of-vocabulary gibberish scores 0 by
        construction and gates nothing -- measured before choosing). The logged misroutes ('counter traders'
        -> dialect emitters; 'purple monkey dishwasher' -> opponent agreement) abstain at z=-0.9/-1.5 while
        real queries route from z=+1.0 up; z_min=0.8 sits in the measured gap. KEPT NEGS: z is calibrated to
        the current vocabulary (not comparable across catalog versions); a genuine query in words the
        catalog never uses abstains CORRECTLY -- the fix is aliases, not a lower line. Returns {abstain, z,
        score, null_mean, null_std, hits, reason}. See Catalog.route_or_abstain."""
        return self._capability_catalog().route_or_abstain(problem, k=k, n_null=n_null, z_min=z_min, seed=seed)


    def wave_state_encoder(self, dim=512, window=32, grid=16, seed=0):
        """ONE window of OHLC bars as ONE state vector (I3): carrier shape (close-based, unit-RMS), BOTH
        envelope excursion channels in scale units (their amplitude IS the close-only-blind information), and
        an energy scalar -- offset/scale normalized so the same shape at 10x the level reads cos ~0.94.
        Built for causal_index recall (fitless resonance: 5/5 right-regime neighbours in the selftest) and as
        a signal_program encode_fn. Two measured design corrections carried in WHY-comments: the 'typical'
        carrier leaks the envelope into the scale (10x swing compressed to 2x -- close-based instead), and
        per-channel RMS erases the envelope's amplitude (carrier-only normalization instead). KEPT NEGS:
        level/scale blindness is the invariance (encode level separately if it matters); and the D4 note --
        an 88%-calibrated forecast on these states had NEGATIVE trading EV in the campaign; calibration is
        not exploitability, run calibration_vs_value before acting. See holographic_candles.WaveStateEncoder."""
        from holographic.misc.holographic_candles import WaveStateEncoder
        return WaveStateEncoder(dim=dim, window=window, grid=grid, seed=seed)


    def decomposition_contract(self, decompose_fn, x, atol=1e-8, residual_key="residual"):
        """Judge ANY series decomposition on the three promises it implicitly makes (I4): COMPLETE (the
        components sum back to x within atol -- else it is a projection wearing a decomposition's name),
        CAUSAL (each component passes lookahead_lint INDIVIDUALLY, so you learn which parts are usable at
        time t and which are diagnosis-only), HONEST RESIDUAL (flags residual_dominates when the 'residual'
        carries the majority: a sliver was removed and the rest renamed). KEPT NEGS: energy shares are NOT
        normalised -- correlated components sum past their expectations and the double-counting stays
        visible; causal means prefix-consistent, not timely; the contract judges the map, not the story in
        the component names. Dogfood on record: the engine's own smooth_sharp_split certifies COMPLETE +
        NON-CAUSAL. See holographic_honesty.decomposition_contract."""
        from holographic.agents_and_reasoning.holographic_honesty import decomposition_contract
        return decomposition_contract(decompose_fn, x, atol=atol, residual_key=residual_key)


    def resting_fill_sim(self, path, events, delta, side=1, horizon=10):
        """RESTING-ORDER adverse selection, measured (G3): rest a limit `delta` off spot at each event, fill
        when the path trades through, mark out at `horizon`. The trap it pins: UNCONDITIONAL mark-out is
        +delta BY CONSTRUCTION (the discount a fill-anything backtest banks) while FILLED mark-out on a pure
        random walk is NEGATIVE -- being CHOSEN claws back more than the whole discount (selection_cost).
        Path-character ordering of the extra adverse beyond the discount: momentum -2.45 << random walk
        -0.53 < mean reversion -0.21 (refuting 'reversion flips it positive'); DEPTH shrinks the per-fill
        extra (overshoot, not deepening toxicity) while fills collapse -- deep resting costs OPPORTUNITY.
        KEPT NEG: price-path only, no queue -- real adverse selection is WORSE; quote these as the
        optimistic bound. See holographic_paperbook.resting_fill_sim."""
        from holographic.agents_and_reasoning.holographic_paperbook import resting_fill_sim
        return resting_fill_sim(path, events, delta, side=side, horizon=horizon)

    def paper_book(self, lag=1, cost=0.0):
        """The walk-forward PAPER ACCOUNT with the gates built in (G4): add_sleeve(name, per-step decisions),
        run(path, gate_mask=None). Entries are ACTIONABLE (lag>=1; lag=0 refused -- simultaneous is not
        past), per-trade cost applied, an optional causal-gate mask stands the book aside, and sleeves are
        tracked separately AND combined with the across-sleeves MEDIAN beside the mean (one lucky sleeve
        drags a mean, not a median). Reports net/t/max_drawdown/equity per sleeve. KEPT NEG, structural: a
        paper book proves PLUMBING, not edge -- one path's realisation; split_half and the selection_ledger
        still apply, and the verdict says so. See holographic_paperbook.PaperBook."""
        from holographic.agents_and_reasoning.holographic_paperbook import PaperBook
        return PaperBook(lag=lag, cost=cost)


    def circular_encoder(self, dim=1024, period=6.283185307179586, seed=0, concentration=0.85):
        """Encode a CIRCULAR variable -- angle, hour-of-day, day-of-week, phase -- with the wrap EXACT (I2):
        encode(x) == encode(x + period) to 1e-12, similarity depends only on the CIRCULAR gap, so 23:59 and
        00:01 read as the 2-minute neighbours they are (the LINE ScalarEncoder reads them at cos 0.21 -- its
        declared limitation, pinned, not fixed there: periodicity needs INTEGER harmonics, a construction,
        not a parameter). Geometric harmonics give the Poisson-kernel-MINUS-DC similarity: near-positive with
        a small antipodal dip (bounded under 0.25, measured); `concentration` trades lobe width for the dip.
        decode() is circular cleanup over [0, period). AUDIT VERDICT carried: the proposed SignedEncoder is
        REFUTED -- signed values are native to ScalarEncoder(lo=-a, hi=a). See
        holographic_encoders.CircularEncoder."""
        from holographic.io_and_interop.holographic_encoders import CircularEncoder
        return CircularEncoder(dim, period=period, seed=seed, concentration=concentration)


    def loss_space_report(self, values, conditions=None, tail_frac=0.05, n_null=400, seed=0):
        """WHERE the losses live (E1): the SHAPE of a loss record on three axes, each vs the null that erases
        only the structure under test -- TAIL (worst 5% share of total loss vs a matched Gaussian; heavier
        means the mean is a comfort blanket), TIME (longest losing streak vs the permutation null; z>2 means
        losses arrive together and independence-based sizing is wrong in the ruinous direction), CONDITION
        (per named mask: loss share vs occupancy, circular-shift null preserving the mask's runs; 10%
        occupancy carrying 60% of the loss is the gate candidate). The loss-side sibling of
        insurance_profile (which asks where the VALUE concentrates). Too few losses returns a scarcity
        report, not a z. See holographic_lossspace.loss_space_report."""
        from holographic.agents_and_reasoning.holographic_lossspace import loss_space_report
        return loss_space_report(values, conditions=conditions, tail_frac=tail_frac,
                                 n_null=n_null, seed=seed)


    def calibration_vs_value(self, probs, outcomes, payoff_act=1.0, loss_act=1.0, cost=0.0,
                             taus=None, n_bins=10):
        """CALIBRATION IS NOT VALUE (D4): score a probabilistic forecast twice -- Murphy-decomposed Brier
        (reliability / resolution / uncertainty) for the statistician, and realized net under act-if-p>=tau
        (swept, with never/always baselines) for the decision-maker -- with the verdicts kept separate.
        Pinned facts: a perfectly calibrated CONSTANT forecast is worthless (resolution is the number that
        failed, and the verdict says so), and the SAME informative forecast monotone-squashed to 38x worse
        reliability keeps 100% of its achievable value -- calibration is a REPAIR (a monotone remap fixes
        it); resolution is the SOURCE and no remap creates it. KEPT NEGATIVE: value_best is an argmax over
        taus -- a SELECTION; choose tau on other data or put the sweep on the SelectionLedger.
        State-dependent payoffs are net_of_costs' ground; the two compose. See
        holographic_forecastvalue.calibration_vs_value."""
        from holographic.agents_and_reasoning.holographic_forecastvalue import calibration_vs_value
        return calibration_vs_value(probs, outcomes, payoff_act=payoff_act, loss_act=loss_act,
                                    cost=cost, taus=taus, n_bins=n_bins)


    def event_study(self, outcome, events, horizon=20, pre=None, n_null=500, seed=0, alpha=0.05):
        """Aligned-window EVENT STUDY (H2): the cumulative mean path around each event, judged against the
        CIRCULAR-SHIFT null -- slide the whole event pattern by a random offset, preserving count and every
        inter-event spacing (so the null inherits the pattern's clustering AND its overlap) and destroying
        only the alignment under test. Reports forward {stat,z,p}, pre_trend {stat,z,p} (a large pre-trend z
        means the event DEFINITION already contains the move -- selection, not prediction), n_overlapping and
        shared_fraction. KEPT NEGATIVE, measured: at spacing 6 vs horizon 20 the naive across-events t
        false-alarms at 28% on pure noise (correlated windows) where this null holds 2%; never rebuild a CI
        from mean_path and n_events. Edge events are dropped and counted, never truncated. Shift null assumes
        a shifted alignment is exchangeable -- difference a trending outcome first. See
        holographic_eventstudy.event_study."""
        from holographic.agents_and_reasoning.holographic_eventstudy import event_study
        return event_study(outcome, events, horizon=horizon, pre=pre, n_null=n_null, seed=seed, alpha=alpha)


    def rolling_stats(self, x, window, stats=("mean", "std", "min", "max"), q=0.9, alpha=0.1, ddof=0):
        """The CAUSAL rolling-statistics kit (H1) in one call: trailing series for any of 'mean', 'std',
        'min', 'max', 'range', 'quantile' (uses q), 'drawdown', 'ewma', 'ewm_std' (use alpha) -- window
        ending AT each position, NaN before warm-up (never a silently-shrunk window), every one
        prefix-consistent under mind.lookahead_lint at 0.0 drift and BIT-identical to the conditioning
        gate's TRAILING_STATS lambdas (so gates and series read the same number). Exact per-window is the
        DEFAULT; the O(n) cumsum path is opt-in via holographic_rolling directly because cancellation on
        offset data destroys it (measured: 1e8 offset -> std off by 8.75 fast vs 2e-9 exact). Streaming
        counterparts with exact warm starts: mind.streaming_stats. See holographic_rolling."""
        from holographic.sampling_and_signal import holographic_rolling as R
        x = list(x) if not hasattr(x, "ravel") else x
        fns = {"mean": lambda: R.rolling_mean(x, window), "std": lambda: R.rolling_std(x, window, ddof=ddof),
               "min": lambda: R.rolling_min(x, window), "max": lambda: R.rolling_max(x, window),
               "range": lambda: R.rolling_range(x, window),
               "quantile": lambda: R.rolling_quantile(x, window, q),
               "drawdown": lambda: R.rolling_drawdown(x, window),
               "ewma": lambda: R.ewma(x, alpha), "ewm_std": lambda: R.ewm_std(x, alpha)}
        unknown = [s for s in stats if s not in fns]
        if unknown:
            raise ValueError("unknown stat(s) %s -- known: %s" % (unknown, sorted(fns)))
        return {s: fns[s]() for s in stats}

    def streaming_stats(self, window=None):
        """Online mean / std / min / max for LIVE data (H1): push(v) one sample at a time; Welford recurrence
        (the numerically-stable answer to cumsum cancellation) + monotonic deques; window=None is expanding,
        an int gives trailing-window values pinned equal to mind.rolling_stats on the same data.
        warm_start(history) replays a backtest tail through the SAME push() path, so live state continues
        bit-for-bit where the backtest ended. See holographic_rolling.StreamingStats."""
        from holographic.sampling_and_signal.holographic_rolling import StreamingStats
        return StreamingStats(window=window)


    def lookahead_lint(self, signal_fn, x, n_checkpoints=8, min_prefix=None, atol=1e-10):
        """LINT a black-box signal pipeline for look-ahead (E3): recompute signal_fn on truncated prefixes and
        demand the shared range be IDENTICAL -- a causal pipeline cannot know whether data exists after t, so
        any drift (full-sample z-score, centred smoother, global min-max, global detrend) is a leak, caught at
        machine precision with a first-bad index. Exact, not statistical. NECESSARY not sufficient: a
        prefix-consistent signal can still leak via its TARGET -- run target_shift_probe too. Precondition:
        signal_fn deterministic (the lint cannot tell nondeterminism from leakage and does not try). See
        holographic_honesty.lookahead_lint."""
        from holographic.agents_and_reasoning.holographic_honesty import lookahead_lint
        return lookahead_lint(signal_fn, x, n_checkpoints=n_checkpoints, min_prefix=min_prefix, atol=atol)

    def target_shift_probe(self, signal, target, max_lag=3):
        """The shift-probe half of the look-ahead lint (E3): is the signal AHEAD of its target, or explaining
        it? Correlates signal_t with target at lags -max_lag..+max_lag; suspicious when the not-ahead side
        (k<=0) more than doubles the ahead side -- the contemporaneous leak's signature (a 'predictor' using
        the bar it predicts reads 0.9 not-ahead vs ~0 ahead). KEPT NEGATIVES pinned: a SYMMETRIC centred-label
        leak is invisible here (equal both sides; that case is lookahead_lint's, run on the label
        constructor), and a trailing stat of an unpredictable target fires as an honest-but-useless false
        positive. Smell test that routes to the lint -- never a verdict. See
        holographic_honesty.target_shift_probe."""
        from holographic.agents_and_reasoning.holographic_honesty import target_shift_probe
        return target_shift_probe(signal, target, max_lag=max_lag)


    def causal_index(self):
        """The APPEND-ONLY, BEFORE-t nearest-neighbour index (D3): append(vector, t) in time order (backfill
        refuses by name), nearest(query, t, k, lag>=1) searches only items with time <= t - lag (lag=0 refused:
        simultaneous is not past), audit_causality VERIFIES the mask by perturbing future items. Structurally
        immune to the leak the demo pins: naive full-history k=1 'history matching' finds the query ITSELF and
        reports perfect skill (100% inflation, zero variance); this index cannot self-match at any k. KEPT
        NEGATIVE: with the self-match hand-excluded, the residual future-neighbour leak on stationary series
        measured DEAD (-0.7%+/-2.5, 10 seeds) -- the value is immunity to the naive call and to nonstationary
        cases where no full-index de-leak recipe exists. Exact scan only (a forest cannot be time-masked
        without re-deriving its guarantees; declared, not a TODO). See holographic_index.CausalIndex."""
        from holographic.caching_and_storage.holographic_index import CausalIndex
        return CausalIndex()


    def tied_candidates(self, ranked, margin=0.1, min_score=None):
        """THE DECISION WITH THE TIE STILL ATTACHED (holographic_relations) -- everything decide_or_abstain
        returns, PLUS the candidates it was nearly and by how little. decide_or_abstain detects a knife-edge
        and then throws the alternatives away, so a caller gets confident=False and one name and cannot see
        what the answer was nearly. This is the missing half of adapt-don't-break: the detection and the
        canonical tie-break already ship, but nothing exposed the SET that needs deciding between.
        A CLEAR WINNER RETURNS A ONE-ELEMENT SET, never an empty one -- "no ambiguity" and "no answer" must
        not look alike. Reports the tie; does NOT resolve it, so determinism is untouched.
        MEASURED, and it is a DEGRADED-REGIME feature: 0% ties on a random codebook at moderate noise, 32% at
        extreme noise, and 84% on a COHERENT codebook under heavy noise. A well-separated store never pays for
        this; an overloaded or near-duplicate one pays constantly."""
        from holographic.misc.holographic_relations import tied_candidates
        return tied_candidates(ranked, margin=margin, min_score=min_score)

    def verify_and_keep(self, candidates, verifier):
        """TRY THE CANDIDATES AND KEEP THE ONE THAT WORKS (holographic_relations) -- rank order, first that
        verifies wins, and all-failed is REPORTED rather than falling back to the top-ranked guess.
        THE RESONATOR'S PATTERN GENERALISED: recursive_factor proposes, re-composes, checks, and reports
        unsolved instead of guessing. That is the honest way to resolve an ambiguity a score could not --
        not by learning a preference, but by TESTING which candidate actually works. Where a downstream
        oracle exists, verification beats learning outright: it is exact, deterministic, needs no training
        data, and returns a proof rather than a probability.
        DETERMINISTIC: candidate order and verifier are both deterministic, so this never makes a run
        irreproducible. ADAPTING AND BEING NON-DETERMINISTIC ARE DIFFERENT THINGS -- returning a verified
        candidate is adaptation; returning a different answer each run is not.
        A raising verifier counts as a failure for that candidate and does not take the search down."""
        from holographic.misc.holographic_relations import verify_and_keep
        return verify_and_keep(candidates, verifier)

    def selection_ledger(self):
        """The SESSION-LEVEL selection ledger (F3): record() every hypothesis test AT THE MOMENT IT IS RUN --
        including the discarded ones -- and correct() computes FDR q-values over the WHOLE book, so the
        correction covers what was actually TRIED, not what survived. Append-only: no remove(); withdraw()
        needs a reason, keeps the entry on the books, and keeps its multiplicity cost. Re-runs of one name are
        recorded as sequences ('ran it until it passed' is countable). to_json/from_json persist with a
        hashlib chain that refuses to load a book with a deleted or edited row. This is the debt SignalProgram
        declares out of scope: batteries correct within themselves; the ledger corrects across them. KEPT
        NEGATIVE: only what is WRITTEN DOWN is covered -- eyeballed-and-discarded looks never reach any
        ledger. See holographic_selectionledger.SelectionLedger."""
        from holographic.agents_and_reasoning.holographic_selectionledger import SelectionLedger
        return SelectionLedger()


    def ledger_record(self, name, p, family="default", effect=None, note=""):
        """Record one hypothesis test on THIS MIND'S session ledger at the moment it is run -- the wire door
        into selection_ledger() (whose object cannot cross HTTP, but whose entries are plain JSON). The ledger
        is created on first use and lives with the mind, so an agent posting /invoke calls accumulates one
        honest book across a whole session. Re-recording a name appends sequence n+1, never overwrites. See
        holographic_selectionledger.SelectionLedger.record."""
        if not hasattr(self, "_session_ledger"):
            from holographic.agents_and_reasoning.holographic_selectionledger import SelectionLedger
            self._session_ledger = SelectionLedger()
        self._session_ledger.record(name, p, family=family, effect=effect, note=note)
        return {"recorded": name, "n_on_book": len(self._session_ledger)}

    def ledger_correct(self, alpha=0.1, family=None):
        """FDR q-values over everything THIS MIND'S session ledger has recorded (family=None = the whole book,
        the honest default for 'what survives this session'). Returns the correction plus n_on_book; refuses
        with a named fix when nothing has been recorded. KEPT NEGATIVE: only what was ledger_record()ed is
        covered -- eyeballed-and-discarded looks never reach any ledger. See
        holographic_selectionledger.SelectionLedger.correct."""
        if not hasattr(self, "_session_ledger") or len(self._session_ledger) == 0:
            raise ValueError("nothing on the session ledger -- ledger_record() each test AT THE MOMENT it is "
                             "run, then correct; a correction over an empty book would certify nothing")
        out = self._session_ledger.correct(alpha=alpha, family=family)
        out = dict(out) if isinstance(out, dict) else {"result": out}
        out["n_on_book"] = len(self._session_ledger)
        return out

    def ledger_book(self):
        """The session ledger serialised (hashlib-chained JSON) -- for persistence, audit, or carrying the
        book to another mind via SelectionLedger.from_json. Empty book returns the honest empty state rather
        than an error: an empty book is a fact, a correction over one is not. See
        holographic_selectionledger.SelectionLedger.to_json."""
        if not hasattr(self, "_session_ledger"):
            return {"n_on_book": 0, "book": None}
        return {"n_on_book": len(self._session_ledger), "book": self._session_ledger.to_json()}


    def signal_program(self, dim=512, seed=0):
        """A BATTERY of detectors screened together, with replication and family-wide multiplicity control
        applied INSIDE the screening pass. add_check(name, encode_fn) to register vectorised detectors, then
        screen(states, targets) returns every check's effect alongside its split-half replication and its
        FDR-corrected verdict -- there is no code path that yields the seductive number alone. Passing checks
        are correlation-clustered, so a battery cannot inflate its apparent breadth (two 0.9-correlated checks
        are ONE finding). An empty pass-list is returned as a populated RESULT with a reason, never as an error
        or a silent fallback to the best raw effect. program_vector(states) fingerprints the whole battery as
        one hypervector for release-over-release comparison. build_committee(report) seats the VETO COMMITTEE
        (E2): one representative per correlation cluster of the passers (an idea cannot vote twice), majority
        vote with tie=abstain, and evaluate() holds the COMBINED signal to the same gates on fresh data --
        members passing individually does not transfer, and an empty committee's decide() refuses with the
        reason rather than falling back to the best member. KEPT NEGATIVE, measured: the batched evaluation is
        ~3x SLOWER than the plain loop (0.22-0.36x across K=12..200) -- the value is that the gates are
        structural, not speed. Second: this handles multiplicity WITHIN one battery only; batteries you ran
        last week and discarded are a session-level debt this does not track. See
        holographic_signalprogram.SignalProgram."""
        from holographic.agents_and_reasoning.holographic_signalprogram import SignalProgram
        return SignalProgram(dim=dim, seed=seed)


    def envelope_forecast(self, series, window=20, alpha=0.1, calib_frac=0.5):
        """D2, forecast the ENVELOPE of the next move (its scale), not its direction: trailing-scale predictor
        + conformal RATIO residuals (one quantile serves every volatility state; an additive margin fails
        per-state, pinned). Returns the band with its own holdout coverage AND a zero-directional-bits note --
        scale forecastability and direction forecastability are different quantities, and the campaign
        measured the second at chance while the first was strong. See holographic_envelope.envelope_forecast."""
        from holographic.sampling_and_signal.holographic_envelope import envelope_forecast
        return envelope_forecast(series, window=window, alpha=alpha, calib_frac=calib_frac)

    def envelope_vs_constant(self, series, window=20, alpha=0.1, calib_frac=0.5):
        """The baseline the envelope must beat: width at EQUAL coverage against a constant unconditional band.
        width_ratio < 1 means the data's vol clustering pays; ~1.0 on iid noise (pinned) -- a sharpness win is
        a claim about the DATA, never a property of the method. Run this before quoting envelope_forecast on a
        new domain. See holographic_envelope.envelope_vs_constant."""
        from holographic.sampling_and_signal.holographic_envelope import envelope_vs_constant
        return envelope_vs_constant(series, window=window, alpha=alpha, calib_frac=calib_frac)


    def reclock(self, series, step, axis=None):
        """Sample when an AXIS moves, not when time passes: emit one event each time the axis traverses `step`,
        carrying source_index / duration / rotation / value. axis=None is the PRICE CLOCK (cumulative |diff| of
        the series itself) -- the only configuration whose sharpening is measured; foreign axes measured to add
        nothing (|z|<1.4). Events completing inside one source sample are counted in `skipped_gap`, never
        fabricated. WARNING: the machinery MANUFACTURES structure -- quote rotation persistence only via
        null_persistence. See holographic_reclock.reclock."""
        from holographic.sampling_and_signal.holographic_reclock import reclock
        return reclock(series, step, axis=axis)

    def rotation_persistence(self, events):
        """Fraction of consecutive reclock events whose rotation agrees -- the NAIVE momentum readout, one
        definition instead of five hand-rolled ones. Meaningless alone: this module's clock manufactures ~0.25
        agreement on pure noise (a fake reversion effect) where renko manufactured ~0.72 (fake momentum). Quote
        it only beside null_persistence's z. See holographic_reclock.rotation_persistence."""
        from holographic.sampling_and_signal.holographic_reclock import rotation_persistence
        return rotation_persistence(events)

    def null_persistence(self, series, step, surrogate="iid_shuffle", n=200, seed=0, **surrogate_kwargs):
        """The HONEST reclock-persistence measurement: the full reclock -> persistence chain on the series AND
        on n surrogates, via pipeline_null (two-sided). Expect null_mean far from 0.5 -- that gap IS the
        manufactured structure, on display. KEPT NEGATIVE: price clock only (a surrogate reorders the series;
        an external axis has no defined reordering). See holographic_reclock.null_persistence."""
        from holographic.sampling_and_signal.holographic_reclock import null_persistence
        return null_persistence(series, step, surrogate=surrogate, n=n, seed=seed, **surrogate_kwargs)

    def duration_stats(self, events):
        """The duration channel of a reclocked series in one report: log-space lag-1 autocorrelation (activity
        clustering) and up/down duration asymmetry (do rises take longer than falls?) with a Welch z. Log space
        because durations are ratio-scaled and heavy-tailed. Run duration_resolution_check FIRST -- a quantised
        duration grid makes every number here an artifact. See holographic_reclock.duration_stats."""
        from holographic.sampling_and_signal.holographic_reclock import duration_stats
        return duration_stats(events)

    def duration_resolution_check(self, events, min_distinct=5, max_unit_frac=0.5):
        """Is the duration channel RESOLVED or a grid artifact? Warns on too few distinct durations, too many
        single-sample events, or events completed INSIDE one sample (the measured -inf log-duration incident:
        25bp bricks inside single 5-minute bars). Returns {ok, warnings} with each warning naming its fix. See
        holographic_reclock.duration_resolution_check."""
        from holographic.sampling_and_signal.holographic_reclock import duration_resolution_check
        return duration_resolution_check(events, min_distinct=min_distinct, max_unit_frac=max_unit_frac)


    def sign_flip(self, x, seed=0):
        """A SIGN-FLIP surrogate: randomise the direction of every sample while keeping its magnitude, so
        magnitude (volatility) clustering is preserved EXACTLY and only the direction channel is destroyed. The
        right null for a DIRECTIONAL claim, where a plain shuffle would over-credit you for magnitude structure
        that was never the claim. KEPT NEGATIVE: useless for a magnitude-only statistic (variance, energy) --
        the null has zero spread. See holographic_surrogate.sign_flip."""
        from holographic.sampling_and_signal.holographic_surrogate import sign_flip
        return sign_flip(x, seed=seed)

    def iid_shuffle(self, x, seed=0):
        """A plain random permutation: exact value histogram, ALL ordering destroyed. The bluntest null -- use it
        when the claim is 'there is any temporal structure here at all'. KEPT NEGATIVE: too strong for most
        continuous signals, because it also destroys the autocorrelation a trivial forecaster exploits; prefer
        phase_randomize or block_shuffle unless total disorder is really the baseline you mean. See
        holographic_surrogate.iid_shuffle."""
        from holographic.sampling_and_signal.holographic_surrogate import iid_shuffle
        return iid_shuffle(x, seed=seed)

    def block_shuffle(self, x, block, seed=0):
        """A moving-block-bootstrap surrogate: cut `x` into contiguous blocks of length `block` and shuffle the
        BLOCK ORDER, so structure shorter than `block` survives and structure longer than it is destroyed. The
        block length is the dial that says which SCALE the claim is about. KEPT NEGATIVE: the joins between
        reordered blocks are discontinuities the real signal never had, so a jump/gap detector sees ~len(x)/block
        fake events per surrogate. block=1 is bit-identical to iid_shuffle. See
        holographic_surrogate.block_shuffle."""
        from holographic.sampling_and_signal.holographic_surrogate import block_shuffle
        return block_shuffle(x, block, seed=seed)

    def surrogate_ensemble(self, x, kind="phase", n=200, seed=0, materialize=False, **surrogate_kwargs):
        """Yield `n` surrogates of `x` one at a time as a GENERATOR -- the memory-light form for long series.
        `kind` names any surrogate ("phase", "aaft", "iaaft", "sign_flip", "iid_shuffle", "block_shuffle" with
        block=...) or is a callable fn(x, seed). Each member gets sub-seed seed+i+1, so the ensemble is
        reproducible and shares its numbering with pipeline_null. Set materialize=True to get an (n, len(x))
        ARRAY instead -- required over the HTTP service, where a generator degrades to a repr stub and dead-ends
        the caller; the in-process default stays the memory-light generator. See
        holographic_surrogate.surrogate_ensemble / surrogate_batch."""
        from holographic.sampling_and_signal.holographic_surrogate import surrogate_ensemble, surrogate_batch
        if materialize:
            return surrogate_batch(x, kind=kind, n=n, seed=seed, **surrogate_kwargs)
        return surrogate_ensemble(x, kind=kind, n=n, seed=seed, **surrogate_kwargs)

    def trev(self, x, lag=1):
        """The TIME-REVERSAL ASYMMETRY statistic: the normalised third moment of the lagged difference, which is
        exactly zero for any process invariant under time reversal and non-zero when rises and falls have
        different SHAPES. Scale-free. A non-zero value alone means nothing -- pair it with time_arrow_test. See
        holographic_surrogate.trev."""
        from holographic.sampling_and_signal.holographic_surrogate import trev
        return trev(x, lag=lag)

    def time_arrow_test(self, x, lag=1, n_surrogates=200, seed=0, kind="iaaft"):
        """Does this series have an ARROW OF TIME? Measures `trev` against a surrogate ensemble and returns
        {value, null_mean, null_std, z, p, n_surrogates, kind}. A large |z| says the process is NONLINEAR (linear
        Gaussian processes are time-reversible) -- a triage flag, not a detection. Defaults to the IAAFT null
        because a merely SKEWED series scores a large z against a phase-randomised one. KEPT NEGATIVE, measured:
        a significant global arrow can be entirely DIFFUSE -- z=+6.4 daily with all three localisation attempts
        null. It is a property of the process, never a per-window signal. See
        holographic_surrogate.time_arrow_test."""
        from holographic.sampling_and_signal.holographic_surrogate import time_arrow_test
        return time_arrow_test(x, lag=lag, n_surrogates=n_surrogates, seed=seed, kind=kind)

    def conditional_coverage(self, residuals_calib, residuals_test, condition_test,
                             alphas=(0.05, 0.1, 0.2), min_side=25):
        """D1, coverage UNDER A CONDITION: the conformal guarantee checked inside/outside a boolean split of
        the test residuals (regime, storm gate, load level) -- marginal coverage is an average, and it can hold
        while both sides are wrong in opposite directions (canon: nominal 90% that was ~97% calm / ~70% storm).
        `degraded` flags a side missing nominal by >2 binomial SEs; sides thinner than min_side report
        reliable=False. KEPT NEGATIVE: the split-conformal guarantee IS marginal -- this diagnoses the gap;
        closing it needs per-condition calibration. See holographic_conformal.conditional_coverage."""
        from holographic.mesh_and_geometry.holographic_conformal import conditional_coverage
        return conditional_coverage(residuals_calib, residuals_test, condition_test,
                                    alphas=alphas, min_side=min_side)


    def net_of_costs(self, event_values, cost=None, per_side=None):
        """G1, THE COST WALL: gross per-event value vs the round-trip cost of acting, one readout -- net mean/t,
        wall_ratio, survives, and breakeven_cost (the portable fact: 'survives at 5, dies at 9'). Canon: four
        real gross edges (+1.9..+10.6 bp), all dead at a 17 bp wall. Pass per-event cost ARRAYS for
        state-dependent costs -- a constant cost is a model, and the error's sign follows the cost-value
        covariance (measured both ways). See holographic_actioncost.net_of_costs."""
        from holographic.agents_and_reasoning.holographic_actioncost import net_of_costs
        return net_of_costs(event_values, cost, per_side=per_side)

    def realizable_fills(self, event_index, path, horizon, lag=1, cost=0.0, side=None, emission_price=None):
        """G2, EMISSION vs ACTIONABLE: forward value measured twice -- entry at the first reachable state after
        the event is KNOWN (lag>=1; lag=0 refused by name) and at the idealized emission price -- with
        latency_cost = the difference, the move that completed during recognition. Canon: z=+20 continuation at
        emission, NEGATIVE at the actionable price. Sweep the lag before believing an edge: lag=1 is the floor,
        not the truth. See holographic_actioncost.realizable_fills."""
        from holographic.agents_and_reasoning.holographic_actioncost import realizable_fills
        return realizable_fills(event_index, path, horizon, lag=lag, cost=cost, side=side,
                                emission_price=emission_price)


    def dpi_guard(self, features, new_feature, seed=0, holdout_frac=0.5, degree=2):
        """IS THIS FEATURE ACTUALLY NEW INFORMATION -- or a transform of what you already have? Fits the
        proposed feature from a linear/quadratic expansion of the existing set on a train split and reports
        R^2 on train AND HOLDOUT, never train alone; novel_frac = the reproducibly-unexplained share, the most
        the feature could add. DPI: a transform can CONCENTRATE information, never create it. KEPT NEGATIVES:
        low holdout R^2 means not-a-transform-under-this-basis, which may be noise (novelty is necessary, not
        sufficient -- it still owes a target-side test in bits); an exotic transform outside the basis can
        slip. See holographic_honesty.dpi_guard."""
        from holographic.agents_and_reasoning.holographic_honesty import dpi_guard
        return dpi_guard(features, new_feature, seed=seed, holdout_frac=holdout_frac, degree=degree)

    def holdout_auc(self, scores_train, labels_train, scores_test, labels_test):
        """AUC (exact Mann-Whitney, ties shared) on train AND holdout as one inseparable pair -- the standard
        separability readout, shaped so the overfit signature has nowhere to hide. Measured canon: a kernel
        lift at train 0.685 / held-out 0.557 -- separability that was mostly the representation's own
        capacity. See holographic_honesty.holdout_auc."""
        from holographic.agents_and_reasoning.holographic_honesty import holdout_auc
        return holdout_auc(scores_train, labels_train, scores_test, labels_test)


    def split_half(self, events, values=None, mode="contiguous", alpha=0.05):
        """SPLIT-HALF REPLICATION -- cut the measurements in two, measure the effect in each half, and PASS only
        when both halves agree in SIGN and each is individually significant. Call as split_half(values) or
        split_half(events, values). mode="contiguous" (default) asks the TEMPORAL question and is the mode that
        does the killing; mode="interleave" shares the regime between halves, so passing interleaved while
        failing contiguous identifies a regime-bound effect. Returns per-half means/t/p plus `passed`. Measured:
        this one gate killed four artifacts that every other readout called real, with no false rejections.
        KEPT NEGATIVE: p is the normal approximation (NumPy-only), anticonservative for halves under 30 --
        `small_sample` flags it; and replication is not multiplicity control, so run bh_fdr as well. See
        holographic_honesty.split_half."""
        from holographic.agents_and_reasoning.holographic_honesty import split_half
        return split_half(events, values=values, mode=mode, alpha=alpha)

    def causal_gate(self, stat="std", window=20, threshold=None, compare="ge", context=None,
                    min_periods=None, name=None):
        """Build a CAUSAL condition -- a gate that may look only at trailing data, and can therefore be ACTED on
        rather than merely described. stat is a name ('std','mean','abs_mean','min','max','range','drawdown',
        'last') or a callable window->float; it is handed context[i-window+1:i+1] only, so the gate is causal by
        construction. With context=None you get a Gate object (composable with & | ~, for use in conditional /
        insurance_profile); pass a context array to get {'mask', 'audit', 'n_true'} instead -- the audit PROVES
        causality by scrambling the future and checking the past does not move. The storm gate that took a
        measured book from +22% to +58.4% CAGR (max drawdown -85.9% -> -47.1%) with the SAME entries is two of
        these or-ed: trailing drawdown <= -15% OR trailing vol in the top decile. KEPT NEGATIVE: a hand-written
        mask_fn's causal=True is a claim, not a proof -- audit it; and a bare boolean array passed to the
        measurement functions is treated as EX-POST on purpose. See holographic_conditioning.trailing_gate."""
        import holographic.agents_and_reasoning.holographic_conditioning as _cond
        g = _cond.trailing_gate(stat=stat, window=window, threshold=threshold, compare=compare,
                                min_periods=min_periods, name=name)
        if context is None:
            return g
        mask = g.mask(context)
        return {"mask": mask.tolist(), "n_true": int(mask.sum()), "audit": g.audit_causality(context)}

    def conditional(self, values, condition, stat_fn=None, context=None, alpha=0.05):
        """Split any measurement by a condition and report it FOUR ways at once -- all, inside, outside, and the
        difference (Welch z + p). condition is a Gate (causal, actionable), an ExPostMask, or a raw boolean array
        (treated as EX-POST on purpose -- trusting the caller is how look-ahead gets into a result). The reframe
        it makes cheap: an unconditional average hid two opposite behaviours in the campaign that paid for this
        module -- trending in calm conditions, whipsawing in storms, and a flat nothing on average. Condition a
        weak effect before abandoning it and a strong one before believing it. Returns n/mean/t/p per group plus
        diff, z_diff, separates, detection floors, causal, and a loud `warning` when the split is ex-post. See
        holographic_conditioning.conditional."""
        import holographic.agents_and_reasoning.holographic_conditioning as _cond
        return _cond.conditional(values, condition, stat_fn=stat_fn, context=context, alpha=alpha)

    def across_regimes(self, values, segments=None, series=None, events=None, min_seg=16, penalty=3.0,
                       alpha=0.05, power=0.8):
        """Evaluate an effect inside EVERY measured regime, and say whether it is one effect or one regime's
        story. Pass `segments` or pass `series` to have the regimes measured for you by the engine's own
        change-point segmenter (the same one behind detect_regimes). Per segment: n/mean/t/p, significance, and a
        DETECTION FLOOR so an empty segment reports 'nothing above X' rather than 'nothing'. Across segments:
        consistent (sign agreement), sign_test_p, and `concentration` -- the share of the effect carried by the
        single biggest regime. Measured: a real effect was positive in 3 of 4 regimes at concentration 0.41; an
        artifact with a comparable headline mean had one regime carrying >0.9 of it. KEPT NEGATIVE: the sign test
        is underpowered by construction (four regimes cannot beat p=0.125) -- read `consistent` and
        `concentration` first; and measured segments know the whole series, so this describes where an effect
        lived, it does not define a tradeable rule. See holographic_conditioning.across_regimes."""
        import holographic.agents_and_reasoning.holographic_conditioning as _cond
        return _cond.across_regimes(values, segments=segments, series=series, events=events,
                                    min_seg=min_seg, penalty=penalty, alpha=alpha, power=power)

    def insurance_profile(self, values, condition, context=None, alpha=0.05):
        """ASK BEFORE YOU FILTER: is the payoff concentrated in the state you were about to exclude? Filtering
        the ugly periods is the most natural move in analysis, and sometimes it deletes the entire phenomenon.
        Measured: a reversion effect paid +36bp per event inside storms and +4bp outside -- it was not damaged by
        storms, it WAS storm insurance, and excluding them removed ~90% of the edge while every other statistic
        on the page improved. Returns the inside/outside decomposition plus share_inside, frac_events, lift,
        `premium_inside`, and a verdict sentence. KEPT NEGATIVE: a premium in a rare state is also the signature
        of too little data in that state -- read premium_inside as 'measure it properly before deleting it', not
        as 'keep it', and follow with split_half on the inside events. See
        holographic_conditioning.insurance_profile."""
        import holographic.agents_and_reasoning.holographic_conditioning as _cond
        return _cond.insurance_profile(values, condition, context=context, alpha=alpha)

    def pipeline_null(self, pipeline_fn, x, surrogate="phase", n=200, stat_fn=None, seed=0, alpha=0.05,
                      side="two-sided", **surrogate_kwargs):
        """Run YOUR WHOLE PIPELINE on surrogates and score the statistic against the null the pipeline itself
        produces. Processing MANUFACTURES structure: any smoothing, quantising, re-clocking or clustering step
        imposes correlations on whatever it is fed, including pure noise, so a null computed on the raw input or
        against a textbook baseline credits the pipeline's own artifacts to the data. Measured: a re-clocking
        step produced 72% direction persistence on pure noise (the referenced truth was ANTI-persistence at
        z=-7.3); a denoiser manufactured 83.6%. Returns {observed, null_mean, null_std, z, p, null_ci, collapsed,
        n, surrogate}. KEPT NEGATIVE: it cannot rescue a badly-chosen surrogate -- picking one that destroys
        something the pipeline needs for unrelated reasons yields a healthy-looking, meaningless z. See
        holographic_honesty.pipeline_null."""
        from holographic.agents_and_reasoning.holographic_honesty import pipeline_null
        return pipeline_null(pipeline_fn, x, surrogate=surrogate, n=n, stat_fn=stat_fn, seed=seed,
                             alpha=alpha, side=side, **surrogate_kwargs)

    def min_detectable_effect(self, test_fn, x, effect_grid, inject_fn=None, surrogate="phase", n_trials=60,
                              seed=0, alpha=0.05, power=0.8, **surrogate_kwargs):
        """DETECTION FLOOR -- turn "we found nothing" into "there is nothing here above X", the only form of a
        null result that can be argued with. Injects synthetic effects of known size into surrogates of your own
        `x` (so the noise level is the one you actually face) and reports the smallest size `test_fn` catches at
        the target power, plus the whole power curve. `floor=None` means the grid needs extending upward, not
        that the floor is zero. KEPT NEGATIVE: a floor is conditional on the injection SHAPE -- a floor for an
        additive shift says nothing about a burst or a variance change of the same nominal size, so quote the
        floor with its injection; and the surrogate must DESTROY the statistic under test or the power curve
        degenerates to a 0/1 step. See holographic_honesty.min_detectable_effect."""
        from holographic.agents_and_reasoning.holographic_honesty import min_detectable_effect
        return min_detectable_effect(test_fn, x, effect_grid, inject_fn=inject_fn, surrogate=surrogate,
                                     n_trials=n_trials, seed=seed, alpha=alpha, power=power, **surrogate_kwargs)


    def phase_randomize(self, x, seed=0):
        """A PHASE-RANDOMIZED surrogate of a 1-D signal: same power spectrum (same autocorrelation) as `x` but
        random phases, so deterministic/nonlinear structure is destroyed while the linear second-order statistics
        are preserved EXACTLY (Theiler et al. 1992). The honest null for a CONTINUOUS, autocorrelated signal --
        unlike a permutation, it does not destroy the autocorrelation a trivial forecaster exploits. See
        holographic_surrogate.phase_randomize."""
        from holographic.sampling_and_signal.holographic_surrogate import phase_randomize
        return phase_randomize(x, seed=seed)

    def surrogate_zscore(self, x, statistic, n_surrogates=64, seed=0):
        """Measure a structure `statistic(x)` against an ensemble of PHASE-RANDOMIZED surrogates and report how
        far the real value exceeds the null, in null std devs (a z-score). Because the surrogates share the real
        signal's autocorrelation, a high z means structure BEYOND linear autocorrelation -- the honest continuous
        analogue of the discrete shuffle-null. The gate for a continuous forecast/structure test. See
        holographic_surrogate.surrogate_zscore."""
        from holographic.sampling_and_signal.holographic_surrogate import surrogate_zscore
        return surrogate_zscore(x, statistic, n_surrogates=n_surrogates, seed=seed)

    def amplitude_adjusted_surrogate(self, x, seed=0):
        """AAFT surrogate -- the STRICTER null for NON-GAUSSIAN signals (holographic_surrogate). Basic
        phase_randomize preserves the spectrum but GAUSSIANIZES the marginal (destroying fat tails, e.g. of price
        returns); AAFT preserves BOTH the exact amplitude DISTRIBUTION and (approximately) the power spectrum, by
        phase-randomizing a Gaussian-ranked copy and mapping the original's sorted amplitudes back on. Use this
        when the amplitude distribution matters (fat tails); use phase_randomize when the signal is ~Gaussian and
        the spectrum must match exactly. See holographic_surrogate.amplitude_adjusted_surrogate."""
        from holographic.sampling_and_signal.holographic_surrogate import amplitude_adjusted_surrogate
        return amplitude_adjusted_surrogate(x, seed=seed)

    def iaaft_surrogate(self, x, n_iter=100, tol=1e-8, seed=0):
        """IAAFT surrogate -- the gold-standard null matching BOTH the exact amplitude distribution AND (to
        convergence) the exact power spectrum (Schreiber & Schmitz 1996). AAFT only approximates the spectrum;
        IAAFT ITERATES two projections (impose target magnitudes / impose the amplitude distribution) until they
        agree -- the same iterate-a-projection move as IK/PBD/the resonator. Prefer over AAFT for strongly-coloured
        non-Gaussian signals (e.g. fat-tailed price returns with real autocorrelation), at the cost of iterations.
        See holographic_surrogate.iaaft_surrogate."""
        from holographic.sampling_and_signal.holographic_surrogate import iaaft_surrogate
        return iaaft_surrogate(x, n_iter=n_iter, tol=tol, seed=seed)

    def candle_carrier(self, candles, kind="typical"):
        """Turn OHLC candles into the one-value-per-bar WAVE they sample (holographic_candles): a price candle is a
        SAMPLE of a continuous wave, and this returns the carrier signal -- 'typical' price (H+L+C)/3, 'close',
        'median' (H+L)/2, or 'ohlc4'. Feed the result to any signal op (spectrum, band-limit, phase_randomize,
        fit_deterministic, ladder_predict). Accepts (N,4) OHLC, (N,5/6) [ts,OHLC(V)], or dicts. See
        holographic_candles.carrier."""
        from holographic.misc.holographic_candles import carrier
        return carrier(candles, kind=kind)

    def candle_envelope(self, candles):
        """The high/low BAND around a candle series' carrier (holographic_candles) -- how far the price wave swung
        INTRA-bar, which a close-only line throws away. Returns (upper, lower) = (highs, lows). The band width is
        the per-bar range, the amplitude of the wave's within-sample excursion. See holographic_candles.envelope."""
        from holographic.misc.holographic_candles import envelope
        return envelope(candles)

    def candle_intrabar_path(self, candles, steps_per_bar=4):
        """Reconstruct a HIGHER-RESOLUTION wave from OHLC candles (holographic_candles): each bar becomes O ->
        {High,Low in the inferred order} -> C, so within-bar excursions enter the signal. The H/L order is inferred
        from bar direction (up bar dips to the low then runs to the high; down bar the reverse). Honest about being
        a PLAUSIBLE path -- OHLC doesn't record the true order. Hits every open/close exactly. See
        holographic_candles.intrabar_path."""
        from holographic.misc.holographic_candles import intrabar_path
        return intrabar_path(candles, steps_per_bar=steps_per_bar)

    def candle_range(self, candles):
        """Per-bar RANGE (High-Low, the within-bar swing amplitude) and BODY (|Close-Open|) of a candle series
        (holographic_candles). A small body inside a big range is a bar that went nowhere despite swinging --
        the range/body ratio is a classic noise-vs-trend measure. Returns (ranges, bodies). See
        holographic_candles.candle_range."""
        from holographic.misc.holographic_candles import candle_range
        return candle_range(candles)

    def guide_structure(self, state, constraints, iters=50, tol=1e-6, omega=1.0):
        """Guide a `state` toward a goal by ITERATING A PROJECTION -- the level-generic form of IK / PBD / denoise
        / resonator (all 'iterate a projection', Macklin). `constraints` is a list of projection callables (pin a
        root to a target, clamp a link length, snap to a codebook, ...) applied in turn until the state settles.
        Returns {state, iters_used, converged, residual}. The constraints ARE the structure of the space;
        iterating them is legal movement through it. Builders: guide_pin / guide_clamp_link / guide_snap. See
        holographic_guide.guide_structure."""
        from holographic.misc.holographic_guide import guide_structure
        return guide_structure(state, constraints, iters=iters, tol=tol, omega=omega)

    def guide_pin(self, index, value):
        """Constraint builder for guide_structure: pin state[index] to `value` (an IK end-effector target / a
        boundary condition). See holographic_guide.pin."""
        from holographic.misc.holographic_guide import pin
        return pin(index, value)

    def guide_clamp_link(self, i, j, length):
        """Constraint builder for guide_structure: clamp the distance between state[i] and state[j] to at most
        `length` (a bone/edge length limit -- the PBD distance constraint). See holographic_guide.clamp_link."""
        from holographic.misc.holographic_guide import clamp_link
        return clamp_link(i, j, length)

    def guide_snap(self, book):
        """Constraint builder for guide_structure: snap every state entry to its nearest value in `book` (the
        resonator's codebook cleanup as a projection). See holographic_guide.snap_to_codebook."""
        from holographic.misc.holographic_guide import snap_to_codebook
        return snap_to_codebook(book)

    def assemble_pipeline(self, x, y, candidates, min_z=3.0, holdout=0.3, bins=16, n_shuffle=48, seed=0):
        """Find which candidate transform(s) connect input `x` to output `y`, VALIDATED against a shuffle null
        (the honest pipeline assembler). `candidates` is a dict name -> callable (x -> y_hat). Each is scored on a
        HELD-OUT segment (search overfits its own tests) AND gated by MI-over-shuffle-null (does the REAL input
        drive the output more than a shuffled one?). Returns survivors sorted by z; a candidate passes only if
        z >= min_z, else it is chance alignment, not a discovery. An empty list is an honest 'nothing connects
        these'. The gate that stops 'any random projection works'. See holographic_assemble.assemble_pipeline."""
        from holographic.agents_and_reasoning.holographic_assemble import assemble_pipeline
        return assemble_pipeline(x, y, candidates, min_z=min_z, holdout=holdout, bins=bins,
                                 n_shuffle=n_shuffle, seed=seed)


    def fetch_asset(self, url, cache_dir=None, sha256=None, timeout=30.0):
        """Fetch an external asset (HDRI/model/texture) into the content-addressed cache -> {path, sha256,
        bytes, cached}. THE NETWORK MEETS THE DETERMINISM RULE the same way randomness does: BY PINNING.
        An unpinned fetch returns the hash to record; a PINNED fetch that is cached is served from disk
        with NO network I/O -- so a scene recipe of (url, sha256) pairs replays bit-identically offline,
        forever, which downloaded-on-demand can never do. A pinned fetch whose bytes mismatch is deleted
        and raises naming BOTH hashes (a silently-different asset is the supply-chain version of a flipped
        decision). Opt-in only: nothing in core imports this; http(s) only; 512 MB ceiling. Feed the result
        straight to load_hdr / import_asset / asset_library.add_hashes. See holographic_assetfetch."""
        from holographic.io_and_interop.holographic_assetfetch import fetch_asset
        return fetch_asset(url, cache_dir=cache_dir, sha256=sha256, timeout=timeout)

    def load_hdr(self, path, exposure=1.0):
        """Read a Radiance .hdr / .pic (RGBE) environment map -> (H,W,3) float32 LINEAR radiance, UNBOUNDED.

        THE LAST MISSING PIECE OF IMAGE-BASED LIGHTING. Everything else was already here: DomeLight's `color`
        accepts a callable f(dirs)->rgb, and sky_dome() samples an equirectangular env by lon/lat. What there
        was no way to do was GET a real environment map in -- load_image reads 8 bits, and an 8-bit env is
        precisely the wrong input, because the whole point of an HDRI is that the sun is thousands of times
        brighter than the sky. A tone-mapped picture of a sky is not an environment light.

        MEASURED, and it is why this and not a sky-field wrapper (160x120, 2 bounces, dome only, matched
        mean radiance): a flat-colour dome and a procedural sky FIELD differ by 0.0054 mean abs -- invisible.
        The same env mirrored left/right differs by 0.0336, six times larger. Smooth gradients do not pay;
        DIRECTIONAL STRUCTURE does, and only a real HDRI has it.

        Use it: `env = m.load_hdr(path)` then `m.scene_light('dome', color=lambda d: m.sky_dome(d, env=env))`.
        Pair with a DARK sky= or the environment is counted twice for diffuse.

        KEPT NEGATIVES: .exr is not supported (a whole container format -- multi-part, tiled, several
        compressors -- and its own project); XYZE files RAISE rather than decode wrongly, because their
        primaries are CIE XYZ and returning them as RGB would silently shift every colour; and the result is
        UNBOUNDED on purpose -- clipping the sun to 1.0 is the exact information loss this exists to avoid.
        See holographic_render.load_hdr."""
        from holographic.rendering.holographic_render import load_hdr
        return load_hdr(path, exposure=exposure)

    def load_image(self, path, mode="rgb01"):
        """Read a PNG back into an array -- the inverse of save_render, and the step that closes a see-then-fix loop.

        The engine could WRITE a PNG and could not READ one: a grep for IHDR found only the encoder. That one
        missing direction blocked every render -> look -> adjust -> render cycle, because "look" had nowhere to
        start, and it is why compare_image_files reached for Pillow. Pure stdlib (zlib + struct), no dependency.

        mode='rgb01' (default) gives (H,W,3) float in [0,1] -- the shape every render and image call here takes,
        so it feeds straight back into compare_images, a denoiser, or another render. mode='raw' gives the array
        as stored (uint8/uint16, 1-4 channels). Greyscale, palette, and both alpha forms decode; alpha is dropped
        in rgb01 because the caller asked for RGB.

        KEPT NEGATIVE: the round trip is to about 1/255, not exact -- save_png quantises to 8 bits, so assert
        against a tolerance and never against equality. Interlaced (Adam7) PNGs RAISE rather than decode
        wrongly. See holographic_render.load_png / png_decode."""
        from holographic.rendering.holographic_render import load_png
        return load_png(path, mode=mode)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p01_read", "_UnifiedPart01")
    print("holographic_unified_p01_read selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

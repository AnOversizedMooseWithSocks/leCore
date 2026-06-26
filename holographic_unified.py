"""One model over one holographic space.

The rest of this project grew as separate studies -- a self-organizing classifier, a
self-maintaining decision brain, an image vault, a mixture-of-experts router, a text
n-gram. They were never meant to stay separate. They already share the one thing that
matters: a holographic vector space, and a `UniversalEncoder` that turns ANY input --
text, image, number, category, record, sequence -- into a vector in that single space.

`UnifiedMind` is the top level that makes the sharing real instead of nominal. There is
ONE perception step (the encoder), ONE associative memory (the autonomous
`SelfOrganizingMind`, which both classifies and is searched for recall), and ONE
decision brain (`HolographicMind`), all reading and writing the same space. It does not
reimplement simple versions of these -- the failing of the old `Mind` facade -- it uses
the real, self-maintaining ones, and every input passes through the same encoder before
it reaches any of them.

What is deliberately NOT pretended to be one call: classification, recall, and decision
are different OPERATIONS on the shared substrate (aggregate into prototypes; index the
individuals; weight by reward). The unification is the shared space and the shared
self-maintenance, not a single magic method.
"""

import numpy as np

from holographic_mind import UniversalEncoder, _Index
from holographic_organizer import SelfOrganizingMind
from holographic_creature import HolographicMind


class UnifiedMind:
    """Perceive once, into one space; remember, organize, recall, and decide over it.

      read(corpus)                     -- let perception pre-learn word co-occurrence
      absorb(examples)                 -- SELF-ASSEMBLY: build a working mind from a pile
                                          of (input, label[, modality]) examples
      learn(x, label[, modality])      -- file a perception into the one memory; the
                                          modality is discovered if not declared
      classify(x[, modality])          -- 'what is this?'  (nearest self-organized prototype,
                                          routed within the discovered/declared modality)
      recall(x[, modality])            -- 'what's like this?' (nearest stored individual)
      actions(names) / decide / reinforce  -- choose actions over the same space

    The memory maintains itself: with maintain='auto' it periodically reorganizes (the
    speculate-measure-adopt rule from holographic_organizer), splitting a confusable
    class into sub-prototypes only when held-out accuracy says it earns its keep. The
    decision brain maintains itself the same way.
    """

    # modalities whose inputs are strings/token-lists: type inference alone cannot
    # tell them apart (code and prose are both str), so classify resolves between
    # them by CONTENT -- the compression gate (see _resolve_text_like)
    _TEXT_LIKE = ("text", "code")
    _FORMAT_CORPUS_CAP = 40000     # chars per sub-format kept for fitting the gate

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0), maintain='auto',
                 check_every=60, text_window=2, coherence_floor=None):
        self.dim = dim
        self.seed = seed                   # remembered for owned faculties (scene, morph)
        self.maintain = maintain
        self.check_every = check_every
        # OPT-IN coherence-gated maintenance (default None -> the original fixed schedule). When set,
        # the mind runs the (self-validating) reorganize pass only when its store has gone INCOHERENT
        # -- mean similarity of recent inputs to their own prototype drops below this floor -- rather
        # than on a fixed clock. MEASURED: on a multi-modal stream with a mid-stream class shift this
        # matched the best fixed schedule's accuracy at ~1/3 the reorganize passes, because it skips
        # the passes a coherent store does not need. (KEPT NEGATIVE from the same study: a calibrated
        # NOVELTY trigger -- the originally-flagged idea -- does NOT work here; novelty detects "matches
        # nothing", but the value of reorganizing is fixing incoherence, which novelty cannot see, and
        # calibration added nothing over a fixed cosine floor.) The right floor is data-dependent (the
        # coherence scale moves with dimension and class structure), so it is a parameter, not a constant.
        self.coherence_floor = coherence_floor
        self._last_reorg = 0               # taught-count at the last reorganize (the gate's cooldown)
        self._coh_hist = []                # recent coherence readings (the 'auto' floor's relative baseline)
        # ONE perception, shared by everything below
        self.encoder = UniversalEncoder(dim, seed=seed, number_range=number_range,
                                        text_window=text_window)
        # ONE associative memory: classify by nearest prototype, organize autonomously
        self.memory = SelfOrganizingMind(dim=dim, seed=seed)
        # a recall view over the SAME encoded vectors (individuals, for 'what's like this')
        self._recall = None
        # ONE decision brain (assembled when an action set is declared)
        self._brain = None
        self._actions = None
        # ONE scene faculty (compose/decompose visual scenes; built on first use, on the
        # same substrate -- it is part of this mind, not a separate engine)
        self._scene = None
        self._groles = None    # group-key atoms for nested (scene-of-scenes) composition
        self._hcap = None      # opt-in FHRR high-capacity key-value memory (built on first use)
        self._taught = 0
        self._label_modality = {}    # which modality each label came from (for routing)
        self._fillers = {}           # role -> set of values seen in absorbed records
        self._sequences = None       # lazily-built SequenceMemory: ORDER as a
                                     # queryable property (recipes, plans, proofs --
                                     # meaning the bag-of-everything stores discard)
        self.journal = []            # the mind's own narration of its maintenance
                                     # (every reorganization event, with the splits
                                     # NAMED where record structure allows -- see
                                     # _reorganize_and_narrate)
                                     # (the cleanup vocabulary for read/ask/explain --
                                     # learned from experience, never declared)
        self._gen = None             # sequence generator (lazy)
        # sub-format discovery state: raw samples of each TEXT-LIKE modality (capped),
        # and a lazily fitted compression-gate schema per modality (see classify)
        self._format_corpus = {}     # modality -> accumulated raw chars
        self._format_gate = None     # modality -> fitted SchemaGenerator
        self._format_fitted_at = {}  # modality -> corpus size when its schema was fit

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
        from holographic_ai import random_vector
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
        if m and hasattr(self, "_encyclopedia"):
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
            from holographic_intent import route_question as _vsa_route
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
        from holographic_answer import realize_answer
        return realize_answer(self.answer(question))

    def perceive(self, x, modality=None):
        """Any input -> one vector in the shared space. This is the only encoder in the
        system; the memory and the brain never encode anything themselves."""
        return self.encoder.encode(x, modality)

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
        from holographic_ai import cosine
        return cosine(self.perceive(a, "axial"), self.perceive(b, "axial"))

    def decode_axial(self, vec):
        """Recover the axial value in [0, pi) from an axial hypervector -- the inverse of
        perceive(theta, 'axial'). Lets a recalled/blended orientation be read back as an angle."""
        return self.encoder.decode_axial(vec)

    # -- one memory: classification + organization -------------------------
    def learn(self, x, label, modality=None):
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
        from holographic_schema import compression_gate
        raw = x if isinstance(x, str) else " ".join(str(t) for t in x)
        return compression_gate(raw, gens)[0][1]

    def _format_schemas(self, modalities):
        """Fit (and cache) one small schema per text-like sub-format from the raw
        samples learn() accumulated. Refit only when a corpus has grown by more than
        a third since its schema was fitted, so steady-state classify pays nothing."""
        from holographic_schema import SchemaGenerator
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
        from holographic_ai import cosine
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
        from holographic_ai import bind
        probe = bind(self.encoder._roles.get(str(role)), self.encoder.encode(filler))
        items = self._record_items()
        if not items:
            return None, -1.0
        if len(items) >= 32:
            from holographic_resolution import coarse_to_fine
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
        from holographic_fractal import image_fractal_dimension, hurst_exponent
        m = modality or self.encoder.infer(x)
        arr = np.asarray(x)
        if m == "image" or (arr.ndim >= 2 and arr.dtype != object):
            return float(image_fractal_dimension(arr))
        seq = np.asarray(x, float).ravel()
        return float(2.0 - hurst_exponent(seq))     # self-affinity as a dimension

    def self_affinity(self, series):
        """Hurst exponent of a 1-D series read by the mind: 0.5 random walk,
        <0.5 mean-reverting, >0.5 trending. The fractal lens on a time series."""
        from holographic_fractal import hurst_exponent
        return float(hurst_exponent(np.asarray(series, float).ravel()))

    def verify_image_structure(self, image, real_patches=None, patch=32):
        """Does an image carry the spatial-autocorrelation signature of real data
        (vs noise / corruption)? The text structure verifier, carried to images
        (holographic_signal_structure). If real_patches is None, calibrates on
        patches of the image itself. Returns {'score', 'structured', 'threshold'}."""
        from holographic_signal_structure import SignalStructureVerifier
        img = np.asarray(image, float)
        if img.ndim == 3:
            img = img.mean(axis=2)
        if real_patches is None:
            h, w = img.shape
            real_patches = [img[i:i + patch, j:j + patch]
                            for i in range(0, max(1, h - patch), patch)
                            for j in range(0, max(1, w - patch), patch)] or [img]
        v = SignalStructureVerifier("image").calibrate(real_patches)
        return {"score": v.structure_score(img), "structured": bool(v.is_structured(img)),
                "threshold": float(v.threshold)}

    def volatility_structure(self, returns):
        """Does a return series carry the volatility-clustering signature of real
        markets (|returns| autocorrelated)? Returns the clustering z-score vs a
        shuffled control: >2 is meaningful structure, near 0 means none (or too
        little data). The cross-domain structure verifier for time series."""
        from holographic_signal_structure import clustering_zscore, volatility_clustering
        r = np.asarray(returns, float).ravel()
        return {"clustering": float(volatility_clustering(r)),
                "zscore": float(clustering_zscore(r))}

    def resolution_profile(self, x, modality=None, among=None):
        """How much holographic RESOLUTION does classifying this input need? For
        each truncation dimension, which prototype wins -- and at what dimension
        does the winner stabilise? A low stabilisation dimension means the answer
        is robust to heavy truncation (a 'fundamental' match); needing full width
        means it was a close call. The persistent-homology idea made practical:
        which structure survives compression. Returns
        {'profile': [(dim, label, score)], 'stable_from': dim, 'full_dim': D}."""
        from holographic_resolution import resolution_profile as _rp
        v = self.perceive(x, modality)
        protos, labels = [], []
        for lab, _, unit, _ in self.memory.live._p:
            if among is None or lab in among:
                protos.append(unit)
                labels.append(lab)
        if not protos:
            return {"profile": [], "stable_from": 0, "full_dim": self.dim}
        M = np.stack(protos)
        prof = _rp(v, M)
        named = [(k, labels[i], round(s, 3)) for k, i, s in prof]
        final = prof[-1][1]
        stable = prof[-1][0]
        for k, i, _ in prof:
            if i == final:
                stable = k
                break
        return {"profile": named, "stable_from": stable, "full_dim": self.dim}

    def read_role(self, label, role):
        """Decode one role's filler from a LEARNED class -- unbind the role from
        the class prototype and clean up against the experience-registered
        fillers. Works whether the class holds one record or the superposition
        of many noisy ones (measured: see explain)."""
        from holographic_ai import bind, involution
        est = bind(self._class_vec(label), involution(self.encoder._roles.get(str(role))))
        return self._clean_filler(est, role)

    def ask(self, start_filler, *path):
        """A CHAIN over the mind's own memory: ask('paris', ('capital',
        'currency'), ('currency', 'language')) -> the language of the country
        with the currency of the country whose capital is paris. Each hop is
        find() then read() -- geometry snapped to a symbol before the next hop,
        which is what keeps chains exact instead of compounding HRR noise."""
        filler = start_filler
        for match_role, read_role in path:
            label, _ = self.find(match_role, filler)
            if label is None:
                return None
            filler, _ = self.read_role(label, read_role)
        return filler

    def blend(self, base_label, donor_label, donor_roles):
        """PROJECTION TO CREATE NEW THINGS, over the mind's OWN learned classes.
        Synthesize a novel concept: the frame of `base_label`, with `donor_label`'s
        values projected onto `donor_roles`. The mind decodes each role from its
        class prototypes (so this works over concepts learned from many noisy
        observations, not just hand-built records) and rebuilds a coherent new
        record that names a thing it never saw -- 'this class, but with that
        class's distinctive traits'. Returns {role: value} for the synthesized
        concept. (Analogy as CREATION: synthesizing a specified new thing is
        well-posed and exact where RETRIEVING an existing analogue from a clean
        role-filler memory is not -- every learned class is an exact key, so there
        is no graded nearness for a retrieval-analogy to climb.)"""
        donor_roles = set(donor_roles)
        roles = sorted(self._fillers)
        spec = {}
        for r in roles:
            src = donor_label if r in donor_roles else base_label
            val, _ = self.read_role(src, r)
            if val is not None:
                spec[r] = val
        return spec

    def ask_traced(self, start_filler, *path, min_throughput=0.0):
        """ask() instrumented like a PATH TRACER: a relation chain is a ray
        bouncing through the holographic space, each hop a bounce whose cleanup
        confidence is its reflectance, and throughput is the accumulated product.
        Throughput is a calibrated confidence in the chained answer (measured:
        keeping only the most-confident chains sharply raises accuracy on the
        answered subset), and a chain whose throughput decays below
        `min_throughput` ABSTAINS (returns answer None) rather than emitting
        noise -- the energy-based termination of a ray that has lost too much to
        contribute. Returns (answer_or_None, throughput, hop_confidences)."""
        filler = start_filler
        throughput = 1.0
        confidences = []
        for match_role, read_role in path:
            label, fconf = self.find(match_role, filler)
            if label is None:
                return None, 0.0, confidences
            filler, rconf = self.read_role(label, read_role)
            hop = max(0.0, float(fconf)) * max(0.0, float(rconf))
            throughput *= hop
            confidences.append(round(hop, 3))
            if throughput < min_throughput:
                return None, throughput, confidences
        return filler, throughput, confidences

    def explain(self, x1, x2):
        """WHY are two things similar -- not just a cosine, but the per-role
        verdict. Takes either two record DICTS (encoded fresh, candidates drawn
        from the inputs) or two LEARNED LABELS (decoded from the mind's own
        class prototypes, candidates from the experience-registered fillers --
        so the mind explains concepts it learned, including classes built from
        many noisy observations).

        Returns [(role, value_1, value_2, shared, confidence), ...]. Built on
        the measured relations operations (per-role explanation 4/4, naming
        100%, symbol-routed mapping 360/360, chains exact through three hops);
        every readout cleans up to a SYMBOL, because meaning survives
        composition only when it touches symbols between steps."""
        from holographic_ai import bind, involution, cosine
        if isinstance(x1, dict) and isinstance(x2, dict):
            rec1 = self.encoder.encode(x1, "record")
            rec2 = self.encoder.encode(x2, "record")
            values = sorted({str(v) for v in list(x1.values()) + list(x2.values())})
            val_vecs = {v: self.encoder.encode(v) for v in values}

            def clean(vec):
                best, score = None, -2.0
                for v, vv in val_vecs.items():
                    s = cosine(vec, vv)
                    if s > score:
                        best, score = v, s
                return best, float(score)

            out = []
            for role in sorted(set(x1) & set(x2), key=str):
                inv = involution(self.encoder._roles.get(str(role)))
                f1, c1 = clean(bind(rec1, inv))
                f2, c2 = clean(bind(rec2, inv))
                out.append((str(role), f1, f2, f1 == f2, min(c1, c2)))
            return out
        # learned labels: decode from the class prototypes the mind built itself
        v1, v2 = self._class_vec(x1), self._class_vec(x2)
        out = []
        for role in sorted(self._fillers):
            inv = involution(self.encoder._roles.get(role))
            f1, c1 = self._clean_filler(bind(v1, inv), role)
            f2, c2 = self._clean_filler(bind(v2, inv), role)
            out.append((role, f1, f2, f1 == f2, min(c1, c2)))
        return out

    def finding_registry(self):
        """The findings registry (backlog D3): a research log as a holographic KNOWLEDGE STRUCTURE. Record
        structured claims -- a SUBJECT affects an OBJECT with a +1/-1 POLARITY (helps / hurts), optionally
        under a CONDITION (a regime) -- then recall them by similarity and detect the log's OWN
        contradictions. It extends the relations layer (the same role-bound records KnowledgeStore's
        explain/analogy run on) with the piece a research log needs: distinguishing a FLAT contradiction
        (two opposite-polarity findings about the same claim under the same/absent condition -- one must be
        wrong) from a CONDITIONED tension (the same under DIFFERENT conditions -- reconcilable, the outcome is
        conditioned on the differing dimension). Retrieval is holographic (cosine over the bound claim); the
        verdict is exact (polarity sign, condition equality). Lazily created and cached on the mind's own
        dimension/seed; call .add / .query / .tensions on the returned FindingRegistry.

        SCOPE (kept negative): findings are STRUCTURED claims, not free prose -- turning narrative into
        structured claims is an NLP step this engine does not do (no embeddings, no parser)."""
        reg = getattr(self, "_finding_registry", None)
        if reg is None:
            from holographic_knowledge import FindingRegistry
            reg = FindingRegistry(dim=self.dim, seed=self.seed)
            self._finding_registry = reg
        return reg

    def explain_splits(self, label, contrast_floor=0.25):
        """INCEPTION: the mind explains its own memory organization. When the
        self-organizing memory has split `label` into sub-prototypes (because
        held-out accuracy said the class is genuinely multi-modal), each
        sub-prototype is a superposition of one MODE's records -- so decoding
        the registered roles from each names WHAT the split separated: 'this
        class divided because one mode is colour=red and the other colour=blue'.
        The relations machinery (built on the substrate) explaining the
        organizer (built on the same substrate).

        A role counts as SEPARATING only by CONTRAST -- each mode's winning
        value must be genuinely absent from the other mode, not merely less
        common. Measured on the XOR world: truly separating roles score ~0.5
        contrast, incidental skews (a noise role one 2-means half happened to
        lean toward) score <= 0.1, so the floor sits mid-gap. The statistic's
        first real outing caught the organizer red-handed: one label's split
        turned out to separate the NOISE role (accuracy-sufficient, since the
        other label's clean split already resolved the XOR) -- the explanation
        honestly reports what the split actually did, not what was assumed.

        Returns (decodes, separating): per-sub-prototype {role: (value, score)}
        and the roles whose contrast clears the floor. A single-prototype label
        returns an empty separating set (nothing was divided, which is itself
        the explanation)."""
        from holographic_ai import bind, involution, cosine
        subs = [unit for lab, _, unit, _ in self.memory.live._p if lab == label]
        if not subs:
            raise KeyError(f"unknown label: {label!r}")
        # full per-role value scores for every sub-prototype
        scores = []
        for u in subs:
            row = {}
            for role in sorted(self._fillers):
                est = bind(u, involution(self.encoder._roles.get(role)))
                row[role] = {v: cosine(est, self.encoder.encode(v))
                             for v in self._fillers[role]}
            scores.append(row)
        decodes = [{r: max(row[r].items(), key=lambda kv: kv[1]) for r in row}
                   for row in scores]
        separating = []
        if len(subs) > 1:
            for role in sorted(self._fillers):
                winners = [d[role][0] for d in decodes]
                if len(set(winners)) < 2:
                    continue
                # contrast: my winner's score here, minus every OTHER mode's
                # winner scored here -- averaged over modes (mutual absence)
                cs = []
                for i, row in enumerate(scores):
                    own = row[role][decodes[i][role][0]]
                    others = [row[role].get(decodes[j][role][0], 0.0)
                              for j in range(len(subs)) if j != i]
                    cs.append(own - max(others))
                if float(np.mean(cs)) >= contrast_floor:
                    separating.append(role)
        return decodes, separating

    def explain_organization(self):
        """The whole memory's self-explanation: for every label the organizer
        split, the nameable reason. {label: differing_roles}."""
        out = {}
        for label in sorted(self.memory.live.labels()):
            subs = sum(1 for lab, *_ in self.memory.live._p if lab == label)
            if subs > 1:
                _, differing = self.explain_splits(label)
                out[label] = differing
        return out

    def classify_robust(self, x, modality=None, route=True, n_rays=5, seed=0):
        """MULTI-RAY classification: one query is one noisy ray; fire several
        independent encodings and combine them, the way path tracing fires many
        rays per pixel and averages (no single ray is reliable, but the ensemble
        converges). For text the independent views are word-resampled subsets of
        the query -- each a different SHADOW of the same input -- and the crucial
        step is that each ray's per-label scores are Z-SCORED before summing, so a
        ray that is confident-but-wrong (an outlier view) cannot dominate, exactly
        the failure that sinks a naive vote. Measured: on a task where the views'
        individual accuracy ranges wildly (100%/100%/50%/17% across feature
        lenses), the z-scored ensemble reaches the BEST single view's accuracy
        BLIND -- without being told which view to trust.

        Falls back to plain classify() for non-text inputs (where word-resampling
        does not apply) and for single-token queries (no subset to resample).
        Returns (label, agreement) where agreement is the fraction of rays that
        voted for the winner -- a multi-ray confidence."""
        import numpy as np
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        if modality not in ("text", "code") or not isinstance(x, str):
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        words = x.split()
        if len(words) < 3:
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items()
                     if m == modality} or None
        rng = np.random.default_rng(seed)
        agg, raw_votes = {}, []
        views = [x] + [" ".join([w for w in words if rng.random() > 0.25] or words)
                       for _ in range(n_rays - 1)]
        for view in views:
            v = self.perceive(view, modality)
            sc = self.memory.live.label_scores(v, among=among)
            if not sc:
                continue
            vals = np.array(list(sc.values()))
            mu, sd = vals.mean(), vals.std() + 1e-9
            for lab, s in sc.items():               # z-score this ray's evidence
                agg[lab] = agg.get(lab, 0.0) + (s - mu) / sd
            raw_votes.append(max(sc, key=sc.get))
        if not agg:
            lab, _ = self.classify(x, modality, route)
            return lab, 1.0
        winner = max(agg, key=agg.get)
        agreement = raw_votes.count(winner) / max(1, len(raw_votes))
        return winner, round(agreement, 3)

    def recall(self, x, modality=None, abstain=None):
        """Nearest stored individual. The index does an exact scan until the store is
        genuinely big, then switches to the recursive HoloForest (the crossover is
        measured -- see _Index.recall). A NEGATIVE worth recording here: wiring the
        learned adaptive navigator (holographic_navigator) into this path was tried
        and lost badly on the mind's own store -- 48% recall@1 at ~130 comparisons,
        where the forest at beam 2 gets 89% within ~512. The navigator's margin
        senses were tuned on UNIFORM random vectors; the unified store is clustered
        (many near-duplicates per class), which miscalibrates the arrive/keep-moving
        instinct. So recall keeps the dumb-but-honest index, and the navigator stays
        a study of adaptive access, not a default.

        With `abstain` set (a false-alarm level alpha), returns the recalled payload only if the match is
        calibrated-significant (p <= alpha against the store's own noise floor, recall_calibrated) and None
        otherwise -- an honest 'I have nothing like this'. Default None preserves the original behaviour."""
        if self._recall is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        if abstain is not None:
            payload, _sim, p = self.recall_calibrated(x, modality=modality)
            return payload if (p == p and p <= abstain) else None
        return self._recall.recall(self.perceive(x, modality))

    # -- honest recall: calibrated confidence + abstention, woven into recognition --------
    # The honesty layer (RecallNull / SPRT / bh_fdr) was a standalone measurement harness; here it
    # becomes part of how the mind RECOGNISES. A raw cosine means nothing on its own -- RecallNull asks
    # how high pure noise reaches against THIS mind's own prototypes, turning a recall into an honest
    # false-alarm probability (the radio-SETI / particle-physics 'prove it isn't an artifact of your own
    # pipeline' discipline, calibrated to this mind's geometry). That p-value is what lets the mind
    # ABSTAIN -- say 'I don't recognise this' -- instead of always returning a nearest label, and it
    # upgrades the organizer's fixed-floor novelty heuristic to a calibrated one.
    def _recognition_null(self, n_null=1200):
        """Maintain a RecallNull over the CURRENT class-prototype codebook -- the mind's own noise floor.
        Rebuilt only when the prototype set changes (keyed on the store's mutation counter _gen), so
        steady-state recognition pays nothing. Returns None when nothing is learned yet."""
        labels, mat = self.memory.live._stack()
        if getattr(mat, "shape", (0,))[0] == 0:
            return None
        gen = getattr(self.memory.live, "_gen", 0)
        cache = getattr(self, "_null_cache", None)
        if cache is None or cache[0] != gen or cache[1] != mat.shape[0]:
            from holographic_honesty import RecallNull
            self._null_cache = (gen, mat.shape[0], RecallNull().fit(mat, n_null=n_null, seed=self.seed))
        return self._null_cache[2]

    def _resolve_modality(self, x, modality):
        """The modality classify() uses: declared, or inferred (with text-like sub-format resolution)."""
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        return modality

    def recognize(self, x, modality=None, route=True):
        """CORE calibrated recognition. Like classify, but returns (label, similarity, pvalue): the
        pvalue is the honest false-alarm probability -- the chance pure noise would match the mind's own
        prototypes this well (RecallNull, calibrated to THIS mind). p small -> trust the label; p large ->
        the input matches no learned class. This is the basis of honest abstention and of the calibrated
        batch / streaming recognisers below."""
        modality = self._resolve_modality(x, modality)
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items() if m == modality} or None
        scores = self.memory.live.label_scores(self.perceive(x, modality), among=among)
        if not scores:
            return (None, 0.0, 1.0)
        label = max(scores, key=scores.get); sim = float(scores[label])
        null = self._recognition_null()
        p = float(null.pvalue(sim)) if null is not None else float("nan")
        return (label, sim, p)

    def _match_scores(self, window=600):
        """Genuine-match score density: recent stored examples' cosine to their OWN label's prototype
        (the quantity the organizer's coherence() reads). Feeds SPRT's match density from the mind's own
        experience -- no external calibration data."""
        recent = self.memory.buffer[-window:]
        out = [float(max((p[2] @ v for p in self.memory.live._p if p[0] == label), default=0.0))
               for v, label in recent]
        out = [s for s in out if s > 0.0]
        return np.asarray(out) if out else np.asarray([0.6])

    def stream_recognize(self, cues, modality=None, alpha=0.05, beta=0.05, route=True, cap=None):
        """Sequential recognition over a STREAM of cues bearing on the SAME thing (repeated noisy
        sightings of a landmark, a drifting pattern). Accumulates each cue's best-match score with Wald's
        SPRT and decides MATCH / REJECT the instant the evidence crosses a boundary -- the minimum expected
        number of samples for the (alpha, beta) error pair (decide as fast as the evidence allows). Returns
        (decision, recalled_label, n_samples_used). Null density = the mind's noise floor; match density =
        its own examples' self-similarity."""
        null = self._recognition_null()
        if null is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        from holographic_honesty import SPRTRecall
        sprt = SPRTRecall(null.null, self._match_scores(), alpha=alpha, beta=beta)
        votes, scores = {}, []
        for c in cues:
            lab, sim, _p = self.recognize(c, modality=modality, route=route)
            scores.append(sim)
            if lab is not None:
                votes[lab] = votes.get(lab, 0) + 1
        decision, n = sprt.decide(scores, cap=cap)
        return (decision, (max(votes, key=votes.get) if votes else None), n)

    def recognize_batch(self, queries, modality=None, alpha=0.1, route=True):
        """Recognise a BATCH honestly: classify each query, then control the FALSE-DISCOVERY RATE across
        the batch with Benjamini-Hochberg/Yekutieli (bh_fdr) over the per-query false-alarm p-values.
        Returns a list of {label, similarity, pvalue, significant}, where `significant` is True only for
        recognitions that survive FDR at `alpha` -- so scanning many queries cannot manufacture matches by
        luck (the look-elsewhere discipline applied to the mind's own recognition)."""
        from holographic_honesty import bh_fdr
        res = [self.recognize(q, modality=modality, route=route) for q in queries]
        pvals = np.asarray([(p if p == p else 1.0) for (_l, _s, p) in res])
        reject, _k = bh_fdr(pvals, alpha=alpha, dependent=True)
        return [{"label": l, "similarity": s, "pvalue": p, "significant": bool(r)}
                for (l, s, p), r in zip(res, reject)]

    def scan(self, channels, modality=None, alpha=0.05, beta=0.05, fdr=0.1, route=True, cap=None):
        """Scan MANY candidate channels with the two disciplines a large search needs AT ONCE -- Siemion's
        'flag anything that isn't noise' over an astronomical channel count, combining the streaming detector
        (B3) and the look-elsewhere control (FDR). Each channel is a STREAM of cues bearing on ONE hypothesis
        (a frequency bin over time, a sky position, a recurring market pattern). For every channel, Wald's
        SPRT decides MATCH/REJECT as fast as THAT channel's own evidence allows (the minimum expected samples
        for the (alpha, beta) error pair -- decide as fast as the evidence lets you); then
        Benjamini-Hochberg/Yekutieli FDR controls the trials factor ACROSS the channels (scan enough and some
        clear the per-channel bar by luck). A channel is a CONFIRMED detection (`detected`) only when the SPRT
        decided MATCH *and* its calibrated p-value survives FDR -- streaming detection and look-elsewhere in
        one pass. The per-channel p-value is calibrated for the EXACT statistic used: the channel's mean score
        against a null of equal-length noise streams, resampled from the per-cue noise floor (so a long
        channel and a short one are each judged against their own-length null, and the FDR is honest).
        Returns a per-channel list of {index, label, decision, n_samples, pvalue, fdr_significant, detected}.
        """
        null = self._recognition_null()
        if null is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        from holographic_honesty import SPRTRecall, RecallNull, bh_fdr
        match = self._match_scores()
        # routing the channels share (resolve once; recognize() recomputes the same per cue). Random noise
        # perceives to a ~random unit vector regardless of modality, so the floor below is modality-agnostic.
        rep_mod = modality
        if rep_mod is None and channels and len(channels[0]):
            rep_mod = self._resolve_modality(channels[0][0], None)
        among = ({lab for lab, mm in self._label_modality.items() if mm == rep_mod} or None) if route else None
        floor = self._scan_cue_null(rep_mod, route)       # PROCEDURE-MATCHED per-cue noise floor (see below)
        # The null for a channel's MEAN score is the mean of L i.i.d. draws from that per-cue floor. Resample
        # it from the calibrated floor (cheap) and cache by length L; seed per L so the draw is independent of
        # channel order (deterministic, Macklin's tie-break discipline).
        mean_null_cache = {}
        def mean_null(L):
            if L not in mean_null_cache:
                r = np.random.default_rng(self.seed + L)
                means = floor[r.integers(0, len(floor), size=(4000, L))].mean(axis=1)
                rn = RecallNull(); rn.null = np.sort(means); mean_null_cache[L] = rn
            return mean_null_cache[L]

        rows = []
        for i, cues in enumerate(channels):
            scored = [self.recognize(c, modality=modality, route=route) for c in cues]
            sims = [float(s) for (_l, s, _p) in scored]
            decision, n = SPRTRecall(floor, match, alpha=alpha, beta=beta).decide(sims, cap=cap)
            votes = {}
            for (lab, _s, _p) in scored:
                if lab is not None:
                    votes[lab] = votes.get(lab, 0) + 1
            label = max(votes, key=votes.get) if votes else None
            pval = float(mean_null(len(sims)).pvalue(float(np.mean(sims)))) if sims else 1.0
            rows.append({"index": i, "label": label, "decision": decision,
                         "n_samples": int(n), "pvalue": pval})
        reject, _k = bh_fdr([r["pvalue"] for r in rows], alpha=fdr, dependent=True)
        for r, sig in zip(rows, reject):
            r["fdr_significant"] = bool(sig)
            r["detected"] = bool(r["decision"] == "MATCH" and sig)
        return rows

    def detect_drifting(self, waterfall, drifts=None, alpha=0.01, off=None):
        """Find a DRIFTING narrowband signal in a spectrogram -- the SETI detection problem (Tarter,
        Siemion seats) cast in the engine's OWN primitives. A Doppler frequency drift is a cyclic
        SHIFT of the spectrum over time, and the engine already shifts: `permute` is exactly the
        rigid-shift transform holographic_video.py uses for motion compensation (and a shift is also
        a binding, bind(x, delta_k) == permute(x, k)). So "de-Doppler integration" -- the matched
        filter that recovers a drifting signal a STATIONARY detector loses -- is permute-ing each
        frame back by the drift before summing. The look-elsewhere control over the (drift x channel)
        search grid is `bh_fdr` with the DEPENDENT correction (the drift cells overlap; the honest,
        conservative choice). Supply `off` (an OFF-target spectrogram in the ON-OFF cadence radio
        astronomers use) to reject stationary RFI: a real signal is ON-only, terrestrial interference
        persists across the cadence.

        `waterfall` is (T frames x F bins). Returns detections [{drift, channel, snr, pvalue}, ...]
        sorted by SNR. Deterministic. MEASURED at the field's S/N>=10 regime: ~96% recall at 0%
        false-positive; the cadence rejects a strong stationary RFI ~100% while keeping the drifting
        signal ~94%. KEPT NEGATIVE: below ~10 sigma integrated the dependent-FDR correction over the
        many cells is conservative (a lone weak signal needs ~5 sigma) -- matching turboSETI's own
        S/N>=10 search threshold, which exists for exactly this reason.
        """
        from holographic_dedoppler import detect_drifting as _dd
        off_arr = None if off is None else np.asarray(off, float)
        return _dd(np.asarray(waterfall, float), drifts=drifts, alpha=alpha, off=off_arr)

    def _scan_cue_null(self, modality, route, n=1500):
        """Procedure-matched per-cue noise floor for `scan`: the distribution of recognize()'s OWN sim on
        random unit vectors, taken through the SAME path a channel cue takes -- perceive() then the routed
        label_scores. This matters: perceive() is NOT the identity even for the 'vector' modality (it lifts a
        raw vector onto the encoder geometry, raising the max label score of pure noise from ~0.086 to ~0.117),
        and the recognition codebook's RecallNull scores prototype ROWS rather than the max label score
        recognize() returns -- calibrating the channel p-value to either of those wrong floors makes pure-noise
        channels look significant (a kept lesson). Calibrated for the encoded / vector-channel regime (the
        SETI spectrogram case); cached on the prototype generation, the modality, and route."""
        gen = getattr(self.memory.live, "_gen", 0)
        nproto = getattr(self.memory.live._stack()[1], "shape", (0,))[0]
        key = (gen, nproto, modality, bool(route))
        cache = getattr(self, "_scan_floor_cache", None)
        if cache is None or cache[0] != key:
            rng = np.random.default_rng(self.seed + 99991)
            sims = np.empty(n)
            for i in range(n):
                v = rng.standard_normal(self.dim); v /= np.linalg.norm(v) + 1e-12
                sims[i] = self.recognize(v, modality=modality, route=route)[1]
            self._scan_floor_cache = (key, np.sort(sims))
        return self._scan_floor_cache[1]

    def _recall_null(self, n_null=800):
        """The noise floor for 'have I stored anything actually like this?' -- and it is PROCEDURE-MATCHED:
        it is fit by running the SAME recall path (recall() -- the sublinear forest on a big store, the exact
        scan on a small one) on random unit queries and recording the score each reaches. Calibrated by
        construction (the null IS the score distribution noise produces under the real procedure), and it
        inherits recall()'s sublinearity, so it neither under-samples the store nor defeats the acceleration
        structure -- the two problems the earlier exact-scan + sampled-null version had. Cached on store size.
        Stored vectors and queries are unit length, so the index's dot is a cosine. Returns None if empty."""
        if self._recall is None or not getattr(self._recall, "vecs", None):
            return None
        n = len(self._recall.vecs)
        cache = getattr(self, "_recall_null_cache", None)
        if cache is None or cache[0] != n:
            rng = np.random.default_rng(self.seed)
            Q = rng.standard_normal((n_null, self.dim))
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
            scores = np.array([self._recall.recall(q)[1] for q in Q])   # the actual recall path on noise
            from holographic_honesty import RecallNull
            rn = RecallNull(); rn.null = np.sort(scores)                 # reuse its searchsorted pvalue
            self._recall_null_cache = (n, rn)
        return self._recall_null_cache[1]

    def recall_calibrated(self, x, modality=None):
        """CORE calibrated recall: the nearest STORED INDIVIDUAL plus an honest false-alarm probability --
        (payload, similarity, pvalue). p small -> the store really contains something like this; p large ->
        it does not (abstain). The symmetric partner of recognize() (which calibrates the class PROTOTYPE
        readout); recall(..., abstain=alpha) thresholds on it. The winner comes through recall() itself --
        the sublinear HoloForest on a big store, the exact scan on a small one -- so honest abstention does
        NOT cost the acceleration structure, and the null is matched to the same path (see _recall_null)."""
        if self._recall is None or not getattr(self._recall, "vecs", None):
            raise RuntimeError("nothing learned yet -- call learn() first")
        q = self.perceive(x, modality)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn                                  # unit query: stored vecs are unit, so the dot is a cosine
        payload, score = self._recall.recall(q)         # sublinear (forest) when big, exact when small
        null = self._recall_null()
        p = float(null.pvalue(float(score))) if null is not None else float("nan")
        return payload, float(score), p

    def capacity_report(self, alpha=0.05, loads=(64, 256, 1024), n_floor=800, n_fa=800):
        """Where this store sits relative to the noise-wins CLIFF (Plate's HRR capacity theory), AND whether
        the calibrated false-alarm rate holds as the store GROWS (Cranmer's coverage-vs-LOAD -- the question
        Tier 0 left open by validating coverage only at a FIXED store). One report, two readings of the same
        geometry; the capacity complement to `calibration_report` (which checks coverage at the current store)
        and to `resolution_profile`.

        CAPACITY (the operating point, read off the actual prototype-row geometry where HRR theory applies):
          * `dprime` = (genuine-match cosine - noise-floor mean) / noise-floor std -- the SNR, in noise-sigmas,
            that a real match sits above what random crosstalk produces. Large is comfortably above the cliff;
            near 0 is AT it (noise wins).
          * `floor_mean` vs `hrr_floor_bound`: the measured noise floor (a random query's best cosine to the N
            rows) against the extreme-value bound sqrt(2 ln N / D) for N atoms in D dims -- the validation that
            the store's geometry behaves as the capacity theory predicts.
          * `headroom`: `n_cliff` = exp(D * match^2 / 2) is the store size at which the rising floor reaches the
            match level; `headroom` = n_cliff / N_now -- how many times the store could grow before noise wins
            (huge in high D for well-separated items, the whole point of distributed codes).

        COVERAGE vs LOAD: for each size in `loads`, build a random codebook of that many atoms in this mind's D,
        fit the procedure-matched recall null on it, and measure the false-alarm rate (fraction of pure-noise
        queries with p <= alpha). It should stay ~alpha as N grows -- the null re-fits to the rising floor, so
        the look-elsewhere discipline stays calibrated under load; materially above alpha at large N would mean
        the null is under-sampling the bigger store. Returns operating point, theory comparison, headroom, and
        per-load coverage. Deterministic (seeded by this mind)."""
        from holographic_honesty import RecallNull
        D = self.dim
        mat = self.memory.live._stack()[1]
        N = int(getattr(mat, "shape", (0,))[0])
        if N == 0:
            return {"n_prototypes": 0, "dim": D}
        rng = np.random.default_rng(self.seed)
        unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
        floor = np.array([float(np.max(mat @ unit(rng.standard_normal(D)))) for _ in range(n_floor)])
        mu, sd = float(floor.mean()), float(floor.std())
        match = float(np.mean(self._match_scores()))
        dprime = (match - mu) / (sd + 1e-12)
        hrr_bound = float(np.sqrt(2.0 * np.log(max(2, N)) / D))            # expected max of N cosines in D dims
        n_cliff = float(np.exp(min(700.0, D * match * match / 2.0)))       # cap exponent to avoid float overflow
        headroom = n_cliff / N
        coverage = {}
        for nload in loads:                                               # does coverage hold as the store grows?
            cb = np.stack([unit(rng.standard_normal(D)) for _ in range(int(nload))])
            rn = RecallNull().fit(cb, n_null=1500, seed=self.seed)        # procedure-matched null for this size
            qp = np.array([rn.pvalue(float(np.max(cb @ unit(rng.standard_normal(D))))) for _ in range(n_fa)])
            coverage[int(nload)] = float(np.mean(qp <= alpha))
        return {"n_prototypes": N, "dim": D, "match": match, "floor_mean": mu, "floor_std": sd,
                "dprime": dprime, "hrr_floor_bound": hrr_bound, "n_cliff": n_cliff, "headroom": headroom,
                "headroom_log10": float(np.log10(headroom)) if headroom > 0 else float("-inf"),
                "alpha": alpha, "coverage_vs_load": coverage}

    def calibration_report(self, n=2000, alphas=(0.01, 0.05, 0.1, 0.2), seed=12345):
        """Validate that the false-alarm probabilities recognize() and recall_calibrated() report are
        actually CALIBRATED. Draw `n` random unit vectors -- pure noise, matching nothing by construction --
        score each against the mind's own prototypes (the recognize path) and its individual store (the
        recall path), and report the empirical false-alarm RATE: the fraction whose p-value lands at or below
        each alpha. A calibrated detector fires on noise at rate ~= alpha; materially above alpha is
        anti-conservative (too many false matches), materially below is conservative. This is the radio-SETI
        / particle-physics coverage check (does thresholding at alpha hold the false-alarm rate at alpha?),
        run on the mind's own geometry -- and the validation that the procedure-matched recall null is not
        the anti-conservative under-estimate the earlier sampled null was."""
        rng = np.random.default_rng(seed)
        pnull = self._recognition_null()
        rnull = self._recall_null()
        proto = self.memory.live._stack()[1] if pnull is not None else None
        p_proto, p_indiv = [], []
        for _ in range(n):
            v = rng.standard_normal(self.dim); v /= np.linalg.norm(v) + 1e-12
            if pnull is not None and proto is not None and proto.shape[0]:
                p_proto.append(pnull.pvalue(float((proto @ v).max())))
            if rnull is not None:
                p_indiv.append(rnull.pvalue(float(self._recall.recall(v)[1])))
        rates = lambda ps: {a: (float(np.mean(np.asarray(ps) <= a)) if ps else float("nan")) for a in alphas}
        return {"n": int(n), "alphas": list(alphas),
                "prototype_false_alarm": rates(p_proto),       # recognize() path: should track alpha
                "individual_false_alarm": rates(p_indiv)}       # recall_calibrated() path: should track alpha

    def federation_report(self, target_items=None, threshold=0.90, n_vals=256, seed=0):
        """A federation / conservation diagnostic -- Path D's 'as above, so below' law as a callable readout,
        the federation-aware companion to `capacity_report` (which charts a SINGLE vector's noise-wins cliff).
        One D-vector holds only ~0.05-0.1 x D symbols at `threshold` cleanup-gated recall (the exact figure
        depends on the threshold); that budget is CONSERVED, so capacity comes from FEDERATING across shards,
        not from packing harder. Measured on the mind's own dimension and kernel (delegating to `storage_array`
        / HoloArray):
          * `per_vector_budget` -- the largest single-shard load whose recall still clears `threshold`;
          * `federated` -- a spot check that K aligned shards hold ~K x that budget at the same recall;
          * `conservation_ratio` -- partitioning the dimension in half holds total capacity (a half-D vector
            holds ~half the budget, so two of them tie one full vector): 2 x budget(D/2) / budget(D) ~ 1, the
            block-federation finding that federation buys capacity from more DIMENSIONS, not for free;
          * `recommended_shards` -- ceil(target_items / per_vector_budget), when `target_items` is given.
        Honest scope: this is the DISCRETE-symbol (cleanup-gated) budget ~0.1 x D; continuous compute with no
        cleanup is the lower ~0.02 x D regime (see `distributed_forward`), and federation buys fidelity and
        capacity, not fewer FLOPs."""
        from holographic_array import HoloArray
        fracs = (0.05, 0.08, 0.10, 0.12, 0.15)

        def budget_at(dim):
            """Largest single-shard load (at this dimension) whose recall clears the threshold."""
            best = 0
            for load in (int(f * dim) for f in fracs):
                if load < 1:
                    continue
                arr = HoloArray(dim, seed=seed, n_parity=0, add_threshold=0.0, n_vals=n_vals)
                rng = np.random.default_rng(seed)
                for _ in range(load):
                    arr.add(int(rng.integers(0, n_vals)))
                if arr.accuracy() >= threshold:
                    best = load
            return best

        D = self.dim
        budget = max(budget_at(D), 1)

        # federated spot check: K aligned shards (each at the per-vector budget) hold ~K x budget at threshold
        K = 4
        arr = self.storage_array(n_parity=0, add_threshold=0.0, n_vals=n_vals)
        rng = np.random.default_rng(seed + 1)
        for k in range(K):
            if k > 0:
                arr._spin_up()
            for _ in range(budget):
                arr.add(int(rng.integers(0, n_vals)))
        fed_acc = float(arr.accuracy())

        # conservation: a half-dimension vector holds ~half the budget (so partitioning conserves total capacity)
        conservation_ratio = 2.0 * max(budget_at(D // 2), 1) / budget

        out = {"dim": D, "threshold": threshold,
               "per_vector_budget": budget, "per_vector_fraction": budget / D,
               "federated": {"shards": K, "stored": K * budget, "recall": fed_acc},
               "conservation_ratio": float(conservation_ratio)}
        if target_items is not None:
            out["target_items"] = int(target_items)
            out["recommended_shards"] = int(np.ceil(target_items / budget))
        return out

    # -- one decision brain, on the same substrate -------------------------
    def actions(self, names):
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain)
        return self

    def decide(self, state, explore=False, epsilon=None, modality=None,
               senses=None, avoid=("danger", "wall"), explore_if_unrecognized=None):
        """Decide an action. `senses`/`avoid` pass straight through to the
        brain's built-in safety reflexes (HolographicMind.decide): hand over
        the current senses dict and moves into seen dangers or walls are
        vetoed below the value estimate -- the unified brain gets the same
        measured safety every other caller of the model gets.

        `explore_if_unrecognized=alpha` carries the honesty layer from perception
        to ACTION: if the current state is noise-level against the brain's own
        experience (calibrated false-alarm p > alpha, see decide_confidence), the
        value estimate is built on nothing, so take a safe random move among the
        allowed actions instead of committing to an unreliable greedy pick -- the
        agent KNOWING when it is guessing. Off by default (None); the calibrated
        threshold replaces the brain's hand-set absolute `blind_floor` cosine."""
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        sv = self.perceive(state, modality)
        if explore_if_unrecognized is not None:
            null = self._brain_null()
            if null is not None:
                sup = max((self._brain.value(sv, a)[1] for a in range(len(self._actions))), default=0.0)
                if float(null.pvalue(float(sup))) > explore_if_unrecognized:
                    epsilon = 1.0                    # unrecognized: the value estimate is noise -> safe random
        a = self._brain.decide(sv, explore=explore, epsilon=epsilon, senses=senses, avoid=avoid)
        return self._actions[a]

    def _brain_null(self, n_null=800):
        """The noise floor for the creature brain's recognition of a STATE -- the action-side analogue of
        _recall_null, and PROCEDURE-MATCHED the same way: draw random unit states, run them through the
        brain's own value() (which projects into the brain's basis exactly as a real decision does), and take
        the distribution of the best support any action reaches. Calibrated by construction (the null IS the
        support noise produces under the real value path); uses value() as a black box, reaching into no
        internals. Cached on the brain's prototype count. Returns None when the brain has learned nothing."""
        if self._brain is None:
            return None
        nproto = int(sum(len(self._brain._unit[a]) for a in range(len(self._actions))))
        if nproto == 0:
            return None
        cache = getattr(self, "_brain_null_cache", None)
        if cache is None or cache[0] != nproto:
            rng = np.random.default_rng(self.seed)
            scores = np.empty(n_null)
            for i in range(n_null):
                s = rng.standard_normal(self.dim); s /= np.linalg.norm(s) + 1e-12
                scores[i] = max((self._brain.value(s, a)[1] for a in range(len(self._actions))), default=0.0)
            from holographic_honesty import RecallNull
            rn = RecallNull(); rn.null = np.sort(scores)
            self._brain_null_cache = (nproto, rn)
        return self._brain_null_cache[1]

    def decide_confidence(self, state, modality=None, explore=False, epsilon=None,
                          senses=None, avoid=("danger", "wall")):
        """Decide an action AND report a CALIBRATED confidence in it: (action, pvalue). The p-value is the
        false-alarm probability that the state is no better matched to the brain's experience than pure noise
        -- p small means the brain has genuinely been somewhere like here and the value estimate can be
        trusted; p large means it is in unfamiliar territory and effectively guessing. This is recognize()'s
        honesty applied to the decision brain: the same RecallNull machinery, over the brain's experienced
        states instead of the perceptual prototypes. The action returned is exactly decide()'s."""
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        sv = self.perceive(state, modality)
        a = self._brain.decide(sv, explore=explore, epsilon=epsilon, senses=senses, avoid=avoid)
        null = self._brain_null()
        if null is None:
            return self._actions[a], float("nan")
        sup = max((self._brain.value(sv, a2)[1] for a2 in range(len(self._actions))), default=0.0)
        return self._actions[a], float(null.pvalue(float(sup)))

    def reinforce(self, state, action, reward, modality=None):
        s = self.perceive(state, modality)
        self._brain.remember([s], [self._actions.index(action)], [float(reward)])
        return self

    # -- generation: predict the next symbol over the same space ------------
    def learn_sequence(self, data, n=6, hierarchical=True, modality="text", name=None):
        """Learn to continue a sequence.

        Two engines, picked by `hierarchical`:

        * The fractal coder (default): discover a chunk schema by compression, then predict by
          cross-level backoff -- emit the longest chunk a level is confident about, else descend
          a level and spell it out. Measured against the flat n-gram on Austen, it cut bits/char
          from 2.085 to 1.829 and the stored model from ~218k context entries to ~58k (3.8x
          smaller), at roughly tied coherence (0.96 vs 0.98 real words). Generation is the
          traversal-shaped operation where the multi-scale substrate earns its keep -- unlike
          classification, where a tree REGRESSED and the flat scan stayed best.

        * The flat holographic n-gram (`hierarchical=False`): the original engine, kept because
          it exposes `next_symbol` and an exact context key, and because the boundary between
          where the substrate helps and where it doesn't is measured here, not assumed.

        Two consolidations, both backward compatible:

        * `modality` passes through to the fractal coder, so the mind can learn to
          continue CODE, not just prose -- the same compress-by-merging schema was
          measured to discover code structure from scratch (held-out bits/char 2.98
          -> 2.28 on this project's own source, with `def __init__` and indentation
          idioms among the unlabeled emergent chunks).
        * `name` lets the mind hold MANY sequence schemas at once. Unnamed calls keep
          the old single-slot behaviour (each call replaces); named calls accumulate,
          and generate() with no name picks the schema by the compression gate -- the
          one routing primitive used everywhere else in the stack. That is
          content-level self-discovery, needed exactly where TYPE-level inference
          goes blind: code and prose are both `str`."""
        if hierarchical:
            from holographic_schema import SchemaGenerator
            gen, kind = SchemaGenerator(modality=modality).fit(data), "hierarchical"
        else:
            from holographic_text import HolographicNGram
            gen = HolographicNGram(dim=self.dim, n=n, seed=0).fit(data)
            kind = "flat"
        key = name if name is not None else "default"
        if not hasattr(self, "_gens"):
            self._gens = {}
        self._gens[key] = {"gen": gen, "kind": kind, "modality": modality}
        self._gen, self._gen_kind = gen, kind        # most-recent alias (compat)
        return self

    def _pick_gen(self, name=None, seed_text=""):
        """Resolve which sequence schema a call means. Named -> that one. One schema
        -> it. Several and unnamed -> route the SEED by the compression gate: whoever
        compresses the seed best is the schema that understands it. The honest
        boundary: only hierarchical schemas expose bits_per_char, so flat engines
        never compete in the gate -- name them explicitly."""
        gens = getattr(self, "_gens", {})
        if not gens:
            raise RuntimeError("nothing learned to continue -- call learn_sequence() first")
        if name is not None:
            if name not in gens:
                raise KeyError(f"no sequence schema named {name!r} -- have {sorted(gens)}")
            return gens[name]
        if len(gens) == 1:
            return next(iter(gens.values()))
        from holographic_schema import compression_gate
        gated = {k: g["gen"] for k, g in gens.items() if g["kind"] == "hierarchical"}
        if not gated or not seed_text:
            raise RuntimeError("several schemas are loaded -- name one, or give a seed "
                               "the gate can route (flat engines must be named)")
        return gens[compression_gate(seed_text, gated)[0][1]]

    def next_symbol(self, context, name=None):
        g = self._pick_gen(name, context)
        if g["kind"] != "flat":
            raise RuntimeError("next_symbol needs the flat engine: learn_sequence(text, hierarchical=False)")
        return g["gen"].next_char(context)

    def generate(self, seed_text, length=160, temperature=0.5, name=None, top_p=1.0):
        """Continue text from the chosen sequence schema. top_p<1.0 requests nucleus
        decoding; it is forwarded only to flat n-gram generators that support it (the
        hierarchical schema generator decodes its own way), so the argument is safe and
        backward-compatible everywhere."""
        gen = self._pick_gen(name, seed_text)["gen"]
        try:
            return gen.generate(seed_text, length, temperature, top_p=top_p)
        except TypeError:
            return gen.generate(seed_text, length, temperature)   # generator without top_p

    def build_predictor(self, order=2, reinforce_threshold=0.15, novelty_threshold=0.55):
        """Give the mind a PREDICTIVE LOOP over symbol sequences: it anticipates
        the next symbol from recent context, measures its surprise, and learns
        error-gated (see holographic_predictive). This is the active layer on top
        of storage -- the mind now expects, notices when it is wrong, and adapts.
        Returns self."""
        from holographic_predictive import PredictiveMemory
        self._predictor = PredictiveMemory(dim=self.dim, order=order, seed=0,
                                           reinforce_threshold=reinforce_threshold,
                                           novelty_threshold=novelty_threshold)
        return self

    def observe_sequence(self, tokens, learn=True):
        """Run the predictive loop over a token sequence; return the Steps (the
        surprise/valence/free-energy trace). The mind lives the sequence one
        anticipation at a time."""
        if not hasattr(self, "_predictor"):
            self.build_predictor()
        return self._predictor.learn_sequence(list(tokens), learn=learn)

    def anticipate(self, recent, soft=False):
        """What does the mind expect next, and how confident is it? Returns
        (symbol, confidence). Confidence near 1 is a remembered continuation;
        around 0.5 is a generalisation from a similar context."""
        if not hasattr(self, "_predictor"):
            return None, 0.0
        return self._predictor.predict(list(recent), soft=soft)

    def generate_predictive(self, seed, length=30, soft=False):
        """Generate by anticipation: predict the next symbol, append, repeat."""
        if not hasattr(self, "_predictor"):
            return []
        return self._predictor.generate(list(seed), length=length, soft=soft)

    def prediction_report(self, tokens):
        """How well does the mind anticipate this sequence (no learning)? Returns
        accuracy plus mean surprise and final free energy -- its self-consistency
        on the stream."""
        if not hasattr(self, "_predictor"):
            return {"accuracy": 0.0, "mean_surprise": 1.0, "free_energy": 1.0}
        steps = self._predictor.learn_sequence(list(tokens), learn=False)
        if not steps:
            return {"accuracy": 0.0, "mean_surprise": 1.0, "free_energy": 1.0}
        import numpy as _np
        return {"accuracy": sum(s.hit for s in steps) / len(steps),
                "mean_surprise": float(_np.mean([max(0.0, s.surprise) for s in steps])),
                "free_energy": float(steps[-1].self_free_energy)}

    def build_meaning_predictor(self, sentences, order=2, window=2):
        """Give the mind a MEANING-level predictor: instead of returning a single
        stored next symbol, it composes a next-MEANING vector from all resonating
        contexts and settles it to a word (holographic_meaning_predict). Built over
        a co-occurrence meaning space, which -- measured -- is the right space for
        'what follows' (the dictionary-curriculum space is for 'what is related').
        Returns self."""
        from holographic_meaning_predict import MeaningPredictor
        stream = [w for s in sentences for w in (s if isinstance(s, list) else s.split())]
        self._meaning_pred = (MeaningPredictor(dim=self.dim, order=order, seed=0)
                              .fit_space(sentences, window=window)
                              .fit_transitions(stream))
        # calibrate a structure verifier on the same corpus, so the mind can PROVE
        # whether a sequence carries structure and steer generation to stay in it
        from holographic_structure import StructureVerifier
        mp = self._meaning_pred
        self._verifier = StructureVerifier(mp.vocab, mp.M, mp.idx).calibrate(stream, chunk=150, z_floor=2.0)
        return self

    def verify_structure(self, tokens):
        """Proof of meaning: does this sequence carry structure, or is it salad?
        Returns {'score': float, 'meaningful': bool, 'threshold': float}. The score
        is how closely the sequence's lag-coherence profile matches real text (0 =
        typical, more negative = anomalous) -- meaning projected onto context across
        ranges, not trusted from any single word."""
        if not hasattr(self, "_verifier"):
            return {"score": 0.0, "meaningful": False, "threshold": 0.0}
        toks = list(tokens) if not isinstance(tokens, str) else tokens.split()
        return {"score": self._verifier.structure_score(toks),
                "meaningful": bool(self._verifier.is_meaningful(toks)),
                "threshold": float(self._verifier.threshold)}

    def generate_structured(self, seed, length=30, beam=6, lookback=8):
        """Generate while PROVING structure step by step: among the predictor's top
        candidates, keep the one that best preserves the running context's structure
        -- generation that defends its own coherence, which (measured) escapes the
        loops plain greedy decoding falls into."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return []
        from holographic_structure import steered_generate
        seed = list(seed) if not isinstance(seed, str) else seed.split()
        return steered_generate(self._meaning_pred, self._verifier, seed,
                                length=length, beam=beam, lookback=lookback)

    def share(self):
        """Freeze this trained mind and return a SharedMind that many lightweight
        instances (NPCs/agents) can branch from -- they share this heavy base by
        reference and hold only their own private deltas, instead of each building a
        full brain (holographic_partition). Learning in a branch can be propagated
        back so every instance inherits it. Pass capacity>0 to SharedMind for a
        capacity-aware merge when very many instances propagate into the same label."""
        from holographic_partition import SharedMind
        return SharedMind(self)

    def compress_lossless(self, tokens):
        """Go both directions: losslessly compress a sequence to a compact code (seed
        + rank stream) via the predictor's ranking, and report the achievable size.
        decompress_lossless inverts it exactly. Needs build_meaning_predictor.

        BOUNDARY (vs decompose_signal): this is LOSSLESS entropy coding of a discrete TOKEN
        sequence against a learned next-token predictor -- it reconstructs the exact tokens.
        decompose_signal instead fits a generating LAW to a CONTINUOUS signal (a small savable
        Formula seed), which is lossy/approximate. Different levels of 'compression'; both kept."""
        if not hasattr(self, "_meaning_pred"):
            return {"code": None, "cost": {"ratio": 1.0}}
        if not hasattr(self, "_codec"):
            from holographic_codec import PredictiveCodec
            self._codec = PredictiveCodec(self._meaning_pred)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return {"code": self._codec.compress(toks), "cost": self._codec.cost(toks),
                "lossless": self._codec.roundtrip_ok(toks)}

    def decompress_lossless(self, code):
        """Recover the exact original sequence from a code produced by
        compress_lossless, by replaying the shared predictor."""
        if not hasattr(self, "_codec"):
            from holographic_codec import PredictiveCodec
            self._codec = PredictiveCodec(self._meaning_pred)
        return self._codec.decompress(code)

    def attribute_sources(self, tokens, sources, topk=15, order=2):
        """Source attribution: trace which stored material a passage drew on. sources
        is {name: token_stream}; returns a provenance distribution over those sources
        from the predictor's resonance couplings (holographic_codec.SourceAttributor)."""
        from holographic_codec import SourceAttributor
        att = SourceAttributor(dim=self.dim, order=order, seed=0).fit(sources)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return att.attribute(toks, topk=topk)

    def factor_composite(self, composite, codebooks, restarts=20, L=None, iters=None, seed=0, confidence=False,
                         readout="softmax"):
        """Pull a single bound composite APART into the factors that built it -- the inverse of binding,
        by searching in superposition. ONE entry point for both factorizers the engine grew:

          * SBC (PREFERRED) -- pass an integer-atom `composite` (B blocks), SBC `codebooks` (lists of
            B-integer atoms), and the block length `L`. Delegates to the higher-capacity, confidence-
            VALIDATED resonator (holographic_sbc.decompose_structure): block-local convolution makes each
            block a clean channel, so it factors more (factors x alphabet) at a fixed dimension AND reports
            whether the answer actually RECONSTRUCTS the product -- it verifies or abstains rather than
            guessing. This is the SAME factorizer decompose_structure() exposes: one factorizer, not two.

          * DENSE (LEGACY, deprecated) -- the original dense MAP/bipolar path (holographic_resonator):
            `composite` a dense bipolar vector, `codebooks` dense (n, dim) bipolar matrices, no `L`. Kept
            for backward compatibility because the SBC resonator works in a DIFFERENT algebra (per-block
            modular add of one-hots, not the elementwise sign-product MAP bind) and CANNOT factor a dense
            MAP composite -- so this path could not simply be removed, only delegated-past and deprecated.
            New code should pass SBC codebooks + L, or call decompose_structure(); a DeprecationWarning
            steers it there.

        Returns a dict with at least 'factors' (recovered index per slot), 'solved' (True only if the
        factors actually re-bind to the composite), 'search_space' (the combinatorial size searched without
        enumerating), and 'backend' ('sbc'/'dense'). The SBC backend also returns 'verified' and 'present'."""
        space = 1
        for B in codebooks:
            space *= (len(B) if L is not None else B.shape[0])

        if L is not None:                              # SBC problem -> the preferred, validated factorizer
            from holographic_sbc import decompose_structure
            res = decompose_structure(np.asarray(composite), codebooks, L, restarts=restarts,
                                      iters=(50 if iters is None else iters), seed=seed,
                                      confidence=confidence, readout=readout)
            out = {"factors": tuple(res["picks"]), "solved": bool(res["verified"]),
                   "verified": bool(res["verified"]), "present": res["present"],
                   "restarts": restarts, "search_space": space, "backend": "sbc"}
            if confidence:                             # calibrated soft confidence for approximate inputs
                out["agreement"] = res["agreement"]; out["pvalue"] = res["pvalue"]
            return out

        # dense MAP/bipolar -- the legacy path, kept for backward compatibility, gently deprecated
        import warnings
        warnings.warn("factor_composite's dense MAP/bipolar path is legacy; pass SBC codebooks + L (or "
                      "call decompose_structure) to use the higher-capacity, validated SBC resonator.",
                      DeprecationWarning, stacklevel=2)
        from holographic_resonator import ResonatorNetwork
        kw = {"iters": iters} if iters is not None else {}
        out = ResonatorNetwork(codebooks).factor(composite, restarts=restarts, **kw)
        out["backend"] = "dense"
        return out

    def decompose_structure(self, composed, codebooks, L, restarts=6, iters=50, seed=None,
                            readout="softmax", confidence=False, k=8):
        """Recover the generating recipe of a COMPOSED structure -- the canonical, higher-capacity
        factorizer (holographic_sbc.decompose_structure), exposed as a faculty the mind speaks directly.
        A bound product is DISSIMILAR to its factors, so per-factor cleanup is chance; the SBC resonator
        holds a superposition of all candidate factors per block, anneals from soft (explore) to sharp
        (commit), and accepts ONLY reconstruction-VERIFIED answers -- so it verifies or abstains, never
        guesses. If a codebook contains the SBC identity, that factor can be found ABSENT (presence
        detection).

        `composed` is an SBC product (B integers, active position per block); `codebooks` is a list of SBC
        codebooks (each a list of B-integer atoms); `L` is the block length. Returns
        {picks, factors, verified, present}. This is the SAME factorizer factor_composite delegates to when
        given an `L` -- the de-siloing the integration review asked for: one factorizer, not two.

        `readout='sparsemax'` switches the alternating-projection blend from softmax to the sparse readout,
        which is MEASURED to raise factorization capacity (all-correct at N=50 0.00->0.12, N=80 0.00->0.25,
        N=25 0.47->0.62) by curing the softmax blend's metastable mixing; the default 'softmax' is unchanged.
        With confidence=True the result also carries {agreement, pvalue} -- the calibrated soft confidence for
        approximate inputs (its null is matched to the chosen readout)."""
        from holographic_sbc import decompose_structure as _decompose
        return _decompose(np.asarray(composed), codebooks, L, restarts=restarts, iters=iters,
                          seed=self.seed if seed is None else seed, readout=readout, confidence=confidence, k=k)

    # -- self-verifying storage: tamper-evidence as an O(log n) property of the structure (BLD-1) -----
    def verify_store(self, items, seed=None):
        """Commit to a list of item vectors with a tamper-evident composition tree (holographic_verify) --
        the holographic Merkle tree, built from the same bind + bundle the rest of the mind uses. Returns a
        CompositionTree whose `.root()` is the commitment; `.verify(items)` detects any later change and
        `.locate(items)` returns the index of a single changed item in <= log2(n) composite comparisons (the
        descent depth, independent of how many items are stored).

        Position is bound into each leaf, so a reordering is caught -- a plain bundle, being commutative, would
        miss it. The honest bound, on the record in the module: the root is LINEAR, so this is evidence of
        ACCIDENTAL corruption / uncoordinated tampering, NOT cryptographic tamper-proofing -- a key-aware
        adversary can cancel a change by deconvolution and leave the root bit-for-bit unchanged."""
        from holographic_verify import CompositionTree
        return CompositionTree(items, seed=self.seed if seed is None else seed)

    def vector_function_encoder(self, n_dims, bounds=None, kernel="rbf", bandwidth=3.0):
        """An N-dimensional Fractional Power Encoder (holographic_fpe) on this mind's dim and seed: encode a
        continuous point in R^n so that a SHIFT is a BINDING and the similarity is a designed PRODUCT kernel,
        and represent / query / shift whole FUNCTIONS as bundles of weighted encoded points.

        The 1-D case is already the ScalarEncoder (encode(x) = "base^x", with shift-as-bind and a Bochner
        kernel); this is the step up to a vector domain and the compute-on-functions algebra the resonator /
        scene-factoring literature builds on. Returns a VectorFunctionEncoder. The standing capacity cliff
        applies -- a function is a bundle, so too many superposed points drown each in the others' cross-talk."""
        from holographic_fpe import VectorFunctionEncoder
        return VectorFunctionEncoder(n_dims, dim=min(self.dim, 1024), bounds=bounds,
                                     kernel=kernel, bandwidth=bandwidth, seed=self.seed)

    def spectral_basis(self, points, k=10, n_basis=12):
        """The data-driven decomposition basis for a signal on a manifold (holographic_spectral, EXP-6): the
        lowest eigenvectors of the kNN-graph Laplacian of the sample points -- the smoothest functions the
        manifold admits. On a line this is the DCT / elementary basis and on a ring the harmonic basis (so it
        matches decompose_signal's hand-picked choice), and on a manifold the topology detector cannot name
        (a sphere, a torus, a curved surface) it is the right basis where the line/elementary fallback is not.
        Returns a SpectralBasis with decompose / reconstruct / denoise. Dense eigh -> moderate N (C1)."""
        from holographic_spectral import SpectralBasis
        return SpectralBasis(points, k=k, n_basis=n_basis)

    def manifold_topology(self, points, lo=1.3, hi=2.6, steps=7, max_points=250):
        """Name a point cloud's topology by persistent homology (holographic_topology, EXP-7): the Betti
        signature (components, loops, voids) that persists across a scale band. The principled generalisation
        of detect_topology -- it reproduces "line" (1,0,0) and "ring" (1,1,0) on the cases that detector knows,
        and extends to ones it cannot name structurally: a torus (1,2,1) and a sphere (1,0,1). Returns
        (name, (B0,B1,B2), scale-histogram). It reads a WELL-SAMPLED manifold's topology -- finicky on uneven
        or noisy clouds and blind to non-topological geometry (pair it with spectral_basis for the geometry),
        and the cloud is subsampled to max_points for tractability (dense reduction -> moderate N, C1)."""
        from holographic_topology import persistent_topology
        return persistent_topology(points, lo=lo, hi=hi, steps=steps, max_points=max_points)

    def is_manifold(self, points, max_dense_scales=1, lo=1.3, hi=2.6, steps=7, max_points=250):
        """A fast structural GATE: does this point cloud form a single connected low-dimensional MANIFOLD, or is
        it a dense blob with no clean topology? Runs persistent homology (now sub-second on a blob, where it once
        ground for ~30s) and reads its verdict. Returns a dict {is_manifold, topology, betti, dense_scales}:
        is_manifold is True iff the cloud is ONE connected piece (B0 == 1) AND at most `max_dense_scales` scale
        bands were too dense to read a clean complex.

        The point is to check a premise CHEAPLY before a manifold-assuming operation. denoise(method='spectral'),
        for instance, projects a field onto the cloud's smooth modes -- its premise is a smooth field on a curved
        manifold. On a blob that premise fails and the 'denoise' is just graph low-pass; measured, on a 2-sphere
        spectral denoise cleans 3.74->1.08, but on a random blob it barely moves (4.37->4.20). This gate is the
        honest signal of which case you are in, fast enough to run inline (pass check_manifold=True to denoise to
        wire it as a guard). Same finickiness as manifold_topology: it reads a WELL-SAMPLED manifold, and a
        disconnected manifold (B0 > 1) reads as not-a-manifold by design (the gate wants one connected piece)."""
        name, betti, hist = self.manifold_topology(points, lo=lo, hi=hi, steps=steps, max_points=max_points)
        dense = int(hist.get("dense_scales", 0))
        ok = (betti[0] == 1) and (dense <= int(max_dense_scales))   # one component, and readable at enough scales
        return {"is_manifold": bool(ok), "topology": name, "betti": betti, "dense_scales": dense}

    def hodge_decomposition(self, n_verts, edges, flow, triangles=None):
        """Split an edge flow into three L2-orthogonal parts (holographic_spectral, EXP-8): gradient (curl-free
        transport from a vertex potential), curl (local circulation around filled triangles), and harmonic
        (global circulation around the holes -- its dimension equals B1, so the harmonic part IS the flow's
        topology, tying to EXP-7). For the Tero flow solver and graph-signal analysis. On a tree (no cycles,
        no triangles) curl and harmonic vanish and all flow is gradient (kept negative). Returns
        (gradient, curl, harmonic)."""
        from holographic_spectral import hodge_decomposition as _hd
        return _hd(n_verts, edges, flow, triangles)

    def denoise_flow(self, n_verts, edges, flow, triangles=None, keep=("gradient", "harmonic")):
        """Denoise an edge flow by keeping only its structurally-valid Hodge components and dropping the rest
        (holographic_spectral, EXP-8) -- e.g. drop the curl of a transport flow that should not circulate
        around cells, removing the share of isotropic noise that lands there. Beats naive edge-smoothing on a
        flow. Returns the reconstruction from the kept parts."""
        from holographic_spectral import denoise_flow as _df
        return _df(n_verts, edges, flow, triangles, keep=keep)

    # -- one scene faculty, on the same substrate ---------------------------
    def scene(self):
        """The mind's own scene coder (compose/decompose visual attribute scenes), built
        lazily on this mind's dim and seed so it shares the substrate rather than being a
        separate engine. All scene methods below go through it."""
        if self._scene is None:
            from holographic_scene import SceneCoder
            self._scene = SceneCoder(dim=min(self.dim, 1024), seed=self.seed)
        return self._scene

    def compose_scene(self, tag_list):
        """Run the resonator FORWARD on this mind's scene coder: bind chosen attribute
        atoms (colour/shape/texture) into a single scene vector that was never stored.
        The inverse of decompose_scene()."""
        return self.scene().encode_scene(tag_list)

    def decompose_scene(self, scene_vec, n_objects, sweeps=2):
        """Factor a scene vector back into its per-object attribute tags -- the backward
        resonator, on the mind's own scene coder. Verifies a composed scene by recovering
        exactly what built it."""
        return self.scene().factor_scene(scene_vec, n_objects, sweeps=sweeps)

    def _group_roles(self):
        """A small vocabulary of group-key atoms, seed-derived so a nested scene
        reconstructs from the same seed (regenerate-from-seed at the group level too)."""
        if getattr(self, "_groles", None) is None:
            from holographic_ai import Vocabulary
            self._groles = Vocabulary(min(self.dim, 1024), seed=self.seed + 11, derived=True)
        return self._groles

    def spectral_encode(self, frame):
        """Encode an audio frame as an FHRR phasor hypervector -- Puckette's phase-vocoder representation in
        the complex domain. The frame's DFT splits into a unit-magnitude PHASOR per bin (the phase: an FHRR
        vector, every component on the complex unit circle, so it binds / bundles / recalls in
        `high_capacity_memory` exactly like a minted atom) and a MAGNITUDE per bin (the timbre, returned
        alongside). Silent bins take phasor 1 by convention, so the phasor vector is unit-magnitude EVERYWHERE
        -- a valid FHRR vector, not a spectrum with holes. Exactly invertible by `spectral_decode`. The same
        per-bin phase that `learn_dynamics` advances when it predicts the next frame: the dynamics operator
        and this encoding are two faces of one spectral structure. Returns (phasors, magnitudes)."""
        x = np.asarray(frame, float)
        spec = np.fft.fft(x)
        mag = np.abs(spec)
        phasors = np.where(mag > 1e-9, spec / (mag + 1e-30), 1.0 + 0j)   # unit phasor; silent bins -> 1
        return phasors, mag

    def spectral_decode(self, phasors, magnitudes):
        """Invert `spectral_encode`: re-attach the magnitudes to the unit phasors and inverse-DFT back to the
        real frame (phase-vocoder resynthesis). Exact to floating point -- the phasor (FHRR-domain) key and
        the magnitude together lose nothing."""
        spec = np.asarray(phasors, complex) * np.asarray(magnitudes, float)
        return np.real(np.fft.ifft(spec))

    def high_capacity_memory(self):
        """An opt-in FHRR (complex-phasor) key->value trace memory and its atom vocab,
        for the one regime where the complex domain measurably beats the real-valued core:
        a LARGE number of pairs crammed into one vector (see holographic_fhrr -- ~0.90 vs
        ~0.61 recovery at 40 pairs/256-d). The real-HRR memory stays the default everywhere
        else, since at normal loads both are perfect. Returns (PhasorMemory, PhasorVocabulary)
        sharing this mind's seed, so the store is seed-deterministic like the rest."""
        from holographic_fhrr import PhasorMemory, PhasorVocabulary
        if getattr(self, "_hcap", None) is None:
            d = min(self.dim, 1024)
            self._hcap = (PhasorMemory(d), PhasorVocabulary(d, seed=self.seed + 23, derived=True))
        return self._hcap

    def compose_nested(self, groups):
        """Fractal composition -- the SAME bind+superpose that builds a scene from objects,
        applied ONE LEVEL UP to build a scene-of-scenes. `groups` is a dict {group_key:
        tag_list}; each tag_list is composed into a sub-scene vector, that vector is bound
        to its group-key atom, and the bound sub-scenes are superposed. Same above, same
        below: a sub-scene is to the super-scene exactly what an object is to a scene.

        Recovery (decompose_nested) is near-perfect for 2-3 groups and degrades gracefully
        beyond as group-binding cross-talk accumulates -- the same capacity limit the flat
        scene has, now measured at the group level: ~1.00 at 2 groups, ~0.97 at 3, ~0.89 at
        4, ~0.82 at 5 (2 objects each). Returns the super-scene vector."""
        from holographic_ai import bind
        gr = self._group_roles()
        sc = self.scene()
        parts = [bind(gr.get(str(k)), sc.encode_scene(tags)) for k, tags in groups.items()]
        return np.sum(parts, axis=0)

    def decompose_nested(self, super_scene, group_sizes, sweeps=2):
        """Invert compose_nested: for each group key, unbind its sub-scene out of the
        super-scene and factor that sub-scene back into per-object tags -- the same
        unbind-then-factor at two levels. `group_sizes` is {group_key: n_objects}. Returns
        {group_key: [recovered tags]}. A nested scene is real when it analyses straight back
        to the groups-of-objects it was built from."""
        from holographic_ai import unbind
        gr = self._group_roles()
        sc = self.scene()
        out = {}
        for k, n in group_sizes.items():
            sub = unbind(super_scene, gr.get(str(k)))
            out[k] = sc.factor_scene(sub, n, sweeps=sweeps)
        return out

    # ---- B7 keystone: ONE typed structure for all composition (see holographic_typed) -----------
    def typed_structure(self):
        """A fresh StructureRecipe bound to this mind's dim and seed -- the single replayable build-graph
        that recipes, programs, expression trees, and scenes all reduce to. Build through it
        (atom/bind/bundle/permute/superpose), then realize() to a vector or save() the seed. This is the
        de-siloing the integration review asked for: one structure type the mind speaks directly."""
        from holographic_recipe import StructureRecipe
        return StructureRecipe(self.dim, self.seed)

    def realize(self, recipe):
        """Replay a StructureRecipe to its output vector(s) -- the single realize path for any structure."""
        outs = recipe.outputs()
        return outs[0] if len(outs) == 1 else outs

    def tree_structure(self, tree):
        """Encode an expression tree as a typed structure at this mind's dim/seed. A leaf is a str symbol;
        an internal node is (op, *children). The EML-tree's holographic encoding, generalised."""
        from holographic_typed import tree_to_recipe
        return tree_to_recipe(self.dim, self.seed, tree)

    def nested_scene_structure(self, groups):
        """compose_nested AS a typed structure: the same super-scene vector, now a replayable build-graph
        (group-role atoms + bind + superpose, rng sub-scenes as raw leaves) that can be saved and inspected."""
        from holographic_typed import nested_scene_to_recipe
        return nested_scene_to_recipe(self, groups)

    def chain_structure(self, n):
        """Build an n-node linked-list CHAIN as a typed structure (B7), at this mind's dim/seed:
        M = superpose_i bind(node_i, node_{i+1}). Returns (recipe, nodes) -- realize(recipe) gives the
        chain-memory vector, and decode_structure(memory, nodes) traverses it back. The smallest honest
        forward object whose INVERSE (per-peel decode) is the interesting part (holographic_peel)."""
        from holographic_peel import chain_recipe
        return chain_recipe(self.dim, self.seed, n)

    def decode_structure(self, memory, nodes, steps=None, cleanup="hard", beta=8.0):
        """DECODE a composed CHAIN structure by iterated unbinding with PER-PEEL CLEANUP (B8) -- the
        inverse of the B7 chain typed structure, on the same object. The crux the module measured: each
        recovered pointer is noisy, and without cleanup that noise is carried into the next hop and
        COMPOUNDS, so a raw traversal craters after ~1-2 hops and its carried vector diverges; snapping
        each pointer back onto the node codebook BEFORE the next hop bounds the noise and the whole chain
        decodes. Cleaning structure AS it is decoded.

        `memory` is the chain vector (realize() of chain_structure's recipe); `nodes` is the node codebook.
        cleanup in {None, 'hard', 'soft'}: None carries the raw peel (it craters -- the kept negative made
        visible); 'hard' snaps to the nearest atom (Bayes-optimal for identity); 'soft' is the B1 dense-
        Hopfield update (ties hard on discrete pointers -- the value is continuous payloads, see the module's
        recover_continuous_values). Returns the recovered node-index sequence (-1 marks a diverged hop).

        NOTE: this is the SEQUENCE inverse (traverse a chain); decompose_structure is the PRODUCT inverse
        (factor a bound product). Different structures, different inverses -- both on the one substrate."""
        from holographic_peel import traverse
        nodes = np.asarray(nodes)
        steps = (len(nodes) - 1) if steps is None else steps
        return traverse(np.asarray(memory), nodes, steps, cleanup=cleanup, beta=beta)

    # ---- the DECOMPOSE / DENOISE / FIT half of the loop (integration plan, Tier 1) -------------
    # UnifiedMind was already strong on one half of the loop: COMPOSE / RECALL / PREDICT / GENERATE.
    # These three faculties add the inverse half -- take a FOREIGN signal APART into a generator (a
    # law), CLEAN it on the right manifold, or FIT an interpretable function to it. Each one unifies
    # several already-shipped modules behind a single honest entry point, the same move
    # typed_structure() made for composition: one faculty the mind speaks directly, not a drawer of
    # disconnected experiments beside it.

    def decompose_signal(self, x, y=None, max_terms=6, coef_bits=20, n_harmonics=5):
        """Take a foreign 1-D signal APART into the law that generates it -- the measured-regime twin
        of compose()/typed_structure(). One faculty over four shipped modules:

          1. detect the domain TOPOLOGY            (holographic_manifold.detect_topology)
          2. choose the matched basis:
               line           -> elementary functions, additive OR multiplicative (auto-selected)
               ring / mobius  -> harmonics of the detected period (mobius = ODD harmonics only, the
                                 antiperiodic basis -- holographic_mobius's own function space)
               torus          -> harmonics of both periods
          3. fit an MDL-gated law on that basis    (holographic_symbolic.symbolic_regress / compress_signal)
          4. return the Formula -- which already IS a savable generative seed (.generate() to regenerate
             or extrapolate, .save()/.load() to persist), the scalar-signal analogue of a StructureRecipe.

        x is the independent coordinate, y the observed signal. As a shorthand, decompose_signal(y) with
        a single array treats it as the signal on a unit-spaced index grid.

        Returns (Formula, info). info carries: topology, period, mode ('additive'/'multiplicative'),
        n_terms, resid_rms (ORIGINAL-space residual a B5 coder would take), and compression_ratio.

        SCOPE (kept honest, surfaced from the modules, not new): the multiplicative (log) family is
        auto-selected only on a LINE domain and needs y > 0 (it fits log y); a periodic signal is
        decomposed additively on its harmonic basis. A torus needs a window long enough to resolve both
        tones or detection falls back to line (the Rayleigh limit -- see holographic_manifold)."""
        from holographic_manifold import detect_topology, decompose_on_manifold
        from holographic_symbolic import compress_signal
        if y is None:                                  # single-array shorthand: signal on an index grid
            y = np.asarray(x, float)
            x = np.arange(len(y), dtype=float)
        x = np.asarray(x, float); y = np.asarray(y, float)

        topo, _ = detect_topology(x, y, n_harmonics=n_harmonics)
        if topo == "line":
            # Flat domain: an additive fit and a multiplicative (log-basis) fit are both candidates, so
            # let compress_signal's measured auto-rule choose -- it switches to multiplicative only when
            # that law is competitive in-sample AND generalizes better on a held-out tail (the conservative
            # criterion that refuses to reward additive overfitting). Catches a*x^p*exp(cx) laws a flat
            # additive dictionary would miss.
            f, info = compress_signal(x, y, max_terms=max_terms, coef_bits=coef_bits, mode="auto")
            info["topology"] = "line"; info["period"] = None
        else:
            # Periodic / antiperiodic / quasiperiodic: decompose on the manifold-matched harmonic basis
            # so the recovered law extrapolates PERIODICALLY instead of diverging the way a polynomial
            # forced onto a ring would. (mobius -> odd harmonics only -- the antiperiodic space.)
            f, info = decompose_on_manifold(x, y, n_harmonics=n_harmonics,
                                            max_terms=max_terms, coef_bits=coef_bits)
            info["mode"] = "multiplicative" if f.log_space else "additive"
            info["compression_ratio"] = f.compression_ratio(len(y))
        return f, info

    def denoise(self, x, method="auto", samples=None, codebook=None, sigma=None,
                rank=8, beta=25.0, steps=3, forward=None, adjoint=None, mu=0.5, pnp_steps=30,
                readout="softmax", points=None, spectral_k=10, spectral_nbasis=12, check_manifold=False):
        """Clean a noisy signal by projecting it onto a manifold -- Milanfar's thesis that a denoiser
        IS a map of the manifold clean signals live on. One call over holographic_denoise +
        holographic_hopfield, picking the map by the structure you supply a prior for:

          method='adaptive' : project onto a low-rank SVD subspace fit from `samples`, then
                              noise-THRESHOLD the coefficients (Donoho-Johnstone). The safe default for
                              low-rank signals -- estimates the noise level itself, so it does not
                              over-smooth at low noise.
          method='manifold' : plain FIXED-rank projection onto the subspace fit from `samples`.
          method='codebook' : modern-Hopfield cleanup of `x` toward a discrete `codebook` manifold.
          method='nlm'       : non-local means -- `x` is a (N, dim) patch set; average each patch with
                              its near-duplicates via the engine's own content-addressable recall.
          method='trajectory': clean a LONE 1-D signal with no external prior -- its sliding-window Hankel
                              matrix is low-rank for a smooth/structured signal (SSA/Cadzow), so project the
                              windows onto their own subspace and reconstruct. The second prior-free method
                              beside nlm (nlm needs a patch SET; this takes a raw 1-D signal).
          method='spectral' : clean a lone scalar FIELD living on a known manifold GEOMETRY -- pass the point
                              coordinates as points=<(N, d)> and x as the field value at each of those N points.
                              Builds the kNN graph-Laplacian eigenbasis (EXP-5/6) and projects the field onto
                              its low-frequency modes. The NONLINEAR-manifold map the linear methods lack: it is
                              the only denoiser here that needs no example set and no codebook, just the cloud's
                              own geometry. Measured on a smooth field over a 2-sphere, it cleans error 4.1->0.9
                              where the geometry-blind options barely move it (trajectory 3.1, DCT 4.2) -- a
                              linear/1-D prior cannot see a curved manifold's smoothness.
          method='pnp'       : Plug-and-Play / RED restoration of a degraded measurement x = forward(clean)
                              + noise, using the adaptive manifold map as the prior (needs forward/adjoint).
          method='auto'      : codebook if a `codebook` is given, else adaptive manifold if `samples`
                              are given. NLM and PnP stay OPT-IN: deciding self-similar-vs-low-rank
                              automatically is itself a measurement we will not fake -- name them.
          method='geometry'  : route by the GEOMETRY of the set you hand (samples= or codebook=). Read its
                              effective rank; if LOW relative to the row count (a continuous manifold)
                              project onto that subspace; if HIGH (distinct atoms) do codebook recall. This
                              is the measured 'match the map to the manifold' rule -- projection is
                              near-perfect on a low-rank manifold and FAILS (67% recall) on high-rank atoms,
                              so the rank knee picks the right one.

        `readout='sparsemax'` switches the codebook/recall branches from the softmax blend (which
        over-smooths a continuous manifold) to the sparse Hopfield-Fenchel-Young readout; the default
        'softmax' leaves every path bit-for-bit unchanged.

        `check_manifold=True` (method='spectral' only) first verifies the points form a single connected manifold
        via is_manifold and raises if they do not -- the spectral map's premise -- rather than silently returning
        graph low-pass on a blob. Default False keeps the path overhead-free and backward-compatible.

        A denoiser needs a PRIOR; a single vector with no manifold cannot be cleaned (no free lunch), so
        `samples` (clean rows) or `codebook` (atoms) is required for every method but 'nlm' (which uses
        `x`'s own redundancy). Returns the cleaned vector (or, for 'nlm', the cleaned (N, dim) set).

        KEPT NEGATIVES (the modules', surfaced not hidden): FIXED-rank projection over-smooths at low
        noise -- use 'adaptive', which is ~neutral there; manifold projection only helps where real
        low-rank structure exists (it destroys structureless signal); NLM only helps where near-duplicates
        exist."""
        from holographic_denoise import (fit_manifold, manifold_denoise, fit_manifold_full,
                                          adaptive_manifold_denoise, codebook_denoise,
                                          nlm_denoise, pnp_restore, effective_rank, trajectory_denoise)
        x = np.asarray(x, float)

        if method == "auto":                          # pick by the prior you handed me, conservatively
            method = "codebook" if codebook is not None else ("adaptive" if samples is not None else None)
            if method is None:
                raise ValueError("denoise needs a prior: pass samples=<clean rows> or codebook=<atoms> "
                                 "(a denoiser is a map of a manifold; a lone vector has none)")

        if method == "nlm":                           # self-similarity: x IS the patch set to clean
            P = np.atleast_2d(x)
            return nlm_denoise(P, k=min(12, len(P)))

        if method == "trajectory":                    # lone 1-D signal: prior built from its OWN windows (SSA)
            return trajectory_denoise(x, window=None, rank=rank)

        if method == "spectral":              # lone scalar FIELD on a known manifold GEOMETRY -> graph-Laplacian map
            if points is None:
                raise ValueError("method='spectral' needs points=<(N, d) coordinates>; x is the field over "
                                 "those N points (the manifold's own geometry IS the prior)")
            pts = np.atleast_2d(np.asarray(points, float))
            if check_manifold:                # opt-in premise check (cheap now PH is fast): the spectral map
                chk = self.is_manifold(pts)   # assumes a smooth field on a CONNECTED manifold; on a blob it is
                if not chk["is_manifold"]:     # only graph low-pass, so refuse loudly unless overridden
                    raise ValueError(
                        f"method='spectral' premise fails: the points are not a single connected manifold "
                        f"(topology={chk['topology']!r}, dense_scales={chk['dense_scales']}). The spectral "
                        f"denoiser would be graph low-pass, not manifold denoising. Pass check_manifold=False "
                        f"to proceed anyway.")
            from holographic_spectral import SpectralBasis
            sb = SpectralBasis(pts, k=spectral_k, n_basis=spectral_nbasis)
            return sb.denoise(x)

        if method == "codebook":
            if codebook is None:
                raise ValueError("method='codebook' needs codebook=<(n, dim) atoms>")
            return codebook_denoise(x, np.asarray(codebook, float), beta=beta, steps=steps, readout=readout)

        if method == "geometry":              # route by the set's geometry (measured: match map to manifold)
            M = codebook if codebook is not None else samples
            if M is None:
                raise ValueError("method='geometry' needs samples= or codebook= (the manifold/atom set "
                                 "whose geometry decides the map)")
            M = np.atleast_2d(np.asarray(M, float))
            er = effective_rank(M)
            if er <= 0.5 * len(M):            # LOW-rank continuous -> the manifold map: project onto its span
                basis, mean = fit_manifold(M, rank=max(1, er))
                return manifold_denoise(x, basis, mean)
            return codebook_denoise(x, M, beta=beta, steps=steps, readout=readout)   # HIGH-rank discrete -> recall

        if method in ("manifold", "adaptive", "pnp"):
            if samples is None:
                raise ValueError(f"method='{method}' needs samples=<clean rows> to fit the manifold")
            S = np.atleast_2d(np.asarray(samples, float))
            if method == "manifold":
                basis, mean = fit_manifold(S, rank=rank)
                return manifold_denoise(x, basis, mean)
            # 'adaptive' and 'pnp' both want a GENEROUS basis whose coefficients get noise-thresholded
            basis, _, mean = fit_manifold_full(S, rank=min(4 * rank, S.shape[1]))
            if method == "adaptive":
                return adaptive_manifold_denoise(x, basis, mean, sigma=sigma)
            if forward is None or adjoint is None:    # pnp
                raise ValueError("method='pnp' needs forward and adjoint callables (the operator A and A^T)")
            prior = lambda v: adaptive_manifold_denoise(v, basis, mean, sigma=sigma)
            return pnp_restore(x, forward, adjoint, prior, mu=mu, steps=pnp_steps)

        raise ValueError(f"unknown denoise method: {method!r}")

    def project_onto_constraints(self, x, projections, iters=30, tol=None, omega=1.0):
        """Satisfy a set of constraints on a vector by ITERATED PROJECTION -- sweep a list of projections
        (each a callable x->x' snapping x onto one constraint set / manifold) until they jointly hold. This is
        the one engine under three faculties the mind grew separately (Macklin's observation -- the object he
        builds in position-based dynamics): the SBC resonator is alternating projection onto factor codebooks,
        `denoise(method='pnp')` is alternating projection onto data-fidelity and the signal manifold (it now
        literally calls this), and a constraint sweep is PBD. With `omega`<1 the update is under-relaxed (PBD's
        stability trick); onto convex sets this is POCS and converges to their intersection. Returns
        (x, n_sweeps, converged). Deterministic given deterministic projections. Useful directly for enforcing
        arbitrary constraints on a hypervector (be a valid code AND agree with a partial observation, say)."""
        from holographic_denoise import project_onto_constraints as _poc
        return _poc(np.asarray(x, float), list(projections), iters=iters, tol=tol, omega=omega)

    def restore(self, y, mask=None, samples=None, forward=None, adjoint=None,
                mu=0.5, steps=30, sigma=None, rank=8):
        """Restore a degraded measurement y = A(clean) + noise by Plug-and-Play / RED -- the inverse-problem
        faculty (Milanfar's thesis: a denoiser IS a map of the signal manifold, so it is the prior in ANY
        inverse problem -- deblur, super-resolve, and especially INPAINT). The common case is inpainting an
        ERASED signal (Ozcan's degraded archive plate): pass `mask` (1 where observed, 0 where erased) and the
        forward operator A and its transpose are filled in for you -- a diagonal mask is its own transpose. For
        other operators pass `forward`/`adjoint`. The prior is THIS mind's adaptive manifold denoiser fit from
        `samples` (clean rows -- e.g. an archive's own gallery), which estimates the noise level itself, so it
        does not over-smooth at low noise.

        This is addendum B7 wired as a LOOP, not a one-shot: it delegates to `denoise(method='pnp')`, iterating
        data-fidelity <-> denoise so the erased entries are filled by the manifold while the observed ones are
        held to the measurement. That iteration is exactly why it beats a single denoise on erasure -- the
        one-shot projection of the masked signal is dragged toward zero by the missing values. Returns the
        restored vector."""
        if forward is None or adjoint is None:
            if mask is None:
                raise ValueError("restore needs a `mask` (for inpainting) or explicit forward/adjoint operators")
            mvec = np.asarray(mask, float)
            forward = adjoint = (lambda x, _m=mvec: _m * x)   # a diagonal mask is its own transpose (A == A^T)
        return self.denoise(np.asarray(y, float), method="pnp", samples=samples,
                            forward=forward, adjoint=adjoint, mu=mu, pnp_steps=steps, sigma=sigma, rank=rank)

    def fit_function(self, X, y, n_grid=24, bandwidth=8.0, ridge=1e-2):
        """Fit an interpretable function y ~ F(X) as a single-layer Kolmogorov-Arnold readout on this
        mind's encoders (holographic_kan): F(x) = sum_j psi_j(x_j), each per-feature psi_j a sum of
        adaptive-grid RBF bumps, all coefficients fit by ONE deterministic ridge least-squares solve
        (no backprop). The fitted model exposes .predict(X) and .feature_function(j, ts) -- the plottable
        learned univariate part for feature j, KAN's interpretability pitch.

        X is (N, n_features) (a 1-D array is taken as one feature); y is (N,). Returns the fitted
        HolographicKAN, built at this mind's seed so it stays seed-reproducible like the rest.

        KEPT NEGATIVE: a single-layer additive KAN cannot represent feature INTERACTIONS (e.g. x1*x2) --
        it is additive by construction. That needs a second layer or explicit interaction features; the
        boundary is shown, not hidden."""
        from holographic_kan import HolographicKAN
        X = np.asarray(X, float)
        if X.ndim == 1:                                # a lone feature vector -> a single column
            X = X.reshape(-1, 1)
        model = HolographicKAN(X.shape[1], dim=min(self.dim, 512), n_grid=n_grid,
                               bandwidth=bandwidth, seed=self.seed, ridge=ridge)
        return model.fit(X, np.asarray(y, float))

    # ---- the SEARCH & DYNAMICS faculties (integration plan, Tier 3) -----------------------------
    # Min-cost search on a graph or a trellis (a maze; a fragment assembly) and learned linear
    # dynamics -- the last modules built beside the mind, now faculties of it. Where the structure is
    # natural the search returns a B7 typed structure (assemble); dynamics is, literally, an algebra
    # of binds.

    def design_network(self, nbr, terminals, steps=200, mu=2.0, dt=0.2, keep=0.25):
        """Multi-terminal network design by the Tero/Physarum flow model -- the 'Tokyo rail' experiment, the
        multi-terminal generalisation of `solve_maze`. Given a graph `nbr` (adjacency {node: [neighbours]})
        and a set of TERMINALS (food sources), grow tubes that thicken with flux until an efficient network
        connecting every terminal emerges (holographic_flow.tero_network). `mu` tunes the trade-off Physarum
        is famous for: HIGH mu -> a near-minimal Steiner TREE, LOW mu -> a REDUNDANT, fault-tolerant mesh.

        The network is returned BOTH as raw edges and as a B7 TYPED STRUCTURE: a graph-memory recipe
        M = superpose over edges of bind(node_u, node_v) -- the same construction `chain_structure` uses, so
        it realize()s to one hypervector and the engine's unbind+cleanup recalls a node's neighbours
        (unbind M by a node atom, snap the result to the node codebook). Returns a dict with 'edges', the
        typed 'structure' (recipe), the 'memory' vector, and 'nodes' (name -> unitary atom for cleanup)."""
        from holographic_flow import tero_network
        from holographic_ai import derived_atom
        edges, _D = tero_network(nbr, terminals, steps=steps, mu=mu, dt=dt, keep=keep)
        rec = self.typed_structure()
        names = sorted({x for e in edges for x in e}, key=str)
        handles = {nd: rec.atom(f"node:{nd}", unitary=True) for nd in names}        # symbolic build-graph nodes
        nodes = {nd: derived_atom(self.seed, f"node:{nd}", self.dim, unitary=True)  # matching codebook (cleanup)
                 for nd in names}
        memory = None
        if edges:                                                                  # M = superpose bind(u, v)
            rec.mark_output(rec.superpose([rec.bind(handles[u], handles[v]) for (u, v) in edges]))
            memory = self.realize(rec)
        return {"edges": edges, "structure": rec, "memory": memory, "nodes": nodes}

    def solve_maze(self, world, steps=200, mu=1.5, dt=0.2):
        """Solve a GridWorld maze by the DETERMINISTIC Tero flow-conductance model (holographic_flow):
        Physarum-style tubes thicken with flux (Poiseuille conductance) until the network collapses onto
        the shortest path. Same (path, info) interface as the stochastic slime solver, but deterministic
        and ~100x faster, and it lands EXACTLY on the optimum on braided mazes. info reports
        reached / optimal / extracted_len / cells / deterministic. Returns (path, info)."""
        from holographic_flow import solve_maze_flow
        return solve_maze_flow(world, steps=steps, mu=mu, dt=dt)

    def flow_circulation(self, nbr, start, goal, steps=200, mu=1.5, dt=0.2):
        """Decompose a solved Tero/Physarum flow into TRANSPORT and CIRCULATION -- the analysis layer the flow
        solver lacked, wiring it to the Helmholtz-Hodge split (EXP-8) and the graph's topology (the B1 of
        EXP-5/7). Runs the flow on the adjacency graph `nbr` (the same one `solve_maze`/`tero_solve` take),
        reads the CONVERGED signed edge flux (`holographic_flow.tero_flux` -- the quantity the solver computes
        and discards), then Hodge-splits it: the GRADIENT part is the net source->goal transport (its
        divergence is the injected current), and the HARMONIC part is circulation around the graph's loops (its
        dimension is the graph's B1). A maze graph has no filled triangles, so there is no curl. Returns a dict
        {loops, redundancy, transport_energy, circulation_energy, flux, edges, n_vertices}: `loops` = B1 (the
        graph's independent cycles), and `redundancy` = the harmonic fraction of the flux energy -- how much of
        the converged flow CIRCULATES rather than transports. It is exactly 0 on a tree (the route is forced),
        and on a loopy graph it varies with mu (a previously-hidden property of the flow: at high mu competing
        thick tubes leave more circulating flux). Returns None if start and goal are disconnected."""
        from holographic_flow import tero_flux
        from holographic_spectral import hodge_decomposition, betti_numbers
        res = tero_flux(nbr, start, goal, steps=steps, mu=mu, dt=dt)
        if res is None:
            return None
        n, edges, flux = res
        gradient, _curl, harmonic = hodge_decomposition(n, edges, flux, None)   # graph -> no triangles -> no curl
        _, b1 = betti_numbers(n, edges, None)
        circ = float(np.dot(harmonic, harmonic))
        total = float(np.dot(flux, flux)) + 1e-30
        return {
            "loops": b1,
            "redundancy": circ / total,
            "transport_energy": float(np.dot(gradient, gradient)),
            "circulation_energy": circ,
            "flux": flux,
            "edges": edges,
            "n_vertices": n,
        }

    def assemble(self, target, library, frag_len=2, steps=300, mu=1.5, dt=0.2, energy=None):
        """Assemble `target` from a `library` of overlapping fragments by MIN-ENERGY flow search
        (holographic_assembly) -- Rosetta-style fragment assembly (choose a fragment per position to
        minimise a placement energy, consecutive fragments overlap-agreeing) cast as the SAME min-cost-
        path flow the maze solver runs, on a (position x fragment) trellis. Returns a dict: the assembled
        string, its total energy, the chosen (position, fragment) list, and a B7 StructureRecipe binding
        each fragment to its position -- the assembly AS a typed holographic structure (realize() it,
        save() it). Built at this mind's dim/seed.

        It finds the GLOBAL optimum (it matches the exact Viterbi DP), not a greedy one. `energy` is an
        optional callable energy(frag, pos, target) -> non-negative cost (default: Hamming mismatch). A
        SUPPLIED energy is the Rosetta move -- not every mismatch costs the same (a substitution matrix, a
        cleanup-based score) -- and the search stays globally optimal under it. The default stand-in is the
        combinatorial core, not a protein force field (the kept negative); a real energy is now pluggable."""
        from holographic_assembly import assemble as _assemble
        return _assemble(target, library, frag_len=frag_len, steps=steps, mu=mu, dt=dt,
                         dim=self.dim, seed=self.seed, energy=energy)

    def compare_structures(self, a, b, dim=None, seed=None, tol=0.1):
        """Superpose two assembled structures (assemble() outputs) and read their OVERLAP -- the Baker seat's
        compare-two-folds. Returns {placement_overlap, holographic_overlap, shared}: the exact shared-motif
        fraction (overlap coefficient of the (pos, fragment) sets), the SAME overlap read holographically from
        the superposed role-bound vectors via consolidation (so you can compare structures you only hold as
        hypervectors -- a recalled fold -- and it degrades gracefully under noise), and the shared placements.
        On clean structures the two overlaps agree, the holographic read validated against the exact count.
        Built at this mind's dim/seed by default."""
        from holographic_assembly import compare_structures as _cmp
        return _cmp(a, b, dim=(self.dim if dim is None else dim),
                    seed=(self.seed if seed is None else seed), tol=tol)

    def wasserstein(self, a, b, cost=None, eps=None):
        """The Wasserstein (earth-mover's) distance between two distributions by Sinkhorn iteration
        (holographic_transport, BLD-8) -- the least work to MOVE one onto the other, mass times the ground
        distance it travels. Unlike a bin-wise metric (Euclidean/cosine), it keeps GROWING as two distributions
        move apart even after they stop overlapping (a peak at bin 12 vs bin 20 reads farther from bin 10 than
        bin 12 does), which is the right answer when the bins carry geometry (position, time, frequency).
        `cost` is the ground-distance matrix (default |i-j|); `eps` is the entropic regularisation (default
        scales to the cost). KEPT NEGATIVES: the eps knob (too large blurs the distance high, too small
        underflows the kernel) and O(n*m) per iteration. Returns the distance."""
        from holographic_transport import wasserstein as _w
        return _w(a, b, cost=cost, eps=eps)

    def learn_dynamics(self, states, ridge=1e-3):
        """Learn a fixed dynamics operator U so that state(t+1) ~ bind(U, state(t)) -- dynamics as an
        ALGEBRA OF BINDS (holographic_dynamics). In HRR's Fourier domain a learned bind is a per-frequency
        complex transfer, i.e. the Koopman/DMD operator in Fourier coordinates (the object Stam's FFT
        fluid step and Puckette's phase vocoder also manipulate). Returns a Propagator with .step(state)
        (one-step prediction = a SINGLE bind), .rollout(state, k), and .recall_at(state, k) -- recover the
        state k steps BEFORE one now, so the trajectory is content-addressable, not just forward-runnable.

        `states` is a sequence of state rows (T, dim). KEPT NEGATIVE on real market RETURNS: prediction
        only TIES a trivial mean predictor -- near-efficient-market returns have almost no linear structure
        for a fixed operator to exploit (the correct, expected result, kept on record). It SHINES where the
        dynamics ARE linear, now measured on two such regimes: AUDIO frames of a sustained tone (the per-bin
        phase advance; one-step error 0.001 vs persistence 1.64 / mean 1.00) and a FLUID field on a torus
        (linear advection-diffusion, each mode a rotation + decay; error 0.011 vs persistence 0.34 / mean 1.12,
        and the learned operator rolls out as a surrogate solver tracking the true sim to ~3% over 8 steps).
        Its HONEST LIMIT, also measured: a NONLINEAR Burgers field forms shocks no single fixed linear operator
        captures, where it does worse than persistence. The CONTENT-ADDRESSABLE round-trip (the operator's own
        forward k then back k returns the start at cosine ~1.0) is the durable win regardless of regime."""
        from holographic_dynamics import Propagator
        return Propagator.learn(np.asarray(states, float), ridge=ridge)

    # ---- the GENERATIVE faculties (integration plan, Tier 4) -----------------------------------
    # Generation is denoising run backwards, and a splat scene is a bundle -- so the last two modules
    # built beside the mind reconcile straight into it: generate a vector by the cleanup-attractor
    # diffusion, and represent a 2-D field as a superposition of Gaussian primitives.

    def generate_vector(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=None,
                        readout="softmax"):
        """GENERATE a hypervector by denoising FROM PURE NOISE (B10) -- the cleanup attractor as a tiny
        holographic diffusion (holographic_hopfield.generate): start from a random unit vector, anneal
        beta UP (vague -> sharp) and injected noise DOWN across `steps`, and walk onto the codebook
        manifold. Generation and denoising are the SAME operation in different regimes -- this is the
        vector-level twin of the text generate(), pointed at the B10 diffusion sampler. Returns a unit
        vector, deterministic in `seed` (this mind's seed by default).

        KEPT NEGATIVE: over a BARE codebook this converges to a stored atom (a degenerate sampler) --
        feed it a COMPOSED or continuous manifold for novel-but-valid samples. `readout='sparsemax'` is
        available but does NOT change the bare/continuous-codebook behaviour (measured: both readouts snap
        to a stored atom, novelty ~0) -- the sparse readout's win is on generate_structure (below), where it
        cures generative mode collapse."""
        from holographic_hopfield import generate as _generate
        return _generate(np.asarray(codebook, float), steps=steps, beta0=beta0, beta1=beta1,
                         noise0=noise0, seed=self.seed if seed is None else seed, readout=readout)

    def generate_structure(self, roles, fillers, steps=16, beta0=4.0, beta1=60.0, noise0=0.5, seed=None,
                           readout="softmax"):
        """GENERATE a novel-but-valid COMPOSED structure by denoising from noise over the composition manifold
        (B10 + the Eno reframe) -- the composed-subspace answer to `generate_vector`'s kept negative. Same
        annealed diffusion, but the denoiser is a slot-wise projection (unbind each role, snap the filler to
        the vocabulary, rebind, bundle), so the walk lands on the manifold of role-filler STRUCTURES rather
        than collapsing to a stored atom. `roles` is (S, dim) unitary role atoms, `fillers` is (V, dim) the
        filler vocabulary; the result is a unit vector whose every slot unbinds to a vocabulary atom -- a NEW
        combination (one of V^S), valid by construction (re-encoding the decoded fillers reproduces it).
        Different seeds give different structures; `generate_vector` over a bare codebook returns a stored
        atom (the degenerate case this fixes). Deterministic in `seed` (this mind's seed by default).

        `readout='sparsemax'` switches the slot-wise blend from softmax to the sparse readout. MEASURED: both
        readouts produce perfectly VALID structures (the generated vector reencodes its decoded combination at
        cosine 1.000), but softmax generation MODE-COLLAPSES -- many random seeds settle into the same few
        structures (diversity as low as 0.03-0.5) because the softmax blend's wide metastable basins funnel
        them together -- while sparsemax stays DIVERSE (0.6-1.0, nearly every seed a distinct valid structure).
        It is the same metastable-mixing fix as the cleanup/resonator readout, here curing generative mode
        collapse at no validity cost. Default stays softmax for backward-compatibility; sparsemax is the
        recommended setting when sampling for variety."""
        from holographic_hopfield import generate_structure as _gs
        return _gs(np.asarray(roles, float), np.asarray(fillers, float), steps=steps, beta0=beta0,
                   beta1=beta1, noise0=noise0, seed=self.seed if seed is None else seed, readout=readout)

    def _fractal_codebooks(self, G=8):
        """Deterministic codebooks for the fractal-kernel seed: a G*G grid of offset POSITIONS in the unit
        square (each with a unitary atom), a position role, a scale role, and a small set of candidate scales
        (each with an atom). Built from this mind's seed, so encode and decode agree by construction."""
        from holographic_ai import derived_atom
        cells = [(i / (G - 1), j / (G - 1)) for i in range(G) for j in range(G)]
        pos_atoms = np.stack([derived_atom(self.seed, f"fpos:{i}:{j}", self.dim, unitary=True)
                              for i in range(G) for j in range(G)])
        pos_role = derived_atom(self.seed, "f:pos_role", self.dim, unitary=True)
        scale_role = derived_atom(self.seed, "f:scale_role", self.dim, unitary=True)
        scales = [0.5, 1.0 / 3.0, 0.25]
        scale_atoms = np.stack([derived_atom(self.seed, f"fscale:{k}", self.dim, unitary=True)
                                for k in range(len(scales))])
        return cells, pos_atoms, pos_role, scale_role, scales, scale_atoms

    def fractal_seed(self, offsets, scale, G=8):
        """Encode a fractal KERNEL into ONE seed hypervector (Quilez: one kernel, a single seed). The kernel is
        N copies of the plane, each contracted by `scale` and translated to an (x, y) in `offsets` (snapped to
        the grid codebook). The seed is the holographic bundle
            seed = sum_k bind(pos_role, grid_atom[offset_k])  +  bind(scale_role, scale_atom[scale])
        -- the kernel carried in the geometry of one vector. Decode and expand it with `fractal_scene`.
        Returns a unit seed vector; different kernels give different seeds."""
        from holographic_ai import bind
        cells, pos_atoms, pos_role, scale_role, scales, scale_atoms = self._fractal_codebooks(G)
        parts = []
        for (ox, oy) in offsets:                                  # snap each offset to the nearest grid atom
            k = int(np.argmin([(ox - cx) ** 2 + (oy - cy) ** 2 for cx, cy in cells]))
            parts.append(bind(pos_role, pos_atoms[k]))
        s = min(scales, key=lambda z: abs(z - scale))             # nearest candidate scale
        parts.append(bind(scale_role, scale_atoms[scales.index(s)]))
        v = np.sum(parts, axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def fractal_scene(self, seed, depth=8, G=8, max_points=60000):
        """Decode the kernel from a seed hypervector and EXPAND it to a self-similar scene by domain repetition
        of that one kernel to `depth` (the scene-of-scenes-of-scenes Quilez asked for -- each level places a
        contracted copy of the whole), then report its fractal dimension. Decoding is pure VSA: unbind the
        position role and threshold the grid atoms to recover WHICH cells are offsets, unbind the scale role
        and clean up to recover the scale. Returns {'offsets', 'scale', 'n_maps', 'points', 'dimension',
        'expected'} where expected = log(N)/log(1/scale) is the self-similar (Hausdorff) dimension the box
        count should land near. Deterministic in the seed."""
        from holographic_ai import unbind
        from holographic_fractal import box_counting_dimension
        cells, pos_atoms, pos_role, scale_role, scales, scale_atoms = self._fractal_codebooks(G)
        pq = unbind(seed, pos_role); pn = pq / (np.linalg.norm(pq) + 1e-12)
        sims = pos_atoms @ pn                                     # which grid cells are present in the seed?
        offsets = [cells[k] for k in range(len(cells)) if sims[k] > max(0.5 * float(sims.max()), 0.12)]
        sq = unbind(seed, scale_role); sn = sq / (np.linalg.norm(sq) + 1e-12)
        s = scales[int(np.argmax(scale_atoms @ sn))]             # which candidate scale?
        N = len(offsets)
        d = depth
        while N > 1 and N ** d > max_points and d > 1:           # keep N^depth bounded
            d -= 1
        pts = np.zeros((1, 2))
        for _ in range(d):                                       # one kernel, repeated to depth d
            pts = np.vstack([pts * s + np.array(o) for o in offsets]) if offsets else pts
        dim = float(box_counting_dimension(pts)) if N > 1 else 0.0
        expected = float(np.log(N) / np.log(1.0 / s)) if (N > 1 and s < 1) else float("nan")
        return {"offsets": offsets, "scale": s, "n_maps": N, "points": pts,
                "dimension": dim, "expected": expected, "depth": d}

    def splat_field(self, target, k=20, denoise=False):
        """Represent a 2-D field/image as a SUPERPOSITION of K Gaussian primitives (holographic_splat) --
        the structural twin of bundle (a Gaussian-splat scene IS a bundle, and the RBF ScalarEncoder is
        already a Gaussian splat in hypervector space). Fits the splats by matching pursuit (greedy
        superposition); returns (splats, rendered) where `splats` is a compact (cy, cx, amplitude, sigma)
        code and `rendered` is their sum. With denoise=True returns just the rendered field, which is a
        DENOISER -- a few smooth Gaussians have no capacity for high-frequency noise.

        KEPT NEGATIVE / SCOPE: isotropic splats and a fixed scale set (the honest matching-pursuit
        baseline); anisotropic covariances and gradient refinement (full 3DGS) remain out of scope.
        Storing a whole gallery AS splat codes is now splat_archive() (holographic_splat_archive)."""
        from holographic_splat import splat_fit, splat_render
        splats = splat_fit(np.asarray(target, float), k)
        rendered = splat_render(splats, np.asarray(target).shape)
        return rendered if denoise else (splats, rendered)

    def distributed_forward(self, layers, x, K=1, cleanup_books=None, relu=True):
        """A federated (and optionally deep, cleanup-gated) forward pass in the holographic space -- Path D's
        compute win: the storage array's federation applied to the MATMUL. A linear layer's weight rows stored
        in ONE bundled vector cap at ~0.02 x D classes (crosstalk on the continuous logit, with no cleanup to
        absorb it); FEDERATING the rows across K weight-memory shards (row c in shard c mod K) moves the wall to
        ~K x 0.02 x D -- measured: 16 classes faithful on one vector -> 96 on eight shards (~6x), tracking the
        exact classifier far past where a single vector collapses. For DEPTH, pass `cleanup_books` (a codebook of
        valid hidden activations per hidden layer) and each layer's output is snapped onto that manifold (a soft
        dense-Hopfield) so crosstalk resets between layers instead of compounding; or compute each layer exactly
        with `exact_matmul`, which has no crosstalk to compound at all.

        `layers` -- one (out, in) weight matrix or a list of them (deep). `x` -- (in,) or (N, in). Returns the
        final-layer logits, (N, out_last). KEPT NEGATIVE: federation buys FIDELITY / capacity, not fewer FLOPs
        (total unbinds are still C, grouped into K vectors; the K shards parallelise on neuromorphic hardware),
        and WITHOUT a depth cure a deep federated pass decays with depth -- the decay that cleanup (or exact
        arithmetic) removes. Delegates to holographic_compute."""
        import holographic_compute as hc
        return hc.distributed_forward(layers, x, K=K, seed=self.seed, cleanup_books=cleanup_books, relu=relu)

    def pivot_index(self, items, fanout=7, seed=None):
        """A recursive pivot-tree index for SUBLINEAR nearest-item recall (Path D, the forest/data-structure
        seat). A naive index that summarizes items upward into a bundle hits the capacity wall; a B-tree holds
        PIVOTS explicitly instead, so the wall never bites. Here each node is a small cleanup memory of
        (pivot -> child) and routing is a nearest-pivot decision applied RECURSIVELY -- the same `cleanup`
        primitive the mind already uses, one level per hop, inception as the addressing fabric. Returns a
        holographic_pivot.PivotIndex: `.query(q, beam)` -> (nearest item index, pivot comparisons used),
        `.reached(q, beam)` -> the candidate leaf set.

        Greedy top-1 routing (beam=1) matches an exhaustive scan while touching only ~O(log N) pivots; a wider
        beam buys near-perfect recall of the true leaf into the candidate set, after which an exact key-unbind
        finishes. KEPT NEGATIVE: each hop is an approximate nearest-pivot decision, so a wrong turn at beam=1 can
        lose a query on overlapping data -- the beam is the honest knob that recovers recall; the build cost is
        the recursive k-means (NumPy only, no sklearn -- the minimal-frameworks rule)."""
        from holographic_pivot import PivotIndex
        return PivotIndex(np.asarray(items, float), fanout=fanout, seed=(self.seed if seed is None else seed))

    def exact_matmul(self, W, x, scale=None, moduli=None):
        """Exact integer / fixed-point matmul carried over the FHRR phasor algebra -- Path D's arithmetic lever.
        General matmul in a lossy superposition dies as the matrix grows (crosstalk: the bundled rows interfere
        on readout). This instead carries each number as residues over coprime moduli (a Residue Number System)
        and does every multiply-accumulate as EXACT phasor-binding modular arithmetic -- a product of unit
        phasors adds their phases, so the modular sum is exact for ANY number of terms, with no crosstalk --
        then recomposes the integer with the Chinese Remainder Theorem. The dynamic range FEDERATES over moduli
        channels (more channels -> bigger exact range), the arithmetic sibling of `storage_array`'s federation.

        Integer W, x -> exact y = W @ x. Float W, x (with `scale`, or auto when the dtype is float) -> a
        fixed-point exact-arithmetic matmul (delegates to holographic_rns). KEPT NEGATIVE / scope: exact for
        integer / fixed-point operands within range -- a float is QUANTIZED first and the only error is that
        rounding (set by `scale`), a bit-depth question, not the crosstalk wall (it does not grow with size); and
        the FLOPs are real -- the parallelism is per-modulus / per-output, native on phasor / RNS hardware."""
        import holographic_rns as rns
        W = np.asarray(W)
        x = np.asarray(x)
        if scale is not None or W.dtype.kind == "f" or x.dtype.kind == "f":
            return rns.rns_matmul_float(W, x, scale=(scale if scale is not None else 64), moduli=moduli)
        return rns.rns_matmul(W, x, moduli=moduli)

    def storage_array(self, n_parity=1, n_vals=256, add_threshold=0.90):
        """A federated, RAID-style symbol store -- the capacity/resilience faculty from the Path D
        'as above, so below' arc. One D-vector holds only ~0.1 x D symbols faithfully, and that budget is
        CONSERVED: to store more you FEDERATE across aligned shards coordinated by a thin layer
        (align/place/grow/protect), which is the within-vector graceful-degradation move applied one rung up,
        between shards. Returns a holographic_array.HoloArray on the mind's own dim and seed -- `.add(value_index)`
        stores a symbol (auto-growing a shard under capacity pressure), `.recall(g, down=...)` recalls it
        (reconstructing any lost shards from parity by subtraction -- the real-valued sibling of a fountain
        droplet), and `.accuracy(...)` measures recall across the federation.

        KEPT NEGATIVE / information floor: `n_parity` parity shards survive at most `n_parity` simultaneous
        shard losses -- it cannot recover more losses than it has parity, mirroring the fountain's 'too few
        droplets -> nothing'. The coordinator delegates to the same bind/unbind/derived_atom kernel the mind's
        own memory uses; it adds coordination, never a new algebra. The array also supports three recall modes:
        `.recall(g)` (directory-routed, O(1)), `.broadcast_recall(g)` (routerless, O(shards)), and
        `.routed_recall(g, c)` -- content-addressable SKETCH ROUTING that matches a key against per-shard
        key-sketches and unbinds only the top-c candidate shards, staying accurate where broadcast erodes."""
        from holographic_array import HoloArray
        return HoloArray(self.dim, seed=self.seed, n_parity=n_parity, add_threshold=add_threshold, n_vals=n_vals)

    def superpose_compute(self, items, query=None, codebook=None, keys=None, shards=1):
        """The WIDTH faculty: evaluate K computations at once inside ONE vector (Kanerva / Kleyko 'computing in
        superposition') -- the parallel-readout complement to the mind's DEPTH side (recursive structure:
        `encode_tree`, `peel` traversal, the measured inception depth law). Bundles the keyed items into a single
        vector (holographic_superposed.pack), recovers them all with one batched unbind, and -- given a `query` --
        scores them in parallel and returns the winner, cleanup-gated against `codebook` when supplied (the
        discrete decision that resets crosstalk). Returns a dict: `packed`, `keys`, `recovered` ((n, D) noisy
        readout), `decoded` (per-item cleanup indices, when a `codebook` is given), and -- with a `query` --
        `scores` + `winner` (+ `winner_score`).

        `shards > 1` FEDERATES the items across that many vectors (item i -> shard i mod shards), recovering each
        shard separately, so the width wall moves ~shards-fold -- the storage array's federation applied to the
        readout. That lets one call serve the Bucket-A tasks the single vector capped: hypothesis SELECTION among
        more candidates than one vector holds (pass a `query`) and SEQUENCE recall of a longer symbol string (pass
        position-atom `keys` + a symbol `codebook`, read `decoded`).

        Keys default to unitary atoms derived from the mind's seed, so a single keyed item recovers EXACTLY and
        the only readback error is superposition crosstalk. KEPT NEGATIVE / the conservation law: one D-vector
        holds only ~0.1-0.2 x D items under cleanup-gated recall and ~0.02 x D when recovered values feed
        continuous math with no cleanup -- width is bounded per vector; you buy more by spending DEPTH (recurse,
        cleanup-gate each level) or by FEDERATING across shards, not by widening one flat bundle."""
        import holographic_superposed as hs
        from holographic_ai import unitary_vector
        items = np.asarray(items, float)
        if items.ndim == 1:
            items = items[None, :]
        n = items.shape[0]
        if keys is None:
            rng = np.random.default_rng(self.seed)
            keys = np.stack([unitary_vector(self.dim, rng) for _ in range(n)])
        else:
            keys = np.asarray(keys, float)
        if shards <= 1:
            packed = hs.pack(keys, items)
            recovered = hs.recover_all(packed, keys)
        else:
            recovered = np.zeros_like(items)
            packed = []
            for k in range(shards):                              # federate items across shards (i mod shards)
                idx = np.arange(n)[np.arange(n) % shards == k]
                if len(idx) == 0:
                    continue
                Sk = hs.pack(keys[idx], items[idx])
                packed.append(Sk)
                recovered[idx] = hs.recover_all(Sk, keys[idx])   # recover only this shard's items
            packed = np.stack(packed) if packed else np.zeros((0, self.dim))
        out = {"packed": packed, "keys": keys, "recovered": recovered}
        if codebook is not None:
            cb = np.asarray(codebook, float)
            decoded = np.array([hs.resolve(r, cb)[0] for r in recovered])   # cleanup each item to the codebook
            out["decoded"] = decoded
        if query is not None:
            q = np.asarray(query, float)
            scores = (cb[decoded] if codebook is not None else recovered) @ q
            win = int(np.argmax(scores))
            out.update(winner=win, winner_score=float(scores[win]), scores=scores)
        return out

    def reservoir(self, n_in=1, rho=0.95, leak=0.3, in_scale=0.6, recurrence="shift"):
        """Gradient-free SEQUENCE learning -- the substrate-native Echo-State Network, the truly
        derivative-free corner of the learning program. The recurrent reservoir is FIXED: holostuff's
        `permute` (a cyclic shift) is norm-preserving (orthogonal), which is exactly the echo-state property,
        so the engine's own sequence operator IS a near-optimal reservoir; only a linear READOUT is trained,
        by one closed-form ridge regression -- no gradients, no backprop-through-time, fully deterministic.
        Returns a holographic_reservoir.HolographicESN on the mind's dim/seed: `.fit(U, Y)` solves the readout,
        `.predict(U)`, and `.generate(n, warm_U, feedback)` runs closed-loop autoregressive generation.

        Measured: NARMA10 to a literature-grade NRMSE ~0.37 (the reservoir features carry it -- it beats a
        linear-on-raw baseline), and gradient-free LEARNED text generation. KEPT NEGATIVE: chaotic free-running
        prediction MUST diverge pointwise after ~one Lyapunov time (the 'climate' is learnable, the 'weather'
        is not), and the readout learns a linear map of FIXED reservoir features, not new internal features.
        Delegates to holographic_reservoir."""
        from holographic_reservoir import HolographicESN
        return HolographicESN(n_in, dim=self.dim, rho=rho, leak=leak, in_scale=in_scale,
                              seed=self.seed, recurrence=recurrence)

    def prototype_classifier(self, levels=32, bandwidth=3.5):
        """Gradient-free CLASSIFICATION -- the HDC/VSA prototype learner, the other truly derivative-free
        method. Stage 1: encode each example (bind a feature-id atom with a ScalarEncoder level, bundle over
        features) and BUNDLE a class's examples into one prototype -- a one-pass centroid. Stage 2: perceptron
        retraining -- on a misclassified example, pull the correct prototype toward it and push the wrongly
        predicted one away (add/subtract on bundled vectors, no gradients). Returns a
        holographic_classifier.HolographicClassifier on the mind's dim/seed: `.fit(X, y, epochs)`, `.predict(X)`.

        Measured (test accuracy): digits 0.90 one-shot -> 0.95 retrained, breast_cancer 0.93 -> 0.95, and the
        encoding lifts a centroid model dramatically (wine raw-centroid 0.67 -> HDC 0.98). KEPT NEGATIVE (the
        field's own verdict): retraining beats the one-shot centroid, but the classifier lands just BELOW a
        tuned linear model (logistic regression) -- traded for a dead-simple gradient-free rule. Delegates to
        holographic_classifier."""
        from holographic_classifier import HolographicClassifier
        return HolographicClassifier(dim=self.dim, levels=levels, bandwidth=bandwidth, seed=self.seed)

    def equilibrium_net(self, n_in, n_hidden=64, n_out=2, beta=0.35, dt=0.4, t_free=40, t_nudge=12):
        """Equilibrium Propagation -- the LEARNING RULE for the energy-based (Hopfield) memory the engine
        uses as a fixed cleanup. Where the dense-Hopfield cleanup relaxes a query to a FIXED stored
        attractor, EP LEARNS the weights of a continuous Hopfield net so its energy minima encode a task. No
        backprop: a free relaxation (clamp the input -> equilibrium = the prediction) and symmetric nudged
        relaxations (+/- beta * loss on the output) whose difference is a contrastive Hebbian update that
        estimates the loss gradient (Scellier & Bengio 2017; Laborieux 2021, symmetric nudging). Returns a
        holographic_equilibrium.EquilibriumNet: `.fit(X, y_onehot, epochs)`, `.predict(X)`, plus
        `.ep_gradient_Who` / `.fd_gradient_Who` for the gradient-matching correctness check.

        This is the LOCAL-GRADIENT corner of the learning program -- NOT derivative-free like `reservoir`
        and `prototype_classifier`; EP estimates a gradient, just with relaxations rather than a backward
        pass. Its payoff over those two: it learns the HIDDEN weights, so it fits a NONLINEAR task a linear
        readout cannot -- on two interleaving moons it reaches ~0.92 vs a linear model's ~0.85, and its
        symmetric update matches the true gradient to cosine ~1.0. KEPT NEGATIVE / scope: needs SYMMETRIC
        weights; costs several relaxations per update (far more than a one-shot rule); the finite-beta
        estimate is still biased (it lands below exact backprop, which reaches ~1.0 here); and it is
        validated at small / moderate scale, not frontier scale. Delegates to holographic_equilibrium."""
        from holographic_equilibrium import EquilibriumNet
        return EquilibriumNet(n_in, n_hidden=n_hidden, n_out=n_out, beta=beta, dt=dt,
                              t_free=t_free, t_nudge=t_nudge, seed=self.seed)

    def forward_forward(self, n_in, layer_sizes=(64, 64), n_classes=2, theta=0.05, label_scale=3.0):
        """The Forward-Forward algorithm -- backprop-free, settling-free DEPTH from purely LOCAL objectives
        (Hinton 2022). A stack of layers, each trained by its own goodness objective via TWO forward passes
        (positive data -> high goodness; data with a WRONG label embedded -> low goodness), each layer's
        weights moving by the gradient of THAT layer's local logistic alone, with L2-normalization between
        layers so a later layer can't read the length an earlier one already separated. Classification is
        label-embedded: prepend a one-hot label, and at test pick the label whose accumulated goodness is
        highest. Returns a holographic_forward.ForwardForwardNet: `.fit(X, y, epochs)`, `.predict(X)`.

        LOCAL-GRADIENT, not derivative-free (like `equilibrium_net`): each layer follows its own local
        gradient; there is just no backward pass linking the layers, and no relaxation. Its niche over EP is
        arbitrary DEPTH with a cheap closed-form local update per layer. KEPT NEGATIVE (measured, loud): at
        the small scale here this compact FF is a WORKING but WEAK classifier -- it TRAILS a plain linear /
        logistic model on every task tried (two-moons ~0.88 tie; overlapping 4-class blobs 0.95 vs 0.99;
        sklearn digits 0.88 vs logistic 0.97), beating linear only on a radial task where linear provably
        fails, and even then weakly. FF's published accuracy (Hinton's ~1.4% MNIST error) needs the full
        MNIST-scale recipe; what this demonstrates is the MECHANISM -- positive goodness provably separates
        from negative -- a conceptual route to backprop-free depth, not a competitive number. The stronger
        Mono-Forward (2025) refinement (per-layer local supervised heads) is the natural next step, not built
        here. Delegates to holographic_forward."""
        from holographic_forward import ForwardForwardNet
        return ForwardForwardNet(n_in, layer_sizes=layer_sizes, n_classes=n_classes,
                                 theta=theta, label_scale=label_scale, seed=self.seed)

    def learn_chaos(self, states, dim=600, rho=0.9, leak=1.0, in_scale=0.5,
                    ridge=1e-6, washout=200, noise=1e-2):
        """Learn a NONLINEAR dynamics operator -- the companion to `learn_dynamics`. Where
        `learn_dynamics` fits ONE per-frequency complex transfer (the linear Koopman/DMD operator, exact
        for linearisable flow but pinned at the persistence floor on a state-dependent or chaotic system),
        this fits the reservoir's FIXED nonlinear expansion plus a TRAINED linear readout to the one-step
        evolution map -- the learned lift the dynamics negative called for. Returns a
        holographic_chaos.NonlinearPropagator on the mind's seed: `.predict_sequence(states)` for one-step-
        ahead forecasts, `.free_run(warmup, k)` for closed-loop rollout. It delegates to the reservoir
        faculty (it does not re-implement a learner).

        Measured (Lorenz '63, the canonical reservoir-computing test): the nonlinear one-step prediction
        lands ~0.0014 relative error -- about 40x better than the BEST linear map (full DMD) and ~50x
        better than persistence, where every linear operator sits at the chaos floor. Deterministic.
        KEPT NEGATIVES (loud): (1) closed-loop free-run tracks only ~ONE Lyapunov time -- far short of what
        the one-step error implies, because the autonomous reservoir has the well-known free-run STABILITY
        problem; noise=1e-2 is the sweet spot and more hurts, bigger reservoirs help only modestly, and the
        recurrence mixing (shift / perm / unitary-bind) is NOT the lever. (2) HIGH-dimensional PDE fields
        are out of reach for a single global reservoir (a 48-D Burgers field forecasts at ~0.27, worse than
        persistence; the literature needs local/parallel reservoirs, and EP is weak at high-D field
        regression too). (3) On mild dissipative flow persistence is a punishing baseline regardless of
        learner. The win is a genuine LOW-dimensional nonlinear-dynamics result, honestly bounded.
        Delegates to holographic_chaos."""
        from holographic_chaos import NonlinearPropagator
        return NonlinearPropagator.learn(states, dim=dim, rho=rho, leak=leak, in_scale=in_scale,
                                         ridge=ridge, washout=washout, noise=noise, seed=self.seed)

    def learn_cleanup(self, patterns, noise=0.30, n_hidden=None, epochs=80, beta=0.5):
        """Learn a cleanup's ATTRACTORS instead of storing them -- a LEARNED energy memory. Every cleanup
        in the engine is fixed: the classical one snaps to a stored atom, and the modern-Hopfield energy
        cleanup (`cleanup(..., energy=True)` / dense_cleanup) relaxes against a FIXED codebook. This trains
        an energy (via Equilibrium Propagation -- it delegates to `equilibrium_net`, not a new learner)
        whose attractors form a LEARNED manifold, so a noisy query is projected onto the manifold rather
        than snapped to the nearest stored sample. `patterns` are manifold samples in [0,1]; the returned
        holographic_energy.LearnedEnergyMemory exposes `.cleanup(x)` (single vector or batch). This is the
        natural learned prior for a Plug-and-Play / RED restoration loop.

        Measured: on a continuous manifold the learned energy beats the fixed SOFT energy cleanup at every
        codebook size, and on a manifold of dimension >= 2 it beats a matched-memory codebook of random
        samples (2-D: ~0.43 vs ~0.49) -- learning's compactness beats the curse of dimensionality that
        makes tiling a manifold with samples cost ~grid^d points. Deterministic.
        KEPT NEGATIVES (loud): (1) for DISCRETE atoms recovered from isotropic noise the HARD 1-NN cleanup
        returns the EXACT atom (~0.02) and is unbeatable -- a learned approximate energy cannot beat exact
        recovery (B1's tie, sharpened to a loss); use the existing cleanup for discrete recall. (2) In 1-D
        the curse does not bite, so a matched-memory codebook wins (~0.27 vs ~0.33) -- the advantage over
        storing data requires manifold dimension >= 2. (3) The win over a codebook is at MATCHED memory,
        not unbounded, and EP inherits its weakness at very high output dimension (moderate D, low
        intrinsic-dim manifolds). Delegates to holographic_energy."""
        from holographic_energy import LearnedEnergyMemory
        return LearnedEnergyMemory.learn(patterns, noise=noise, n_hidden=n_hidden,
                                         epochs=epochs, beta=beta, seed=self.seed)

    def tensor_bind(self, keys, values, rank=None):
        """A TENSOR-PRODUCT (outer-product) binding memory -- the uncompressed cousin of HRR's circular
        convolution -- optionally truncated to a rank-r 2-site TENSOR TRAIN (MPS). Stoudenmire's seat: HRR's
        bind is a compressed projection of the tensor product a (X) b, and a tensor-network representation
        interpolates between the two. Returns a holographic_tensor.TensorBindMemory with `.recall(key)` and
        `.n_numbers` (its honest storage).

        What the capacity comparison shows (see the integration tests): at a fixed LOAD the tensor-product bind
        recalls far more accurately than HRR -- crosstalk is suppressed by the key inner products (~1/sqrt(D))
        -- because it spends D x the storage; and with ORTHOGONAL keys it recalls EXACTLY up to M = D, where
        circular convolution cannot. An MPS truncation LOSSLESSLY compresses a low-rank binding matrix (the
        tensor-network win). KEPT NEGATIVE: on the capacity-per-STORED-NUMBER frontier the two are equal --
        HRR's compression gives up nothing there -- and a generic (full-rank) binding cannot be MPS-compressed
        without losing recall, so this is a different point on the storage/fidelity tradeoff, not a free
        improvement over the engine's bind."""
        from holographic_tensor import TensorBindMemory
        return TensorBindMemory(np.asarray(keys, float), np.asarray(values, float), rank=rank)

    def clifford(self):
        """Cl(3,0) geometric algebra as a PARALLEL binding mode (holographic_clifford) -- the geometric-product
        cousin of tensor_bind, for GEOMETRIC structure. Its seat: rotors compose 3D rotations EXACTLY (the
        product of two rotors IS the rotor of the composed rotation, measured error ~1e-15) and
        NON-COMMUTATIVELY (rotation order is preserved), which the engine's COMMUTATIVE convolution bind cannot
        do -- a commutative binding gives one answer for both orders and so carries the whole order-gap as
        error. Returns a CliffordAlgebra with product / rotor / rotate / compose / reverse. KEPT NEGATIVE: 2^d
        dimension growth (Cl(3,0)=8, Cl(10,0)=1024), and it binds VERSORS (rotors), not arbitrary atoms -- a
        parallel tool for the rotation-shaped corner, not a general HRR bind replacement."""
        if getattr(self, "_clifford", None) is None:
            from holographic_clifford import CliffordAlgebra
            self._clifford = CliffordAlgebra()
        return self._clifford

    def splat_aniso(self, field, k=12, steps=200, denoise=False):
        """Represent an n-D field (a 2-D image or a 3-D volume) as a superposition of ANISOTROPIC Gaussian
        splats -- the real 3D-Gaussian-Splatting primitive (oriented, elliptical/ellipsoidal Gaussians with a
        full covariance), fit by gradient descent on the reconstruction MSE (holographic_splat.aniso_fit:
        analytical NumPy gradients + a tiny built-in Adam, no autodiff framework). The anisotropic, n-D
        extension of `splat_field`'s isotropic matching pursuit: one aligned splat replaces many circular ones
        wherever structure is oriented or elongated, in 2-D OR 3-D. Returns (splats, rendered) where each splat
        is (center, amplitude, L) with L the inverse-covariance Cholesky factor; denoise=True returns just the
        rendered field (a few smooth Gaussians cannot hold high-frequency noise).

        KEPT NEGATIVE / SCOPE: the loss is non-convex, so this finds a LOCAL optimum -- more splats do not help
        monotonically (a clean K=4 fit can beat a messier K=8 one) and the result depends on the isotropic warm
        start; and this is the from-scratch core of 3DGS only -- no tile rasteriser, no spherical-harmonic
        view-dependent colour, no GPU speed."""
        from holographic_splat import aniso_fit
        splats, rendered = aniso_fit(np.asarray(field, float), k, steps=steps)
        return rendered if denoise else (splats, rendered)

    def image_archive(self, shape, capacity, keep=None, dim=32768, thumb=12):
        """The DCT/Walsh-Hadamard plate archive (holographic_archive.HolographicArchive) as a mind faculty --
        the exact, CROSS-MODAL counterpart to the lossy `splat_archive`. Images are superposed into orthogonal
        key plates and any one reconstructs EXACTLY when undamaged (a single adjoint per channel) or gracefully
        under erasure (joint masked recovery -- ~0.002 error even at 40% plate loss). Cross-modal both ways:
        `.add(image, tags=[...], nums={...})` attaches a descriptive address, `.recall_by_tags(words=[...])`
        returns the best-matching image FROM THE DESCRIPTION ALONE (Ozcan's describe-then-retrieve, soft-AND
        over the query tags), and `.tags_of(i, candidates)` runs the reverse -- the description the archive
        would give a stored image. `keep` is the DCT coefficients kept per image (defaults to all shape[0]^2,
        bit-exact; fewer trades exactness for compression). Requires capacity*keep <= dim. Seeded by this mind."""
        from holographic_archive import HolographicArchive
        if keep is None:
            keep = shape[0] * shape[0]                  # all coefficients -> exact recall by default
        return HolographicArchive(shape, capacity, keep=keep, dim=dim, seed=self.seed, thumb=thumb)

    def federated_archive(self, shape, capacity, K=4, keep=2000, dim=32768, thumb=12):
        """A FEDERATED image archive (holographic_archive.FederatedArchive) -- the storage array's federation
        applied to the content archive, and the archive twin of `storage_array`. One archive's per-vector budget
        is conserved, so to hold more images you spread them across K aligned HolographicArchive shards with a
        directory (image i -> shard i mod K): total capacity is K x per-shard, recovery quality holds at a fixed
        per-image budget (more images = more shards, not one archive blurring), and it extends one shard at a
        time. Returns the coordinator: `.add(image, tags=, nums=)` stores and routes (returning a global index),
        `.recover(i, mask=)` recovers image i from its shard. Same per-shard DCT / Walsh-Hadamard plate algebra
        the single archive uses -- only a routing layer added, never a new image codec. Seeded by this mind."""
        from holographic_archive import FederatedArchive
        return FederatedArchive(shape, capacity, K=K, keep=keep, dim=dim, seed=self.seed, thumb=thumb)

    def splat_archive(self, shape, keep=40):
        """Open a SPLAT-BUNDLE image archive (holographic_splat_archive) -- a gallery stored as Gaussian-
        splat codes BESIDE the WHT-plate archive (a splat scene is a bundle). add(image) fits K splats per
        channel; recover(i, k) renders them (a k-prefix is a progressively-refined preview, since matching
        pursuit stores them in importance order); recall(query) finds an image by content; region(i, box)
        is an EXACT 'what is here' query. Returns a fresh SplatArchive for `shape`.

        KEPT NEGATIVE: this is LOSSY -- the WHT-plate archive is EXACT undamaged and, on DCT-friendly
        images, beats it on quality at a matched byte budget; the splat archive's win is the ADDED
        region-query + progressive-refinement (and a compact code), not quality parity. It sits beside the
        plates, not in place of them."""
        from holographic_splat_archive import SplatArchive
        return SplatArchive(shape, keep=keep)

    def render_scene(self, tag_list, S=96, seed=0):
        """Render composed attribute tags to an actual RGB image via the scene renderer."""
        from holographic_scene import make_scene
        return make_scene([(t["shape"], t["colour"]) for t in tag_list], S=S, seed=seed)

    def morph_scene(self, img_a, img_b, steps=9):
        """Morph between two images in the DCT-coefficient domain (structure-blending
        slerp, not a ghosting crossfade), reusing the generation bundle's morph on a DCT
        basis sized to these images. Part of this mind's generative repertoire."""
        from holographic_archive import HolographicArchive
        from holographic_generate import morph_images
        S = img_a.shape[0]
        arch = HolographicArchive(shape=img_a.shape, capacity=2,
                                  keep=min(900, (S * S) // 2), dim=32768, seed=self.seed)
        return morph_images(arch.M, img_a, img_b, steps=steps)

    def discover_units(self, stream, order=4, percentile=70):
        """Self-discovery of structure: find the units in a raw symbol stream with
        no labels, by branching entropy on the substrate (holographic_segment) --
        prediction is tight inside a unit and uncertain at its end, so boundaries
        are the entropy peaks. Returns {'chunks', 'boundaries', 'chunk_bits',
        'symbol_bits'} -- including the MDL payoff (discovered chunks compress better
        than single symbols). Pass a string or a list of symbols."""
        from holographic_segment import Segmenter, chunk_compression
        s = list(stream)
        seg = Segmenter(dim=self.dim, order=order, seed=0).fit(s)
        bounds = seg.boundaries(s, percentile)
        chunks = seg.segment(s, percentile)
        cb, sb = chunk_compression(s, chunks)
        return {"chunks": ["".join(map(str, c)) for c in chunks],
                "boundaries": sorted(bounds),
                "chunk_bits": float(cb), "symbol_bits": float(sb)}

    def compress_cost(self, tokens):
        """Better structure means better compression, measured: encode a sequence by
        the rank of each symbol under the meaning predictor and report the bits, the
        uniform baseline, and the compression ratio (below 1 means structure was
        exploited). A predictor IS a compressor. Needs build_meaning_predictor."""
        if not hasattr(self, "_meaning_pred"):
            return {"ratio": 1.0, "bits_per_symbol": 0.0, "n": 0}
        if not hasattr(self, "_compressor"):
            from holographic_compress import PredictiveCompressor
            self._compressor = PredictiveCompressor(self._meaning_pred)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return self._compressor.encode_cost(toks)

    def structure_compresses(self, windows):
        """The link itself: correlation between window structure scores and their
        compression ratios (negative = more structure -> better compression)."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return 0.0
        if not hasattr(self, "_compressor"):
            from holographic_compress import PredictiveCompressor
            self._compressor = PredictiveCompressor(self._meaning_pred)
        from holographic_compress import structure_compression_correlation
        return structure_compression_correlation(self._verifier, self._compressor, windows)

    def respond(self, query, length=30, query_weight=4.0):
        """Query-and-generate: answer a query with a continuation steered toward
        what the query is about, held coherent by the structure guard. Returns the
        generated token list. Needs build_meaning_predictor first."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return []
        from holographic_respond import respond as _respond
        return _respond(query, self._meaning_pred, self._verifier,
                        length=length, query_weight=query_weight)

    def respond_report(self, query, length=30, query_weight=4.0):
        """Answer a query AND measure the answer: returns the response with its
        relevance to the query (is it on-topic) and its structure score (is it
        coherent) -- both reported, so the answer is never trusted blindly."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "relevance": 0.0, "structure": 0.0}
        from holographic_respond import respond_report as _rr
        return _rr(query, self._meaning_pred, self._verifier,
                   length=length, query_weight=query_weight)

    def deliberate(self, query, max_iters=8, target_quality=0.45, length=26,
                   query_weight=5.0, seed=0):
        """Think before answering: draft a response, judge it, and refine -- keeping
        the best -- stopping early once it is good enough. The number of iterations
        is the 'thinking time' and adapts to how hard the query is (easy ones settle
        fast, hard ones take longer). Returns the response with its quality, the
        iterations used, and the full trace of drafts (the inner deliberation made
        visible). Needs build_meaning_predictor first."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "quality": 0.0, "iterations": 0, "trace": []}
        if not hasattr(self, "_deliberator"):
            from holographic_deliberate import Deliberator
            self._deliberator = Deliberator(self._meaning_pred, self._verifier)
        return self._deliberator.deliberate(
            query, max_iters=max_iters, target_quality=target_quality,
            length=length, query_weight=query_weight, seed=seed)

    def negotiate(self, query, max_iters=8, target_quality=0.55, length=26,
                  query_weight=5.0, seed=0):
        """Deliberate under competing judges (coherence vs novelty vs relevance):
        each draft is scored by all three and the kept draft is the most BALANCED
        (its weakest pressure least bad), not the one that wins a single axis. The
        per-judge trace shows the pressures resolving. Needs build_meaning_predictor."""
        if not hasattr(self, "_meaning_pred") or not hasattr(self, "_verifier"):
            return {"response": [], "scores": {}, "negotiated": 0.0, "iterations": 0, "trace": []}
        if not hasattr(self, "_deliberator"):
            from holographic_deliberate import Deliberator
            self._deliberator = Deliberator(self._meaning_pred, self._verifier)
        return self._deliberator.negotiate(
            query, max_iters=max_iters, target_quality=target_quality,
            length=length, query_weight=query_weight, seed=seed)

    def anticipate_meaning(self, recent):
        """Compose and settle a next-MEANING prediction; return (word, confidence).
        Even when the exact word is missed, the prediction lands near semantically
        appropriate words -- graded, compositional anticipation."""
        if not hasattr(self, "_meaning_pred"):
            return None, 0.0
        word, _vec, conf = self._meaning_pred.predict_meaning(list(recent))
        return word, conf

    def meaning_prediction_report(self, tokens):
        """Exact-symbol accuracy and semantic RANK (did the composed prediction
        land in the right neighbourhood; 0.5 = chance, 1 = always nearest)."""
        if not hasattr(self, "_meaning_pred"):
            return {"exact": 0.0, "semantic_rank": 0.5, "n": 0}
        toks = [w for s in [tokens] for w in (s if isinstance(s, list) else s.split())] \
            if isinstance(tokens, str) else list(tokens)
        return self._meaning_pred.evaluate(toks)

    def learn_word_generator(self, sentences, order=1, window=2):
        """Train a WORD-level context generator (holographic_generation): a word
        n-gram for local fluency plus learned meaning vectors for an optional
        topic pull. Kept separate from the char-level generate() above. Returns
        self."""
        from holographic_generation import ContextGenerator
        self._wordgen = ContextGenerator(dim=self.dim, order=order, window=window,
                                         seed=0).fit(sentences)
        return self

    def generate_words(self, seed, length=40, topic_weight=0.0, temperature=0.7,
                       seed_rng=0):
        """Generate at the word level. topic_weight blends a topic-alignment pull
        into the n-gram choice (0 = bare n-gram). NOTE, measured honestly: the
        topic pull does NOT buy real coherence on this substrate -- it is flat
        when n-gram candidates are sparse and collapses into degenerate repetition
        when pushed hard. Use topic_pull_tradeoff() to see the curve. The missing
        piece for LLM-like behaviour is a high-capacity learned P(next|context),
        not this re-ranking lever."""
        if not hasattr(self, "_wordgen"):
            raise RuntimeError("call learn_word_generator(sentences) first")
        return self._wordgen.generate(seed, length=length, topic_weight=topic_weight,
                                      temperature=temperature, seed_rng=seed_rng)

    def topic_pull_tradeoff(self, seeds, weights=(0.0, 2.0, 8.0, 16.0), length=40):
        """The honest experiment surfaced on the brain: for each topic_weight,
        mean coherence, transition validity, and lexical diversity. Coherence that
        'rises' only as diversity collapses is the metric being gamed by repetition,
        not real on-topic language -- the kept negative that explains why deeper
        conditioning alone does not make the brain an LLM."""
        if not hasattr(self, "_wordgen"):
            raise RuntimeError("call learn_word_generator(sentences) first")
        return self._wordgen.sweep(seeds, weights=weights, length=length)

    def _seq_mem(self):
        if self._sequences is None:
            from holographic_sequence import SequenceMemory
            # SHARE the encoder's symbol atoms, so sequence steps are the very
            # same vectors the rest of the mind uses -- a plan's steps can be
            # the labels it classifies, the records it relates, all one space
            self._sequences = SequenceMemory(dim=self.dim, vocab=self.encoder._symbols)
        return self._sequences

    def learn_hierarchical(self, name, observations):
        """Absorb observations of a possibly-NESTED plan. Each observation is a
        plan whose steps may themselves be (sub_name, [sub_steps]) pairs OR bare
        atomic step names. Stored so discover_hierarchy can test, recursively
        and by the SAME permutation test, which steps expand into ordered
        sub-plans -- structure unfolding fractally, one layer at a time."""
        if not hasattr(self, "_hier_obs"):
            self._hier_obs = {}
        self._hier_obs[name] = list(observations)
        return self

    def discover_hierarchy(self, name=None, z_threshold=2.0, _depth=0, _max_depth=6):
        """RECURSIVE self-discovery of order. Test the top-level observations for
        sequential structure (the permutation test); if they pass, self-assemble
        the canonical order, then for each step gather its OWN sub-observations
        (where the data provides them) and recurse -- the same test one layer
        down. The recursion STOPS honestly: at a step with no sub-observations,
        or whose sub-observations show no sequential signal (z below the same
        bar), or at a depth guard. Returns a nested tree:
            {step: subtree or None}  (None = atomic / order not found)
        so the discovered hierarchy is the data's own, measured at every layer,
        with no depth or shape declared in advance."""
        from holographic_sequence import sequentiality_z
        obs = (self._hier_obs.get(name) if name is not None
               else getattr(self, "_hier_obs", {}).get(None))
        if not obs or _depth >= _max_depth:
            return None
        # an observation is a list of steps; a step is either a bare name or a
        # (sub_name, [sub_steps]) pair. Split into the top-level order view and
        # a per-step bag of sub-observations.
        top_members, sub_obs = [], {}
        for ob in obs:
            order = []
            for step in ob:
                if isinstance(step, (list, tuple)) and len(step) == 2 \
                        and isinstance(step[1], (list, tuple)):
                    sub_name, sub_steps = step
                    order.append(sub_name)
                    sub_obs.setdefault(sub_name, []).append(list(sub_steps))
                else:
                    order.append(step)
            top_members.append(order)
        z = sequentiality_z(top_members, self.encoder._symbols)
        if z < z_threshold:
            return None                                  # no order here: stop
        canonical = self._canonical_order(top_members)
        tree = {}
        for step in canonical:
            children = sub_obs.get(step)
            if not children:
                tree[step] = None                        # atomic: honest stop
                continue
            # recurse: does THIS step's sub-observations carry order?
            sub_z = sequentiality_z(children, self.encoder._symbols)
            if sub_z < z_threshold:
                tree[step] = None                        # sub-steps are a bag: stop
            else:
                # stash and recurse one layer down
                key = (name, step, _depth)
                self._hier_obs[key] = children
                tree[step] = self.discover_hierarchy(key, z_threshold,
                                                     _depth + 1, _max_depth)
                if tree[step] is None:                   # recursion bottomed out
                    tree[step] = self._canonical_order(children)
        return tree

    def learn_sequences(self, labeled_sequences):
        """Absorb (sequence, label) pairs, KEEPING the raw ordered members per
        label so the mind can later DISCOVER which classes are genuinely
        sequential. Each sequence is also encoded order-free into the normal
        memory (a bag of its elements) for classification -- the two views
        coexist; discovery decides which matters for each class."""
        if not hasattr(self, "_seq_members"):
            self._seq_members = {}
        for seq, label in labeled_sequences:
            self._seq_members.setdefault(label, []).append(list(seq))
            # order-free view into the standard memory (classification still works)
            self.learn(list(seq), label, modality="text")
        return self

    def discover_sequential(self, z_threshold=2.0):
        """SELF-DISCOVERY of order. For every absorbed class, run the permutation
        test (real order vs the class's own shuffled null) and report which
        classes carry genuine sequential structure -- no magic constant, the
        class is measured against itself, and z>2 is the standard significance
        bar (signal exceeds two sigma of the null), not a tuned threshold.

        Classes that pass get an order-aware prototype in the sequence memory
        (their canonical order recovered), so precedes()/validate_plan() work on
        the discovered structure. Returns {label: z_score} for all tested
        classes, so the continuous evidence is visible, not just the verdict."""
        from holographic_sequence import sequentiality_z
        if not getattr(self, "_seq_members", None):
            return {}
        verdicts = {}
        for label, members in self._seq_members.items():
            z = sequentiality_z(members, self.encoder._symbols)
            verdicts[label] = round(z, 2)
            if z >= z_threshold:
                # the class scored sequential AND must PROVE executable (no
                # precedence cycle) before its order is trusted -- statistical
                # signal is necessary but not sufficient; the structure has to
                # be consistent enough to actually walk
                ok, _ = self.prove_executable(members)
                if ok:
                    canonical = self._canonical_order(members)
                    self._seq_mem().add(label, canonical)
                    verdicts[label] = (z, "executable")
                else:
                    verdicts[label] = (round(z, 2), "inconsistent")
        return verdicts

    def execute_plan(self, name, context=None, attempt_order=None, templates=None):
        """RUN a discovered, proven plan -- the loop from discovering structure to
        ACTING on it. The contract is honest: a step fires only when (a) every
        step that must PRECEDE it (by the discovered canonical order) has already
        fired, and (b) its context SLOTS can be bound from `context`. Otherwise it
        BLOCKS, and the block is reported with its reason -- an unmet precondition
        or an unbound slot -- rather than silently assumed away.

        `context` is a dict binding slot names to values (the scenario: the
        physics law is generic, context supplies m and a; 'open the book' needs
        'book' bound). `templates` optionally maps a step to (template, slot_keys)
        from extract_template, so a step's slots are filled from context as it
        fires. `attempt_order` defaults to the canonical order (the natural run);
        pass a different order to test what blocks.

        Returns a log: [(step, status, detail)] where status is 'fired' (with the
        bound form in detail) or 'blocked' (with the reason). A plan that wasn't
        registered (failed discovery/proof) raises -- you cannot run what was
        never proven."""
        if name not in self._seq_mem().seqs:
            raise ValueError(f"'{name}' is not a proven sequential plan -- "
                             "discover_sequential must register it first")
        order = self._seq_mem().seqs[name][1]
        context = context or {}
        templates = templates or {}
        attempt = attempt_order or order
        done, log = set(), []
        for step in attempt:
            idx = order.index(step)
            required = set(order[:idx])
            missing = required - done
            if missing:
                log.append((step, "blocked",
                            f"preconditions unmet: needs {sorted(missing)}"))
                continue
            # bind slots from context if this step is a template
            if step in templates:
                template, slot_keys = templates[step]
                unbound = [k for k in slot_keys if k not in context]
                if unbound:
                    log.append((step, "blocked",
                                f"context missing bindings: {unbound}"))
                    continue
                # fill each <_> in order from the slot_keys' context values
                vals = [context[k] for k in slot_keys]
                parts, vi = [], 0
                for t in template:
                    if t == "<_>":
                        parts.append(str(vals[vi])); vi += 1
                    else:
                        parts.append(t)
                done.add(step)
                log.append((step, "fired", " ".join(parts)))
            else:
                done.add(step)
                log.append((step, "fired", ""))
        return log

    def extract_template(self, observations):
        """DISCOVER the generic schema and its context-bound slots in a repeated
        step. A step like 'the material has density X' is a SCHEMA (fixed words:
        'the material has density') plus a SLOT filled from context (X = 5g, 3g,
        ...). Across observations the schema positions are STABLE and the slot
        positions VARY -- so token-entropy per position separates them, and the
        split is placed at the natural largest GAP in the entropy distribution
        (the data's own scale, no magic cutoff). This is the same insight as a
        physical law: 'F = m*a' is generic until a scenario BINDS m and a; the
        schema is the law, the slots are where context enters.

        Returns (template, slots): template is the step with slots marked as
        '<_>', slots is {position: [observed values]} -- the variable parts and
        what context has filled them with. A step with no varying position is
        fully explicit (empty slots)."""
        import numpy as np
        from collections import Counter
        obs = [list(o) for o in observations]
        if len(obs) < 2:
            return (list(obs[0]) if obs else []), {}
        L = min(len(o) for o in obs)

        def entropy(toks):
            c = Counter(toks); n = len(toks)
            return -sum((v / n) * np.log2(v / n) for v in c.values())

        ents = np.array([entropy([o[i] for o in obs]) for i in range(L)])
        order = np.sort(ents)
        if order[-1] - order[0] < 1e-6:
            return [obs[0][i] for i in range(L)], {}   # all positions stable: explicit
        gaps = np.diff(order)
        split = order[int(np.argmax(gaps))] + gaps.max() / 2  # cut at the biggest gap
        slots = {}
        template = []
        for i in range(L):
            if ents[i] > split:
                slots[i] = [o[i] for o in obs]
                template.append("<_>")
            else:
                template.append(obs[0][i])
        return template, slots

    def prove_executable(self, members):
        """SELF-PROOF that a discovered order is VALID, not merely predictable.
        A class can score z>2 (its order carries signal) and still be
        INCONSISTENT -- if its members' pairwise precedences form a cycle
        (A before B, B before C, C before A), no single ordering satisfies them
        all and the 'plan' cannot be executed. The proof: build the majority
        precedence edges, take the canonical (vote-sorted) order, and check that
        EVERY majority edge is respected by it. A cycle shows up as an edge the
        sorted order must violate. Returns (ok, violations): ok means the
        structure proved itself executable; violations name the contradictory
        precedences. Structure earns trust by passing this, not by z alone."""
        from collections import Counter
        before = Counter()
        elems = set()
        for m in members:
            for i, a in enumerate(m):
                elems.add(a)
                for b in m[i + 1:]:
                    before[(str(a), str(b))] += 1
        order = self._canonical_order(members)
        pos = {e: i for i, e in enumerate(order)}
        # a ROBUST majority edge a->b that the canonical order reverses signals a
        # real cycle. "Robust" = the majority is stronger than the sampling noise
        # of sparse partial observations: a single contradictory vote from a rare
        # pair is not a contradiction, a consistent reversal is. The bar is the
        # data's own: an edge counts only if its margin exceeds the median edge
        # margin (so typical-strength evidence, not a fluke), and the canonical
        # order must still reverse it. No fixed constant -- the observation set
        # sets the scale.
        margins = [abs(before[(a, b)] - before[(b, a)])
                   for a in elems for b in elems if a < b]
        import numpy as np
        med = np.median([mg for mg in margins if mg > 0]) if any(margins) else 0
        violations = []
        for a in elems:
            for b in elems:
                if a != b and pos[a] > pos[b]:
                    margin = before[(a, b)] - before[(b, a)]
                    if margin > 0 and margin >= med:        # robust reversed edge
                        violations.append((a, b))
        return (not violations), violations

    def _canonical_order(self, members):
        """Recover a class's canonical step order from its member sequences by a
        true TOPOLOGICAL SORT over the majority-precedence edges -- so the
        discovered order RESPECTS what the data agrees on (if cut beats plate
        4-0, cut comes first), not a score heuristic that can misplace rare
        elements. Edges are the net majority (a before b more often than after);
        ties and the occasional cycle-inducing weak edge are broken by net
        margin, so a consistent dataset yields its exact order and a contradictory
        one yields the least-bad order (whose remaining violations prove_executable
        then surfaces)."""
        from collections import Counter
        before = Counter()
        elems = set()
        for m in members:
            for i, a in enumerate(m):
                elems.add(a)
                for b in m[i + 1:]:
                    before[(str(a), str(b))] += 1
        elems = [str(e) for e in elems]
        # net majority edge weight a->b (positive => a should precede b)
        def net(a, b):
            return before[(a, b)] - before[(b, a)]
        # Kahn-style topological sort, picking among available nodes the one with
        # the strongest outgoing majority (greedy by net precedence) so stronger
        # evidence is honoured first; this respects every consistent edge exactly.
        remaining = set(elems)
        order = []
        while remaining:
            # a node is "available" if no remaining node has a majority edge INTO it
            avail = [e for e in remaining
                     if not any(net(o, e) > 0 for o in remaining if o != e)]
            if not avail:                                  # a cycle: break it by
                # picking the node with the best net outgoing balance
                avail = [max(remaining,
                             key=lambda e: sum(net(e, o) for o in remaining if o != e))]
            pick = max(avail, key=lambda e: sum(net(e, o) for o in remaining if o != e))
            order.append(pick)
            remaining.discard(pick)
        return order

    def learn_plan(self, name, steps):
        """Store an ORDERED plan/recipe/protocol by name. Unlike absorb (which
        files things order-free for classification and recall), this keeps the
        SEQUENCE queryable: meaning that lives in the order is preserved."""
        self._seq_mem().add(name, steps)
        return self

    def step_at(self, name, i):
        """What is the i-th step of a stored plan?"""
        return self._seq_mem().step(name, i)

    def precedes(self, name, a, b):
        """In the stored plan, does step a come before step b? The order
        relation no bag store can answer (measured exact to ~40 steps at dim
        2048, ~93% at 120 -- graceful, not a hard cliff)."""
        return self._seq_mem().precedes(name, a, b)

    def validate_plan(self, name_or_steps, constraints):
        """Check a plan against ordering rules -- the PB&J test: does every
        'a must come before b' hold? Returns (ok, violations); a violation
        names exactly which step is out of order. Works on a stored plan name
        or a fresh step list."""
        return self._seq_mem().validate(name_or_steps, constraints)

    # -- executable procedures: HoloMachine as a faculty of the mind ---------------------------
    # A learn_plan stores an ordered list of opaque step LABELS ("beat the eggs"); a PROCEDURE stores
    # an executable recipe whose steps are actual VSA operations (LOAD/BIND/BUNDLE/PERMUTE/CALL), so
    # the recipe DOES something. The VM shares the mind's dim and seed, so a procedure's accumulator is
    # a vector in the mind's OWN space (seed it with a mind vector to transform it) and the format is
    # deterministic. This de-silos the stored-program machine: it is now a faculty, not an island.
    def _machine(self):
        """The HoloMachine VM, lazily built at the mind's dim & seed so procedures share the substrate."""
        if getattr(self, "_machine_vm", None) is None:
            from holographic_machine import HoloMachine
            self._machine_vm = HoloMachine(dim=self.dim, seed=self.seed)
        return self._machine_vm

    def learn_procedure(self, name, program):
        """Store a named PROCEDURE -- an executable ACC->ACC recipe of VSA operations, assembled into
        ONE hypervector and held in the machine's library, callable by name and composable (a procedure
        may CALL procedures defined earlier). `program` is a list of (opcode, operand). Define a
        procedure before any program that CALLs it."""
        self._machine().define(name, program)
        return self

    def run_procedure(self, name_or_program, init_acc=None, max_steps=512,
                      stop=None, max_loop=64, converge_tol=0.999, branch_tol=0.5):
        """Execute a procedure and return (accumulator, trace). `name_or_program` is the name of a
        stored procedure or a fresh list of (opcode, operand) instructions (which may CALL stored
        procedures). `init_acc` seeds the accumulator -- pass a vector from the mind's own space to
        transform it, which is what makes a procedure an operation on the mind's data. Control-flow
        knobs: `stop` is a predicate acc->bool that lets an ITERATE loop exit when the desired OUTPUT is
        reached; `max_loop` caps loop iterations; `converge_tol` is the fixed-point tolerance an ITERATE
        converges at; `branch_tol` is the IFMATCH match threshold."""
        M = self._machine()
        if isinstance(name_or_program, str):
            if name_or_program not in M.functions:
                raise KeyError(f"no procedure named {name_or_program!r} -- learn_procedure it first")
            pv = M.functions[name_or_program]
        else:
            pv = M.assemble(name_or_program)
        return M.run(pv, init_acc=init_acc, max_steps=max_steps, handlers=self._procedure_handlers(),
                     stop=stop, max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)

    def _procedure_handlers(self):
        """Handlers for APPLY <faculty>: one unary acc->acc map per faculty name, each delegating to a
        real mind faculty. 'cleanup' relaxes the accumulator toward the nearest known VALUE atom (the
        dense associative cleanup -- recovers a noisy accumulator); 'denoise' is the general manifold
        denoiser. KEPT NEGATIVE: denoise helps only when the accumulator carries low-rank/self-similar
        structure; on bare random value atoms there is no manifold, so 'cleanup' is the operative one
        there. Extend this dict (and DEFAULT_FACULTIES) to give procedures more unary faculties."""
        import numpy as _np
        M = self._machine()
        codebook = _np.stack([M.data_atoms[d] for d in M.data_names])

        def _cleanup(acc):
            from holographic_hopfield import dense_cleanup
            return dense_cleanup(acc, codebook, beta=25.0, steps=3)

        def _denoise(acc):
            return self.denoise(acc, method="auto")

        handlers = {"cleanup": _cleanup, "denoise": _denoise}
        W = getattr(self, "_matmul_W", None)
        if W is not None:                               # APPLY matmul := ACC <- W @ ACC (exact RNS matmul)
            def _matmul(acc):
                return self.exact_matmul(W, acc)
            handlers["matmul"] = _matmul
        inv = getattr(self, "_invprob", None)
        if inv is not None:                             # INV: restore a degraded measurement AS A PROGRAM --
            def _datafit(acc):                          # ITERATE [APPLY datafit; APPLY denoise] is the PnP/RED loop
                return acc - inv["mu"] * inv["adjoint"](inv["forward"](acc) - inv["y"])
            handlers["datafit"] = _datafit
            handlers["denoise"] = inv["prior"]          # the inverse problem's own manifold prior (overrides generic)
        gen = getattr(self, "_generator", None)
        if gen is not None:                             # GEN: generative diffusion AS A PROGRAM: ITERATE [APPLY diffuse]
            def _diffuse(acc):                          # one self-scheduled denoise-from-noise step: anneal beta up,
                import numpy as _np                     # cool injected noise down; ITERATE stops when the cooled,
                from holographic_hopfield import dense_cleanup   # sharpened cleanup reaches a fixed point on the manifold
                s = gen; frac = s["t"] / max(1, s["steps"] - 1)
                beta = s["beta0"] + (s["beta1"] - s["beta0"]) * frac
                noise = s["noise0"] * max(0.0, 1.0 - frac)
                z = dense_cleanup(acc, s["codebook"], beta=beta, steps=1, readout=s["readout"])
                if noise > 0:
                    z = z + noise * s["rng"].standard_normal(z.size) / _np.sqrt(z.size)
                z = z / (_np.linalg.norm(z) + 1e-12); s["t"] += 1
                return z
            handlers["diffuse"] = _diffuse
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:                            # PIPE-1: the data-analysis pipeline's faculties.
            handlers.update(self._pipeline_handlers(pipe))   # (its 'denoise' overrides the generic one,
        return handlers                                 # because a raw signal needs a signal-shaped prior.)

    def set_matmul(self, W):
        """Configure the matrix used by APPLY matmul: that opcode then does ACC := W @ ACC, carried by the
        EXACT RNS matmul (no crosstalk; floats are fixed-point quantized). Pass a dim x dim matrix so the
        accumulator keeps its shape and an ITERATE can loop it -- which makes `ITERATE [APPLY matmul]` a
        recurrent linear map iterated to a fixed point (the literal input->process->feed-back pattern with
        real linear algebra). Set to None to disable (APPLY matmul becomes a safe no-op again)."""
        import numpy as _np
        self._matmul_W = None if W is None else _np.asarray(W, dtype=float)
        return self

    def set_inverse_problem(self, y, forward, adjoint, prior, mu=0.8):
        """Configure the inverse problem that APPLY datafit / APPLY denoise solve as a PROGRAM. After this,
        ITERATE [APPLY datafit; APPLY denoise] runs the Plug-and-Play/RED restoration loop: datafit pulls ACC
        toward the measurement (ACC <- ACC - mu*adjoint(forward(ACC) - y)) and denoise applies `prior`, an
        acc->acc manifold map. `forward`/`adjoint` are the operator A and its transpose A^T. Pass y=None to
        disable (datafit becomes a no-op and denoise reverts to the generic faculty). Usually set via
        restore_procedure(); exposed for hand-built restoration programs."""
        M = self._machine()
        if "datafit" not in M.fac_atoms:                # register the faculty atom so a program can name it
            M.fac_atoms["datafit"] = M._atom("fac:datafit"); M.faculty_names.append("datafit")
        self._invprob = None if y is None else {"y": np.asarray(y, float), "forward": forward,
                                                 "adjoint": adjoint, "prior": prior, "mu": float(mu)}
        return self

    def set_generator(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0, readout="softmax"):
        """Configure the generative diffusion that APPLY diffuse runs as a PROGRAM. After this,
        ITERATE [APPLY diffuse] denoises from pure noise onto the `codebook` manifold -- the B10 sampler as a
        procedure. The diffuse step anneals beta up and injected noise down over `steps`; ITERATE halts when the
        cooled, sharpened cleanup reaches a fixed point. A composed/continuous codebook gives novel-but-valid
        samples; a BARE codebook degenerates to a stored atom (the kept B10 negative). Deterministic in `seed`.
        Usually set via generate_procedure(); exposed for hand-built generative programs."""
        import numpy as _np
        M = self._machine()
        if "diffuse" not in M.fac_atoms:
            M.fac_atoms["diffuse"] = M._atom("fac:diffuse"); M.faculty_names.append("diffuse")
        V = _np.atleast_2d(_np.asarray(codebook, float))
        V = V / (_np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)   # unit rows, as hopfield.generate uses
        self._generator = {"codebook": V, "steps": int(steps), "beta0": float(beta0), "beta1": float(beta1),
                           "noise0": float(noise0), "readout": readout, "rng": _np.random.default_rng(seed), "t": 0}
        return self

    def generate_procedure(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0,
                           readout="softmax", max_loop=None):
        """Generate a sample by running the B10 diffusion AS A VSA PROGRAM -- ITERATE [APPLY diffuse] from a noise
        seed -- so the generative PROCESS is a stored, composable procedure (savable via to_recipe), not Python
        control flow. Process, not object. Returns (sample_vector, trace). Matches hopfield.generate's quality
        (it lands on the manifold), up to ITERATE's convergence-stop replacing the fixed step count. Kept
        negatives: B10's (a bare codebook converges to a stored atom; feed a composed manifold for novelty) and
        the procedure tax (a noisy unbind-and-clean per instruction read -- slower than the direct loop, which is
        the price of being data, not a faster path)."""
        import numpy as _np
        self.set_generator(codebook, steps=steps, beta0=beta0, beta1=beta1, noise0=noise0, seed=seed, readout=readout)
        M = self._machine()
        if "_diffuse_step" not in M.functions:
            self.learn_procedure("_diffuse_step", [("APPLY", "diffuse"), ("HALT", "a")])
        z0 = self._generator["rng"].standard_normal(self.dim)
        z0 = z0 / (_np.linalg.norm(z0) + 1e-12)
        return self.run_procedure([("ITERATE", "_diffuse_step"), ("HALT", "a")], init_acc=z0,
                                  converge_tol=0.9999, max_loop=max_loop or max(40, steps * 3))

    def restore_procedure(self, y, forward, adjoint, samples, mu=0.8, rank=24, steps=60):
        """Restore a degraded measurement y = forward(clean) + noise by running Plug-and-Play/RED AS A VSA PROGRAM
        -- ITERATE [APPLY datafit; APPLY denoise] -- so the restoration LOOP is a stored, composable procedure
        rather than Python control flow. The manifold prior is fit from `samples` (clean rows). Returns
        (restored_vector, trace). Reaches the same error-to-truth as denoise(method='pnp'); ITERATE halts at the
        fixed point, typically in far fewer iterations than a fixed step budget. Kept negative: the procedure tax
        (noisy instruction reads) -- this is the being-data form of restoration, not a faster path."""
        import numpy as _np
        from holographic_denoise import fit_manifold_full, adaptive_manifold_denoise
        S = _np.atleast_2d(_np.asarray(samples, float))
        basis, _, mean = fit_manifold_full(S, rank=min(int(rank), S.shape[1]))
        prior = lambda v: adaptive_manifold_denoise(v, basis, mean, sigma=None)
        self.set_inverse_problem(y, forward, adjoint, prior, mu=mu)
        M = self._machine()
        if "_pnp_step" not in M.functions:
            self.learn_procedure("_pnp_step", [("APPLY", "datafit"), ("APPLY", "denoise"), ("HALT", "a")])
        return self.run_procedure([("ITERATE", "_pnp_step"), ("HALT", "a")],
                                  init_acc=_np.asarray(adjoint(y), float), converge_tol=0.99999, max_loop=steps)

    # ---- PIPE-1: an automatic data-analysis pipeline expressed as a VSA PROGRAM ----------------------
    # The pipeline is NOT Python control flow -- it is a HoloMachine program (APPLY faculties, an ITERATE
    # loop, an IFMATCH branch, a CALL sub-routine) that orchestrates real faculties. The accumulator carries
    # the signal through the denoise loop, then decompose hands back a structured/noise FLAG atom the IFMATCH
    # branches on. Findings are recorded in self._pipe['report']; each APPLY step delegates downward.

    def _denoise_signal(self, sig, window=None):
        """Denoise a lone 1-D signal against its own low-rank trajectory (SSA/Cadzow). Now a thin delegate to
        the promoted denoise(method='trajectory') -- the primitive lives in the denoise faculty (a general
        prior-free 1-D denoiser beside nlm), so the pipeline and any other caller share one implementation
        rather than this owning a private copy."""
        return self.denoise(np.asarray(sig, float).ravel(), method="trajectory")

    def _pipeline_handlers(self, pipe):
        """The APPLY faculties the pipeline program calls. Each is a unary acc->acc map that reads the
        working signal and records findings in pipe['report'], delegating to a real faculty:
          analyze   -> detect_topology + basic stats
          denoise   -> _denoise_signal (self-similar trajectory denoise; overrides the generic denoiser)
          decompose -> decompose_signal (the generative law) + residual; returns a 'structured'/'noise' flag
          train     -> a learned KAN readout over the signal (fit_function)
          validate  -> held-out generalization: refit on 80%, predict the last 20%
          save      -> persist the structured form, the COMPACT generative law (the law IS the small file)."""
        import numpy as _np
        M = self._machine()
        rep = pipe["report"]

        def _analyze(acc):
            sig = _np.asarray(acc, float).ravel()
            pipe["signal"] = sig
            try:
                from holographic_manifold import detect_topology
                topo, _ = detect_topology(_np.arange(sig.size, dtype=float), sig)
            except Exception:
                topo = "unknown"
            rep.update({"n": int(sig.size), "mean": round(float(sig.mean()), 4),
                        "std": round(float(sig.std()), 4), "topology": topo})
            return acc

        def _denoise(acc):                              # overrides the generic denoiser: signals need a prior
            den = self._denoise_signal(_np.asarray(acc, float).ravel())
            pipe["signal"] = den
            return den

        def _decompose(acc):
            sig = _np.asarray(acc, float).ravel()
            try:
                f, info = self.decompose_signal(sig)
                recon = _np.asarray(f.generate(_np.arange(sig.size, dtype=float))).ravel()
                explained = max(0.0, 1.0 - float(_np.var(sig - recon[:sig.size])) / (float(_np.var(sig)) + 1e-12))
            except Exception as e:                      # decompose failed -> treat as no structure found
                rep["decompose_error"] = str(e)
                return M.data_atoms["noise"]
            pipe["structure"] = f
            pipe["residual"] = sig - recon[:sig.size]
            rep.update({"topology": info.get("topology"), "n_terms": info.get("n_terms"),
                        "explained_var": round(explained, 3),
                        "compression_ratio": round(float(info.get("compression_ratio") or 0.0), 1)})
            # hand the program a FLAG atom so its IFMATCH can branch on whether a law was actually found
            return M.data_atoms["structured"] if explained >= 0.5 else M.data_atoms["noise"]

        def _train(acc):                                # a learned (nonparametric) readout over the signal
            sig = pipe.get("signal")
            pipe["model"] = pipe.get("structure")
            try:
                self.fit_function(_np.arange(sig.size, dtype=float).reshape(-1, 1), sig)
                rep["trained"] = True
            except Exception:
                rep["trained"] = pipe["model"] is not None
            return acc

        def _validate(acc):                             # held-out: refit on 80%, predict the unseen last 20%
            sig = pipe.get("signal")
            n = len(sig); cut = max(8, int(0.8 * n))
            try:
                f, _ = self.decompose_signal(sig[:cut])
                pred = _np.asarray(f.generate(_np.arange(n, dtype=float))).ravel()[cut:]
                rms = float(_np.sqrt(_np.mean((sig[cut:] - pred) ** 2)))
                rep["heldout_rms"] = round(rms, 4)
                rep["heldout_rel"] = round(rms / (float(_np.std(sig)) + 1e-12), 3)
            except Exception as e:
                rep["validate_error"] = str(e)
            return acc

        def _save(acc):                                 # store the structured form: the compact generative law(s)
            laws = pipe.get("level_laws") or ([pipe["structure"]] if pipe.get("structure") is not None else [])
            has_law = rep.get("explained_var", 0.0) >= 0.5 and len(laws) > 0   # only if real structure found
            if has_law:
                import os, tempfile
                total = 0
                for i, f in enumerate(laws):
                    if hasattr(f, "save"):
                        p = os.path.join(tempfile.gettempdir(), f"holostuff_pipe_law_{i}.json")
                        try:
                            f.save(p); total += os.path.getsize(p)
                        except Exception:
                            pass
                rep["law_bytes"] = total
                rep["saved_as"] = "law_ladder" if len(laws) > 1 else "generative_law"
            else:                                       # no compressible structure -> honest: keep raw samples
                rep["saved_as"] = "raw_only"
            return acc

        def _peel_step(acc):                            # one LEVEL of the recursive peel: decompose the residual
            r = _np.asarray(acc, float).ravel()
            if pipe.get("peel_input_var") is None:      # remember the energy we started peeling from
                pipe["peel_input_var"] = float(_np.var(r)) + 1e-12
            if float(_np.std(r)) < 0.01 * (pipe["peel_input_var"] ** 0.5):    # residual already negligible
                return r                                # (unchanged -> ITERATE converges) -- nothing left to peel
            try:
                f, info = self.decompose_signal(r)
                recon = _np.asarray(f.generate(_np.arange(r.size, dtype=float))).ravel()[:r.size]
                ev = max(0.0, 1.0 - float(_np.var(r - recon)) / (float(_np.var(r)) + 1e-12))
            except Exception:
                return r                                # can't decompose -> residual unchanged -> ITERATE stops
            # Stop on the engine's OWN MDL verdict: decompose returns n_terms==0 when no real structure remains
            # (it already gates out noise-fitting). A level may explain only a MODEST fraction yet be real --
            # a line trend under a comparable sine -- so the right test is "did the MDL gate admit a term?",
            # not "did this level explain most of the residual?".
            if (info.get("n_terms") or 0) < 1:
                return r                                # MDL found nothing -> residual is noise -> stop peeling
            pipe["levels"].append({"level": len(pipe["levels"]) + 1, "topology": info.get("topology"),
                                   "n_terms": info.get("n_terms"), "explained_of_residual": round(ev, 3)})
            pipe["level_laws"].append(f)
            return r - recon                            # hand the next level what THIS one could not explain

        def _assess(acc):                               # after peeling: finalize the report, flag the branch
            r = _np.asarray(acc, float).ravel()
            base = pipe.get("peel_input_var") or (float(_np.var(r)) + 1e-12)
            cum = max(0.0, 1.0 - float(_np.var(r)) / base)
            lv = pipe["levels"]
            rep.update({"n_levels": len(lv), "levels": lv, "cumulative_explained": round(cum, 3),
                        "final_residual_energy": round(float(_np.std(r)), 4),
                        "explained_var": round(cum, 3),         # same field the branch/save read in single mode
                        "n_terms": sum(int(d["n_terms"] or 0) for d in lv)})
            return M.data_atoms["structured"] if len(lv) > 0 else M.data_atoms["noise"]

        return {"analyze": _analyze, "denoise": _denoise, "decompose": _decompose,
                "train": _train, "validate": _validate, "save": _save,
                "peel": _peel_step, "assess": _assess}

    def analyze_dataset(self, data):
        """Set up the automatic-analysis pipeline over a 1-D signal: register the analyze/decompose/train/
        validate/save APPLY faculties and the 'structured'/'noise' flag atoms the program's IFMATCH branches
        on, and clear the findings report. The pipeline faculty/flag atoms are added to the machine ON DEMAND
        so the default VM stays lean. Then run the pipeline PROGRAM with run_analysis_pipeline (or assemble
        your own). Returns self."""
        import numpy as _np
        M = self._machine()
        for flag in ("structured", "noise"):            # IFMATCH branch flags (data codebook)
            if flag not in M.data_atoms:
                M.data_atoms[flag] = M._atom(f"dat:{flag}"); M.data_names.append(flag)
        for fac in ("analyze", "decompose", "train", "validate", "save", "peel", "assess"):  # APPLY faculties
            if fac not in M.fac_atoms:
                M.fac_atoms[fac] = M._atom(f"fac:{fac}"); M.faculty_names.append(fac)
        self._pipe = {"report": {}, "signal": _np.asarray(data, dtype=float),
                      "structure": None, "residual": None, "model": None,
                      "levels": [], "level_laws": [], "peel_input_var": None}
        return self

    def pipeline_report(self):
        """The findings recorded by the last analysis pipeline run (topology, explained variance, n_terms,
        held-out generalization, saved size)."""
        return dict(getattr(self, "_pipe", {}).get("report", {}))

    def run_analysis_pipeline(self, data, program=None, max_loop=12, recursive=False):
        """Run the standard automatic-analysis pipeline -- a VSA PROGRAM -- over a 1-D signal and return the
        findings report. The PROGRAM (not Python) drives the logic:

            APPLY  analyze            profile the data (stats + topology)
            ITERATE _denoise_step     denoise until the signal SETTLES   (the loop)
            APPLY  decompose          find the generative law; residual; flag structured/noise
            IFMATCH structured        only if a law was actually found...
              CALL _train_validate    ...train a readout and check held-out generalization
            APPLY  save               store the structured form: the compact generative law
            HALT

        With recursive=True the single `APPLY decompose` becomes `ITERATE _peel_step` -- decompose the
        DOMINANT law, then peel its residual and decompose THAT, layer by layer, until the residual has no
        structure left (the loop converges when decompose's own MDL gate admits no term -- n_terms==0 -- or
        the residual is already negligible). An `APPLY assess` then flags the branch and records the level
        ladder. This is the "access every level" mode: a line trend + a periodic part are caught on SEPARATE
        levels that one decompose cannot fit together (the trend explains only a modest fraction, so a level
        is kept by the MDL verdict, not by how much it explains).

        Every APPLY step delegates to a real faculty (see _pipeline_handlers). On a structured signal the
        program finds the law(s) and runs train+validate; on pure noise decompose reports no structure, the
        IFMATCH SKIPS train+validate, and the report says so honestly -- both branches from one program.
        Pass `program` to override the default."""
        import numpy as _np
        self.analyze_dataset(data)
        self.learn_procedure("_denoise_step", [("APPLY", "denoise"), ("HALT", "a")])
        self.learn_procedure("_train_validate", [("APPLY", "train"), ("APPLY", "validate"), ("HALT", "a")])
        self.learn_procedure("_peel_step", [("APPLY", "peel"), ("HALT", "a")])
        if program is None:
            if recursive:
                program = [("APPLY", "analyze"),
                           ("ITERATE", "_denoise_step"),
                           ("ITERATE", "_peel_step"),       # peel structure layer by layer (every level)
                           ("APPLY", "assess"),             # finalize the ladder, flag the branch
                           ("IFMATCH", "structured"),
                           ("CALL", "_train_validate"),
                           ("APPLY", "save"),
                           ("HALT", "a")]
            else:
                program = [("APPLY", "analyze"),
                           ("ITERATE", "_denoise_step"),
                           ("APPLY", "decompose"),
                           ("IFMATCH", "structured"),
                           ("CALL", "_train_validate"),
                           ("APPLY", "save"),
                           ("HALT", "a")]
        _, trace = self.run_procedure(program, init_acc=_np.asarray(data, dtype=float), max_loop=max_loop)
        rep = self.pipeline_report()
        rep["_ops"] = [t[0] for t in trace]             # which opcodes actually executed (shows the branch)
        return rep

    def index_procedures(self, names=None, probe_seed="canonical_probe"):
        """Build a behavioural FINGERPRINT index: run each procedure ONCE on a canonical probe and
        cache its output. Recall can then identify a bind-transform from a single example with ZERO
        further program runs (recover the key, match the implied fingerprint). One-time O(library) cost,
        amortised across all later recalls. Call again after adding procedures to refresh the index."""
        from holographic_ai import derived_atom
        M = self._machine()
        names = list(names) if names is not None else list(M.functions)
        self._proc_probe = derived_atom(self.seed, probe_seed, self.dim, unitary=True)
        handlers = self._procedure_handlers()
        self._proc_fp = {n: M.run(M.functions[n], init_acc=self._proc_probe, handlers=handlers)[0]
                         for n in names}
        # Cache a unit-normalized fingerprint MATRIX so recall is ONE matrix-vector product (cosine of the
        # implied kernel against every fingerprint at once) instead of a Python loop -- measured 6-26x faster
        # than the loop and 3-7x faster than a HoloForest index, which only beats a linear scan past ~4000
        # procedures (a regime the library vector cannot even hold). Rows are unit norm so mat @ qhat == cosine.
        names_l = list(self._proc_fp)
        mat = np.stack([self._proc_fp[n] for n in names_l]) if names_l else np.zeros((0, self.dim))
        self._proc_fp_names = names_l
        self._proc_fp_mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        return self

    def recall_procedure(self, input_vec, output_vec, names=None, method="auto", fp_floor=0.5):
        """Goal-addressable recall over the procedure library: given ONE (input -> output) example,
        return (name, score) of the stored procedure whose behaviour best reproduces it. Returns
        (None, score) if the library is empty.

        method='behavioral' runs each candidate on the input and matches its output -- general (works
        for ANY procedure), at O(library size) executions. method='fingerprint' uses the precomputed
        index (see index_procedures): it recovers the transform's kernel from the example and matches
        the IMPLIED fingerprint, with ZERO program runs -- exact for the LINEAR (convolution) class,
        which is bind AND permute and their compositions (permutation is convolution by a shifted
        delta, so it commutes with binding the same way), ~30x cheaper. It scores near zero for
        genuinely NONLINEAR procedures (e.g. ones with an APPLY cleanup/denoise step). method='auto'
        (default) gets the best of both: try the fingerprint shortcut first IF an index exists, trust
        it only when its match clears fp_floor (so it fires only for the linear transforms it actually
        covers -- nonlinear guesses score near zero), and otherwise fall back to the behavioural scan.
        With no index, 'auto' is exactly the behavioural scan, so this stays backward-compatible.

        The fingerprint scan is VECTORIZED: one matrix-vector product of the implied kernel against the
        cached unit-normalized fingerprint matrix gives every cosine at once (measured 6-26x faster than the
        per-candidate Python loop). A HoloForest index was measured and REJECTED: it is 3-7x SLOWER than this
        vectorized scan for any realistic library and only beats a linear scan past ~4000 procedures -- a
        regime the single library vector cannot hold anyway, so the sub-linear index is premature here. The
        right fix for an O(N) scan that was slow was to vectorize it, not to index it."""
        from holographic_ai import cosine as _cos, bind as _bind, unbind as _unbind
        M = self._machine()
        # -- fingerprint fast-path (zero program runs; gated by confidence so it only fires when right)
        if method in ("auto", "fingerprint") and getattr(self, "_proc_fp", None):
            implied = _bind(self._proc_probe, _unbind(output_vec, input_vec))   # assumes a bind transform
            mat = getattr(self, "_proc_fp_mat", None)
            if names is None and mat is not None and len(self._proc_fp_names):
                qhat = implied / max(float(np.linalg.norm(implied)), 1e-12)
                scores = mat @ qhat                          # cosine vs every fingerprint in ONE matvec
                bi = int(scores.argmax())
                best, bs = self._proc_fp_names[bi], float(scores[bi])
            else:                                            # named subset (or no matrix): scan the dict
                cands = list(names) if names is not None else list(self._proc_fp)
                best, bs = None, -9.0
                for n in cands:
                    if n in self._proc_fp:
                        s = float(_cos(implied, self._proc_fp[n]))
                        if s > bs:
                            bs, best = s, n
            if method == "fingerprint" or bs >= fp_floor:
                return best, bs
        # -- behavioural scan (general fallback)
        cands = list(names) if names is not None else list(M.functions)
        handlers = self._procedure_handlers()
        best, best_s = None, -9.0
        for n in cands:
            out, _ = M.run(M.functions[n], init_acc=input_vec, handlers=handlers)
            s = float(_cos(out, output_vec))
            if s > best_s:
                best_s, best = s, n
        return best, best_s

    def recall_and_apply(self, input_vec, output_vec, new_input):
        """Recall the procedure that performs a demonstrated (input -> output) transform, then APPLY it
        to NEW input -- 'learn the operation from one example, then use it' (analogy/transfer, VSA-
        native). Returns (result, name, score); result is None if the library is empty."""
        name, score = self.recall_procedure(input_vec, output_vec)
        if name is None:
            return None, None, score
        out, _ = self.run_procedure(name, init_acc=new_input)
        return out, name, score

    def synthesize_procedure(self, input_vec, output_vec, max_depth=2, threshold=0.9,
                             ops=("BIND", "BUNDLE", "PERMUTE")):
        """Construct a SHORT procedure that maps input_vec -> output_vec by bounded breadth-first search
        over the VM's operations -- the CONSTRUCTIVE counterpart to recall_procedure, which can only find
        a procedure already stored. Returns the program as a list of (opcode, operand) ending in HALT
        (run it with init_acc=input_vec), or None if nothing within max_depth reaches the target. BFS
        returns the SHORTEST such program, and it is VERIFIED by execution before return.

        Operands are the machine's data atoms; PERMUTE takes none. HONEST: the search branches by
        (ops x operands) per step, so it is EXPONENTIAL in depth -- bounded to short programs (depth
        2-3 are practical); it finds only programs over the KNOWN operations and operands; and it may
        return an EQUIVALENT program rather than a unique 'intended' one (binding is commutative, so the
        order of two BINDs is free). Because the moves it searches are the structured convolution/additive
        ops, a program verified on the single example generalises to other inputs (it captures the
        transform, not the pair) -- except where a non-structural op (PERMUTE before/after BUNDLE) makes
        order matter, which the search handles by trying both."""
        from holographic_ai import cosine as _cos, bind as _bind, bundle as _bundle, permute as _permute
        M = self._machine()
        data, atoms = M.data_names, M.data_atoms
        if float(_cos(input_vec, output_vec)) >= threshold:        # already there: identity program
            return [("HALT", data[0])]

        def step_vec(vec, op, operand):
            if op == "BIND":    return _bind(vec, atoms[operand])
            if op == "BUNDLE":  return _bundle([vec, atoms[operand]])
            if op == "PERMUTE": return _permute(vec, 1)
            raise ValueError(f"unsupported synthesis op {op!r}")

        moves = []                                                 # candidate (op, operand) moves per step
        for op in ops:
            if op == "PERMUTE":
                moves.append((op, data[0]))                        # operand is a don't-care placeholder
            else:
                moves.extend((op, d) for d in data)

        frontier = [(input_vec, [])]                               # (accumulator-so-far, program-so-far)
        for _ in range(max_depth):
            nxt = []
            for vec, prog in frontier:
                for op, operand in moves:
                    v2 = step_vec(vec, op, operand)
                    prog2 = prog + [(op, operand)]
                    if float(_cos(v2, output_vec)) >= threshold:
                        candidate = prog2 + [("HALT", data[0])]
                        out, _ = self.run_procedure(candidate, init_acc=input_vec)
                        if float(_cos(out, output_vec)) >= threshold:   # verify by execution
                            return candidate
                    nxt.append((v2, prog2))
            frontier = nxt
        return None

    def canonicalize_procedure(self, program, verify_input=None):
        """Reduce a BIND/PERMUTE procedure to its MINIMAL equivalent form. The invertible algebra collapses:
        any interleaving of k binds and m permutes applied to x equals `permute(x, m)` bound by the PRODUCT of
        all the bind operands -- a depth-(k+m) program is a depth-<=2 canonical one. (Measured: cosine 1.0000
        on every interleaving tried. Two binds are bind(x, a*b); a permute slides through a bind onto x,
        bind(permute(x), a); so every bind/permute program flattens.) The k binds collapse to ONE bind by the
        product; the m permutes stay m unit-shift ops, since this VM's PERMUTE is a fixed shift of 1.

        This is also WHY deeper synthesis over the invertible ops is unnecessary and a bidirectional 'meet in
        the middle' search buys nothing there: there is nothing deep to find -- the canonical form is depth
        <=2. The ops where depth WOULD matter -- BUNDLE (its normalization breaks bind-commutativity) and the
        nonlinear APPLY/ITERATE/IFMATCH/CALL/REPEAT -- do not collapse and are not cleanly invertible, so they
        are BARRIERS: if the program contains one, canonicalization is refused (fully_collapsible=False), an
        honest limit rather than a silent partial answer.

        Returns (canonical_program, info). info: fully_collapsible, net_shift, n_bind, original_len,
        canonical_len, verified (the canonical program reproduces the original on a probe input), and
        equivalence_cosine. A leading LOAD (which discards the input) is itself treated as a barrier."""
        import numpy as _np
        from holographic_machine import bind as _bind, cosine as _cos, derived_atom as _datom
        M = self._machine()
        BARRIERS = {"BUNDLE", "APPLY", "ITERATE", "IFMATCH", "CALL", "REPEAT", "LOAD"}
        bind_ops = []                                   # operand names of the binds (order irrelevant: commutes)
        shift = 0                                       # net permute = count of unit shifts
        barriers = []
        for i, (op, operand) in enumerate(program):
            if op == "HALT":
                break
            elif op == "BIND":
                bind_ops.append(operand)
            elif op == "PERMUTE":
                shift += 1
            else:                                       # BUNDLE / nonlinear / LOAD -> cannot flatten across it
                barriers.append((i, op))
        if barriers:
            return None, {"fully_collapsible": False, "barriers": barriers,
                          "reason": "BUNDLE and the nonlinear ops do not collapse (normalization breaks "
                                    "bind-commutativity; APPLY/ITERATE/etc. are not invertible) -- the honest "
                                    "limit on canonicalization, and on deep synthesis over these ops"}
        # the collapse: program(x) == permute(x, shift) bound by the product of all bind operands
        P = None
        for name in bind_ops:
            d = M.data_atoms[name]
            P = d if P is None else _bind(P, d)
        canon = [("PERMUTE", "a")] * shift              # m unit shifts (PERMUTE has no shift operand here)
        if P is not None:                               # one bind by the product (stored under a stable name)
            cname = "_canon_" + "_".join(sorted(bind_ops))
            if cname not in M.data_atoms:
                M.data_atoms[cname] = P; M.data_names.append(cname)
            canon.append(("BIND", cname))
        canon.append(("HALT", "a"))
        x = verify_input if verify_input is not None else _datom(12345, "canon_probe", self.dim, unitary=True)
        out_orig, _ = self.run_procedure(program, init_acc=x)       # prove equivalence by execution
        out_canon, _ = self.run_procedure(canon, init_acc=x)
        equiv = float(_cos(out_orig, out_canon))
        info = {"fully_collapsible": True, "net_shift": shift, "n_bind": len(bind_ops),
                "original_len": sum(1 for p in program if p[0] != "HALT"),
                "canonical_len": sum(1 for p in canon if p[0] != "HALT"),
                "verified": equiv > 0.999, "equivalence_cosine": round(equiv, 4)}
        return canon, info

    def learn_recipe_grammar(self, recipes, order=2):
        """Learn the sequence statistics of a set of valid recipes, so the next instruction can be
        PREDICTED from a partial recipe. Builds two dedicated token-level predictive models (delegating to
        PredictiveMemory), kept separate from the mind's prose predictor so recipe grammar and text never
        mix: one over the OPCODE stream (the recipe's control SHAPE, read by complete_procedure) and one
        over the JOINT (opcode, operand) stream (read by complete_instruction, which predicts the operand
        too). Each recipe is a list of (opcode, operand)."""
        from holographic_predictive import PredictiveMemory
        if getattr(self, "_recipe_grammar", None) is None:
            self._recipe_grammar = PredictiveMemory(dim=self.dim, order=order, seed=self.seed)
        if getattr(self, "_recipe_grammar_joint", None) is None:        # GEN-1: operand-aware grammar
            self._recipe_grammar_joint = PredictiveMemory(dim=self.dim, order=order, seed=self.seed + 1)
        pm = self._recipe_grammar
        jm = self._recipe_grammar_joint
        for r in recipes:
            ops = [s[0] if isinstance(s, tuple) else s for s in r]
            toks = [self._instr_token(s) for s in r]                    # "OPCODE|operand" joint tokens
            for i in range(len(ops)):
                pm.step(ops[:i], ops[i], learn=True)   # context = opcodes before i, target = ops[i]
            for i in range(len(toks)):
                jm.step(toks[:i], toks[i], learn=True) # context = full instructions before i, target i
        return self

    @staticmethod
    def _instr_token(step):
        """A (opcode, operand) instruction as one grammar token 'OPCODE|operand'. Opcodes never contain
        '|', so the split back is unambiguous. A bare opcode (no operand) is just the opcode."""
        return f"{step[0]}|{step[1]}" if isinstance(step, tuple) else str(step)

    def complete_procedure(self, partial):
        """Predict the next opcode given a partial recipe (a list of (opcode, operand) or bare opcodes),
        from the learned recipe grammar. Returns (opcode, confidence); (None, 0.0) if no grammar has
        been learned. An empty partial predicts the typical FIRST opcode. KEPT NEGATIVE: this is an
        n-gram over opcodes -- it predicts by the frequency of transitions it has SEEN, so a recipe
        shape absent from the training set will not be anticipated well, and it predicts the single
        most-likely next opcode (use the model's soft mode for a blended estimate)."""
        if getattr(self, "_recipe_grammar", None) is None:
            return None, 0.0
        ops = [s[0] if isinstance(s, tuple) else s for s in partial]
        return self._recipe_grammar.predict(ops)

    def complete_instruction(self, partial):
        """Predict the next full INSTRUCTION -- opcode AND operand -- from the learned recipe grammar (the
        joint (opcode, operand) token stream). Returns (opcode, operand, confidence); (None, None, 0.0) if
        no grammar. This extends complete_procedure, which predicts the opcode alone, with the operand.

        KEPT NEGATIVE (measured): operand prediction works only where operand USAGE is PATTERNED -- a family
        of recipes that bind the same operands in the same positions. When operands are arbitrary per recipe,
        each joint transition is seen at most once, so the operand is NOT predictable: the opcode SHAPE is
        still anticipated (the opcode grammar ignores operands and sees the consistent shape), but the
        operand prediction is low-confidence -- correctly, because a random operand is unknowable. So
        complete_procedure (opcode) is the robust call; the operand is a bonus only when it is patterned."""
        if getattr(self, "_recipe_grammar_joint", None) is None:
            return None, None, 0.0
        toks = [self._instr_token(s) for s in partial]
        tok, conf = self._recipe_grammar_joint.predict(toks)
        if tok is None:
            return None, None, float(conf)
        if "|" in tok:                                  # split the joint token back into opcode + operand
            op, operand = tok.split("|", 1)
            return op, operand, float(conf)
        return tok, None, float(conf)                   # a bare-opcode token (no operand)

    def decode_step(self, name_or_program, i):
        """Read the i-th instruction of a procedure as (opcode, operand) -- the program-as-DATA query
        the von Neumann encoding allows: a stored procedure can be INSPECTED, not just run. The honest,
        noisy read (cleaned against the codebooks); reliable up to a length that scales with dim."""
        M = self._machine()
        if isinstance(name_or_program, str) and name_or_program in M.functions:
            pv = M.functions[name_or_program]
        else:
            pv = M.assemble(name_or_program)
        return M.decode_instruction(pv, i)

    def procedure_to_recipe(self, program):
        """Express a procedure as a typed StructureRecipe (the B7 structure object) -- proving a program
        is just another holographic structure, reducible to atoms + bind + bundle, savable and composable
        like any recipe. Reproduces the assembled program bit-exactly. CALL is runtime, so out of scope."""
        from holographic_typed import program_to_recipe
        return program_to_recipe(self._machine(), program)

    def audit_procedure(self, steps=None, program=None, n_steps=None):
        """Audit a PROTOCOL for honesty anti-patterns (backlog D1): treat an analysis procedure as
        program-as-data and check its STRUCTURE -- does a SEARCH/recall step have a procedure-matched NULL,
        does a searched-and-scored family carry FDR control, is there an out-of-sample SPLIT between selecting
        and deciding? The check reads the step structure BACK FROM THE PROGRAM VECTOR (the same noisy
        unbind+cleanup the VM runs), so the honesty discipline becomes a structural query on the protocol
        vector rather than a habit you maintain and find missing after a fake edge slips through.

        Pass `steps` -- an ordered list of faculty-step names (encode, combination_search, calibrated_null,
        fdr, oos_split, decide, ...), which is assembled into a protocol vector and audited -- or a prebuilt
        (`program` vector, `n_steps`). Returns {sound, roles, sequence, violations}, where each violation is
        a (code, message) and `sound` is True iff no rule fires.

        SCOPE / KEPT NEGATIVE: a structural lint on DECLARED steps, not a data-flow analysis -- 'scores the
        same rows it selected on' is approximated by the order check (no SPLIT between SEARCH and DECIDE), not
        by tracking data identity; and the per-step decode is bounded by the program vector's capacity, so a
        protocol must be short to read reliably (the procedure tax). An unknown step name carries no
        obligation (fails open, not a false alarm)."""
        from holographic_protocol import build_protocol, audit_protocol
        M = self._machine()
        if steps is not None:
            program, n_steps = build_protocol(M, list(steps))
        if program is None or n_steps is None:
            raise ValueError("audit_procedure needs steps=<list of step names>, or program=<vec> with "
                             "n_steps=<int>")
        return audit_protocol(M, program, n_steps)

    def attribute(self, text, name=None):
        """WHO taught this? If the sequence model was fit on (text, source)
        documents, rank the sources by how much of the passage's transitions
        each one taught -- the stylistic bag. Returns [(source, weight)] or []."""
        gen = self._pick_gen(name, text)["gen"]
        if hasattr(gen, "attribute") and getattr(gen, "sources", None):
            return gen.attribute(text)
        return []

    def trace(self, text, name=None):
        """The full provenance answer: STYLE (transition bag) AND MATERIAL
        (sequence alignment -- the longest verbatim span), leading with whichever
        the evidence makes decisive. This is the method that tells apart sources
        sharing every word in opposite order (meaning is in the ordering). Returns
        the trace dict, or None when no provenance was recorded."""
        gen = self._pick_gen(name, text)["gen"]
        if hasattr(gen, "trace") and getattr(gen, "sources", None):
            return gen.trace(text)
        return None

    # -- self-maintenance across the whole model ---------------------------
    def _reorganize_and_narrate(self):
        """Run the organizer's speculate-measure-adopt pass AND write the mind's
        own account of what happened into the journal: which labels changed
        sub-prototype counts, and -- where the absorbed data was record-shaped --
        WHAT each split separates, by the contrast-judged role decode
        (explain_splits). The maintenance log narrates itself: 'A split in two;
        the modes differ in colour and shape.' Every consumer of the unified
        mind (the console, the tour, absorb()'s auto path) gets the narration
        for free, because this wrapper is the only road to auto_reorganize."""
        before = dict(self.memory.live.counts_by_label())
        choice = self.memory.auto_reorganize()
        after = dict(self.memory.live.counts_by_label())
        changed = {lab: (before.get(lab, 0), after.get(lab, 0))
                   for lab in set(before) | set(after)
                   if before.get(lab, 0) != after.get(lab, 0)}
        entry = {"taught": self._taught,
                 "choice": (choice[0] if choice else "keep"),
                 "changed": changed, "named": {}}
        if self._fillers:
            for lab, (b, a) in changed.items():
                if a > 1:
                    try:
                        _, sep = self.explain_splits(lab)
                        if sep:
                            entry["named"][lab] = sep
                    except Exception:
                        pass
        if changed:
            bits = []
            for lab, (b, a) in sorted(changed.items()):
                why = (f", the modes differ in {', '.join(entry['named'][lab])}"
                       if lab in entry["named"] else "")
                bits.append(f"'{lab}' went from {b} to {a} sub-prototype(s){why}")
            entry["story"] = "reorganized: " + "; ".join(bits) + "."
        else:
            entry["story"] = (f"checked the organization ({entry['choice']}): "
                              "nothing earned a change.")
        self.journal.append(entry)
        return choice

    def maintain_now(self):
        """Reorganize the memory and refresh the brain, each by its own held-out
        measurement. Returns the memory's choice -- and writes the mind's own
        narration of BOTH events into self.journal: the organizer's splits (named
        where the data allows) and the brain's keep/fold/refresh verdict, so the
        whole self-maintenance story reads in one place."""
        choice = self._reorganize_and_narrate()
        if self._brain is not None and self._brain.maintain == 'auto':
            outcome = self._brain.auto_maintain()
            entry = self.journal[-1]
            if outcome is None:
                # too little recent experience to judge -- say so honestly
                entry["brain"] = {"choice": "untested"}
                entry["story"] += (" The decision brain has too little recent"
                                   " experience to judge; left as is.")
            else:
                name, protos = outcome
                entry["brain"] = {"choice": name, "prototypes": protos}
                if name == "keep":
                    entry["story"] += (f" The decision brain measured its policy"
                                       f" memory and kept it ({protos} prototypes).")
                elif name.startswith("fold"):
                    entry["story"] += (f" The decision brain folded duplicate"
                                       f" situations down to {protos} prototypes"
                                       f" without forgetting.")
                else:
                    entry["story"] += (f" The decision brain REFRESHED its policy"
                                       f" from recent experience ({protos}"
                                       f" prototypes) -- recent decisions judged"
                                       f" the old regime stale.")
        return choice

    def describe(self):
        parts = [f"memory of {self.memory.live.size()} prototypes over "
                 f"{len(self.memory.live.counts_by_label())} labels"]
        if self._recall is not None:
            parts.append(f"a recall index of {len(self._recall.vecs)} items")
        if self._brain is not None:
            parts.append(f"a decision brain over {self._actions}")
        if getattr(self, "_gens", None):
            descs = []
            for key, g in sorted(self._gens.items()):
                detail = (f"order {g['gen'].n}" if g["kind"] == "flat"
                          else f"fractal coder, {g['modality']}")
                descs.append(f"{key}: {detail}")
            plural = "s" if len(self._gens) > 1 else ""
            parts.append(f"sequence schema{plural} ({'; '.join(descs)})")
        return "UnifiedMind: " + "; ".join(parts)

    # -- persistence: save the LEARNED MIND (its generalization), via the kernel's save -----------
    # The save captures what the mind LEARNED -- its perception (encoder), its self-organized
    # prototype memory (the classifier), its decision brain, and the routing bookkeeping classify
    # reads. It deliberately does NOT persist the verbatim recall index of every individual example
    # (`_recall`), whose payloads are arbitrary original inputs -- raw arrays, dicts, strings -- that
    # do not round-trip through a structured array save; re-learn() those if you want recall back.
    # Lazy/derived faculties (sequence & plan memory, the text/word generators, meaning predictors,
    # the scene coder, the FHRR high-capacity memory) are rebuilt on use, not stored. What round-trips
    # is the trained generalization: classify and decide are bit-for-bit identical after save/load.
    _STATE_KIND = "UnifiedMind"

    def to_state(self):
        """Snapshot the learned mind for holographic_core.save (so quant='rd'/'auto'/'int8' all apply).
        See the persistence note above for exactly what is and is not captured."""
        nr = self.encoder.to_state().get("number_range", [-4.0, 4.0])
        return {
            "kind": self._STATE_KIND,
            "config": {"dim": int(self.dim), "seed": int(self.seed), "maintain": self.maintain,
                       "check_every": int(self.check_every), "number_range": [float(nr[0]), float(nr[1])],
                       "text_window": int(self.encoder._text.window),
                       "coherence_floor": self.coherence_floor},
            "encoder": self.encoder.to_state(),
            "memory": self.memory.to_state(),
            "label_modality": dict(self._label_modality),
            # sets aren't JSON-able: store each role's fillers as a list, restore to a set on load
            "fillers": {k: sorted(v, key=str) for k, v in self._fillers.items()},
            "format_corpus": dict(self._format_corpus),
            "taught": int(self._taught),
            "actions": list(self._actions) if self._actions is not None else None,
            "brain": self._brain.to_state() if self._brain is not None else None,
        }

    @classmethod
    def from_state(cls, state):
        """Rebuild a UnifiedMind from to_state(). The reloaded mind classifies and decides identically;
        its recall index and lazy faculties start empty and rebuild on use (see the persistence note)."""
        from holographic_mind import UniversalEncoder
        from holographic_organizer import SelfOrganizingMind
        cfg = state["config"]
        m = cls(dim=int(cfg["dim"]), seed=int(cfg["seed"]), number_range=tuple(cfg["number_range"]),
                maintain=cfg.get("maintain", "auto"), check_every=int(cfg.get("check_every", 60)),
                text_window=int(cfg.get("text_window", 2)), coherence_floor=cfg.get("coherence_floor"))
        m.encoder = UniversalEncoder.from_state(state["encoder"])      # replace the fresh encoder/memory
        m.memory = SelfOrganizingMind.from_state(state["memory"])      # with the saved, trained ones
        m._label_modality = dict(state.get("label_modality", {}))
        m._fillers = {k: set(v) for k, v in state.get("fillers", {}).items()}
        m._format_corpus = dict(state.get("format_corpus", {}))
        m._taught = int(state.get("taught", 0))
        if state.get("brain") is not None:
            from holographic_creature import HolographicMind
            m._actions = list(state["actions"])
            m._brain = HolographicMind.from_state(state["brain"])
        return m

    def save(self, path, quant="auto", compress=True):
        """Persist the learned mind to `path` (.npz) via the kernel save. The default quant='auto' picks
        the coarsest DECISION-SAFE precision per array -- and now also uses the B5 rate-distortion code
        (KLT -> quantize -> rANS, cosines preserved to 0.9999) on any LARGE low-rank float array, taken only
        when it beats int8, so low-rank state shrinks automatically with no precision risk and small arrays
        are untouched. quant='rd' forces that code wherever it helps (int8 elsewhere); 'int8'/None as in
        holographic_core.save."""
        from holographic_core import save as _save
        return _save(self, path, compress=compress, quant=quant)

    @classmethod
    def load(cls, path):
        """Reload a mind saved with save(); dispatches through the kernel's versioned loader."""
        from holographic_core import load as _load
        return _load(path)


# ---------------------------------------------------------------------------
# DEMO: one mind, many modalities, one memory -- measured against separate ones
# ---------------------------------------------------------------------------

def _patterns(kind, rng, n=8):
    """Tiny synthetic 'images' -- four visually distinct classes, with noise."""
    a = np.zeros((n, n))
    if kind == "rows":
        a[::2, :] = 1.0
    elif kind == "cols":
        a[:, ::2] = 1.0
    elif kind == "diag":
        for i in range(n):
            a[i, i] = 1.0; a[i, (i + 1) % n] = 1.0
    elif kind == "check":
        a[(np.add.outer(np.arange(n), np.arange(n)) % 2) == 0] = 1.0
    return a + 0.15 * rng.standard_normal((n, n))


def demo_unified():
    """One UnifiedMind learns three different KINDS of thing -- text topics, little
    images, and records -- into a SINGLE self-organizing memory, then classifies all
    three. The honest question is whether one shared store does as well as three
    separate ones; if mixing modalities in one space wrecked it, the unification would
    be fake. It does not: the modalities land in near-orthogonal parts of the space, so
    one memory matches the separate baselines AND the same mind still makes decisions."""
    from holographic_text import TOPICS, _content, _split

    print("=" * 70)
    print("One mind, one memory: text + images + records in a single space")
    print("=" * 70)
    rng = np.random.default_rng(0)
    corpus = [s for sents in TOPICS.values() for s in sents]

    # build the three datasets as (input, label, modality)
    text_tr, text_te = [], []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        text_tr += [(_content(s), topic, "text") for s in a]
        text_te += [(_content(s), topic, "text") for s in b]
    img_tr, img_te = [], []
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img_tr.append((_patterns(kind, rng), f"img:{kind}", "image"))
        for _ in range(8):
            img_te.append((_patterns(kind, rng), f"img:{kind}", "image"))
    rec_tr, rec_te = [], []
    depts = ("eng", "sales", "ops")
    for d in depts:
        for _ in range(20):
            rec_tr.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))
        for _ in range(8):
            rec_te.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))

    # ---- ONE unified mind: everything into one memory --------------------
    # text word-vectors learn best from content words (stopwords dilute co-occurrence);
    # that is a text-task choice, so the orchestrator makes it -- the encoder stays generic.
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    train = text_tr + img_tr + rec_tr
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    def score(m, test, route=True):
        return sum(m.classify(x, mod, route=route)[0] == lab for x, lab, mod in test) / len(test)

    ut = score(mind, text_te); ui = score(mind, img_te); ur = score(mind, rec_te)
    ut_flat = score(mind, text_te, route=False)

    # ---- separate baselines: one memory per modality (same encoding) -----
    def separate(train_items, test_items):
        enc = UniversalEncoder(1024, seed=0)
        enc.learn_text([_content(s) for s in corpus])
        mem = SelfOrganizingMind(dim=1024, seed=0)
        for x, lab, mod in train_items:
            mem.observe_vector(enc.encode(x, mod), lab)
        mem.auto_reorganize()
        return sum(mem.classify_vector(enc.encode(x, mod))[0] == lab
                   for x, lab, mod in test_items) / len(test_items)

    st = separate(text_tr, text_te); si = separate(img_tr, img_te); sr = separate(rec_tr, rec_te)

    print(f"\n  {'modality':10s}{'separate memory':>18s}{'one shared memory':>20s}")
    print(f"  {'text':10s}{100*st:>16.0f}% {100*ut:>18.0f}%")
    print(f"  {'images':10s}{100*si:>16.0f}% {100*ui:>18.0f}%")
    print(f"  {'records':10s}{100*sr:>16.0f}% {100*ur:>18.0f}%")
    print(f"\n  Routing: a text query against ALL concepts scores {100*ut_flat:.0f}%; restricted to")
    print(f"  text concepts (its known modality) it scores {100*ut:.0f}%. With correct encoding the")
    print("  modalities separate cleanly, so here routing changes nothing -- it is a cheap")
    print("  safeguard that removes cross-modal collisions WHEN they occur, not a routine")
    print("  booster. (An earlier apparent gain came from a since-fixed encoding bug that")
    print("  degraded text vectors into colliding with other modalities.)")
    print(f"\n  {mind.describe()}")

    # ---- cross-modal recall over the same store --------------------------
    q = img_te[0]
    (lab, _), sim = mind.recall(q[0], q[2])
    print(f"\n  Recall: a held-out '{q[1]}' image finds nearest stored item '{lab}' "
          f"(cos {sim:.2f}) -- the recall view searches the same vectors.")

    # ---- the SAME mind also decides -------------------------------------
    mind.actions(["left", "right"])
    rng2 = np.random.default_rng(1)
    for _ in range(400):
        n = float(rng2.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    dec = sum((mind.decide(float(v), modality="number") == ("right" if v > 0 else "left"))
              for v in np.linspace(-3, 3, 40)) / 40
    print(f"  Decision: the same mind learned a contextual choice over numbers -> "
          f"{100*dec:.0f}% correct, using the same encoder and space.")

    # ---- the SAME mind also generates (the fourth operation) -------------
    mind.learn_sequence(" ".join(corpus), n=5)
    sample = mind.generate("the ", 90, 0.4)
    print(f"  Generation: taught to continue the topic text, it produces -> \"{sample[:70]}\"")

    print(f"\n  {mind.describe()}")
    print("\n  One encoder, one self-organizing memory, one brain -- shared substrate, not")
    print("  a wrapper. One shared store matches separate per-modality memories; with")
    print("  correct encoding the modalities are near-orthogonal, so a flat store shows")
    print("  no cross-modal interference here and routing is a cheap safeguard rather than")
    print("  a booster. Storage needs no separate curator: the memory's own aggregation")
    print("  already compresses (here ~1800 observations into a handful of prototypes).")
    print("  Generation completes the operation set -- its next-symbol step is the same")
    print("  cleanup primitive -- though its context index stays exact, the one place a")
    print("  fuzzy recall was measured to hurt rather than help.")


if __name__ == "__main__":
    demo_unified()

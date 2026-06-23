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
                 check_every=60, text_window=2):
        self.dim = dim
        self.seed = seed                   # remembered for owned faculties (scene, morph)
        self.maintain = maintain
        self.check_every = check_every
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
        if self.maintain == 'auto' and self._taught % self.check_every == 0:
            self._reorganize_and_narrate()
        return self

    def classify(self, x, modality=None, route=True):
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
          down."""
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

    def recall(self, x, modality=None):
        """Nearest stored individual. The index does an exact scan until the store is
        genuinely big, then switches to the recursive HoloForest (the crossover is
        measured -- see _Index.recall). A NEGATIVE worth recording here: wiring the
        learned adaptive navigator (holographic_navigator) into this path was tried
        and lost badly on the mind's own store -- 48% recall@1 at ~130 comparisons,
        where the forest at beam 2 gets 89% within ~512. The navigator's margin
        senses were tuned on UNIFORM random vectors; the unified store is clustered
        (many near-duplicates per class), which miscalibrates the arrive/keep-moving
        instinct. So recall keeps the dumb-but-honest index, and the navigator stays
        a study of adaptive access, not a default."""
        if self._recall is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        return self._recall.recall(self.perceive(x, modality))

    # -- one decision brain, on the same substrate -------------------------
    def actions(self, names):
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain)
        return self

    def decide(self, state, explore=False, epsilon=None, modality=None,
               senses=None, avoid=("danger", "wall")):
        """Decide an action. `senses`/`avoid` pass straight through to the
        brain's built-in safety reflexes (HolographicMind.decide): hand over
        the current senses dict and moves into seen dangers or walls are
        vetoed below the value estimate -- the unified brain gets the same
        measured safety every other caller of the model gets."""
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        a = self._brain.decide(self.perceive(state, modality), explore=explore,
                               epsilon=epsilon, senses=senses, avoid=avoid)
        return self._actions[a]

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

    def factor_composite(self, composite, codebooks, restarts=20, L=None, iters=None, seed=0):
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
                                      iters=(50 if iters is None else iters), seed=seed)
            return {"factors": tuple(res["picks"]), "solved": bool(res["verified"]),
                    "verified": bool(res["verified"]), "present": res["present"],
                    "restarts": restarts, "search_space": space, "backend": "sbc"}

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

    def decompose_structure(self, composed, codebooks, L, restarts=6, iters=50, seed=None):
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
        given an `L` -- the de-siloing the integration review asked for: one factorizer, not two."""
        from holographic_sbc import decompose_structure as _decompose
        return _decompose(np.asarray(composed), codebooks, L, restarts=restarts, iters=iters,
                          seed=self.seed if seed is None else seed)

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
                rank=8, beta=25.0, steps=3, forward=None, adjoint=None, mu=0.5, pnp_steps=30):
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
          method='pnp'       : Plug-and-Play / RED restoration of a degraded measurement x = forward(clean)
                              + noise, using the adaptive manifold map as the prior (needs forward/adjoint).
          method='auto'      : codebook if a `codebook` is given, else adaptive manifold if `samples`
                              are given. NLM and PnP stay OPT-IN: deciding self-similar-vs-low-rank
                              automatically is itself a measurement we will not fake -- name them.

        A denoiser needs a PRIOR; a single vector with no manifold cannot be cleaned (no free lunch), so
        `samples` (clean rows) or `codebook` (atoms) is required for every method but 'nlm' (which uses
        `x`'s own redundancy). Returns the cleaned vector (or, for 'nlm', the cleaned (N, dim) set).

        KEPT NEGATIVES (the modules', surfaced not hidden): FIXED-rank projection over-smooths at low
        noise -- use 'adaptive', which is ~neutral there; manifold projection only helps where real
        low-rank structure exists (it destroys structureless signal); NLM only helps where near-duplicates
        exist."""
        from holographic_denoise import (fit_manifold, manifold_denoise, fit_manifold_full,
                                          adaptive_manifold_denoise, codebook_denoise,
                                          nlm_denoise, pnp_restore)
        x = np.asarray(x, float)

        if method == "auto":                          # pick by the prior you handed me, conservatively
            method = "codebook" if codebook is not None else ("adaptive" if samples is not None else None)
            if method is None:
                raise ValueError("denoise needs a prior: pass samples=<clean rows> or codebook=<atoms> "
                                 "(a denoiser is a map of a manifold; a lone vector has none)")

        if method == "nlm":                           # self-similarity: x IS the patch set to clean
            P = np.atleast_2d(x)
            return nlm_denoise(P, k=min(12, len(P)))

        if method == "codebook":
            if codebook is None:
                raise ValueError("method='codebook' needs codebook=<(n, dim) atoms>")
            return codebook_denoise(x, np.asarray(codebook, float), beta=beta, steps=steps)

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

    def solve_maze(self, world, steps=200, mu=1.5, dt=0.2):
        """Solve a GridWorld maze by the DETERMINISTIC Tero flow-conductance model (holographic_flow):
        Physarum-style tubes thicken with flux (Poiseuille conductance) until the network collapses onto
        the shortest path. Same (path, info) interface as the stochastic slime solver, but deterministic
        and ~100x faster, and it lands EXACTLY on the optimum on braided mazes. info reports
        reached / optimal / extracted_len / cells / deterministic. Returns (path, info)."""
        from holographic_flow import solve_maze_flow
        return solve_maze_flow(world, steps=steps, mu=mu, dt=dt)

    def assemble(self, target, library, frag_len=2, steps=300, mu=1.5, dt=0.2):
        """Assemble `target` from a `library` of overlapping fragments by MIN-ENERGY flow search
        (holographic_assembly) -- Rosetta-style fragment assembly (choose a fragment per position to
        minimise a placement energy, consecutive fragments overlap-agreeing) cast as the SAME min-cost-
        path flow the maze solver runs, on a (position x fragment) trellis. Returns a dict: the assembled
        string, its total energy, the chosen (position, fragment) list, and a B7 StructureRecipe binding
        each fragment to its position -- the assembly AS a typed holographic structure (realize() it,
        save() it). Built at this mind's dim/seed.

        It finds the GLOBAL optimum (it matches the exact Viterbi DP), not a greedy one. KEPT NEGATIVE:
        the energy is a placement-mismatch / Rosetta-score STAND-IN -- the combinatorial core, not a
        protein force field."""
        from holographic_assembly import assemble as _assemble
        return _assemble(target, library, frag_len=frag_len, steps=steps, mu=mu, dt=dt,
                         dim=self.dim, seed=self.seed)

    def learn_dynamics(self, states, ridge=1e-3):
        """Learn a fixed dynamics operator U so that state(t+1) ~ bind(U, state(t)) -- dynamics as an
        ALGEBRA OF BINDS (holographic_dynamics). In HRR's Fourier domain a learned bind is a per-frequency
        complex transfer, i.e. the Koopman/DMD operator in Fourier coordinates (the object Stam's FFT
        fluid step and Puckette's phase vocoder also manipulate). Returns a Propagator with .step(state)
        (one-step prediction = a SINGLE bind), .rollout(state, k), and .recall_at(state, k) -- recover the
        state k steps BEFORE one now, so the trajectory is content-addressable, not just forward-runnable.

        `states` is a sequence of state rows (T, dim). KEPT NEGATIVE on real market RETURNS: prediction
        only TIES a trivial mean predictor -- near-efficient-market returns have almost no linear structure
        for a fixed operator to exploit (the correct, expected result, kept on record). It shines on signals
        that DO have linear dynamics (audio, fluids, a bind-shaped control). The CONTENT-ADDRESSABLE
        round-trip (forward k then back k returns the start at cosine ~1.0) is the durable win regardless."""
        from holographic_dynamics import Propagator
        return Propagator.learn(np.asarray(states, float), ridge=ridge)

    # ---- the GENERATIVE faculties (integration plan, Tier 4) -----------------------------------
    # Generation is denoising run backwards, and a splat scene is a bundle -- so the last two modules
    # built beside the mind reconcile straight into it: generate a vector by the cleanup-attractor
    # diffusion, and represent a 2-D field as a superposition of Gaussian primitives.

    def generate_vector(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=None):
        """GENERATE a hypervector by denoising FROM PURE NOISE (B10) -- the cleanup attractor as a tiny
        holographic diffusion (holographic_hopfield.generate): start from a random unit vector, anneal
        beta UP (vague -> sharp) and injected noise DOWN across `steps`, and walk onto the codebook
        manifold. Generation and denoising are the SAME operation in different regimes -- this is the
        vector-level twin of the text generate(), pointed at the B10 diffusion sampler. Returns a unit
        vector, deterministic in `seed` (this mind's seed by default).

        KEPT NEGATIVE: over a BARE codebook this converges to a stored atom (a degenerate sampler) --
        feed it a COMPOSED or continuous manifold for novel-but-valid samples."""
        from holographic_hopfield import generate as _generate
        return _generate(np.asarray(codebook, float), steps=steps, beta0=beta0, beta1=beta1,
                         noise0=noise0, seed=self.seed if seed is None else seed)

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
        relation no bag store can answer (measured 100% on plans up to ~8
        steps)."""
        return self._seq_mem().precedes(name, a, b)

    def validate_plan(self, name_or_steps, constraints):
        """Check a plan against ordering rules -- the PB&J test: does every
        'a must come before b' hold? Returns (ok, violations); a violation
        names exactly which step is out of order. Works on a stored plan name
        or a fresh step list."""
        return self._seq_mem().validate(name_or_steps, constraints)

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
                       "text_window": int(self.encoder._text.window)},
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
                text_window=int(cfg.get("text_window", 2)))
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
        """Persist the learned mind to `path` (.npz) via the kernel save. quant='rd' uses the B5
        rate-distortion code on any low-rank float arrays (KLT -> quantize -> rANS, ~11x under int8),
        falling back to int8 where there is no low-rank structure, so it is never larger; 'auto' picks
        the coarsest decision-safe precision per array; 'int8'/None as in holographic_core.save."""
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

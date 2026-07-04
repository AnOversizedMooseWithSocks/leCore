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

    THE THREE MINDS -- one division of labour, so this never gets confusing again:
      - UnifiedMind     : THE ONE MIND  <<< THIS CLASS.  Every general, composable faculty lives here -- the
                          single encoder (perceive), the self-organising memory, recall/recognize, planning,
                          denoising, and the decision machinery. Everything is built on this; it depends on
                          nothing domain-specific. New capability that ANY mind could use belongs here.
      - CreatureMind    : a SPECIALIZED LAYER on this one mind (holographic_creature_mind.py) -- subclasses
                          UnifiedMind, inherits every faculty, and adds only domain wiring (sense/act/learn).
                          The reference demo of the pattern; build any new specialized mind the same way.
      - HolographicMind : the RL ENGINE (holographic_creature.py) -- a per-action prototype value memory +
                          greedy policy that THIS mind uses internally for value-learning (decide/reinforce
                          delegate to it). MEASURED to beat value-learning built on the unified memory
                          (exp_value_memory.py), so it is kept, not a remnant. It is NOT an agent-building
                          pattern: build agents from CreatureMind, never directly from the engine.

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

    def lookup(self, word):
        """Look a word up in the VENDORED DICTIONARY (~144k English words): its definition, part of speech, synonyms,
        an example, and its 'is a kind of' parent. This is real world-knowledge the engine carries with it -- distinct
        from define(), which returns nearest words by LEARNED meaning. Returns None for an unknown word.
        See holographic_dictionary."""
        import holographic_dictionary as _dict
        return _dict.entry(word)

    def word_taxonomy(self, word):
        """The 'what kind of thing is this?' chain from the vendored dictionary's taxonomy: e.g. 'dog' -> ['domestic
        animal', 'animal', 'organism', ... 'entity']. Contextual grounding straight from the dictionary's is_a
        hierarchy. See holographic_dictionary.hypernym_chain."""
        import holographic_dictionary as _dict
        return _dict.hypernym_chain(word)

    def dictionary_size(self):
        """How many words the vendored dictionary holds (and its source/license via .manifest())."""
        import holographic_dictionary as _dict
        return _dict.size()

    def learn_vocabulary(self, vocab, iters=3, alpha=0.7):
        """Bootstrap the mind's word meanings from the VENDORED DICTIONARY over a given vocabulary: builds the
        {word: [definition words]} map from real definitions and runs learn_dictionary on it. So instead of supplying
        your own dictionary, the mind can learn meaning from the batteries-included one. Returns self."""
        import holographic_dictionary as _dict
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

    def hypervector(self, x, modality=None, tag=None):
        """The same encode, but returned as a first-class Hypervector (consolidation D1): the raw array plus its
        dim / encoder / tag, with the five verbs (bind/unbind/bundle/cleanup/permute) as methods. The encoder is the
        'constructor'; the raw array stays one attribute away (.array / np.asarray(hv)). See holographic_hypervector.
        """
        from holographic_hypervector import Hypervector
        return Hypervector(self.encoder.encode(x, modality), encoder=self.encoder,
                           tag=tag if tag is not None else repr(x)[:40])

    def adaptive_record(self, expected_pairs, exact=False, max_numbers=None, seed=0):
        """A role->filler memory whose representation is GATED BY LOAD and FIDELITY NEED (the FHRR/tensor re-enables):
        cheap real-HRR at low load, FHRR phasors past the capacity knee, or tensor-product binding for EXACT recall
        (perfect to M~dim, at dim*dim storage) when `exact=True` and the D*D budget fits `max_numbers`. Uniform
        add/recall. The deciders are exact integers/flags, and there is no harm mode on recall (FHRR >= real-HRR,
        tensor is exact in-regime), so the gate just avoids paying for capacity until it's worth it. See
        holographic_loadmemory."""
        from holographic_loadmemory import AdaptiveRoleFillerMemory
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
        from holographic_ai import cosine
        return cosine(self.perceive(a, "axial"), self.perceive(b, "axial"))

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

    def spectral_bandwidth(self, x, energy_fraction=0.95):
        """The BANDWIDTH a signal occupies (holographic_bandwidth) -- the fraction of Nyquist (in [0,1]) holding
        `energy_fraction` of the spectral energy: small for band-limited content, near 1 for broadband noise. The
        number that drives a band-limited encoder's bandwidth knob. Complements the shipped fractal_dimension (which
        says how rough) with how much spectrum to keep. Kept negative: it is an ENERGY rolloff, so a fractal's
        front-loaded 1/f^b energy can read a small bandwidth even though its self-similar detail extends higher --
        honest about fidelity-for-a-budget, not lossless bandwidth."""
        from holographic_bandwidth import spectral_bandwidth
        return spectral_bandwidth(x, energy_fraction=energy_fraction)

    def fractal_confidence(self, x, tol=0.15):
        """A singularity CROSS-CHECK for a 1-D signal's fractal dimension (holographic_bandwidth): two independent
        slope estimators -- the power-spectrum slope D=(5-gamma)/2 and an increment-variance estimator -- and
        whether they AGREE. The shipped fractal_dimension is a single estimator and silently returns a wrong number
        for a step or a pure tone; this flags those (agree=False). Returns (d_spectral, d_increment, agree). Trust a
        fractal dimension only when agree is True. (R/S Hurst is deliberately NOT a co-validator here -- it weights
        coarse scales differently and disagrees even on clean fBm.)"""
        from holographic_bandwidth import fractal_confidence
        return fractal_confidence(x, tol=tol)

    def density_estimate(self, samples, lo, hi, query, dim=1024, seed=None, method="lcv", bandwidth=None):
        """Kernel DENSITY ESTIMATE via the encoder (holographic_kde): bundle the encoded samples, then density(x) ~
        bundle . encode(x) = (1/n) sum K(x - s_i), with the RBF kernel bandwidth AUTO-SELECTED (the band-limit
        matched to the data) and the output normalized to integrate to ~1 over [lo,hi]. method='lcv' (leave-one-out
        likelihood, robust -- beats a fixed default ~5-7x by matching the kernel to the data) or 'silverman' (cheap,
        over-smooths multimodal). Returns (density_at_query, chosen_bandwidth). The disciplined form of the
        band-limited-encoding faculty: the encoder's documented RBF-as-KDE use, where the bandwidth IS the
        band-limit. Kept negatives: the sinc kernel's bandwidth is NOT tunable (only RBF); bandwidth selection fixes
        smoothing, not capacity (a too-small dim cannot be rescued)."""
        from holographic_kde import density_estimate
        return density_estimate(samples, lo, hi, query, dim=dim, seed=self.seed if seed is None else seed,
                                method=method, bandwidth=bandwidth)

    def kde_bandwidth(self, samples, lo, hi, method="lcv"):
        """The RBF kernel bandwidth for a density estimate over [lo,hi] (holographic_kde): 'lcv' (leave-one-out
        likelihood, robust, matches the data's structure) or 'silverman' (rule of thumb, over-smooths multimodal).
        The band-limit auto-matched to the data. Returns a float."""
        from holographic_kde import kde_bandwidth
        return kde_bandwidth(samples, lo, hi, method=method)

    def spectral_flatness(self, v):
        """SPECTRAL FLATNESS of a vector (holographic_flatness): the Wiener entropy of its power spectrum (geometric
        mean / arithmetic mean), in (0,1]. 1.0 = a perfectly flat spectrum = a UNITARY, distortion-free binding key;
        lower = peakier = more lossy as a key. The diagnostic for "is this safe to bind/unbind repeatedly?" --
        unbind(bind(x,k),k) returns x convolved with |K|^2, which is x only when |K|=1 everywhere (flatness 1).
        Returns a float."""
        from holographic_flatness import spectral_flatness
        return spectral_flatness(v)

    def binding_stability(self, v, tol=0.05):
        """BINDING-STABILITY regime diagnostic for a key (holographic_flatness): {'flatness', 'distortion',
        'stable'}, where distortion is the measured single-round bind/unbind error and 'stable' iff it is within tol
        (effectively unitary). Spectral flatness PREDICTS distortion monotonically. The band-limit-preservation
        regime test grounded in the real bind: the engine already mints the stable (unitary) regime via
        unitary_vector; this measures where any vector sits on it. Kept finding: the Trefethen transient-growth
        concern does not materialize -- linear ops preserve a white spectrum and the cleanup contracts monotonically;
        the real stability axis is this linear flatness."""
        from holographic_flatness import binding_stability
        return binding_stability(v, tol=tol)

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

    def combine_estimators(self, pairs, power=1.0):
        """Veach's balance/power heuristic (MIS): combine several estimators of the SAME quantity, weighting
        each by its per-query RELIABILITY (the 'pdf' analog). `pairs` is a list of (estimate, reliability);
        returns sum_i w_i * estimate_i with w_i = r_i**power / sum_j r_j**power (power=1 = the balance
        heuristic, 2 = the power heuristic). The reusable combiner behind mis_recover -- the point being that
        this BEATS a naive average, which instead carries each estimator's error into the other's weak regime
        (holographic_mis)."""
        from holographic_mis import combine_estimators
        return combine_estimators(pairs, power=power)

    def mis_recover(self, x, codebook, beta=10.0, power=1.0):
        """Recover a vector by combining HARD 1-NN and SOFT (dense-Hopfield) cleanups per-query via the balance
        heuristic -- the B1 kept-negative ('hard wins on discrete atoms, soft wins on continuous off-grid
        values') turned into one combiner that needs NO regime label. The cosine distribution's peakiness is
        the reliability: a sharp single winner trusts the exact atom; a close runner-up trusts the interpolating
        soft blend. `codebook` is the matrix of atoms to recover against. Measured to beat naive averaging
        always, and both singles in the crossover regime where neither estimator dominates (holographic_mis)."""
        from holographic_mis import mis_recover
        return mis_recover(np.asarray(x, float), np.asarray(codebook, float), beta=beta, power=power)

    def gradient_cache(self, anchors, values, jacobians):
        """Package sparse anchors with cached values AND local Jacobians for first-order (Ward irradiance-
        gradient) decode (CACHE-1). `anchors` (N,d), `values` (N,) or (N,M), `jacobians` (N,d) or (N,M,d). Read
        back with cache_interp -- the gradient lets each anchor cover more ground, ~halving the anchor count a
        smooth decode needs vs a nearest-neighbour grid argmax. (Use holographic_cache.gradient_cache_fd to
        build from a field function alone, with finite-difference Jacobians.)"""
        from holographic_cache import gradient_cache
        return gradient_cache(anchors, values, jacobians)

    def cache_interp(self, cache, q, validity_radius, global_weights=False):
        """Read a gradient cache at query `q` by first-order interpolation with a VALIDITY-RADIUS locality guard
        (CACHE-1): each anchor within the radius extrapolates its linear model v_i + J_i.(q-a_i), blended by
        1/distance. The guard is load-bearing -- global_weights=True removes it and a distant anchor dumps a bad
        long-range extrapolation into the query (measured ~2.7x worse); kept callable so the failure is visible."""
        from holographic_cache import interp_first_order
        return interp_first_order(cache, q, validity_radius, global_weights=global_weights)

    def optimize(self, loss, x0, grad=None, steps=200, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                 tol=0.0, patience=10, min_steps=20, fd_eps=1e-5, stats=None):
        """GENERAL GRADIENT-DESCENT OPTIMIZER (holographic_optimize, GRAD-2) -- minimize any scalar `loss(x)` from
        `x0` by Adam (the exact bias-corrected update the splat fit uses, generalized). Supply `grad(x)` for the
        analytic gradient (fast); omit it for the finite-difference fallback (2*n loss evaluations per step). This
        promotes the 3DGS work's gradient machinery -- the splat fit's hand-derived-gradient Adam, plus the cache
        module's finite differences -- to a first-class engine capability: gradients on the fly, with no autodiff
        (the NumPy-only rule). `tol > 0` enables convergence-gated early stop; pass stats={} for steps and the loss
        trajectory. Underpins IHT recovery (GRAD-1) and any fit/alignment problem. Kept negative: no autodiff (FD is
        the general fallback and costs 2*n evals/step); gradient descent finds a LOCAL minimum on a non-convex loss."""
        from holographic_optimize import optimize
        return optimize(loss, x0, grad=grad, steps=steps, lr=lr, b1=b1, b2=b2, eps=eps,
                        tol=tol, patience=patience, min_steps=min_steps, fd_eps=fd_eps, stats=stats)

    def fd_gradient(self, f, x, eps=1e-5):
        """The central finite-difference gradient of a scalar function f: R^n -> R at x (holographic_optimize,
        GRAD-2) -- 2*n evaluations, perturbing a copy one coordinate at a time. The general scalar-loss companion to
        the cache module's field-map finite differences; the 'gradient' for optimize() when no analytic one exists."""
        from holographic_optimize import fd_gradient
        return fd_gradient(f, x, eps=eps)

    def adaptive_anchors(self, x, y, n, floor=0.05, power=0.5):
        """Place n cache/codebook anchor positions along `x` so they crowd where the field `y` bends (high
        curvature) and thin out where it is flat -- irradiance caching's adaptive record density instead of a
        uniform grid (CACHE-3). Density ~ |y''|^power equidistributes the piecewise-linear reconstruction error, so
        this matches uniform-placement quality at MATERIALLY fewer anchors (~7x on a non-uniformly-smooth field).
        Scope kept honest: on a UNIFORMLY-smooth field there is no concentration to exploit, so it ~ties uniform --
        the win is quality MOVED to where the field needs it, not free quality. Returns the anchor x-positions."""
        from holographic_adaptive_cache import adaptive_anchors
        return adaptive_anchors(x, y, n, floor=floor, power=power)

    def reconstruct_from_anchors(self, x, anchor_x, y):
        """Piecewise-linear reconstruction of field `y` (sampled at `x`) from its values at `anchor_x` (CACHE-3) --
        the cache read paired with adaptive_anchors: sample at the anchors, then interpolate between them."""
        from holographic_adaptive_cache import reconstruct_from_anchors
        return reconstruct_from_anchors(x, anchor_x, y)

    def robust_accumulate(self, samples, schedule="harmonic", alpha=0.2, clamp_k=None):
        """Average noisy estimates of one quantity robustly, for the engine's averaging paths (consolidation over
        a growing store, forest vote-averaging). schedule='harmonic' uses 1/n weights (ACCUM-2: converges, best
        for a STATIONARY target; 'ema' tracks a DRIFTING target but plateaus; 'mean' is the plain mean). clamp_k
        (ACCUM-3), if set, winsorizes outlier samples to clamp_k robust-scales from the median first, so one
        firefly can't dominate -- measured ~100x lower error under outliers, with no loss on clean data."""
        from holographic_accumulate import robust_accumulate
        return robust_accumulate(samples, schedule=schedule, alpha=alpha, clamp_k=clamp_k)

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

    def conformance_report(self, dim=64, seed=0):
        """Run the ISA conformance suite (ISA-2): check every production base instruction against its
        definitional reference implementation, per the contract in ISA.md. Returns {op: {'passed', 'class', 
        'max_diff'}} where class is 'TOL' (a continuous output, conformant within numeric tolerance) or 'EXACT'
        (a decision / exact reindex, conformant bit-for-bit). The kernel is conformant iff every op passes -- and
        this is what makes a vectorized op safe to adopt: it is 'conformant' iff it passes here. The bind_batch
        class (a value-conformant change that flips a decision) is caught because decisions are pinned
        separately and exactly (see test_isa_conformance.py)."""
        from holographic_reference import run_conformance
        return run_conformance(dim=dim, seed=seed)

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
    def actions(self, names, robust_returns=False, value_backend="table"):
        """Declare the creature brain's action set. `robust_returns=True` (D2, opt-in) winsorises outlier rewards
        in each prototype's running-mean value: a fluke reward (a jackpot, a sensor glitch) is clamped to a few
        robust-scales before it folds in, so it cannot swing the value estimate -- measured ~3x lower value error
        under outlier rewards, no cost on clean data. Off by default (the plain running average).

        `value_backend` picks the brain's value/policy representation: 'table' (default, the prototype memory),
        'holo' (the two-bundle hypervector policy), or 'routed' (the hypervector policy with the routing fabric
        pushing the capacity cliff back). With 'holo'/'routed' the whole brain runs on a fixed-size, savable,
        composable hypervector policy -- so anywhere the creature is used in the holographic space, the
        holographic creature can be used instead (decide/reinforce are unchanged)."""
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain, robust_returns=robust_returns,
                                      value_backend=value_backend)
        return self

    def use_holographic_brain(self, routed=False):
        """Swap the (already-declared) creature brain to a holographic backend in place -- 'routed' for the
        cliff-pushed-back routing variant, else the plain two-bundle policy. Lets existing code that calls
        decide/reinforce run on a hypervector policy without re-plumbing. (Re-declares a fresh brain on the
        same action set; any table-mode learning is not carried over.)"""
        if not self._actions:
            raise RuntimeError("declare actions() before switching the brain backend")
        return self.actions(self._actions, value_backend=("routed" if routed else "holo"))

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
        """Reinforcement-learning update: teach the decision brain that taking `action` in `state` earned `reward`.
        Perceives `state` into a hypervector (modality inferred when None), then records the (state, action, reward)
        experience so future decide()/act() calls prefer higher-reward actions in similar states. `action` must be one
        the mind already knows (see self._actions). Returns self (chainable), so you can stream experiences."""
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
        """Predict the next symbol (character) that should follow `context`, using a learned FLAT n-gram sequence
        schema. `name` selects among several learned schemas (defaults to the one matching the context). Returns the
        single most likely next character. Requires a flat engine -- train one with
        learn_sequence(text, hierarchical=False); raises RuntimeError if the selected schema is the hierarchical
        (fractal) kind, which decodes its own way. For whole continuations rather than one step, use generate()."""
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

    def video_codec(self, dim=None, key_keep=400, res_keep=80, bits=8, gop_len=6, max_shift=8, seed=0):
        """MOTION-COMPENSATED VIDEO/FRAME CODEC (holographic_video, HolographicVideo) -- the rigid-shift-is-a-bind
        property made into a temporal codec. Encodes a grayscale frame sequence as a group-of-pictures: every
        gop_len-th frame is stored whole (a keyframe), the rest as a one-number MOTION VECTOR plus a holographically-
        compressed RESIDUAL against the motion-shifted previous reconstruction. A rigid pan is exactly a shift, so the
        residual nearly vanishes and the codec is a strict rate-distortion win over per-frame storage; non-rigid
        change leaves a large residual and is an honest loss (the kept boundary). This is the image-domain twin of the
        token codec (compress_lossless) -- both spend bits only on what a predictor cannot foresee. Returns a
        HolographicVideo on the mind's dim: `encode(frames)` -> (packets, total_bytes), `decode(packets)` -> frames,
        `mean_psnr(frames, packets)`, and the static `intra_baseline(frames, keep, ...)` to compare against. Serves
        the Stam/Puckette (temporal) and Duda (compression) seats. Delegates to holographic_video."""
        from holographic_video import HolographicVideo
        return HolographicVideo(dim=dim or self.dim, key_keep=key_keep, res_keep=res_keep, bits=bits,
                                gop_len=gop_len, max_shift=max_shift, seed=seed)

    def attribute_sources(self, tokens, sources, topk=15, order=2):
        """Source attribution: trace which stored material a passage drew on. sources
        is {name: token_stream}; returns a provenance distribution over those sources
        from the predictor's resonance couplings (holographic_codec.SourceAttributor)."""
        from holographic_codec import SourceAttributor
        att = SourceAttributor(dim=self.dim, order=order, seed=0).fit(sources)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return att.attribute(toks, topk=topk)

    def gated_traverse(self, step, start, floor=0.15, max_steps=64, min_steps=1):
        """Drive an iterative holographic traversal with a THROUGHPUT GATE -- Russian roulette for a path
        through the space (a multi-hop recall, the resonator's peeling, a recursive descent). In the phasor
        domain a bind is multiplicative, so a chain of binds is a ray whose recoverable signal attenuates;
        `step(state) -> (next_state, throughput, payload)` reports a cheap per-step confidence (a cleanup
        cosine, a convergence margin) and the traversal STOPS the instant it falls below `floor` -- the ray
        has gone dark -- abstaining on that step. Returns TraversalResult(payloads, throughputs, steps,
        stopped, final_throughput). Measured: it recovers the valid prefix and abstains exactly when the
        signal is gone, without ground truth, at lower average cost than a fixed depth (holographic_traverse).
        `step` may return None for a natural end; `floor` is on whatever scale the step reports as throughput."""
        from holographic_traverse import gated_traverse
        return gated_traverse(step, start, floor=floor, max_steps=max_steps, min_steps=min_steps)

    def occlusion_recall(self, cue, codebook, m=None, min_share=0.05, gram=None, cache=False):
        """OCCLUSION RECALL (holographic_occlusion, RT-V) -- recover the components present in `cue` (a bundle /
        superposition of `codebook` atoms) by an ordered, saturating front-to-back readout: take the most-relevant
        atom, record its share, SUBTRACT its explained part (the transmittance), repeat. The alpha-compositing
        transfer: the front explains the cue, the tail is OCCLUDED rather than summed, so multi-component recall
        survives FAR past the linear-bundle capacity cliff (measured F1 ~1.0 at high load where the linear /
        softmax / TopK top-m readouts wash out to ~0.91). Returns (index, weight) pairs in descending-relevance
        order; `m` fixes the count, else stop below `min_share`.

        SPEED (SPEED-1, Batch-OMP): pass a cached `gram` from build_occlusion_gram(codebook) -- the per-step
        dictionary rescan becomes a Gram-column update (the D factor leaves the inner loop), EXACT (identical
        atoms/order, weights to ~1e-16) and measured ~23x faster at D=1024. RAM (RAM-1): pass `cache=True` instead and
        this mind keeps a bounded GramCache, so a vocabulary queried many times pays the Gram precompute ONCE and the
        second call is a zero-precompute hit -- the caller need not manage the Gram. Kept negative: at LOW load it
        TIES plain linear recall; the Gram cache assumes immutable codebooks."""
        from holographic_occlusion import occlusion_recall
        if cache and gram is None:
            if not hasattr(self, "_gram_cache"):
                from holographic_occlusion import GramCache
                self._gram_cache = GramCache()
            gram = self._gram_cache.gram(codebook)         # RAM-1: build once, reuse across cues (id-keyed, GC-safe)
        return occlusion_recall(cue, codebook, m=m, min_share=min_share, gram=gram)

    def build_occlusion_gram(self, codebook):
        """The cached Gram matrix for occlusion_recall's fast path (holographic_occlusion, SPEED-1) -- G = codebook @
        codebook.T, computed ONCE and reused across cues. Pass it as occlusion_recall(..., gram=G): the readout then
        updates correlations through a Gram column per pick instead of rescanning the dictionary (O(N) vs O(N*D) per
        step). Pays whenever the same codebook is queried more than once (the engine's normal case); costs O(N^2)
        memory -- the storage-for-speed trade (Rubinstein-Zibulevsky-Elad 2008, Batch-OMP)."""
        from holographic_occlusion import build_gram
        return build_gram(codebook)

    def build_occlusion_forest(self, codebook, n_trees=4, leaf_size=64, seed=0):
        """Build a HoloForest over `codebook` for forest-routed occlusion selection (holographic_occlusion, SPEED-2,
        the N-factor) -- built once, reused across cues like the SPEED-1 Gram. Pass it as occlusion_recall_forest(...,
        forest=F). See occlusion_recall_forest for the measured trade-off (it is a kept negative at current scale)."""
        from holographic_occlusion import build_occlusion_forest
        return build_occlusion_forest(codebook, n_trees=n_trees, leaf_size=leaf_size, seed=seed)

    def occlusion_recall_forest(self, cue, codebook, m, forest=None, beam=4, n_trees=4, seed=0):
        """OCCLUSION RECALL, FOREST-ROUTED (holographic_occlusion, SPEED-2) -- occlusion recall with the per-step
        atom selection routed through a HoloForest instead of an exact O(N) scan. The N-FACTOR: the pick-the-most-
        relevant-atom step is a max-inner-product search, and the forest answers it by comparing only the atoms ROUTED
        to the query's leaves -- genuinely sub-linear in the dictionary size N. Returns (index, weight) descending,
        like occlusion_recall.

        SHIPPED AS A KEPT NEGATIVE (the capability is real, its limits are loud and measured): the comparison count IS
        sub-linear (~12% of the atoms at N=5000), but at this engine's operating scale it is a REGRESSION -- the exact
        selection is a single vectorized BLAS matrix-vector product that the Python-level tree routing cannot beat
        until N is very large (measured ~0.1x speed at N=500, ~0.6x at N=5000, still slower), AND the forest is
        APPROXIMATE, so when N is finally large enough to compare few candidates the approximate pick drops recovery
        F1 to ~0.77 (exact is 1.0). For everything at current scale, exact occlusion_recall (with a cached Gram) wins
        on BOTH speed and accuracy; this path is for the very-large-N, approximate-acceptable regime only. The
        N-factor of the occlusion-speed analysis, measured to its honest conclusion."""
        from holographic_occlusion import occlusion_recall_forest
        return occlusion_recall_forest(cue, codebook, m, forest=forest, beam=beam, n_trees=n_trees, seed=seed)

    def iht_recall(self, cue, codebook, K, steps=300, mu=None, tol=1e-12):
        """IHT RECALL (holographic_iht, GRAD-1) -- recover the K active atoms of `cue` (a bundle of `codebook` rows)
        by ITERATIVE HARD THRESHOLDING: projected gradient descent, a gradient step on the reconstruction loss then
        keep the K largest coefficients, ITERATED. The gradient-native sibling of occlusion_recall (greedy matching
        pursuit) and the linear readout: unlike greedy MP it REVISES its support, so a coefficient dropped at one step
        can return -- which is why it holds up on a COHERENT dictionary where greedy MP's early wrong picks become
        unrecoverable. Built on GRAD-2: the gradient step is the descent optimize() generalized, and with K=N (no
        threshold) IHT reduces to plain gradient descent = the least-squares solution; the hard-threshold projection
        is the one thing that makes it sparse recovery. Returns (index, weight) pairs descending by |weight|, the same
        shape as occlusion_recall. MEASURED: ties occlusion when incoherent (both ~perfect), BEATS it at high
        coherence (F1 0.71 vs 0.54). Kept negative: greedy MP wins at LOW-MILD coherence -- IHT is the coherent-regime
        method, not a strict upgrade; it needs the sparsity K and a step size mu (defaulted to 1/Lipschitz)."""
        from holographic_iht import iht_recall
        return iht_recall(cue, codebook, K, steps=steps, mu=mu, tol=tol)

    def cosamp_recall(self, cue, codebook, K, iters=15, tol=1e-10, stats=None):
        """CoSaMP RECALL (holographic_cosamp, SPEED-3) -- recover the K active atoms of `cue` by BATCH selection with
        a least-squares solve each round: correlate the residual with every atom, take the 2K most-correlated, MERGE
        with the current support, solve least-squares over that merged set, PRUNE to the K largest, repeat. The
        strongest member of the recovery family (linear / occlusion / IHT / CoSaMP): the least-squares solve gets
        EXACT coefficients and corrects errors the greedy and gradient methods cannot, so it recovers PERFECTLY across
        dictionary coherence where occlusion falls to ~0.54 and IHT to ~0.71 -- and converges in ~2-3 ROUNDS, not M
        sequential picks (the M-factor companion to SPEED-1's D-factor Gram). Returns (index, weight) descending by
        |weight|, the same shape as occlusion_recall / iht_recall; pass stats={} for stats['rounds']. Kept negatives:
        each round costs a least-squares solve (cost grows with K), and it FALLS OFF at the underdetermined phase
        transition when the load M approaches the dimension D (recovery lives below ~M < D/3 -- no method recovers
        above it)."""
        from holographic_cosamp import cosamp_recall
        return cosamp_recall(cue, codebook, K, iters=iters, tol=tol, stats=stats)

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
                            readout="softmax", confidence=False, k=8, early_stop=False, stats=None):
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
        approximate inputs (its null is matched to the chosen readout).

        early_stop=True (ADAPT-2) stops the resonator the moment its picks VERIFY: an exact reconstruction cannot be
        improved by more iterations, so this returns the SAME verified answer the fixed count would, only sooner --
        matched quality at lower average cost on easily-solved problems, a no-op on hard ones. Pass stats={} to read
        back stats['iters'] (the inner iterations actually run), so the saving is measurable."""
        from holographic_sbc import decompose_structure as _decompose
        return _decompose(np.asarray(composed), codebooks, L, restarts=restarts, iters=iters,
                          seed=self.seed if seed is None else seed, readout=readout, confidence=confidence, k=k,
                          early_stop=early_stop, stats=stats)

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

    def structured_index(self, keys, payloads=None, n_trees=4, leaf_size=64, keying="projection",
                         nbuckets=None, tile=1, normalize=True):
        """A content-addressable structured index over a list of keys (holographic_tree.StructuredIndex)
        -- the one shared lookup the route/sequence chunkers and the content store all draw from: file each
        item under its OWN key, find it in SUB-LINEAR (or zero) comparisons, and get back a meaningful label
        (the `payloads`), not a row number. Returns a StructuredIndex.

        It exists so the next caller that needs "find the item this query points at, at scale" reaches for one
        primitive instead of re-growing a fourth near-copy and rediscovering the same two limits. Both rules
        are enforced and explained in the class: KEY ON THE ITEMS THEMSELVES (a hyperplane tree only routes
        when query ~= key; a bundle-summary the query is weakly correlated with mis-routes -- measured), and
        NEVER STORE THE INDEX AS A BUNDLE (a superposed index caps with set size -- measured).

        `keying` picks the routing regime -- the pivot that fits the query (the RAM / page-table lesson):
          'projection' (default) routes a CONTENT vector through the RP-tree forest (sub-linear, approximate,
        with the agreement/abstention signal); 'hash' makes a stable hash of a LABEL the address (the page-
        table / RAM regime -- zero-comparison, exact); 'spatial' floor-divides a COORDINATE into a tile (the
        splat-tiler regime). locate_exact() is the flat guaranteed answer for small sets (what RouteIndex's
        flat scan already is). For INTEGRITY instead of lookup -- has anything changed, which item -- use
        verify_store (the holographic Merkle tree): a different job, and comparing whole composites by cosine
        is an evaluation that does NOT cap."""
        from holographic_tree import StructuredIndex
        return StructuredIndex(self.dim, n_trees=n_trees, leaf_size=leaf_size, seed=self.seed,
                               keying=keying, nbuckets=nbuckets, tile=tile, normalize=normalize).build(keys, payloads)

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

    def tile_field(self, enc, function, period, counts):
        """Tile an FPE field hypervector over an n-D lattice -- domain repetition as bind+bundle, so the result
        is itself a composable hypervector (works in 2-D and 3-D). See holographic_tiling.tile."""
        from holographic_tiling import tile
        return tile(enc, function, period, counts)

    def tile_field_recursive(self, enc, function, period, counts, levels):
        """Recursive tiling (inception): tile the tiling `levels` deep -- count^levels copies per axis from
        linear binds, in one fixed-size vector. See holographic_tiling.tile_recursive."""
        from holographic_tiling import tile_recursive
        return tile_recursive(enc, function, period, counts, levels)

    def fractal_field(self, enc, function, base_period, levels, count=3, decay=1.0):
        """A multi-scale (fBm-like) tiling: the motif tiled at base_period, /2, /4, ... summed -- richness at
        many scales from one motif and a few binds. See holographic_tiling.fractal_bands."""
        from holographic_tiling import fractal_bands
        return fractal_bands(enc, function, base_period, levels, count=count, decay=decay)

    def fractal_volume(self, enc, period, counts, levels, motif=None, beta=2.0, seed=0, motif_size=5,
                       motif_grid=None, motif_coords=None):
        """ONE call: inception over ANY VSA object -> one hypervector. The seed can be a precomputed motif
        hypervector (a smoke puff, an SDF surface, an archive image, or ANOTHER fractal_volume's output --
        inception over the engine itself), a NumPy grid (motif_grid, crossed into VSA once), or, by default, a
        synthesised localized fractal grain. tile_recursive replicates it count^levels deep in one fixed-size
        vector, composable as any VSA object. See holographic_tiling.fractal_volume."""
        from holographic_tiling import fractal_volume
        return fractal_volume(enc, period, counts, levels, motif=motif, beta=beta, seed=seed,
                              motif_size=motif_size, motif_grid=motif_grid, motif_coords=motif_coords)

    def inception(self, enc, period, counts, depth, motif=None, beta=2.0, seed=0, motif_size=5):
        """One-parameter recursion DEPTH over fractal_volume + an honest capacity-ceiling measurement. Returns
        (volume, profile): the volume is the recursive tiling carried `depth` levels (composable as any VSA
        object); the profile reports, at each depth, copies-per-axis, mean per-copy read, and round-trip
        recovery -- so the SNR fall as counts**depth instances share one fixed dim is a measured table. See
        holographic_tiling.inception."""
        from holographic_tiling import inception
        return inception(enc, period, counts, depth, motif=motif, beta=beta, seed=seed, motif_size=motif_size)

    def grid_to_hypervector(self, enc, grid, coords, threshold=1e-3):
        """Encode a NumPy field (a fluid density, an SDF slice) as an FPE hypervector so it can be tiled /
        bound / bundled / stored like any VSA object -- the one crossing from grid into VSA. See
        holographic_tiling.grid_to_function."""
        from holographic_tiling import grid_to_function
        return grid_to_function(enc, grid, coords, threshold=threshold)

    def hypervector_to_grid(self, enc, function, coords):
        """Read an FPE field hypervector back onto a grid (the inverse of grid_to_hypervector). See
        holographic_tiling.function_to_grid."""
        from holographic_tiling import function_to_grid
        return function_to_grid(enc, function, coords)

    def harmonic_atom(self, thetas, meanings, n_harmonics):
        """CONTEXT-CONDITIONED ATOM (holographic_harmonic, RT-VI) -- a polysemous atom whose decoded MEANING is a
        function of a context angle, represented in a CIRCULAR-harmonic (Fourier) basis on this engine's own FHRR/FPE
        phase substrate (the spherical-harmonics transfer: phase = a point on the circle = a direction). Fit from
        (context angle, meaning) pairs; `n_harmonics`=K keeps the DC plus K-1 harmonics. The DC term is the
        context-FREE meaning -- exactly the plain fixed atom, so a context-free atom reduces to the plain atom at K=1
        (backward-compatible). Returns a harmonic atom to read with harmonic_decode / harmonic_dc. Measured: distinct
        senses at distinct contexts are each recovered (cos>0.999) and blend between; a smooth (band-limited) meaning
        is exact at K=B+1, beating per-context storage. Kept negative: for context-free atoms the DC suffices (ties
        the plain atom by construction); if the variation is not smooth it degenerates to storing every context."""
        from holographic_harmonic import harmonic_atom
        return harmonic_atom(thetas, meanings, n_harmonics)

    def harmonic_decode(self, atom, theta):
        """Read a context-conditioned atom (holographic_harmonic, RT-VI) at context angle `theta` -- the harmonic
        sum giving the meaning in that context. The analog of unbinding with an FPE-encoded role(theta)."""
        from holographic_harmonic import harmonic_decode
        return harmonic_decode(atom, theta)

    def harmonic_dc(self, atom):
        """The DC (degree-0), context-FREE meaning of a context-conditioned atom (holographic_harmonic, RT-VI) --
        exactly the plain fixed atom, the backward-compatible fallback."""
        from holographic_harmonic import harmonic_dc
        return harmonic_dc(atom)

    def spectral_basis(self, points, k=10, n_basis=12):
        """The data-driven decomposition basis for a signal on a manifold (holographic_spectral, EXP-6): the
        lowest eigenvectors of the kNN-graph Laplacian of the sample points -- the smoothest functions the
        manifold admits. On a line this is the DCT / elementary basis and on a ring the harmonic basis (so it
        matches decompose_signal's hand-picked choice), and on a manifold the topology detector cannot name
        (a sphere, a torus, a curved surface) it is the right basis where the line/elementary fallback is not.
        Returns a SpectralBasis with decompose / reconstruct / denoise. Dense eigh -> moderate N (C1)."""
        from holographic_spectral import SpectralBasis
        return SpectralBasis(points, k=k, n_basis=n_basis)

    def nystrom_embedding(self, points, n_basis=4, m=None, sigma=None, landmarks="fps", seed=0):
        """The SCALABLE spectral embedding (SCALE-1): the smooth eigenbasis of `spectral_basis` without the dense
        O(N^3) eigh. It does the high-precision eigh on a small m x m LANDMARK block (farthest-point-sampled so
        every cluster is covered -- the irradiance-cache idea applied to the latent space) and extends to all N
        in O(m^3 + N*m), forming only an N x m affinity block, never N x N. Measured ~286x faster / 38x less
        memory at N=2400 (and the win grows with N). Drop-in for the smooth-embedding use of laplacian_eigenbasis.
        KEPT NEGATIVE: it is an APPROXIMATION -- exact for low-rank / well-separated structure (alignment ~1.0 on
        clustered data), but only ~0.76 on a curved high-rank manifold; spend more landmarks for more accuracy.
        Use `spectral_basis` when N is small and exactness matters; this when N is large. Returns (eigenvalues,
        eigenvectors (N, n_basis)). See holographic_nystrom."""
        from holographic_nystrom import nystrom_embedding
        return nystrom_embedding(points, n_basis=n_basis, m=m, sigma=sigma, landmarks=landmarks, seed=seed)

    def spectral_landmarks(self, points, m, seed=0):
        """Farthest-point-sampled landmark indices: a coverage set (every local manifold gets an anchor) for the
        Nystrom embedding -- the discrete cousin of the engine's blue-noise sampling. See holographic_nystrom."""
        from holographic_nystrom import farthest_point_landmarks
        return farthest_point_landmarks(points, m, seed=seed)

    def holo_octree(self, bounds, points=None, capacity=48, dim=2048, bandwidth=8.0, max_depth=8, seed=0):
        """A capacity-adaptive 3D holographic octree (TILE3D-1): tile 3D space so each node carries its points as
        ONE FPE 'wave' hypervector and AUTO-SPLITS into 8 octants when it exceeds `capacity` -- 'spin up another
        vector when the first is too full', in 3D and automatic. The tree IS the bidirectional spatial index:
        descend a position to its leaf (forward), read the leaf's points / occupancy wave back (backward), keeping
        the engine's content-addressable semantic recall. This is the fix for the capacity cliff: one global wave
        loses the ability to tell a stored point from empty space as N grows (measured AUC 0.85 at N=50 -> ~0.5 by
        N=800), while the octree holds AUC ~1.0 at any N by bounding each leaf's load -- at a cost of one vector
        per non-empty leaf (proportional storage, the same trade splat_bundle_tiled makes in 2D). Each child
        encoder is scaled to its smaller box, so local resolution sharpens with depth. Pass `points` (N,3) to
        build immediately, or insert later. See holographic_octree.HoloOctree."""
        from holographic_octree import HoloOctree
        tree = HoloOctree(bounds, capacity=capacity, dim=dim, bandwidth=bandwidth, max_depth=max_depth, seed=seed)
        if points is not None:
            tree.insert(points)
        return tree

    def similarity_graph(self, vectors, k=6, weighted=True):
        """A geometry-weighted kNN graph over hypervectors (holographic_simgraph, ARCH-3): the cotangent-Laplacian
        idea turned inward. weighted=True makes each edge carry the COSINE SIMILARITY (the geometry of the vector
        space) -- the analogue of cotangent weights; weighted=False is the engine's existing BINARY kNN graph.
        Returns a symmetric adjacency matrix."""
        from holographic_simgraph import similarity_adjacency
        return similarity_adjacency(vectors, k, weighted=weighted)

    def graph_spectral_embedding(self, vectors, k=6, dims=2, weighted=True):
        """Laplacian eigenmaps of a hypervector set (holographic_simgraph, ARCH-3): the `dims` lowest non-trivial
        eigenvectors of the (similarity-weighted) graph Laplacian -- the data-driven coordinates the manifold's
        points live on. Recovers a ring as a circle, a curve as a line. Returns (N, dims). Kept negative: under
        UNIFORM high-D sampling the weighted and binary graphs essentially tie (concentration of measure, unlike a
        mesh's sharp cotangent gap); weighting helps most under IRREGULAR sampling."""
        from holographic_simgraph import spectral_embedding
        return spectral_embedding(vectors, k=k, dims=dims, weighted=weighted)

    def graph_ring_order(self, vectors, k=6, weighted=True):
        """For hypervectors sampled on a 1-D ring, the recovered cyclic coordinate (holographic_simgraph, ARCH-3):
        atan2 of the first two non-trivial Laplacian eigenvectors -- a ring's eigenmap is a circle, so this recovers
        the points' order around the ring from the high-D vectors alone. Returns (N,) angles."""
        from holographic_simgraph import ring_order
        return ring_order(vectors, k=k, weighted=weighted)

    def subdivide_sequence(self, points, levels=1, closed=False):
        """Subdivide a SEQUENCE of hypervectors into a smooth limit curve (holographic_subdivcurve, ARCH-5):
        Chaikin corner-cutting, the 1-D inward mirror of FWD-8's mesh subdivision. Each level doubles the point
        count (refine) and low-pass smooths (corner-cutting). Reproduces a straight line of vectors exactly (affine,
        like FWD-8's 'flat stays flat'), converges to a limit curve, and shrinks a zig-zag's roughness. Returns the
        refined (M, dim) sequence. Kept negative: Chaikin is APPROXIMATING -- the original control points are cut,
        not interpolated (the same approximating nature as Loop; an interpolating 4-point scheme is deferred)."""
        from holographic_subdivcurve import subdivide_sequence
        return subdivide_sequence(points, levels=levels, closed=closed)

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

    def decompose_scene_tiled(self, tile_scenes, counts, sweeps=2):
        """Factor a scene too big to recover whole, by TILING (chunking-transfer item X1). A many-object scene
        exceeds the resonator's per-scene object cap (~5 at dim 1024; past it whole-scene recovery collapses,
        ~30% at 15 objects), so split objects into spatial tiles of <= cap each, factor every tile's sub-scene
        independently, and merge -- lifting recovery from ~30% to ~93% at 15 objects, dim 1024. The same move
        chunk_route makes for a long route: beat a fixed structure's capacity with composition, the tile size
        playing the chunk's role (keep it under the per-tile cap). `tile_scenes`: per-tile scene vectors (each
        an encode_scene of that tile's objects, grouped by region by the caller); `counts`: objects per tile.
        Returns the flat list of recovered tag-triples across all tiles."""
        return self.scene().factor_scene_tiled(tile_scenes, counts, sweeps=sweeps)

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

    def phase_morph(self, a, b, t):
        """Morph between two FHRR phasor vectors in the PHASE domain (phase shift = motion) -- interpolate each
        component's phase along the shortest arc, staying on the unit-phasor manifold (PHASE-1). This moves the
        decoded feature at CONSTANT velocity and keeps the morph a valid full-energy phasor at every t, where the
        amplitude-domain blend ((1-t)a + t*b) eases non-uniformly and collapses in magnitude where components fall
        out of phase. KEPT NEGATIVE / scope: the shortest arc wraps once a component's phase difference exceeds pi,
        so under extreme (near-orthogonal) change it stops tracking the true intermediate -- the win holds while the
        change keeps per-component phase differences under pi. `t` in [0, 1]."""
        from holographic_phasemorph import phase_morph
        return phase_morph(a, b, t)

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

    def validate_recipe(self, recipe):
        """Check a StructureRecipe is WELL-FORMED (holographic_recipeops, ARCH-1) -- the recipe's is_manifold():
        every op references only EARLIER existing results (a DAG, no forward/dangling/out-of-range refs), raw
        indices and repeat templates in range. Returns (ok, problems). Pairs with the recipe EDIT operators below
        -- the recipe equivalent of the mesh Euler operators (validate + local invariant-preserving edits)."""
        from holographic_recipeops import validate
        return validate(recipe)

    def recipe_commute_bind(self, recipe, handle):
        """Edit a recipe (holographic_recipeops, ARCH-1): swap the two arguments of the bind at `handle`,
        bind(a,b)->bind(b,a). Because bind is circular convolution (commutative), the realized vector is unchanged
        (FFT precision). ITS OWN INVERSE -- the recipe analogue of mesh flip_edge. Returns a new recipe."""
        from holographic_recipeops import commute_bind
        return commute_bind(recipe, handle)

    def recipe_reorder_members(self, recipe, handle, perm):
        """Edit a recipe (holographic_recipeops, ARCH-1): permute the members of the bundle/superpose at `handle`
        by `perm`. Because bundle/superpose are sums (commutative), the realized vector is unchanged; invertible by
        the inverse permutation. The parameterised cousin of recipe_commute_bind. Returns a new recipe."""
        from holographic_recipeops import reorder_members
        return reorder_members(recipe, handle, perm)

    def recipe_substitute_atom(self, recipe, handle, new_name):
        """Edit a recipe (holographic_recipeops, ARCH-1): rename the atom leaf at `handle` -- the recipe analogue
        of moving a vertex (keeps the STRUCTURE valid while changing the realized vector predictably; reversed
        exactly by substituting the original name back). Returns a new recipe."""
        from holographic_recipeops import substitute_atom
        return substitute_atom(recipe, handle, new_name)

    def template_names(self):
        """The names of the available parameterized recipe templates (ISA-6, the macro layer)."""
        from holographic_template import STARTER_LIBRARY
        return sorted(STARTER_LIBRARY)

    def instantiate_template(self, name, **args):
        """Instantiate a named parameterized template (ISA-6) at this mind's dim/seed, filling its HOLES with
        `args` (a string is a named atom; an array is a literal vector). Returns the built structure vector --
        different arguments give distinct, BIT-EXACT structures, and the templates are hygienic (internal atoms
        are namespaced under a reserved prefix and cannot collide with the caller's atoms)."""
        from holographic_template import STARTER_LIBRARY
        if name not in STARTER_LIBRARY:
            raise ValueError(f"unknown template {name!r}; available: {sorted(STARTER_LIBRARY)}")
        return STARTER_LIBRARY[name].build_vector(self.dim, self.seed, **args)

    def compile_structure(self, spec):
        """Compile a structure-description spec (ISA-7) to a StructureRecipe at this mind's dim/seed. The spec is
        S-expression text or a parsed AST -- a declarative surface (atoms, bind/bundle/permute, ISA-6 templates)
        that lowers to the recipe IR. Scoped to structure description, not a general language."""
        from holographic_lang import compile_spec
        return compile_spec(spec, self.dim, self.seed)

    def realize_structure(self, spec):
        """Compile a structure-description spec (ISA-7) and materialize its vector -- bit-exact for a given
        spec/dim/seed. The high-level surface for the same structures realize/compose build by hand."""
        from holographic_lang import realize_spec
        return realize_spec(spec, self.dim, self.seed)

    def reversibility_audit(self):
        """Classify each base instruction as reversible (bind/unbind/permute/involution) or information-
        destroying (bundle/superpose/cleanup) -- the ISA-8 reversibility model. cleanup is error correction;
        capacity is the coherence budget. (Framing; the practical payoff is run_with_auto_cleanup.)"""
        from holographic_reversible import reversibility_audit
        return reversibility_audit()

    def run_with_auto_cleanup(self, initial, steps, codebook, floor=0.9, schedule="adaptive", k=3):
        """Run a vector 'program' (a list of vector->vector steps) under an error-correction policy (ISA-8),
        inserting a cleanup before the crosstalk cliff. 'adaptive' cleans only when the nearest-atom health drops
        below `floor`; it holds fidelity at far fewer cleanups than a fixed cadence under variable damage.
        Returns (final_vector, n_cleanups)."""
        from holographic_reversible import auto_cleanup_run
        return auto_cleanup_run(initial, steps, codebook, floor=floor, schedule=schedule, k=k)

    def steering_regress(self, X, y, X_query, bounds, base=2.0, dim=None):
        """Anisotropic (steering) kernel regression (RT-IV1): fit a PER-AXIS bandwidth to the data's directional
        structure (a sharp axis gets a large bandwidth, a flat axis a small one), build an anisotropic FPE
        encoder, and predict X_query by FPE-kernel-weighted averaging. Returns (predictions, bandwidths). Beats
        the isotropic RBF on DENSE directional data (an edge/ridge); on sparse or isotropic data the advantage is
        marginal and the bandwidth estimate is unreliable -- isotropic is the honest baseline there."""
        from holographic_steering import steer_bandwidths, kernel_regress
        from holographic_fpe import VectorFunctionEncoder
        bw = steer_bandwidths(X, y, base=base)
        enc = VectorFunctionEncoder(len(bounds), dim=(dim or self.dim), bounds=bounds, bandwidth=bw, seed=self.seed)
        return kernel_regress(enc, X, y, X_query), bw

    def propagator_jump(self, states, state, k):
        """Jump a learned dynamics operator k steps in ONE eval (RT-I1): the closed-form k-step iterate via the
        FREE Fourier spectrum (a bind is diagonal in the Fourier basis), matching the k-bind rollout to FFT
        tolerance. Diagonalise once, evaluate any level -- the same math Stam's subdivision eval uses."""
        from holographic_iterate import step_k
        U = self.learn_dynamics(states).U
        return step_k(state, U, k)

    def propagator_spectrum(self, states):
        """Read a learned dynamics operator's convergence off its FREE FFT spectrum (RT-I1) WITHOUT running:
        the regime (contractive -> decays; marginal -> persists; divergent -> blows up), the spectral_gap
        (small -> slow/near-degenerate stall, the linear cousin of a resonator stall), and the dominant frequency.
        The eigendecomposition of the bind operator is just its rfft -- no dense O(n^3) work."""
        from holographic_iterate import spectral_profile
        U = self.learn_dynamics(states).U
        return spectral_profile(U)

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

    def scene_graph(self, transform=None, mesh=None, children=None, name=None):
        """Build a SCENE-GRAPH node (holographic_scenegraph): a 4x4 `transform`, an optional leaf `mesh`, optional
        child nodes -- the geometry capstone that joins the FWD mesh kernel to the ARCH-1 recipe algebra. The node
        can be read two ways (see scene_flatten / scene_to_recipe). Transform builders are on the mind:
        scene_translation / scene_scaling / scene_rotation / scene_compose_transforms. Returns a SceneNode."""
        from holographic_scenegraph import SceneNode
        return SceneNode(transform=transform, mesh=mesh, children=children, name=name)

    def scene_flatten(self, node):
        """The GEOMETRY view of a scene graph (holographic_scenegraph): instance every leaf mesh through its
        ACCUMULATED transform (parent transforms composed down the graph) and MERGE into one Mesh. Returns a Mesh.
        Kept negative: this INSTANCES and concatenates -- it does not weld or boolean-merge overlapping geometry
        (that is mesh_csg's job); two touching cubes flatten to two components, not one solid."""
        from holographic_scenegraph import flatten_scene
        return flatten_scene(node)

    def scene_to_recipe(self, node, dim=None, seed=None):
        """The STRUCTURE view of a scene graph (holographic_scenegraph): encode the graph as a StructureRecipe --
        transforms BOUND to content, siblings BUNDLED -- realising to one hypervector. A well-formed recipe that the
        ARCH-1 operators (validate_recipe / recipe_reorder_members) apply to. The consistency theorem: swapping
        siblings leaves BOTH the flattened geometry and this vector identical (merge and bundle are commutative) --
        the scene is one object in two costumes, and they agree. Returns a StructureRecipe."""
        from holographic_scenegraph import scene_to_recipe
        return scene_to_recipe(node, dim=self.dim if dim is None else dim, seed=self.seed if seed is None else seed)

    def scene_delta(self, base, variant):
        """The component DIFF between two scenes (holographic_scenedelta): {'added', 'removed'} content-hashed
        component ids, so a variant can be TRANSMITTED as its delta (send the base once, then small deltas). A
        one-subtree change is a couple of components; apply_scene_delta rebuilds the variant's component set exactly.
        Kept negative (honest): the component SHARING itself is AUTOMATIC from content-addressed atoms (shared
        subtrees share ids for free) -- this adds the explicit diff/transmission, not a new dedup mechanism."""
        from holographic_scenedelta import scene_delta
        return scene_delta(base, variant)

    def scene_dedup_saving(self, scenes):
        """Measure the content-addressed dedup saving across a set of scenes (holographic_scenedelta): {'naive',
        'unique', 'saving_x'} -- how much the automatic component sharing buys (measured ~4-6x across a base + its
        variants). The saving is automatic from content-hashing; this quantifies it. Returns a dict."""
        from holographic_scenedelta import scene_dedup_saving
        return scene_dedup_saving(scenes)

    def versioned_store(self, gop_len=8):
        """VERSIONED STORE with rollback (holographic_history, VersionedStore) -- a store whose every version is
        committed and exactly recoverable: the undo/redo and scene-versioning piece for the editable-mesh authoring
        vision, a natural companion to scene_delta. State is rows (vectors) keyed by stable integer ids plus their
        order; the history is keyframes + lossless row-keyed deltas (the same keyframe/GOP structure the video codec
        uses, here for an edit timeline). Build: `new_id()` for stable row ids, `commit(rows, order, proof=None,
        note='')` to record a version (an optional `proof(rows, order)` gate must return True or the commit is
        rejected and only logged -- proof-gated reorganization), `checkout(version)` to reconstruct any past state
        EXACTLY, `rollback(version)` to revert (itself recorded, so history is never erased), `head()`, and
        `history()` (the audit of every attempt). Returns a VersionedStore on the mind's dim. Delegates to
        holographic_history."""
        from holographic_history import VersionedStore
        return VersionedStore(self.dim, gop_len=gop_len)

    def scene_translation(self, t):
        """A 4x4 translation transform for scene_graph nodes (holographic_scenegraph)."""
        from holographic_scenegraph import translation
        return translation(t)

    def scene_scaling(self, s):
        """A 4x4 scale transform (uniform scalar or per-axis length-3) for scene_graph nodes."""
        from holographic_scenegraph import scaling
        return scaling(s)

    def scene_rotation(self, axis, angle):
        """A 4x4 rotation transform (Rodrigues, radians) for scene_graph nodes."""
        from holographic_scenegraph import rotation
        return rotation(axis, angle)

    def scene_compose_transforms(self, *matrices):
        """Compose 4x4 transforms (the product M0 @ M1 @ ..., parent then child) for scene_graph nodes."""
        from holographic_scenegraph import compose_transforms
        return compose_transforms(*matrices)

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

    def _plan_vocab(self):
        """The seed-deterministic atom source for shaped plans/records on this mind (regenerable from the
        mind's seed, like every other store). Cached."""
        if getattr(self, "_pvocab", None) is None:
            from holographic_planshape import ShapeVocab
            self._pvocab = ShapeVocab(min(self.dim, 1024), seed=self.seed + 31)
        return self._pvocab

    def plan_shape(self, actions, scopes, branch_skeleton=None):
        """Build the SHAPE (the decode key) for a contingency plan: the action and scope value codebooks and a
        nested-dict branch skeleton {name: {name: {...}}}. Hand this to decode_plan / descend so reading a plan
        vector is a deterministic unbind-and-clean walk, not the resonator's blind search."""
        from holographic_planshape import plan_shape
        return plan_shape(actions, scopes, branch_skeleton)

    def encode_plan(self, plan):
        """Encode a PlanNode contingency tree (a primary action, a scope, named branches each a PlanNode) as ONE
        hypervector -- the structured branching output the planner was missing, the HRR role-filler nested
        encoding. import PlanNode from holographic_planshape to build the tree."""
        from holographic_planshape import encode_plan
        return encode_plan(plan, self._plan_vocab())

    def decode_plan(self, vec, shape):
        """Decode a plan vector back to a PlanNode GIVEN ITS SHAPE (from plan_shape) -- schema-guided, so it
        stays exact far past the resonator's blind-parse cap. The returned node's confidence is the measured
        decode cosine of its action."""
        from holographic_planshape import decode_plan
        return decode_plan(vec, shape, self._plan_vocab())

    def descend(self, vec, situation, shape):
        """Walk a plan vector to the branch matching the current SITUATION (a branch-name str, or a state
        vector), returning the actions along that path -- the generalisation of IFMATCH from one gated
        instruction to a named branch tree WITH ABSTENTION (no branch clears the measured noise floor -> the
        node's primary action). Togelius's behavior-tree selector + Cranmer's measured floor."""
        from holographic_planshape import descend
        return descend(vec, situation, shape, self._plan_vocab())

    def encode_record(self, fields):
        """Encode a flat record {field_name: value_name} as one vector -- the GENERAL 'bring your own shape'
        path for any structured output (a scientific decision record, a classified state), not just plans. One
        bundle of role-bound value atoms."""
        from holographic_planshape import encode_record
        return encode_record(fields, self._plan_vocab())

    def decode_record(self, vec, schema):
        """Decode a flat record GIVEN ITS SHAPE: schema = {field_name: [possible_value_names]} (the per-field
        codebooks). Unbinds each role and cleans against that field's codebook -- a deterministic walk. Returns
        {field_name: value_name}."""
        from holographic_planshape import decode_record
        return decode_record(vec, schema, self._plan_vocab())

    def directed_structure(self, n, edges=None, seed=None):
        """Encode a directed SEQUENCE or GRAPH with a permutation DIRECTION ROLE (RAY-3): the successor of
        each edge is bound through a fixed permutation, M = superpose bind(node_i, perm(node_j)), so unbinding
        a node and undoing the permutation recovers its successor while the predecessor term is pushed into
        noise. The substrate-correct counterpart to chain_structure (B7, UNDIRECTED), whose predecessor leak
        otherwise needs holographic_peel's per-peel cleanup to suppress -- the permutation does at ENCODE time
        what the peel cleanup does at DECODE time. `n` node atoms are minted at this mind's dim/seed; `edges`
        is a list of (src, dst) index pairs (default: the linear chain 0->...->n-1; pass your own for a
        graph). Returns a DirectedStructure(memory, nodes, perm, perm_inv) -- query it with
        directed_successor() or walk it with directed_traverse()."""
        from holographic_directed import build
        s = self.seed if seed is None else seed
        rng = np.random.default_rng(s)
        nodes = rng.standard_normal((n, self.dim))
        nodes = nodes / np.linalg.norm(nodes, axis=1, keepdims=True)
        return build(nodes, edges=edges, seed=s + 1)

    def directed_successor(self, ds, node_index, topk=1, thresh=None):
        """Recover the successor(s) of a node in a DirectedStructure (RAY-3): perm_inv(unbind(M, node))
        cleaned up against the node codebook. Returns [(index, cosine), ...] -- the strongest `topk`, or every
        node at/above `thresh` (a branching node's whole successor set). The forward step of a directed walk;
        unlike the undirected baseline it returns the successor only, not both neighbours."""
        from holographic_directed import successors
        return successors(ds, node_index, topk=topk, thresh=thresh)

    def directed_traverse(self, ds, start_index=0, floor=0.15, max_steps=64, min_steps=1):
        """Walk a directed chain FORWARD from `start_index`, gated by recovery confidence -- the directed
        substrate (RAY-3) under the throughput-gated traversal (RAY-1). Each hop recovers the successor and
        reports its cleanup cosine as throughput; the walk stops when that drops below `floor` (the chain
        exhausted, the ray dark). Returns the TraversalResult (payloads = the recovered node indices in
        order). Unambiguously forward, because the direction role suppressed the predecessor leak."""
        from holographic_directed import make_step
        from holographic_traverse import gated_traverse
        return gated_traverse(make_step(ds), ds.nodes[start_index], floor=floor,
                              max_steps=max_steps, min_steps=min_steps)

    def plan(self, start, field_step, max_steps=14, floor=0.15, seed=None,
             action_of=None, is_branch=None):
        """Bake one CORRIDOR -- a short executable route to the next decision point -- on the directed
        substrate, the way PAST the per-structure capacity cap. A route stored as one bundle decodes only
        a handful of tiles before crosstalk wins; rather than push one structure past its reliable depth,
        roll out the goal field's downhill path for ~12-16 steps (`field_step(node) -> next_or_None`, the
        caller's gradient/flow/policy step; stop early at `is_branch(node)` -- a junction worth a real
        decision -- or at `max_steps`), bake it as a directed chain, and return a Plan: the compact plan
        hypervector, the decoded tile route, the decoded direction labels (if `action_of` is given), and a
        per-step throughput. The courier executes the baked steps with NO further thinking and re-anchors at
        the decision point via replan_needed(); the brain is consulted once per corridor, not once per tile.
        Keep max_steps at or under the dim's reliable decode depth (~15 at dim 512-1024) so the plan never
        claims steps it cannot carry. Built on directed_structure (RAY-3) + gated_traverse (RAY-1)."""
        from holographic_plan import plan as _plan
        return _plan(start, field_step, max_steps=max_steps, floor=floor,
                     seed=self.seed if seed is None else seed,
                     action_of=action_of, is_branch=is_branch)

    def replan_needed(self, p, executed, tile_ok=None, floor=0.15):
        """The cheap per-tick guard for a baked Plan: should the courier abandon it and re-anchor (call
        plan() again)? True when the plan is exhausted, the next baked step's throughput has fallen below
        `floor`, or `tile_ok(next_tile)` reports the next tile is no longer clear/on-route. Otherwise False
        -- execute the next baked step. No value() calls, no decode work: a list index and a comparison."""
        from holographic_plan import replan_needed as _replan
        return _replan(p, executed, tile_ok=tile_ok, floor=floor)

    def plan_route(self, start, field_step, max_total=200, corridor=14, floor=0.15,
                   seed=None, action_of=None, is_branch=None):
        """Bake a WHOLE arbitrarily-long route in one call, by chaining cap-sized corridors and re-anchoring
        internally at each leg's reliably-decoded end. This is the way past the per-structure ~15 cap
        delivered as a single result: a 45-tile route that collapses to noise if crammed into one plan()
        comes back correct here as a sequence of clean corridors. `corridor` is the per-leg length and must
        stay at/under the dim's reliable decode depth (default 14, safe at dim 512-1024) -- an over-long leg
        overstuffs its own structure, the same cliff per leg; `max_total` caps the whole route. Use this when
        you want the full route in hand (display / validate / pre-plan a leg); a real-time courier reacting to
        traffic still wants plan() + replan_needed (bake-as-you-go). Returns a Route: the full action sequence,
        the chained corridors, why it stopped, the re-anchor count, and the step total."""
        from holographic_plan import plan_route as _plan_route
        return _plan_route(start, field_step, max_total=max_total, corridor=corridor, floor=floor,
                           seed=self.seed if seed is None else seed,
                           action_of=action_of, is_branch=is_branch)

    def chunk_route(self, items, chunk=14, floor=0.15, seed=None, action_of=None):
        """Store/replay an EXPLICIT ordered sequence you ALREADY HAVE -- a GPS route from a planner, a fixed
        experiment protocol, any known list of N steps -- past the per-structure cap, by splitting it into
        <=chunk-element directed-structure pieces, each individually clean. The explicit-list twin of
        plan_route (which DISCOVERS its route by following a field): here the sequence is given, so it skips
        the rollout and just chunks, bakes, and replays it EXACTLY. The per-piece cap is HRR physics (a fixed
        structure can't hold unbounded order); chunking makes the EFFECTIVE length unbounded at LINEAR cost --
        a 200-step route is ~15 chunks, a 1000-step one ~72 -- and each chunk is ONE compact vector you can
        store or compose. `chunk` must stay at/under the dim's reliable decode depth (default 14); elements
        must be distinguishable so each chunk decodes. Returns a Route (full replayable actions + the chunk
        Plans + step total)."""
        from holographic_plan import chunk_route as _chunk_route
        return _chunk_route(items, chunk=chunk, floor=floor,
                            seed=self.seed if seed is None else seed, action_of=action_of)

    def index_route(self, route):
        """Build a sub-linear RANDOM-ACCESS index over a chunked route (from plan_route / chunk_route): a BVH
        over the chunks. "Where am I on this route?" becomes a jump, not a replay from the start -- index each
        chunk by a summary vector and locate a query two-level (nearest chunk summary, then nearest tile within
        it), ~(#chunks + chunk_size) comparisons instead of #tiles. Build once, query many (the courier asking
        its position every tick). Returns a RouteIndex; call its .locate(query) -> (chunk, position, global_step)."""
        from holographic_plan import RouteIndex
        return RouteIndex(route)

    def dedup_chunks(self, chunk_vectors, tol=1e-9):
        """Content-addressed deduplication of chunk vectors (chunking-transfer item C1): a route that REVISITS
        the same corridor, or a program with repeated motifs, stores the same compact chunk vector many times.
        Keep each unique chunk once and replace repeats with a reference -- storage shrinks by exactly the
        repetition ratio (measured 65% on a 17-corridor loop of 6 distinct chunks), and by nothing when there
        is no repetition (the honest bound). `chunk_vectors` is the ordered chunk list (e.g. the corridors'
        `.memory` vectors). Returns (unique, refs) where `[unique[r] for r in refs]` rebuilds the original list
        EXACTLY. The storage twin of structured_index: that finds an item by content, this stores by content so
        identical chunks coalesce -- and comparing whole chunks by cosine is an evaluation, so it never caps."""
        from holographic_plan import dedup_chunks
        return dedup_chunks(chunk_vectors, tol=tol)

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

    def find_pattern_by_downscale(self, data, kind="vectors", k=3, n_null=80, seed=0):
        """Find a pattern in noisy data by DOWNSCALING -- project to a coarse representation where independent
        noise averages out and structure survives (XDATA-1, the Group G entry). kind='vectors' pools correlated
        vectors to a top-k subspace (consolidation/SVD); kind='signal' keeps a signal's k strongest spectral
        components (low-pass FFT). 'found' is decided against a PERMUTATION NULL so it FAILS SAFE -- pure noise
        reports nothing rather than a hallucinated pattern. Returns PatternResult(pattern, score, null_mean,
        null_std, found). Same mechanism, any data type: downscale = low-pass = noise removal = pattern reveal."""
        from holographic_downscale import find_pattern_by_downscale
        return find_pattern_by_downscale(np.asarray(data, float), kind=kind, k=k, n_null=n_null, seed=seed)

    def multires_pyramid(self, signal, n_levels=5):
        """Build an anti-aliased mipmap of `signal` -- [full, half, quarter, ...], each level low-pass filtered
        before downsampling by two (SCALE-1). The decisive property is anti-aliasing on a COARSE read: a coarse
        pyramid level is a clean (alias-free), smaller view, where naively subsampling the full store folds
        high-frequency content into the low band. The levels are also a progressive code (coarsest is a usable
        approximation, finer levels add detail back, exact at the top). Returns the list of levels, coarsest last."""
        from holographic_multires import build_pyramid
        return build_pyramid(signal, n_levels=n_levels)

    def pyramid_reconstruct(self, level, n):
        """Resample a pyramid level (from multires_pyramid) back to length `n` -- the LOD read, so a coarse,
        anti-aliased level can be used or compared at full length (SCALE-1)."""
        from holographic_multires import upsample_to
        return upsample_to(level, n)

    def manifold_denoise(self, x, manifold, beta=18.0, steps=8):
        """Settle a (noisy) point ONTO a sample-defined manifold by looping a dense-Hopfield step (XDATA-2) --
        denoising as iterated projection. Generalises the codebook cleanup to ANY manifold given as a point cloud
        (a curved manifold, or a consolidation subspace from find_pattern_by_downscale). Idempotent: once on the
        manifold, further steps leave it fixed. Beats interpolation on a curved manifold (the chord midpoint
        leaves the manifold; this settles it back)."""
        from holographic_diffuse import settle
        return settle(np.asarray(x, float), np.asarray(manifold, float), beta=beta, steps=steps)

    def manifold_generate(self, manifold, steps=30, beta_lo=2.0, beta_hi=25.0, noise_hi=0.5,
                          noise_lo=0.0, settle_steps=5, seed=0):
        """Generate a NOVEL-but-VALID sample on a sample-defined manifold by annealed diffusion (XDATA-2): from
        noise, loop the denoise step with beta rising and injected noise falling, then settle. Lands ON the
        manifold (valid) but BETWEEN the stored samples (novel) -- where bare-codebook generation just returns a
        stored sample. The B10 diffusion generalised off the discrete codebook to a learned/composed manifold."""
        from holographic_diffuse import generate
        return generate(np.asarray(manifold, float), steps=steps, beta_lo=beta_lo, beta_hi=beta_hi,
                        noise_hi=noise_hi, noise_lo=noise_lo, settle_steps=settle_steps, seed=seed)

    def sharpen_loop(self, x, blur=None, sigma=3.0, lam=1.0, iters=60, noise_level=0.0):
        """Recover detail an over-smoothed signal LOST, by looping a converging negative-lobe (Van Cittert)
        sharpening (XDATA-3, the sharpen half of Group G). `blur` is the smoothing operator that over-smoothed it
        (callable; default a Gaussian low-pass with `sigma`). The accumulated correction is the INVERSE blur, a
        sharpening filter with negative lobes. With `noise_level` > 0 it stops by the discrepancy principle
        (residual hits the noise floor) to avoid amplifying noise -- the kept negative is that running past that
        over-sharpens, and an over-large `lam` diverges into ringing. Data-type-agnostic: the partner to the
        splat negative-lobe sharpening, for any smeared signal."""
        from holographic_denoisehome import Denoise                    # the Denoise home (consolidation R5)
        return Denoise.sharpen(np.asarray(x, float), blur=blur, sigma=sigma, lam=lam, iters=iters, noise_level=noise_level)

    def smooth_sharp_split(self, x, k_smooth, k_sharp):
        """Split a signal into a SMOOTH layer (its k_smooth lowest-frequency coefficients) and a SHARP layer (the
        k_sharp largest residual samples -- sparse in the sample domain) (CACHE-2). At a budget covering both
        layers this beats any single basis, because no single basis is cheap across smooth-plus-sharp content (the
        spikes are broadband in frequency but sparse in samples). Returns a TwoLayerCode; reconstruct with
        smooth_sharp_reconstruct. The right sharp basis matches the sharp content (sample-sparse for spikes)."""
        from holographic_twolayer import smooth_sharp_split
        return smooth_sharp_split(np.asarray(x, float), k_smooth, k_sharp)

    def smooth_sharp_reconstruct(self, code):
        """Reconstruct a signal from a two-layer code (CACHE-2): the smooth layer everywhere plus the exact sharp
        residual at the stored sharp positions."""
        from holographic_twolayer import smooth_sharp_reconstruct
        return smooth_sharp_reconstruct(code)

    def graph_denoise(self, vectors, k=8, method="taubin", lam=0.55, mu=-0.58, iters=8, sublinear=False):
        """Denoise / regularize a SET of vectors (a noisy codebook, an embedding, a value function) over its
        own k-NN similarity graph -- the graph-signal filter the stack lacked (reverse-transfer RT-III1; mesh
        smoothing mapped back onto the concept graph). `method='taubin'` is Taubin's lambda|mu no-shrink
        low-pass; 'laplacian' is the naive shrinking baseline. Where `denoise` cleans ONE vector against a
        manifold, this cleans a whole set USING its own redundancy (non-local means on the graph).

        Helps most at HIGH noise on a curved manifold whose local neighbourhoods survive when the global linear
        subspace is corrupted (measured: beats per-vector consolidation 6/6 seeds at rel-noise 1.2, and Taubin
        keeps its norm where the naive Laplacian collapses). KEPT NEGATIVE: at low noise a per-vector
        consolidation denoiser is better and this over-smooths. `sublinear=True` builds the k-NN graph from a
        HoloForest's recall_k instead of the O(n^2) dense scan -- reuse the index for large sets."""
        from holographic_graphsignal import graph_denoise
        forest = None
        if sublinear:
            from holographic_tree import HoloForest
            V = np.asarray(vectors, float)
            forest = HoloForest(V.shape[1], seed=self.seed).build(V)   # index over the vectors' own dim
        return graph_denoise(vectors, k=k, method=method, lam=lam, mu=mu, iters=iters, forest=forest)

    def manifold_chart(self, vectors, dim=2, method="isomap", k=10, sublinear=False):
        """Flatten a CURVED hypervector manifold to a low-D coordinate chart -- the nonlinear extension of
        `consolidation` (which is a LINEAR SVD chart and folds a curved manifold) (reverse-transfer RT-II1; UV
        unwrapping mapped onto the concept/state manifold). `method='isomap'` is the geodesic-preserving chart
        (recommended -- unrolls the manifold); 'spectral' is Laplacian Eigenmaps (local cluster structure, the
        graph-spectral cousin of `graph_denoise`'s Laplacian). Use it to SEE the concept space / a brain's state
        space, or as a tighter storage coordinate where the manifold is curved.

        Measured: on a swiss roll lifted into high-D, Isomap beats the linear SVD chart on geodesic-distance
        fidelity and class separation 5/5 seeds. SCOPE: a chart assumes disk topology -- a CLOSED manifold (a
        torus, genus>0) needs a cut first (the `topology` faculty finds the genus); a flat manifold is better
        served by the linear `consolidation`. `sublinear=True` finds neighbours via a HoloForest (RT-III1's
        index reuse); the geodesic step is otherwise O(N^3), so subsample for very large sets."""
        from holographic_chart import manifold_chart
        forest = None
        if sublinear:
            from holographic_tree import HoloForest
            V = np.asarray(vectors, float)
            forest = HoloForest(V.shape[1], seed=self.seed).build(V)
        return manifold_chart(vectors, dim=dim, method=method, k=k, forest=forest)

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

    # ---- the EXPLICIT 3-D GEOMETRY faculties (forward DCC backlog, FWD-1 / FWD-2) ----------------
    # The first explicit-mesh layer on the mind: a polygon mesh kernel (half-edge adjacency, Euler
    # invariants, normals, OBJ/buffer export) and the glTF (.glb) boundary that hands a scene to a
    # three.js front end. These are explicit-geometry I/O, NOT VSA hypervector ops -- a mesh is integer
    # connectivity over float positions, not a hypervector, and this layer does not pretend otherwise.
    # The bridge to the native VSA reps (mesh <-> SDF <-> splat, a mesh AS a StructureRecipe) is a later
    # item (FWD-11 / ARCH); this is the honest explicit substrate those bridges will connect to.

    def mesh_box(self, width=1.0, height=1.0, depth=1.0, center=(0.0, 0.0, 0.0)):
        """An axis-aligned box as a quad Mesh (holographic_mesh, FWD-1): the canonical first explicit mesh
        -- 8 vertices, 6 quad faces, V - E + F = 2 (a genus-0 closed surface). The returned Mesh carries
        the half-edge adjacency, Euler/manifold checks, normals, triangulation and OBJ/buffer export."""
        from holographic_mesh import box
        return box(width, height, depth, center)

    def mesh_tetrahedron(self, scale=1.0, center=(0.0, 0.0, 0.0)):
        """A regular tetrahedron as 4 triangles (holographic_mesh, FWD-1) -- the smallest closed manifold,
        a second primitive proving the Euler machinery is not box-specific."""
        from holographic_mesh import tetrahedron
        return tetrahedron(scale, center)

    def mesh_grid(self, nx=4, ny=4, width=1.0, height=1.0, center=(0.0, 0.0, 0.0)):
        """A flat subdivided plane as an (nx by ny) quad grid (holographic_mesh, FWD-1): an OPEN mesh
        (chi = 1, has a boundary) -- the test surface for boundary handling and bigger builds."""
        from holographic_mesh import grid
        return grid(nx, ny, width, height, center)

    def mesh_euler(self, mesh):
        """The combinatorial well-formedness signature of a Mesh (holographic_mesh, FWD-1): vertices,
        edges, faces, the Euler characteristic chi = V - E + F, whether the surface is closed and manifold,
        and -- for a closed surface -- the genus from chi = 2 - 2g. The exact-integer health check the
        explicit-geometry edit operators (FWD-7) will be required to preserve."""
        return {
            "vertices": mesh.n_vertices,
            "edges": mesh.n_edges,
            "faces": mesh.n_faces,
            "characteristic": mesh.euler_characteristic(),
            "closed": mesh.is_closed(),
            "manifold": mesh.is_manifold(),
            "genus": mesh.genus(),
        }

    def mesh_to_gltf(self, mesh, base_colour=(0.8, 0.8, 0.8, 1.0), material=None):
        """Serialise a Mesh to single-file binary glTF (.glb) bytes (holographic_gltf, FWD-2): the boundary
        a three.js GLTFLoader consumes. POSITION/NORMAL/TEXCOORD_0/COLOR_0 attributes plus a triangle index
        buffer, with the glTF-required position bounds. `material` is an optional PBRMaterial (full
        metallic-roughness + emissive factors); without it, a default base-colour PBR material is written.
        Deterministic -- the same mesh yields byte-identical output."""
        from holographic_gltf import mesh_to_glb
        return mesh_to_glb(mesh, base_colour=base_colour, material=material)

    def pbr_material(self, name="material", base_color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.8,
                     emissive=(0.0, 0.0, 0.0)):
        """A standard glTF 2.0 metallic-roughness PBR material (the ISO factor model: base colour RGBA, metallic,
        roughness, emissive RGB) -- the representation every export/import maps through. See
        holographic_materialio.PBRMaterial."""
        from holographic_materialio import PBRMaterial
        return PBRMaterial(name=name, base_color=base_color, metallic=metallic, roughness=roughness,
                           emissive=emissive)

    def material_to_mtl(self, material):
        """A PBRMaterial as an MTL block (the OBJ companion): modern PBR keywords (Pr/Pm/Ke) + legacy Kd/d/Ns so
        it round-trips with PBR-aware tools and still opens in old viewers. See holographic_materialio."""
        return material.to_mtl()

    def materials_from_mtl(self, text):
        """Parse an MTL file's text into a list of PBRMaterial. See holographic_materialio.materials_from_mtl."""
        from holographic_materialio import materials_from_mtl
        return materials_from_mtl(text)

    def material_to_vsa_record(self, material, scalar_encoder):
        """Carry a PBRMaterial as ONE hypervector (each factor scalar-encoded, bound to its channel role, then
        bundled) -- so a material transmits, composes, and BLENDS with the engine's bind/bundle algebra, like a
        splat scene or a typed record. Round-trips to the encoder's resolution (crosstalk-limited; use the exact
        MTL/glTF path for lossless). See holographic_materialio.PBRMaterial.to_vsa_record."""
        return material.to_vsa_record(scalar_encoder)

    def material_from_vsa_record(self, record, scalar_encoder, name="material"):
        """Recover a PBRMaterial from its hypervector record (unbind each channel role, decode the scalar).
        See holographic_materialio.PBRMaterial.from_vsa_record."""
        from holographic_materialio import PBRMaterial
        return PBRMaterial.from_vsa_record(record, scalar_encoder, name=name)

    def mesh_from_gltf(self, data):
        """Parse binary glTF (.glb) bytes back into a Mesh (holographic_gltf, FWD-2): the inverse of
        `mesh_to_gltf`. Faces return as triangles (glTF's buffer form); positions, normals and uvs are
        recovered. The offline round-trip that proves the boundary is lossless before three.js ever sees it."""
        from holographic_gltf import glb_to_mesh
        return glb_to_mesh(data)

    # ---- FWD-7: local Euler EDIT operators -- the rewrites that let the modeler CHANGE a mesh ---------
    # The four workhorses every remesher/decimator decomposes into, on the shipped half-edge kernel. Each
    # returns a NEW Mesh, keeps the surface a valid manifold, and bookkeeps the Euler characteristic exactly;
    # the make/kill pairs (flip's self-inverse, split/collapse) give exact do-then-undo round-trips. Higher
    # ops (subdivide, bevel, decimate) are sequences of these and are later items.

    def mesh_flip_edge(self, mesh, a, b):
        """Rotate the edge shared by two triangles (holographic_eulerops, FWD-7): remove edge a-b, add edge
        c-d between the opposite apexes. V/E/F (hence chi) unchanged -- the Delaunay-remeshing primitive.
        Refuses if c-d already exists (would be non-manifold). Returns a new Mesh."""
        from holographic_eulerops import flip_edge
        return flip_edge(mesh, a, b)

    def mesh_split_edge(self, mesh, a, b):
        """Insert a midpoint vertex on edge {a,b}, splitting the incident triangle(s) (holographic_eulerops,
        FWD-7): the refinement primitive. V+1, chi unchanged. Returns (new_mesh, m) where m is the new vertex
        index; collapse_edge(new_mesh, keep=a, remove=m) is its exact inverse."""
        from holographic_eulerops import split_edge
        return split_edge(mesh, a, b)

    def mesh_collapse_edge(self, mesh, keep, remove):
        """Merge edge endpoint `remove` into `keep` (holographic_eulerops, FWD-7): the decimation/LOD
        primitive and the inverse of split_edge. V-1, chi unchanged. Returns a new Mesh, or None if the
        collapse would break the manifold (the LINK CONDITION) -- a true precondition the caller must handle."""
        from holographic_eulerops import collapse_edge
        return collapse_edge(mesh, keep, remove)

    def mesh_qem_decimate(self, mesh, target_faces):
        """QEM (quadric error metric) DECIMATION (holographic_meshqem, Garland-Heckbert): greedily collapse the
        lowest-cost edge -- cost = how far the collapse moves the surface, read from a per-vertex quadric (an
        accumulated BUNDLE of incident-plane constraints) -- via the guarded mesh_collapse_edge, until <=
        target_faces. Preserves features instead of eroding them; beats a naive shortest-edge collapse on surface
        error (~1.8x mean, ~3x max on a sphere). Returns a new Mesh. Kept negatives: closed meshes (open-boundary
        preservation deferred); minimizes plane-distance, not radius; halts above target if no safe collapse
        remains. Pair with mesh_surface_deviation to measure the result."""
        from holographic_meshqem import qem_decimate
        return qem_decimate(mesh, target_faces)

    def mesh_surface_deviation(self, mesh_a, mesh_b):
        """A decimation QUALITY metric (holographic_meshqem): (mean, max) point-to-surface distance from mesh_a's
        vertices to mesh_b's triangles -- how far one mesh's surface sits from the other's points. Use to measure
        how much a decimation (e.g. mesh_qem_decimate) moved the surface. Returns (mean, max)."""
        from holographic_meshqem import surface_deviation
        return surface_deviation(mesh_a, mesh_b)

    def mesh_lod_chain(self, mesh, targets=(0.5, 0.25, 0.125)):
        """Build a level-of-detail CHAIN (holographic_lod): QEM-decimate `mesh` to successively coarser levels at
        the given face-count fractions, measuring each level's surface deviation from the original. Returns a
        fine->coarse list of LODLevel(mesh, n_faces, mean_error, max_error); the first is the original (zero error).
        Pair with mesh_select_lod to choose a level by viewing distance."""
        from holographic_lod import build_lod_chain
        return build_lod_chain(mesh, targets=targets)

    def mesh_select_lod(self, chain, distance, pixel_threshold, screen_height_px=1080, fov_deg=60.0):
        """Choose a level of detail by SCREEN-SPACE ERROR (holographic_lod): the index of the coarsest level in
        `chain` whose error, projected to the screen at `distance`, stays under `pixel_threshold` -- the cheapest
        mesh that looks right. The engine's error-budget resolution selection (coarse_to_fine) carried to meshes:
        full detail up close, coarser far away. Returns an int index into the chain. Kept negative: the error is
        geometric surface deviation, not a perceptual/silhouette metric."""
        import math
        from holographic_lod import select_lod
        return select_lod(chain, distance, pixel_threshold, screen_height_px=screen_height_px,
                          fov_rad=math.radians(fov_deg))

    def mesh_cluster_decimate(self, mesh, grid=16):
        """PARALLEL decimation by vertex clustering (holographic_meshqem, Rossignac-Borrel / Lindstrom) -- the O(n)
        counterpart of the greedy mesh_qem_decimate, for an IMPORTED mesh with no field behind it. Bins vertices into
        a grid^3 spatial lattice (the engine's floor-divide tiling), collapses each cell to ONE representative, remaps
        faces, drops degenerate ones. Every step is a vectorized array op -- no greedy edge-collapse search -- so it
        runs hundreds-to-thousands x faster than QEM (a 22k-face mesh in tens of ms vs minutes). The representative is
        VSA-native: a cell's quadric is the SUM (a bundle, superposition) of its faces' plane tensors, and the
        representative is that bundle's minimizer, clamped to the cell. Returns a new Mesh. KEPT NEGATIVE: clustering
        trades quality and manifoldness for parallel speed (a coarse grid can go non-manifold) -- mesh_qem_decimate
        stays the quality option, this is the fast one. Higher `grid` = finer = more faces kept."""
        from holographic_meshqem import cluster_decimate
        return cluster_decimate(mesh, grid)

    def mesh_cluster_lod_chain(self, mesh, grids=(48, 24, 12)):
        """A fast PARALLEL level-of-detail chain (holographic_lod) for an IMPORTED mesh: vertex-cluster
        (mesh_cluster_decimate) at decreasing grid resolutions, measuring each level's deviation from the original.
        The O(n)-per-level counterpart of mesh_lod_chain (greedy QEM), for large meshes where the QEM chain is too
        slow. Returns a fine->coarse list of LODLevel(mesh, n_faces, mean_error, max_error); pair with
        mesh_select_lod. For a field-backed surface, prefer surface_mesh's FIELD-NATIVE LOD (re-march the source
        coarser) -- this is the path for a mesh that arrives with no field. Kept negative: inherits cluster_decimate's
        quality/manifoldness trade."""
        from holographic_lod import build_cluster_lod_chain
        return build_cluster_lod_chain(mesh, grids=grids)

    def mesh_to_field(self, mesh, bounds, res=48, band=None):
        """The mesh -> FIELD direction (holographic_meshbridge.mesh_distance_grid): decompose a mesh into a SIGNED
        banded distance field (a banded SDF) by TILING -- each triangle updates only the local block of grid voxels
        within `band` of it (a vectorized sub-array scatter-min by magnitude, the apply_local pattern), so the cost is
        O(F * block) not O(F * res^3). Signed (negative inside, by nearest face normal) so that |sample| near the
        surface is accurate to WELL UNDER a voxel -- an unsigned field's kink cannot resolve sub-voxel distances.
        Returns (grid res^3, (xs,ys,zs)). This is the gateway that lets an imported mesh be queried like a field:
        build once, then any number of points are O(V) samples (mesh_sample_field) -- e.g. a fast point-to-surface
        distance for decimation/LOD error.

        KEPT HONEST: the build is currently an F-triangle Python loop (~1ms/triangle) -- a batched closest-point would
        vectorize it (backlog); the nearest-normal sign can mis-sign deep concavities / non-watertight meshes
        (magnitude is always right); far interior voxels default to +band (no flood-fill sign yet), so this is a
        sample-near-the-surface field, not yet a re-marchable full SDF."""
        from holographic_meshbridge import mesh_distance_grid
        return mesh_distance_grid(mesh, bounds, res=res, band=band)

    def mesh_sample_field(self, grid, axes, points):
        """Trilinearly sample a banded SDF (from mesh_to_field) at query points (N,3) -> (N,) SIGNED distances; take
        abs for unsigned surface distance (holographic_meshbridge.sample_distance_grid). The O(V) read that turns a
        once-built field into a point-to-surface distance for any points -- the cheap query the brute O(Va*Fb) scan
        could not give once the field exists."""
        from holographic_meshbridge import sample_distance_grid
        return sample_distance_grid(grid, axes, points)

    def mesh_to_sdf_grid(self, mesh, bounds, res=48, band=None):
        """Convert an imported mesh into a FULL, re-marchable signed distance field
        (holographic_meshbridge.mesh_to_sdf_grid): the banded SDF by tiling (mesh_to_field) followed by a flood fill
        that fills the interior negative. Unlike mesh_to_field (a sample-near-the-surface band), this is a complete
        SDF with the surface as a true zero level set -- so the mesh can be re-marched at any resolution, sampled, or
        composited like any other field. Returns (grid res^3, (xs,ys,zs)). This is the gateway that lets an imported
        mesh JOIN the field-native world. KEPT HONEST: nearest-normal sign + needs a watertight band (>= ~2 voxels)."""
        from holographic_meshbridge import mesh_to_sdf_grid
        return mesh_to_sdf_grid(mesh, bounds, res=res, band=band)

    def mesh_to_field_vector(self, mesh, bounds, dim=2048, bandwidth=18.0, grid=12, seed=0):
        """FS-5: carry a surface as a SINGLE hypervector (edit = bind). Samples the mesh's signed distance on a
        coarse `grid`^3 lattice over `bounds` and bundles it into one FPE vector (HolographicField). On the result:
        `.value(points)` reads the (smoothed) signed field, `.translate(delta)` moves the WHOLE surface with one
        binding (exact -- value_shifted(x)=value_orig(x-delta)), `.union(other)` merges two surfaces by bundling, and
        `.surface(bounds, res)` re-extracts a mesh from the 0-level. This is the array-domain mesh_to_field's
        hypervector cousin -- moving and merging become VSA algebra. KEPT HONEST: a demonstration representation, not
        the fast path (FFT-bound build/extract); the marched extract is a smoothed, ~15%-biased, not-guaranteed-
        watertight estimate (bandwidth is the bias knob, dim the noise floor); the encoder bounds must exceed
        |sample|+|shift| or the FPE wraps; valid only within the sampled cloud. See holographic_fpefield."""
        from holographic_fpefield import HolographicField
        return HolographicField.from_mesh(mesh, bounds, dim=dim, bandwidth=bandwidth, grid=grid, seed=seed)

    def mesh_point_distance(self, mesh, points, radius=2, signed=False):
        """Distance from query points (N,3) to a mesh, ACCELERATED by a vectorized spatial grid that culls the work
        (holographic_meshbridge.point_set_to_mesh_grid) -- ~20-110x faster than the brute O(N*F) scan and exact for
        near-surface queries (the regime that matters: decimation/LOD error, contact, snapping). Returns (N,) unsigned
        distance, or signed (negative inside) if `signed`. KEPT HONEST: APPROXIMATE by construction -- a query whose
        nearest triangle lies beyond `radius` cells returns +inf (raise `radius`, or use the exact brute path for
        far-field queries); see point_set_to_mesh_grid."""
        from holographic_meshbridge import point_set_to_mesh_grid
        return point_set_to_mesh_grid(points, mesh.vertices, mesh.faces, radius=radius, signed=signed)

    def mesh_field_lod(self, mesh, bounds, res=64, strides=(1, 2, 4)):
        """FIELD-NATIVE level-of-detail for an IMPORTED mesh (the decomposition closure): convert it to a full SDF
        once (mesh_to_sdf_grid), then RE-MARCH that field at coarser strides -- so the imported mesh coarsens exactly
        like a field-backed surface, no mesh decimation in the loop. Returns a fine->coarse list of meshes. This is
        the same field-native LOD that surface_mesh gives a native field, now reached by a mesh that arrived with no
        field. KEPT HONEST: quality is bounded by the SDF grid resolution and the nearest-normal sign; for a mesh that
        is already field-backed, use surface_mesh directly (no conversion needed)."""
        from holographic_meshbridge import mesh_to_sdf_grid, marching_tetrahedra_vec
        grid, axes = mesh_to_sdf_grid(mesh, bounds, res=res)
        out = []
        for s in strides:
            s = int(s)
            sub = grid[::s, ::s, ::s]
            subax = (axes[0][::s], axes[1][::s], axes[2][::s])
            out.append(marching_tetrahedra_vec(sub, subax, 0.0))
        return out

    def oct_encode_normals(self, normals, bits=8):
        """OCTAHEDRAL-encode unit normals to compact integer codes (holographic_octnormal, Cigolle et al. 2014):
        map each S^2 unit vector to 2 numbers (project onto the octahedron, unfold the lower hemisphere) and
        quantize to `bits` per component -- spending the budget on the sphere's 2 intrinsic DOF, not 3 ambient
        x/y/z. Returns integer codes (N,2) in [0, 2^bits). At equal storage this beats naive xyz quantization ~3x
        (the manifold-quantization win, reverse item R3's S^2 instance). Decode with oct_decode_normals. Kept
        negative: this is the S^2 map specifically -- the PRINCIPLE generalizes to FHRR phasors (phase angle) and
        normalized codes, the literal map does not."""
        from holographic_octnormal import oct_quantize
        return oct_quantize(normals, bits=bits)

    def oct_decode_normals(self, codes, bits=8):
        """Decode octahedral integer codes (N,2) back to unit normals (N,3) (holographic_octnormal). Inverse of
        oct_encode_normals."""
        from holographic_octnormal import oct_dequantize
        return oct_dequantize(codes, bits=bits)

    def mesh_split_face(self, mesh, f_index, i, j):
        """Cut polygon face `f_index` with a diagonal between its i-th and j-th corners (holographic_eulerops,
        FWD-7): MEF, the one operator that works on n-gons, not just triangles. E+1, F+1, chi unchanged.
        Returns a new Mesh."""
        from holographic_eulerops import split_face
        return split_face(mesh, f_index, i, j)

    def mesh_smooth(self, mesh, lam=0.55, mu=-0.58, iters=8, weights="cotangent"):
        """Taubin lambda|mu no-shrink mesh smoothing / denoising (holographic_meshsmooth, FWD-4): the shipped
        `graphsignal.taubin_filter` WIRED onto explicit mesh geometry -- 3-D vertex positions as the signal,
        the mesh's 1-ring as the graph, cotangent (discrete Laplace-Beltrami) weights by default. Removes
        vertex noise while preserving the overall extent (no volume shrink, unlike a naive Laplacian). Faces
        are untouched, so connectivity and every Euler invariant are preserved -- only vertices move. Returns a
        new Mesh. `weights='uniform'` for the umbrella baseline. Kept negative: fixed strength over-smooths an
        already-clean mesh (it is a low-pass) -- needs a noise estimate; this exposes lam/mu/iters, no auto-tune."""
        from holographic_meshsmooth import taubin_smooth
        return taubin_smooth(mesh, lam=lam, mu=mu, iters=iters, weights=weights)

    def mesh_curvature(self, mesh, kind="mean"):
        """Per-vertex curvature of a mesh (holographic_meshcurvature, FWD-6). kind='mean' -> |H| via the
        cotangent Laplacian of positions (reusing FWD-4's weights; H=1/R on a sphere); kind='gaussian' -> the
        area-normalised angle defect (K=1/R^2 on a sphere), whose TOTAL equals 2*pi*chi exactly by discrete
        Gauss-Bonnet. Returns an array of length V. Kept negative: per-vertex values are noisy on coarse/
        irregular meshes (the mean is accurate); `mesh_curvature_confidence` scores per-vertex reliability."""
        from holographic_meshcurvature import mean_curvature, gaussian_curvature
        if kind == "mean":
            return mean_curvature(mesh)
        if kind == "gaussian":
            return gaussian_curvature(mesh)
        raise ValueError(f"kind must be 'mean' or 'gaussian', got {kind!r}")

    def mesh_curvature_confidence(self, mesh):
        """A per-vertex confidence in [0,1] for the curvature estimate (holographic_meshcurvature, FWD-6), from
        1-ring regularity -- low where the neighbourhood is sliver-heavy or sparse, so a caller can down-weight
        unreliable curvature rather than trust it (the noise kept-negative made actionable)."""
        from holographic_meshcurvature import curvature_confidence
        return curvature_confidence(mesh)

    def mesh_creases(self, mesh, threshold_deg=30.0):
        """The sharp edges of a mesh (holographic_meshcurvature, FWD-6): interior edges whose DIHEDRAL angle
        (between the two adjacent faces) exceeds `threshold_deg`. A cube returns its 12 edges; a smooth sphere
        returns none. Feeds crease-aware smoothing, adaptive subdivision, and shading-normal splitting. Returns
        a sorted list of (lo,hi) vertex-index edges."""
        from holographic_meshcurvature import detect_creases
        return detect_creases(mesh, threshold_deg=threshold_deg)

    def mesh_geodesic(self, mesh, source):
        """Single-source surface geodesic distance (holographic_meshgeodesic, FWD-5): shortest path ALONG mesh
        edges from vertex `source` to every vertex (Dijkstra with Euclidean edge weights) -- distance over the
        surface, not the straight line through the void. Returns an array of length V. The along-surface metric
        UV seams (FWD-3), soft selections, and remesh spacing all want. Kept negative: the edge-graph distance
        is approximate (a few percent over the smooth geodesic, with a tiny chord-effect undercut near source)."""
        from holographic_meshgeodesic import geodesic_distances
        return geodesic_distances(mesh, source)

    def mesh_soft_selection(self, mesh, source, radius, falloff="smooth"):
        """A soft-selection falloff in [0,1] per vertex by GEODESIC distance from `source` within `radius`
        (holographic_meshgeodesic, FWD-5): 1 at the source, smoothly to 0 at the radius. Because it measures
        along the surface it does NOT bleed to vertices near in 3-D space but far across the surface (a Euclidean
        brush's failure on a thin neck or a folded region). `falloff`: 'smooth' or 'linear'. Returns length-V."""
        from holographic_meshgeodesic import geodesic_soft_selection
        return geodesic_soft_selection(mesh, source, radius, falloff=falloff)

    def mesh_uv_unwrap(self, mesh, method="isomap"):
        """UV-unwrap a (disk-topology) mesh to 2-D coordinates (holographic_meshuv, FWD-3): classical MDS of the
        mesh's GEODESIC distance matrix -- Isomap on explicit edges -- so surface distances are preserved as well
        as a plane allows. The shipped manifold-chart machinery pointed at real mesh connectivity. Returns (V,2)
        UV in ~[0,1]^2. `method`: 'isomap' (geodesic; wins on curved surfaces), 'planar' (linear; exact on
        developable surfaces), 'spectral' (Laplacian eigenmaps). Kept negative: a CLOSED surface needs a seam
        (cut) first -- `holographic_meshuv.puncture` opens it crudely; a real seam is the ARCH-4 atlas piece."""
        from holographic_meshuv import uv_unwrap
        return uv_unwrap(mesh, method=method)

    def mesh_stable_uv(self, mesh, bounds=None, mode="triplanar", axis=2):
        """UVs that are a deterministic function of WORLD POSITION, so they DON'T move under local edits -- the
        stable counterpart to mesh_uv_unwrap (whose global MDS/eigenmap re-solves and can flip on any edit).
        Normalised by the FIXED `bounds` (field domain) so the scale is edit-invariant. mode='triplanar' picks
        each vertex's projection plane by its normal (curves don't fold); 'planar' drops `axis`. Stable texturing,
        not a seam-cut chart. See holographic_meshuv.stable_uv."""
        from holographic_meshuv import stable_uv
        return stable_uv(mesh, bounds=bounds, mode=mode, axis=axis)

    def mesh_face_type(self, mesh, face_type="quad", planarity=0.90, normal_tol=0.999):
        """Convert ANY triangle mesh's face standard (holographic_meshpoly) WITHOUT moving vertices (so stable
        keys survive): 'quad' merges coplanar triangle pairs into quad-dominant output; 'ngon' merges connected
        coplanar regions into single n-gons where the boundary is a clean loop (a flat wall -> one face);
        'triangle' returns it unchanged. Use after surface_mesh / on an imported mesh. See holographic_meshpoly."""
        if face_type == "quad":
            from holographic_meshpoly import triangles_to_quads
            return triangles_to_quads(mesh, planarity=planarity)
        if face_type == "ngon":
            from holographic_meshpoly import merge_coplanar
            return merge_coplanar(mesh, normal_tol=normal_tol)
        return mesh

    def mesh_face_counts(self, mesh):
        """{3: triangles, 4: quads, 5: n-gons} -- the face-standard summary. See holographic_meshpoly."""
        from holographic_meshpoly import face_type_counts
        return face_type_counts(mesh)

    def mesh_uv_distortion(self, mesh, uv):
        """The per-edge STRETCH distortion of a UV map (holographic_meshuv, FWD-3): spread of (UV edge length /
        3-D edge length), 0 = isometric, growing with Gaussian curvature (Gauss: a curved surface cannot flatten
        without stretch). The scale-invariant flatness score -- lower is a better parameterisation."""
        from holographic_meshuv import uv_distortion
        return uv_distortion(mesh, uv)

    def mesh_cut_seam(self, mesh, seam):
        """Cut a mesh open along a SEAM (holographic_meshseam, ARCH-4): given `seam` (an ordered list of vertex
        indices forming an edge path), duplicate the seam's interior vertices on a consistent side, opening a
        closed surface into a disk -- the REAL seam that FWD-3's crude `puncture` stood in for. Non-destructive
        (every face preserved, unlike puncture which deletes faces). A meridian-cut sphere becomes a disk (chi=1).
        Returns a new Mesh. Kept negative: seam CHOICE matters (a full pole-to-pole meridian unwraps worse than a
        pole-to-equator cut); a good atlas needs several cuts (deferred)."""
        from holographic_meshseam import cut_seam
        return cut_seam(mesh, seam)

    def mesh_shortest_seam(self, mesh, a, b):
        """A seam path between two vertices (holographic_meshseam, ARCH-4): the shortest edge path from `a` to `b`
        (Dijkstra on the mesh edge graph), e.g. a meridian from a pole to its antipode. Returns an ordered list of
        vertex indices to hand to mesh_cut_seam."""
        from holographic_meshseam import shortest_seam
        return shortest_seam(mesh, a, b)

    def mesh_extrude(self, mesh, face_index, distance):
        """EXTRUDE a face (holographic_meshverbs, FWD-7): lift face `face_index` along its outward normal by
        `distance` and wall it in -- the iconic modelling verb. Preserves the Euler characteristic and keeps a
        closed mesh a closed manifold; the cap moves exactly `distance` along the normal. Returns a new Mesh."""
        from holographic_meshverbs import extrude_face
        return extrude_face(mesh, face_index, distance)

    def mesh_inset(self, mesh, face_index, ratio):
        """INSET a face (holographic_meshverbs, FWD-7): shrink face `face_index` toward its centroid by `ratio`,
        ringing it with new faces around a smaller central face (in-plane, so the central face stays coplanar; its
        area is exactly (1-ratio)^2 of the original). Preserves chi + closed + manifold. Returns a new Mesh."""
        from holographic_meshverbs import inset_face
        return inset_face(mesh, face_index, ratio)

    def mesh_dissolve_vertex(self, mesh, vertex):
        """DISSOLVE a vertex (holographic_meshverbs, FWD-7; the Euler KEV verb): remove `vertex` and its incident
        faces, then fan-triangulate the hole, leaving the surrounding ring fixed. Preserves chi + closed +
        manifold, one fewer vertex. Distinct from `mesh_collapse_edge` (the decimation cousin). Returns a Mesh."""
        from holographic_meshverbs import dissolve_vertex
        return dissolve_vertex(mesh, vertex)

    def mesh_bevel_vertex(self, mesh, vertex, ratio=0.25):
        """BEVEL a corner (holographic_meshverbs2, FWD-7 remainder): pull each edge incident to `vertex` back toward
        its neighbour by `ratio`, chamfer every incident face, and cap the hole with a new face -- the corner
        becomes a small facet. Preserves chi + closed + manifold. Returns a Mesh. Kept negative: this is the VERTEX
        bevel; the EDGE bevel (two-sided fan split) is deferred; ratio must be in (0,1)."""
        from holographic_meshverbs2 import bevel_vertex
        return bevel_vertex(mesh, vertex, ratio=ratio)

    def mesh_bridge(self, verts, loop_a, loop_b, closed=True):
        """BRIDGE two edge loops (holographic_meshverbs2, FWD-7 remainder): join two equal-length ordered vertex
        loops with a band of quads -- the verb that builds a tube between two openings. `verts` holds all points;
        `loop_a`/`loop_b` are equal-length index lists. Returns a Mesh of the band. Kept negative: requires
        equal-length, already-aligned loops (the caller supplies the correspondence); matching unequal loops is
        deferred."""
        from holographic_meshverbs2 import bridge_loops
        return bridge_loops(verts, loop_a, loop_b, closed=closed)

    def mesh_loop_cut(self, mesh, start_face, start_edge):
        """LOOP-CUT: insert an edge loop (holographic_meshverbs2, FWD-7 remainder): trace the perpendicular loop of
        quads (enter through one edge, leave through the OPPOSITE edge) carrying `start_edge` of quad `start_face`,
        and split every crossed quad in two with a new mid-loop -- the verb that adds a ring of resolution.
        Preserves chi. Returns a Mesh. Kept negative: QUADS only (the opposite-edge trace is undefined on
        triangles); the trace stops at a boundary or when it returns to the start."""
        from holographic_meshverbs2 import loop_cut
        return loop_cut(mesh, start_face, start_edge)

    def mesh_subdivide(self, mesh, levels=1):
        """Loop-SUBDIVIDE a triangle mesh (holographic_meshsubdiv, FWD-8): refine each triangle into four and
        low-pass smooth with the Loop masks (a C2 limit surface). Two operations braided -- a topological refine
        (an Euler-operator sequence, the new part) and a graph-signal low-pass smooth (the spectral family FWD-4
        also uses). Each level multiplies faces by 4, adds one vertex per edge, preserves chi, and keeps a closed
        mesh a closed manifold; a flat mesh stays flat exactly, a curved one is smoothed toward the Loop limit.
        A non-triangle input is triangulated first (Loop is a triangle scheme). Returns a new Mesh."""
        from holographic_meshsubdiv import loop_subdivide
        return loop_subdivide(mesh, levels=levels)

    def solve_ik(self, joints, target, iters=20, tol=None):
        """Inverse kinematics by FABRIK (holographic_meshik, FWD-10): move a chain of `joints` (n+1, 3) so the
        end-effector reaches `target`, keeping every bone's rest length and the root fixed. Implemented LITERALLY
        through this mind's own `project_onto_constraints` engine -- FABRIK's forward/backward reaching IS a
        Gauss-Seidel sweep of bone-length + endpoint-pin projections, so IK is the iterate-a-projection faculty in
        the kinematic-chain costume. Returns (new_joints (n+1,3), n_sweeps). For an UNREACHABLE target the chain
        fully extends toward it (the honest degenerate outcome). Kept negative: plain FABRIK has no joint-angle
        limits -- a per-joint cone projection would slot into the same sweep but is not shipped here."""
        from holographic_meshik import solve_ik as _solve_ik
        return _solve_ik(joints, target, iters=iters, tol=tol)

    def skin_mesh(self, mesh, transforms, weights):
        """Linear-blend-SKIN a mesh (holographic_meshskin, FWD-9): deform each vertex as the weighted combination
        of what each bone transform would do to it -- v' = sum_b w_b (M_b v), weights a partition of unity. This
        is a SOFT mixture of expert bone-transforms (the soft/dense cousin of this engine's hard/sparse top-1
        `moe.GatedMixture` -- same experts+gating skeleton, different gating regime). `transforms` is (B,4,4),
        `weights` is (V,B). Returns a new Mesh (deformed vertices, faces untouched). A shared rigid transform is
        reproduced exactly. Kept negative: LBS averages matrices not rotations, so a 50/50 twist collapses the
        radius to cos(theta/2) (the candy-wrapper artifact) -- dual-quaternion skinning is the fix, not shipped."""
        from holographic_meshskin import skin_mesh as _skin_mesh
        return _skin_mesh(mesh, transforms, weights)

    def blend_pose(self, targets, weights):
        """The forward blendshape/skinning map for STRUCTURES (holographic_blendpose, ARCH-6): a soft weighted blend
        of pose-target structures, normalize(sum_i w_i targets_i) -- FWD-9's soft mixture, one rung up (mixing whole
        structures, not transforms). `targets` is (m,dim), `weights` (m,). Returns the (dim,) pose. Paired with
        solve_pose (the inverse)."""
        from holographic_blendpose import blend_pose
        return blend_pose(targets, weights)

    def solve_pose(self, targets, goal, iters=400):
        """Inverse kinematics for a blendshape rig of STRUCTURES (holographic_blendpose, ARCH-6): solve the blend
        weights so blend_pose(targets, w) reaches `goal`, by handing two constraints to the SAME
        project_onto_constraints sweeper FWD-10 used for FABRIK -- FIT the goal (least-squares step, FABRIK's reach)
        and stay a VALID CONVEX BLEND (simplex projection, FABRIK's length constraint). The blend weights are the
        'joint angles'. Returns the weights (a valid convex blend). Exact when the goal is a blend of the targets;
        the CLOSEST valid blend otherwise (residual <= any single target). Kept negative: a goal outside the
        targets' convex blend is unreachable -- that needs a richer rig (more targets), not a better solver."""
        from holographic_blendpose import solve_pose
        return solve_pose(targets, goal, iters=iters)

    def mesh_from_sdf(self, sdf, bounds, res=24, level=0.0, vectorized=False):
        """Extract a MESH from an implicit field (holographic_meshbridge, FWD-11; SDF -> mesh): sample the scalar
        field `sdf` (a callable points(N,3)->values, e.g. `sphere_sdf` or a `metaball_field` of Gaussian splats) on
        a res^3 grid over `bounds`=((x0,y0,z0),(x1,y1,z1)) and extract its `level` isosurface by MARCHING
        TETRAHEDRA -- the isosurface extractor the mesh kernel deliberately lacked, and the bridge that lets the
        engine's implicit/splat representations enter the mesh world. The result is a watertight, outward-oriented
        triangle Mesh (a closed genus-0 field gives chi=2). vectorized=True uses the parallel array-op marcher
        (marching_tetrahedra_vec, the case-table-RAM path) -- geometrically identical, ~6-14x faster at working grid
        sizes (default False keeps the per-cell marcher's exact vertex ordering for backward compatibility). Kept
        negative: resolution is the grid's -- features below the cell size are rounded. Returns a Mesh."""
        from holographic_meshbridge import sample_field, marching_tetrahedra, marching_tetrahedra_vec
        values, axes = sample_field(sdf, bounds, res)
        march = marching_tetrahedra_vec if vectorized else marching_tetrahedra
        return march(values, axes, level=level)

    def surface_mesh_stable(self, field, bounds, resolution=40, level=0.0, validate=True, face_type="triangle"):
        """Project a field to a mesh with STABLE vertex identity and a topology guarantee -- the entry point for
        a 3-D modeling app that needs faces/edges/verts that DON'T move on their own. Returns a dict:
          'mesh'     -- the watertight, 2-manifold Mesh.
          'keys'     -- a STABLE per-vertex identity array: keys[v] is the grid edge the vertex sits on, the SAME
                        integer in any extraction at this (resolution, bounds). Track vertices by KEY across
                        edits, NOT by array index -- a local edit renumbers indices (a crossing added/removed
                        shifts everything after it in sorted-key order) but never changes keys, so unchanged
                        geometry keeps its identity and the user sees no phantom movement of distant faces.
          'topology' -- (when validate) the validate_topology() report (manifold edges+verts, watertight, genus).
        `face_type` chooses the face standard projected out (marching tetrahedra emits triangles; quads/ngons are
        a deterministic merge ON TOP, leaving vertices -- and their keys -- untouched):
          'triangle' -- the raw marched triangles (default).
          'quad'     -- quad-DOMINANT: adjacent coplanar triangle pairs merged into convex quads, leftovers stay
                        triangles (triangles_to_quads).
          'ngon'     -- connected coplanar faces merged into single n-gons where the region's boundary is a clean
                        loop (merge_coplanar) -- a flat wall becomes one face, curved areas keep triangles.
        GUARANTEE: exact-corner field samples (the one case that makes marching tetrahedra non-manifold, where a
        vertex would land on a shared grid corner) are nudged a deterministic epsilon off the level so the
        crossing lands on the edge interior instead -- yielding clean 2-manifold output. Keys are tied to the
        grid, so they are stable across EDITS at a fixed resolution, not across resolution changes (a different
        resolution is a different mesh by definition). KEPT HONEST: the face GROUPING (which tris became a quad)
        is itself NOT edit-stable -- faces are a derived view; the vertices are the stable identity."""
        from holographic_meshbridge import sample_field, marching_tetrahedra_vec
        import numpy as _np
        values, axes = sample_field(field, bounds, resolution)
        scale = float(_np.abs(values).mean()) or 1.0
        exact = (values == level)
        if exact.any():
            values = values.copy()
            values[exact] = level - 1e-9 * scale               # push exact-corner hits just inside -> no bowtie
        mesh, keys = marching_tetrahedra_vec(values, axes, level=level, return_keys=True)
        if face_type == "quad":
            from holographic_meshpoly import triangles_to_quads
            mesh = triangles_to_quads(mesh)                    # vertices unchanged -> keys still valid
        elif face_type == "ngon":
            from holographic_meshpoly import merge_coplanar
            mesh = merge_coplanar(mesh)
        out = {"mesh": mesh, "keys": keys}
        if validate:
            out["topology"] = mesh.validate_topology()
        return out

    def validate_topology(self, mesh):
        """The full well-formedness report for a mesh (manifold edges AND vertices/bowties, watertight,
        degenerate faces, euler/genus) -- the guarantee a 3-D modeling app gates on before handing a mesh to a
        user. See Mesh.validate_topology."""
        return mesh.validate_topology()

    def mesh_to_sdf(self, mesh, points):
        """Signed distance from a MESH at query points (holographic_meshbridge, FWD-11; mesh -> implicit): the
        unsigned distance to the nearest triangle, signed by the nearest face normal (negative inside). The reverse
        of mesh_from_sdf, completing the mesh<->SDF bridge. `points` is (N,3); returns (N,). Kept negative: the
        nearest-normal sign is exact for convex-ish closed meshes but can mis-sign deep concavities (the magnitude
        is always right; a generalized winding number is the fix, not shipped)."""
        from holographic_meshbridge import mesh_to_sdf as _mesh_to_sdf
        return _mesh_to_sdf(mesh, points)

    def sculpt(self, field_fn, kind, p, radius, strength=0.3, **kw):
        """SCULPT a field with a falloff-weighted brush (holographic_sculpt, FS-1) -- a local edit of a field
        FUNCTION (vectorized: P of shape (N,3) -> (N,)) in a ball around point `p`, returning the EDITED field
        function. `kind` is one of inflate / carve / smooth / grab / flatten / pinch (grab needs drag=..., flatten
        needs level=...). Because a surface is carried as a field whose level-set IS the surface, sculpting the field
        and RE-EXTRACTING (marching_tetrahedra) gives resolution-independent topology editing -- the move a fixed mesh
        cannot do. The brush leaves the field BIT-IDENTICAL outside the ball (the falloff is exactly 0 past the
        radius), so the surface changes only where you brushed and the re-extract stays watertight/manifold. Works on
        ANY field, not just the surface SDF: the same operator reshapes the creature's value landscape (reward
        shaping) or a density/memory field -- the radius+falloff the bare `reinforce` lacks. KEPT HONEST: on a DENSE
        field the re-extract is still O(res^3) per stroke -- the narrow-band sparse field (the next FS item) is what
        makes a stroke cost O(brush). Delegates to holographic_sculpt.apply_brush."""
        from holographic_sculpt import apply_brush
        return apply_brush(field_fn, kind, p, radius, s=strength, **kw)

    def sparse_field(self, field_fn, bounds, voxel, band, tile=8):
        """Build a NARROW-BAND SPARSE field from a dense field function (holographic_sparsefield, FS-2) -- store,
        edit, and re-extract only the thin shell of voxels around the surface (|f| < band), so a brush stroke touches
        O(brush) voxels instead of O(res^3) and only the dirtied bricks re-mesh. This is what turns FS-1 sculpting
        from batchy into interactive. `field_fn(points (N,3)) -> values (N,)` is the SDF (negative inside), `bounds`
        = (min_corner, max_corner), `voxel` the cell size, `band` the half-width to store, `tile` voxels per brick.
        Returns a SparseField; call .apply_local(delta_fn, p, r) to edit (returns dirty bricks + touched count),
        .reinitialize() to restore the signed-distance property after edits, and .extract_local(dirty_bricks) to
        re-mesh only the dirty region (welded watertight). KEPT HONEST: the per-brick marching is pure Python and
        belongs on the GPU past a few hundred active bricks (holostuff is the authoring brain -- the band bookkeeping
        and SDF numerics -- the per-frame voxel grind is the GPU's muscle); the band must be reinitialized or
        distances drift; topology growth into unseeded interior space needs a re-seed. Delegates to
        holographic_sparsefield.SparseField.from_field."""
        from holographic_sparsefield import SparseField
        return SparseField.from_field(field_fn, bounds[0], bounds[1], voxel, band, tile=tile)

    def surface_mesh(self, field, bounds=None, resolution=24, level=0.0, pixel_budget=None, distance=1.0,
                     lod_targets=(0.5, 0.25, 0.125), screen_height_px=1080, fov_deg=60.0, cache=False):
        """THE SCULPT LOOP'S RE-EXTRACT STEP (FS-4): turn ANY field representation into the drawable mesh, at the right
        detail for the view -- one entry point composing the parts FS-1..FS-3 shipped. `field` is either a field
        FUNCTION (points(N,3)->values; needs `bounds`=(min,max)) OR a SparseField from sparse_field().

        Without a `pixel_budget`, the full-resolution surface is projected and returned.

        With a `pixel_budget`, LOD is FIELD-NATIVE: rather than projecting the fine mesh and then QEM-DECIMATING it
        (greedy edge collapse -- O(collapses*edges), seconds), the SOURCE FIELD is coarsened (re-marched at a coarser
        grid stride / resolution) and re-projected, and the COARSEST level whose screen-space error at `distance`
        stays under the budget is returned. This is the whole thesis made load-bearing: the mesh is a PROJECTION of
        the field, so a coarser mesh is a coarser field projected -- measured ~thousands x faster than decimating the
        projection, because re-marching is a vectorized pass and the per-level error is read straight from the field
        (the full-res field value at a coarse vertex IS its distance to the true surface -- O(V) field samples, no
        O(V*F) mesh-to-mesh distance, no greedy collapse). The legacy QEM LOD (mesh_lod_chain) remains a separate
        faculty for an IMPORTED mesh that has no field behind it.

        cache=True (SparseField, no budget) uses the brick-mesh WORKING-SET CACHE (the ReflexCache idea: re-mesh only
        the bricks a stroke dirtied). The field's edits must go through apply_local; else call its cache_clear().

        This NAMES THE LOOP as iterate-a-projection -- each sculpt step EDITS the field then RE-PROJECTS to the
        surface, the same shape as the resonator/denoiser/dynamics. sculpt -> surface_mesh -> export_splats is the
        authoring cycle; the field is the source of truth, the mesh and splats are two projections. KEPT HONEST: the
        per-level error is the field-distance at the marched vertices (a true geometric deviation), not a
        silhouette/perceptual metric; the function-field path is dense O(res^3) per level (use the SparseField path
        for the interactive, O(brush) loop). `lod_targets` is retained for signature compatibility; the field-native
        LOD coarsens by RESOLUTION STRIDE, not face fraction."""
        from holographic_sparsefield import SparseField
        is_sparse = isinstance(field, SparseField)
        if not is_sparse and bounds is None:
            raise ValueError("a field FUNCTION needs bounds=(min_corner, max_corner)")
        # No budget: project the field once, at full resolution.
        if pixel_budget is None:
            if is_sparse:
                return field.extract_cached() if cache else field.extract_local()
            return self.mesh_from_sdf(field, bounds, res=resolution, level=level, vectorized=True)
        # Budget: FIELD-NATIVE LOD -- coarsen the SOURCE and re-project, never decimate the projection.
        if is_sparse:
            chain = field.lod_chain()
        else:
            chain = self._function_lod_chain(field, bounds, resolution, level)
        idx = self.mesh_select_lod(chain, distance, pixel_budget, screen_height_px=screen_height_px, fov_deg=fov_deg)
        return chain[idx].mesh

    def _function_lod_chain(self, field, bounds, resolution, level):
        """Field-native LOD for a field FUNCTION: re-sample + march at decreasing resolutions (coarsen the source),
        with each level's error read from the function at the coarse vertices (|f(vertex) - level| = the vertex's
        distance to the true level set, for an SDF). Returns a fine->coarse list of LODLevel for select_lod -- the
        same 'coarsen the source, project, read the error from the field' move as SparseField.lod_chain."""
        from holographic_lod import LODLevel
        from holographic_meshbridge import sample_field, marching_tetrahedra_vec
        import numpy as _np
        levels = []
        res = int(resolution)
        for r in (res, res // 2, res // 4):
            if r < 2:
                break
            vals, axes = sample_field(field, bounds, r)
            m = marching_tetrahedra_vec(vals, axes, level=level)
            if m.n_faces == 0:
                continue
            if levels and m.n_faces >= levels[-1].n_faces:
                continue
            if not levels:
                mean_e = max_e = 0.0
            else:
                d = _np.abs(_np.asarray(field(m.vertices), float) - level)    # field deviation at marched vertices
                mean_e, max_e = float(d.mean()), float(d.max())
            levels.append(LODLevel(m, m.n_faces, mean_e, max_e))
        return levels

    def route_representation(self, operation):
        """The routing POLICY (holographic_route, ARCH-7): the representation whose capability set supports
        `operation` -- e.g. booleans ("union"/"intersection"/"difference") route to "sdf" (field min/max), explicit
        "boundary"/"render" to "mesh", soft "blend" to "splat". The decision layer on top of the FWD-11 bridge."""
        from holographic_route import representation_for
        return representation_for(operation)

    def mesh_csg(self, operation, mesh_a, mesh_b, res=28, bounds=None):
        """Boolean of two solids by ROUTING through the SDF (holographic_route, ARCH-7's flagship): `operation` in
        {"union","intersection","difference"}. The mesh kernel has no native booleans; this routes mesh_a, mesh_b
        -> SDF (mesh_to_sdf), combines the fields (min / max / max-with-negation), and extracts back to a mesh
        (marching tetrahedra). The result can CHANGE TOPOLOGY -- overlapping solids merge to one component, separate
        ones stay two -- which a mesh cannot do to itself. Geometrically correct (satisfies inclusion-exclusion).
        Returns a Mesh. Kept negative: resolution is the grid's (sharp seams round); the input meshes' SDF sign must
        be reliable (convex-ish), per FWD-11."""
        from holographic_route import route_csg
        return route_csg(operation, mesh_a, mesh_b, res=res, bounds=bounds)

    def mesh_volume(self, mesh):
        """The enclosed volume of a closed mesh (holographic_route, ARCH-7; divergence theorem over outward-wound
        triangles). Used to verify CSG booleans are geometrically -- not just topologically -- correct."""
        from holographic_route import mesh_volume
        return mesh_volume(mesh)

    def mesh_connected_components(self, mesh):
        """The number of connected components of a mesh (holographic_route, ARCH-7; flood fill over edge adjacency).
        A CSG union of overlapping solids has 1; of separate solids, 2."""
        from holographic_route import connected_components
        return connected_components(mesh)

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

    def kinematics(self, dim=None, lo=-50.0, hi=50.0, seed=1):
        """CLOSED-FORM KINEMATICS on the substrate (holographic_physics, Kinematics) -- physics as an algebra of
        binds. Position advances by velocity as ONE binding (x += v is bind(state_x, state_v)), acceleration advances
        velocity the same way, and the velocity BETWEEN two observed positions is read by UNBIND (bind x_b with the
        involution of x_a, decode). The direct embodiment of the engine's core thesis, 'binding is a rigid shift,'
        pointed at motion -- the Stam/Macklin seats' territory. This is the CLOSED-FORM twin of learn_dynamics
        (Propagator), which LEARNS its operator from data; here the operator is the encoder's own shift, exact by
        construction rather than fitted. Returns a Kinematics over [lo, hi] (the mind's dim by default): state(x),
        step(S_x, S_v), trajectory(x0, v0, a, steps) -- integrate by pure binding and decode each position, which
        RAISES if the true path leaves the encoder's range (the honest boundary) -- and read_velocity(x_a, x_b).
        Delegates to holographic_physics."""
        from holographic_physics import Kinematics
        return Kinematics(dim=dim or self.dim, lo=lo, hi=hi, seed=seed)

    # ---- the GENERATIVE faculties (integration plan, Tier 4) -----------------------------------
    # Generation is denoising run backwards, and a splat scene is a bundle -- so the last two modules
    # built beside the mind reconcile straight into it: generate a vector by the cleanup-attractor
    # diffusion, and represent a 2-D field as a superposition of Gaussian primitives.

    def low_discrepancy_sample(self, n, d=2, seed=None):
        """`n` low-discrepancy (quasi-random) points in [0, 1)^d -- even coverage of a domain. The right
        sampler wherever you PLACE points to COVER (generation seeds, codebook / anchor placement, sub-pixel
        jitter) rather than to draw an INDEPENDENT sample. Roberts' generalised golden-ratio sequence
        (holographic_lowdiscrepancy): deterministic, progressive (any prefix is well-distributed), and
        measurably tighter coverage than default_rng -- a quasi-Monte-Carlo integrator with far lower error
        than plain MC at equal count. Use default_rng where genuine independence is wanted (these points are
        correlated by construction). `seed` defaults to the mind's seed."""
        from holographic_samplinghome import Sampling                 # via the Sampling home (consolidation R4)
        return Sampling.low_discrepancy(n, d=d, seed=self.seed if seed is None else seed)

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
                           readout="softmax", early_stop=False, min_steps=None, stats=None):
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
        recommended setting when sampling for variety.

        ADAPTIVE STOP (B3, opt-in early_stop=True; pass stats={} to read stats['steps']): the decoded structure
        settles well before the fixed schedule ends, so stop once it has been STABLE for a few steps past a
        floor (Eno's condition: stability, not first-convergence -- novelty preserved). Measured ~50% fewer
        steps, the SAME structure as the full run on every seed, and a final crisp snap restores validity to
        1.000 -- essentially FREE (the hard decoded combination is an effective certificate, unlike the splat
        fit's soft plateau). Off by default (bit-identical to the fixed schedule)."""
        from holographic_hopfield import generate_structure as _gs
        return _gs(np.asarray(roles, float), np.asarray(fillers, float), steps=steps, beta0=beta0,
                   beta1=beta1, noise0=noise0, seed=self.seed if seed is None else seed, readout=readout,
                   early_stop=early_stop, min_steps=min_steps, stats=stats)

    def svg_canvas(self):
        """The holographic vector-graphics (SVG) faculty (holographic_svg.HolographicSVG) -- the sharp,
        resolution-INDEPENDENT cousin of splat_archive. Encode a scene of typed primitives (rect/circle/triangle,
        each with a continuous position, size, and palette colour) into ONE hypervector, decode it back, MORPH two
        scenes by interpolating their vectors (vector arithmetic that tracks a parameter lerp), GENERATE novel
        scenes via the composed-manifold diffusion (generate_structure), and render any scene as crisp SVG. A
        vector <rect>/<circle> has analytically exact edges at any zoom, so this sidesteps the Gaussian-basis blur
        the splat work had to fight with smaller splats or supersampling; SVG emission is pure string formatting,
        no new dependency. Cached on the mind, built at this mind's dim/seed -- round-trip fidelity scales with
        dimension (a few primitives are faithful at 2048+; a crowded scene wants more, the bundle's honest
        capacity limit). MEASURED: round-trip type/colour exact and position within ~0.03 on [0,1]."""
        if getattr(self, "_svg_canvas", None) is None:
            from holographic_svg import HolographicSVG
            self._svg_canvas = HolographicSVG(dim=self.dim, seed=self.seed)
        return self._svg_canvas

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

    def splat_field(self, target, k=20, denoise=False, refit=True, noise_thresh=None, k_min=4, k_max=200):
        """Represent a 2-D field/image as a SUPERPOSITION of K Gaussian primitives (holographic_splat) --
        the structural twin of bundle (a Gaussian-splat scene IS a bundle, and the RBF ScalarEncoder is
        already a Gaussian splat in hypervector space). Fits the splats by matching pursuit (greedy
        superposition); returns (splats, rendered) where `splats` is a compact (cy, cx, amplitude, sigma)
        code and `rendered` is their sum. With denoise=True returns just the rendered field, which is a
        DENOISER -- a few smooth Gaussians have no capacity for high-frequency noise.

        refit=True (default) re-solves the amplitudes JOINTLY after placement (`splat_refit`) -- greedy
        matching pursuit double-counts overlapping splats, and one least-squares solve removes that for
        ~2-4 dB (the gain grows with k). It is closed-form and gradient-FREE.

        noise_thresh (default None) switches the COUNT from fixed to ADAPTIVE (`adaptive_fit`, V-Ray's
        adaptive sampler): placement runs until the residual is below noise_thresh*range, bounded to [k_min,
        k_max], so a simple field uses few splats and a busy one uses more at MATCHED quality. Orthogonal to
        refit -- the count is WHERE the splats go, refit is HOW STRONG they are. None keeps the fixed-k path
        unchanged. (Meaningful only for fields the smooth Gaussian basis can represent: a hard edge runs to k_max.)

        KEPT NEGATIVE / SCOPE: isotropic splats and a fixed scale set (the honest matching-pursuit
        baseline); the *amplitude* refit is the gradient-free half of 'looping', but the gradient
        optimisation of positions/scales and anisotropic covariances (full 3DGS) needs autodiff and stays
        out of scope. Storing a whole gallery AS splat codes is now splat_archive() (holographic_splat_archive)."""
        from holographic_splat import splat_fit, splat_render
        target = np.asarray(target, float)
        if noise_thresh is not None:                          # ADAPT-1: let the COUNT adapt to content
            from holographic_splat import adaptive_fit
            splats, _ = adaptive_fit(target, noise_thresh=noise_thresh, k_min=k_min, k_max=k_max, refit=refit)
        else:
            splats = splat_fit(target, k, refit=refit)
        rendered = splat_render(splats, target.shape)
        return rendered if denoise else (splats, rendered)

    def export_splats(self, splats, path=None, fmt="ply", colors=None):
        """EXPORT Gaussian splats to a browser-renderer format (holographic_splatexport, FS-3) -- so a field/scene
        can be DISPLAYED as splats (the GPU's job; the engine stays the authoring brain). `splats` is a list of
        (center, amplitude, L), L the Cholesky of the inverse covariance (aniso_fit's native format; field_to_splats
        produces it from a metaball field). fmt='ply' writes the STANDARD 3D-Gaussian-Splatting .ply to `path` (opens
        in any 3DGS viewer) and returns the count; fmt='json' returns a compact JSON string for a three.js
        Gaussian-billboard shader. The core conversion is L -> scale + rotation quaternion by eigen-decomposing the
        precision (principal_axes). KEPT HONEST: base colour only -- holostuff splats carry no view-dependent
        spherical-harmonic colour (a further add, not faked); a degenerate/flat covariance is RAISED, not garbage.
        Delegates to holographic_splatexport."""
        from holographic_splatexport import splats_to_ply, splats_to_json
        if fmt == "json":
            return splats_to_json(splats, colors=colors)
        if path is None:
            raise ValueError("fmt='ply' needs a path to write to")
        return splats_to_ply(splats, path, colors=colors)

    def field_to_splats(self, centers, radius=0.5, amp=1.0):
        """Pull a metaball FIELD's Gaussians directly as exportable splats (holographic_splatexport, FS-3) -- no fit:
        the centres ARE the splat positions and the metaball `radius` IS the isotropic standard deviation. Returns a
        list of (center, amp, L) with L = (1/radius) I, ready for export_splats. For an already-fitted anisotropic
        field, pass aniso_fit's (center, amp, L) to export_splats directly."""
        from holographic_splatexport import field_to_splats
        return field_to_splats(centers, radius=radius, amp=amp)

    def export_splats_2d(self, splats2d, path=None, fmt="ply", colors=None, z=0.0, pixel_scale=1.0):
        """Export 2-D image splats (splat_fit's (cy, cx, amp, sigma)) to the standard 3DGS .ply by lifting them
        to the z=`z` plane (splats_2d_to_records: center=(cx,cy,z), isotropic L=1/sigma), so 2-D and 3-D splats
        export through one path. fmt='ply' writes to `path`; fmt='json' returns the three.js string. See
        holographic_splatexport.splats_2d_to_records."""
        from holographic_splatexport import splats_2d_to_records, splats_to_ply, splats_to_json
        recs = splats_2d_to_records(splats2d, z=z, pixel_scale=pixel_scale)
        if fmt == "json":
            return splats_to_json(recs, colors=colors)
        if path is None:
            raise ValueError("fmt='ply' needs a path to write to")
        return splats_to_ply(recs, path, colors=colors)

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

    def mixture_of_experts(self, dim=None, seed=0, number_range=(-4.0, 4.0)):
        """MIXTURE OF EXPERTS with a LEARNED GATE (holographic_moe, GatedMixture) -- a bank of specialists plus a
        trained holographic gate (itself a creature brain) that routes each input to ONE expert, learned from reward.
        This is the genuinely distinct routing the mind's own dispatch is NOT: `decide`/`classify`/`recognize` route
        by RULE (which verb you called, what type the input is), whereas the MoE gate is TRAINED -- so it routes by
        the input's CONTENT, which a type check cannot do (two experts owning different halves of the number line; the
        gate sends each value to the right one). Build it: `add_expert(name, examples)` / `add_linear_expert(...)` for
        specialists, `train_gate(examples, epochs)` to learn the routing from outcomes, then `predict(x, modality)`.
        Returns a GatedMixture on the mind's dim/seed. Measured: the learned gate beats any single expert by a wide
        margin and approaches the oracle; it also beats CONFIDENCE routing when a specialist can be confidently wrong
        (the outcome-trained gate is not fooled). Serves the Olshausen/Togelius seats -- learned, interpretable
        routing. Delegates to holographic_moe."""
        from holographic_moe import GatedMixture
        return GatedMixture(dim=dim or self.dim, seed=seed, number_range=number_range)

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

    def procedural_noise(self, n_dims=2, dim=1024, bounds=None, octaves=4, lacunarity=2.0,
                         gain=0.5, base_bandwidth=2.0, seed=None):
        """G1 -- holographic band-limited procedural noise as a FIELD; fBm as an octave BUNDLE.

        Returns a holographic_noise.FractalNoise: `.query(point)` evaluates the amplitude-weighted sum of
        per-octave band fields, each a single hypervector (an FPE bundle of random-weighted RBF kernels).
        Frequency is bandwidth (base_bandwidth * lacunarity^o), amplitude is gain^o. Seats: Stam (spectral
        noise), Berry (fBm/band-limit), Quilez (noise as a procedural-SDF primitive).
        KEPT NEGATIVE: band-limited/smooth by construction (no sharp/discontinuous noise), and FFT-bound --
        each kernel is one encode, so deep fully-filled fBm is expensive (kernels are capped per octave)."""
        from holographic_noise import FractalNoise
        return FractalNoise(n_dims, dim=dim, bounds=bounds, octaves=octaves, lacunarity=lacunarity,
                            gain=gain, base_bandwidth=base_bandwidth,
                            seed=self.seed if seed is None else seed)

    def material(self, channels=None, dim=1024, bandwidth=3.0, bounds=None):
        """G2 -- a PBR material as a role-filler HRR record; textures as FPE functions over UV.

        `channels` maps a name (albedo, roughness, normal, height, ...) to (uv_points, values); each becomes
        a texture field over the UV square, and the material binds them under per-name role atoms into one
        record = sum_r bind(role_r, channel_r). Returns a holographic_material.Material: sample() is exact
        (stored field), the record composes/blends/transmits as one vector, and transform_uv re-UVs every
        channel with a single bind. Seats: Plate (the record), Pharr (PBR channels), Drettakis (per-primitive
        material). KEPT NEGATIVE: band-limited (smooth textures; sharp masks stay raster) and the bare-record
        channel recovery carries ~sqrt(n)/sqrt(dim) crosstalk (raise dim to buy capacity)."""
        from holographic_fpe import VectorFunctionEncoder
        from holographic_material import Material, texture_field
        if bounds is None:
            bounds = [(0.0, 1.0), (0.0, 1.0)]
        enc = VectorFunctionEncoder(2, dim=dim, bounds=bounds, bandwidth=bandwidth, seed=self.seed)
        mat = Material(enc)
        if channels:
            for name, (pts, vals) in channels.items():
                mat.add(name, texture_field(enc, pts, vals))
        return mat

    def displace(self, target, scalar_fn, amount, use_uv=False):
        """G3 -- displace a surface along its normal by amount*scalar_fn. Dispatches on the target type.

        On a HolographicField the offset is a field DELTA (apply_delta of -amount*scalar at the field points),
        O(edit) with EXACT remove_delta undo; returns (displaced_field, delta). On a Mesh each vertex moves
        along its normal; returns the displaced Mesh. Seats: Pharr (displacement), Quilez (SDF displace = add
        to distance). KEPT NEGATIVE: the SDF path is the near-surface shader approximation (exact only where
        |grad sdf|=1); mesh displacement can self-intersect for large amounts on concavities."""
        from holographic_displace import displace_sdf, displace_mesh
        from holographic_fpefield import HolographicField
        if isinstance(target, HolographicField):
            return displace_sdf(target, scalar_fn, amount)
        return displace_mesh(target, scalar_fn, amount, use_uv=use_uv)

    def bump(self, mesh, scalar_fn, amount, eps=1e-3):
        """G3 -- bump mapping: perturb a mesh's shading normals from a scalar field's slope, NO vertices move.
        Returns an (V,3) array of tilted unit normals. The cheap fake-detail path (silhouette unchanged)."""
        from holographic_displace import bump_normals
        return bump_normals(mesh, scalar_fn, amount, eps=eps)

    def terrain(self, bounds=None, octaves=5, lacunarity=2.0, gain=0.5, base_bandwidth=2.0,
                dim=1024, seed=None):
        """G4 -- a holographic fBm heightfield, liftable to a displaced-grid mesh or a heightfield SDF.

        Returns a holographic_terrain.Terrain (`.height(xy)`, `.heightmap(res)`); use holographic_terrain's
        terrain_to_mesh / terrain_to_sdf to lift it. A composition of G1 (height) and the displacement idea,
        so LOD is just re-sampling. Seats: Stam, Berry. KEPT NEGATIVE: no erosion (pure fBm statistics), and
        the heightfield SDF (z - height) is sign-correct but not a true Euclidean distance where it is steep."""
        from holographic_terrain import Terrain
        return Terrain(bounds=bounds, octaves=octaves, lacunarity=lacunarity, gain=gain,
                       base_bandwidth=base_bandwidth, dim=dim, seed=self.seed if seed is None else seed)

    def lsystem(self, axiom, productions, stochastic=None):
        """G5 -- a context-free L-system grammar; productions are a holographic record, output is a scenegraph.

        Returns a holographic_grammar.LSystem (`.expand(n)` parallel-rewrites the string); interpret it with
        turtle_to_segments and assemble with segments_to_scene / grow_plant (each segment instanced through a
        transform -- a recursive bundle that scene_to_recipe turns back into a holographic recipe). The one
        genuinely new geometry capability. Seat: Plate (HRR productions). KEPT NEGATIVE: recursive composition,
        not a biological growth simulation (no tropism/competition); deterministic context-free (optionally
        seeded-stochastic)."""
        from holographic_grammar import LSystem
        return LSystem(axiom, productions, stochastic=stochastic, rng_seed=self.seed)

    def attribute_field(self, encoder, points, values, weights=None):
        """G6 -- a per-vertex/texel attribute as a RESOLUTION-INDEPENDENT field over the surface domain.

        Same construction as a texture (an FPE function), but the intent is a data channel: bake it to a
        coarse mesh, subdivide, re-bake, and shared points keep their values because the field never changed
        -- only the sample points densified. Returns the field vector; sample with holographic_attributes.
        sample_attribute. A light raster store (attach_attribute/get_attribute) coexists for hard masks.
        KEPT NEGATIVE: band-limited (smooth attributes interpolate; hard 0/1 masks come back smoothed)."""
        from holographic_attributes import attribute_field
        return attribute_field(encoder, points, values, weights=weights)

    def sdf_object(self, seed=None, complexity=3):
        """S2 -- a procedurally generated 3D OBJECT as an SDF tree, from a seed (the demoscene seed->world).

        Returns a holographic_sdf.SDF: a few transformed primitives combined by CSG / smooth-union with an
        occasional round/twist. Render with sdf_render, emit a shader with sdf_shader, represent it with
        .to_tree() (a holographic recipe). Seat: Quilez (tiny seed, structured world). KEPT NEGATIVE: a
        generator, not an art director -- a random tree can subtract most of itself away or leave a
        disconnected surface; marching rounds sub-cell features."""
        from holographic_procgen import procedural_object
        return procedural_object(self.seed if seed is None else seed, complexity=complexity)

    def sdf_render(self, sdf_node, bounds=((-2, -2, -2), (2, 2, 2)), res=40):
        """S1 -- march an SDF tree to a triangle Mesh through the engine's existing marching bridge."""
        from holographic_procgen import object_to_mesh
        return object_to_mesh(sdf_node, bounds=bounds, res=res)

    def sdf_shader(self, sdf_node, name="map"):
        """S1 -- emit a complete Shadertoy-ready GLSL fragment shader (map() + raymarch + normals + light)
        for an SDF tree -- the demoscene OUTPUT. The shader embeds its own DSL in a header comment, so it
        round-trips back to a tree. Seat: Quilez (raymarched SDFs). KEPT NEGATIVE: twist/displace are
        domain warps (not exact distances) -- the emitter flags them and the raymarcher must shorten steps."""
        return sdf_node.to_glsl(name=name)

    def sdf_parse(self, dsl_text):
        """S1 -- parse a compact SDF DSL string back into an SDF tree -- the INPUT side of shader I/O.
        (kind p0 p1 ... child0 child1 ...); the inverse of node.to_dsl()."""
        from holographic_sdf import parse_dsl
        return parse_dsl(dsl_text)

    def menger_fractal(self, iterations=3, size=1.0):
        """S1 -- the canonical Menger-sponge FRACTAL model as an SDF (a box minus recursive crosses). Evals,
        marches to a mesh, AND emits a GLSL loop -- the demoscene fractal. Seat: Quilez."""
        from holographic_sdf import menger
        return menger(iterations, size)

    def greeble(self, base_mesh, seed=None, density=0.7, max_height=0.15, footprint=0.5):
        """S2 -- cover a base mesh's faces with extruded greeble boxes (the G5 panel idea on any surface) ->
        a merged Mesh of mechanical hull detail. Seat: Quilez/demoscene. KEPT NEGATIVE: instancing, not CSG
        -- greebles can intersect the hull (which is how greebling actually looks)."""
        from holographic_procgen import greeble_mesh
        return greeble_mesh(base_mesh, seed=self.seed if seed is None else seed,
                            density=density, max_height=max_height, footprint=footprint)

    def vegetated_terrain(self, seed=None, n_plants=10, plant_iterations=3, terrain_kwargs=None):
        """S2 -- a fBm terrain (G4) with L-system plants (G5) scattered on its surface -> one scenegraph.
        Returns (scene_node, terrain). Composes terrain + grammar + scatter; flatten_scene gives the mesh.
        KEPT NEGATIVE: a deterministic scatter at the terrain height, not an ecology (no collision/clustering)."""
        from holographic_procgen import vegetated_terrain
        return vegetated_terrain(self.seed if seed is None else seed, n_plants=n_plants,
                                 plant_iterations=plant_iterations, terrain_kwargs=terrain_kwargs)

    def procedural_compression(self, sdf_node, bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=48):
        """S3/C1 -- measure procedural representation AS compression: the tiny generator (DSL) vs the
        expanded geometry it marches to. Returns {dsl, dsl_bytes, mesh_faces, mesh_bytes, ratio}. The
        finding it makes concrete: a generator's size is CONSTANT in its output's complexity (a Menger
        DSL is 12 bytes at any depth), so storing the LAW escapes the capacity/complexity wall -- the same
        MDL principle as symbolic_regress/compress_signal, for geometry. KEPT NEGATIVE: only compressible
        content has a short generator (an arbitrary mesh does not); procedural compression is lossy and
        content-restricted, not a universal codec."""
        from holographic_procbridge import procedural_compression
        return procedural_compression(sdf_node, bounds=bounds, res=res)

    def soft_min(self, a, b, k):
        """S3/C4 -- the log-sum-exp soft minimum, -k*log(exp(-a/k)+exp(-b/k)); k->0 gives min(a,b). This is
        the SAME log-sum-exp the modern-Hopfield/softmax cleanup uses (softmax is a soft-arg-MAX; this is a
        soft-arg-MIN over distances): a smooth CSG union of geometry and a soft recall of a memory are one
        temperature-controlled operator, with k playing the role of 1/beta. The SDF's smooth_union and the
        memory cleanup are the same math seen in two domains."""
        from holographic_procbridge import soft_min
        return soft_min(a, b, k)

    def evolving_atom(self, n_harmonics=3, dim=None, forgetting=1.0, delta=1e-6):
        """SUBSTRATE EVOLUTION -- a context-conditioned atom that updates its OWN harmonic coefficients as
        (context angle, meaning) pairs stream in, via online Recursive Least Squares (the batch
        harmonic_atom fit made autonomous). Returns a holographic_harmonic.OnlineHarmonicAtom: `.observe(
        theta, meaning)` folds in one observation by a rank-1 Sherman-Morrison step, `.decode(theta)` reads
        the current meaning. forgetting=1.0 converges to the batch least-squares fit; forgetting<1.0 turns
        the codebook into a dynamical system that TRACKS a drifting meaning function. No RNG, no autodiff --
        the engine's own least squares, online. KEPT NEGATIVE: forgetting<1 trades steady-state accuracy on
        a stationary function for tracking a non-stationary one."""
        from holographic_harmonic import OnlineHarmonicAtom
        return OnlineHarmonicAtom(n_harmonics, self.dim if dim is None else dim,
                                  forgetting=forgetting, delta=delta)

    def optimize_toolchain(self, tool_vecs, goal_sig, length, steps=200, lr=0.5):
        """DIFFERENTIABLE ORCHESTRATION -- optimize a whole tool-chain JOINTLY against a chain-level
        structural score, instead of scoring tools independently. `tool_vecs` (N, D) are the registry's
        tool vectors (tools already live in hyperspace); `goal_sig` is the desired composed chain signature
        (the order-encoded superposition a working chain would produce). Optimizes a soft selection over
        tools per step by gradient ASCENT on cosine(chain_signature, goal_sig) -- the gradient derived
        analytically through cosine/superposition/permute/softmax in numpy, NO autodiff (the same
        gradient-without-a-framework method as holographic_optimize). Returns (indices, score). KEPT
        NEGATIVE: a local optimum of a non-convex landscape; on orthogonal/easy tool sets per-position
        greedy already wins -- the gain is on CORRELATED tool sets where independent scoring is misled by
        cross-talk between positions."""
        from holographic_orchestrator import optimize_toolchain
        return optimize_toolchain(tool_vecs, goal_sig, length, steps=steps, lr=lr)

    def synthesize_program(self, library, goal_sig, max_length=4, threshold=0.85, steps=200, lr=0.5):
        """Fill a VOID CAPABILITY GAP (SYNTH-1): when no registered tool chain reaches a goal, SYNTHESISE one in
        the latent space instead of failing. Optimises a chain over `library` toward `goal_sig` (growing the
        length if a short program won't reach -- the structural 're-bundle'), VERIFIES the discrete chain's
        coherence (never trusts the soft optimum), and GATES: returns status 'synthesized' with the chain if
        coherence >= `threshold`, else 'abstain' (the gap genuinely could not be filled -- it declines rather than
        running an incoherent program). Measured: 20/20 reachable goals synthesized (coh ~1.0), 20/20 unreachable
        abstained (best coh ~0.19) -- the gate cleanly separates fillable from void gaps. The latent ascent is a
        hand-derived analytic gradient (numpy, NO autodiff); abstention is the load-bearing safety property. See
        holographic_voidsynth."""
        from holographic_voidsynth import synthesize_for_goal
        return synthesize_for_goal(library, goal_sig, max_length=max_length, threshold=threshold, steps=steps, lr=lr)

    def blend_programs(self, sig_a, sig_b, weights=(1.0, 1.0)):
        """BLEND two program signatures into one by bundling -- composition in the shared substrate. The blend
        stays coherent to BOTH source goals (measured ~0.72/0.74 for a graphics + an audio program), so one
        vector can carry two intents: a program + a program, or a program + data. Because every domain's tools
        live in the SAME space, this cross-domain blend is what 'synesthesia across domains' actually is -- the
        project's one-algebra thesis, not a new sense. See holographic_voidsynth.blend_programs."""
        from holographic_voidsynth import blend_programs
        return blend_programs(sig_a, sig_b, weights=weights)

    def fill_capability_gap(self, library, goal_sig, registry_hit=None, threshold=0.85, max_length=4, steps=200):
        """The orchestration: if a registered tool/chain already reaches the goal (`registry_hit` >= threshold),
        use it (status 'registry', no gap); otherwise synthesise and gate/abstain. The bridge from the
        orchestrator's plan()='gap' to verified latent synthesis. See holographic_voidsynth.fill_capability_gap."""
        from holographic_voidsynth import fill_capability_gap
        return fill_capability_gap(library, goal_sig, registry_hit=registry_hit, threshold=threshold,
                                   max_length=max_length, steps=steps)

    def agent(self, actions, dim=512, seed=0, value_floor=0.25, pain_reflex=0.6, synth_threshold=0.8):
        """An upgraded creature agent (AGENT-1): an action LIBRARY as VSA atoms, reward AND pain affect, a
        pain-avoidance REFLEX (one painful experience blocks an action, faster than value learning), and VOID-GAP
        ACTION SYNTHESIS -- when no learned single action fits the situation, it synthesises a multi-step action
        program toward a goal and gates it (commit if coherent, else abstain to a safe default), reusing the
        SYNTH-1 loop. Self-explaining (decide returns WHY) and deterministic. Because actions are atoms, a plan has
        a composed signature embeddable in / blendable with other VSA programs -- the agent can DRIVE a program.
        Build agents from this for roles beyond a maze NPC; the bespoke per-action value engine still lives in
        HolographicMind. See holographic_agent.Agent."""
        from holographic_agent import Agent
        return Agent(actions, dim=dim, seed=seed, value_floor=value_floor, pain_reflex=pain_reflex,
                     synth_threshold=synth_threshold)

    def drive_system(self, weights=None):
        """A set of homeostatic DRIVES (DRIVE-1): internal needs (clarity, understanding, coverage, energy) that
        decide which faculty an agent should apply next -- the most under-satisfied applicable need wins. The
        mechanism that lets the agent DRIVE denoising / pattern recognition / descent decisions through a deeply
        nested process where the schedule is too large to hand-script. See holographic_drives.DriveSystem."""
        from holographic_drives import DriveSystem
        return DriveSystem(weights=weights)

    def drive_process(self, root, codebook, drives=None, energy=24, recog_thresh=0.5, policy="drive", seed=0):
        """Walk a NESTED/fractal process under homeostatic drives, choosing at each node whether to DENOISE,
        RECOGNISE, or DESCEND by which need is most starved (policy='drive'); 'denoise'/'recognize'/'descend' are
        fixed-priority baselines and 'random' is the naive control. Faculties are real (codebook cleanup + cosine
        recognition; recognition only succeeds on a cleaned signal). MEASURED: the drive schedule matches the best
        fixed priority WITHOUT being told it (~0.46 vs ~0.45 worst-served need) and beats naive scheduling 2-4x --
        an adaptive default for processes too nested to schedule by hand. See holographic_drives.drive_process and
        make_nested_process."""
        from holographic_drives import drive_process
        return drive_process(root, codebook, drives=drives, energy=energy, recog_thresh=recog_thresh,
                             policy=policy, seed=seed)

    def abstract_program(self, examples, name=None, max_depth=2, threshold=0.9):
        """Abstract a reusable PROGRAM from a TRACE -- a set of (input_vec, output_vec) examples demonstrating one
        transform (a recorded behaviour, a demonstration, the in/out of a creature's moves). Synthesise a
        procedure that reproduces the FIRST example, then VERIFY it reproduces ALL the others: the abstraction is
        the program CONSISTENT ACROSS examples, not a fit to one. If it generalises, optionally store it by `name`
        (callable later). Returns dict{program, generalizes, fit (mean cosine over examples), worst}. WHY this
        beats raw prototypes (the 'transfers better' claim): a stored prototype only matches near-identical
        states; an abstracted program captures the TRANSFORM itself, so it transfers to inputs never seen --
        measured in the tests against a prototype-nearest-neighbour baseline. HONEST: it only abstracts transforms
        expressible in the VM's ops within max_depth (BIND/BUNDLE/PERMUTE), and returns generalizes=False (rather
        than a wrong program) when the examples don't share one such transform."""
        import numpy as _np
        from holographic_ai import cosine as _cos
        examples = list(examples)
        prog = self.synthesize_procedure(examples[0][0], examples[0][1], max_depth=max_depth, threshold=threshold)
        if prog is None:
            return {"program": None, "generalizes": False, "fit": 0.0, "worst": 0.0}
        fits = []
        for inp, outp in examples:                           # VERIFY the synthesised program on EVERY example
            out, _ = self.run_procedure(prog, init_acc=inp)
            fits.append(float(_cos(out, outp)))
        worst = float(min(fits))
        generalizes = worst >= threshold                     # consistent across all examples -> a real abstraction
        if generalizes and name is not None:
            self.learn_procedure(name, prog)
        return {"program": prog, "generalizes": generalizes, "fit": float(_np.mean(fits)), "worst": worst}

    def path_trace(self, sdf, camera, width=96, height=96, spp=16, max_bounce=4, material=None, sky=None, seed=0):
        """Monte-Carlo PATH TRACER for true multi-bounce global illumination -- the core of V-Ray/Redshift/Arnold.
        Solves the full rendering equation over an SDF scene by following random light paths and averaging:
        BRDF importance sampling (cosine + GGX), Russian roulette, vectorised over rays. Indirect light (color
        bleeding, soft GI in concavities) falls out for free, unlike the engine's single-bounce irradiance cache.
        Returns an (H,W,3) HDR image. MEASURED: unbiased (white-furnace -> albedo), noise ~1/sqrt(spp), color
        bleed reproduced; 128^2/96spp ~13-16s (OFFLINE NumPy brain, NOT GPU-realtime). KEPT NEGATIVE: no
        next-event estimation, so light is gathered only when a bounce hits the emissive environment -- great for
        a big sky, very noisy for small emitters (NEE/MIS is the next step). See holographic_pathtrace.path_trace."""
        from holographic_pathtrace import path_trace
        return path_trace(sdf, camera, width=width, height=height, spp=spp, max_bounce=max_bounce,
                          material=material, sky=sky, seed=seed)

    def render_auto(self, sdf, camera, width=96, height=96, material=None, sky=None, quality="high",
                    max_bounce=4, seed=0, return_stats=False, **kw):
        """AUTO-CALIBRATING render -- one quality knob, no per-scene spp or denoise tuning. It wires together
        machinery the engine already had but never connected into a render loop: it samples in PASSES and, after
        each, asks the calibrated stop rule (holographic_adaptive_sample.converged_mask) which pixels have reached
        the target confidence interval -- those STOP, the rest keep sampling (path_trace's `active` mask) -- so
        hard pixels (glass, silhouettes, grazing reflections) automatically get more samples than flat regions.
        It then denoises with a VARIANCE-GUIDED SVGF whose per-pixel strength is set by the variance the sampler
        measured, so residual grain is smoothed and converged detail is preserved. `quality` is a target CI
        half-width (name 'draft'/'medium'/'high'/'ultra' or a float). MEASURED: at equal average sample budget it
        beats a raw path trace at draft/medium and ties near convergence at high (denoise stops mattering once a
        pixel is already converged -- the documented crossover). See holographic_gbuffer.render_auto."""
        from holographic_gbuffer import render_auto
        return render_auto(sdf, camera, width, height, material, sky=sky, quality=quality,
                           max_bounce=max_bounce, seed=seed, return_stats=return_stats, **kw)

    def render_scene_document(self, scene, camera, width=96, height=72, quality="medium", max_bounce=4,
                              seed=0, sky=None, default_material="matte_gray", return_stats=False, sss_dir=None,
                              sss_depth=0.6, sss_sigma=4.0, lights=None, dome_cache=False, demodulate=False, soft_light_cache=False,
                              indirect_cache=False):
        """Render the canonical SCENE DOCUMENT (holographic_scene_doc.Scene) -- the 'a modeling app builds a
        document, then renders it' path. The document is a table of objects (each a stable handle + transform +
        SDF geometry + library material); this flattens it to ONE scene SDF (nearest-object distance) plus a
        material_fn that shades each hit with its owning object's material, then renders with render_auto. So the
        renderer consumes the authoritative scene instead of a hand-built Python class per scene (backlog H7).
        `sss_dir` (a light direction) turns on the SUBSURFACE glow for translucent materials (wax/jade/skin).
        `dome_cache` (default off) serves any DomeLight via the cheap cached-dome pass (holographic_domecache)
        instead of ray-traced ambient occlusion. `demodulate` (default off) denoises by dividing the albedo out
        (holographic_modulate, M4) -- cleaner on textured diffuse surfaces. See
        holographic_scene_render.render_scene_document."""
        from holographic_scene_render import render_scene_document
        return render_scene_document(scene, camera, width=width, height=height, quality=quality,
                                     max_bounce=max_bounce, seed=seed, sky=sky,
                                     default_material=default_material, return_stats=return_stats, sss_dir=sss_dir,
                                     sss_depth=sss_depth, sss_sigma=sss_sigma, lights=lights, dome_cache=dome_cache,
                                     demodulate=demodulate, soft_light_cache=soft_light_cache,
                                     indirect_cache=indirect_cache)

    def bake(self, evaluator, vary="position", **kw):
        """CONSOLIDATION CACHE (H2) -- bake a slow `evaluator` over the thing that VARIES, then look it up cheaply.
        vary='position' (kw: lo, hi, res) returns a BakedGrid with .sample(points); 'constant' computes once; 'view'
        and 'time' delegate to the BRDF LUT / frame bakes. See holographic_cachehome.Cache."""
        from holographic_cachehome import Cache
        return Cache.bake(evaluator, vary=vary, **kw)

    def build_index(self, vectors, labels=None, method="auto", seed=0):
        """CONSOLIDATION INDEX (H1) -- build a nearest-neighbour index over `vectors` with one interface: an exact
        cosine scan for small sets, the sub-linear RP-forest for large ones (chosen by `method='auto'`). The result
        has `.nearest(query, k, abstain=alpha)` -> [(label_or_index, score), ...], with an optional calibrated
        abstain. See holographic_index.Index."""
        from holographic_index import Index
        return Index(vectors, labels=labels, method=method, seed=seed)

    def find_capability(self, problem, k=3):
        """CONSOLIDATION CATALOG (C1) -- 'search before you build'. Describe a problem in plain English and get the
        engine homes that already solve it, best first, so you don't build a duplicate. The catalog is seeded with
        the consolidation homes (Index/Cache/Field/...) AND this mind's own public faculties (their docstrings), so
        both curated homes and live methods are findable. Returns a list of holographic_catalog.Capability (each has
        .name, .does, .example, .native). See holographic_catalog."""
        return self._capability_catalog().find_capability(problem, k=k)

    def suggest(self, task, k=5):
        """AGENT-FRIENDLY autocomplete: turn a plain-English task into the best capabilities to use, each with a
        CONFIDENCE (0..1) and the concrete call to make. Like find_capability, but scored + call-ready so an agent (or
        a person) can decide what to invoke. See holographic_skills.suggest."""
        import holographic_skills as _sk
        return _sk.suggest(task, k=k)

    def route(self, task):
        """AGENT-FRIENDLY decision node: when one skill clearly wins, returns {'decision':'act', 'skill':..., 'call'}
        so the agent just does it; when it's ambiguous, returns {'decision':'choose', 'options':[...]} so it asks
        instead of guessing. 'Act when confident, ask when not', score-based. See holographic_skills.route."""
        import holographic_skills as _sk
        return _sk.route(task)

    def describe_skill(self, name):
        """A machine-readable SKILL CARD for a capability or a UnifiedMind method by name: what it does + how to CALL
        it (real signature for methods). The 'skill description' an agent reads before invoking. See holographic_skills."""
        import holographic_skills as _sk
        return _sk.skill_card(name)

    def complete_method(self, prefix, k=15):
        """Method-name AUTOCOMPLETE: this mind's methods starting with `prefix`, each with its signature -- what an
        agent (or an IDE) offers while constructing a `mind.<prefix...` call. See holographic_skills.complete."""
        import holographic_skills as _sk
        return _sk.complete(prefix, k=k)

    def skills(self, include_methods=True):
        """The full machine-readable SKILL MANIFEST: every curated capability home plus every public method (with its
        signature). What an agent loads ONCE to know the whole surface it can drive. See holographic_skills.manifest."""
        import holographic_skills as _sk
        return _sk.manifest(include_methods=include_methods)

    def _capability_catalog(self):
        """Lazily build (once) and cache the capability catalog: the curated homes + this mind's faculties + EVERY
        engine module (by docstring), so a problem description can surface anything built -- nothing stays buried."""
        cat = getattr(self, "_catalog_cache", None)
        if cat is None:
            from holographic_catalog import default_catalog, seed_from_mind, seed_from_modules
            cat = seed_from_modules(seed_from_mind(default_catalog(), self))
            self._catalog_cache = cat
        return cat

    def register_capability(self, name, does, example="", native=True, aliases=()):
        """Register a capability in the catalog so future `find_capability` calls surface it (backlog C1: as each
        consolidation home lands, register it here). Additive; returns the entry."""
        return self._capability_catalog().register_capability(name, does, example=example, native=native,
                                                              aliases=aliases)

    def scene_to_render(self, scene, default_material="matte_gray"):
        """Flatten a Scene document to (sdf, material_fn) for the path tracer, without rendering -- the bridge
        render_scene_document uses. Useful when you want the scene's SDF/material to hand to a custom render
        (dispersion, caustics, a preview session). See holographic_scene_render.scene_to_render."""
        from holographic_scene_render import scene_to_render
        return scene_to_render(scene, default_material=default_material)

    def render_demodulated_upscale(self, sdf, camera, low_wh, high_wh, material_fn, sky=None, quality="medium",
                                   max_bounce=3, seed=0, lights=None):
        """M5 -- render a HIGH-resolution frame at LOW-resolution lighting cost. Render the expensive lighting at
        `low_wh`=(w,h), read the cheap high-res G-buffer at `high_wh`, and combine by demodulation: upscale the
        smooth irradiance, multiply the crisp high-res albedo back (holographic_modulate). ~2.6x faster than a full
        high-res render and cleaner than a plain upscale ON TEXTURED surfaces (the detail lives in the albedo).
        Kept negative: neutral on uniform-albedo (no texture to restore); diffuse only."""
        from holographic_modulate import render_demodulated_upscale
        return render_demodulated_upscale(sdf, camera, low_wh, high_wh, material_fn, sky=sky, quality=quality,
                                          max_bounce=max_bounce, seed=seed, lights=lights)

    def fluid_solver(self, shape, **kwargs):
        """A grid-based STABLE-FLUIDS solver (Stam 1999) for smoke, buoyant plumes, and combustion/FIRE -- the
        method professional smoke engines (Houdini, Bifrost Aero) are built on. Incompressibility is enforced by
        an FFT pressure projection (a Helmholtz-Hodge decomposition = the engine's periodic circular-convolution
        algebra), and advection is unconditionally-stable semi-Lagrangian. Returns a holographic_fluid.StableFluid
        carrying velocity + smoke density + temperature + fuel; call .step() to advance, .add_source() to emit.
        MEASURED: divergence-free to machine precision; 128^2 ~10ms/step, 64^3 ~0.5s/step (offline NumPy BRAIN,
        NOT GPU-realtime -- the method matches the pros, the throughput does not); vorticity confinement keeps
        ~88x more swirl. KEPT NEGATIVE: semi-Lagrangian advection is dissipative (~20% smoke mass lost over 60
        steps to interpolation; a MacCormack/BFECC or FLIP scheme conserves better); boundaries are periodic."""
        from holographic_fluid import StableFluid
        return StableFluid(shape, **kwargs)

    def delta_chain(self, base, tol=0.0, codebook=None):
        """A chunked DELTA CHAIN (DELTA-1): store a SEQUENCE of (N, D) chunks as a base + per-chunk deltas, each
        delta taken against the BASE or the PRIOR chunk -- whichever is smaller -- so memory is O(actual change).
        A SHA-256 hash chain + Merkle root make integrity PROVABLE: append() folds each chunk into the chain,
        get(i) reconstructs AND verifies (a corrupted delta / wrong base / broken propagation raises
        IntegrityError), root() is the one 'fractal' proof of the whole sequence. Bit-exact, deterministic,
        vectorized (no per-element Python on the data). With `codebook` set, changed rows that EXACTLY equal an
        atom store an index, not a float row -- lossless, and its size win scales with D and sequence length
        (measured 86x on the delta portion at D=256). HONEST: exact integrity is hashlib, not a (lossy) VSA
        bundle -- the case where VSA-native is NOT beneficial. See holographic_deltachain.DeltaChain."""
        from holographic_deltachain import DeltaChain
        return DeltaChain(base, tol=tol, codebook=codebook)

    def execution_replay(self, program, chunk=14, init_acc=None):
        """Run a long program CHUNKED and record a verifiable, O(change) REPLAY LOG (DELTA-1 x WIRE-2): each
        chunk's full machine state (acc + registers + stack, as rows) is appended to a DeltaChain, so the
        execution can be audited, RESUMED from any seam, and integrity-checked. This is where the run_chunked
        state-threading and the delta chain meet: consecutive seam states share most rows (registers that didn't
        change), so the log is O(actual change), and the hash chain proves it wasn't tampered with. Returns
        (acc, trace, replay) -- replay a DeltaChain whose chunk i is the state after seam i+1 (its base is the
        first seam's state), or None if the program produced no seams."""
        M = self._machine()
        acc, trace, states = M.run_chunked(program, chunk=chunk, init_acc=init_acc,
                                           handlers=self._procedure_handlers(), record=True)
        if not states:
            return acc, trace, None
        from holographic_deltachain import DeltaChain
        replay = DeltaChain(states[0])
        for s in states[1:]:
            replay.append(s)
        return acc, trace, replay

    def nystrom_field(self, points, sources, weights, sigma, m=None):
        """Approximate a kernel-weighted field f(p) = sum_j weights[j]*K(p, sources[j]) (Gaussian RBF) via m
        LANDMARK sources -- O((Np+Ns)*m) instead of the exact O(Np*Ns), for LARGE memory or physics sims
        (sources=particles+charges, or stored items+payload; points=where you sample). MEASURED ~13x at N=2000
        on a smooth field, the win growing with N. KEPT NEGATIVE: exact only for a LOW-RANK (smooth) field; a
        high-frequency field is full-rank and the landmark approximation degrades (corr ~0.2). See
        holographic_nystrom.nystrom_kernel_apply."""
        from holographic_nystrom import nystrom_kernel_apply
        return nystrom_kernel_apply(points, sources, weights, sigma, m=m)

    def consolidate_subspace(self, memories, k=8, landmarks=None):
        """The consolidated low-rank SUBSPACE of stored memories (top-k principal directions) + mean. With
        `landmarks`=m, approximate it from m farthest-point memories instead of all N (the Nystrom sketch for a
        LARGE store). Returns (basis, mean). See holographic_dream.dream_subspace."""
        from holographic_dream import dream_subspace
        return dream_subspace(memories, k=k, landmarks=landmarks)

    def dream(self, basis, mean, n=8, seed=0, noise=1.0, codebook=None):
        """DREAM = generative replay over the consolidated subspace: draw noise, project onto the subspace (the
        manifold denoiser run from noise), optionally clean -> samples ON the manifold (valid) yet NOVEL (not a
        stored item). Over the consolidated (composed) subspace this yields novel COMPOSITIONS, the interesting
        regime B10 flagged. Returns an (n, D) array. See holographic_dream.dream."""
        from holographic_dream import dream
        return dream(basis, mean, n=n, seed=seed, noise=noise, codebook=codebook)

    def holographic_value_head(self, n_actions, dim=None, routed=False, n_buckets=64):
        """The creature's value/policy AS a pure-VSA program. Returns a HolographicValueHead (or, with
        routed=True, a RoutedValueHead whose routing fabric pushes the capacity cliff back ~n_buckets-fold):
        a drop-in for the creature's value backend (same value(state, action)->(value, support) and
        absorb(state, action, ret) API), but the whole per-action policy is bundles -- Q_a (return-weighted
        state superposition) and N_a (the normaliser) -- so value(s,a) = <s,Q_a>/<s,N_a> reproduces the
        brain's Nadaraya-Watson average while learning is one bundling step and the policy is a fixed-size,
        savable, COMPOSABLE hypervector ({Q, N}, or policy_atom() folded into two bindable vectors) instead of
        a growing (vector, scalar) table. KEPT NEGATIVE: a single bundle pair has finite capacity (matches the
        tabular brain at low load, degrades past ~dim distinct situations); routing trades n_buckets-fold
        memory to push that cliff back."""
        d = self.dim if dim is None else dim
        if routed:
            from holographic_valuehead import RoutedValueHead
            return RoutedValueHead(d, n_actions, n_buckets=n_buckets)
        from holographic_valuehead import HolographicValueHead
        return HolographicValueHead(d, n_actions)

    def fast_creature_encoder(self, dim=None, seed=1):
        """Compiled, fully in-VSA perception for the creature loop: a FastCreatureEncoder whose per-step
        role/filler binds (FFT convolutions) are precomputed once into a codebook, so perceiving recurring
        senses is a gather+sum -- no per-step FFT. Bit-identical to the plain CreatureEncoder, ~8x faster at
        steady state. With the holographic value head (decide=dot, learn=bundle) the whole perceive->decide->
        learn loop becomes array ops, keeping the creature inside the holographic space (no Python<->VSA
        round-trip per step). `perception_codebook()` exposes the compiled (features, dim) matrix."""
        from holographic_creature import FastCreatureEncoder
        return FastCreatureEncoder(self.dim if dim is None else dim, seed=seed)

    def soft_body(self, positions, inv_mass=None, velocities=None):
        """A PBD/XPBD softbody: particles + distance constraints, time-stepped under gravity/forces. Inverse
        mass 0 pins a particle. The constraint sweep is the same iterate-a-projection engine as the resonator/
        denoiser/IK; XPBD adds time-step-independent stiffness. See holographic_softbody.SoftBody."""
        from holographic_softbody import SoftBody
        return SoftBody(positions, inv_mass=inv_mass, velocities=velocities)

    def mesh_to_softbody(self, mesh, compliance=0.0, pin=None):
        """Turn ANY mesh into a simulatable SoftBody (vertices -> particles, edges -> distance constraints) so a
        PROJECTED surface mesh can be driven by the whole physics layer: gravity, fluid drag (drag_force_3d as
        external_force), self-collision (add_self_collision), the constraint solver. The bridge that lets the
        mesh pipeline take advantage of the physics -- sculpt -> surface_mesh -> mesh_to_softbody -> simulate.
        A surface mesh is a shell (behaves like cloth); `pin` anchors vertices. See SoftBody.from_mesh."""
        from holographic_softbody import SoftBody
        return SoftBody.from_mesh(mesh, compliance=compliance, pin=pin)

    def dynamics_to_mesh(self, source, bounds=None, radius=0.5, level=0.5, resolution=48,
                         axes=None, face_type="triangle"):
        """Export ANY dynamics state as a watertight mesh -- the unified surfacing entry point. `source` is one of:
          * a SoftBody or RigidBody built via from_mesh -> its current (deformed/moved) positions + faces
            (source.to_mesh()); the soft/rigid case.
          * a point cloud (N,3 array) -- particles or a LIQUID/SPH front -> surfaced by wrapping the points in a
            metaball field (sum of Gaussians, `radius`) and marching its `level` isosurface over `bounds`
            (= surface_mesh_stable); the particle/liquid case.
          * a (density_grid, axes) pair OR a grid with `axes=` given -- a SMOKE/volume density field -> its
            `level` isosurface marched directly. The smoke case.
        `face_type` ('triangle'|'quad'|'ngon') applies the meshpoly merge to the result. Returns a Mesh (or, for
        a point cloud / field, the surface_mesh_stable dict via the field path). This keeps everything on the
        engine's own field<->mesh bridge -- particles are a metaball field, smoke is a field, a soft body is the
        mesh it was built from, all surfaced by the same marching tetrahedra."""
        from holographic_softbody import SoftBody, RigidBody
        from holographic_meshbridge import metaball_field, marching_tetrahedra_vec
        import numpy as _np
        if isinstance(source, (SoftBody, RigidBody)):          # soft / rigid: re-export the carried faces
            mesh = source.to_mesh()
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        if isinstance(source, tuple) and len(source) == 2:     # (density_grid, axes): smoke/volume isosurface
            values, ax = source
            mesh = marching_tetrahedra_vec(_np.asarray(values, float), ax, level=level)
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        arr = _np.asarray(source, float)
        if axes is not None:                                   # a bare density grid + axes: smoke
            mesh = marching_tetrahedra_vec(arr, axes, level=level)
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        if arr.ndim == 2 and arr.shape[1] in (2, 3):           # a point cloud: particles / liquid -> metaball surface
            if bounds is None:
                lo = arr.min(0) - 3 * radius; hi = arr.max(0) + 3 * radius
                bounds = (lo, hi)
            field = metaball_field(arr, radius=radius)
            return self.surface_mesh_stable(field, bounds, resolution=resolution, level=level, face_type=face_type)
        raise ValueError("source must be a SoftBody/RigidBody, a point cloud (N,3), or (density_grid, axes)")

    def cloth(self, rows, cols, spacing=1.0, compliance=0.0):
        """A rectangular cloth softbody (structural + shear distance constraints, top row pinned)."""
        from holographic_softbody import SoftBody
        return SoftBody.cloth(rows, cols, spacing=spacing, compliance=compliance)

    def cloth3d(self, rows, cols, spacing=1.0, compliance=0.0, bending=None):
        """A 3-D cloth that drapes under gravity; pass `bending` (a compliance) to add bend springs that
        resist folding so the sheet stays flatter. See holographic_softbody.SoftBody.cloth3d."""
        from holographic_softbody import SoftBody
        return SoftBody.cloth3d(rows, cols, spacing=spacing, compliance=compliance, bending=bending)

    def soft_box(self, nx, ny, nz, spacing=1.0, compliance=0.0, volume_compliance=0.0):
        """A soft 3-D solid (a lattice of tetrahedra with volume constraints) that resists being squashed and
        springs back. See holographic_softbody.SoftBody.soft_box."""
        from holographic_softbody import SoftBody
        return SoftBody.soft_box(nx, ny, nz, spacing=spacing, compliance=compliance,
                                 volume_compliance=volume_compliance)

    def rope(self, n, spacing=1.0, compliance=0.0, start=(0.0, 0.0)):
        """A hanging rope softbody of n particles (particle 0 pinned)."""
        from holographic_softbody import SoftBody
        return SoftBody.rope(n, spacing=spacing, compliance=compliance, start=start)

    def rigid_body(self, positions, inv_mass=None, velocities=None):
        """A hardbody via shape matching (polar decomposition): falls and rotates under forces but never
        deforms. See holographic_softbody.RigidBody."""
        from holographic_softbody import RigidBody
        return RigidBody(positions, inv_mass=inv_mass, velocities=velocities)

    def diffuse_field(self, field, amount):
        """Diffuse a scalar/component grid field by the heat kernel via the FFT -- a Gaussian bind on the
        torus (the engine's own operator). Conserves mass. See holographic_fields."""
        from holographic_fields import diffuse
        return diffuse(field, amount)

    def make_incompressible(self, vx, vy):
        """Pressure-project a velocity field to divergence-free (the fluid solver's Helmholtz step, an FFT
        solve). Returns (vx, vy) with divergence ~0. See holographic_fields.project_divergence_free."""
        from holographic_fields import project_divergence_free
        return project_divergence_free(vx, vy)

    def field_divergence(self, vx, vy):
        """The divergence field of a velocity grid (compression/expansion), computed spectrally."""
        from holographic_fields import divergence
        return divergence(vx, vy)

    def advect_field(self, field, vx, vy, dt=0.1):
        """Transport a scalar field (density/temperature) along a velocity field (semi-Lagrangian)."""
        from holographic_fields import advect
        return advect(field, vx, vy, dt)

    def fluid_step(self, vx, vy, density, dt=0.1, viscosity=0.0, fx=None, fy=None, source=None, solid=None):
        """One Stable-Fluids step on the torus (add force -> diffuse -> project -> advect), built on the FFT.
        Returns (vx, vy, density). Pass `solid` (a 0/1 mask) for an obstacle the flow goes around. See
        holographic_fields.fluid_step."""
        from holographic_fields import fluid_step
        return fluid_step(vx, vy, density, dt=dt, viscosity=viscosity, fx=fx, fy=fy, source=source, solid=solid)

    def smoke_step(self, vx, vy, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
                   buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None, solid=None):
        """One smoke step: temperature drives velocity by buoyancy (hot rises), vorticity confinement keeps it
        curly, density + temperature advect with the flow. Pass `solid` for an obstacle smoke flows around.
        Returns (vx, vy, density, temperature). See holographic_fields.smoke_step."""
        from holographic_fields import smoke_step
        return smoke_step(vx, vy, density, temperature, dt=dt, viscosity=viscosity, ambient=ambient,
                          buoyancy=buoyancy, gravity=gravity, confinement=confinement,
                          dens_source=dens_source, temp_source=temp_source, solid=solid)

    def fluid_step_3d(self, vx, vy, vz, density, dt=0.1, viscosity=0.0, fx=None, fy=None, fz=None, source=None,
                      solid=None):
        """One 3-D Stable-Fluids step on a 3-D periodic grid (the same FFT solver, generalised via the n-D real
        FFT). Pass `solid` (a 3-D mask, e.g. from sphere_mask) for an obstacle the flow goes around. Returns
        (vx, vy, vz, density). See holographic_fields.fluid_step_3d."""
        from holographic_fields import fluid_step_3d
        return fluid_step_3d(vx, vy, vz, density, dt=dt, viscosity=viscosity, fx=fx, fy=fy, fz=fz,
                             source=source, solid=solid)

    def smoke_step_3d(self, vx, vy, vz, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
                      buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None,
                      solid=None):
        """One 3-D smoke step: buoyancy lifts hot fluid along +y, 3-D vorticity confinement keeps it curly,
        density + temperature advect. Pass `solid` for a 3-D obstacle smoke rises and curls around. Returns
        (vx, vy, vz, density, temperature). See holographic_fields.smoke_step_3d."""
        from holographic_fields import smoke_step_3d
        return smoke_step_3d(vx, vy, vz, density, temperature, dt=dt, viscosity=viscosity, ambient=ambient,
                             buoyancy=buoyancy, gravity=gravity, confinement=confinement,
                             dens_source=dens_source, temp_source=temp_source, solid=solid)

    def make_incompressible_3d(self, vx, vy, vz):
        """Pressure-project a 3-D velocity field to divergence-free (the 3-D FFT Helmholtz solve). Returns
        (vx, vy, vz). See holographic_fields.project_divergence_free_3d."""
        from holographic_fields import project_divergence_free_3d
        return project_divergence_free_3d(vx, vy, vz)

    def field_divergence_3d(self, vx, vy, vz):
        """The 3-D divergence field of a velocity grid, computed spectrally."""
        from holographic_fields import divergence_3d
        return divergence_3d(vx, vy, vz)

    def spectral_field(self, shape, beta=2.0, seed=0):
        """Synthesise a SEAMLESS FRACTAL volume (2-D or 3-D) in the Fourier domain -- a 1/f^beta procedural
        field that tiles with no seam (periodic by construction) and is reproducible from just (shape, beta,
        seed). The demoscene 'rich volume from a tiny seed' move on the engine's FFT; composes with tile_field
        / the fluid solver / the archive. See holographic_fields.spectral_field."""
        from holographic_fields import spectral_field
        return spectral_field(shape, beta=beta, seed=seed)

    def seam_continuity(self, field, axis=0):
        """How seamlessly a field tiles along `axis` (~1.0 = seamless, >>1 = visible seam). See
        holographic_fields.seam_continuity."""
        from holographic_fields import seam_continuity
        return seam_continuity(field, axis=axis)

    def disc_mask(self, shape, center, radius):
        """A circular solid-obstacle mask (1 inside the disc) for fluid_step/smoke_step `solid`. See
        holographic_fields.disc_mask."""
        from holographic_fields import disc_mask
        return disc_mask(shape, center, radius)

    def enforce_solid(self, vx, vy, solid_mask, solid_vx=0.0, solid_vy=0.0, iters=2):
        """Force the flow to respect a solid obstacle (velocity -> solid velocity inside the mask, then
        re-project so the flow diverts around). See holographic_fields.enforce_solid."""
        from holographic_fields import enforce_solid
        return enforce_solid(vx, vy, solid_mask, solid_vx=solid_vx, solid_vy=solid_vy, iters=iters)

    def sphere_mask(self, shape, center, radius):
        """A spherical solid-obstacle mask (1 inside the ball) for fluid_step_3d/smoke_step_3d `solid` -- the
        3-D lift of disc_mask. See holographic_fields.sphere_mask."""
        from holographic_fields import sphere_mask
        return sphere_mask(shape, center, radius)

    def enforce_solid_3d(self, vx, vy, vz, solid_mask, solid_vx=0.0, solid_vy=0.0, solid_vz=0.0, iters=2):
        """The 3-D immersed boundary: force flow to the solid's velocity inside the mask, then re-project so it
        diverts around the ball. See holographic_fields.enforce_solid_3d."""
        from holographic_fields import enforce_solid_3d
        return enforce_solid_3d(vx, vy, vz, solid_mask, solid_vx=solid_vx, solid_vy=solid_vy,
                                solid_vz=solid_vz, iters=iters)

    def spatial_hash_pairs(self, positions, radius):
        """Find all index pairs within `radius` via a uniform-grid cull -- O(N) expected vs O(N^2), the
        'cull, don't batch' primitive (any dimension). Reusable for collision, particle interaction, patch
        matching. See holographic_fields.spatial_hash_pairs."""
        from holographic_fields import spatial_hash_pairs
        return spatial_hash_pairs(positions, radius)

    def buoyancy_force(self, temperature, density=None, alpha=0.0, beta=1.0, ambient=0.0):
        """Boussinesq buoyancy force from a temperature (and optional density) field -- hot rises, heavy sinks.
        Returns (fx, fy). See holographic_fields.buoyancy_force."""
        from holographic_fields import buoyancy_force
        return buoyancy_force(temperature, density=density, alpha=alpha, beta=beta, ambient=ambient)

    def vorticity_confinement(self, vx, vy, epsilon=0.5):
        """Vorticity confinement force (Fedkiw 2001) -- restores the small vortices semi-Lagrangian advection
        damps, keeping smoke curly. Returns (fx, fy). See holographic_fields.vorticity_confinement."""
        from holographic_fields import vorticity_confinement
        return vorticity_confinement(vx, vy, epsilon=epsilon)

    def particle_system(self, positions, velocities=None):
        """A ParticleSystem on the grid: particles feel forces (gravity, attractors, any (N,2) force a VSA
        program supplies) and can ride a solved velocity field. See holographic_fields.ParticleSystem."""
        from holographic_fields import ParticleSystem
        return ParticleSystem(positions, velocities)

    def attractor_force(self, positions, center, strength=1.0, softening=1.0):
        """A force pulling particles toward a point (negative strength repels). See holographic_fields."""
        from holographic_fields import attractor_force
        return attractor_force(positions, center, strength=strength, softening=softening)

    def pairwise_repulsion(self, positions, radius, strength=1.0):
        """Short-range particle-particle repulsion (the n-body short-range force), CULLED by spatial_hash_pairs
        so it is O(N + pairs) not O(N^2). Returns an (N, D) force array to pass to particle_system.step(force=)
        -- granular piles, collision avoidance, flocking separation. See holographic_fields.pairwise_repulsion."""
        from holographic_fields import pairwise_repulsion
        return pairwise_repulsion(positions, radius, strength=strength)

    def blue_noise_sample(self, radius, bounds, k=30, seed=0):
        """Poisson-disk (blue-noise) point sampling by Bridson dart-throwing: a maximal point set with every
        pair >= `radius` apart and the blue-noise spectrum (suppressed low frequencies, a ring at the spacing).
        The exclusion principle done right -- nearly matches adaptive matching-pursuit placement and beats random
        by ~3 dB on a fixed-budget splat fit, and is the gold standard for stippling / particle init / Monte
        Carlo. `bounds`=(min,max), any dimension. See holographic_sampling.poisson_disk_sample."""
        from holographic_samplinghome import Sampling                 # via the Sampling home (consolidation R4)
        return Sampling.poisson_disk(radius, bounds, k=k, seed=seed)

    def sample_field(self, field, positions):
        """Read a grid field at continuous particle positions (bilinear, periodic) -- how particles feel a
        VSA-encoded or solved field. See holographic_fields.sample_field."""
        from holographic_fields import sample_field
        return sample_field(field, positions)

    def plan_stage_execution(self, stages, frames):
        """Decide bake-vs-compute PER pipeline stage (PW3), extending adaptive.plan_render's break-even to the
        stage level: static stages bake once reused across enough frames, dynamic (and unannotated) stages compute.
        Returns a per-stage plan with reasons. See holographic_stageplan."""
        from holographic_stageplan import plan_stages
        return plan_stages(stages, frames)

    def plan_pipeline_bakes(self, cfg, frames, registry=None, options=None, cache=None):
        """Compile a pipeline (PW2) then decide which of its stages to BAKE vs COMPUTE over `frames` frames (PW3) --
        the decision layer that tells PW1's bake_pipeline which stages are worth baking. Returns (pipeline, plan).
        See holographic_stageplan / holographic_pipecompile."""
        from holographic_stageplan import plan_stages
        pipe = self.compile_pipeline(cfg, registry=registry, options=options, cache=cache)
        return pipe, plan_stages(pipe.stages, frames)

    def diffuse_readout(self, field, amount, k):
        """Read out the matter model's LINEAR diffusion sub-step at any time k in ONE evaluation (PW4): diagonalise
        the diffusion bind in the Fourier basis and raise its transfer to the k-th power, instead of marching k
        steps (k may be fractional). Matches k marched diffusions exactly. See holographic_simreadout. Nonlinear
        advection/buoyancy/tension still march."""
        from holographic_simreadout import diffuse_at
        return diffuse_at(field, amount, k)

    def diffuse_steady_state(self, field):
        """The closed-form limit of unbounded diffusion (PW4): the flat mean field -- every non-DC mode decays, the
        mean is preserved -- without marching to it. See holographic_simreadout."""
        from holographic_simreadout import diffuse_limit
        return diffuse_limit(field)

    def compile_pipeline(self, cfg, registry=None, options=None, cache=None):
        """Compile a render/sim pipeline's plan ONCE per config (PW2): select+auto-include+toposort the stages,
        keyed by the config's content, so repeated frames reuse the ordered Pipeline with no re-planning. Reuses the
        content-addressed compile cache. See holographic_pipecompile."""
        from holographic_pipecompile import compiled_pipeline
        return compiled_pipeline(cfg, registry=registry, options=options, cache=cache)

    def run_pipeline(self, cfg, scene=None, seed=0, prev_frame=None, renderer=None, registry=None, options=None, cache=None):
        """Run a pipeline for `cfg`, compiling its plan on the first frame and reusing it thereafter (PW1/PW2) -- the
        everyday frame-loop entry point. Returns the final FrameState. See holographic_pipecompile."""
        from holographic_pipecompile import run_compiled
        return run_compiled(cfg, scene=scene, seed=seed, prev_frame=prev_frame, renderer=renderer,
                            registry=registry, options=options, cache=cache)

    def bake_view_lut(self, metallic=1.0, base_color=(1.0, 1.0, 1.0), res_view=16, res_rough=16, samples=8192, seed=0):
        """Pre-integrate the view-DEPENDENT specular (MC3): bake directional_albedo over a (view_cos, roughness)
        grid ONCE, so per-pixel specular reflectance is a bilinear LUT lookup instead of a hemisphere integral --
        the 'add a dimension' move that turns the last view-dependent axis into a query. Returns a ViewLUT with
        .sample(view_cos, roughness). See holographic_viewlut."""
        from holographic_viewlut import bake_view_lut
        return bake_view_lut(metallic=metallic, base_color=base_color, res_view=res_view,
                             res_rough=res_rough, samples=samples, seed=seed)

    def bake_material(self, material, lo, hi, res=24):
        """Bake a material's VIEW-INDEPENDENT channels into field lookups (MC2): a procedural texture becomes a
        trilinear grid sample (O(1) per hit, no field re-evaluation), constants stay folded (MC1). `lo,hi` are the
        object bounds to bake over. Returns a shade(points)->channels kernel of folds + lookups. See
        holographic_matbake. The remaining view-DEPENDENT channels (specular) are MC3's LUT."""
        from holographic_matbake import bake_material
        return bake_material(material, lo, hi, res=res)

    def compile_material(self, material, cache=None):
        """Compile a material's socket graph into ONE cached shade(points)->channels kernel (MC1): built once, keyed
        by the material's content spec, reused for every hit/instance/frame; constant channels are folded so only the
        procedural sockets re-resolve per hit. Reuses the content-addressed compile cache. See holographic_matcompile."""
        from holographic_matcompile import compiled_shader
        return compiled_shader(material, cache=cache)

    def scatter_surface(self, instance, sdf, bounds, count, scale=1.0, density=None, cell_size=0.25, seed=0):
        """Scatter instances (grass, rocks, barnacles...) onto ANY surface -- a scatter LAYER that emits geometry
        instead of colour, weighted by an optional density map, reusing emit_from_surface. Returns the placements
        (points, normals, per-placement bound vectors) + the bundled, region-queryable layer vector. See
        holographic_scatterlayer."""
        from holographic_scatterlayer import ScatterLayer
        layer = ScatterLayer(instance, count, scale=scale, density=density, cell_size=cell_size, seed=seed)
        return layer.apply(sdf, bounds)

    def scale_node(self, scene, lod_px=8.0):
        """A cosmic-scale summariser over a scene hierarchy: a parent carries the monoid-accumulated value of its
        children (mass SUMs, appearance BUNDLES) via the wired distribute_compute; zoom out reads the summary, zoom
        in descends. Same accumulation at every scale (atom -> ... -> galaxy). See holographic_scalenode."""
        from holographic_scalenode import ScaleNode
        return ScaleNode(scene, mind=self, lod_px=lod_px)

    def make_mixture(self, shape, solvent_density=1.0, buoyancy=1.0, tension=0.0):
        """Create a Mixture -- the multi-channel matter model (smoke/dye/milk/oil-water are this with different
        dials). Add components with .add(name, field, density, diffusivity), then advance with matter_step. See
        holographic_mixture."""
        from holographic_mixture import Mixture
        return Mixture(shape, solvent_density=solvent_density, buoyancy=buoyancy, tension=tension)

    def matter_step(self, mix, vx, vy, dt=0.1, drift_strength=0.0):
        """Advance a Mixture one step on ONE shared incompressible flow: blend density -> buoyancy -> a single
        fluid_step -> per-channel advect + diffuse (+ optional double-well tension and drift) -> renormalise. This
        delegates to the wired advect/diffuse/buoyancy_force/fluid_step; it is NOT a second solver. Returns the
        updated (vx, vy); the mixture is mutated in place. See holographic_mixture."""
        from holographic_mixture import matter_step
        return matter_step(mix, vx, vy, dt=dt, drift_strength=drift_strength)

    def smoke_preset(self, name="rising", nx=48, ny=48, steps=40, dt=0.1, seed=0):
        """Run one of the six named SMOKE PRESETS on the wired FFT smoke solver and return its fields
        (density/temperature/vx/vy). The presets (rising, wispy, billow, heavy, still_room, stratified) are just
        dial settings -- smoke is the 1-channel, tension-0 corner of the matter model -- so this delegates to
        smoke_step, it does not add a solver. Use smoke_preset_names() for the list. See holographic_smokepresets."""
        from holographic_smokepresets import simulate
        return simulate(name, nx=nx, ny=ny, steps=steps, dt=dt, seed=seed)

    def smoke_preset_names(self):
        """The available smoke preset names."""
        from holographic_smokepresets import preset_names
        return preset_names()

    def scatter_to_field(self, shape, positions, values):
        """The adjoint of sample_field: imprint per-particle values onto a grid (bilinear, periodic) -- e.g. a
        moving body depositing momentum into the fluid velocity grid (cloth->fluid coupling)."""
        from holographic_fields import scatter_to_field
        return scatter_to_field(shape, positions, values)

    def drag_force(self, positions, velocities, vx, vy, k=1.0):
        """Drag force on particles from a fluid, F = k*(v_fluid - v_particle) (fluid->cloth coupling). See
        holographic_fields.drag_force."""
        from holographic_fields import drag_force
        return drag_force(positions, velocities, vx, vy, k=k)

    def sample_field_3d(self, field, positions):
        """Read a 3-D grid field at continuous positions (N,3), trilinear+periodic -- how a softbody/particle
        feels a 3-D solved or VSA-encoded field. See holographic_fields.sample_field_3d."""
        from holographic_fields import sample_field_3d
        return sample_field_3d(field, positions)

    def scatter_to_field_3d(self, shape, positions, values):
        """The exact adjoint of sample_field_3d: imprint per-node values onto a 3-D grid (trilinear, periodic)
        -- the body->fluid half of 3-D two-way coupling. See holographic_fields.scatter_to_field_3d."""
        from holographic_fields import scatter_to_field_3d
        return scatter_to_field_3d(shape, positions, values)

    def drag_force_3d(self, positions, velocities, vx, vy, vz, k=1.0):
        """Drag on nodes from a 3-D fluid: k*(v_fluid - v_node), sampled trilinearly -- so a softbody couples to
        fluid_step_3d exactly as it does to the 2-D solver (pass as external_force). See
        holographic_fields.drag_force_3d."""
        from holographic_fields import drag_force_3d
        return drag_force_3d(positions, velocities, vx, vy, vz, k=k)

    def encode_pairs(self, keys, values):
        """Encode parallel arrays of keys and values -- bundle of bind(key_i, value_i) -- in ONE batched FFT
        (bundle_bind), the vectorised form of the role/filler encode loop VSA programs run constantly. Keeps
        the operation inside the holographic space (one array op) instead of a Python loop of per-pair binds.
        (Distinct from encode_record(fields): that takes a {field: value} record and pairs with decode_record;
        this takes two parallel arrays. Renamed from a former encode_record overload that shadowed the record
        encoder -- one name, one faculty.)"""
        from holographic_ai import bundle_bind
        return bundle_bind(keys, values)

    def unbind_keys(self, trace, keys):
        """Unbind one trace against many keys in ONE batched FFT (unbind_all) -- the vectorised form of the
        per-key unbind loop decoders/resonators run. Returns (k, dim): row i is the estimate freed by key i."""
        from holographic_ai import unbind_all
        return unbind_all(trace, keys)

    def nearest_in(self, query, matrix):
        """Nearest row of a codebook matrix to `query` -- argmax(matrix @ query) (nearest), the reusable
        matmul form of a per-row cosine loop. Exact. Returns (index, score)."""
        from holographic_ai import nearest
        return nearest(query, matrix)

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

    def splat_aniso(self, field, k=12, steps=200, denoise=False, early_stop=False, stats=None):
        """Represent an n-D field (a 2-D image or a 3-D volume) as a superposition of ANISOTROPIC Gaussian
        splats -- the real 3D-Gaussian-Splatting primitive (oriented, elliptical/ellipsoidal Gaussians with a
        full covariance), fit by gradient descent on the reconstruction MSE (holographic_splat.aniso_fit:
        analytical NumPy gradients + a tiny built-in Adam, no autodiff framework). The anisotropic, n-D
        extension of `splat_field`'s isotropic matching pursuit: one aligned splat replaces many circular ones
        wherever structure is oriented or elongated, in 2-D OR 3-D. Returns (splats, rendered) where each splat
        is (center, amplitude, L) with L the inverse-covariance Cholesky factor; denoise=True returns just the
        rendered field (a few smooth Gaussians cannot hold high-frequency noise).

        ADAPTIVE STOP (C3, opt-in early_stop=True; pass stats={} to read stats['steps']): stop the fixed Adam
        schedule once the reconstruction MSE has converged -- measured ~40% fewer steps at a few-percent MSE
        cost. KEPT CAVEAT: this is a SPEED/QUALITY knob, not free -- a continuous fit has only a soft plateau,
        not the resonator's exact certificate, so stopping always costs a little MSE; off by default
        (bit-identical to the fixed-step fit). A min_steps floor inside aniso_fit guards Adam's ~30-step warm-up
        (a naive relative test would mistake the warm-up for convergence and stop with a terrible fit).

        KEPT NEGATIVE / SCOPE: the loss is non-convex, so this finds a LOCAL optimum -- more splats do not help
        monotonically (a clean K=4 fit can beat a messier K=8 one) and the result depends on the isotropic warm
        start; and this is the from-scratch core of 3DGS only -- no tile rasteriser, no spherical-harmonic
        view-dependent colour, no GPU speed."""
        from holographic_splat import aniso_fit
        splats, rendered = aniso_fit(np.asarray(field, float), k, steps=steps,
                                     early_stop=early_stop, stats=stats)
        return rendered if denoise else (splats, rendered)

    def splat_densify(self, field, k=12, stage_steps=(50, 80, 210), denoise=False, stats=None):
        """Fit an n-D field with K ANISOTROPIC Gaussian splats COARSE-TO-FINE (C1) -- 3D-Gaussian-Splatting
        densification, from scratch (holographic_splat.densify_fit). Rather than placing all K splats at once and
        running one joint gradient fit (`splat_aniso`), grow the set in stages: place a fraction on the current
        residual (coarse scales first), jointly optimise, then place more where the re-optimised reconstruction
        still errs, and optimise again. `stage_steps` is the Adam steps per stage (the last should be long enough
        to fully converge the whole set). Returns (splats, rendered); denoise=True returns just the rendered
        field; pass stats={} to read stats['stages'].

        WHY USE THIS over splat_aniso (measured): the staged placement is a far better WARM START for the final
        joint fit, landing in a better basin of the non-convex loss. On a multi-scale target (a broad blob + small
        sharp details) it reaches MSE the one-shot CANNOT reach at any step count (~1e-6 vs ~1e-3, where the
        one-shot then DIVERGES past ~300 steps) -- directly addressing splat_aniso's local-optimum kept negative
        (its result 'depends on the isotropic warm start'; a staged warm start is a much better one). The trade is
        more total compute (several optimisation rounds); the win is on MULTI-SCALE content -- on a single-scale
        field the one-shot is already near-optimal."""
        from holographic_splat import densify_fit
        splats, rendered = densify_fit(np.asarray(field, float), k, stage_steps=stage_steps, stats=stats)
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

    def splat_scene(self, field, grid=16, tile=8, levels=5, k=30, dim=4096, seed=0):
        """Build a CONTENT-ADDRESSABLE splat scene from a field -- a coarse occupancy map stored as ONE
        hypervector per tile that you can query by region ('what is at this cell?'). Fits k Gaussian splats,
        then encodes them with TILED bundling so region recall stays accurate at FINE resolution. The plain
        single-bundle scene's region readback is decode-via-cleanup (unbind a cell role, clean up to a level)
        and so caps as the grid gets finer -- measured ~98% at grid 16 down to ~75% at grid 32 at dim 4096 --
        while routing each cell to a tile bundle of at most tile*tile bindings holds recall ~100% at any total
        resolution, the same chunk-to-beat-the-cap trade the route/sequence faculties make (one vector per
        tile). The EXACT complement is splat_archive's region (explicit per-splat); this is the compact,
        content-addressable, coarse-but-robust one. Returns the scene; read a cell back with splat_region."""
        from holographic_splat import splat_fit, splat_bundle_tiled
        field = np.asarray(field, float)
        splats = splat_fit(field, k)
        return splat_bundle_tiled(splats, field.shape, dim=dim, grid=grid, levels=levels, tile=tile, seed=seed)

    def splat_region(self, scene, cell):
        """Read a region's occupancy in [0, 1] back from a splat scene built by splat_scene -- the
        content-addressable 'what is at grid cell (gy, gx)?' query, routed to the cell's tile bundle."""
        from holographic_splat import recall_region_tiled
        return recall_region_tiled(scene, tuple(cell))

    def splat_prune(self, splats, target, keep):
        """PRUNE a splat set to its `keep` highest-contribution splats (largest |amplitude|, since each splat's
        reconstruction energy is amp^2 for the engine's unit-norm gaussians) and refit the survivors (holographic_
        splatprune). Contribution-ranked prune + refit degrades gracefully and beats naive pruning by a wide margin
        (~20 dB at half the splats on a smooth field). Returns the pruned, refitted splat list. The splat twin of
        mesh decimation."""
        from holographic_splatprune import splat_prune
        return splat_prune(splats, target, keep)

    def splat_merge(self, splats, target, radius):
        """MERGE splats closer than `radius` into one (amplitude-weighted centre and scale, summed amplitude) and
        refit (holographic_splatprune) -- reduces the count with bounded quality loss. Returns the merged list. Kept
        negative: merge is lossy by construction (one Gaussian cannot equal two); the radius trades count for
        quality."""
        from holographic_splatprune import splat_merge
        return splat_merge(splats, target, radius)

    def splat_lod_chain(self, splats, target, keeps=(40, 20, 10, 5)):
        """Build a splat LEVEL-OF-DETAIL chain (holographic_splatprune): prune to each count in `keeps`, measuring
        reconstruction PSNR at each. Returns a fine->coarse list of (splats, count, psnr); the first is the refitted
        full set. The splat-domain twin of mesh_lod_chain -- the engine's error-budget resolution selection, with the
        budget in PSNR. Pair with splat_select_lod."""
        from holographic_splatprune import splat_lod_chain
        return splat_lod_chain(splats, target, keeps=keeps)

    def splat_select_lod(self, chain, min_psnr):
        """Choose a splat LOD by QUALITY budget (holographic_splatprune): the index of the FEWEST-splat level whose
        PSNR still meets `min_psnr` -- the cheapest splat set that looks right -- falling back to the finest level if
        none clears it. The PSNR-budget analog of mesh_select_lod's pixel budget."""
        from holographic_splatprune import select_splat_lod
        return select_splat_lod(chain, min_psnr)

    def splat_clone_split(self, splats, target, n_densify=None, scale_thresh=None):
        """SCALE-AWARE CLONE-VS-SPLIT DENSIFICATION (holographic_splatdensify) -- the 3DGS densification DISTINCTION
        that the engine's existing splat_densify (`densify_fit`, staged residual placement) was missing: it adds
        capacity where error is high but is SCALE-BLIND. This refines an EXISTING splat set by asking, per high-error
        splat, whether the region needs COVERING or RESOLVING. Rank the splats by the residual error in their
        footprint and, for the highest-error ones, CLONE if narrow (sigma < scale_thresh -- add a same-scale copy to
        COVER an under-served region) else SPLIT (replace a wide splat with two narrower ones to RESOLVE fine
        structure it was smearing). `n_densify` caps how many splats to densify (default all); `scale_thresh` defaults
        to the set's median sigma. The inverse of splat_prune/merge: those COARSEN, this REFINES -- and it sharpens
        WHERE new capacity goes (measured: scale-aware beats always-clone and always-split on a mixed-error target at
        a fixed budget, and the WRONG move can be worse than nothing -- splitting a small splat loses coverage). Kept
        negative: complements splat_densify's from-scratch placement (the new part is the cover-vs-resolve decision on
        an existing set); isotropic splats."""
        from holographic_splatdensify import clone_split_densify
        return clone_split_densify(splats, target, n_densify=n_densify, scale_thresh=scale_thresh)

    def splat_relocate(self, splats, target, dead_frac=0.05):
        """MCMC BIRTH-DEATH RELOCATION (holographic_relocate) -- the successor to evict-rarest. The engine's bounded
        memory DROPS the rarest when a store is full (the creature's `memory_cap` path); 3DGS-as-MCMC instead
        RELOCATES a dead atom to an under-represented region, CONSERVING the budget. This moves the DEAD splats
        (|amplitude| below `dead_frac` of the largest) each to the current residual peak -- the most under-represented
        region -- subtracting after each so successive relocations find distinct peaks, and keeping the splat COUNT
        fixed (a birth-death move, not a drop). The discrete kin of the B10 generative-denoising sampler. Measured:
        relocating to residual peaks beats DROPPING (~4x lower MSE -- eviction shrinks and wastes the budget) and
        beats RANDOM relocation (the principled target matters), at a conserved count. Kept negative: the drop was
        already in the box (the creature's eviction is unchanged); this conserves capacity instead -- isotropic
        splats, and a no-op when nothing is dead."""
        from holographic_relocate import birth_death_relocate
        return birth_death_relocate(splats, target, dead_frac=dead_frac)

    def render_scene(self, tag_list, S=96, seed=0):
        """Render composed attribute tags to an actual RGB image via the scene renderer."""
        from holographic_scene import make_scene
        return make_scene([(t["shape"], t["colour"]) for t in tag_list], S=S, seed=seed)

    def camera(self, eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
               fov_deg=50.0, aspect=1.0, near=0.05, far=100.0):
        """A pinhole Camera (eye looks at target, vertical fov) -- view+projection matrices and per-pixel rays.
        The viewpoint for rasterised and volumetric renders. See holographic_render.Camera."""
        from holographic_render import Camera
        return Camera(eye=eye, target=target, up=up, fov_deg=fov_deg, aspect=aspect, near=near, far=far)

    def light(self, kind="directional", direction=(-0.4, -0.8, -0.5), position=(2.0, 3.0, 2.0),
              color=(1.0, 1.0, 1.0), intensity=1.0):
        """A Light: 'directional' (sun), 'point', or 'ambient' (fill). See holographic_render.Light."""
        from holographic_render import Light
        return Light(kind=kind, direction=direction, position=position, color=color, intensity=intensity)

    def render_mesh(self, mesh, camera, width=512, height=512, lights=None, base_color=(0.8, 0.8, 0.8),
                    background=(0.05, 0.06, 0.08), ambient=0.15, vectorized=True):
        """Rasterise a mesh to an (H,W,3) RGB image with a z-buffer and Lambert shading (frustum + back-face
        culled). `base_color` may be a PBRMaterial's base_color. vectorized=True (default) uses the batched
        fragment-scatter path (the per-triangle Python loop ported to one array op -- ~8-15x faster, image
        identical); vectorized=False is the readable reference loop. CPU renderer -- the authoring brain's
        offline / preview frame; the GPU stays the muscle for a heavy interactive viewport. See holographic_render."""
        from holographic_render import rasterize_mesh
        if hasattr(base_color, "base_color"):
            base_color = base_color.base_color
        return rasterize_mesh(mesh, camera, width=width, height=height, lights=lights,
                              base_color=base_color, background=background, ambient=ambient, vectorized=vectorized)

    def ray_path_index(self, objects, camera, width=256, height=256, sun="bright", sky="clear"):
        """Build a BIDIRECTIONAL ray<->object index for a scene: which objects each camera ray TOUCHED along its path
        (primary hit + objects seen THROUGH glass). `index.pixels_touching(ids)` then returns the exact pixels to
        re-shade when those objects change -- INCLUDING indirect pixels (an object seen through glass) that a
        primary-id-only incremental renderer misses. Pair with `delta_reshade_scene` for a bounded, bit-exact update.
        The trace already knows where every ray went; this keeps it instead of re-gathering it every frame.
        See holographic_rayindex."""
        from holographic_semantic import _scene_setup
        from holographic_rayindex import build_ray_index
        ctx = _scene_setup(objects, True, sky, sun, (0.75, 0.9, 0.85))
        if ctx is None:
            return None
        return build_ray_index(ctx, camera, width, height)

    def delta_reshade_scene(self, edited_objects, index, changed_ids, base_frame, camera, sun="bright", sky="clear"):
        """Apply a material/colour/light EDIT as a bounded delta: re-shade ONLY the pixels whose ray touched a changed
        object (from `ray_path_index`), composite into `base_frame`. Deterministic -> the updated pixels are bit-exact
        vs a full re-render, at a fraction of the work, and the through-glass pixels update correctly. Geometry must be
        unchanged (a colour/material/light edit); a MOVE needs the index rebuilt for the affected region first."""
        from holographic_semantic import _scene_setup
        from holographic_rayindex import delta_reshade
        ctx_new = _scene_setup(edited_objects, True, sky, sun, (0.75, 0.9, 0.85))
        return delta_reshade(ctx_new, index, changed_ids, base_frame, camera)

    def brick_ray_index(self, objects, camera, width=256, height=256, grid=10, samples=12, sun="bright", sky="clear"):
        """Build a REGION-keyed ray index: which spatial bricks each ray traversed (eye->hit). Where ray_path_index
        keys edits by object, this keys them by REGION, so a MOVE (geometry change) gets the same bounded delta -- the
        affected pixels are the rays that passed through the bricks the object vacated or now occupies. Pair with
        delta_reshade_move. See holographic_rayindex.BrickRayIndex."""
        from holographic_semantic import _scene_setup
        from holographic_rayindex import build_brick_index
        ctx = _scene_setup(objects, True, sky, sun, (0.75, 0.9, 0.85))
        if ctx is None:
            return None
        self._last_brick_ctx = ctx
        return build_brick_index(ctx, camera, width, height, grid=grid, samples=samples)

    def delta_reshade_move(self, obj_id, delta, brick_index, base_frame, camera):
        """Apply a MOVE of object `obj_id` by `delta` as a bounded, bit-exact delta: re-shade only the rays that
        traversed the bricks the object vacated (old position) or now occupies (new position), from `brick_ray_index`
        (which cached the scene ctx). Returns (updated_frame, mask, ctx_new). See holographic_rayindex."""
        from holographic_rayindex import delta_reshade_move as _drm
        ctx = getattr(self, "_last_brick_ctx", None)
        if ctx is None:
            raise ValueError("call brick_ray_index(...) first to build the index and cache the scene")
        return _drm(ctx, obj_id, delta, brick_index, base_frame, camera)

    def incremental_renderer(self, camera, width=256, height=256, sun="bright", sky="clear", ground=True, ss=1):
        """A render SESSION: render the first frame, then re-render the SAME scene for FREE (cached), apply colour/
        material/light edits and geometry moves as bounded bit-exact DELTAS, and stream only the changed pixels. This
        is the path for repeated rendering / live editing / pixel streaming -- calling render_scene every frame re-does
        the whole trace even when nothing changed; the session pays only for what changed. Default ss=1 (delta-exact,
        for streaming); use a one-shot render_scene(ss=2) for a final still. See holographic_rayindex.IncrementalRenderer.

        Usage:
            r = mind.incremental_renderer(camera, 256, 256)
            frame, mask = r.render(objects)        # first frame (full); mask = whole frame
            frame, mask = r.render(objects)        # SAME scene -> free, mask empty
            frame, mask = r.edit(0, 'color', 'gold')   # delta; mask = changed pixels
            frame, mask = r.move(1, (0.3, 0, 0))       # delta; mask = changed pixels
            ys, xs, rgb = r.stream_delta(mask)     # the wire payload: only changed pixels
        """
        from holographic_rayindex import IncrementalRenderer
        return IncrementalRenderer(camera, width, height, sun=sun, sky=sky, ground=ground, ss=ss)

    def region_field(self, regions):
        """Compose a LABELLED REGION FIELD: a set of boundaries (SDFs), each tagging the points inside it with how to
        REGARD them -- a material, or a behaviour (cloth / fire / smoke / fluid), or a biome -- resolved by priority.
        One `classify(points)` call then drives material lookup, behaviour (which SIM) lookup, and precise culling
        (points outside every region are known-empty, skipped with no marching). `slice(origin, u, v)` cuts the volume
        open to reveal the material LAYERS. This is the composable substrate for treating anything as mesh / particle /
        smoke / fluid / light over one field. `regions` is a list of holographic_regionfield.Region. See
        holographic_regionfield.

        Example:
            from holographic_regionfield import Region
            from holographic_semantic import _SphereSDF
            rf = mind.region_field([
                Region(_SphereSDF((0,0,0), 1.0), 'crust',  priority=1, material=(0.4,0.3,0.2)),
                Region(_SphereSDF((0,0,0), 0.4), 'core',   priority=2, material=(1.0,0.85,0.3)),
            ])
            img, labels = rf.slice((0,0,0), (1,0,0), (0,1,0))   # cut it open -> see the layers
            keep = rf.cull(points)                              # precise culling for free
        """
        from holographic_regionfield import RegionField
        return RegionField(list(regions))

    def reflect_transform(self, O, D, P_hit, N, bounce=None):
        """A secondary (bounce) ray as a TRANSFORM of its parent: origin -> hit point, direction -> reflected about the
        normal, bounce counter -> +1. N bounces are N applications of this one transform. See holographic_raycoherence."""
        from holographic_raycoherence import reflect_transform as _rt
        return _rt(O, D, P_hit, N, bounce=bounce)

    def coherent_reflection(self, ctx, P, N, D, ids, mirror, width, height, stride=4, var_tol=0.03):
        """Reconstruct the reflection over reflective pixels from a SPARSE trace + gated bilinear interpolation of the
        perpendicular neighbours, with an exact-trace fallback on reflection edges -- because neighbouring reflection
        rays off a smooth surface are coherent, this traces far fewer rays than per-pixel for a close result. Returns
        (reflected (H*W,3), n_traced, n_mirror). Honest limit: sharp reflected-CONTENT edges blur (the coherence is in
        the reflector geometry, not the reflected image). See holographic_raycoherence."""
        from holographic_raycoherence import coherent_reflection as _cr
        return _cr(ctx, P, N, D, ids, mirror, width, height, stride=stride, var_tol=var_tol)

    def ray_pencil(self, O, D, C, R, eps=0.03):
        """Emit a ray's PERPENDICULAR FRAME (centre + 4 marginal rays offset +-eps) and transport it through a
        reflection off a sphere: the reflected pencil converges (concave far wall -> caustic) or diverges (convex cap).
        Returns (P (5,3), D2 (5,3), hit). The pencil's cross-section IS the Gaussian of secondary rays. See
        holographic_raydiff."""
        from holographic_raydiff import transport_pencil
        return transport_pencil(O, D, C, R, eps)

    def caustic_focus(self, O, D, C, R, eps=0.03, s_max=6.0):
        """Where a reflected pencil is tightest -- the focus / caustic point -- and the pencil radius there. A 5-ray
        frame predicts the focus a dense bundle would show. Returns (s_focus, radius). See holographic_raydiff."""
        from holographic_raydiff import transport_pencil, find_focus
        P, D2, hit = transport_pencil(O, D, C, R, eps)
        return find_focus(P, D2, s_max=s_max)

    def lobe_sigma(self, O, D, C, R, s, eps=0.03, roughness=0.0, light_half_angle=0.0):
        """The Gaussian lobe half-width of the whole secondary bundle at distance s: geometric pencil spread combined
        with surface roughness (micro-imperfections -> glossy) and a soft light's angular size (penumbra). One number
        standing in for the entire bundle of secondary rays. See holographic_raydiff."""
        from holographic_raydiff import transport_pencil, lobe_sigma as _ls
        P, D2, hit = transport_pencil(O, D, C, R, eps)
        return _ls(P, D2, s, roughness=roughness, light_half_angle=light_half_angle)

    def dispersion_spread(self, D, N, iors):
        """The chromatic angular fan from refracting one ray at several wavelength IORs (eta = n_in/n_out per colour) --
        the same pencil split by wavelength, which IS dispersion. See holographic_raydiff."""
        from holographic_raydiff import dispersion_spread as _ds
        return _ds(D, N, iors)

    def grid_graph(self, shape, blocked=None):
        """Adjacency dict {cell: [neighbours]} for an N-D grid (a 2D/3D/.../ND maze is the same object). Feed to any
        graph solver. See holographic_ndfield."""
        from holographic_ndfield import grid_graph
        return grid_graph(shape, blocked)

    def solve_grid_maze(self, shape, blocked, start, goal, steps=200, mu=1.5, dt=0.2):
        """Solve an N-D grid maze with the Tero slime-mould flow solver -- the SAME solver the 2D maze used, unchanged,
        because it operates on the graph not the coordinates. 3D (or 7D) is trivial. See holographic_ndfield."""
        from holographic_ndfield import solve_grid_maze
        return solve_grid_maze(shape, blocked, start, goal, steps=steps, mu=mu, dt=dt)

    def sparse_reconstruct(self, oracle, lo, hi, n_seed=96, n_refine=96, bandwidth=None, seed=0):
        """The reusable sparse-probe-interpolate-refine pattern: reconstruct a known deterministic field (oracle) over
        an N-D box from a sparse ADAPTIVE sample (Nadaraya-Watson kernel + refine where uncertain). The pattern under
        coherent reflection, ray differentials, and the radiance field, named once. Returns (points, values,
        reconstruct_fn). See holographic_ndfield."""
        from holographic_ndfield import sparse_reconstruct
        return sparse_reconstruct(oracle, lo, hi, n_seed=n_seed, n_refine=n_refine, bandwidth=bandwidth, seed=seed)

    def navigate_cost_field(self, cost, shape, start, goal, blocked=None, lo=None, hi=None):
        """NAVIGATE a known N-D COST FIELD: discretize it to a grid, weight edges by the field, and return the
        LEAST-COST path start->goal. The one primitive for routing through smoke density (volumetrics), a potential
        (physics), or any resistance/terrain (particles); the uniform maze is the special case. `cost` is a callable
        f(points)->cost or an array of shape `shape`. See holographic_ndfield. (Distinct from navigate_field, which is
        the gravity/attractor field navigator.)"""
        from holographic_ndfield import navigate_field
        return navigate_field(cost, shape, start, goal, blocked=blocked, lo=lo, hi=hi)

    def path_cost(self, path, cost, shape, lo=None, hi=None):
        """Total field cost accumulated along a path -- for comparing a navigated route to a naive straight shot."""
        from holographic_ndfield import path_cost
        return path_cost(path, cost, shape, lo=lo, hi=hi)

    def cost_to_go_field(self, cost, shape, goal, blocked=None, lo=None, hi=None):
        """SOLVE THE WHOLE VALUE FIELD ONCE, then route from ANYWHERE for free. One Dijkstra sweep from the goal yields
        V (cost-to-go at every cell) and a next-step field; after that, routing from any start is an O(path) descent, no
        re-search -- the 'precompute once, read out anywhere' pattern (the SDF bake, PRT) carried into navigation. The
        win grows with the number of agents/queries to that goal (measured ~5x at 8 starts, and each extra route is
        ~free). V is also a POTENTIAL / VALUE FUNCTION: its negative gradient is a physics force, and descent on it is an
        optimal policy -- the same object as a distance field, a physics potential, and an RL value. Returns
        (V, nxt, route) where route(start) -> the cell path to the goal. See holographic_ndfield."""
        from holographic_ndfield import field_weighted_graph, cost_to_go, route_from
        nbr, edge_cost = field_weighted_graph(shape, cost, blocked=blocked, lo=lo, hi=hi)
        V, nxt = cost_to_go(nbr, edge_cost, goal)
        return V, nxt, (lambda start: route_from(nxt, start, goal))

    def straight_line_cells(self, start, goal):
        """The grid cells a straight line start->goal crosses -- the tie-break-independent baseline a naive shot pays."""
        from holographic_ndfield import straight_line_cells
        return straight_line_cells(start, goal)

    def navigate_scene(self, sdf_eval, lo, hi, shape, start_world, goal_world, clearance=0.25):
        """Route an agent through a LIVE SCENE using its own SDF as the cost field: inside geometry is impassable,
        near a surface is costly, so the path threads free space around objects. One structure for drawing AND moving.
        `sdf_eval` is a callable P->signed distance. Returns world-space waypoints. See holographic_ndfield."""
        from holographic_ndfield import navigate_scene
        return navigate_scene(sdf_eval, lo, hi, shape, start_world, goal_world, clearance=clearance)

    def encode_path(self, path, dim=2048, seed=0):
        """A navigated path -> ONE hypervector, COMPOSABLE in a VSA program (bind to a label, bundle several routes,
        query order). Returns (vector, SequenceMemory, keys). See holographic_ndfield."""
        from holographic_ndfield import encode_path
        return encode_path(path, dim=dim, seed=seed)

    def decode_path_step(self, vec, sm, keys, i):
        """Read waypoint i back out of a path hypervector -- the route survives as composable VSA data."""
        from holographic_ndfield import decode_path_step
        return decode_path_step(vec, sm, keys, i)

    def emit_from_surface(self, sdf_eval, n, bounds, speed=1.0, weight=None, seed=0):
        """Spawn particles ON a surface to DRIVE a particle system: samples the zero level-set of `sdf_eval`, returns
        (positions, normals, velocities) with velocity along the outward normal. `speed` and `weight` (emission
        density on the surface) each take a constant OR a map / field / wired output. See holographic_emitter."""
        from holographic_emitter import emit_from_surface
        return emit_from_surface(sdf_eval, n, bounds, speed=speed, weight=weight, seed=seed)

    def advance_particles(self, pos, vel, force=None, dt=0.05, damping=0.0, wrap_to=None):
        """One integration step for an N-D particle set (gravity / attractor / sampled-field force). See holographic_emitter."""
        from holographic_emitter import advance
        return advance(pos, vel, force=force, dt=dt, damping=damping, wrap_to=wrap_to)

    def param(self, value=None, field=None, map=None, domain=None, source=None, default=0.0):
        """Make a connectable parameter SOCKET: a value that is a constant OR wired to a map / field / named output --
        the 'choose a map instead of a number' affordance. Pass the result anywhere a faculty resolves parameters
        (region reflect/roughness, emit speed/weight, a field cost). See holographic_param."""
        from holographic_param import Param
        return Param(value=value, field=field, map=map, domain=domain, source=source, default=default)

    def resolve_param(self, p, points=None, ctx=None, n=None):
        """Resolve any parameter (constant / map / field / socket) to concrete values at `points`. See holographic_param."""
        from holographic_param import resolve_param
        return resolve_param(p, points=points, ctx=ctx, n=n)

    def collide_sdf(self, X, sdf_eval, radius=0.0):
        """ENVIRONMENT collision: push every point inside `sdf_eval` (signed distance < radius) out to the surface --
        keep particles / cloth outside scene geometry. The positional contact resolve behind SoftBody.step(collider=...).
        See holographic_collide."""
        from holographic_collide import resolve_sdf_collision
        return resolve_sdf_collision(X, sdf_eval, radius=radius)

    def sdf_collision_projection(self, sdf_eval, N, D, radius=0.0):
        """A collision PROJECTION callable for project_onto_constraints -- so 'stay outside this surface' is just one
        more constraint in the SAME unified sweep as distance/bend/denoise/resonator (Macklin's one-solver-many-uses).
        See holographic_collide."""
        from holographic_collide import sdf_collision_projection
        return sdf_collision_projection(sdf_eval, N, D, radius=radius)

    def dirty_field(self, shape, lo=None, hi=None, base=0.0):
        """A navigation / physics cost field with DIRTY-FLAG deltas: add movable colliders, then `move` one and only
        its footprint is re-evaluated (O(footprint), grid-size-independent), staying bit-identical to a full rebuild.
        The 'recompute only what changed' render discipline, carried into physics/nav. Returns a DirtyField whose
        `cost_grid()` feeds navigate_field. See holographic_dirtyfield."""
        from holographic_dirtyfield import DirtyField
        return DirtyField(shape, lo=lo, hi=hi, base=base)

    def bake_sdf(self, sdf, lo, hi, res):
        """PRECOMPUTE a scene SDF (anything with `.eval`, and optionally `.ids`) onto a grid, then sample it O(1) --
        the realtime distance-field shortcut. Cost of a sample is independent of the number of primitives, so the ONE
        baked grid speeds every SDF consumer at once (the shader's trace/shadows/AO/reflections, navigation, collision,
        emission). Amortises over many rays/frames/queries. Returns a GridSDF (a drop-in union). See holographic_sdfbake."""
        from holographic_sdfbake import GridSDF
        return GridSDF.bake(sdf, lo, hi, res)

    def dispatch_methods(self, x, tags, ops, default=None):
        """COMPOSABILITY OF CALCULATION METHODS -- apply a DIFFERENT operator to different elements of one structure,
        chosen per-element by `tags`, and recombine. The same "part fluid, part static, by a field" idea the engine
        uses for DATA, now applied to WHICH COMPUTATION runs where: trace the first hit, then per bounce dispatch to
        collapse (a PRT dot product) on diffuse, trace on a mirror, a glossy bundle on a rough patch -- switching on the
        fly. It is the per-element generalization of the whole-signal method selection `denoise(method='auto')` and
        `decompose_signal` already do. `ops` is {label: fn(sub_x)->sub_y}. See holographic_dispatch."""
        from holographic_dispatch import dispatch_field
        return dispatch_field(x, tags, ops, default=default)

    def render_dispatch(self, sdf, camera, width, height, methods, colors, light, order=3, n=400):
        """RENDER by dispatching each hit to its best method and get a RELIGHT handle -- the pipeline form of "collapse
        on diffuse, trace on a mirror, switch on the fly". `methods` maps object id -> 'collapse' (PRT dot product) or
        'trace' (a mirror bounce whose diffuse hits themselves collapse). Returns (frame, relight, info): `relight(new
        light)` re-shades the collapsed parts for free, and `info` reports the dispatch counts. This is how PRT and the
        method dispatch are USED in a real render, not just measured. See holographic_dispatch.render_dispatch."""
        from holographic_dispatch import render_dispatch
        return render_dispatch(sdf, camera, width, height, methods, colors, light, order=order, n=n)

    def bake_scene(self, sdf, camera, width, height, methods, colors, order=3, n=400):
        """PRECOMPUTE / BAKE a scene BEFORE any render, so the first render is already a relight, not a cold trace. Call
        this once at scene-load: it traces primary visibility, dispatches each hit to its method, and precomputes the
        PRT transfer for every diffuse hit and every diffuse surface behind a mirror bounce. Returns a BakedScene to hand
        to `render_baked(scene, light)` -- interactive relighting is then a dot product from frame one. See
        holographic_dispatch.bake_scene."""
        from holographic_dispatch import bake_scene
        return bake_scene(sdf, camera, width, height, methods, colors, order=order, n=n)

    def render_baked(self, scene, light):
        """Relight a BakedScene (from bake_scene) -- shade every pixel from its precomputed transfer, no tracing. Every
        frame, including the first, is this cheap dot-product relight. Returns a (H,W,3) frame. See
        holographic_dispatch.render_baked."""
        from holographic_dispatch import render_baked
        return render_baked(scene, light)

    def render_adaptive(self, objects, camera, width=256, height=256, frames=1, relight=False, light=None,
                        sun="bright", sky="clear", post=None, **kw):
        """ONE render call that ADAPTS -- it looks at the scene and the workload and picks the methods itself instead of
        you choosing bake/relax/collapse/trace by hand. Grounded in the MEASURED break-evens: it bakes the SDF only when
        primitives or frames make it pay, keeps the exact active-only marcher (over-relaxation stays a manual opt-in),
        and -- when relighting -- COLLAPSES diffuse surfaces (PRT, free relight) while TRACING reflective ones, deriving
        the per-surface method from each material's reflectivity. Returns (frame, relight, plan); `plan['reasons']`
        explains every choice so the automation stays legible. The separate options (render_scene bake=/relax=,
        render_dispatch, radiance_transfer) remain for manual control. See holographic_adaptive.render_adaptive."""
        from holographic_adaptive import render_adaptive
        return render_adaptive(objects, camera, width=width, height=height, frames=frames, relight=relight,
                               light=light, sun=sun, sky=sky, post=post, **kw)

    def plan_render(self, objects, frames=1, relight=False):
        """The DECISION LAYER of the adaptive pipeline, on its own: given a scene and workload, return the plan (bake
        resolution, relax factor, path, per-surface methods) with a human-readable reason for each choice -- so you can
        see what render_adaptive WOULD do, and why, without rendering. See holographic_adaptive.plan_render."""
        from holographic_adaptive import plan_render
        return plan_render(objects, frames=frames, relight=relight)

    def distribute_compute(self, buckets, worker, reduce="sum", cache=None):
        """DISTRIBUTED COMPUTATION the holostuff way -- decompose a job into buckets, hand every bucket the same shared
        read-only `cache` (the "GI cache on the main node"), run `worker(bucket, cache)` on each, and reassemble with a
        COMMUTATIVE monoid so the result is independent of bucket order (=> the buckets could run on separate machines /
        VMs with no stitch pass). `reduce` selects the reassembly operator that matches the computation: 'sum' (linear
        superposition -- forces, fields, radiance, densities), 'min' (SDF union), 'max' (occupancy), 'bundle' (VSA
        scene/memory), or a callable. Runs in-process here (no speedup claimed); it builds the STRUCTURE that makes
        distribution correct. Returns (result, info). See holographic_distribute."""
        from holographic_scalehome import Scale                      # the Scale home (consolidation H3)
        return Scale.map_reduce(buckets, worker, reduce=reduce, cache=cache)

    def partition_domain(self, n, k, costs=None):
        """Decompose a domain of n items into k disjoint buckets for distribution. With `costs` (a per-item work
        estimate) it LOAD-BALANCES -- heaviest-first onto the lightest bucket -- so the slowest bucket, which bounds a
        farm's wall-time, is minimised (adaptive bucket sizing). Returns a list of index arrays. See
        holographic_distribute.partition / adaptive_partition."""
        from holographic_scalehome import Scale                      # the Scale home (consolidation H3)
        return Scale.partition(n, k, costs=costs)

    def partition_grid(self, shape, blocks):
        """Decompose a 2D image/field (shape=(H,W)) into TILES or a 3D volume/grid (shape=(X,Y,Z)) into BRICKS -- the
        render-farm bucket layout generalised to 2D and 3D. `blocks` is an int or a per-axis tuple. Returns a list of
        slice-tuples covering the domain disjointly; each is an independent bucket (a separate VM/node), and a 3D brick
        with no surface can be skipped (sparse volumes). Also the cache-blocking layout -- a tile/brick sized to a
        working budget streams through a fast cache level. See holographic_distribute.partition_2d / partition_3d."""
        from holographic_scalehome import Scale                      # the Scale home (consolidation H3)
        return Scale.tiles(shape, blocks)

    def distribute_bricks(self, out_shape, regions, worker, cache=None, fill=0.0, skip=None):
        """Run `worker(region, cache)` on each tile/brick and PLACE its result at that region -- disjoint, so
        order-independent and seamless (the shared read-only cache makes borders agree). `skip(region)->bool` drops
        EMPTY bricks (sparse volumes: most of a volume is empty space -- the real speed win of bricking 3D, beyond
        parallelism). Returns (out, info) with the ran/skipped counts. See holographic_distribute.distribute_bricks."""
        from holographic_scalehome import Scale                      # the Scale home (consolidation H3)
        return Scale.bricks(out_shape, regions, worker, cache=cache, fill=fill, skip=skip)

    def surface_material(self, name=None, color=(0.7, 0.7, 0.7), **channels):
        """The FIRST-CLASS render material: every channel (color, roughness, reflect, emission, opacity) is a Param
        SOCKET -- a constant, a `Param`, a callable field (e.g. a `pattern_field`), or a map array -- resolved PER HIT
        by `render_surface`. With `name`, channels start from the ONE canonical MATERIAL_RENDER table (no more
        per-demo copies) and your overrides apply on top. This is the object that ties param -> pattern -> material ->
        render together. See holographic_surface.SurfaceMaterial."""
        from holographic_surface import SurfaceMaterial
        m = SurfaceMaterial.from_name(name, color=color) if name is not None else SurfaceMaterial(color=color)
        for k, v in channels.items():
            setattr(m, k, v)
        return m

    def realize_recipe_fused(self, recipe, spectrum_cache=None):
        """Fill 4 integration at Layer 4: realize a StructureRecipe's outputs through the SCHEDULER, fusing its
        straight-line bind/bundle/permute runs so a long structure build does fewer FFTs than the op-by-op
        `recipe.build()`. Returns (outputs, stats). THROUGHPUT path -- fusion is ~1e-15, so this does NOT keep the
        recipe's bit-exact-replay guarantee; `recipe.build()`/`realize` stay the exact default. See
        holographic_schedule.run_recipe."""
        from holographic_schedule import run_recipe
        return run_recipe(recipe, fused=True, spectrum_cache=spectrum_cache)

    def spectrum_cache(self, max_items=4096):
        """Fill 1 (residency): a content-addressed cache of atom -> rfft(atom), so binds/unbinds against KNOWN
        atoms skip the forward transform. Bit-identical to recompute. Pass it to `fuse`/`run_scheduled` to make
        their leaf transforms free for known atoms. See holographic_residency.SpectrumCache."""
        from holographic_memoryhome import Memory                # the Memory home (consolidation H6)
        return Memory.spectrum_cache(max_items=max_items)

    def fuse_record(self, keys, values, spectrum_cache=None):
        """Fill 2 (spectral fusion): build a role/filler record -- bundle([bind(k_i, v_i)]) -- in ONE fused FFT
        pass (leaves+1 transforms instead of ~3*len), equal to the op-by-op result to ~1e-15. THROUGHPUT path:
        tie-sensitive encoders (the maze-rescue path) must NOT use this. See holographic_fuse."""
        from holographic_computehome import Compute                  # the Compute home (consolidation H7)
        return Compute.fuse_record(keys, values, spectrum_cache=spectrum_cache)

    def fuse_expression(self, expr, spectrum_cache=None):
        """Fill 2: evaluate a five-op (bind/unbind/bundle/permute) expression tree in the FFT domain -- one
        transform per leaf, one out. Build `expr` with holographic_fuse.{leaf,fbind,funbind,fbundle,fpermute}."""
        from holographic_computehome import Compute                  # the Compute home (consolidation H7)
        return Compute.fuse(expr, spectrum_cache=spectrum_cache)

    def superpose_batch(self, keys, items, gated=True):
        """Fill 3 (auto-superposition + spill): pack N independent keyed items into the FEWEST superposed vectors
        that keep each bucket under the capacity dial -- one vector if it fits, else SPILL across buckets (not
        abstain). Returns (packed_vectors, buckets). `apply_in_superposition` does one op on each bundle at once.
        See holographic_superschedule."""
        from holographic_superschedule import superpose_batch
        return superpose_batch(keys, items, gated=gated)

    def apply_in_superposition(self, keys, items, op, gated=True):
        """Fill 3: the latency-hiding move -- hold items in superposition and apply ONE bind by `op` to each
        bucket's whole bundle at once (transforming every item in flight), then recover. Spills past the dial."""
        from holographic_superschedule import apply_in_superposition
        return apply_in_superposition(keys, items, op, gated=gated)

    def schedule_program(self, ops, min_run=2, spectrum_cache=None, sequential=False):
        """Fill 4 (the scheduler capstone): run a VSA program DAG (built with holographic_schedule.{leaf,op_bind,
        op_unbind,op_bundle,op_permute,op_cleanup}) with the linear runs FUSED, tie-sensitive runs kept op-by-op
        and bit-exact, and Python crossings only at the cleanups. Returns (values, stats) where stats reports the
        FFT count, kernel-op calls, and crossings. `sequential=True` runs the op-by-op baseline for comparison.
        See holographic_schedule."""
        from holographic_schedule import run_scheduled, run_sequential
        if sequential:
            return run_sequential(ops)
        return run_scheduled(ops, min_run=min_run, spectrum_cache=spectrum_cache)

    def measure_area(self, mesh):
        """Modeling-app backlog (measurement + units): total surface area of a mesh, as a dimensioned [m^2]
        Quantity measured from the geometry. See holographic_metrology.surface_area."""
        from holographic_metrology import surface_area
        return surface_area(mesh)

    def measure_volume(self, mesh):
        """Modeling-app backlog: enclosed volume of a CLOSED mesh (divergence theorem), a [m^3] Quantity. See
        holographic_metrology.volume."""
        from holographic_metrology import volume
        return volume(mesh)

    def measure_bbox(self, mesh):
        """Modeling-app backlog: the axis-aligned bounding box of a mesh (extents + diagonal as [m] Quantities).
        See holographic_metrology.bounding_box."""
        from holographic_metrology import bounding_box
        return bounding_box(mesh)

    def measure_distance(self, p, q):
        """Modeling-app backlog: the length between two points, as a dimensioned [m] Quantity (convert with
        .to('ft') etc). See holographic_metrology.distance."""
        from holographic_metrology import distance
        return distance(p, q)

    def guided_upsample(self, low_color, guide_normal, guide_albedo=None, guide_depth=None, levels=4, sigma_color=2.0):
        """Inverse-rendering ST3: guided (joint-bilateral) super-resolution -- render colour SMALL, then upscale it
        steered by the full-res G-buffer (normal/depth/albedo, which render_channels exposes), so colour edges snap to
        the geometry the cheap render already knows at full res. Reuses the shipped SVGF bilateral. Invents plausible,
        not true, detail (below learned SR). See holographic_superres.guided_upsample."""
        from holographic_superres import guided_upsample
        return guided_upsample(low_color, guide_normal, guide_albedo=guide_albedo, guide_depth=guide_depth,
                               levels=levels, sigma_color=sigma_color)

    def synthesize_texture(self, sample, out_h, out_w, psize=24, overlap=6, seed=0, seam="mincut"):
        """Inverse-rendering ST2: grow a larger texture from a small sample by Image Quilting -- lay overlapping
        patches chosen by a patch search (HoloForest recall_k), stitched along min-cut seams. For material synthesis
        and feeding IR1 auto-bump with tileable maps. Patch-copying (can repeat/seam), best for texture/material, not
        free-form restyle. See holographic_texturesynth.synthesize_texture."""
        from holographic_texturesynth import synthesize_texture
        return synthesize_texture(sample, out_h, out_w, psize=psize, overlap=overlap, seed=seed, seam=seam)

    def complete_object(self, archive, front, match_floor=0.85):
        """Inverse-rendering IR11: given a partial FRONT view of an object and an ObjectArchive of complete objects,
        recall the nearest stored WHOLE object (including the unobserved back) by its view fingerprint, or ABSTAIN
        when nothing in the library matches. Retrieval, not hallucination -- the archive's 'recover the whole from a
        partial measurement' move, one dimension up. See holographic_objectarchive.ObjectArchive."""
        return archive.complete_from_front(front, match_floor=match_floor)

    def render_checkerboard(self, sdf, camera, width, height, parity=0, **kw):
        """Inverse-rendering IR13: checkerboard/sparse render -- shade only ~50% of the pixels (a 2x2 pattern) and
        reconstruct the rest as masked recovery (the unshaded pixels are 'damage'; their four cross-neighbours are
        all shaded). Roughly halves the shading cost for a near-full-resolution result. Flip `parity` per frame to
        fill the other half over time. Returns (image, mask). See holographic_checkerboard.render_checkerboard."""
        from holographic_checkerboard import render_checkerboard
        return render_checkerboard(sdf, camera, width, height, parity=parity, **kw)

    def upscale(self, image, scale=2.0, sharpness=0.4):
        """Inverse-rendering IR12: FSR1-style spatial upscale -- EASU (edge-adaptive Lanczos with anti-ringing) then
        RCAS (the shipped noise-aware sharpen). Take a low-res render up to display resolution edge-adaptively; beats
        plain bilinear on PSNR and edge sharpness. Reconstructs, cannot invent absent detail. See holographic_fsr."""
        from holographic_fsr import fsr_upscale
        return fsr_upscale(image, scale=scale, sharpness=sharpness)

    def render_channels(self, sdf, camera, want=None, width=32, height=32, objects=None, **render_kw):
        """Inverse-rendering IR14: render selectable, separate AOV channels (depth/normal/position/mask G-buffer,
        per-object Cryptomatte mattes), each with its own alpha, for compositing/science/debug. A channel is an
        UNBIND; the scene is a bundle at every level. Default (no selection) = the beauty pass, bit-identical to
        render_sdf. Lighting passes need trace-time accumulation (not in v1). See holographic_renderchannels."""
        from holographic_renderchannels import render_channels
        return render_channels(sdf, camera, want=want, width=width, height=height, objects=objects, **render_kw)

    def scene_hypothesis(self, image, k=4):
        """Inverse-rendering IR3: an archetype-level scene READING of an image -- dominant palette, the horizon row
        (sky/ground split), and a coarse sun direction. The perception seed that warm-starts the IR4 loop. A gist,
        not a segmentation (abstain-worthy outside its vocabulary). See holographic_perception.scene_hypothesis."""
        from holographic_perception import scene_hypothesis
        return scene_hypothesis(image, k=k)

    def estimate_light_direction(self, image, power=2.0):
        """Inverse-rendering IR3: a COARSE sun-direction estimate (azimuth, elevation) from an image's brightest
        region -- a warm-start cue for IR4 to refine, not a measurement. See holographic_perception."""
        from holographic_perception import estimate_light_direction
        return estimate_light_direction(image, power=power)

    def recover_scene(self, sdf, target_img, init_params, accept_threshold=None, **kw):
        """Inverse-rendering IR4 (the headline): analysis-by-synthesis. Given a TARGET image, gradient-free-search
        the camera + sun-direction parameters whose render best matches it (perceptual distance, not MSE), from a
        warm-start guess, with an optional conformal accept/abstain gate. The measurable milestone is self-recovery:
        render a known scene, recover its camera + light within tolerance. See holographic_inverserender."""
        from holographic_inverserender import recover_scene
        return recover_scene(sdf, target_img, init_params, accept_threshold=accept_threshold, **kw)

    def compare_images(self, x, y, w_struct=0.5, w_color=0.3, w_edge=0.2):
        """Inverse-rendering IR4: a PERCEPTUAL render-vs-target similarity in [0,1] (1 = identical) -- multi-scale
        SSIM + colour-histogram agreement + edge alignment. Shift/lighting-tolerant, unlike raw pixel MSE. This is
        the compare step of the analysis-by-synthesis loop. See holographic_imagecompare.perceptual_similarity."""
        from holographic_imagecompare import perceptual_similarity
        return perceptual_similarity(x, y, w_struct=w_struct, w_color=w_color, w_edge=w_edge)

    def image_distance(self, x, y, **kw):
        """Inverse-rendering IR4: 1 - compare_images -- the objective the analysis-by-synthesis loop MINIMIZES
        (0 = a perfect match). See holographic_imagecompare.perceptual_distance."""
        from holographic_imagecompare import perceptual_distance
        return perceptual_distance(x, y, **kw)

    def recolor_image(self, image, reference, mode="covariance", strength=1.0):
        """2D EDIT -- grade an image toward another image's COLOUR statistics (a colour-transfer / recolour). `mode`
        'meanstd' matches per-channel mean+std, 'covariance' does full whitening/colouring; `strength` blends
        0->original .. 1->full transfer. Returns an image the same shape as the input. See holographic_colortransfer."""
        from holographic_colortransfer import color_transfer
        return color_transfer(image, reference, mode=mode, strength=strength)

    def blend_images(self, image_a, image_b, steps=21):
        """2D GENERATE -- a crossfade/morph sequence between two images (the midpoint is the 0.5*a+0.5*b double
        exposure). Returns `steps` frames from a to b. See holographic_generate.crossfade_images."""
        from holographic_generate import crossfade_images
        return crossfade_images(image_a, image_b, steps=steps)

    def auto_displace(self, mesh, rgb, amount=0.1, sigma=4.0, min_confidence=0.02):
        """Inverse-rendering IR5: promote an auto-bump height (IR1) from a shading bump to REAL geometry -- move a
        mesh's vertices along their normals by the derived height, but ONLY if the bump-confidence clears a
        (stricter) geometry threshold; otherwise ABSTAIN and return the mesh unchanged. Returns (mesh, info). See
        holographic_autodisplace.auto_displace."""
        from holographic_autodisplace import auto_displace
        return auto_displace(mesh, rgb, amount=amount, sigma=sigma, min_confidence=min_confidence)

    def color_transfer(self, img, reference, mode="covariance", strength=1.0, clip=True):
        """Inverse-rendering ST1: grade an image toward a REFERENCE image's colour statistics (Reinhard 2001) --
        the 'match the sunset's mood' knob. mode='meanstd' (per-channel) or 'covariance' (full mean+covariance,
        whitening/colouring). Moves colour, not content. See holographic_colortransfer.color_transfer."""
        from holographic_colortransfer import color_transfer
        return color_transfer(img, reference, mode=mode, strength=strength, clip=clip)

    def integrate_normals(self, nmap):
        """Inverse-rendering IR7: integrate a tangent-space normal map into a single-valued, CONSISTENT height
        field by FFT (Frankot-Chellappa) -- the inverse of auto_bump's normal-from-height. Drift-free and
        seamlessly TILEABLE (periodic boundary). See holographic_surfaceint.height_from_normals."""
        from holographic_surfaceint import height_from_normals
        return height_from_normals(nmap)

    def auto_bump(self, rgb, strength=2.0, sigma=4.0, abstain_below=0.005):
        """Inverse-rendering IR1: derive a plausible tangent-space normal map (and height) from an albedo image
        alone -- 'auto bump' when no bump/normal map is supplied. Grayscale -> high-pass -> normal, with an honest
        confidence gate that ABSTAINS to flat when there is too little fine detail. Returns a dict with the normal
        map, height, confidence, and whether it abstained. See holographic_autobump.auto_bump."""
        from holographic_autobump import auto_bump
        return auto_bump(rgb, strength=strength, sigma=sigma, abstain_below=abstain_below)

    def sampler(self, shape, target, mode="point", radius=1.0, falloff="smooth", weight=None):
        """Modeling-app backlog (capstone): a placeable read-probe -- the read-dual of a FieldEffect. Reads a
        field/material at a point, surface patch, or volume region with a falloff weighting, and handles overlap
        with a labeled bundle. See holographic_sampler.Sampler."""
        from holographic_sampler import Sampler
        return Sampler(shape, target, mode=mode, radius=radius, falloff=falloff, weight=weight)

    def place_sampler(self, scene, sampler, transform=None, name="Sampler"):
        """Modeling-app backlog (capstone): drop a Sampler into the Scene as an object (handle + transform), so it
        is placed/moved/animated like anything else. See holographic_sampler.place_sampler."""
        from holographic_sampler import place_sampler
        return place_sampler(scene, sampler, transform=transform, name=name)

    def resolve_override(self, scene, handle, prop, defaults=None, default=None):
        """Modeling-app feature layer: resolve a render property for an object -- its own override, else its
        material's, else the scene defaults, else a bare default (a bound role with fallback). See
        holographic_overrides.resolve."""
        from holographic_overrides import resolve
        return resolve(scene, handle, prop, defaults=defaults, default=default)

    def set_override(self, scene, handle, prop, value):
        """Modeling-app feature layer: bind a render override on an object (undoable). See
        holographic_overrides.set_override."""
        from holographic_overrides import set_override
        set_override(scene, handle, prop, value)

    def snapper(self, grid=None, vertices=None, tol=0.25):
        """Modeling-app feature layer: a Snapper that snaps a dragged point to the nearest grid node or vertex
        within a tolerance (snapping = cleanup). See holographic_snap.Snapper."""
        from holographic_snap import Snapper
        return Snapper(grid=grid, vertices=vertices, tol=tol)

    def group_objects(self, scene, handles, name="Group"):
        """Modeling-app feature layer: group objects under a null parent (grouping = a bundle); one undo step.
        See holographic_grouping.group_objects."""
        from holographic_grouping import group_objects
        return group_objects(scene, handles, name=name)

    def instance(self, scene, source, transform=None, name=None):
        """Modeling-app feature layer: create an instance sharing a source's geometry with its own transform
        (instancing = a bind); editing the source updates all instances. See holographic_grouping.instance."""
        from holographic_grouping import instance
        return instance(scene, source, transform=transform, name=name)

    def camera_controller(self, eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
        """Modeling-app feature layer: a viewport camera controller -- orbit/pan/dolly/zoom/frame around a target.
        See holographic_camera.CameraController."""
        from holographic_camera import CameraController
        return CameraController(eye=eye, target=target, up=up)

    def selection(self, scene):
        """Modeling-app feature layer: a Selection helper bound to a Scene -- query objects into a set of handles,
        save named sets, do set algebra (union/intersect/minus/invert), and push the current selection. See
        holographic_scene_query.Selection."""
        from holographic_scene_query import Selection
        return Selection(scene)

    def select_objects(self, scene, **predicates):
        """Modeling-app feature layer: select object handles from a Scene by exact predicates (name/material/tag/
        substring/where). For semantic 'select the metal-ish parts' with confidence, use
        holographic_scene_query.select_fuzzy. See holographic_scene_query.select."""
        from holographic_scene_query import select
        return select(scene, **predicates)

    def look_at(self, eye, target, up=(0.0, 1.0, 0.0)):
        """Modeling-app backlog (item G): an OpenGL view matrix for a camera at `eye` looking at `target`. See
        holographic_transform.look_at."""
        from holographic_transform import look_at
        return look_at(eye, target, up)

    def decompose_transform(self, M):
        """Modeling-app backlog (item G): split a 4x4 transform into (translate, rotation quaternion, scale) --
        what a move/rotate/scale gizmo reads off a matrix. See holographic_transform.decompose."""
        from holographic_transform import decompose
        return decompose(M)

    def cancel_token(self):
        """Modeling-app backlog (item F): a cooperative CancelToken to pass as should_stop= to a long render/sim,
        so it can be stopped mid-run and return a partial result. See holographic_cancel.CancelToken."""
        from holographic_cancel import CancelToken
        return CancelToken()

    def modifier_stack(self, base):
        """Modeling-app backlog (item C): a per-object MODIFIER STACK + dependency graph over any payload (mesh /
        field / vector) -- an ordered, non-destructive op chain with stable handles that re-evaluates O(change)
        (only downstream of a changed parameter). See holographic_modifier.ModifierStack."""
        from holographic_modifier import ModifierStack
        return ModifierStack(base)

    def new_scene(self, dim=None, seed=0):
        """Modeling-app backlog (item 0): a fresh canonical Scene document -- the single source of truth a modeling
        app is built around (a table of object records + hierarchy, owning selection and undo history, firing
        change events, with STABLE identity handles that survive edits). See holographic_scene_doc.Scene."""
        from holographic_scene_doc import Scene
        return Scene(dim=dim if dim is not None else self.dim, seed=seed)

    def scatter_to_grid(self, points, values, shape, kernel="bilinear", periodic=False):
        """The shared kernel SCATTER = a BUNDLE: deposit each point's value onto a grid through a kernel (bilinear
        or B-spline) -- the superposition that MPM's P2G, a fluid deposit, and a splat all are. `points` (N,D) in
        grid-cell units, `values` (N,) or (N,C). See holographic_transfer.scatter."""
        from holographic_transfer import scatter
        return scatter(points, values, shape, kernel=kernel, periodic=periodic)

    def gather_from_grid(self, field, points, kernel="bilinear", periodic=False):
        """The shared kernel GATHER = the READOUT: read a grid back at each point through the same kernel -- the
        adjoint of scatter, and what MPM's G2P, field sampling, and a texture lookup all are. See
        holographic_transfer.gather."""
        from holographic_transfer import gather
        return gather(field, points, kernel=kernel, periodic=periodic)

    def snow_mpm(self, grid=48, dx=1.0, E=140.0, nu=0.2, gravity=9.81, seed=0):
        """Physics backlog (#8B, rung 4): a SNOW solver by the Material Point Method (Stomakhin 2013). Seed it
        (.seed_block) and .run(). Thinking holographically: its P2G scatter IS bundling (the grid is a
        superposition of kernel-weighted particle contributions -- verified equal to a bundle of splats) and G2P is
        the readout; only the elasto-plastic grid update is grid-native. See holographic_mpm.MPMSnow."""
        from holographic_mpm import MPMSnow
        return MPMSnow(grid=grid, dx=dx, E=E, nu=nu, gravity=gravity, seed=seed)

    def simulate_snow(self, cx=24, cy=12, w=10, h=8, n=400, grid=48, gravity=9.81, dt=2e-3, steps=600, seed=0):
        """Physics backlog (#8B): seed a snow block and run it -- it falls, piles, and compresses plastically.
        Returns the settled MPMSnow. See holographic_mpm.MPMSnow."""
        from holographic_mpm import MPMSnow
        snow = MPMSnow(grid=grid, gravity=gravity, seed=seed).seed_block(cx=cx, cy=cy, w=w, h=h, n=n)
        return snow.run(dt=dt, steps=steps)

    def free_surface(self, g=9.81, ground=0.0, damping=0.3):
        """Physics backlog (#8, rung 4): the OVERTURNING free-surface solver -- particles that can fold the water
        surface over itself (a breaking wave), which a height field fundamentally cannot. Seed it and call
        .advance(). See holographic_freesurface.FreeSurface."""
        from holographic_freesurface import FreeSurface
        return FreeSurface(g=g, ground=ground, damping=damping)

    def break_wave(self, length=10.0, n=40, crest_speed=8.0, phase_speed=3.0, height=4.0, dt=0.05, steps=20):
        """Physics backlog (#8): set up and run a PLUNGING BREAKER -- a crest whose tip outruns the wave, throwing
        it forward until the surface folds (overturns) into a multi-valued sheet. Returns the FreeSurface mid-plunge
        (query .is_overturning() / .is_multivalued()). See holographic_freesurface.seed_breaking_crest."""
        from holographic_freesurface import FreeSurface, seed_breaking_crest
        fs = FreeSurface()
        seed_breaking_crest(fs, length=length, n=n, crest_speed=crest_speed, phase_speed=phase_speed, height=height)
        fs.advance(dt, steps=steps)
        return fs

    def grow_ice(self, shape=(81, 81), eta=1.0, steps=200, seed=0):
        """Physics backlog (#7): grow an ICE / frost dendrite by diffusion-limited branching (the dielectric-
        breakdown model) -- a cluster racing into the steepest gradient of a Laplace field, branching. Returns a
        DielectricBreakdown; read its .cluster mask. See holographic_dendrite.ice_dendrite."""
        from holographic_dendrite import ice_dendrite
        return ice_dendrite(shape=shape, eta=eta, steps=steps, seed=seed)

    def grow_lightning(self, shape=(81, 81), eta=3.0, steps=120, seed=0):
        """Physics backlog (#7): grow a LIGHTNING bolt -- the SAME diffusion-limited branching engine as the ice
        dendrite (N11: build once, get frost and bolts), only the seed (the cloud) and the source (the ground it
        reaches toward) differ. See holographic_dendrite.lightning."""
        from holographic_dendrite import lightning
        return lightning(shape=shape, eta=eta, steps=steps, seed=seed)

    def dielectric_breakdown(self, shape, eta=1.0, seed=0):
        """Physics backlog (#7): the raw diffusion-limited branching engine (Niemeyer-Pietronero-Wiesmann). Seed
        it (seed_point / seed_line), set a source boundary (set_source_border), and grow(). eta tunes the shape
        (bushy -> fractal -> stringy). See holographic_dendrite.DielectricBreakdown."""
        from holographic_dendrite import DielectricBreakdown
        return DielectricBreakdown(shape, eta=eta, seed=seed)

    def lorentz_force(self, q, E, v, B):
        """Physics backlog (#6): the Lorentz force F = q(E + v x B) on a charge q moving at v through fields E, B.
        See holographic_em.lorentz_force."""
        from holographic_em import lorentz_force
        return lorentz_force(q, E, v, B)

    def push_charge(self, pos, vel, q, m, E, B, dt, steps):
        """Physics backlog (#6): integrate a charged particle through uniform fields E, B with the Boris pusher
        (energy-conserving) -- a cyclotron orbit in a magnetic field, an E-cross-B drift in crossed fields.
        Returns (trajectory, final_velocity). See holographic_em.push_particle."""
        from holographic_em import push_particle
        return push_particle(pos, vel, q, m, E, B, dt, steps)

    def maxwell_field(self, n, dx=1.0, eps=1.0, mu=1.0):
        """Physics backlog (#6): a 1-D coupled Maxwell field (Yee/FDTD) -- Ez and Hy feed each other so a pulse
        propagates at c = 1/sqrt(mu*eps). Set .Ez, call .step(). This is the genuine E<->B coupling the spectral
        backbone's single-component em_field doesn't have. See holographic_em.Maxwell1D."""
        from holographic_em import Maxwell1D
        return Maxwell1D(n, dx=dx, eps=eps, mu=mu)

    def plan_waves(self, height, depth=None, obstacles=None, dx=1.0, tile=8):
        """Physics backlog (#5, the AdaptiveSolver): the DECISION LAYER for the ocean stack -- per tile, pick the
        wave method (fft_ocean / wave_packets / shallow_water / free_surface) from the local regime and say WHY,
        exactly as plan_render picks bake/analytic/trace. No solving; the plan is inspectable before running, and
        deterministic (breaking > shallow > obstacle > open). Pair with solve_waves. See
        holographic_waveadaptive.plan_waves / plan_cost."""
        from holographic_waveadaptive import plan_waves
        return plan_waves(height, depth=depth, obstacles=obstacles, dx=dx, tile=tile)

    def solve_waves(self, plan, field, dt=1.0, methods=None, halo=2):
        """Physics backlog (#5): EXECUTE a wave plan -- run each tile's chosen method on the shared surface field
        and blend the tile borders (overlap-add, no seam). The dear grid solver runs only where the plan marked a
        breaking tile; the cheap spectral path runs everywhere else. See holographic_waveadaptive.solve_waves."""
        from holographic_waveadaptive import solve_waves
        return solve_waves(plan, field, dt=dt, methods=methods, halo=halo)

    def wave_packets(self, size=64.0, g=9.81, envelope=6.0, seed=0):
        """Physics backlog (N8): a water surface as localized WAVE PACKETS -- each a Gaussian-enveloped wave train
        that lives at a place, so unlike the global FFT ocean it can REFLECT off walls, SHOAL over depth changes,
        and diffract. A packet is a role-bound record and the surface is a bundle (content-addressable). Add
        packets with .add_packet, advance with .advance, read the surface with .render. See
        holographic_wavepacket.WavePacketField."""
        from holographic_wavepacket import WavePacketField
        return WavePacketField(size=size, g=g, envelope=envelope, seed=seed)

    def spectral_pde(self, field, velocity=None, order="parabolic", rate=None, omega=None, dx=1.0):
        """Physics backbone (Part 3 #1): a linear field advanced in FOURIER space by a per-frequency transfer --
        (named spectral_pde to avoid colliding with the fractal-volume spectral_field synthesizer above.)
        'advancing time is one bind, any t in closed form.' order='parabolic' (diffusion, decay rate(|k|)) or
        'hyperbolic' (waves, oscillation omega(|k|), carries velocity). Superposition is add_source (bundle); a
        calibrated trigger_mask fires where a potential crosses threshold. See holographic_spectralfield."""
        from holographic_spectralfield import SpectralField
        return SpectralField(field, velocity=velocity, order=order, rate=rate, omega=omega, dx=dx)

    def spectral_diffusion(self, field, D, dx=1.0):
        """A diffusion/heat/gas field as a SpectralField: rate(|k|) = -D|k|^2, closed-form any t. Beats the grid
        diffuse_heat baseline (machine-precision exact in one eval vs accumulated step error). See
        holographic_spectralfield.diffusion_field."""
        from holographic_spectralfield import diffusion_field
        return diffusion_field(field, D, dx=dx)

    def spectral_wave(self, field, velocity=None, c=1.0, dx=1.0):
        """A wave/acoustic/EM(vacuum) field as a SpectralField: omega(|k|) = c|k|, a pulse propagates at speed c;
        closed-form any t. See holographic_spectralfield.wave_field."""
        from holographic_spectralfield import wave_field
        return wave_field(field, velocity=velocity, c=c, dx=dx)

    def spectral_ocean(self, height, velocity=None, g=9.81, dx=1.0):
        """A deep-water ocean surface as a SpectralField: the dispersive omega(|k|) = sqrt(g|k|) (long swells
        outrun short chop). Seed the height with phillips_spectrum for a real sea state. See
        holographic_spectralfield.ocean_field / phillips_spectrum."""
        from holographic_spectralfield import ocean_field
        return ocean_field(height, velocity=velocity, g=g, dx=dx)

    def electrostatic_potential(self, source, dx=1.0, eps0=1.0):
        """The electrostatic potential of a charge distribution in ONE spectral step: phi_hat = source_hat /
        (eps0|k|^2) -- the closed-form steady (t->inf) limit of the diffusion field (Thesis A: electrostatics is
        the limit). See holographic_spectralfield.poisson_solve."""
        from holographic_spectralfield import poisson_solve
        return poisson_solve(source, dx=dx, eps0=eps0)

    def photo_to_3d(self, depth, colour, fx, fy, cx, cy, confidence_floor=0.3):
        """Forecasting sweep (sec.5, depth delegation) / photo-to-3D: lift a depth map + image into per-pixel 3D
        Gaussians, but ONLY where the reconstruction is observed -- unproject the CONFIDENT front-facing,
        continuous pixels and ABSTAIN on invalid depth, occlusion edges (where unprojecting stretches fake
        geometry), grazing surfaces, and -- loudest -- the unobserved BACK of every object. A single view
        reconstructs the visible front, not a watertight guess. Returns positions/colours/radii/confidences + an
        abstain mask + coverage. See holographic_photo3d."""
        from holographic_photo3d import photo_to_gaussians
        return photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=confidence_floor)

    def unproject_depth(self, depth, fx, fy, cx, cy):
        """Turn a depth map into 3D points in camera space (the pinhole unprojection). Returns (H, W, 3). See
        holographic_photo3d.unproject."""
        from holographic_photo3d import unproject
        return unproject(depth, fx, fy, cx, cy)

    def make_scene(self, objects, dim=2048, seed=0):
        """Query Interface (Phase 4): build a SCENE from a list of nested objects -- each object is encoded as a
        nested VSA record (a nested field is bind(role, sub_record)). Pair with `query_scene`. See
        holographic_graphql.Scene."""
        from holographic_graphql import Scene
        return Scene(objects, dim=dim, seed=seed)

    def query_scene(self, graphql, scene):
        """Query Interface (Phase 4): run a GraphQL query over a scene -- 'ask for exactly the nested fields you
        want,' which maps onto unbinding exactly those roles. Filters objects by a `where` arg and returns only
        the requested (possibly nested) fields per object. GraphQL is the natural fit for the nested scene where
        SQL fits the flat tables. See holographic_graphql.resolve."""
        from holographic_graphql import resolve
        return resolve(scene, graphql)

    def database(self, dim=2048, seed=0):
        """Query Interface (Phases 9-13): a DATABASE you OWN -- user namespaces over a read-only 'system'
        namespace. The mind's capability registry is published as `system.actions` out of the box, so you can
        SELECT from it, CREATE your own databases/tables beside it, INSERT rows, bookmark system rows
        (`insert_select`), define live views (`create_view`), and persist by replay (`to_state`/`from_state`) --
        but never write to system.* (the wall). See holographic_query.Database."""
        from holographic_query import Database, capability_registry
        db = Database()
        db.register_system("actions", capability_registry(self, dim=dim, seed=seed))
        return db

    def db_query(self, sql, db):
        """Run a SQL statement over a Database: CREATE DATABASE, CREATE TABLE ns.t (cols), INSERT INTO ns.t (cols)
        VALUES (...), or SELECT ... FROM ns.table. Writes to system.* are refused by the wall. Bookmarks and views
        use the object API (db.insert_select / db.create_view). See holographic_query.run_db_sql."""
        from holographic_query import run_db_sql
        return run_db_sql(sql, db)

    def capabilities(self, dim=2048, seed=0):
        """Query Interface (Phase 6): introspect this mind into a capability REGISTRY -- a VSA table with one row
        per public faculty (name, a heuristic domain, its one-line doc). 'What can this mind do?' then becomes an
        ordinary data query: `mind.query("SELECT name FROM actions WHERE domain = 'render'", mind.capabilities())`
        or a GROUP BY domain census. See holographic_query.capability_registry."""
        from holographic_query import capability_registry
        return capability_registry(self, dim=dim, seed=seed)

    def explain_program(self, machine, program_vec, init_acc=None):
        """Query Interface (Phase 7): EXPLAIN a program WITHOUT running it. A DRY RUN with no handlers -- every
        APPLY is a no-op so the heavy work is skipped, but the machine walks the whole program, so the trace names
        which faculties it WOULD call and how many steps it takes. The program-level twin of the pipeline's
        plan()->EXPLAIN. See holographic_query.explain_program."""
        from holographic_query import explain_program
        return explain_program(machine, program_vec, init_acc=init_acc)

    def make_table(self, rows, roles, dim=1024, seed=0):
        """Query Interface (Phase 1): ingest tabular data (a list of {column: value} dicts) into a VSA Table --
        each row becomes a role-bound record, with the exact values kept beside the vectors. Pair with `query`.
        See holographic_query.from_rows."""
        from holographic_query import from_rows
        return from_rows(rows, roles, dim=dim, seed=seed)

    def query(self, sql, table):
        """Query Interface (Phases 2-3): run a small SQL subset (SELECT/FROM/WHERE/ORDER BY/LIMIT) over a VSA
        Table. Exact predicates (=, >, <) run on the stored props; the FUZZY predicate (~) ranks rows by semantic
        cosine and returns a per-row `_confidence` -- the two things a plain database can't do natively. See
        holographic_query.run_sql."""
        from holographic_query import run_sql
        return run_sql(sql, table)

    def svgf_denoise(self, image, normal, albedo, depth, levels=5, **kw):
        """Interactive Render Speed (technique E): edge-aware denoise a noisy (1-spp-style) image the engine's
        way -- a holographic bilateral filter whose edge-stopping is a cosine in the bound (normal, albedo, depth)
        feature space, run coarse-to-fine over the a-trous hierarchy. Similar surfaces blend, edges don't; it
        beats a plain blur measurably (kept negative: it denoises, it can't add detail). The sibling render pieces
        already exist -- robust_accumulate (firefly clamp), SPRTRecall (adaptive sampling), TemporalReuse
        (reproject). See holographic_svgf."""
        from holographic_denoisehome import Denoise                    # the Denoise home (consolidation R5)
        return Denoise.image(image, normal, albedo, depth, method="svgf", levels=levels, **kw)

    def forecast(self, series, d=20, alpha=0.1, abstain_width=None, seed=0):
        """Forecasting backlog (F3): the "forecast any data" door. Routes a 1-D series to the producer that
        calibrates tightest (linear AR vs analog recall), wraps it in a calibrated conformal interval, and
        abstains when uncertain. Returns a RoutedForecaster -- `.predict(last_window)` gives {point, interval,
        coverage, abstain, producer}. A misroute fails SAFE (wide interval), never a confident wrong answer. See
        holographic_forecast."""
        from holographic_forecast import route_and_forecast
        rf, info = route_and_forecast(series, d=d, alpha=alpha, abstain_width=abstain_width, seed=seed)
        return rf

    def analog_forecaster(self, contexts, successors, sim_floor=0.5, seed=0):
        """Forecasting backlog (F4): analog forecasting -- "find the past that looks like now, return what
        followed." Pure VSA recall (sublinear via HoloForest); yields a DISTRIBUTION over outcomes natively and
        ABSTAINS when no near analog exists. Use holographic_analog.delay_embed to build (context, successor)
        pairs from a series. See holographic_analog.AnalogForecaster."""
        from holographic_analog import AnalogForecaster
        return AnalogForecaster(sim_floor=sim_floor, seed=seed).fit(contexts, successors)

    def multi_horizon_forecaster(self, rollout_fn, alpha=0.1, kind="scalar"):
        """Forecasting backlog (F6): multi-horizon forecast with a TRUSTED-HORIZON gate -- calibrate a per-step
        interval that widens with the horizon, and report how far ahead a closed-loop `rollout_fn(state, H)` can
        be trusted so a sim/renderer substitutes a cheap forecast up to there and recomputes beyond it. Kept loud:
        chaotic systems have a short trusted horizon by nature (Lyapunov time). See holographic_horizon."""
        from holographic_horizon import MultiHorizonForecaster
        return MultiHorizonForecaster(rollout_fn, alpha=alpha, kind=kind)

    def generate_gated(self, codebook, confidence_floor=0.6, steps=12, seed=None, **kw):
        """Forecasting backlog (F5): confidence-gated generation -- generate a vector, then score how VALID it is
        (cosine to the nearest codebook atom) and ACCEPT or flag it. Turns open-ended generation into calibrated
        generation: a low-confidence sample is flagged so a caller can resample or abstain. Kept scoped: for
        open-ended generation this is a filter/abstention aid, not a correctness guarantee."""
        import numpy as _np
        from holographic_ai import cosine as _cos
        v = self.generate_vector(codebook, steps=steps, seed=seed, **kw)
        if isinstance(v, tuple):
            v = v[0]
        cb = _np.asarray(codebook, float)
        vv = _np.asarray(v, float)
        sims = cb @ vv / (_np.linalg.norm(cb, axis=1) * (_np.linalg.norm(vv) + 1e-12) + 1e-12)
        conf = float(sims.max())
        return {"vector": vv, "confidence": conf, "accepted": conf >= confidence_floor,
                "nearest": int(sims.argmax())}

    def recurrent_forecaster(self, kind="esn", n_in=1, n_res=600, dim=1024, seed=0):
        """Forecasting backlog (F7, de-silo): a gradient-free sequence producer, now reachable through the mind.
        `kind='esn'` -> EchoStateNetwork (nonlinear-lift readout); `kind='vsa'` -> VSAReservoir (permute IS the
        recurrence). `.fit(inputs, targets)` then `.predict(inputs)`. Pairs with `forecast`/conformal for a
        calibrated interval. See holographic_recurrent."""
        from holographic_recurrent import EchoStateNetwork, VSAReservoir
        return EchoStateNetwork(n_in, n_res=n_res, seed=seed) if kind == "esn" else VSAReservoir(dim=dim, seed=seed)

    def market_projector(self, dim=512, K=5, H=3, R=80, seed=1):
        """Forecasting backlog (F7, de-silo): the RayProjector time-series study -- casts rays into a data field
        and reads held-out quantiles -- now reachable through the mind. `.fit(moves, burst)` then `.project(row)`.
        See holographic_market.RayProjector."""
        from holographic_market import RayProjector
        return RayProjector(dim=dim, K=K, H=H, R=R, seed=seed)

    def adaptive_sample_budget(self, variance_of_mean, current_n, target_half_width, z=1.959963984540054):
        """Forecasting sweep (sec.5, renderer delegation): a CALIBRATED adaptive-sampling stop. Given a renderer's
        per-pixel variance-of-the-mean at `current_n` samples, return the EXTRA samples each pixel needs to reach
        `target_half_width` at confidence z (0 where already converged) -- "sample where the estimate is still
        uncertain, stop where it is confident," replacing a hand-set threshold. Honest: a pixel mean's interval is
        Gaussian/CLT (var falls as sigma^2/n; halving the interval costs 4x samples), NOT conformal -- a single
        pixel has no calibration set. See holographic_adaptive_sample."""
        from holographic_adaptive_sample import sample_budget
        return sample_budget(variance_of_mean, current_n, target_half_width, z=z)

    def scheduler_capacity(self, dim=None, gated=True, target_recall=0.9, seed=0):
        """Forecasting sweep (sec.5.5): the scheduler's cost model IS a forecaster. Instead of assuming the
        theoretical packing wall (~0.10*D), MEASURE it -- probe growing superposition loads, measure gated cleanup
        recall, and return the largest load whose recall stays >= target_recall (a CALIBRATED capacity) plus the
        recall curve, so the scheduler packs as many as it is confident it can. `should_superpose` (in
        holographic_superschedule) gates a batch on the measured wall. Honest finding: the measured wall is often
        BELOW the theoretical dial at a strict target -- assuming overpacks. See holographic_superschedule."""
        from holographic_superschedule import calibrated_capacity, pack_capacity
        d = dim if dim is not None else self.dim
        cap, curve = calibrated_capacity(d, gated=gated, target_recall=target_recall, seed=seed)
        return {"capacity": cap, "curve": curve, "theoretical": pack_capacity(d, gated=gated),
                "target_recall": target_recall, "gated": gated}

    def calibrate_forecast(self, preds, actuals, alpha=0.1, kind="scalar", abstain_width=None):
        """Forecasting backlog (F1): wrap ANY producer's forecasts in a CALIBRATED prediction interval that
        abstains when too wide to trust. Fit on a held-out set of (prediction, truth) pairs -- scalar (scored by
        |error|) or vector (scored by 1 - cosine, the engine's own metric) -- and get a ConformalForecaster whose
        `.predict(point)` returns {point, interval/cosine_radius, coverage, abstain}. Distribution-free, no learned
        weights -- the forecasting twin of RecallNull. See holographic_conformal."""
        from holographic_conformal import ConformalForecaster
        cf = ConformalForecaster(alpha=alpha, kind=kind, abstain_width=abstain_width)
        cf.calibrate(list(preds), list(actuals))
        return cf

    def adaptive_conformal(self, alpha=0.1, gamma=0.05, window=200):
        """Forecasting backlog (F2): temporal conformal for time series (which break exchangeability). Adaptive
        Conformal Inference holds LONG-RUN coverage at 1-alpha under drift by widening after a miss / narrowing
        after a hit -- `.step(residual)` per observation, `.realized_coverage()` reads the held rate. Kept loud:
        under a fundamental regime change (~0% overlap) no feedback rule recovers coverage -- abstain and flag
        drift. See holographic_conformal.AdaptiveConformal."""
        from holographic_conformal import AdaptiveConformal
        return AdaptiveConformal(alpha=alpha, gamma=gamma, window=window)

    def forecast_coverage_report(self, residuals_calib, residuals_test, alphas=(0.01, 0.05, 0.1, 0.2)):
        """Forecasting backlog (F8): the coverage instrument -- for each alpha, confirm the empirical coverage on
        held-out residuals tracks the nominal 1-alpha. The forecasting twin of calibration_report; an interval you
        cannot verify is one you cannot trust. See holographic_conformal.coverage_report."""
        from holographic_conformal import coverage_report
        return coverage_report(residuals_calib, residuals_test, alphas=alphas)

    def forecast_crps(self, samples, actual):
        """Forecasting backlog (F8): CRPS -- the proper score for a probabilistic (sample) forecast; coverage says
        the interval is wide enough, CRPS says the forecast is GOOD (rewards accuracy AND sharpness). Lower is
        better; a sharp-and-accurate forecast scores strictly below a vague one. See holographic_conformal."""
        from holographic_conformal import crps_sample
        return crps_sample(samples, actual)

    def render_pipeline(self, preset="preview", **overrides):
        """Render/Sim Pipeline (Phases 1-2): build a configured, validated render+sim pipeline. `preset` is
        'preview' / 'final' / 'interactive', or a holographic_pipeline.PipelineConfig; `**overrides` tweak
        individual flags. Returns a Pipeline -- call `.plan()` to see exactly which stages will run and WHY
        (without rendering), or `.run(scene, seed, renderer=...)` to execute a frame. The builder auto-includes
        prerequisites (ask for SVGF, get the G-buffer) and rejects impossible combos up front with a clear
        message. See holographic_pipeline."""
        from holographic_pipeline import PipelineConfig, build_pipeline
        if isinstance(preset, str):
            cfg = {"preview": PipelineConfig.preview, "final": PipelineConfig.final,
                   "interactive": PipelineConfig.interactive}[preset]()
        else:
            cfg = preset
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return build_pipeline(cfg)

    def field_effect(self, sdf, effect, radius=1.0, falloff="smooth", strength=1.0, texture=None):
        """Render/Sim Pipeline (Part 4): a shaped zone of influence -- the SDF is the shape, its distance is the
        falloff, `effect(points, weight)` is what it does (attractor/wind/drag/density). Compose several with
        holographic_fieldeffect.FieldGroup (they add); attach one to a moving node with AttachedFieldEffect.
        Ready effects: attract_to / repel_from / uniform_force. See holographic_fieldeffect."""
        from holographic_fieldeffect import FieldEffect
        return FieldEffect(sdf, effect, radius=radius, falloff=falloff, strength=strength, texture=texture)

    def particle_sim(self, pos, vel, force_fn, integrator="symplectic"):
        """Render/Sim Pipeline (Phase 0 / G2): a point-mass sim advanced by the shared, energy-stable symplectic
        integrator -- exactly what a FieldEffect's summed forces drive. `force_fn(pos, vel) -> accelerations`.
        `.advance(dt)` steps it. See holographic_integrate.ParticleSim / SimStep."""
        from holographic_integrate import ParticleSim
        return ParticleSim(pos, vel, force_fn, integrator=integrator)

    def near_surface_to_sdf(self, near_surface, h=1.0, threshold=0.0):
        """Sweep 3 local completion (photo-to-3D): turn a NEAR-SURFACE signed field (accurate only in a thin band
        around the surface -- e.g. from depth unprojection) into a FULL, globally-consistent signed distance
        field. The band's sign is all we trust away from the surface, so we threshold it to an inside/outside
        occupancy and run the existing fast-sweeping eikonal (`signed_distance_field`) to redistance everywhere.
        Honest reuse: the eikonal solver already does the extension; this just prepares its input from a band."""
        import numpy as _np
        inside = _np.asarray(near_surface, float) < threshold        # sign is the reliable part away from the surface
        return self.signed_distance_field_3d(inside, h=h) if inside.ndim == 3 else self.signed_distance_field(inside, h=h)

    def texture_map(self, image, wrap="repeat"):
        """Sweep 3 local completion: an image-based TEXTURE MAP sampled by UV with bilinear interpolation -- the
        per-texel detail a factor-level PBRMaterial couldn't carry. Pass one to a PBRMaterial's `*_map` argument
        (base_color_map/metallic_map/roughness_map/emissive_map) and call material.sample(u, v) for the effective
        shaded values. See holographic_materialio.TextureMap."""
        from holographic_materialio import TextureMap
        return TextureMap(image, wrap=wrap)

    def graph_namespace(self, branching=6, beam=1, seed=0):
        """Sweep 3 local completion (fit-correct): a hierarchical namespace/navigation tree over labelled vectors
        (GraphMemory) -- observe_vector(v, label) grows the tree, classify_vector(v) routes a query to its region.
        Use it for HIERARCHY / NAMESPACE / NAVIGATION (a DB namespace tree, a region index), NOT for exact recall
        -- its recall accuracy is a documented negative that collapses at scale, which is why the sweep re-homed
        it here. See holographic_graph_memory.GraphMemory."""
        from holographic_graph_memory import GraphMemory
        return GraphMemory(self.dim, branching=branching, beam=beam, seed=seed)

    def directional_field(self, dirs, values, order=3):
        """Sweep 3 item 8: project a DIRECTIONAL function (sampled `values` at unit directions `dirs`) onto
        spherical-harmonic coefficients -- the SAME primitive for directional LIGHT (PRT radiance transfer, splat
        view-dependent colour) and directional SOUND (ambisonic encoding). `sample_directional` reconstructs.
        Reuses prt's SH basis (no fork). See holographic_spharm."""
        from holographic_spharm import sh_project
        return sh_project(dirs, values, order=order)

    def sample_directional(self, coeffs, dirs, order=3):
        """Sweep 3 item 8: reconstruct a directional function from its spherical-harmonic coefficients at `dirs`
        (radiance toward a view direction, or ambisonic gain toward a listening direction). Inverse of
        directional_field. See holographic_spharm."""
        from holographic_spharm import sh_reconstruct
        return sh_reconstruct(coeffs, dirs, order=order)

    def conditional_propagator(self, transitions, ridge=1e-3):
        """Sweep 3 item 9: a CONDITIONAL Propagator -- one learned dynamics operator per ACTION, so predict is a
        bind of that action's transform onto the state (dynamics' 'predict = bind a transform to a state', made
        action-conditional). Unifies lookahead's per-action forward model with the dynamics module and gives
        model-based planning (`.plan(state, actions, codebook)` re-anchors each hop). `transitions` maps each
        action to its (state, next_state) pairs. See holographic_condprop.ConditionalPropagator."""
        from holographic_condprop import ConditionalPropagator
        return ConditionalPropagator.learn(transitions, ridge=ridge)

    def storage_spine(self, block_size=32):
        """Sweep 3 item 7: one content-addressed, deduplicated, erasure-robust byte store -- uri KEYS a record,
        a content hash DEDUPS identical payloads, a fountain code makes retrieval robust to lost droplets. The
        shared spine for the query DBs, texture atlases, scene deltas, and the compile cache. `.put(tags, bytes)`
        / `.get(key, loss=...)`. See holographic_storage.StorageSpine."""
        from holographic_storage import StorageSpine
        return StorageSpine(block_size=block_size)

    def faculties(self):
        """The capability table (Sweep 3 item 1): the sorted names of every faculty currently callable from a VSA
        program as `APPLY <name>` -- built-ins (cleanup/denoise/...) plus anything `register_apply_handler` added
        (an octree query, an agent behaviour, a fitted embedding). This is the SAME live handler set the machine's
        APPLY uses, so introspection, a drives scheduler, or an moe gate all read one registry -- the convergence
        point the sweep flagged. Registration is `register_apply_handler(name, fn)`; this is the read side."""
        return sorted(self._procedure_handlers().keys())

    def spatial_index(self, points, cell_size):
        """Sweep 3 item 2: ONE shared uniform-grid spatial index over a point set -- radius / knn / closest-point
        queries in O(1)-ish (nearby cells only), byte-identical to a brute-force scan. The widest-fanout mechanism
        (cull, navigation, collision broadphase, sampling, Walk-on-Spheres closest-point all ask the same query).
        See holographic_spatial.SpatialGrid."""
        from holographic_spatial import SpatialGrid
        return SpatialGrid(points, cell_size)

    def reaction_diffusion(self, size=64, dim=48, steps=40, seed=0, **kw):
        """Sweep 3 item 3: a reaction-diffusion cellular automaton (HyperCA) -- a local update rule over a
        hypervector field from which global patterns emerge (spots/stripes/fronts). One solver, many domains
        (patina/weathering, procedural texture, fur/skin patterns, crystal growth, erosion). Runs `steps` and
        returns the stepped HyperCA (read `.grid`). See holographic_automaton.HyperCA."""
        from holographic_automaton import HyperCA
        ca = HyperCA(size=size, dim=dim, seed=seed, **kw)
        for _ in range(int(steps)):
            ca.step()
        return ca

    def emergent_concepts(self, vigilance=0.45, commit=3.0, prune=0.5, seed=0, **kw):
        """Sweep 3 item 4: online, label-free concept growth -- watch a stream, grow concepts with no fixed
        category count, and COMMIT one (via the double-diffusion staircase, which pulls in the diffusion
        mechanism) when its slowly-integrating support survives. An online GROUP BY / clustering reusable across
        agent situations, vision classes, market regimes, or the knowledge registry. Feed it with `.perceive(x)`.
        See holographic_emergence.EmergentConcepts."""
        from holographic_emergence import EmergentConcepts
        return EmergentConcepts(vigilance=vigilance, commit=commit, prune=prune, seed=seed, **kw)

    def temporal_reuse(self):
        """Sweep 3 item 5: the temporal-reuse loop -- reuse last frame's per-cell result, reproject it
        (backward-warp), and re-solve ONLY the dirty region, optionally accumulating (running average) for noisy
        estimators. The render/solve SPEED discipline (path tracer, Walk-on-Spheres, fluid/wave). Call
        `.solve(solve_fn, n, dirty=..., reproject=..., accumulate=...)`. See holographic_temporal.TemporalReuse."""
        from holographic_temporal import TemporalReuse
        return TemporalReuse()

    def cosserat_strand(self, points_or_strand, bend_stiffness=0.5, shape_stiffness=0.6):
        """A hair as a COSSERAT ROD (H2b): each segment carries an orientation frame, so the strand HOLDS its
        curl under gravity and can carry a TWIST -- the quality upgrade over plain bend springs. Accepts an
        (n,3) point array or a groom Strand. `.step()/.settle()` simulate it; `.set_root_twist(a)` twists it;
        `.curl_amount()`/`.twist_of(i)` read it out. See holographic_cosserat.CosseratStrand."""
        from holographic_cosserat import CosseratStrand, from_strand
        if hasattr(points_or_strand, "points"):
            return from_strand(points_or_strand, bend_stiffness=bend_stiffness, shape_stiffness=shape_stiffness)
        return CosseratStrand(points_or_strand, bend_stiffness=bend_stiffness, shape_stiffness=shape_stiffness)

    def groom_hair(self, surface_sdf, n_strands, bounds, length=1.0, n_pts=8, curl=0.0, lean=0.0,
                   width=0.02, seed=0, length_jitter=0.0):
        """HAIR GROOM (H1): grow `n_strands` rooted on an SDF surface, each along its outward normal (+ optional
        lean), straight or curly, smoothed for rendering. `bounds`=(lo_vec,hi_vec). Returns a list of Strand.
        See holographic_groom.groom."""
        from holographic_groom import groom
        return groom(surface_sdf, n_strands, bounds, length=length, n_pts=n_pts, curl=curl, lean=lean,
                     width=width, seed=seed, length_jitter=length_jitter)

    def simulate_hair(self, strands, steps=60, dt=1.0 / 60.0, gravity=(0.0, -9.8, 0.0), wind=None,
                      body_sdf=None, collide_radius=0.0, bend_compliance=1e-3):
        """HAIR DYNAMICS (H2): simulate strands as PBD chains (root pinned, inextensible with Follow-The-Leader,
        bend springs for stiffness) under gravity, optional wind force, and body collision. Returns new strands.
        See holographic_groom.simulate_strands."""
        from holographic_groom import simulate_strands
        return simulate_strands(strands, steps=steps, dt=dt, gravity=gravity, wind=wind, body_sdf=body_sdf,
                                collide_radius=collide_radius, bend_compliance=bend_compliance)

    def interpolate_hair(self, guides, render_roots, k=3, clump=0.4):
        """GUIDE INTERPOLATION (H3): make many render strands from a few simulated guide strands by blending the
        k nearest guides and clumping toward one -- what makes full fur affordable. See holographic_groom."""
        from holographic_groom import interpolate_strands
        return interpolate_strands(guides, render_roots, k=k, clump=clump)

    def hair_wind(self, strength=2.0, res=24, bounds=((-2, 2), (-2, 2), (-2, 2)), octaves=3, seed=0, base=(1.0, 0.0, 0.0)):
        """CURL-NOISE WIND (H7): a divergence-free (volume-preserving) turbulent wind field; call `.force(strand)`
        for the per-point force to pass to simulate_hair. Fur ripples without ballooning. See holographic_groom.CurlWind."""
        from holographic_groom import CurlWind
        return CurlWind(strength=strength, res=res, bounds=bounds, octaves=octaves, seed=seed, base=base)

    def render_hair(self, strands, camera, light_dir=(0.3, 0.6, 0.6), width=400, height=400,
                    shader="kajiya", hair_color=(0.55, 0.35, 0.15), smooth_levels=2, lod_stride=1):
        """RENDER HAIR (H4/H5/H6): project each strand's smoothed centerline and shade its segments by their
        TANGENT -- `shader`='kajiya' (anisotropic sheen) or 'marschner' (physical R/TT/TRT with a colored
        secondary highlight). Returns an (H,W,3) image. See holographic_hairshade.render_hair."""
        from holographic_hairshade import render_hair
        return render_hair(strands, camera, light_dir=light_dir, width=width, height=height, shader=shader,
                           hair_color=hair_color, smooth_levels=smooth_levels, lod_stride=lod_stride)

    def solve_pde(self, sdf, boundary_value, points, source=None, n_walks=256, eps=1e-3, seed=0):
        """WALK ON SPHERES: solve Laplace (Delta u = 0) or Poisson (-Delta u = source) on the interior of an SDF,
        with NO meshing, by random walks that step by the distance-to-boundary (one SDF eval) until they hit the
        boundary and read `boundary_value` there. Returns (solution, standard_error) at each query point. Works on
        any shape you can write an SDF for. SIGGRAPH list #7. See holographic_wos.solve_on_sdf."""
        from holographic_wos import solve_on_sdf
        return solve_on_sdf(sdf, boundary_value, points, source=source, n_walks=n_walks, eps=eps, seed=seed)

    def steady_heat(self, sdf, boundary_temperature, points, n_walks=256, eps=1e-3, seed=0):
        """STEADY-STATE heat on any SDF shape: hold the boundary at `boundary_temperature(point)` and find the
        equilibrium temperature at interior `points`. This is Laplace's equation via Walk on Spheres -- the
        grid-free, mesh-free steady complement to the transient `holographic_heat` diffusion and the `wave`
        field. Returns (temperature, standard_error). See holographic_wos."""
        from holographic_wos import solve_on_sdf
        return solve_on_sdf(sdf, boundary_temperature, points, n_walks=n_walks, eps=eps, seed=seed)

    def curl_noise(self, res=64, bounds=((0.0, 8.0), (0.0, 8.0)), octaves=4, seed=0, obstacle_sdf=None, ramp=1.0):
        """CURL NOISE: divergence-free procedural turbulence (u, v) on a grid -- the curl of an fBm streamfunction,
        so it never compresses (no sources/sinks). Optional `obstacle_sdf` makes the flow go AROUND a shape. Cheap
        wind/smoke detail with no fluid solve. SIGGRAPH list #1. See holographic_curlnoise.curl_noise."""
        from holographic_curlnoise import curl_noise
        return curl_noise(res, bounds=bounds, octaves=octaves, seed=seed, obstacle_sdf=obstacle_sdf, ramp=ramp)

    def tearable_cloth(self, rows=12, cols=12, spacing=1.0, compliance=2e-3, material="paper",
                       tear_strain=None, pin="top"):
        """A TEARABLE thin sheet: a PBD cloth whose links SNAP when stretched past the material's tear strength,
        so it rips and separates into pieces when yanked. `.step(pull=..., gravity=...)` advances and tears;
        `.connected_components()` / `.piece_sizes()` report the split. SIGGRAPH list #2 (fracture -- a new
        capability). See holographic_tear.TearableCloth."""
        from holographic_tear import TearableCloth
        return TearableCloth(rows=rows, cols=cols, spacing=spacing, compliance=compliance, material=material,
                             tear_strain=tear_strain, pin=pin)

    def levitation_chamber(self, height=0.10, wavelength=0.0086, amplitude=4000.0, n_beads=40,
                           gravity=9.81, bead_radius=1e-3, bead_density=25.0, seed=0):
        """ACOUSTIC LEVITATION: beads in a vertical standing wave feel the Gor'kov radiation force and are trapped
        at the pressure NODES (spaced lambda/2) against gravity. `.settle(field_on=True)` holds them aloft;
        `field_on=False` lets them fall. The 'sound moves objects' showpiece. Acoustics A7 -- reuses the standing
        field idea (A3), the particle system, and gravity. See holographic_levitate.LevitationChamber."""
        from holographic_levitate import LevitationChamber
        return LevitationChamber(height=height, wavelength=wavelength, amplitude=amplitude, n_beads=n_beads,
                                 gravity=gravity, bead_radius=bead_radius, bead_density=bead_density, seed=seed)

    def room_acoustics(self, size=(5.0, 4.0, 3.0), material="plaster", absorption=None, c=343.0):
        """GEOMETRIC ROOM ACOUSTICS: how a room echoes. `.rt60()` is the reverberation time (Sabine), `.reflections
        (source, listener)` the early echoes via the image-source method (arrival = path/c, level from the wall
        reflectance), `.impulse_response(...)` the sampled room response. Hard rooms ring, soft rooms are dead.
        Acoustics A6 -- the acoustic twin of the path tracer, reusing A2's reflectance. See
        holographic_roomacoustic.ShoeboxRoom."""
        from holographic_roomacoustic import ShoeboxRoom
        return ShoeboxRoom(size=size, material=material, absorption=absorption, c=c)

    def power_law_viscosity(self, shear_rate, K=1.0, n=1.8, eta_min=1e-4, eta_max=1e4):
        """The non-Newtonian power-law viscosity eta = K * shear_rate^(n-1): a viscosity that DEPENDS on how fast
        the fluid is sheared. n>1 thickens under shear (cornstarch/oobleck), n<1 thins (ketchup, paint), n=1 is
        Newtonian. Returns a per-cell viscosity field. See holographic_nonnewtonian.power_law_viscosity."""
        from holographic_nonnewtonian import power_law_viscosity
        return power_law_viscosity(shear_rate, K, n, eta_min=eta_min, eta_max=eta_max)

    def nonnewtonian_fluid(self, shape, power_law_n=1.8, consistency_K=1.0, **kwargs):
        """A fluid solver with NON-NEWTONIAN rheology: its viscosity is the shear-rate-dependent power law, so it
        can carry cornstarch (n>1, shear-thickening -- stiffens where you shear it hard) or a shear-thinning fluid
        (n<1). A StableFluid in its power-law mode; n=1 would be ordinary Newtonian. See holographic_fluid.StableFluid
        (power_law_n / consistency_K) and holographic_nonnewtonian."""
        from holographic_fluid import StableFluid
        return StableFluid(shape, power_law_n=power_law_n, consistency_K=consistency_K, **kwargs)

    def wave_field(self, shape, c=343.0, dx=1.0, damping=0.0, absorb_border=0):
        """A scalar acoustic PRESSURE field that PROPAGATES (the compressible wave the incompressible fluid can't
        carry): d2p/dt2 = c^2 grad^2 p by leapfrog. `.pulse(center)` taps it, `.step(dt)` advances it (auto-
        subdivided to stay CFL-stable), an `absorb_border` sponge stops edge reflections. `c` may be a per-cell
        field from material sound_speed. Acoustics A3 -- the low-frequency wave complement to ray acoustics, and
        the standing field levitation (A7) will use. See holographic_wave.WaveField."""
        from holographic_wave import WaveField
        return WaveField(shape, c=c, dx=dx, damping=damping, absorb_border=absorb_border)

    def read_wav(self, path):
        """Read a PCM WAV file -> (samples in [-1,1] mono, sample_rate). The front door for driving acoustics/
        cymatics from a real sound. Acoustics A1. See holographic_audio.read_wav."""
        from holographic_audio import read_wav
        return read_wav(path)

    def audio_spectrum(self, samples, rate, k=6):
        """The `k` dominant frequencies (Hz) and their amplitudes in a signal -- the tones that drive a plate or
        fluid. Acoustics A1. See holographic_audio.dominant_frequencies (and `spectrum`, `frames` there)."""
        from holographic_audio import dominant_frequencies
        return dominant_frequencies(samples, rate, k=k)

    def acoustic_impedance(self, material):
        """Characteristic acoustic impedance Z = rho * c (rayl) of a material, reused from its density x speed of
        sound -- the acoustic twin of a refractive index. Acoustics A2. See holographic_acoustic.impedance."""
        from holographic_acoustic import impedance
        return impedance(material)

    def acoustic_interface(self, mat_a, mat_b):
        """Sound crossing from material A into B: (R, T) = fractions of energy reflected and transmitted, from the
        impedance mismatch (a big mismatch like air/steel reflects nearly all). Energy conserved. Acoustics A2. See
        holographic_acoustic.interface; `wall_absorption` for a surface's absorbed fraction."""
        from holographic_acoustic import interface
        return interface(mat_a, mat_b)

    def chladni_plate(self, shape="square", grid=40, medium="sand", n_modes=48, base_hz=200.0,
                      n_grains=6000, seed=0):
        """A vibrating plate whose CYMATIC figures are its Laplacian eigenmodes: `.drive(freqs, amps)` (or
        `.drive_mode(k)`) sets the displacement from a sound's spectrum, `.step_medium(dt)`/`.settle()` drift sand
        to the nodes, `.render()` shows the figure. The headline acoustics demo, reusing the spectral eigenmodes.
        Acoustics A4. See holographic_cymatics.ChladniPlate."""
        from holographic_cymatics import ChladniPlate
        return ChladniPlate(shape=shape, grid=grid, medium=medium, n_modes=n_modes, base_hz=base_hz,
                            n_grains=n_grains, seed=seed)

    def oxidation_field(self, shape, exposure=None, moisture=1.0, seed=None):
        """A CORROSION front over a surface grid: rust/patina that NUCLEATES at exposed/wet faces (default: the
        border) and SPREADS inward as a reaction-diffusion front. `.step(material, dt)` advances it, `.albedo(
        material)` gives the per-cell base->oxide colour blend (steel->rust, copper->patina). Process M4. See
        holographic_oxidation.OxidationField; `oxide_color` for a single whole-object sample."""
        from holographic_oxidation import OxidationField
        return OxidationField(shape, exposure=exposure, moisture=moisture, seed=seed)

    def oxide_color(self, material, ox_fraction):
        """The blended colour of a material at oxidation fraction 0..1 -- pristine base to full oxide (rust orange,
        patina green). The weathering interpolation for a single sample. See holographic_oxidation.oxide_color."""
        from holographic_oxidation import oxide_color
        return oxide_color(material, ox_fraction)

    def burn_object(self, material, mass_kg, temp_K=293.15):
        """An object BEING CONSUMED by fire: `.light()` ignites it, `.step(dt)` advances the burn -- it loses mass
        (drives an M6 Fire), its appearance marches base->char->ash, and it emits that material's smoke, ending as
        ash. Process M7, tying M6 (combustion) and, via `evaporate`, M5 (a puddle drying up). See
        holographic_burn.BurningObject; `char_color` for the appearance blend at a burn fraction."""
        from holographic_burn import BurningObject
        return BurningObject(material, mass_kg, temp_K=temp_K)

    def char_color(self, material, burn_fraction):
        """The surface colour at burn fraction 0..1: pristine base -> char (blackened) -> ash (grey). The burning
        interpolation. See holographic_burn.char_color."""
        from holographic_burn import char_color
        return char_color(material, burn_fraction)

    def element(self, symbol):
        """A chemical ELEMENT's engine-relevant properties by symbol: name, atomic number, atomic mass, density,
        melt/boil points, flame-test colour (or None), category. The atomic ingredients materials are made of. See
        holographic_elements.element."""
        from holographic_elements import element
        return element(symbol)

    def material_elemental(self, name):
        """A material's ELEMENTAL makeup and everything derived from it: {composition (element:count/ratio),
        molar_mass, flame_color, mass_fractions}. molar_mass feeds the gas law (T1); flame_color is the
        ratio-weighted BLEND of the constituents' flame-test colours (the emission-line colour a copper compound
        burns green, which the blackbody continuum alone can't give -- feeds M6). None if no composition on file.
        This is a material referencing its elements + ratio, feeding simulation. See holographic_elements."""
        from holographic_elements import material_elemental
        return material_elemental(name)

    def fire(self, material, fuel_kg, temp_K=293.15):
        """A material-aware BURNING body: `.step(dt)` checks ignition against the material's autoignition point,
        consumes fuel at its burn rate once lit (a fire latches and sustains until the fuel runs out), and reports
        the smoke it makes (that material's colour + soot) and the flame colour (blackbody at its temperature) --
        so wood smoke and plastic smoke genuinely differ. Process M6, standing on the heat model (T4) and blackbody
        (T3). See holographic_combustion.Fire; `ignites` gates it; couplings configure_fluid / emit_smoke feed the
        fluid solver and surface emitter."""
        from holographic_combustion import Fire
        return Fire(material, fuel_kg, temp_K=temp_K)

    def ignites(self, material, temperature_K):
        """True if `material` is at or above its autoignition temperature (hot enough to catch fire); False below
        it, or if the material is not flammable. The honest ignition gate. See holographic_combustion.ignites."""
        from holographic_combustion import ignites
        return ignites(material, temperature_K)

    def phase_state(self, material, mass_kg, temp_K=293.15, pressure_Pa=101325.0):
        """A parcel of `material` with mass across solid/liquid/gas at one temperature: `.add_heat(Q)` warms it and
        drives melt/boil/freeze/condense, HOLDING temperature flat during a transition while the latent heat is
        paid (the boiling plateau). Boiling point tracks pressure (from the gas model T1). Process M5, standing on
        the heat model (T4). See holographic_phase.PhaseState / boiling_point_at."""
        from holographic_phase import PhaseState
        return PhaseState(material, mass_kg, temp_K=temp_K, pressure_Pa=pressure_Pa)

    def blackbody_color(self, temp_K, normalize="hue"):
        """The sRGB colour a blackbody GLOWS at temperature `temp_K` -- red ember (~900 K), orange flame, white
        filament (~2800 K), blue-white star (~12000 K) -- from Planck's law integrated against the CIE curves.
        Thermodynamics T3; the ember/flame/glowing-char colour the combustion & burn processes (M6/M7) will paint
        by temperature. `normalize='hue'` gives the pure hue at full value; 'none' keeps the luminance ratio. See
        holographic_blackbody.blackbody_rgb."""
        from holographic_blackbody import blackbody_rgb
        return blackbody_rgb(temp_K, normalize=normalize)

    def diffuse_heat(self, temp_field, alpha, dx=1.0, dt=None, steps=1):
        """Spread heat through a temperature FIELD by Fourier conduction (dT/dt = alpha*laplacian(T)) for `steps`
        steps, auto-substepped to stay stable at any dt, insulated boundaries (total heat conserved). `alpha` =
        thermal diffusivity k/(rho c) (see thermal_diffusivity / material_thermal). Thermodynamics T4 -- the
        temperature source the phase-change / combustion / decay processes read. See holographic_heat.diffuse_heat."""
        from holographic_heat import diffuse_heat
        return diffuse_heat(temp_field, alpha, dx=dx, dt=dt, steps=steps)

    def heat_body(self, material, mass_kg, temp_K=293.15):
        """A lumped body of a named `material` at a uniform temperature: `.add_energy(Q)` raises it by Q/(m c),
        `.newton_cool(ambient, hA, dt)` relaxes it toward ambient. Specific heat is pulled from the material
        definition. Thermodynamics T4. See holographic_heat.HeatBody / material_thermal."""
        from holographic_heat import HeatBody, material_thermal
        c = material_thermal(material)["specific_heat"]
        return HeatBody(mass_kg, c, temp_K=temp_K)

    def material_thermal(self, material):
        """Thermal properties of a named material -- {'density','specific_heat','thermal_conductivity'} in SI,
        reusing the definition library and the enrichment data (conductivity), with thermal_diffusivity easily
        derived. See holographic_heat.material_thermal."""
        from holographic_heat import material_thermal
        return material_thermal(material)

    def ideal_gas(self, name="air", temp_K=293.15, pressure_Pa=101325.0):
        """A parcel of gas in a definite state: `.density()`, `.sound_speed()`, `.adiabatic_change(V2/V1)`. The
        ideal gas law P V = m R_specific T. Thermodynamics T1 -- and its speed of sound cross-checks the
        definitions' tabulated value. See holographic_gas.IdealGas."""
        from holographic_gas import IdealGas
        return IdealGas(name=name, temp_K=temp_K, pressure_Pa=pressure_Pa)

    def boiling_point(self, pressure_Pa, material="water"):
        """The boiling temperature (K) at a given pressure, from Clausius-Clapeyron (lower pressure -> boils
        cooler, the mountain effect). Defaults to water. Thermodynamics T1 -- the fact phase change (M5) consumes
        to know WHEN a liquid turns to vapour. See holographic_gas.boiling_point."""
        from holographic_gas import boiling_point
        return boiling_point(pressure_Pa)

    def grain_material(self, axis=(0, 1, 0), light=(0.72, 0.52, 0.32), dark=(0.40, 0.26, 0.14),
                       ring_scale=8.0, fibre=0.35, warp=0.6, seed=0, center=(0.0, 0.0, 0.0)):
        """A WOOD-GRAIN colour socket f(points)->(M,3): concentric rings along an axis + lengthwise fibre streaks +
        an fBm domain-warp that bends rings into knots. Volumetric in object space, so a cut board shows the rings
        continue. Drop it into a channel: `surface_material(color=Param(field=mind.grain_material(...)))`. Structure
        primitive M1. See holographic_grainmat.wood_albedo (also `substrate_layers` there for plywood/strata)."""
        from holographic_grainmat import wood_albedo
        return wood_albedo(axis=axis, light=light, dark=dark, ring_scale=ring_scale, fibre=fibre, warp=warp,
                           seed=seed, center=center)

    def material_inclusions(self, base, inclusions, seed=0):
        """An IMPURITY/INCLUSION colour socket f(points)->(M,3): the `base` material everywhere except in
        noise-blob pockets where each inclusion shows (carbon in steel, bubbles in glass, veins in stone).
        `inclusions` is [(material, fraction, scale), ...] with fraction the CALIBRATED covered fraction. base and
        materials are matlib preset names or rgb triples. Volumetric, deterministic. Structure primitive M3 (the
        planet's ore-deposit noise-threshold pattern, scoped to a material). See holographic_inclusions."""
        from holographic_inclusions import with_inclusions
        return with_inclusions(base, inclusions, seed=seed)

    def crystal_material(self, n_seeds=32, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), seed=0, jitter=1.0,
                         base=(0.55, 0.57, 0.62), spread=0.18, crack=(0.05, 0.05, 0.06), crack_width=0.03):
        """A POLYCRYSTALLINE colour socket f(points)->(M,3): a Worley/Voronoi partition where each grain is a
        slightly different facet colour, darkened along the cell boundaries (cracks). Volumetric, deterministic.
        Structure primitive M2. Returns (cells, socket) so you can also query cells.ids / cells.edge_distance (e.g.
        for a crack roughness channel via holographic_cellular.crack_mask). See holographic_cellular."""
        from holographic_cellular import VoronoiCells, cell_albedo
        cells = VoronoiCells(n_seeds=n_seeds, bounds=bounds, seed=seed, jitter=jitter)
        return cells, cell_albedo(cells, base=base, spread=spread, crack=crack, crack_width=crack_width, seed=seed)

    def render_material(self, name, color=None):
        """A PHYSICALLY-PLAUSIBLE render material from the library (holographic_matlib) as a first-class
        SurfaceMaterial -- so any of its ~130 glTF-PBR presets (metals, woods, stones, gems, biomes, planetary layers,
        ore deposits) drives preview / path_trace / RenderSession directly. This is the fork's physical definitions
        plugged into our render pipeline: data-driven materials, not hand-set demo colours. See
        SurfaceMaterial.from_matlib and holographic_matlib.material."""
        from holographic_surface import SurfaceMaterial
        return SurfaceMaterial.from_matlib(name, color=color)

    def material_catalog(self):
        """The whole render-material library grouped by class (diffuse/metal/wood/stone/glass/gem/emissive/biome/
        layer/deposit/liquid/fabric/organic) -- for a UI picker. See holographic_matlib.catalog."""
        import holographic_matlib as _ml
        return _ml.catalog()

    def fractal_planet(self, radius=1.0, seed=0, dim=256, octaves=4, relief=0.10, **kw):
        """A data-driven, physically-plausible PLANET as one region field: a fBm-displaced sphere painted by a
        Whittaker biome classifier (elevation/temperature/moisture -> ocean/desert/forest/ice), wrapped around
        interior shells (crust/mantle/core) with ore-deposit pockets. `.cross_section()` slices it; `.material_at()`
        colours any point; `.biome_histogram()` reports the surface mix. 'As above, so below' -- a planet is region
        composition all the way down. See holographic_matlib.fractal_planet."""
        import holographic_matlib as _ml
        return _ml.fractal_planet(radius=radius, seed=seed, dim=dim, octaves=octaves, relief=relief, **kw)

    def physical_material(self, name):
        """The PHYSICAL properties of a named material (density kg/m3, viscosity, Young's modulus, refractive index,
        sound speed, specific heat, phase) from the definition library -- the numbers a SOLVER needs, so a simulation
        can be data-driven by real materials instead of raw parameters. Raises KeyError with near matches if unknown.
        See holographic_definitions.MATERIALS."""
        from holographic_definitions import MATERIALS
        if name not in MATERIALS:
            near = [n for n in MATERIALS if name.split("_")[0] in n][:6]
            raise KeyError("unknown material %r%s" % (name, (" -- did you mean: %s" % near) if near else ""))
        return dict(MATERIALS[name])

    def material_info(self, name):
        """EVERYTHING the engine knows about a named material in one view: its RENDER appearance (PBR: base colour,
        metallic, roughness, ior, class -- if it's a preset) AND its PHYSICAL properties (density, refractive index,
        viscosity, Young's modulus, sound speed, specific heat, phase -- if defined). This bridges the two material
        libraries so 'tell me about gold' returns both how it LOOKS (for rendering) and how it BEHAVES (for science).
        See holographic_materialindex.material_info."""
        import holographic_materialindex as _mi
        return _mi.material_info(name)

    def find_materials(self, query, k=10):
        """Discover materials across BOTH libraries (render presets + physical definitions) by plain-English keywords
        -- name, render class (metal/gem/liquid/...), or phase. Returns matches with which library each lives in.
        See holographic_materialindex.find_materials."""
        import holographic_materialindex as _mi
        return _mi.find_materials(query, k=k)

    def materials(self):
        """The whole material roster: every material with which library it lives in, plus a summary (counts by render
        class, physical count, overlap). The 'what materials do we have?' entry point. See holographic_materialindex."""
        import holographic_materialindex as _mi
        return {"summary": _mi.summary(), "materials": _mi.all_materials()}

    def material_units(self):
        """The units of the physical properties (density -> kg/m^3, youngs -> GPa, ...) -- so a returned value is
        self-describing for a scientist. See holographic_materialindex.physical_units."""
        import holographic_materialindex as _mi
        return _mi.physical_units()

    def materials_by_category(self, category):
        """The physical materials in a category (metal / liquid / gas / polymer / ceramic / glass / mineral / stone /
        wood / biological / building / semiconductor). See holographic_materialindex.physical_by_category."""
        import holographic_materialindex as _mi
        return _mi.physical_by_category(category)

    def validate_materials(self):
        """Plausibility-check the physical material database (units, ranges, category/phase). Empty list = clean --
        the honest self-audit of the library. See holographic_materialindex.validate_physical."""
        import holographic_materialindex as _mi
        return _mi.validate_physical()

    def resolve_scenario(self, description):
        """Turn a physical description ('a block of wood floating in water', 'a steel ball sinking in oil') into a
        VALIDATED, parameterised Scenario: it grounds each named thing to its physical properties, checks the physics
        (wood floats, steel sinks -- `.consistent`), and emits `.build_spec()` -- the phenomenon + solver family +
        per-body masses/volumes a shipped solver consumes. This is how a description becomes a physically-accurate
        simulation. See holographic_definitions.resolve_scenario."""
        from holographic_definitions import resolve_scenario
        return resolve_scenario(description, dim=self.dim, seed=self.seed)

    def quantity(self, value, unit, uncertainty=0.0, source=None):
        """A physical QUANTITY = value + unit + uncertainty + source, in the dimensional GRAMMAR (holographic_
        quantities): multiplication composes dimensions (density * volume -> mass), addition requires matching
        dimensions (a length plus a mass is refused as the grammar error it is), conversion is one call (`.to('ft')`),
        and uncertainty propagates. Extensible via register_unit. See holographic_quantities.Quantity."""
        from holographic_quantities import Quantity
        return Quantity(value, unit, uncertainty=uncertainty, source=source)

    def estimate_bill(self, bill, price_per_kg=None, carbon_factor=None):
        """'Render' the mass, cost, and embodied carbon of a bill of materials [(material, volume_m3), ...] by
        composing recipes over the definition library's densities (reused, never duplicated): mass = Sigma density*vol,
        cost = Sigma mass*price, carbon = Sigma mass*carbon_factor -- each dimensionally checked by the grammar.
        Returns {'mass','cost','carbon','missing'}; SAMPLE price/carbon tables are used if none supplied (flagged,
        pending a real USGS/ICE ingest). See holographic_quantities.bill_mass/bill_cost/bill_embodied_carbon."""
        from holographic_quantities import (bill_mass, bill_cost, bill_embodied_carbon,
                                            SAMPLE_PRICE_USD_PER_KG, SAMPLE_CARBON_KG_PER_KG)
        from holographic_definitions import build_standard_library
        lib = build_standard_library(dim=256, seed=0)              # small dim: only the density table is used
        price = price_per_kg if price_per_kg is not None else SAMPLE_PRICE_USD_PER_KG
        carbon = carbon_factor if carbon_factor is not None else SAMPLE_CARBON_KG_PER_KG
        mass, miss_m = bill_mass(lib, bill)
        cost, miss_c = bill_cost(lib, bill, price)
        co2, miss_k = bill_embodied_carbon(lib, bill, carbon)
        return {"mass": mass, "cost": cost, "carbon": co2,
                "missing": sorted(set(miss_m) | set(miss_c) | set(miss_k))}

    def render_session(self, sdf, materials, camera, width=256, height=256, bounds=None):
        """Open a RENDER SESSION over one scene -- the object that ties the renderers together so a preview and a
        photoreal final can't drift apart. Holds an SDF + a SurfaceMaterial per object id (or one for the whole SDF) +
        a camera, and derives every output from that SINGLE scene: `.preview()` (fast render_surface), `.render_final(
        spp, on_progress=)` (progressive path_trace), `.to_splats()` (a browser splat proxy), and `.edit_channel(id,
        channel, value)` (a live material edit that shows in both). This is what a demo page drives instead of
        re-wiring render_surface / path_trace / splat export by hand. See holographic_session.RenderSession."""
        from holographic_session import RenderSession
        return RenderSession(sdf, materials, camera, width=width, height=height, bounds=bounds)

    def sdf_surface_points(self, sdf, bounds, n=2000, seed=0, eps=0.02):
        """Sample points that lie ON an SDF's surface (random points + one Newton step onto the zero level, kept where
        |sdf|<eps) -- the front half of the SDF->splat bridge, so any SDF scene becomes splat-viewable via
        field_to_splats. Deterministic. See holographic_session.sdf_surface_points."""
        from holographic_session import sdf_surface_points
        return sdf_surface_points(sdf, bounds, n=n, seed=seed, eps=eps)

    def render_surface(self, sdf, camera, width, height, materials, **kw):
        """Render an SDF scene resolving every material channel PER HIT from its socket -- so a procedural pattern on
        any channel is a solid 3-D texture (wraps curved surfaces, no UV unwrap), and opacity alpha-composites one
        transparency layer. `materials` maps object id -> SurfaceMaterial (or one material for the whole SDF). Honest
        scope: environment reflection only (use render_dispatch / render_scene for object-object mirrors). See
        holographic_surface.render_surface."""
        from holographic_surface import render_surface
        return render_surface(sdf, camera, width, height, materials, **kw)

    def pattern_field(self, name, **params):
        """A named deterministic procedural pattern FIELD f(points)->[0,1] (checker, stripes, gradient, dots, noise,
        fbm) that plugs into ANY Param socket -- a material channel, a region field, an emitter rate. Deterministic by
        integer-lattice hash (PYTHONHASHSEED-independent). Use holographic_pattern.field_lerp to drive a channel
        lo..hi by the pattern. See holographic_pattern."""
        from holographic_pattern import make_pattern
        return make_pattern(name, **params)

    def radiance_transfer(self, sdf, points, normals, order=3, n=512):
        """PRECOMPUTED RADIANCE TRANSFER -- collapse the light-transport integral into a per-point transfer vector once,
        then RELIGHT with a dot product (no rays). For a STATIC scene the way a surface point turns incident lighting
        into outgoing radiance (including its own soft self-shadowing) depends only on geometry, so it is precomputed in
        a spherical-harmonic basis; runtime shading is `shade_prt(transfer, project_env_to_sh(light))`. The 'don't
        path-trace, just read out' idea: expensive to precompute, ~free to relight -- wins when the light changes often
        over fixed geometry. Returns the transfer matrix (len(points), order^2). See holographic_prt."""
        from holographic_prt import precompute_transfer
        return precompute_transfer(sdf, points, normals, order=order, n=n)

    def holographic_radiance_field(self, points, rgb, bounds=None, grid=14, dim=768, bandwidth=None, halo=1, seed=0):
        """Bake scene RADIANCE (colour leaving each surface point) into a TILED holographic field: space is split into
        a deterministic grid of bricks, each a small FPE radiance field within capacity, only occupied bricks stored
        (holographic_radiance.TiledRadianceField). `query(points) -> (rgb, coverage)` reads the kernel-weighted colour
        at any point (Nadaraya-Watson, self-normalising -- no calibration); coverage ~0 marks empty space, known from
        the field. This is the capacity answer to the single-vector wall (the HoloOctree move, for radiance): refine
        `grid` and the wall moves (measured 15.6 dB single -> 28.9 dB tiled). Changes to a region rebuild only its
        bricks (rebuild_cells -- an O(change) delta). Pairs with holographic_fog_volume so a render becomes a QUERY of
        geometry + density + radiance fields. HONEST: stores radiance (a solver bakes it), view-independent/diffuse-ish
        from one view, RBF-smooth. See holographic_radiance."""
        import numpy as _np
        from holographic_radiance import TiledRadianceField as _TRF
        P = _np.atleast_2d(_np.asarray(points, float))
        if bounds is None:
            lo = P.min(0) - 0.5; hi = P.max(0) + 0.5
            bounds = list(zip(lo.tolist(), hi.tolist()))
        bw = float(bandwidth) if bandwidth is not None else 2.2 * grid    # sharp kernel scales with grid resolution
        return _TRF(bounds, grid=grid, dim=dim, bandwidth=bw, halo=halo, seed=seed).bake(P, rgb)

    def holographic_fog_volume(self, centers, weights=None, bounds=None, dim=2048, bandwidth=1.1, seed=0):
        """Encode a volumetric DENSITY field (fog/atmosphere) as ONE hypervector via Fractional Power Encoding, and
        return a HolographicVolume whose `optical_depth(O, D, L)` integrates the density along any ray in CLOSED FORM
        -- one inner product per ray, no marching (holographic_volint). Unlike a marcher, the field is a property of
        ALL space: empty regions read ~0 optical depth without being discovered. `centers` are fog-blob positions in
        R^3; `bounds` defaults to the centres' extent padded. Use render_fog (or the returned .optical_depth) to
        composite atmospheric fog over a rendered frame using its depth buffer. Closed-form integral verified exact
        vs a marched reference; ~steps-fold faster than marching the same field. See holographic_volint."""
        import numpy as _np
        from holographic_fpe import VectorFunctionEncoder as _VFE
        from holographic_volint import HolographicVolume as _HV
        C = _np.atleast_2d(_np.asarray(centers, float))
        if bounds is None:
            lo = C.min(0) - 2.0; hi = C.max(0) + 2.0
            bounds = list(zip(lo.tolist(), hi.tolist()))
        enc = _VFE(C.shape[1], dim=dim, bounds=bounds, kernel="rbf", bandwidth=bandwidth, seed=seed)
        return _HV.from_blobs(enc, [tuple(c) for c in C], weights)

    def render_volume(self, field, camera, bounds, width=256, height=256, steps=96, mode="smoke",
                      sigma=12.0, emission_color=None, albedo=(0.9, 0.9, 0.95), lights=None,
                      background=(0.0, 0.0, 0.0)):
        """Volumetrically render a density FIELD (smoke/fire/water/particles) by marching camera rays and
        accumulating the volume integral -- vectorised over all pixels (the field IS the volume, so this is
        field-native). mode='smoke' (absorption), 'fire' (emission/blackbody ramp), 'density' (raw). Returns
        (RGB image, alpha). See holographic_render.volume_render."""
        from holographic_render import volume_render
        return volume_render(field, camera, bounds, width=width, height=height, steps=steps, mode=mode,
                             sigma=sigma, emission_color=emission_color, albedo=albedo, lights=lights,
                             background=background)

    def save_render(self, path, rgb01):
        """Write a render (an (H,W,3) image in [0,1]) to a PNG via the stdlib encoder. See holographic_render."""
        from holographic_render import save_png
        return save_png(path, rgb01)

    def render_frame_delta(self, prev, curr, tile=32, thresh=1e-3):
        """The pixel-streaming primitive: return only the `tile`x`tile` image blocks that CHANGED between two
        frames, as (row, col, pixels), plus the fraction changed -- so a viewport pushes just the dirty tiles
        after a local edit / small camera move, the rendering analogue of the engine's O(change) delta protocol.
        See holographic_render.frame_delta_tiles."""
        from holographic_render import frame_delta_tiles
        return frame_delta_tiles(prev, curr, tile=tile, thresh=thresh)

    def deform(self, geometry, kind="bend", angle=0.0, factor=0.0, axis=2, up=None, center=None):
        """Apply a classic vectorised deformer to ANY point set -- a Mesh (returns a deformed Mesh, faces kept)
        OR an (N,3) array (a particle cloud / point set, returns the deformed array). kind: 'bend' (arc in the
        (axis,up) plane by `angle` rad), 'twist' (screw by `angle` rad along `axis`), 'taper' (cone by `factor`
        along `axis`). One array op per deformer -- no Python per-point loop, so a mesh and a particle cloud run
        the same path. See holographic_deform."""
        from holographic_deform import bend, twist, taper
        from holographic_mesh import Mesh
        is_mesh = hasattr(geometry, "vertices")
        P = geometry.vertices if is_mesh else np.asarray(geometry, float)
        if kind == "bend":
            Q = bend(P, angle, axis=(0 if axis == 2 else axis), up=(2 if up is None else up), center=center)
        elif kind == "twist":
            Q = twist(P, angle, axis=axis)
        elif kind == "taper":
            Q = taper(P, factor, axis=axis)
        else:
            raise ValueError("kind must be 'bend', 'twist' or 'taper'")
        return Mesh(Q, [tuple(f) for f in geometry.faces]) if is_mesh else Q

    def lattice_deform(self, geometry, bounds, control_offsets):
        """Free-form (FFD) deformation through a control lattice: each point moves by the TRILINEAR interpolation
        of the lattice's per-control displacements `control_offsets` (nx,ny,nz,3) over `bounds`. Works on a Mesh
        (returns a Mesh) or an (N,3) array. The sculpt-by-cage deformer, vectorised (8 gathers, no point loop).
        See holographic_deform."""
        from holographic_deform import lattice_deform as _ld
        from holographic_mesh import Mesh
        is_mesh = hasattr(geometry, "vertices")
        P = geometry.vertices if is_mesh else np.asarray(geometry, float)
        Q = _ld(P, bounds, control_offsets)
        return Mesh(Q, [tuple(f) for f in geometry.faces]) if is_mesh else Q

    def blend_shapes(self, base, targets, weights):
        """Morph-target / blendshape mix as a WEIGHTED BUNDLE: base + sum_i w_i (target_i - base). `base` and each
        target are (N,3) (mesh vertices or particles); `weights` length K. Vectorised (`weights @ deltas`) -- this
        is the engine's superposition primitive on geometry, so animating the weights over time IS the blendshape
        animation. Pass a base Mesh to get a Mesh back. See holographic_deform."""
        from holographic_deform import blendshapes
        from holographic_mesh import Mesh
        is_mesh = hasattr(base, "vertices")
        B = base.vertices if is_mesh else np.asarray(base, float)
        T = [t.vertices if hasattr(t, "vertices") else np.asarray(t, float) for t in targets]
        Q = blendshapes(B, T, weights)
        return Mesh(Q, [tuple(f) for f in base.faces]) if is_mesh else Q

    def timeline(self):
        """A keyframe Timeline: `.key(channel, t, value)` then `.sample(channel, t)` for the lerp-interpolated
        value at time t (t may be an array -- vectorised). Key blendshape weights, deform params, or transforms
        and drive the animation from it. See holographic_anim.Timeline."""
        from holographic_anim import Timeline
        return Timeline()

    def frame_cache(self, base, hot=8, tol=1e-9):
        """A tiered delta FrameCache for playback: `.put(frame, state)` stores each frame as a sparse DELTA vs
        `base` (O(change) memory -- the engine's patch idea on the time axis), `.get(frame)` reconstructs it
        exactly, and the `hot` most-recent frames stay full in RAM for instant scrubbing. `.full_bytes()` /
        `.memory_bytes()` report the saving (big when frames change locally, ~full when a deformation is global).
        See holographic_anim.FrameCache."""
        from holographic_anim import FrameCache
        return FrameCache(base, hot=hot, tol=tol)

    def bake_deformation(self, base, n_frames, frame_fn):
        """Evaluate an animation into a FrameCache: for each frame f, cache frame_fn(base, f) as a delta. Returns
        the FrameCache, ready to scrub. `frame_fn` is any vectorised deformer with a time-varying parameter; the
        bake never loops in Python over vertices, only over frames. See holographic_anim.bake_deformation."""
        from holographic_anim import bake_deformation
        return bake_deformation(base, n_frames, frame_fn)

    def mirror_mesh(self, mesh, axis=0, plane=0.0, weld=True, tol=1e-5):
        """Mirror a mesh across the `axis`=const `plane`: append a reflected copy with reversed winding (normals
        stay consistent) and optionally WELD the seam -- the standard way to model a symmetric object from one
        half. Vectorised. See holographic_meshtools.mirror."""
        from holographic_meshtools import mirror
        return mirror(mesh, axis=axis, plane=plane, weld=weld, tol=tol)

    def weld_mesh(self, mesh, tol=1e-5):
        """Merge-by-distance: weld vertices within `tol` into one (mean position), remap faces, drop the faces
        that collapse -- the cleanup after a mirror / import / boolean. Vectorised for triangle meshes. See
        holographic_meshtools.merge_by_distance."""
        from holographic_meshtools import merge_by_distance
        return merge_by_distance(mesh, tol=tol)

    def solidify_mesh(self, mesh, thickness, flip=False):
        """Give a surface thickness (the shell / solidify modifier): offset an inner copy along the vertex
        normals by `thickness`, reverse its winding, and BRIDGE the boundary so an open sheet becomes a
        watertight solid (a closed mesh becomes a hollow double wall). Vectorised offset; the bridge loops over
        boundary edges only. See holographic_meshtools.solidify."""
        from holographic_meshtools import solidify
        return solidify(mesh, thickness, flip=flip)

    def render_sdf(self, sdf, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                   base_color=(0.85, 0.5, 0.35), sky=None, ao=True, shadows=True, reflect=0.25,
                   refract=0.0, ior=1.5, sss=0.0, sss_color=(1.0, 0.4, 0.3), ambient=0.25):
        """Field-native SDF renderer: sphere-trace primary rays, then shade hits with Lambert direct light gated
        by a SOFT SHADOW, ambient gated by AMBIENT OCCLUSION, an HDRI-sky environment REFLECTION (Schlick
        fresnel), optional REFRACTION (the sky bent through the surface) and SUBSURFACE glow; misses show the sky
        dome. `sky` may be an equirectangular HDRI (H,W,3) array. These are light-transport effects that fall out
        cheaply because the engine is SDF-native (the field answers nearest-surface / occlusion / normal). All
        vectorised over pixels. See holographic_raymarch.render_sdf."""
        from holographic_raymarch import render_sdf
        if hasattr(base_color, "base_color"):
            base_color = base_color.base_color[:3]
        return render_sdf(sdf, camera, width=width, height=height, light_dir=light_dir, base_color=base_color,
                          sky=sky, ao=ao, shadows=shadows, reflect=reflect, refract=refract, ior=ior, sss=sss,
                          sss_color=sss_color, ambient=ambient)

    def ambient_occlusion(self, sdf, points, normals, samples=6, step=0.06, k=1.6):
        """SDF ambient occlusion at `points` with `normals`: march the normal and read the field -- a near
        surface darkens the point. Field-native, no hemisphere rays. See holographic_raymarch.ambient_occlusion."""
        from holographic_raymarch import ambient_occlusion
        return ambient_occlusion(sdf, points, normals, samples=samples, step=step, k=k)

    def soft_shadow(self, sdf, points, light_dir, k=12.0):
        """SDF soft shadow: march each point toward the light; the closest approach to any surface is the
        penumbra (0 blocked .. 1 clear). Field-native. See holographic_raymarch.soft_shadow."""
        from holographic_raymarch import soft_shadow
        return soft_shadow(sdf, points, light_dir, k=k)

    def sky_dome(self, directions, sun_dir=(-0.4, 0.7, -0.3), env=None):
        """HDRI sky dome: the environment radiance from `directions`. With `env` (an equirectangular (H,W,3)
        image) it samples a real HDRI by lon/lat; otherwise a procedural sky+sun+ground. The incoming light is a
        superposition (bundle) of directional radiance. See holographic_raymarch.sky_dome."""
        from holographic_raymarch import sky_dome
        return sky_dome(directions, sun_dir=sun_dir, env=env)

    def refract(self, directions, normals, ior=1.5):
        """Snell's-law refraction of rays at a surface (total-internal-reflection falls back to reflection).
        This is optics -- plain vector math -- exposed as a composable helper. See holographic_raymarch.refract_dir."""
        from holographic_raymarch import refract_dir
        return refract_dir(directions, normals, ior=ior)

    def subsurface(self, sdf, points, normals, light_dir, depth=0.6, sigma=4.0):
        """Field-native subsurface translucency: measure how much SOLID the light crosses inside the object to
        reach each point (the SDF interior); thin regions transmit more and glow. See holographic_raymarch.subsurface."""
        from holographic_raymarch import subsurface
        return subsurface(sdf, points, normals, light_dir, depth=depth, sigma=sigma)

    def irradiance_cache(self, sdf, points, normals, light_dir, base_color=(0.8, 0.6, 0.5),
                         n_cache=64, n_dirs=16, seed=0):
        """Global illumination via a sparse IRRADIANCE CACHE (Ward = the engine's adaptive-anchor idea): compute
        one-bounce indirect light at `n_cache` surface points (the slow integral, paid sparsely), returning a
        cache to read with `read_irradiance`. Indirect light is smooth, so a sparse cache reconstructs it cheaply.
        See holographic_globalillum.irradiance_cache."""
        from holographic_globalillum import irradiance_cache
        return irradiance_cache(sdf, points, normals, light_dir, base_color=base_color,
                                n_cache=n_cache, n_dirs=n_dirs, seed=seed)

    def read_irradiance(self, cache, query_points, k=4):
        """Read an irradiance cache at query points by inverse-distance interpolation of the k nearest cached
        samples (the GI cache read). See holographic_globalillum.read_cache."""
        from holographic_globalillum import read_cache
        return read_cache(cache, query_points, k=k)

    def caustics(self, sdf, light_dir=(0, -1, 0), receiver_y=-0.9, extent=2.0, res=128, ior=1.5, n_side=200):
        """Caustics by forward light tracing: shoot parallel light rays, refract them through the object, and
        SPLAT where they land on the receiver plane with np.add.at -- the scatter that is the engine's bundle.
        Where refracted rays converge the bundle piles up: the caustic. Returns a (res,res) intensity map.
        See holographic_globalillum.caustics."""
        from holographic_globalillum import caustics
        return caustics(sdf, light_dir=light_dir, receiver_y=receiver_y, extent=extent, res=res, ior=ior, n_side=n_side)

    def morph_scene(self, img_a, img_b, steps=9, method="dct", post=None):
        """Morph between two images. method='dct' (default) blends in the DCT-coefficient domain (structure
        slerp, not a ghosting crossfade). method='phase' (C2) blends in the 2-D FFT domain, interpolating each
        bin's magnitude and PHASE separately -- the phase-vocoder move (holographic_phasemorph.morph_image_phase):
        by the Fourier shift theorem a translation is a phase ramp, so the phase morph SLIDES a translated feature
        to its intermediate position (a compact moving blob) where the DCT slerp interpolates its SHAPE and smears
        it. Measured win for SMALL displacements; the kept BOUND is that a large translation wraps the phase ramp
        (bin phase differences exceed pi) and falls back to a crossfade -- so 'phase' is for small-motion morphs,
        'dct' for arbitrary structure change. With `post` (a holographic_postfx.PostChain) each output frame is run
        through the post-processing pipeline -- generate-and-polish in one call (the same polish post_process applies
        to any image; here it is wired at the generation site). Part of this mind's generative repertoire."""
        if method == "phase":
            from holographic_phasemorph import morph_image_phase
            frames = morph_image_phase(np.asarray(img_a, float), np.asarray(img_b, float), steps=steps)
        else:
            from holographic_archive import HolographicArchive
            from holographic_generate import morph_images
            S = img_a.shape[0]
            arch = HolographicArchive(shape=img_a.shape, capacity=2,
                                      keep=min(900, (S * S) // 2), dim=32768, seed=self.seed)
            frames = morph_images(arch.M, img_a, img_b, steps=steps)
        if post is not None:                                  # polish each generated frame (post-fx pipeline)
            frames = [post.apply(np.clip(np.asarray(f, float), 0.0, 1.0)) for f in frames]
        return frames

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

    def learn_plan(self, name, steps, chunk=0):
        """Store an ORDERED plan/recipe/protocol by name. Unlike absorb (which
        files things order-free for classification and recall), this keeps the
        SEQUENCE queryable: meaning that lives in the order is preserved.

        For a LONG plan (a scientist's many-step protocol, a long itinerary), pass
        `chunk` (e.g. 14): the plan is stored as positional blocks so step_at /
        precedes / validate_plan stay EXACT past the single-bundle cap (the positional
        encoding alone caps with length -- ~100% to length ~50-100, decaying to ~15%
        by 800 at dim 2048). chunk=0 (default) is the original storage, ideal for short
        plans where chunking is a no-op."""
        self._seq_mem().add(name, steps, chunk=chunk)
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

    def gradient_cache_symbolic(self, expr, anchors, variables=("x", "y", "z")):
        """Build an irradiance/GI-style GradientCache with EXACT Jacobians from a symbolic field (SymPy) instead of
        finite differences -- no truncation error in the cached gradients, so first-order interpolation is more
        accurate at the same anchors. Needs sympy. See holographic_cache.gradient_cache_symbolic."""
        from holographic_cache import gradient_cache_symbolic
        return gradient_cache_symbolic(expr, anchors, variables)

    def render_sdf_fast(self, expr, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                        base_color=(0.85, 0.5, 0.35), ao=True, shadows=True, ambient=0.25, sky=None):
        """Render an analytic SDF (given as a symbolic expression) with the fully-JIT'd renderer: the whole march --
        primary ray, exact normal, AO, soft shadow -- compiles into one njit kernel (the closure barrier is gone),
        ~9-15x the numpy renderer for the field-native shading. Compiled renderer cached per SDF. Needs sympy+numba;
        falls back is the caller's (use render_sdf without jit_expr). See holographic_sdf_render.render_analytic."""
        from holographic_sdf_render import render_analytic
        return render_analytic(expr, camera, width=width, height=height, light_dir=light_dir,
                               base_color=base_color, ao=ao, shadows=shadows, ambient=ambient, sky=sky)

    def compiled_sdf_numba(self, expr, variables=("x", "y", "z")):
        """SymPy -> Numba, cached: compile a symbolic 3-D SDF to njit scalar+grid value/normal kernels ONCE and
        reuse them. The scalar njit SDF composes into other njit loops (a sphere-trace march) -- the closure barrier
        that blocked Numba from the raymarch is gone. Needs sympy + numba. See holographic_compile.compiled_sdf_numba."""
        from holographic_compile import compiled_sdf_numba
        return compiled_sdf_numba(expr, variables)

    def compile_program(self, program):
        """Assemble a HoloMachine program (list of (opcode, operand)) into its program vector ONCE via the compile
        cache and reuse it -- re-running the SAME program skips the ~L-bind assembly (measured ~15 ms / 60 instr).
        Returns the cached program vector. See holographic_compile.compiled_program."""
        from holographic_compile import compiled_program
        return compiled_program(self._machine(), program)

    def compiled_sdf_normal(self, expr, variables=("x", "y", "z")):
        """Compile a symbolic SDF's exact normal ONCE and reuse it via the content-addressed compile cache: the
        same expr returns the cached (value_fn, normal_fn) instantly instead of re-running the ~140-390 ms sympy
        lambdify, and recompiles only when the expr changes. The runtime use of the codegen pipeline -- compile a
        spec, cache the compiled version, hand it out everywhere. See holographic_compile.compiled_sdf_normal."""
        from holographic_compile import compiled_sdf_normal
        return compiled_sdf_normal(expr, variables)

    def compile_cache_stats(self):
        """Stats for the process-wide compile cache: hits/misses/compiles/evictions, size, hit_rate. See
        holographic_compile.DEFAULT_CACHE."""
        from holographic_compile import DEFAULT_CACHE
        return dict(DEFAULT_CACHE.stats, size=len(DEFAULT_CACHE), hit_rate=round(DEFAULT_CACHE.hit_rate(), 3))

    def exact_sdf_normal(self, expr, variables=("x", "y", "z")):
        """Derive an EXACT SDF surface normal from a symbolic SDF expression (SymPy, design-time) and return
        (value_fn, normal_fn) of pure NumPy -- no finite-difference step-size error, no autodiff. The Quilez-seat
        path: e.g. exact_sdf_normal('sqrt(x**2+y**2+z**2)-1.0'). Needs sympy (requirements-accel.txt); the returned
        functions are pure NumPy. See holographic_codegen.sdf_normal_fn."""
        from holographic_codegen import sdf_normal_fn
        return sdf_normal_fn(expr, variables)

    def symbolic_gradient(self, expr, variables):
        """Exact gradient of a symbolic scalar field as a pure-NumPy function (force = -symbolic_gradient(energy)).
        The Baker-seat analytic-force path, autodiff-free. See holographic_codegen.gradient_fn."""
        from holographic_codegen import gradient_fn
        return gradient_fn(expr, variables)

    def fft_backend(self, use_pyfftw=None):
        """Report or switch the FFT backend behind bind/bundle. Default 'numpy' is bit-exact and deterministic;
        pass use_pyfftw=True to opt into pyFFTW (MEASURED to regress at typical dims -- see fft_benchmark; off by
        default for good reason). Returns the active backend name."""
        from holographic_fft import use_pyfftw as _u, fft_backend as _b
        if use_pyfftw is not None:
            return _u(bool(use_pyfftw))
        return _b()

    def fft_benchmark(self):
        """Reproduce the numpy-vs-pyFFTW comparison (ratios <1 mean pyFFTW is slower). Documents why numpy stays
        the default. See holographic_fft.benchmark."""
        from holographic_fft import benchmark
        return benchmark()

    def signed_distance_field(self, inside_mask, h=1.0):
        """Occupancy/inside mask -> signed distance field (negative inside, positive outside) via the fast-sweeping
        eikonal solver, Numba-accelerated when numba is installed (measured ~270x on a 256^2 grid, bit-identical to
        the pure path) and pure-Python otherwise. The occupancy->SDF step the modelling/raymarch pipeline wants.
        See holographic_jit.signed_distance_2d."""
        from holographic_jit import signed_distance_2d
        return signed_distance_2d(inside_mask, h=h)

    def parse_scene_description(self, text):
        """Parse a controlled 3-D scene DESCRIPTION into objects + environment (holographic_semantic). Each object is
        {shape, color, material, size, relation}. Controlled vocabulary + keyword grammar, deterministic -- NOT a
        general language model (see the module SCOPE note)."""
        from holographic_semantic import parse_description
        return parse_description(text)

    def build_scene(self, text):
        """DESCRIBE a scene in plain words and get back a LIVE, adjustable scene the system built for you -- a
        SemanticScene of NAMED objects plus the environment. Then talk to it: scene.adjust('make the sphere bigger'),
        scene.render(), scene.simulate(), scene.describe(). The one-call 'describe it and let the engine create it'
        entry point; controlled-vocabulary + deterministic (see holographic_scene_semantic)."""
        from holographic_scene_semantic import scene_from_description
        return scene_from_description(text, mind=self)

    def semantic_scene(self, objects, environment=None):
        """Wrap an EXISTING list of scene objects ({shape,color,material,size,...}) as a SemanticScene so you can
        reference its named objects and adjust them in words -- scene.set('the red sphere', material='glass'),
        scene.adjust('make everything matte'). For starting from a scene you already have rather than a description.
        See holographic_scene_semantic.SemanticScene."""
        from holographic_scene_semantic import SemanticScene
        return SemanticScene(objects, environment=environment, mind=self)

    # ---- composable texture map graph (CMP1) --------------------------------------------------------------
    def texture_leaf(self, source=None, value=None, **kw):
        """Make a LEAF for a texture graph. Give `source=` a Texture name ('fbm','voronoi','synth',...) to wrap that
        field, e.g. mind.texture_leaf('fbm', n_dims=2); or give `value=` a number or a length-3/4 colour for a
        constant, e.g. mind.texture_leaf(value=[1,0,0]). Leaves are the inputs you feed to texture_op. See
        holographic_texturegraph."""
        from holographic_texturegraph import field_leaf, Const
        if source is not None:
            return field_leaf(source, **kw)
        if value is not None:
            return Const(value)
        raise ValueError("texture_leaf needs source= (a Texture name) or value= (a number/colour)")

    def texture_op(self, op, **inputs):
        """Compose a texture-map NODE: an op ('mix','multiply','over','scale','add','remap','min','max','clamp',
        'saturate') over TYPED inputs, each of which may be a leaf OR another texture_op -- so graphs nest to any
        depth. The input types are checked HERE, at compose time, so a bad graph (a colour used as a weight, a missing
        input, an unknown op) is refused up front with a clear message. Sample it with mind.sample_texture(node, uv);
        wrap it in 'saturate' to keep colours in [0,1]. See holographic_texturegraph."""
        from holographic_texturegraph import Map
        return Map(op, **inputs)

    def sample_texture(self, node, uv):
        """Sample a texture graph at a UV/point -> a value (a scalar, or an rgb colour if the graph produces one).
        Walks the tree: evaluate each child, then apply the op."""
        return node.sample(uv)

    def encode_texture(self, node):
        """The texture graph as ONE hypervector (this mind's dim/seed), for CACHING a baked result by its graph
        identity or SEARCHING a library of graphs -- structurally identical graphs encode identically. The readable
        object tree stays the source of truth; this is the derived vector form."""
        from holographic_texturegraph import encode
        return encode(node, self.dim, self.seed)

    # ---- multi-material blended/selected by masks (CMP3) --------------------------------------------------
    def multi_material(self, materials, weights, mode="blend", normalize=True):
        """Combine N Materials by per-point MASKS -- generalises Material.blend (2-way, one scalar) to N materials
        each weighted by a mask that varies over the surface. Each weight is a CMP1 texture graph (mind.texture_op),
        a field, or a constant; at a point a channel reads sum_i w_i(uv) * material_i.sample(channel, uv). mode='blend'
        is a soft weighted sum (weights normalised to a partition of unity so brightness stays put -- the kept
        negative); mode='select' hard-picks the dominant material (a material-ID / splat map). Returns a MultiMaterial
        you sample with .sample(channel, uv) / .sample_all(uv). See holographic_multimaterial."""
        from holographic_multimaterial import MultiMaterial
        return MultiMaterial(materials, weights, mode=mode, normalize=normalize)

    # ---- layered materials with an order schema (CMP2) ---------------------------------------------------
    def material_layer(self, kind, material, alpha=1.0):
        """Make one LAYER of a layered material: a `kind` ('base','diffuse','specular','reflection','coat','clearcoat')
        that fixes its place in the order, the Material carrying its channels, and a coverage `alpha` (how much shows
        through what's below -- a number, a field, or a CMP1 texture graph, so coverage can vary over the surface).
        Feed these to mind.layered_material. See holographic_layeredmaterial."""
        from holographic_layeredmaterial import Layer
        return Layer(kind, material, alpha=alpha)

    def layered_material(self, layers):
        """Stack material LAYERS bottom-to-top with the ORDER enforced: base < diffuse < specular/reflection <
        coat/clearcoat, so you cannot put a reflection under a diffuse -- an out-of-order stack is refused at COMPOSE
        time with a clear message, not rendered wrong. Each layer composites OVER the one below by its coverage alpha.
        Returns a LayeredMaterial you sample with .sample(channel, uv). Honest boundary: this fixes the STACKING, not
        the radiometry -- a physically energy-conserving layered BRDF is a separate, harder thing. See
        holographic_layeredmaterial."""
        from holographic_layeredmaterial import LayeredMaterial
        return LayeredMaterial(layers)

    # ---- type-correct binding + shared-definition instancing (CMP4) --------------------------------------
    def shared_definition(self, name, geometry, material, geometry_kind=None):
        """A shared, editable scene DEFINITION -- geometry bound to a material, with the binding TYPE-CHECKED at
        compose time: a surface material (paint/metal/glass) only binds to a mesh, a volumetric material (fog/smoke/
        fire) only to a volume; a bad binding is refused with a clear message. Place it many times in an
        instanced_scene and edit it ONCE (def.set_material / set_geometry) to update every instance. Returns a
        Definition. See holographic_instancing."""
        from holographic_instancing import Definition
        return Definition(name, geometry, material, geometry_kind=geometry_kind)

    def instanced_scene(self):
        """An empty INSTANCED scene: place shared Definitions through transforms (scene.place(defn, transform)) so many
        instances share one definition -- editing the definition updates them all (edit-once). flatten_surface()
        materialises the surface instances into one mesh. Honest boundary: sharing is edit-once at the graph level;
        flatten is where instances become concrete geometry. See holographic_instancing."""
        from holographic_instancing import InstancedScene
        return InstancedScene()

    # ---- pipeline composes the graphs: bake vs live (CMP5) -----------------------------------------------
    def render_graph(self, res=64):
        """An ORCHESTRATOR that prepares the CMP1-CMP4 graphs into a render-ready scene as pipeline stages. Register
        texture graphs (rg.add_texture(name, graph, static=True/False)) and a CMP4 instanced scene (rg.set_scene(...));
        rg.plan() shows what it will do and WHY (bake each STATIC texture to a grid for O(1) lookup, keep dynamic ones
        live, then bind+flatten the scene); rg.prepare() runs it and returns a PreparedScene. This is where 'adaptive'
        reaches down to the maps. See holographic_rendergraph."""
        from holographic_rendergraph import RenderGraph
        return RenderGraph(res=res)

    def bake_texture(self, graph, res=64, lo=0.0, hi=1.0):
        """Bake a CMP1 texture graph to a res x res grid -> a BakedTexture you sample in O(1) (bilinear lookup),
        instead of walking the graph every hit. Do this for a STATIC map sampled many times; the trade is memory +
        interpolation error (blurs detail finer than a cell -- raise res, or keep sharp maps live). See
        holographic_rendergraph."""
        from holographic_rendergraph import bake_texture
        return bake_texture(graph, res=res, lo=lo, hi=hi)

    # ---- SEE what you composed: previews for the composability stack -------------------------------------
    def preview_texture(self, graph, res=256):
        """Render a CMP1 texture graph as a flat RGB SWATCH -- a (res,res,3) float image in [0,1] you can save/view.
        Colour graph -> its rgb; scalar graph -> greyscale; out-of-range values are clamped for display. The missing
        step between 'I composed a texture' and 'let me look at it'. See holographic_preview."""
        from holographic_preview import texture_image
        return texture_image(graph, res=res)

    def preview_material(self, material, res=192, base_color=(0.82, 0.80, 0.78)):
        """Render a material on a preview SPHERE -- the classic MATERIAL BALL. Works on a plain Material or a CMP2/CMP3
        layered/multi material: uses its roughness/metallic channels (else defaults), modulates base_color by an albedo
        channel if present, and shades with the same Cook-Torrance BRDF the real renderer uses. Returns a (res,res,3)
        float image in [0,1]. See holographic_preview."""
        from holographic_preview import material_ball
        return material_ball(material, res=res, base_color=base_color)

    def render_textured(self, scene, textures, width=256, height=192, **kw):
        """Render a SemanticScene with COMPOSED textures/materials painted onto its objects -- the composability stack
        driving a FULL 3-D render, not just a swatch. `textures` maps an object NAME (from scene.names()) to a CMP1
        texture graph or a Material; the texture is UV-wrapped onto that object's surface (spherical map on a sphere,
        planar on a box) and shaded with the real Cook-Torrance BRDF + a light + a hard shadow. Objects with no entry
        keep their scene colour. Returns an (H,W,3) image in [0,1]. Honest limits: textbook UV mapping (seams/pole
        pinch), single hard light (no GI). See holographic_texturerender."""
        from holographic_texturerender import render_textured
        return render_textured(scene, textures, width=width, height=height, **kw)

    # ---- message bus + optional agent (LLM) bridge: person and agent both connected -----------------------
    def bus(self):
        """The mind's MessageBus -- one shared bus for this mind that the app, the person, and an agent all talk
        through (publish/subscribe/mailboxes/history). Created on first use. See holographic_bus."""
        if getattr(self, "_bus", None) is None:
            from holographic_bus import MessageBus
            self._bus = MessageBus()
        return self._bus

    def agent_bridge(self, llm=None, name="agent"):
        """Connect an OPTIONAL agent to this mind's bus. `llm` is any callable text->reply (your wrapper around any
        model -- nothing here depends on an LLM library, so it's fully optional and leCore runs with no agent). Use
        bridge.notify_on('render.done', 'does it look right?') so the app REACHES the agent when a job finishes, and
        bridge.ask('...') to ask it directly. See holographic_agent_bridge."""
        from holographic_agent_bridge import AgentBridge
        return AgentBridge(self.bus(), llm=llm, name=name)

    def run_task(self, name, fn, *args, background=False, summarize=None, **kwargs):
        """Run `fn(...)` as a named task on this mind's bus: publishes '<name>.start' then '<name>.done' (with a small
        summary an agent can read) or '<name>.error'. background=True runs it off-thread and returns immediately -- the
        bus's '<name>.done' is how everyone (including a watching agent) learns it finished, so nobody has to poll. See
        holographic_agent_bridge.run_task."""
        from holographic_agent_bridge import run_task
        return run_task(self.bus(), name, fn, *args, background=background, summarize=summarize, **kwargs)

    # ---- OPTIONAL semantic index over the dictionary: find words by meaning -------------------------------
    def build_semantic_index(self, words=None, dim=256, seed=0, include_synonyms=True, max_words=None):
        """Build an OPT-IN semantic index over the dictionary so you can find words by MEANING (idx.find('unexpected
        good luck') -> 'serendipity'). Nothing loads until you call this -- a user who just wants the library pays
        nothing for it. `words` scopes the vocabulary (default all ~144k, ~150 MB at dim=256 -- pass a list or
        max_words to keep it small). Approximate (random indexing over glosses) -- great for the top hit, noisy in the
        tail. See holographic_word_index."""
        from holographic_word_index import build_semantic_index
        return build_semantic_index(words=words, dim=dim, seed=seed,
                                    include_synonyms=include_synonyms, max_words=max_words)

    # ---- external asset paths: track files, repair them when they move --------------------------------------
    def asset_library(self):
        """A fresh AssetLibrary for tracking the EXTERNAL files a scene depends on (textures, models, ...) and repairing
        their paths when they move: add() them, ask which are missing(), then relink(one, new_path) to re-find the rest
        by the moved-parent + structural search, changed() to spot on-disk edits, and resolve(..., roots=) to find a
        file by content hash across machines. See holographic_assets."""
        from holographic_assets import AssetLibrary
        return AssetLibrary()

    def ingest_files(self, source, extract_to=None, with_hash=True, index_text=True, max_text_bytes=1_000_000):
        """Digest a FOLDER, a .zip, or a single file into a queryable FILE MAP: mind.ingest_files('project/') or
        mind.ingest_files('bundle.zip'). The returned FileMap lets you query by name (find('*.png')), kind
        (by_kind('model')), metadata, text content (search_text('shader normal')), and -- after build_meaning_index() --
        meaning (find_by_meaning('lighting')); inspect its tree(); and repair paths when files move via its built-in
        AssetLibrary (missing/changed/relink/resolve_assets). See holographic_filemap."""
        from holographic_filemap import ingest
        return ingest(source, extract_to=extract_to, with_hash=with_hash, index_text=index_text,
                      max_text_bytes=max_text_bytes)

    # ---- cold storage: compress INACTIVE structures, inflate on demand -------------------------------------
    def cold_store(self, keep_warm=8, codec="zlib", spill_dir=None):
        """A keyed store that bounds memory: keeps at most `keep_warm` values live and compresses the rest, warming any
        of them transparently on get(). Park inactive tables/arrays/databases here. codec='lzma' packs smaller (slower);
        spill_dir writes cold blobs to disk to free RAM entirely. See holographic_coldstore."""
        from holographic_coldstore import ColdStore
        return ColdStore(keep_warm=keep_warm, codec=codec, spill_dir=spill_dir)

    def cool(self, value, codec="zlib", spill_dir=None):
        """Wrap ONE value so it can be folded up (compressed) when idle and inflated on demand: c = mind.cool(big_table);
        c.cool() frees its RAM, c.get() brings it back bit-identical. See holographic_coldstore.Cold."""
        from holographic_coldstore import Cold
        return Cold(value, codec=codec, spill_dir=spill_dir)

    # ---- import the file formats artists hand you: OBJ/MTL, glTF/GLB, texture sets, volumetric grids -------
    def import_asset(self, path):
        """Import an artist file by extension: .obj (+its .mtl) or .glb/.gltf -> a LoadedMesh (geometry + PBR
        materials + textures); a volumetric .npy/.raw -> (GridField, bounds) for render_volume. For a Substance 3D
        Painter export, point load_texture_set at the folder. See holographic_assetimport."""
        from holographic_assetimport import import_asset
        return import_asset(path)

    def load_obj(self, path):
        """Load a Wavefront .obj and its .mtl into a LoadedMesh (positions, UVs, normals, per-face material, and the
        materials with their map_* textures)."""
        from holographic_assetimport import load_obj
        return load_obj(path)

    def load_glb(self, path):
        """Load a .glb/.gltf into a LoadedMesh with its PBR materials (base colour / metallic-roughness / normal /
        emissive) and embedded textures."""
        from holographic_assetimport import load_glb
        return load_glb(path)

    def load_texture_set(self, folder, name=None):
        """Build one PBRMaterial from a folder of maps exported by Adobe Substance 3D Painter (or any tool):
        basecolor/roughness/metallic/normal/height/ao/emissive matched by file name. Reads the exported maps (the
        .spp/.sbsar project files are proprietary)."""
        from holographic_assetimport import load_texture_set
        return load_texture_set(folder, name=name)

    def load_volume(self, path, dims=None, dtype="float32", bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))):
        """Load a 3-D density grid (.npy, or raw floats + dims) into (GridField, bounds) you can hand to
        render_volume. OpenVDB .vdb is proprietary/sparse -- export a dense grid instead."""
        from holographic_assetimport import load_volume
        return load_volume(path, dims=dims, dtype=dtype, bounds=bounds)

    def deform_mesh(self, loaded, clip=None, t=0.0):
        """DEFORM an imported rig at time t: morph-blend the base shape (if it has blend shapes) then apply
        linear-blend skinning (if it has a skeleton), returning a deformed Mesh with the vertices moved. Pass one of
        loaded.animations as `clip` to pose it; clip=None gives the rest pose. This is what makes a loaded glTF
        actually move. See holographic_skindeform."""
        from holographic_skindeform import deform
        return deform(loaded, clip=clip, t=t)

    def encode_scene(self, objects):
        """Encode parsed objects into ONE composable scene hypervector: superpose bind(OBJ_i, record_i). Returns
        (scene_vector, [record_vectors], [role_atoms]). Query it back by slot with query_scene_slot -- the scene is
        content-addressable, every attribute recoverable through the superposition by cleanup."""
        from holographic_semantic import encode_scene
        return encode_scene(objects, self)

    def query_scene_slot(self, scene_vector, roles, slot):
        """Read object `slot` back OUT of the bundled scene hypervector (unbind its role, then decode each attribute
        against its codebook). The bidirectional semantic read."""
        from holographic_semantic import query_scene
        return query_scene(scene_vector, roles, self, slot)

    def render_scene_description(self, text, camera, width=256, height=256, post=None, quality="fast", spp=24,
                                 adaptive_spp=0, bake=None, relax=1.0):
        """The full text -> 3-D pipeline in one call: parse the description and render. quality='fast' uses the
        single-pass adaptive-AA renderer (seconds; inter-object shadows, see-through glass, volumetric fog/smoke/
        fire). quality='hyperreal' routes through the Monte-Carlo PATH TRACER with real Cook-Torrance/GGX materials
        (true global illumination, colour bleeding, emissive objects that light the scene, REFRACTIVE glass) --
        offline, spp-controlled. `adaptive_spp`>0 enables variance-driven adaptive sampling (extra samples only on
        noisy pixels). `post` is an optional holographic_postfx.PostChain. `bake`=grid-resolution precomputes the SDF to
        a grid so the shader samples it O(1) (fast on complex/animated scenes; see bake_sdf); `relax`>1 turns on opt-in
        over-relaxed marching (faster on grazing scenes, a small quality trade). The fast path always uses the free,
        bit-exact active-only marcher."""
        from holographic_semantic import parse_description, render_scene, render_scene_pbr
        scene = parse_description(text)
        env = scene["environment"]
        sun = env.get("sun") or "bright"; sky = env.get("sky") or "clear"
        if quality == "hyperreal":
            return render_scene_pbr(scene["objects"], camera, width=width, height=height, spp=spp,
                                    post=post, sun=sun, sky=sky, adaptive_spp=adaptive_spp)
        return render_scene(scene["objects"], camera, width=width, height=height, post=post,
                            sun=sun, sky=sky, bake=bake, relax=relax)

    def scene_control_spec(self, command):
        """Turn a control phrase ('control the ball size and how metallic it is') into UI control descriptors
        (sliders / selects) a front-end can render directly -- the engine emits the spec, the browser draws the
        widgets. See holographic_semantic.control_spec."""
        from holographic_semantic import control_spec
        return control_spec(command)

    def post_process(self, image, chain=None, depth=None):
        """Apply a post-processing PROGRAM (a holographic_postfx.PostChain -- an ordered, named chain of effects) to a
        rasterized (H,W,3) frame. `chain` defaults to postfx.default_chain(). The convolution family (bloom, glare,
        DOF, blur) rides the engine's FFT-convolution primitive (the same operator as bind, one dimension up); the
        tone/colour curves are plain NumPy in the same pipeline. Pass `depth` (the renderer's depth buffer) for DOF.
        See holographic_postfx."""
        from holographic_postfx import default_chain
        if chain is None:
            chain = default_chain()
        return chain.apply(image, depth=depth)

    def postfx_chain(self, *steps):
        """Build a post-processing PostChain program from (name, params) steps, e.g.
        postfx_chain(("exposure", {"ev": 0.3}), ("aces", {}), ("vignette", {"strength": 0.4})). With no steps,
        returns the default preset. See holographic_postfx.PostChain / default_chain / cinematic_chain."""
        from holographic_postfx import PostChain, default_chain
        if not steps:
            return default_chain()
        return PostChain(list(steps))

    def signed_distance_field_3d(self, inside_mask, h=1.0):
        """Occupancy VOLUME -> signed distance field via 3-D fast-sweeping eikonal (Numba ~230x on 96^3, bit-exact
        vs pure; pure-Python fallback). The 3-D twin of signed_distance_field for mesh import / sculpt volumes. See
        holographic_jit.signed_distance_3d."""
        from holographic_jit import signed_distance_3d
        return signed_distance_3d(inside_mask, h=h)

    def distance_transform(self, seed_mask, h=1.0):
        """Distance from every cell to the nearest True seed cell, via fast sweeping (same optional-Numba path).
        See holographic_jit.distance_transform."""
        from holographic_jit import distance_transform
        return distance_transform(seed_mask, h=h)

    def field_deflect(self, query, attractors, masses=None, sigma=0.5, strength=0.1):
        """Slide a query toward a local mass concentration in a field of attractors -- a soft, continuous cousin of
        cleanup (drift toward the weighted local centre of mass rather than hard-snapping to one atom). Extracted
        from leOS's gravitational lens. Returns (lensed_vector, deflection_radians, force_magnitude). See
        holographic_lens.deflect."""
        from holographic_lens import deflect
        return deflect(query, attractors, masses=masses, sigma=sigma, strength=strength)

    def detect_caustic(self, query, attractors, masses=None, sigma=0.5):
        """Routing-ambiguity (caustic) score at a query: high when the two strongest attractors pull in opposite
        directions with similar strength -- a fold/decision-boundary where a tiny move flips the winner.
        Complementary to RecallNull ('is this a match?') -- this asks 'is this AMBIGUOUS between matches?'. Returns
        (caustic_score in [0,1], n_significant_attractors). See holographic_lens.detect_caustic."""
        from holographic_lens import detect_caustic
        return detect_caustic(query, attractors, masses=masses, sigma=sigma)

    def navigate_field(self, query, attractors, masses=None, sigma=0.5, strength=0.6):
        """Climb the attractor field from a query toward an attractor (iterated, decaying-step deflection),
        reporting the strongest caustic met en route. A heuristic drift that APPROACHES an attractor, not an exact
        nearest-cluster solver. See holographic_lens.navigate."""
        from holographic_lens import navigate
        return navigate(query, attractors, masses=masses, sigma=sigma, strength=strength)

    def local_structure(self, point, cloud, k=12):
        """Classify a point by the shape of its local neighbourhood (the 'cosmic web' method extracted from leOS):
        VOID / FILAMENT (1-D thread) / WALL (2-D sheet) / NODE (dense cluster), from a local PCA of its k nearest
        neighbours, plus a continuous intrinsic_dim (participation ratio of the eigenvalues). Tells you what kind
        of structure a point lives in -- useful before denoising (project a filament point along its one
        direction), sampling (avoid voids), or summarising a cloud's geometry. holostuff already had GLOBAL
        dimension estimates (box-counting, spectral); this is the PER-POINT local type. KEPT NEGATIVE:
        high-dimensional noise inflates the apparent dimension. See holographic_cosmic.local_structure."""
        from holographic_cosmic import local_structure
        return local_structure(point, cloud, k=k)

    def classify_cloud(self, cloud, k=12):
        """Classify every point of a cloud into void/filament/wall/node and return (labels, info, summary) -- the
        summary giving the fraction of each structure type and the mean intrinsic dimension, a compact geometric
        fingerprint of the cloud. See holographic_cosmic.classify_cloud."""
        from holographic_cosmic import classify_cloud
        return classify_cloud(cloud, k=k)

    def frechet_mean(self, vectors, weights=None, max_iters=12):
        """The Frechet (Karcher) mean of unit vectors -- the geometrically-correct average on the sphere, the
        point minimizing the sum of squared geodesic distances (extracted from leOS's spherical geometry). This is
        the right operation for a class PROTOTYPE, a cluster centre, or a consolidation anchor -- distinct from
        `bundle`, which is a SUPERPOSITION (stays similar to every part) for binding records. Provably lower
        geodesic variance than a re-normalized Euclidean mean. HONEST: for well-separated/tight clusters its
        downstream edge over Euclidean-normalize is marginal; the geometry pays when distributions are genuinely
        spread or skewed. See holographic_sphere.frechet_mean."""
        from holographic_sphere import frechet_mean
        return frechet_mean(vectors, weights=weights, max_iters=max_iters)

    def parallel_transport(self, v, p, q):
        """Transport a tangent vector `v` (a 'displacement', a move from one state to another) from the tangent
        plane at `p` to the tangent plane at `q`, along the geodesic -- preserving its length and surface
        relationship (extracted from leOS). This is how a displacement measured at one point is correctly reused
        at another, which is what lets displacements be composed/compared across distant regions of the space.
        See holographic_sphere.parallel_transport."""
        from holographic_sphere import parallel_transport
        return parallel_transport(v, p, q)

    def use_gpu(self, enable=True):
        """User setting: turn the optional CuPy GPU backend on/off for the heavy array-parallel kernels (fluid
        solver, and any kernel that allocates via the backend). Returns whether the GPU is now ACTIVE (requested
        AND a CUDA device is present). HONEST: GPU is for throughput, not for the deterministic/tie-sensitive
        paths -- it matches NumPy only to a tolerance and can vary run-to-run, so the bit-exact guarantees are a
        CPU property. Falls back to NumPy silently when no GPU is available."""
        from holographic_backend import enable_gpu
        return enable_gpu(enable)

    def backend_status(self):
        """A human-readable line describing the current compute backend (GPU enabled/available/unavailable)."""
        from holographic_backend import device_report
        return device_report()

    def run_procedure_batch(self, name_or_program, init_accs, max_steps=512):
        """Run a straight-line procedure over a BATCH of accumulators (N, D) in ONE interpret pass -- the
        data-parallel form of run_procedure (the architecture sweep's vectorisation of a Python-loop hot spot).
        Decodes each instruction once and applies it to all N rows at once; ~10x faster than looping
        run_procedure per item at N=2000, matching it to machine epsilon. SCOPE: value+register programs only
        (LOAD/BIND/BUNDLE/PERMUTE/STORE/RECALL/HALT); control/host ops (IFMATCH/CALL/APPLY/...) raise -- loop
        run_procedure for those. Returns the (N, D) accumulator batch. See HoloMachine.run_batch."""
        M = self._machine()
        if isinstance(name_or_program, str):
            if name_or_program not in M.functions:
                raise KeyError(f"no procedure named {name_or_program!r} -- learn_procedure it first")
            pv = M.functions[name_or_program]
        else:
            pv = M.assemble(name_or_program)
        return M.run_batch(pv, init_accs, max_steps=max_steps)

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
        user = getattr(self, "_apply_handlers", None)   # user-registered faculties (octree/nystrom/agent/...)
        if user:                                        # a registered name overrides a built-in of the same name
            handlers.update(user)
        return handlers                                 # because a raw signal needs a signal-shaped prior.)

    def register_apply_handler(self, name, fn):
        """Make any unary acc->acc faculty callable from a HoloMachine program as `APPLY <name>` -- INCLUDING
        stateful spatial ops (an octree query, a Nystrom approximation) and agent behaviours, since `fn` is a
        closure that may capture a built octree, a fitted embedding, an Agent, a DriveSystem, etc. The name is
        registered as a faculty atom on the VM so APPLY's operand cleans to it, and the handler is merged into
        the live handler set. This is the general extension point the APPLY docstring invites: the engine's
        faculties (and your own) become PROGRAMMABLE STEPS, so a synthesised or hand-written VSA program can
        denoise, recall, query space, approximate, or act, all inline -- the bridge from 'the agent drives a
        program' to 'a program drives the engine'. Backward-compatible: built-ins are untouched; a registered
        name overrides a built-in of the same name. `fn` must be callable acc->acc."""
        if not callable(fn):
            raise TypeError("an APPLY handler must be callable as acc -> acc")
        if getattr(self, "_apply_handlers", None) is None:
            self._apply_handlers = {}
        self._apply_handlers[name] = fn
        M = self._machine()
        if name not in M.faculty_names:                 # register the faculty atom so APPLY <name> cleans
            M.faculty_names.append(name)
            M.fac_atoms[name] = M._atom(f"fac:{name}")
        return self

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
        """A human-readable one-line summary of what this mind currently HOLDS: how many memory prototypes over how
        many labels, the size of the recall index, the decision brain and its action set, and any learned sequence
        generators. Takes no arguments and returns a string -- handy for logging or a quick 'what do you know?' check.
        (For a machine-readable skill card of a method or capability, use describe_skill(name) instead.)"""
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

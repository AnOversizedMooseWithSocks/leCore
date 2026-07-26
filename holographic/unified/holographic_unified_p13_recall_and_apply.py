"""Part 13 of UnifiedMind's faculty surface -- 93 methods, recall_and_apply .. mantis_falsecolor.

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


class _UnifiedPart13:

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
        from holographic.agents_and_reasoning.holographic_ai import cosine as _cos, bind as _bind, bundle as _bundle, permute as _permute
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
        from holographic.agents_and_reasoning.holographic_machine import bind as _bind, cosine as _cos, derived_atom as _datom
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
        from holographic.agents_and_reasoning.holographic_predictive import PredictiveMemory
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

    def sample_from(self, distribution, temperature=1.0, top_p=1.0, seed_rng=None):
        """Draw one symbol from a {symbol: weight} distribution with temperature + optional nucleus (top-p) --
        the GENERAL stochastic-choice primitive (holographic_tokensample.sample_from_distribution), the same
        draw the char generator, the topic generator, and the recipe grammar all now delegate to. weights may
        be probabilities, similarities, or raw counts. temperature<1 sharpens (->argmax), >1 flattens; top_p<1
        keeps only the smallest set of top symbols reaching that mass. Returns the chosen symbol, or None if the
        distribution is empty / has no positive mass. Use this when you have a distribution in hand; use
        sample_recipe / generate for full sequences. See holographic_tokensample.sample_from_distribution."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_tokensample import sample_from_distribution
        rng = _np.random.default_rng(seed_rng) if seed_rng is not None else None
        return sample_from_distribution(distribution, temperature=temperature, top_p=top_p, rng=rng)

    def sample_instruction(self, partial, temperature=1.0, top_p=1.0, rng=None):
        """SAMPLE the next full instruction (the GENERATION dual of complete_instruction). Same learned
        joint (opcode, operand) grammar, but drawn stochastically instead of argmax -- because a greedy
        generator LIMIT-CYCLES (measured: greedy recipe generation gave MMD2 0.599 and ~15x the real
        verbatim-copy rate, looping on the single top continuation; a sampled arm gave MMD2 0.011). Use
        complete_instruction to PREDICT the next step; use this to GENERATE a diverse, distribution-faithful
        continuation. Returns (opcode, operand); (None, None) if no grammar or the context is exhausted.

        Delegates the temperature+nucleus draw to PredictiveMemory.sample -> holographic_tokensample, the
        same primitive the character generator uses. See sample_recipe for a full-sequence generator.

        KEPT NEGATIVES (measured on real physics traces): on HEAVY-TAILED token streams (one token at
        ~98% marginal), top_p<1 or T<1 silently deletes the rare events -- default T=1.0/top_p=1.0 is the
        safe setting there. And well-formedness is the CALLER's job: if the language alternates (e.g.
        run/event), enforce it in the decode loop -- 18% ill-formed emissions measurably destroyed the
        generated stream's autocorrelation until the caller enforced alternation."""
        if getattr(self, "_recipe_grammar_joint", None) is None:
            return None, None
        toks = [self._instr_token(s) for s in partial]
        tok, _ = self._recipe_grammar_joint.sample(toks, temperature=temperature, top_p=top_p, rng=rng)
        if tok is None:
            return None, None
        if "|" in tok:                                  # split the joint token back into opcode + operand
            op, operand = tok.split("|", 1)
            return op, operand
        return tok, None                                # a bare-opcode token (no operand)

    def sample_recipe(self, seed, length=16, temperature=1.0, top_p=1.0, seed_rng=0):
        """Generate a whole recipe by SAMPLING the learned grammar step by step (the non-limit-cycling
        generator). `seed` is a short list of (opcode, operand) instructions to start from. Returns the
        generated instructions (opcode, operand pairs) after the seed. This is the sampler the transform-
        codebook program needed: the grammar can now be SPOKEN, not just predicted."""
        import numpy as _np
        rng_ = _np.random.default_rng(seed_rng)
        out = list(seed)
        for _ in range(length):
            op, operand = self.sample_instruction(out[-self._recipe_grammar_joint.order:],
                                                  temperature=temperature, top_p=top_p, rng=rng_)
            if op is None:
                break
            out.append((op, operand))
        return out[len(seed):]

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
        from holographic.misc.holographic_typed import program_to_recipe
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
        from holographic.scene_and_pipeline.holographic_protocol import build_protocol, audit_protocol
        M = self._machine()
        if steps is not None:
            program, n_steps = build_protocol(M, list(steps))
        if program is None or n_steps is None:
            raise ValueError("audit_procedure needs steps=<list of step names>, or program=<vec> with "
                             "n_steps=<int>")
        return audit_protocol(M, program, n_steps)

    def selftest_coverage(self):
        """Which engine modules carry a real selftest, and which don't -- the engine's own test-coverage census,
        answerable through the mind (an above/below sweep found the CI selftest walker had no mind door, so an
        agent driving leCore over HTTP could not ask 'is the engine covered?'). Returns {runnable, missing,
        missing_modules, coverage}: `runnable` have a __main__ AND a _selftest, `missing` advertise an entry
        point but assert nothing (a false green -- and the exact backfill worklist), `coverage` is the fraction.
        Pure AST, no subprocess -- safe to call from a served mind; the actual RUN of the walk is the CLI/CI tool
        tools/run_selftests.py. See holographic_codestructure.selftest_census."""
        from holographic.io_and_interop.holographic_codestructure import selftest_census
        return selftest_census()

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
        from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder
        from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
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
            from holographic.misc.holographic_creature import HolographicMind
            m._actions = list(state["actions"])
            m._brain = HolographicMind.from_state(state["brain"])
        return m

    def save(self, path, quant="auto", compress=True):
        """Persist the learned mind to `path` (.npz) via the kernel save. The default quant='auto' picks
        the coarsest DECISION-SAFE precision per array -- and now also uses the B5 rate-distortion code
        (KLT -> quantize -> rANS, cosines preserved to 0.9999) on any LARGE low-rank float array, taken only
        when it beats int8, so low-rank state shrinks automatically with no precision risk and small arrays
        are untouched. quant='rd' forces that code wherever it helps (int8 elsewhere); 'int8'/None as in
        holographic_core.save.

        NOT TO BE CONFUSED WITH `restore()` -- that is an inverse-problem SOLVER (deblur/inpaint a degraded
        measurement), nothing to do with persistence; a downstream integrator "nearly wired the Loader to it".
        The state family is: save/save_state (write) <-> load/load_state/from_file (read, CLASSMETHODS), and
        to_state/from_state for the in-memory dict form."""
        from holographic.misc.holographic_core import save as _save
        return _save(self, path, compress=compress, quant=quant)

    @classmethod
    def load(cls, path):
        """Reload a mind saved with save() -- a CLASSMETHOD that RETURNS A NEW MIND. Use the return value:

            mind = lecore.UnifiedMind.load(path)      # right
            mind.load(path)                           # WRONG -- loads into a new object and throws it away

        THE TRAP, measured: this does NOT mutate the instance you call it on. `m = UnifiedMind(...); m.load(p)`
        looks like a restore and silently leaves `m` exactly as it was -- an empty mind that then behaves as if
        the save file were empty. A downstream integrator described doing precisely that ("the Loader constructs
        then calls load()"). It is a classmethod because rebuilding a mind means rebuilding every faculty, which
        is construction, not mutation -- so it hands you the new object instead of pretending to edit the old one.
        `from_file` is the same call under the name that says so.

        NOT TO BE CONFUSED WITH `restore()`, which is an inverse-problem solver (deblur/inpaint), not persistence.
        Dispatches through the kernel's versioned loader."""
        from holographic.misc.holographic_core import load as _load
        return _load(path)

    @classmethod
    def from_file(cls, path):
        """Construct a mind FROM a saved state file: `mind = lecore.UnifiedMind.from_file(path)`. Exactly
        `load(path)` under the name a caller looks for when they want a constructor -- an integrator asked for
        `from_file`/`state_path=` having not recognised that the classmethod `load` already IS construct-from-file.
        Additive alias: one implementation, two honest names, so neither can drift."""
        return cls.load(path)

    def save_state(self, path, quant="auto", compress=True):
        """Alias of `save(path)` -- persist this mind to a .npz. Exists because `save`/`load` sit next to
        `restore()` (an inverse-problem SOLVER) and `to_state`/`from_state` (the in-memory dict form), and an
        integrator reported nearly wiring a loader to `restore`. `save_state`/`load_state` name the STATE family
        unambiguously. Delegates; no second implementation to drift."""
        return self.save(path, quant=quant, compress=compress)

    @classmethod
    def load_state(cls, path):
        """Alias of the CLASSMETHOD `load(path)` -- RETURNS A NEW MIND; it does not mutate the instance. See
        `load` for the trap this family exists to make obvious."""
        return cls.load(path)

    def doppler_velocity(self, lambda_obs, lambda_rest, relativistic=False):
        """Line-of-sight velocity (m/s, positive = receding) from a spectral shift: classical v=c*z, or set
        relativistic=True (stays below c for any redshift). The physical reading of a shifted line. Field-native.
        See holographic_dedoppler.doppler_velocity."""
        from holographic.sampling_and_signal.holographic_dedoppler import doppler_velocity as _f
        return _f(lambda_obs, lambda_rest, relativistic=relativistic)

    def redshift(self, lambda_obs, lambda_rest):
        """Redshift z = lambda_obs/lambda_rest - 1 (positive = receding). Field-native. See
        holographic_dedoppler.redshift."""
        from holographic.sampling_and_signal.holographic_dedoppler import redshift as _f
        return _f(lambda_obs, lambda_rest)

    def doppler_shift(self, lambda_rest, velocity, relativistic=False):
        """Forward model: observed wavelength when a source at `lambda_rest` recedes at `velocity` (m/s). The exact
        inverse of doppler_velocity. See holographic_dedoppler.doppler_shift."""
        from holographic.sampling_and_signal.holographic_dedoppler import doppler_shift as _f
        return _f(lambda_rest, velocity, relativistic=relativistic)

    def drift_acceleration(self, drift_rate, freq):
        """Line-of-sight acceleration (m/s^2) from a narrowband frequency drift rate (Hz/s) at frequency `freq`:
        a = -c*(df/dt)/f. Turns a detect_drifting result into the emitter's acceleration -- the SETI reading. See
        holographic_dedoppler.drift_acceleration."""
        from holographic.sampling_and_signal.holographic_dedoppler import drift_acceleration as _f
        return _f(drift_rate, freq)

    def stokes_unpolarized(self, intensity=1.0):
        """Unpolarised light of `intensity` as a Stokes vector [I,0,0,0] -- also what a scalar radiance IS
        in Stokes terms. `intensity` may be a scalar or a field (a trailing length-4 axis is added). See
        holographic_stokes.unpolarized."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.unpolarized(intensity)

    def stokes_linear(self, intensity=1.0, angle=0.0, dop=1.0):
        """Linearly polarised light: e-vector at `angle` RADIANS, degree-of-linear-polarization `dop` in
        [0,1]. Field-native (all args broadcast). See holographic_stokes.linear."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.linear(intensity, angle, p=dop)

    def stokes_circular(self, intensity=1.0, handedness=1, dop=1.0):
        """Circularly polarised light: `handedness` +1=right / -1=left, degree-of-circular-polarization
        `dop`. The channel the mantis shrimp uniquely detects. See holographic_stokes.circular."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.circular(intensity, handedness=handedness, p=dop)

    def stokes_report(self, stokes):
        """Read a Stokes vector/field OUT in one call: {intensity, dop, dolp, docp, evector_angle,
        handedness}. All broadcast over a field. See holographic_stokes."""
        import holographic.rendering.holographic_stokes as _stk
        return {"intensity": _stk.intensity(stokes), "dop": _stk.dop(stokes),
                "dolp": _stk.dolp(stokes), "docp": _stk.docp(stokes),
                "evector_angle": _stk.evector_angle(stokes), "handedness": _stk.handedness(stokes)}

    def stokes_complex_linear(self, stokes):
        """The complex linear polarization P = Q + iU as a phasor. Sampled over wavelength^2, its FFT is
        rotation-measure synthesis (Faraday depth) -- the telescope arc reuses this. See
        holographic_stokes.complex_linear."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.complex_linear(stokes)

    def radiance_to_stokes(self, radiance):
        """Lift a scalar/RGB radiance to UNPOLARISED Stokes (S0=value, rest 0). Round-trips byte-identically
        via stokes_to_radiance -- polarization is purely additive. See holographic_stokes.from_radiance."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.from_radiance(radiance)

    def stokes_to_radiance(self, stokes):
        """Collapse a Stokes field to plain intensity S0 -- the byte-identical 'polarization off' path. See
        holographic_stokes.to_radiance."""
        import holographic.rendering.holographic_stokes as _stk
        return _stk.to_radiance(stokes)

    def mueller_matrix(self, kind, angle=0.0, delta=None, rho=0.0, factor=0.0, n1=1.0, n2=1.5, theta=0.0):
        """Build the 4x4 Mueller matrix of an optical element by `kind`: 'identity', 'polarizer' (at
        `angle`), 'retarder' (`delta` rad at `angle`), 'quarter_wave'/'half_wave' (at `angle`), 'rotator'
        (`rho` rad -- an optical / Faraday rotator), 'depolarizer' (`factor` in [0,1]), or 'fresnel'
        (dielectric reflection n1->n2 at incidence `theta`). Feed it to apply_mueller. See
        holographic_mueller."""
        import holographic.rendering.holographic_mueller as _mu
        if kind == "identity": return _mu.identity()
        if kind == "polarizer": return _mu.linear_polarizer(angle)
        if kind == "retarder": return _mu.retarder(np.pi / 2 if delta is None else delta, angle)
        if kind == "quarter_wave": return _mu.quarter_wave(angle)
        if kind == "half_wave": return _mu.half_wave(angle)
        if kind == "rotator": return _mu.rotator(rho)
        if kind == "depolarizer": return _mu.depolarizer(factor)
        if kind == "fresnel": return _mu.fresnel_reflection(n1, n2, theta)
        raise ValueError("unknown mueller element kind: %r" % (kind,))

    def apply_mueller(self, element, stokes):
        """Transform a Stokes vector/field by a Mueller matrix (or a per-pixel matrix-field). See
        holographic_mueller.apply."""
        import holographic.rendering.holographic_mueller as _mu
        return _mu.apply(element, stokes)

    def compose_mueller(self, *elements):
        """Fold a light path (elements IN THE ORDER LIGHT PASSES THROUGH) into one Mueller matrix. See
        holographic_mueller.compose."""
        import holographic.rendering.holographic_mueller as _mu
        return _mu.compose(*elements)

    def rm_synthesis(self, lambda2, phi, P=None, Q=None, U=None, weights=None, lambda2_0=None):
        """Rotation-measure synthesis: the Faraday-depth spectrum F(phi) of polarized light vs wavelength^2
        (Brentjens & de Bruyn 2005). Pass complex P (from stokes_complex_linear) or real Q and U, each
        (...,nchan); returns (...,nphi), field-native over an image cube. This is the SEQUENCE costume of the
        Stokes state -- the telescope's magnetic-field probe, reusing the engine's phasor. See
        holographic_rmsynth.rmsynth."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.rmsynth(lambda2, phi, P=P, Q=Q, U=U, weights=weights, lambda2_0=lambda2_0)

    def rmtf(self, lambda2, phi, weights=None, lambda2_0=None):
        """The rotation-measure transfer function (the 'dirty beam' in Faraday space) for a given wavelength^2
        sampling -- its width is the resolution, its sidelobes are the artefacts a raw F(phi) carries. See
        holographic_rmsynth.rmtf."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.rmtf(lambda2, phi, weights=weights, lambda2_0=lambda2_0)

    def rm_resolution(self, lambda2):
        """Faraday-depth resolution (FWHM, rad/m^2) from the wavelength^2 coverage: 2*sqrt(3)/span. Wider
        band -> finer RM separable. See holographic_rmsynth.resolution_fwhm."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.resolution_fwhm(lambda2)

    def rm_phi_grid(self, lambda2, oversample=5.0, extent=None):
        """Build a sensible grid of Faraday depths to evaluate, derived from the wavelength^2 sampling itself
        (no magic numbers to guess). See holographic_rmsynth.phi_grid."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.phi_grid(lambda2, oversample=oversample, extent=extent)

    def rm_peak(self, F, phi, lambda2_0=None):
        """Reduce a Faraday spectrum (or image cube of them) to its brightest source: {rm, polarized_intensity,
        angle0}, with sub-bin parabolic RM. Pass lambda2_0 to get the true intrinsic angle at lambda^2=0. See
        holographic_rmsynth.peak_rm."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.peak_rm(F, phi, lambda2_0=lambda2_0)

    def stokes_faraday_depth(self, lambda2, stokes, phi=None, weights=None):
        """CONVERGENCE door: from a Stokes field + its wavelength^2 axis straight to the Faraday-depth spectrum,
        in one call. Forms P = Q + iU (stokes_complex_linear) and runs rm_synthesis; if `phi` is None a grid is
        chosen from the data. This is the single step from 'polarization state' to 'line-of-sight magnetism'.
        Delegates to holographic_stokes.complex_linear + holographic_rmsynth.rmsynth."""
        import holographic.rendering.holographic_stokes as _stk
        import holographic.rendering.holographic_rmsynth as _rm
        P = _stk.complex_linear(stokes)               # Q + iU per channel, field-native
        if phi is None:
            phi = _rm.phi_grid(lambda2)
        return {"phi": phi, "F": _rm.rmsynth(lambda2, phi, P=P, weights=weights)}

    def faraday_rotate(self, stokes0, lambda2, rm):
        """FORWARD Faraday model -- the polarized sky a telescope receives: rotate an intrinsic Stokes signal
        (...,4) by rm*lambda^2 across a band (nchan) -> (...,nchan,4). Intensity and circular are untouched;
        only the linear plane turns. Field-native over a whole sky; its output is what rm_synthesis inverts. See
        holographic_rmsynth.faraday_rotate."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.faraday_rotate(stokes0, lambda2, rm)

    def faraday_rm_map(self, lambda2, stokes_cube, phi=None, weights=None):
        """TELESCOPE-AS-OBSERVER: recover a per-pixel Faraday-depth (line-of-sight magnetism) MAP from a sky
        Stokes cube (...,nchan,4), in one call. Composes rm synthesis + peak over the whole field. Returns
        {rm, polarized_intensity, angle0, phi, F}. The same polarization core that reads a mantis eye reads a
        radio dish. See holographic_rmsynth.faraday_rm_map."""
        import holographic.rendering.holographic_rmsynth as _rm
        return _rm.faraday_rm_map(lambda2, stokes_cube, phi=phi, weights=weights)

    def make_sky_axis(self, name, n=None, unit="", crval=0.0, crpix=0.0, cdelt=1.0):
        """One WCS-lite axis descriptor (world = crval + (pixel-crpix)*cdelt, crpix 0-based) for a sky cube. See
        holographic_skydata.make_axis."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.make_axis(name, n=n, unit=unit, crval=crval, crpix=crpix, cdelt=cdelt)

    def make_skydata(self, data, axes, meta=None):
        """Assemble a sky observation: a data cube + one world axis per data dim (+ freeform meta). The container
        the astro tools ingest. See holographic_skydata.make_skydata."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.make_skydata(data, axes, meta=meta)

    def sky_world_coords(self, sky, axis):
        """The world-coordinate array (RA/Dec/freq...) along one axis of a sky cube, by index or name. See
        holographic_skydata.world_coords."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.world_coords(sky, axis)

    def sky_pix_to_world(self, sky, axis, pix):
        """Pixel -> world coordinate on one sky axis (linear). See holographic_skydata.pix_to_world."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.pix_to_world(sky, axis, pix)

    def sky_world_to_pix(self, sky, axis, world):
        """World -> pixel on one sky axis (exact inverse). See holographic_skydata.world_to_pix."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.world_to_pix(sky, axis, world)

    def sky_lambda2(self, sky, axis=None):
        """The lambda^2 (m^2) vector of a cube's spectral axis -- the input Faraday RM synthesis wants (frequency
        is converted via c/f then squared). Auto-finds the spectral axis. The observation->Faraday bridge. See
        holographic_skydata.lambda2_axis."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.lambda2_axis(sky, axis=axis)

    def sky_stokes_cube(self, sky, spectral=None, stokes=None):
        """Reshape a sky observation into (...,nchan,4) -- spatial, then spectral channel, then Stokes -- ready for
        faraday_rm_map. Auto-finds the spectral and Stokes axes. See holographic_skydata.stokes_cube."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.stokes_cube(sky, spectral=spectral, stokes=stokes)

    def save_skydata(self, sky, path):
        """Persist a sky observation deterministically (JSON header + .npy cube in one .npz, no pickle). See
        holographic_skydata.save_skydata."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.save_skydata(sky, path)

    def load_skydata(self, path):
        """Load a sky observation saved by save_skydata (exact round-trip, no pickle). See
        holographic_skydata.load_skydata."""
        import holographic.io_and_interop.holographic_skydata as _sd
        return _sd.load_skydata(path)

    def star_system(self, params, seed=0):
        """PLUG DATA IN, GET A STAR SYSTEM: assemble physical parameters (star temp/radius/mass; planets a/e/radius/
        temp) into a deterministic, JSON-serializable scene RECIPE -- star (blackbody colour at the origin) + planets
        (biome by temperature, Kepler orbit, position, seed to regenerate the surface). Delegates to blackbody,
        fractal_planet, and Kepler geometry. See holographic_starsystem.star_system."""
        from holographic.scene_and_pipeline.holographic_starsystem import star_system as _f
        return _f(params, seed=seed)

    def kepler_ellipse(self, a, e, n=128):
        """A whole orbit as n (x,y) points with the star at a focus (perihelion a(1-e), aphelion a(1+e)); uniform in
        phase so points bunch near aphelion as a real planet does. See holographic_starsystem.kepler_ellipse."""
        from holographic.scene_and_pipeline.holographic_starsystem import kepler_ellipse as _f
        return _f(a, e, n=n)

    def kepler_position(self, a, e, mean_anomaly):
        """Position (x,y) on a Kepler orbit at a given phase (mean anomaly, radians), star at a focus. Field-native
        over an array of phases. See holographic_starsystem.kepler_position."""
        from holographic.scene_and_pipeline.holographic_starsystem import kepler_position as _f
        return _f(a, e, mean_anomaly)

    def temperature_to_biome(self, temp_K):
        """Map a planet's equilibrium temperature (K) to a surface regime (frozen/cold/temperate/hot/molten) -- the
        biome class its surface is painted with. A documented CHOICE of thresholds. See
        holographic_starsystem.temperature_to_biome."""
        from holographic.scene_and_pipeline.holographic_starsystem import temperature_to_biome as _f
        return _f(temp_K)

    def planet_field(self, planet_spec, dim=256):
        """Regenerate a planet's actual surface field from its star_system recipe entry, via fractal_planet (the
        world is never stored, only its seed+knobs -> reproduced on demand). See holographic_starsystem.planet_field."""
        from holographic.scene_and_pipeline.holographic_starsystem import planet_field as _f
        return _f(planet_spec, dim=dim)

    def star_cluster(self, n, seed=0, extent=1.0, density_field=None, planets_per_star=0):
        """Place `n` star systems in a field -- the UP direction of star_system (a cluster is many systems). Masses
        from a Salpeter IMF, coloured by main-sequence temperature (blue giants, red dwarfs). Even low-discrepancy
        placement, or pass a density_field (e.g. a cosmic-web map) to cluster them along structure. Deterministic
        recipe. See holographic_starsystem.star_cluster."""
        from holographic.scene_and_pipeline.holographic_starsystem import star_cluster as _f
        return _f(n, seed=seed, extent=extent, density_field=density_field, planets_per_star=planets_per_star)

    def sample_imf(self, n, seed=0, m_low=0.1, m_high=50.0, alpha=2.35):
        """Draw `n` stellar masses (solar units) from a Salpeter initial mass function (p(m)~m^-alpha, alpha=2.35).
        Bottom-heavy: mostly red dwarfs, few blue giants. Closed-form inverse-CDF. See holographic_starsystem.sample_imf."""
        from holographic.scene_and_pipeline.holographic_starsystem import sample_imf as _f
        return _f(n, seed=seed, m_low=m_low, m_high=m_high, alpha=alpha)

    def mass_to_temperature(self, mass):
        """A star's main-sequence temperature (K) from its mass (solar units): T ~ 5772*M^0.525 (1 Msun -> Sun). A
        rough MS scaling, monotonic. See holographic_starsystem.mass_to_temperature."""
        from holographic.scene_and_pipeline.holographic_starsystem import mass_to_temperature as _f
        return _f(mass)

    def nebula_volume(self, res=48, seed=0, level=0.5, gain=3.0, ridged=True, star_positions=None, cavity_radius=0.16, dim=256, octaves=5):
        """Build a 3-D NEBULA density volume (res^3, [0,1]) -- turbulent gas/dust with wispy filaments and dark
        voids, from the engine's FractalNoise. Pass star_positions to carve cavities where stars blow bubbles (ties
        to star_cluster). Feeds the volume renderer via nebula_field_fn. See holographic_nebula.nebula_volume."""
        from holographic.scene_and_pipeline.holographic_nebula import nebula_volume as _f
        return _f(res=res, seed=seed, level=level, gain=gain, ridged=ridged, star_positions=star_positions, cavity_radius=cavity_radius, dim=dim, octaves=octaves)

    def nebula_column(self, volume, axis=2):
        """Column density: sum a nebula volume along one axis -> a 2-D image (looking through the cloud), the cheap
        look without a full ray-march. See holographic_nebula.nebula_column."""
        from holographic.scene_and_pipeline.holographic_nebula import nebula_column as _f
        return _f(volume, axis=axis)

    def nebula_field_fn(self, volume, bounds=None):
        """Wrap a nebula volume as the callable points(N,3)->density the volume renderer marches (trilinear). Drop a
        nebula straight into render_volume. See holographic_nebula.nebula_field_fn."""
        from holographic.scene_and_pipeline.holographic_nebula import nebula_field_fn as _f
        return _f(volume, bounds=bounds)

    def nbody_simulate(self, positions, velocities, masses, dt, steps, G=6.674e-11, softening=0.0, record_every=0):
        """Integrate an N-BODY gravity system forward `steps` velocity-Verlet steps (symplectic -> energy stays
        bounded). Returns final positions/velocities, the honest energy DRIFT, and optionally a trajectory. Softened
        Newtonian, O(N^2). The dynamics counterpart to star_system's closed-form orbits. See holographic_nbody.nbody_simulate."""
        from holographic.simulation_and_physics.holographic_nbody import nbody_simulate as _f
        return _f(positions, velocities, masses, dt, steps, G=G, softening=softening, record_every=record_every)

    def nbody_step(self, positions, velocities, masses, dt, G=6.674e-11, softening=0.0):
        """One velocity-Verlet step of N-body gravity -> (positions, velocities, accel). See holographic_nbody.nbody_step."""
        from holographic.simulation_and_physics.holographic_nbody import nbody_step as _f
        return _f(positions, velocities, masses, dt, G=G, softening=softening)

    def nbody_accel(self, positions, masses, G=6.674e-11, softening=0.0):
        """Softened Newtonian acceleration on each body from all others (the O(N^2) direct sum). See
        holographic_nbody.nbody_accel."""
        from holographic.simulation_and_physics.holographic_nbody import nbody_accel as _f
        return _f(positions, masses, G=G, softening=softening)

    def nbody_energy(self, positions, velocities, masses, G=6.674e-11, softening=0.0):
        """Total mechanical energy KE + PE of an N-body system -- the quantity a good integrator keeps bounded. See
        holographic_nbody.nbody_energy."""
        from holographic.simulation_and_physics.holographic_nbody import nbody_energy as _f
        return _f(positions, velocities, masses, G=G, softening=softening)

    def circular_orbit_velocity(self, central_mass, radius, G=6.674e-11):
        """The speed for a circular orbit at `radius` around `central_mass`: sqrt(G*M/r). Seeds a stable orbit. See
        holographic_nbody.circular_orbit_velocity."""
        from holographic.simulation_and_physics.holographic_nbody import circular_orbit_velocity as _f
        return _f(central_mass, radius, G=G)

    def best_period(self, times, values, min_period=None, max_period=None, samples_per_peak=5.0, n_null=0, seed=0):
        """Find the PERIOD of an unevenly-sampled series (Lomb-Scargle): returns {period, frequency, power, fap}.
        fap (false-alarm probability) is filled when n_null>0. Closes the loop: a light curve -> a period -> Kepler
        -> star_system. See holographic_lombscargle.best_period."""
        from holographic.sampling_and_signal.holographic_lombscargle import best_period as _f
        return _f(times, values, min_period=min_period, max_period=max_period, samples_per_peak=samples_per_peak, n_null=n_null, seed=seed)

    def lomb_scargle(self, times, values, freqs):
        """The Lomb-Scargle normalised power at each trial frequency (cycles/time) for unevenly-sampled data --
        the periodogram a plain FFT can't do. Field-native over freqs. See holographic_lombscargle.lomb_scargle."""
        from holographic.sampling_and_signal.holographic_lombscargle import lomb_scargle as _f
        return _f(times, values, freqs)

    def lomb_scargle_auto(self, times, values, min_period=None, max_period=None, samples_per_peak=5.0):
        """Give me the periodogram of this light curve: builds the frequency grid from the data and returns
        (freqs, power). See holographic_lombscargle.lomb_scargle_auto."""
        from holographic.sampling_and_signal.holographic_lombscargle import lomb_scargle_auto as _f
        return _f(times, values, min_period=min_period, max_period=max_period, samples_per_peak=samples_per_peak)

    def phase_fold(self, times, values, period, t0=0.0):
        """Fold a series on `period` -> (phase in [0,1), values) sorted by phase. Coherent on the true period,
        scattered on a wrong one -- the way to SEE a period is right. See holographic_lombscargle.phase_fold."""
        from holographic.sampling_and_signal.holographic_lombscargle import phase_fold as _f
        return _f(times, values, period, t0=t0)

    def period_false_alarm(self, times, values, observed_power, freqs, n_null=200, seed=0):
        """The chance a peak this strong comes from the sampling window alone, via a permutation null (times fixed).
        Low => the period is real. The honesty gate on a periodicity detection. See
        holographic_lombscargle.false_alarm_probability."""
        from holographic.sampling_and_signal.holographic_lombscargle import false_alarm_probability as _f
        return _f(times, values, observed_power, freqs, n_null=n_null, seed=seed)

    def human_observer(self, samples=90):
        """The human eye as an OBSERVER (CIE 1931 colour-matching functions on 380-780 nm). Feed it a spectrum
        with observe_spectrum; blackbody_rgb is exactly this observer applied to a Planck spectrum. See
        holographic_observer.human_cie."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.human_cie(samples)

    def make_observer(self, wavelengths_nm, sensitivities, names=None):
        """Assemble a custom sensor from a wavelength grid + per-channel sensitivity curves ((nchan,nlam)).
        A human eye, a mantis eye, or a telescope bandpass are all just different channel sets. See
        holographic_observer.make_observer."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.make_observer(wavelengths_nm, sensitivities, names=names)

    def observer_receptor_bank(self, wavelengths_nm, centers_nm, widths_nm, gains=None):
        """Build a bank of Gaussian receptor sensitivities (the generic shape of biological cones) on a
        wavelength grid -- the parts a multi-band eye is made of. See holographic_observer.receptor_bank."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.receptor_bank(wavelengths_nm, centers_nm, widths_nm, gains=gains)

    def observe_spectrum(self, spectrum, observer):
        """Integrate a spectrum (sampled on the observer's wavelengths) against each channel -> readings
        (...,nchan). Field-native: a hyperspectral image (...,nlam) yields a per-pixel reading image in one
        call (the observer-over-a-field convergence). See holographic_observer.observe."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.observe(spectrum, observer)

    def spectrum_to_rgb(self, spectrum, mode="hue", samples=90):
        """What the HUMAN eye sees from a spectrum, as sRGB -- the general spectrum->colour door (blackbody_rgb
        is the special case for a Planck spectrum, reproduced byte-identically). mode 'hue' lifts chromaticity,
        'none' keeps luminance. See holographic_observer.human_rgb."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.human_rgb(spectrum, mode=mode, samples=samples)

    def xyz_to_srgb(self, xyz, mode="hue"):
        """Convert CIE XYZ readings (from the human observer) to sRGB, matching blackbody's exact conversion.
        Field-native over an image of readings. See holographic_observer.to_srgb."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.to_srgb(xyz, mode=mode)

    def mantis_receptors(self, wavelengths_nm):
        """The mantis shrimp's ~12 spectral receptors (deep UV to far red) as an observer on the given wavelength
        grid. Pass a grid reaching into the UV (e.g. 300-720 nm). See holographic_observer.mantis_receptors."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.mantis_receptors(wavelengths_nm)

    def polarization_readout(self, stokes):
        """Read polarization from a Stokes field the mantis way: linear detectors + circular detectors via a
        quarter-wave retarder (the R8 mechanism). Returns linear_*/circular_* channels, e-vector angle and
        handedness -- the last is the circular-polarization sense the mantis uniquely sees. See
        holographic_observer.polarization_readout."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.polarization_readout(stokes)

    def mantis_view(self, spectral_stokes, wavelengths_nm):
        """See a spectral-Stokes signal (...,nlam,4) as a mantis shrimp does: 12 spectral bands + linear +
        circular polarization, in one call. Spectral channels are a DIRECT readout (no colour-opponent
        processing -- Thoen et al. 2014). See holographic_observer.mantis_view."""
        import holographic.rendering.holographic_observer as _ob
        return _ob.mantis_view(spectral_stokes, wavelengths_nm)

    def wavelength_to_rgb(self, nm):
        """The approximate sRGB a human sees for a monochromatic light at wavelength `nm` (UV/IR map to black --
        invisible). Field-native. Reuses the CIE curves. See holographic_falsecolor.wavelength_to_rgb."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.wavelength_to_rgb(nm)

    def hsv_to_rgb(self, h, s, v):
        """Vectorised HSV->RGB (all args in [0,1], hue wraps) -- the natural map for cyclic quantities like a
        polarization angle. See holographic_falsecolor.hsv_to_rgb."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.hsv_to_rgb(h, s, v)

    def falsecolor_spectral(self, readings, centers_nm, uv_hue=0.80):
        """False-colour N spectral-band readings (...,nchan) into an RGB image, with UV bands made VISIBLE in a
        chosen hue -- 'what a multi-band eye sees'. A CHOICE, not true colour. See
        holographic_falsecolor.spectral_falsecolor."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.spectral_falsecolor(readings, centers_nm, uv_hue=uv_hue)

    def falsecolor_polarization(self, evector_angle, dolp, value=1.0):
        """Standard polarization false-colour: hue=e-vector angle, saturation=degree of linear polarization,
        value=intensity (unpolarised -> grey). Field-native; a CHOICE of mapping. See
        holographic_falsecolor.polarization_falsecolor."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.polarization_falsecolor(evector_angle, dolp, value=value)

    def falsecolor_handedness(self, circular_R, circular_L):
        """Diverging false-colour for circular polarization sense: right-handed->red, left-handed->blue,
        unpolarised->white -- the channel the mantis uniquely sees. A CHOICE of convention. See
        holographic_falsecolor.handedness_falsecolor."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.handedness_falsecolor(circular_R, circular_L)

    def mantis_falsecolor(self, view, centers_nm=None):
        """SEE WHAT THE MANTIS SEES: turn a mantis_view() reading into three human-viewable RGB images -- color
        (12 bands incl. UV made visible), polarization (angle+strength), and handedness (circular sense). Every
        image is a false-colour CHOICE, not the mantis' percept. See holographic_falsecolor.mantis_falsecolor."""
        import holographic.rendering.holographic_falsecolor as _fc
        return _fc.mantis_falsecolor(view, centers_nm=centers_nm)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p13_recall_and_apply", "_UnifiedPart13")
    print("holographic_unified_p13_recall_and_apply selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

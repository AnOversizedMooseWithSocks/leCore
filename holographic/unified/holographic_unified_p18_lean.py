"""Part 18 of UnifiedMind's faculty surface -- formal logic & Lean 4 export.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which remains the only import path anyone uses.

WHY THIS PART EXISTS: the 2026-08-16 Rule-0 audit found NO proof/theorem/verification
capability under ten user phrasings -- the engine could recover laws from data
(holographic_symbolic) but could not PROVE a stated proposition, check the proof
independently, or hand the derivation to an external authority. holographic_lean fills
that gap; these faculties make it reachable by an agent over POST /invoke, speaking
plain JSON (the module's wire format), never its classes.

Every method DELEGATES to holographic.agents_and_reasoning.holographic_lean; none
reimplements. All are additive -- no existing behavior touched.
"""

from holographic.unified import check_part


class _UnifiedPart18:

    def logic_prove(self, goal, rules, max_steps=10000, strategy="naive"):
        """Prove a ground goal from Horn facts/rules by deterministic forward chaining.
        `goal` is ["pred", [args...]]; `rules` is a list of {"head": atom, "body": [atoms],
        "name": str} (empty body = fact). Returns a JSON-safe proof tree dict, or None if
        the goal is not derivable -- an honest None, never a manufactured proof.
        strategy="seminaive" (Bancilhon & Ramakrishnan 1986, opt-in) derives the SAME atom
        set >=22x faster on large bases (measured: the repo's own 708-module import graph,
        naive DNF at 300s vs 13.5s) but may pick a different valid proof tree.
        See holographic_lean.prove."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        p = _L.prove(_L.atom_from_wire(goal), rs, max_steps=max_steps, strategy=strategy)
        return None if p is None else _L.proof_to_wire(p)

    def logic_check_proof(self, proof, rules):
        """Independently verify a wire-format proof tree against the rule set. The checker
        shares no state with the prover and trusts nothing it says: forged premises raise.
        Returns True or raises AssertionError/KeyError loudly. See holographic_lean.check_proof."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        return _L.check_proof(_L.proof_from_wire(proof, rs), rs)

    def lean_export(self, goal, rules, theorem_name="derived", check=True):
        """Prove a goal and emit self-contained Lean 4 source (axioms + term-mode theorem).
        HONEST SCOPE: Lean verifies the proof FOLLOWS from the rules; it does NOT verify the
        rules are consistent -- an inconsistent rule set proves anything and typechecks doing
        it (see logic_consequences' absurdity smoke). check levels: True/"internal" runs the
        independent in-process checker before emitting; "external" ALSO round-trips through an
        installed lean binary and refuses ok=True unless BOTH agree (the de Bruijn criterion:
        two independent checkers, agreement as the deliverable) -- with no binary, ok is False
        and external.available says why, never faked. Returns {"ok","lean","proof"[,"external"]};
        ok=False, lean=None when underivable. See holographic_lean.to_lean."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        p = _L.prove(_L.atom_from_wire(goal), rs)
        if p is None:
            return {"ok": False, "lean": None, "proof": None}
        if check:
            _L.check_proof(p, rs)
        src = _L.to_lean(p, rs, theorem_name=theorem_name)
        out = {"ok": True, "lean": src, "proof": _L.proof_to_wire(p)}
        if check == "external":
            res = _L.lean_check(src)
            out["external"] = res
            # agreement is the deliverable: internal passed above; ok stands only if the
            # external kernel ALSO said proved (available and ok) -- absence is a loud False
            out["ok"] = bool(res.get("available")) and bool(res.get("ok"))
        return out

    def logic_query(self, goal, rules, budget=2000, fallback=True):
        """GOAL-DIRECTED evaluation with TABLING: answer a goal that may contain variables
        (e.g. ['ancestor',['tom','?w']]) by working backward from it, returning every ground
        binding with a checkable proof. Terminates on LEFT RECURSION and CYCLES where plain
        SLD diverges (tabling: a subgoal that is a variant of one in progress reads the
        answer table instead of recursing) -- measured on a cyclic left-recursive graph.

        MEASURED LAW and its NEGATIVE, so callers can choose honestly: speedup over the full
        fixpoint depends on the goal's DEMAND CLOSURE, not graph size -- 304x at demand 1,
        68x at 9, but 0.3x (SLOWER) at demand 690 on the repo's own import graph, break-even
        near demand ~200. `budget` caps tabled answers; with fallback=True (default) a
        blown budget transparently reruns as a seminaive fixpoint filtered to the goal, so
        the caller always gets a complete answer by whichever route is cheaper. The result
        reports which route ran. See holographic_lean.query."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        g = _L.atom_from_wire(goal)
        r = _L.query(g, rs, budget=budget)
        if not r["budget_exceeded"]:
            return {"answers": [[a.pred, list(a.args)] for a in r["answers"]],
                    "route": "query", "rounds": r["rounds"],
                    "proofs": {k: _L.proof_to_wire(v) for k, v in r["proofs"].items()}}
        if not fallback:
            return {"answers": None, "route": "query", "budget_exceeded": True}
        # the demand closure was too wide for goal-direction to pay -- take the fixpoint
        hits = [a for a in _L.consequences(rs, max_steps=10 ** 9, strategy="seminaive")
                if _L.unify(g, a) is not None]
        return {"answers": [[a.pred, list(a.args)] for a in hits], "route": "fixpoint",
                "rounds": r["rounds"]}

    def logic_consequences(self, rules, absurd=("false", "absurd", "bottom"),
                           strategy="naive"):
        """ALL derivable ground atoms -- the least fixpoint of the rule set (the van Emden-
        Kowalski T_P fixpoint, which prove() was computing and discarding), PLUS the cheap
        consistency smoke: whether any designated absurdity predicate is derivable, with its
        proof when so. Completeness as a measured property. Returns {"atoms": [[pred,args],...],
        "count", "absurd": {...}}. See holographic_lean.consequences / detect_absurdity."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        atoms = _L.consequences(rs, strategy=strategy)
        return {"atoms": [[a.pred, list(a.args)] for a in atoms], "count": len(atoms),
                "absurd": _L.detect_absurdity(rs, absurd=tuple(absurd))}

    def logic_proof_measure(self, proof, rules):
        """Honest complexity meter for a wire-format proof: size (nodes), height (longest
        branch), rule-usage multiset -- the derivation's shape as data that travels with the
        result (Gentzen's ordinal-assignment instinct, engineering shadow). Verifies the proof
        against the rules first: measuring an unchecked proof would report the shape of a
        possible lie. See holographic_lean.proof_measure."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        pr = _L.proof_from_wire(proof, rs)
        _L.check_proof(pr, rs)
        return _L.proof_measure(pr)

    def lean_verify(self, source, timeout=60):
        """Round-trip Lean 4 source through an installed `lean` binary (opt-in bridge,
        numba-style; the engine never requires it). Returns {"available", "ok", ...} --
        {"available": False} when no binary exists, stated honestly rather than pretended.
        See holographic_lean.lean_check."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        return _L.lean_check(source, timeout=timeout)

    def logic_decode_atom(self, vec, preds, symbols, max_args=2, floor=0.25):
        """Decode a fact vector back to (pred, args) with honest abstention -- encode_atom's
        inverse via the engine's own unbind + nearest cleanup (Rule-0 record: the resonator
        already ships in three costumes and is NOT needed for this known-role structure).
        Returns {"pred","args","score","abstained"[,"best"]}. See holographic_lean.decode_atom."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        from holographic.agents_and_reasoning.holographic_ai import (derived_atom, bind,
                                                                     unbind, nearest)
        import numpy as _np
        dim, seed = self.encoder.dim, self.encoder.seed
        sym = lambda name: derived_atom(seed, "lean:" + name, dim)
        return _L.decode_atom(_np.asarray(vec, float), list(preds), list(symbols),
                              int(max_args), sym, bind, unbind, nearest, floor=floor)

    def logic_fact_capacity(self, dim=None, n_symbols=32, n_preds=4, arity=2,
                            loads=(1, 2, 4, 8, 16, 32), seeds=6, floor=0.25):
        """PLATE'S QUESTION measured on OUR construction: how many facts survive in one
        bundled trace at dimension D (exact whole-atom recall, mean + bootstrap CI per load)?
        THE MEASURED VERDICT, kept loud: recall follows the 1/sqrt(M)-independent-of-D law
        (interfering facts are unit-norm whatever D is), with predicate-collision chimeras on
        top -- load 1-2 exact, a cliff at 4-8 that widening D does NOT move. Consequence:
        store fact bases as INDEXED rows (matmul search), never one bundled trace.
        See holographic_lean.fact_capacity."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        d = int(dim) if dim else self.encoder.dim
        return _L.fact_capacity(d, n_symbols=n_symbols, n_preds=n_preds, arity=arity,
                                loads=tuple(loads), seeds=range(int(seeds)), floor=floor)

    def logic_induce(self, background, positives, negatives, target, body_preds,
                     max_body=2, max_vars=3, theorem_name="conjecture"):
        """THE ENO LOOP: INDUCE Horn rules from ground examples (learning-from-failures,
        Cropper & Morel 2021 -- generate/test/constrain, honest scope: LFF-style on the
        finite fragment, not Popper parity), DEDUCE the surviving theory's full fixpoint,
        REFUTE against negatives, and emit Lean 4 source proving the first positive FROM
        THE LEARNED RULES. Recursion comes free (test is our own T_P; ancestor learns).
        Wire format throughout; refuted conjectures are counted, not hidden. Returns
        {"rules","lean","consequences","refuted_count","stats"}; rules=None when the
        bounded space exhausts uncovered -- never a guess dressed as an answer.
        See holographic_lean.conjecture_and_refute / induce_rules."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        bg = _L.rules_from_wire(background)
        pos = [_L.atom_from_wire(a) for a in positives]
        neg = [_L.atom_from_wire(a) for a in negatives]
        return _L.conjecture_and_refute(bg, pos, neg, target, dict(body_preds),
                                        max_body=max_body, max_vars=max_vars,
                                        theorem_name=theorem_name)

    def lean_fuzz(self, n=30, seed=0):
        """Differential oracle over the whole logic chain: n random HOSTILE theories (Lean
        keywords, collision pairs, digit-led names) through prove-both-strategies -> check
        -> export -> external Lean when installed (which is itself probed with a corrupted
        term each run). Failures return with their seed for pinning as Lean-free regression
        tests -- the distillation contract: Lean finds a bug once, the repo keeps the pin,
        the binary stays optional. Standing result on record: 300 theories, 793 exports,
        0 failures. An empty list is a measured statement about n seeds, not a proof.
        See holographic_lean.fuzz_export."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        return _L.fuzz_export(n=int(n), seed=int(seed))

    def _proof_mem(self):
        """Lazy per-mind store behind proof_store/proof_recall: parallel lists of goal
        vectors (INDEXED ROWS -- the logic_fact_capacity measurement is WHY this is rows
        and never one bundled trace), wire proofs, provenance, and trace vectors."""
        if not hasattr(self, "_proof_memory_state"):
            self._proof_memory_state = {"goal_vecs": [], "tree_vecs": [], "trace_vecs": [],
                                        "keys": [], "records": []}
        return self._proof_memory_state

    def proof_store(self, goal, rules, verify="internal"):
        """VERIFIED-KNOWLEDGE MEMORY, the Lean distillation into the substrate: prove the
        goal, run the INDEPENDENT checker (mandatory -- unchecked proofs never enter), and
        store the result as indexed rows in THIS mind's hypervector space: goal atom via
        logic_encode_atom, proof TREE via encode_tree_carrier (depth survives), rule TRACE
        via seq_encode. verify="external" additionally runs an installed Lean and records
        its verdict; provenance travels with the record ("checked" or "lean_verified") so a
        consumer can demand the stronger tier -- the binary stays optional, its verdict is
        what we keep. Returns {"stored", "key", "provenance"}; {"stored": False} when the
        goal is underivable -- nothing unproven is ever remembered."""
        import numpy as _np
        from holographic.agents_and_reasoning import holographic_lean as _L
        rs = _L.rules_from_wire(rules)
        g = _L.atom_from_wire(goal)
        pr = _L.prove(g, rs, strategy="seminaive")
        if pr is None:
            return {"stored": False, "key": None, "provenance": None}
        _L.check_proof(pr, rs)
        provenance = "checked"
        if verify == "external":
            res = _L.lean_check(_L.to_lean(pr, rs, theorem_name="stored"))
            if res.get("available") and res.get("ok"):
                provenance = "lean_verified"
        def tree(q):
            return ((q.rule.name,) + tuple(tree(c) for c in q.children)
                    if q.children else (q.rule.name, q.atom.key()))
        def trace(q, out):
            for c in q.children:
                trace(c, out)
            out.append(q.rule.name)
            return out
        names = sorted({r.name for r in rs})
        toks = [names.index(x) for x in trace(pr, [])]
        mem = self._proof_mem()
        mem["goal_vecs"].append(_np.asarray(self.logic_encode_atom(g.pred, list(g.args)), float))
        mem["tree_vecs"].append(_np.asarray(self.encode_tree_carrier(tree(pr)), float))
        # seq_encode returns COMPLEX (FHRR phases) -- casting to float discards the
        # imaginary half and corrupts every trace (caught live by the ComplexWarning;
        # the warning WAS the instrument). Store complex; recall uses a conj-aware cosine.
        mem["trace_vecs"].append(_np.asarray(self.seq_encode(
            toks, dim=self.encoder.dim, seed=self.encoder.seed,
            vocab_size=max(16, len(names)))))
        mem["keys"].append(g.key())
        mem["records"].append({"goal": [g.pred, list(g.args)], "proof": _L.proof_to_wire(pr),
                               "provenance": provenance,
                               "measure": _L.proof_measure(pr)})
        return {"stored": True, "key": g.key(), "provenance": provenance}

    def proof_recall(self, goal, k=3, min_provenance="checked", by="goal"):
        """Recall verified knowledge from the substrate: exact hit when the goal was stored,
        otherwise the k NEAREST stored records by cosine over goal vectors (by="goal"),
        proof-tree structure (by="tree"), or rule-trace shape (by="trace") -- structural
        neighbours are how a stored derivation SUGGESTS an approach to a new goal.
        min_provenance="lean_verified" filters to the externally judged tier. Honest empty:
        {"exact": None, "similar": []} when nothing qualifies."""
        import numpy as _np
        from holographic.agents_and_reasoning import holographic_lean as _L
        mem = self._proof_mem()
        g = _L.atom_from_wire(goal)
        tiers = {"checked": 0, "lean_verified": 1}
        ok = [i for i, r in enumerate(mem["records"])
              if tiers[r["provenance"]] >= tiers[min_provenance]]
        exact = next((mem["records"][i] for i in ok if mem["keys"][i] == g.key()), None)
        field = {"goal": "goal_vecs", "tree": "tree_vecs", "trace": "trace_vecs"}[by]
        sims = []
        if ok:
            q = _np.asarray(self.logic_encode_atom(g.pred, list(g.args)), float)                 if by == "goal" else (mem[field][ok[0]] * 0)  # tree/trace need a stored anchor
            if by != "goal":
                # for structural recall the query IS a stored key: nearest to its own row
                idx = next((i for i in ok if mem["keys"][i] == g.key()), None)
                if idx is None:
                    return {"exact": exact, "similar": []}
                q = mem[field][idx]
            M = _np.stack([mem[field][i] for i in ok])
            # conj-aware cosine: correct for the complex FHRR trace vectors, and reduces to
            # the ordinary cosine for the real goal/tree vectors
            cs = _np.real(M @ _np.conj(q)) / (
                _np.linalg.norm(M, axis=1) * _np.linalg.norm(q) + 1e-12)
            order = _np.argsort(-cs, kind="stable")
            for j in order:
                # exclude self FIRST, then take k -- slicing before exclusion returned an
                # empty list whenever the query's own row topped the ranking (caught by the
                # tree-recall pin: k=1 came back [])
                i = ok[int(j)]
                if mem["keys"][i] == g.key():
                    continue
                sims.append({"key": mem["keys"][i], "cos": float(cs[int(j)]),
                             "record": mem["records"][i]})
                if len(sims) >= k:
                    break
        return {"exact": exact, "similar": sims}

    def morphogenesis_grow(self, n_cells=64, radius=0.5, seed=0, steps=200,
                           k_rep=1.0, k_att=0.35, start="slab"):
        """Grow a soft-cell aggregate by alternating proliferation and ANALYTIC-gradient
        relaxation (backlog F1) -- stage one of energy-based morphogenesis, producing the
        compact genus-0 cell population that later stages sculpt into a body plan. No
        autodiff: pairwise potentials have closed-form gradients, verified against the
        engine's own fd_gradient to 3.5e-9. steps=0 runs proliferation WITHOUT relaxation,
        the honest control (measured contrast: sphericity 1.000 relaxed vs 0.008 control,
        so the ball comes from the energy, not from division jitter). Deterministic per
        seed. Returns {"positions","radii","energy","sphericity","history"}.
        See holographic_morphogen.grow_aggregate."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.grow_aggregate(n_cells=int(n_cells), radius=float(radius), seed=int(seed),
                                 steps=int(steps), k_rep=float(k_rep), k_att=float(k_att),
                                 start=start)

    def morphogenesis_relax(self, positions, radii, steps=300, k_rep=1.0, k_att=0.35):
        """Relax an existing cell population to its pair-potential minimum by gradient
        descent with backtracking (energy monotonically decreases -- pinned). Useful on its
        own for packing any soft-sphere set. KEPT NEGATIVE worth knowing before you call it:
        a perfectly SYMMETRIC configuration (e.g. a planar slab) is a critical point that
        descent cannot escape -- break the symmetry first. See holographic_morphogen.relax."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        import numpy as _np
        pos, hist = _M.relax(_np.asarray(positions, float), _np.asarray(radii, float),
                             steps=int(steps), k_rep=float(k_rep), k_att=float(k_att))
        return {"positions": pos, "history": hist, "energy": hist[-1] if hist else 0.0,
                "sphericity": _M.sphericity(pos)}

    def morphogenesis_differentiate(self, positions, radii, steps=250, k_adh=0.8,
                                    rd_weight=1.0, pi_weight=1.0, pi_axis=0, seed=0,
                                    rd_steps=400, width=0.25):
        """F2: run morphogens on the cell graph and relax under DIFFERENTIAL ADHESION --
        cells with similar morphogen values adhere, dissimilar ones do not, which breaks the
        aggregate's spherical symmetry into a body plan.

        MODE 2 (the current SOTA composition for limb patterning): the morphogen is an
        emergent Gray-Scott RD pattern (rd_weight) MODULATED BY a prescribed Wolpert
        positional gradient (pi_weight) -- set either weight to 0 to ablate, which is the
        experiment the selftest runs. MEASURED with the no-adhesion control: sphericity
        0.824 (control) vs 0.257 (Mode 2), so the shape change is the adhesion.
        KEPT NEGATIVE: at these cell counts the RD makes ONE front, not multiple spots --
        multi-lobe patterning needs a domain several pattern-wavelengths across.
        Returns {"positions","morphogen","u","v","energy","sphericity","lobes","history"}.
        See holographic_morphogen.differentiate."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        import numpy as _np
        return _M.differentiate(_np.asarray(positions, float), _np.asarray(radii, float),
                                steps=int(steps), k_adh=float(k_adh),
                                rd_weight=float(rd_weight), pi_weight=float(pi_weight),
                                pi_axis=int(pi_axis), seed=int(seed),
                                rd_steps=int(rd_steps), width=float(width))

    def tetrahedralize(self, positions, radii=None, alpha_scale=1.6, jitter=0.0, seed=0):
        """F3: turn a cell population into a volumetric TET MESH (Bowyer-Watson Delaunay +
        alpha-complex carving, NumPy only -- no scipy/Qhull). SCOPE, honestly: for clean
        POINT SETS like morphogenesis output, not a TetGen/fTetWild replacement (those solve
        the different problem of surviving broken triangle soup); no quality optimisation or
        sliver removal. Returns tets, face adjacency, boundary faces, NON-MANIFOLD faces
        (reported, never swallowed), component count and Euler numbers.
        See holographic_tetmesh.tetrahedralize."""
        from holographic.mesh_and_geometry import holographic_tetmesh as _T
        import numpy as _np
        return _T.tetrahedralize(_np.asarray(positions, float),
                                 None if radii is None else _np.asarray(radii, float),
                                 alpha_scale=float(alpha_scale), jitter=float(jitter),
                                 seed=int(seed))

    def tet_connectivity_certificate(self, mesh, source_tet, target_tets):
        """PROVE that each target tet reaches the source through face adjacency -- "every
        limb is attached to the torso" as a DERIVATION over the mesh's own facts, not a flood
        fill. Uses the tabled query (small demand closure: the regime E1 measured at 60-300x).
        Orphaned limbs come back in "unreachable" -- a certificate that never fails certifies
        nothing, and the severed case is pinned by test. MEASURED DESIGN LAW: an attachment
        1-2 cells across is NOT volumetrically connected (collinear points make no tets);
        3 cells across is the minimum. See holographic_tetmesh.connectivity_certificate."""
        from holographic.mesh_and_geometry import holographic_tetmesh as _T
        return _T.connectivity_certificate(mesh, int(source_tet),
                                           [int(t) for t in target_tets])

    def tet_certificate_lean(self, mesh, source_tet, target_tet,
                             theorem_name="limb_connected"):
        """Emit Lean 4 source proving one connectivity claim about this mesh, so an external
        kernel can confirm it (Tier 1, opt-in -- EMITTING needs no binary). Returns None when
        the claim is underivable, never a fabricated proof.
        See holographic_tetmesh.certificate_lean."""
        from holographic.mesh_and_geometry import holographic_tetmesh as _T
        return _T.certificate_lean(mesh, int(source_tet), int(target_tet),
                                   theorem_name=theorem_name)

    def fem_simulate(self, positions, tets, steps=200, mu=1.0, lam=10.0, fibers=None,
                     rest_lengths=None, activation=1.0, k_muscle=10.0, gravity=0.0,
                     pinned=None, rest=None, record_every=0, step0=0.01):
        """F4: quasistatic STABLE NEO-HOOKEAN solve over a tet mesh, with optional muscle
        fibers. Uses Smith/De Goes/Kim 2018 rather than the classical log-J neo-Hookean
        because log J is UNDEFINED for inverted elements and generated meshes DO invert --
        this energy stays finite and differentiable through inversion (pinned by test).
        NO autodiff: the first Piola-Kirchhoff stress is hand-derived and checked against
        fd_gradient to 2e-11, and the rest state is exactly stress-free (7e-17).
        `pinned` holds vertices fixed; `activation` < 1 contracts fibers.
        Returns {"positions","energy","history","rest_quality"}. See holographic_fem.simulate."""
        from holographic.simulation_and_physics import holographic_fem as _F
        import numpy as _np
        return _F.simulate(_np.asarray(positions, float), _np.asarray(tets, int),
                           steps=int(steps), mu=float(mu), lam=float(lam), fibers=fibers,
                           rest_lengths=rest_lengths, activation=activation,
                           k_muscle=float(k_muscle), gravity=float(gravity),
                           pinned=pinned, rest=rest, record_every=int(record_every), step0=step0)

    def fem_select_fibers(self, positions, tets, axis=0, fraction=0.25):
        """Choose muscle fibers as the tet edges best ALIGNED with an axis (deterministic).
        Alignment rather than random selection because a muscle pulling every direction at
        once does no net work. Returns (fibers, rest_lengths).
        See holographic_fem.select_fibers."""
        from holographic.simulation_and_physics import holographic_fem as _F
        import numpy as _np
        return _F.select_fibers(_np.asarray(positions, float), _np.asarray(tets, int),
                                axis=int(axis), fraction=float(fraction))

    def fem_rest_quality(self, positions, tets):
        """Element-quality report for a REST mesh before anyone simulates it: degenerate
        count, INVERTED count, volume extremes. Exists because a generated mesh can be born
        inverted and a simulator that silently accepts that produces confident nonsense --
        this report is what caught the tetrahedraliser emitting mixed winding.
        See holographic_fem.rest_quality."""
        from holographic.simulation_and_physics import holographic_fem as _F
        import numpy as _np
        return _F.rest_quality(_np.asarray(positions, float), _np.asarray(tets, int))

    def tet_lod_chain(self, positions, radii=None, fractions=(1.0, 0.6, 0.35, 0.2),
                      seed=0, alpha_scale=1.6, source_tet=0, require_connected=True):
        """F5: a CERTIFIED volumetric LOD chain where a level is a RULE, not a stored mesh.
        Farthest-point ordering is nested, so level k is a PREFIX of one permutation -- the
        whole chain costs one point set + one ordering (measured 9.1x smaller than storing
        the meshes). Each level is re-tetrahedralised and must re-pass F3's certificate;
        a level that fragments or ORPHANS A LIMB comes back ok=False with its reason,
        REFUSED rather than shipped looking fine. This is NOT a better QEM -- leCore already
        ships mesh_qem_decimate/mesh_lod_chain for surface LOD; this is the strategy only a
        GENERATED body allows. See holographic_tetmesh.lod_chain."""
        from holographic.mesh_and_geometry import holographic_tetmesh as _T
        import numpy as _np
        return _T.lod_chain(_np.asarray(positions, float),
                            None if radii is None else _np.asarray(radii, float),
                            fractions=tuple(fractions), seed=int(seed),
                            alpha_scale=float(alpha_scale), source_tet=int(source_tet),
                            require_connected=bool(require_connected))

    def tet_lod_storage_cost(self, positions, chain):
        """Measure the 'store the rule, not the bytes' claim for an LOD chain: rule units
        (points + ordering) vs stored units (every accepted level's mesh), and their ratio.
        See holographic_tetmesh.lod_storage_cost."""
        from holographic.mesh_and_geometry import holographic_tetmesh as _T
        import numpy as _np
        return _T.lod_storage_cost(_np.asarray(positions, float), chain)

    def genome_encode(self, params, dim=1024, seed=0):
        """F6: encode a body-plan genome (k_rep, k_att, k_adh, width, rd_weight, pi_weight)
        as ONE hypervector -- roles bound to fractional-power-encoded scalars, so NEARBY
        PARAMETERS GIVE NEARBY VECTORS. That locality property is what the encoding
        literature calls decisive for search, and here it comes from the encoder rather than
        being asserted (mind.genome_locality measures the curve). A DIRECT encoding lifted
        into the substrate -- not a generative/latent one, and no learned weights.
        See holographic_morphogen.genome_encode."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.genome_encode(dict(params), dim=int(dim), seed=int(seed))

    def genome_decode(self, vec, dim=1024, seed=0, samples=64, floor=0.15):
        """Recover genome parameters from a vector, ABSTAINING per field below `floor`
        rather than confabulating (noise must decode to nothing -- pinned).
        See holographic_morphogen.genome_decode."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        import numpy as _np
        return _M.genome_decode(_np.asarray(vec, float), dim=int(dim), seed=int(seed),
                                samples=int(samples), floor=float(floor))

    def genome_locality(self, dim=1024, seed=0, deltas=(0.01, 0.05, 0.1, 0.25, 0.5),
                        trials=8):
        """MEASURE the encoding's locality curve (mean cosine vs relative perturbation, with
        spread) -- the evolutionary-encoding literature's decisive quality criterion turned
        into a number. MEASURED: 1.000 / 0.996 / 0.986 / 0.921 / 0.792 -- smooth and
        monotone, no cliff. See holographic_morphogen.genome_locality."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.genome_locality(dim=int(dim), seed=int(seed), deltas=tuple(deltas),
                                  trials=int(trials))

    def genome_interpolate(self, pa, pb, t):
        """Blend two genomes in PARAMETER space (bodies are grown from parameters; vector
        space here is for search and comparison, not breeding -- interpolating the VECTORS
        yields a superposition that decodes to one endpoint, not a blend). MEASURED: 5/5
        interpolants at t=0..1 produced certificate-clean single-component bodies.
        See holographic_morphogen.genome_interpolate."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.genome_interpolate(dict(pa), dict(pb), float(t))

    def shape_memory_store(self, shapes, bins=8):
        """F7: store target morphologies as a descriptor codebook (radial mass profile --
        translation- and scale-free). See holographic_morphogen.shape_memory_store."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.shape_memory_store(shapes, bins=int(bins))

    def shape_memory_recall(self, positions, codebook, beta=25.0, steps=3, bins=8):
        """Retrieve which stored morphology a (possibly perturbed) body is, via the engine's
        OWN dense/modern-Hopfield cleanup -- Rule 0: the associative memory already shipped.
        Low confidence means "resembles nothing stored", not a confident wrong answer.
        See holographic_morphogen.shape_memory_recall."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        import numpy as _np
        return _M.shape_memory_recall(_np.asarray(positions, float),
                                      _np.asarray(codebook, float), beta=float(beta),
                                      steps=int(steps), bins=int(bins))

    def shape_memory_probe(self, n_shapes=3, n_cells=45, noise=0.35, trials=6, seed=0, bins=8):
        """THE EXPERIMENT, not a demo: does recovery depend on the STORED PATTERN or merely
        on a well existing? Reports accuracy against a DEPTH-MATCHED SCRAMBLED control.
        MEASURED: noise 0.1 -> 1.00 vs control 0.00; 0.3 -> 0.80 vs 0.07; 0.6 -> 0.47 vs
        0.20 (chance 0.33). KEPT NEGATIVE: varying only GROWTH parameters gives bodies with
        0.99+ descriptor similarity and recall exactly at chance -- discriminability is a
        property of the GENERATOR, not the memory; distinct shapes need F2 differentiation.
        See holographic_morphogen.shape_memory_probe."""
        from holographic.simulation_and_physics import holographic_morphogen as _M
        return _M.shape_memory_probe(n_shapes=int(n_shapes), n_cells=int(n_cells),
                                     noise=float(noise), trials=int(trials), seed=int(seed), bins=bins)

    def tier_certify_plan(self, tiers, plan, forbid_tiers=(), min_recall=None):
        """D1: certify a memory plan against TIER CONTRACTS before it runs -- {pre} plan
        {post} in Hoare's sense, not a roofline (the Cache-Aware Roofline Model is
        descriptive; this is a precondition check that REFUSES). Three clauses, each
        reported separately: capacity, a forbidden-tier ban DERIVED through the Horn kernel
        rather than scanned, and -- the clause a classical cache has no analogue for --
        FIDELITY, because a holographic tier is lossy-but-graceful. The fidelity ladder is
        MEASURED (D5 sweep, four dimensions): recall collapses onto D/M, so 'recall >= 0.98'
        is discharged by 'load <= dim/32'. See holographic_tiercontract.certify_plan."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.certify_plan(dict(tiers), list(plan), forbid_tiers=tuple(forbid_tiers),
                               min_recall=min_recall)

    def tier_fidelity_floor(self, dim, load):
        """The recall a superposed tier is CONTRACTUALLY good for at this load, from the
        measured D/M ladder. Conservative between rungs on purpose: interpolating a
        measurement would promise a number nobody measured.
        See holographic_tiercontract.fidelity_floor."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.fidelity_floor(int(dim), int(load))

    def bake_certify(self, evaluate, lookup, n_cells, n_samples=256, seed=0, tol=1e-9,
                     k_corrupt=None, confidence=0.99):
        """D2: certify a baked artifact against its generating rule, WITH A STATED
        GUARANTEE. store_procedural already verifies pointwise; what it cannot tell you is
        how much confidence "it passed" carries. This samples deterministically (an auditor
        can re-run the SAME plan) and reports the hypergeometric detection probability --
        the established spot-check bound -- for a corruption of `k_corrupt` cells, plus how
        many samples the requested confidence would need. HONEST LIMIT: a pass bounds the
        chance of missing a k-cell corruption; a SINGLE bad cell is hard to catch at small
        m, and the number says so rather than hiding it.
        See holographic_tiercontract.certify_bake."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.certify_bake(evaluate, lookup, int(n_cells), n_samples=int(n_samples),
                               seed=int(seed), tol=float(tol), k_corrupt=k_corrupt,
                               confidence=float(confidence))

    def bake_samples_for_confidence(self, n_cells, k_corrupt, confidence=0.99):
        """How many spot-checks does a bake of `n_cells` need to catch a `k_corrupt`-cell
        corruption with `confidence`? Returns None when the requested confidence is
        unreachable within the search cap -- stated, not silently clamped.
        See holographic_tiercontract.samples_for_confidence."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.samples_for_confidence(int(n_cells), int(k_corrupt), float(confidence))

    def differential_agreement(self, implementations, cases, tol=1e-9, reference=None, compare=None):
        """THE TWO-INSTRUMENT PATTERN, NAMED ONCE (house rule: consolidate at three
        customers; this had five -- SDF emitters, the logic fuzzer, the tetmesh certificate
        vs an independent flood fill, seminaive-vs-naive equality, and query-vs-fixpoint).
        Runs the same cases through several implementations against a REFERENCE oracle and
        reports where they deviate, with the case index so a disagreement is reproducible
        rather than merely counted. A crash counts as a disagreement.

        `tol` is first-class because the differential-testing literature is explicit that a
        STRICT oracle produces false alarms on numeric backends; the report always states
        the WORST deviation so a caller sees how much tolerance was actually consumed.
        KEPT NEGATIVE: agreement is not correctness -- two implementations of the same wrong
        idea agree perfectly. This shows a TRANSLATION preserved meaning, not that the
        meaning is right. See holographic_tiercontract.differential_agreement."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.differential_agreement(dict(implementations), list(cases),
                                         tol=float(tol), reference=reference, compare=compare)

    def schedule_certify(self, waves, resources):
        """D4: certify that no two tasks in the SAME wave share a declared resource, and
        that the waves are a well-formed partition (every task exactly once -- a schedule
        that silently DROPS a task is a worse bug than one that races). Violations name the
        wave, the pair, and the resource, so a failure is actionable.

        HONEST SCOPE, and it matters: this certifies the SCHEDULE, not the PROGRAM. General
        static race verification is hard (the 2025 Faial study found 98% of race-free GPU
        programs needed specific thread configs to be analysable); ours is easy only because
        the schedule is explicit and the resources are DECLARED. A task touching a resource
        it did not declare is outside the certificate.
        See holographic_tiercontract.certify_schedule."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.certify_schedule(list(waves), dict(resources))

    def schedule_conflict_edges(self, resources):
        """Derive the conflict graph from resource declarations, so colour_waves and the
        certificate are built from the SAME source. Writing the edge list twice is how a
        schedule and its check quietly stop describing the same system.
        See holographic_tiercontract.resource_conflict_edges."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.resource_conflict_edges(dict(resources))

    def demux_gated(self, x, noise_limit=0.05, **kw):
        """A2's GATE: run demux_series and REFUSE the answer when the implied substreams are
        too noisy for the MEASURED envelope (stride recovery is 1.00 to 5% noise and
        collapses by 10%). Adds noise_ratio / noise_limit / trusted to the result.

        Noise is estimated by the Donoho-Johnstone robust sigma -- MAD of SECOND differences
        (they annihilate a locally linear trend, so on a smooth source what survives is
        noise). It is measured on the substreams IMPLIED BY THE RETURNED K, so the gate
        validates the ANSWER rather than the input: a wrong K yields rough substreams, a
        large ratio, and a refusal. Both failure directions land on refuse.

        TWO HONEST CAVEATS. (1) The estimate carries a FLOOR from the signal's own curvature
        (~0.036 on the test waveform at zero noise), so the gate is CONSERVATIVE and will
        refuse some correct answers near the boundary. (2) It gates on noise only; a source
        that is not SMOOTH violates demux_series's precondition and is outside this gate
        entirely. Measured: 0 false-trust across 15 (K, noise) cells -- it never blesses a
        wrong stride, which is the property worth having.
        See holographic_tiercontract.demux_gated."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.demux_gated(self, x, noise_limit=float(noise_limit), **kw)

    def estimate_noise_sigma(self, y):
        """Robust noise sigma of a SMOOTH series: MAD of second differences over
        0.6745*sqrt(6) (Donoho-Johnstone, the wnoisest estimator generalised to second
        differences). Measures noise; the denoise* family REMOVES it -- different verbs.
        See holographic_tiercontract.estimate_noise_sigma."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.estimate_noise_sigma(y)

    def pose_certify(self, joints, limits, rest_lengths=None, target=None,
                     root_ref=(0.0, 1.0, 0.0)):
        """B4: certify a solved pose against the SAME limit spec the solver was given --
        bone lengths preserved, every joint inside its hinge/cone limit. Violations name the
        joint and the amount.

        SCOPE, and it is deliberately narrow: this certifies the pose that was RETURNED. It
        does NOT claim optimality, because constrained IK genuinely can miss a feasible
        solution that exists (the FABRIK literature says so of itself: each joint is placed
        without considering the next joint's restriction). Target error is REPORTED, never
        certified -- an unreachable target is a fact about the target, not a defect in the
        pose. See holographic_tiercontract.certify_pose."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.certify_pose(joints, limits, rest_lengths=rest_lengths, target=target,
                               root_ref=root_ref)

    def conservation_ledger(self, history, exact=(), bounded=(), exact_tol=1e-9,
                            ramp_tol=0.6):
        """C1: audit a run's conserved quantities, testing the RIGHT thing for each kind.
        `exact` (mass; linear momentum under symmetric internal forces) is judged on absolute
        drift. `bounded` (energy under a symplectic scheme) is judged on SECULAR TREND only,
        because symplectic integrators conserve a SHADOW Hamiltonian -- energy oscillates
        and stays bounded rather than being exactly conserved, so an |dE|~0 test would
        condemn the best integrators for behaving correctly. Bounded wobble passes; a slow
        ramp fails. MEASURED on velocity Verlet over leCore's own pair potential: energy
        ramp 0.28 (passes), momentum exact to 1.1e-14.
        See holographic_tiercontract.conservation_ledger."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.conservation_ledger(dict(history), exact=tuple(exact),
                                      bounded=tuple(bounded), exact_tol=float(exact_tol),
                                      ramp_tol=float(ramp_tol))

    def lyapunov_certify(self, witness, residuals=None, rise_tol=1e-9, settle_frac=0.02):
        """Upgrade a settle from GUESSED to CERTIFIED, when the run qualifies.

        The settle gate is sound only for stagnation plateaus up to about its window
        (MEASURED: window 96 falsely settles at plateau 128). The escape is a theorem, not a
        bigger window: for a TRUE gradient flow a state plateau means grad E ~ 0, a CRITICAL
        POINT, which cannot spontaneously resume -- so the stagnation trap is IMPOSSIBLE
        there. This checks the PRECONDITION rather than the plateau: the witness never rises,
        it has arrived, and the residual quieted with it. certified=False is not a failure;
        it means only the window heuristic applies and `window` should be sized accordingly.
        See holographic_tiercontract.lyapunov_certify."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.lyapunov_certify(witness, residuals=residuals, rise_tol=float(rise_tol),
                                   settle_frac=float(settle_frac))

    def plan_certify(self, plan, actions, initial_state, goal=None):
        """C4: certify a GOAP-style plan -- every action's PRECONDITIONS hold when it runs,
        and the GOAL holds at the end. Violations name the step index, the action, and the
        missing precondition ("step 1 fire requires has_weapon"), because that is actionable
        where "invalid plan" is not.

        WHY THIS EXISTS even though GOAP planners promise valid plans: that promise covers
        plans the PLANNER built. It says nothing about a plan that was hand-authored,
        learned, replanned mid-execution, or handed over from another system -- which is
        most plans that reach a creature at runtime. Complements mind.validate_plan, which
        checks ORDERING constraints and cannot see a missing precondition because it does
        not simulate state. KEPT NEGATIVE: this certifies FEASIBILITY, not optimality -- a
        plan that reaches the goal by a ludicrous route certifies exactly like a good one,
        because cost is the planner's business. See holographic_tiercontract.certify_plan_actions."""
        from holographic.caching_and_storage import holographic_tiercontract as _T
        return _T.certify_plan_actions(list(plan), dict(actions), dict(initial_state),
                                       goal=goal)

    def template_wrap(self, vertices, faces, field, rounds=6, step0=0.35, step1=1.0,
                      smooth_iters=6, level=0.0):
        """O1 (overhaul keystone): wrap a template mesh onto a target field KEEPING ITS
        TOPOLOGY EXACTLY, so vertex i is the same anatomical point on every body -- the
        precondition for blendshapes, shared textures and cross-species morphing, none of
        which are possible while each creature meshes from scratch.

        Follows non-rigid ICP's annealed schedule (Amberg et al. 2007): projection step
        rising over rounds, Taubin NO-SHRINK relaxation between them. Better conditioned
        than N-ICP because the target is an ANALYTIC field -- correspondence is not
        estimated by nearest-point search, it is a Newton step along the exact gradient.
        MEASURED: the wrap IMPROVES triangle quality (p95/p5 edge ratio 66.6 -> 38.3).
        KEPT NEGATIVE: valid only where template and target share TOPOLOGY -- wrap a biped
        onto a snake and vertices pile into the missing limbs, with correct connectivity and
        meaningless correspondence. See holographic_templatewrap.wrap_to_field."""
        from holographic.mesh_and_geometry import holographic_templatewrap as _TW
        return _TW.wrap_to_field(vertices, faces, field, rounds=int(rounds),
                                 step0=float(step0), step1=float(step1),
                                 smooth_iters=int(smooth_iters), level=float(level),
                                 mind=self)

    def template_wrap_quality(self, vertices, faces, field, level=0.0):
        """Did the wrap land, and is it still a usable mesh? surface_error (is it ON the
        target), edge_ratio as a ROBUST p95/p5 (bunching), degenerate_edges counted
        SEPARATELY, and flipped faces. The split matters: a max/min ratio read 59,000,000 on
        a mesh whose bulk triangles were fine, because one sliver dominates it.
        See holographic_templatewrap.wrap_quality."""
        from holographic.mesh_and_geometry import holographic_templatewrap as _TW
        return _TW.wrap_quality(vertices, faces, field, level=float(level))

    def blend_corrective(self, mesh, source_vertex, radius, direction, amplitude,
                         falloff="smoothstep"):
        """O2: author ONE blendshape target with DECLARED local support -- displace vertices
        within `radius` GEODESIC distance of `source_vertex`, along `direction` (a 3-vector,
        or 'normal' to inflate).

        SMPL's pose correctives are dense and "relate every vertex to all the joints",
        capturing spurious long-range correlations; STAR fixes that by spending scan data to
        LEARN each joint's activation region. Authoring a basis, we DECLARE the region
        instead -- so STAR's headline improvement is the default here, and the support is
        provably exact (measured overreach 0.000e+00). Geodesic, not Euclidean: a hand
        resting on a hip is millimetres away in space and a metre away across the surface.
        KEPT NEGATIVE: a declared radius guarantees LOCALITY, not anatomical realism -- a bad
        radius gives a local, smooth, wrong bulge, and no proof supplies a shape
        distribution only scans can measure.
        See holographic_blendbasis.make_corrective."""
        from holographic.mesh_and_geometry import holographic_blendbasis as _BB
        return _BB.make_corrective(mesh, int(source_vertex), float(radius), direction,
                                   float(amplitude), self, falloff=str(falloff))

    def blend_locality_report(self, base, targets, mesh, sources, radii):
        """Is every corrective ACTUALLY local? Reports each target's farthest geodesic
        influence against its declared radius; max_overreach > 0 means the spurious
        long-range coupling STAR exists to remove has crept back in.
        See holographic_blendbasis.locality_report."""
        from holographic.mesh_and_geometry import holographic_blendbasis as _BB
        return _BB.locality_report(base, targets, mesh, sources, radii, self)

    def conv_calibrated_segments(self, segments, kernel=2.2, iso=0.35):
        """O4: rescale convolution-surface segment radii so the iso-surface lands at the
        radius the CALLER asked for.

        SOTA states the weakness that makes this necessary: "while convolution surfaces
        eliminate bulge artifacts, they also reduce geometric control, since the target
        iso-surface is no longer located at the expected distance from the skeleton"
        (SCALIS, Zanni et al.). MEASURED: the surface lands ~26% INSIDE the request at
        kernel 2.2, and the shortfall depends on the KERNEL not the radius (1.6 -> 0.926,
        2.2 -> 0.747, 3.0 -> 0.590), so it is a one-dimensional constant -- solved once,
        cached, divided out. Radius error 25.4% -> 0.1% across the working range.
        HONEST RESIDUAL: 5.3% at the thinnest radii. That is the SCALIS scale-invariance
        effect ("thin components excessively smoothed when blended into larger ones");
        calibration removes the CONSTANT error, only a scale-invariant kernel removes the
        rest, and this is not SCALIS. See holographic_creatureconv.calibrated_segments."""
        from holographic.mesh_and_geometry import holographic_creatureconv as _CC
        return _CC.calibrated_segments(list(segments), kernel=float(kernel),
                                       iso=float(iso))

    def conv_radius_ratio(self, kernel=2.2, iso=0.35):
        """Where a convolution iso-surface actually lands, as a fraction of the requested
        radius -- the calibration constant, solved once per kernel and cached.
        See holographic_creatureconv.radius_ratio."""
        from holographic.mesh_and_geometry import holographic_creatureconv as _CC
        return _CC.radius_ratio(kernel=float(kernel), iso=float(iso))

    def face_landmarks(self, head_centre, head_height, head_width, depth=None,
                       proportions=None):
        """O3: skull-canon landmark positions for a head -- crown, brow, eye, nose, mouth,
        chin, jaw, cheek, ear, temple -- with bilateral pairs MIRRORED structurally so
        symmetry cannot be forgotten. `proportions` is the slider surface.

        WHY A PART GRAPH RATHER THAN A 3DMM: FLAME/DECA are the standard, and OmniFaceRig
        (2026) states their limit -- "bound to a fixed mesh topology and expression basis
        defined at scan-collection time ... primarily assume ADULT HUMAN ANATOMY", so a
        novel asset with stylized proportions or non-human features fits unstably. An engine
        for salamanders and centaurs IS that asset. SCULPTOR's skeleton-consistency idea is
        kept (landmarks sit on skull canon; soft tissue grows outward), without its CT-scan
        basis. NOT a likeness of anyone and NOT a reconstruction from a photo: there is no
        fitting step because there is no scan basis. See holographic_face.face_landmarks."""
        from holographic.mesh_and_geometry import holographic_face as _F
        return _F.face_landmarks(head_centre, float(head_height), float(head_width),
                                 depth=depth, proportions=proportions)

    def face_part_graph(self, landmarks, scale=1.0):
        """Which rigblock goes at which landmark, as DATA -- so a four-eyed, noseless face is
        an edit to a list rather than a new code path. Feed each entry to build_part.
        See holographic_face.face_part_graph."""
        from holographic.mesh_and_geometry import holographic_face as _F
        return _F.face_part_graph(dict(landmarks), scale=float(scale))

    def face_expression(self, landmarks, name, amount=1.0):
        """An expression as per-landmark DISPLACEMENTS, ready to drive O2's local
        correctives (blend_corrective) -- not a learned basis, so a new expression is a dict
        entry. Linear in `amount` and extrapolable past 1. MEASURED composing with O2:
        overreach 0.000e+00, each facial corrective moving 0.18-0.38% of the mesh.
        See holographic_face.expression."""
        from holographic.mesh_and_geometry import holographic_face as _F
        return _F.expression(dict(landmarks), str(name), amount=float(amount))

    def skin_twist_shrink(self, weights, angles):
        """L4: how much volume LBS will lose under a twist, in CLOSED FORM --
        |sum_b w_b exp(i theta_b)|. 1.0 preserves volume, 0.0 is total collapse. The classic
        two-bone case reduces to |cos(theta/2)|, which is the candy-wrapper artifact: 0.707
        at 90 degrees, ZERO at 180. VERIFIED against the shipped skinning path to 1.1e-16,
        so this is a theorem about the code rather than a model of it.
        See holographic_skinbound.twist_shrink."""
        from holographic.mesh_and_geometry import holographic_skinbound as _SB
        return _SB.twist_shrink(weights, angles)

    def skin_pose_is_safe(self, weights, angles, min_shrink=0.85):
        """Would this pose PINCH? The point of L4 -- a rig can refuse before deforming,
        instead of shipping a collapsed elbow and finding it in a render. Reports the worst
        vertex and its shrink. Does NOT propose a new skinning method: the field's fixes
        (DQS, spherical blending, optimised centres of rotation) are runtime model changes
        that trade one artifact for another (DQS "reveals its own artefact, joint-bulging");
        this supplies the missing PREDICATE. CONSERVATIVE for non-coaxial rotations -- the
        closed form is exact for a pure twist, which is the worst case.
        See holographic_skinbound.pose_is_safe."""
        from holographic.mesh_and_geometry import holographic_skinbound as _SB
        return _SB.pose_is_safe(weights, angles, min_shrink=float(min_shrink))

    def skin_max_safe_twist(self, weights, min_shrink=0.85):
        """The largest two-bone twist that keeps volume above `min_shrink`, SOLVED not
        searched (the closed form inverts directly). Even 50/50 weights allow only ~63.6
        degrees at a 0.85 floor -- which is why twist-bone chains exist.
        See holographic_skinbound.max_safe_twist."""
        from holographic.mesh_and_geometry import holographic_skinbound as _SB
        return _SB.max_safe_twist(weights, min_shrink=float(min_shrink))

    def wrap_is_injective(self, vertices, faces, offset, sdf, samples=1500, seed=0):
        """L3: would this offset/shrink-wrap FOLD the mesh through itself? The predicate that
        makes O1's template_wrap correct rather than hopeful -- a folded wrap still reads
        clean on surface_error, because every vertex IS on the surface.

        Checks BOTH classical conditions, and the second is the one that bites for creatures:
        LOCAL, the offset must stay under the smallest radius of curvature in concave
        regions; and GLOBAL, "a pair of COLLINEAR NORMAL POINTS whose distance is equal or
        smaller than twice the offset distance". An armpit, a limb beside a torso, or the gap
        between fingers is LOW CURVATURE with two surfaces FACING each other -- curvature
        alone passes exactly the cases a creature rig hits, which is worse than no check.
        Together the two are the REACH. KEPT NEGATIVE: this SAMPLES the reach rather than
        computing the medial axis, so a pass is evidence and not proof.
        See holographic_offsetreach.wrap_is_injective."""
        from holographic.mesh_and_geometry import holographic_offsetreach as _OR
        return _OR.wrap_is_injective(vertices, faces, float(offset), sdf, mind=self,
                                     samples=int(samples), seed=int(seed))

    def surface_safe_offset(self, sdf, points, normals=None):
        """The largest offset that keeps a normal projection injective: min(curvature limit,
        facing limit). Reports both terms and the LIMITING PAIR, so a caller can see WHERE
        the geometry is tight rather than only being told a number.
        See holographic_offsetreach.safe_offset."""
        from holographic.mesh_and_geometry import holographic_offsetreach as _OR
        return _OR.safe_offset(sdf, points, normals=normals, mind=self)

    def convolution_field_scalis(self, segments, iso=0.35, samples=24, kernel=2.2):
        """SCALIS (Zanni et al. 2013): a scale-invariant convolution field, so THIN FEATURES
        SURVIVE next to thick ones.

        Plain convolution integrates over ABSOLUTE arc length, so a long thick segment
        deposits more total field than a short thin one -- which is why "thin shape
        components are excessively smoothed out when blended into larger ones" and why
        prescribed radii are not reconstructed. SCALIS changes the NORMALIZATION FACTOR,
        integrating over the homothetic measure ds/tau. Exactly invariant under
        (r, L, d) -> lam*(r, L, d): the exponent d^2/r^2 and the weight L/(n*r) are both
        unchanged, MEASURED constant at 0.13241 across a 16x scale range where plain scales
        by lam.

        MEASURED on the case it exists for -- a spike 5.7x thinner than its trunk: plain
        renders it at 9% of the requested radius (nearly swallowed, which is the salamander's
        vanishing tail tip); SCALIS at 123%. Default-off elsewhere; this is the opt-in entry
        point. See holographic_creatureconv.convolution_field."""
        from holographic.mesh_and_geometry import holographic_creatureconv as _CC
        return _CC.convolution_field(list(segments), iso=float(iso), samples=int(samples),
                                     kernel=float(kernel), scalis=True)

    def tissue_pbr(self, tissue, scale=1.0):
        """Physically-based material for one TISSUE -- the fix for flat-shaded interiors.
        Returns base_color, roughness, metallic, sss_weight and a PER-CHANNEL sss_radius.

        Per-channel because red light scatters deeper than blue in every soft tissue; a
        scalar radius cannot make flesh read warm at the silhouette, which is the difference
        between "red plastic" and "meat". Christensen-Burley parameterisation (albedo +
        scattering distance), with the ORDERING grounded in measured SDOCT coefficients --
        bone and skin 1.95-2.13 /mm, liver and brain 1.30-1.46, spleen 0.52-0.63 -- so
        viscera scatter furthest and bone least, by measurement rather than art direction.
        KEPT NEGATIVE: single-medium per tissue; real skin needs a MIXTURE of media, which
        our layered stack only partly recovers. See holographic_creaturematerial.tissue_pbr."""
        from holographic.materials_and_texture import holographic_creaturematerial as _CM
        return _CM.tissue_pbr(str(tissue), scale=float(scale))

    def tissue_pbr_table(self, scale=1.0):
        """Every tissue material at once -- what a renderer or an editor's material picker
        enumerates. See holographic_creaturematerial.tissue_pbr_table."""
        from holographic.materials_and_texture import holographic_creaturematerial as _CM
        return _CM.tissue_pbr_table(scale=float(scale))

    def groom_region_map(self, vertices, regions, default=0.0, faces=None,
                         smooth=0):
        """A per-vertex groom attribute in [0,1] -- the surface-defined control that replaces
        groom_hair's axis-aligned bounds box. Density says WHERE hair grows, length says HOW
        LONG, and they are separate maps because a beard is not scalp hair. This is the
        production workflow (Houdini paints density and length as skin attributes and
        overrides hair generation with them). See holographic_groommap.region_map."""
        from holographic.mesh_and_geometry import holographic_groommap as _GM
        attr = _GM.region_map(vertices, list(regions), default=float(default))
        # OPTIONAL SURFACE BLUR, additive and default-off. smooth_map blurs
        # ALONG THE MESH rather than in space, so a region's edge follows the
        # surface instead of cutting through it -- the difference between a
        # beard that stops at the jaw and one that bleeds through it.
        # Added by EXTENDING this faculty rather than adding a sibling: the
        # duplicate audit caught a second groom_region_map shadowing this one,
        # which is dead code with a nicer docstring.
        if smooth and faces is not None:
            attr = _GM.smooth_map(vertices, faces, attr, self,
                                  iters=int(smooth))
        return attr

    def groom_smooth_map(self, vertices, faces, attr, iters=6):
        """Blur a groom attribute over the surface: a hard density edge reads as a shaved
        line, and real hairlines fade. See holographic_groommap.smooth_map."""
        from holographic.mesh_and_geometry import holographic_groommap as _GM
        return _GM.smooth_map(vertices, faces, attr, self, iters=int(iters))

    def groom_apply_maps(self, strands, vertices, density, length, base_length=1.0, seed=0,
                         length_range=(0.25, 1.0)):
        """Filter and rescale a groom by DENSITY and LENGTH maps -- one groom, many regions.
        MEASURED on a head: scalp length 0.67 vs beard 0.02, with density culling 14,000
        strands to the 1,399 that belong on the surface.
        KEPT NEGATIVE: this masks AFTER generation, so density is a filter, not a sampling
        density -- 4,000 strands at 0.3 coverage yields ~1,200, not 4,000 concentrated.
        See holographic_groommap.groom_with_maps."""
        from holographic.mesh_and_geometry import holographic_groommap as _GM
        return _GM.groom_with_maps(strands, vertices, density, length,
                                   base_length=float(base_length), seed=int(seed),
                                   length_range=tuple(length_range))

    def skin_sss_shade(self, base_rgb, ndl, thickness, sss_weight=0.75,
                       sss_radius=(1.0, 0.42, 0.28)):
        """Wrapped-diffuse SUBSURFACE shading for mammal skin. Skin is not Lambertian: light
        enters, scatters and leaves nearby, so the terminator wraps PAST 90 degrees and the
        light that travels furthest returns RED -- which is why ears and nostrils glow. The
        wrap width comes from tissue_pbr('skin')'s MEASURED scatter radius (1.0, 0.42, 0.28),
        not an invented tint. Honest scope: a wrap term is the standard real-time
        approximation, not a diffusion profile and not path-traced.
        See holographic_groommap.sss_shade."""
        from holographic.mesh_and_geometry import holographic_groommap as _GM
        return _GM.sss_shade(base_rgb, ndl, thickness, sss_weight=float(sss_weight),
                             sss_radius=tuple(sss_radius))

    def sfs_orient_convex(self, depth, mask=None):
        """Resolve the global CONVEX/CONCAVE flip in a shape-from-shading depth map -- the
        discrete ambiguity that renders a face as a CAVE. SOTA is explicit that "when
        lighting is unknown, a global shape has a discrete counterpart that corresponds to a
        global convex/concave flip"; for a head the centre must be nearer than the border, so
        the sign is decidable from that prior alone. Returns (depth, flipped).
        See holographic_sfsprior.orient_convex."""
        from holographic.mesh_and_geometry import holographic_sfsprior as _SP
        return _SP.orient_convex(depth, mask=mask)

    def sfs_debas_relief(self, depth, mask=None):
        """Remove the generalized bas-relief degrees of freedom -- "a three-parameter global
        ambiguity that corresponds to flattenings and tiltings of the global shape". Fits and
        subtracts a plane (the two tilts) and renormalises scale (the flatten); what survives
        is the part shape-from-shading actually determines.
        See holographic_sfsprior.debas_relief."""
        from holographic.mesh_and_geometry import holographic_sfsprior as _SP
        return _SP.debas_relief(depth, mask=mask)

    def sfs_blend_prior(self, depth, prior, mask=None, cut=6, iters=40):
        """Take the PRIOR's low frequencies and the SFS depth's high frequencies -- the
        concrete meaning of "regularize toward the prior". Shape-from-shading is reliable for
        FINE relief (a nostril crease, a brow furrow) and unreliable for GLOBAL shape (head
        or bowl?); a parametric prior is exactly the reverse. MEASURED on a real portrait:
        centre-to-edge relief 0.077 (nearly flat, unusable) -> 0.516 after blending.
        See holographic_sfsprior.blend_toward_prior."""
        from holographic.mesh_and_geometry import holographic_sfsprior as _SP
        return _SP.blend_toward_prior(depth, prior, mask=mask, cut=cut, iters=iters)

    def sfs_contour_normals(self, mask):
        """Normals along the OCCLUDING CONTOUR -- free and exact, because at a silhouette the
        surface normal is perpendicular to the view. SIRFS uses this same prior, and it is
        the only place in a single image where a normal is known without assuming anything
        about lighting. See holographic_sfsprior.contour_normals."""
        from holographic.mesh_and_geometry import holographic_sfsprior as _SP
        return _SP.contour_normals(mask)

    def fur_length_for(self, bounds, fraction=0.04):
        """Turn "short fur" into a DISTANCE in model units -- the control whose absence made
        every groom the wrong scale. ~0.02 stubble, 0.04 short fur, 0.10 thick coat, as a
        fraction of the model's largest extent, so the INTENT survives a resize.
        See holographic_furshell.fur_length_for."""
        from holographic.mesh_and_geometry import holographic_furshell as _FS
        return _FS.fur_length_for(bounds, fraction=float(fraction))

    def fur_shell(self, sdf, length, density_fn=None, length_fn=None, strand_scale=180.0,
                  seed=0, taper=2.0):
        """FUR AS AN SDF SHELL -- the region between the surface and an outward offset,
        following Kajiya & Kay's volumetric-texture formulation (SIGGRAPH 1989) and HISR's
        hard-SDF/soft-SDF hybrid. Returns f(P) -> occupancy in [0,1].

        FIXES THE TWO RECURRING GROOM FAULTS BY CONSTRUCTION: length is an SDF OFFSET so it
        is in model units and cannot be mis-scaled; coverage is a FIELD evaluated per point,
        so density per unit area is uniform and cannot clump (MEASURED: covered fraction
        0.876 vs 0.883 under 2x sampling). Fibre structure comes from a positional hash
        projected to the skin, so strands stay coherent along their length without storing
        any. COMPLEMENTS groom_hair rather than replacing it -- strands remain right for
        long styled hair; the shell is for short dense fur, stubble and beards.
        See holographic_furshell.fur_shell."""
        from holographic.mesh_and_geometry import holographic_furshell as _FS
        return _FS.fur_shell(sdf, float(length), density_fn=density_fn,
                             length_fn=length_fn, strand_scale=float(strand_scale),
                             seed=int(seed), taper=float(taper))

    def fur_shell_is_valid(self, length, reach):
        """Would this fur length make the shell self-intersect? Reuses L3's reach bound.
        USE A LOCAL REACH: measured on a real head, the GLOBAL reach was 0.0004 -- set by the
        crevice between the lips -- which would forbid fur on a scalp nowhere near it.
        See holographic_furshell.shell_is_valid."""
        from holographic.mesh_and_geometry import holographic_furshell as _FS
        return _FS.shell_is_valid(float(length), float(reach))

    def surface_lfs(self, sdf, points, normals=None, iters=24, r0=None, eps=0.002):
        """LOCAL FEATURE SIZE by the shrinking-ball algorithm -- the correct definition of how
        much room a surface has locally, and the right input to fur_shell_is_valid.

        SOTA: LFS is "the distance from a query point to its closest point on the medial
        axis", the reach is its MINIMUM (Federer), and the medial axis is the locus of
        MAXIMAL EMPTY BALLS. With an SDF the maximal inward tangent ball at p is found by
        iterating on the nearest surface point: r = |p-q|^2 / (2 (p-q).n).

        WHY NOT THE PAIRWISE FACING TEST: that asks whether two points face each other and
        are close, which geodesically ADJACENT points on a wrinkly surface also satisfy. It
        collapsed to 0.0003 on a real head -- refusing fur everywhere -- where the shrinking
        ball gives p05 0.0274 / median 0.0578 over the furred region, ~90x larger. A tangent
        ball cannot contain a neighbour on the same smooth patch, so adjacency is excluded BY
        CONSTRUCTION rather than by a threshold. VERIFIED against analytic planted truths:
        exact on slabs (0.00% error), 0.5% on a unit sphere.
        See holographic_offsetreach.shrinking_ball_lfs."""
        from holographic.mesh_and_geometry import holographic_offsetreach as _OR
        return _OR.shrinking_ball_lfs(sdf, points, normals=normals, iters=int(iters), r0=r0, eps=eps)

    def head_spec(self, params=None):
        """A skull skeleton FROM PARAMETERS -- the head equivalent of quadruped_spec, and the
        piece whose absence meant every head was 26 hand-typed magic numbers.

        Returns (segments, landmarks); the segments are exactly what convolution_field
        consumes, so a head is one call rather than forty lines of coordinates.

        WHY THE PARAMETERISATION IS CONSTRAINED, and it is the fix for the failure that
        recurred three times: fits kept converging to meaningless geometry (9 capsules at
        3.34x baseline that looked like blobs; a 44%-better fit that was a PANCAKE) because
        the objective had a null space. Proving an objective identifiable is hard;
        constraining the PARAMETERISATION so every point in it is anatomically well-formed is
        tractable and STRICTLY STRONGER -- a pancake stops being reachable, so no objective,
        however badly designed, can return one. Brow and chin are FRACTIONS of nose
        projection (before coupling, 292/400 random vectors put the chin or brow in FRONT of
        the nose); skull and face heights are fractions of skull WIDTH, which bounds the
        aspect ratio. MEASURED: 400/400 random parameter vectors satisfy every invariant, and
        absurd input clamps back into the manifold.
        See holographic_headspec.head_spec."""
        from holographic.mesh_and_geometry import holographic_headspec as _HS
        return _HS.head_spec(params)

    def head_invariants(self, params=None):
        """Do the anatomical invariants hold for these head parameters? crown>brow>eye>nose>
        mouth>chin, nose frontmost, ear behind eye, aspect ratio sane. The runtime mirror of
        lean/LeCoreHeadSpec.lean. See holographic_headspec.check_invariants."""
        from holographic.mesh_and_geometry import holographic_headspec as _HS
        return _HS.check_invariants(params)

    def lean_status(self):
        """Report the Lean 4 dependency tier without downloading or requiring anything.
        TIER 0 (always on, NumPy+stdlib): kernel, independent checker, Lean-source EMITTER,
        induction, fuzz oracle's non-Lean stages, proof memory at provenance 'checked'.
        TIER 1 (opt-in, ~1.3 GB installed): an external Lean binary -- buys exactly the
        'lean_verified' provenance tier. Install/remove via tools/install_lean.py (version
        and sha256 PINNED; a verifier downloaded unverified would be a joke at our own
        expense). Returns {"tier", "on_path", "local_install", "version", "pinned_version",
        "path_hint", "install_hint"}."""
        import importlib.util, os, sys
        spec = importlib.util.spec_from_file_location(
            "lecore_install_lean", os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "tools", "install_lean.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        st = mod.status()
        st["tier"] = 1 if st["version"] else 0
        st["install_hint"] = None if st["version"] else "python3 tools/install_lean.py"
        return st

    def wrap_to_field(self, vertices, faces, field, rounds=6, **kw):
        """Wrap a template mesh onto a target field while KEEPING IT A USABLE MESH.
        The point is the second half: a wrap that lands on the isosurface but self-intersects or inverts
        triangles has moved the problem rather than solved it, which is why wrap_quality exists beside it
        and this faculty returns both. See holographic_templatewrap.wrap_to_field."""
        from holographic.mesh_and_geometry.holographic_templatewrap import (
            wrap_to_field, wrap_quality)
        out = wrap_to_field(vertices, faces, field, rounds=rounds, **kw)
        v = out[0] if isinstance(out, tuple) else out
        return {"vertices": v, "quality": wrap_quality(v, faces, field)}

    def pose_is_safe(self, weights, angles, min_shrink=0.85):
        """Would this skinning pose PINCH? {ok, min_shrink, ...} rather than a guess.
        Linear blend skinning shrinks radially under twist -- |sum w_i R_i| < 1 -- and the failure is a
        collapsed wrist nobody attributes to the rig. max_safe_twist gives the angle where it starts.
        See holographic_skinbound.pose_is_safe / max_safe_twist."""
        from holographic.mesh_and_geometry.holographic_skinbound import (
            pose_is_safe, max_safe_twist)
        rep = dict(pose_is_safe(weights, angles, min_shrink=min_shrink))
        rep["max_safe_twist"] = float(max_safe_twist(weights,
                                                     min_shrink=min_shrink))
        return rep

    def shape_from_shading_prior(self, depth, mask=None, prior=None):
        """Fix the two degrees of freedom shape-from-shading CANNOT resolve on its own.
        SFS is ambiguous up to a global convex/concave flip and a bas-relief tilt -- both invisible to the
        shading term, so no amount of solving removes them. orient_convex picks the flip, debas_relief
        removes the tilt, and blend_toward_prior takes the prior's low frequencies with the SFS detail.
        See holographic_sfsprior."""
        from holographic.mesh_and_geometry import holographic_sfsprior as S
        # orient_convex returns (depth, flipped) -- the flag is the interesting
        # half, since "we flipped your surface inside out" is something the
        # caller should be told rather than have silently done. Unpacked here
        # instead of chained, which is what turned an ndarray into a 2-tuple and
        # made debas_relief raise on an inhomogeneous shape.
        d, flipped = S.orient_convex(depth, mask=mask)
        d = S.debas_relief(d, mask=mask)
        if prior is not None:
            d = S.blend_toward_prior(d, prior, mask=mask)
        return {"depth": d, "flipped": bool(flipped)}

    def make_corrective(self, mesh, source_vertex, radius, direction,
                        amplitude, falloff=None):
        """One blendshape TARGET as a LOCAL displacement, with locality something you can check.
        A corrective that leaks outside its radius fights every other shape in the basis, and the symptom
        is a rig that drifts as shapes stack. locality_report measures it instead of trusting the falloff.
        See holographic_blendbasis.make_corrective / locality_report."""
        from holographic.mesh_and_geometry.holographic_blendbasis import (
            make_corrective)
        kw = {} if falloff is None else {"falloff": falloff}
        return make_corrective(mesh, source_vertex, radius, direction,
                               amplitude, self, **kw)

    def _levers_base(self, problem=None, measured=None):
        """THE SIX LEVERS: what to do when you hit a measured wall, in cost order.
        The most reused idea in this engine lived only as PRACTICE -- named in NOTES, applied correctly by
        whoever had read them, findable by nobody else. Asked five ways a stranger would ask ("what do I do
        when I hit a wall", "ways to beat a capacity limit", "the six levers") find_capability returned
        advise_scale, crystal_habit and time_of_impact. THE MOST GENERALISABLE THING IN THE ENGINE WAS THE
        LEAST DISCOVERABLE, and an LLM driving leCore has exactly the problem the levers solve with no way
        to learn them: it hits a limit, concludes "impossible", and stops.
        RENAMED from levers() in sweep 63: part 19's seven-lever levers() EXTENDS this
        body via delegation, and two parts defining one public name is exactly the
        silent-shadow hazard test_unified_split guards. Composition now goes through
        this private base; the ONE public levers() lives in p19_lever7.
            1 cache locality -- bake once, sample O(1)      (prefix cache: 61x)
            2 partition into a commutative monoid            (bundling IS one)
            3 determinism instead of storage                 (registers from a seed)
            4 more dimensions                                (both branch arms + a gate)
            5 tile the domain under an orchestrator          (memory bounded by the tile)
            6 a measured limit is a TILE SIZE                (4,096 facts at 100% recall)
        EACH CARRIES ITS OWN MEASUREMENT AND ITS OWN COST, because a lever recommended without a case where
        it worked is advice, and one without a cost is a sales pitch. Ranking never hides a lever -- the
        doctrine is to walk them in cost order and stop at the first that applies, so the cheapest lever
        that works beats the best-matching one. Pass `measured` to get a wall report.
        See holographic_levers.LEVERS."""
        from holographic.agents_and_reasoning.holographic_levers import (
            levers, wall_report)
        if measured is not None:
            return wall_report(problem, measured=measured)
        return levers(problem)

    def ouroboros(self, dim=1024, seed=0, namespace="ouroboros"):
        """OUROBOROS: the closed memory loop -- read and write a running model's state with NO forward pass.
        The linear-attention state matrix inside a hybrid model IS a holographic memory (a theorem about its
        algebra, not a metaphor), so leCore can address it directly. Measured on the production algebra at
        read cosine 0.935 / write 0.951, with measured deletion and a PREDICTIVE capacity law -- 0.932
        predicted against 0.905 measured, 1.000 exact at reference scale.
        THE LOOP WAS REAL AND HAD NO NAME ON THE MIND. Every piece was wired (delta_write, delta_read,
        reserve, the capacity law) and "ouroboros" returned NOTHING from find_capability, so the one thing
        a caller would search for was the one thing absent.
        THREE VERBS WITH DIFFERENT PRICES, which is the part worth knowing before using it:
            write  adds content and PAYS CROSSTALK against everything already there
            pose   reshapes stored values as an isometry -- no crosstalk, but it moves what is there
            erase  is directional and exact, which is why registers survive 200k writes
        AND A KEPT NEGATIVE THAT TRAVELS WITH IT: rehearsing a state's own reads back into it DEGRADES it,
        0.767 -> 0.730. Consolidation is transcript-only BY API SHAPE for that reason -- the obvious
        alternative was measured and refuted. See holographic_keyreserve, unicron_self_heal."""
        from holographic.caching_and_storage.holographic_keyreserve import (
            reserve, delta_write, delta_read)
        import numpy as _np

        d = int(dim)
        state = {"S": _np.zeros((d, d)), "keys": reserve(d, 64, seed=int(seed)),
                 "n": 0, "namespace": str(namespace)}

        def write(slot, vec):
            state["S"] = delta_write(state["S"], state["keys"][int(slot)],
                                     _np.asarray(vec, float))
            state["n"] = max(state["n"], int(slot) + 1)
            return state["n"]

        def read(slot):
            v = delta_read(state["S"], state["keys"][int(slot)])
            n = float(_np.linalg.norm(v))
            return v / n if n > 1e-12 else v

        def erase(slot):
            # ERASE IS A WRITE OF ZERO, NOT A WRITE OF THE NEGATIVE. I wrote
            # -v first and MEASURED a residual norm of 1.0 -- because
            # delta_write is a GATED REPLACE, S <- a S (I - b k k^T) + b v k^T,
            # not an accumulation. The (I - k k^T) term already removes whatever
            # that key held; writing -v then stores -v in the slot it just
            # cleared, which reads back as the NEGATION of the fact rather than
            # its absence. That is the worst kind of wrong: a confident answer
            # pointing the opposite way.
            # THE ERASE TERM IS THE DIRECTIONAL PART OF THE UPDATE ITSELF, which
            # is exactly why the register file survives 200k writes -- and the
            # correct verb takes no value at all.
            state["S"] = delta_write(state["S"], state["keys"][int(slot)],
                                     _np.zeros(d))
            return state["n"]

        return {"state": state, "write": write, "read": read, "erase": erase,
                "capacity": lambda: len(state["keys"])}

    def optional_backends(self):
        """WHAT IS OPTIONAL, WHETHER IT IS HERE, AND THE ONE COMMAND THAT INSTALLS IT.
        leCore RUNS COMPLETE ON NumPy + stdlib, and that is verified rather than asserted: with cupy,
        numba, torch, scipy, sklearn, pyfftw, matplotlib, faiss and sympy ALL HARD-BLOCKED at the import
        hook, the mind boots, find_capability answers, the levers list, Ouroboros round-trips at cosine
        1.0000, and lean_export emits Lean 4 SOURCE WITHOUT LEAN INSTALLED. The accelerators buy speed on
        specific kernels and the verifier buys an external kernel's verdict; NEITHER BUYS A CAPABILITY.
        THE ASYMMETRY THIS CLOSES: Lean had a one-command installer with --status and --remove, while the
        GPU backends had a report that NAMED the pip command and no way to run it -- so the error message
        handed back a research task (which wheel does my driver take?). tools/install_gpu.py is the
        sibling: it reads nvidia-smi, picks cuda11x vs cuda12x, and REFUSES to offer a CUDA wheel with no
        driver visible, because that install yields a package which imports and finds no device -- harder
        to diagnose than an absence.
        Returns {lean, gpu} with `install` commands. Nothing here installs anything; ask, then run it."""
        out = {}
        try:
            from holographic.agents_and_reasoning import holographic_lean as _L
            out["lean"] = dict(_L.lean_status())
        except Exception as exc:
            out["lean"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        out["lean"]["install"] = "python3 tools/install_lean.py"
        out["lean"]["buys"] = ("the lean_verified provenance tier -- an "
                               "EXTERNAL kernel's verdict. Proving, checking "
                               "and EMITTING Lean source all work without it.")
        try:
            out["gpu"] = dict(self.gpu_report())
        except Exception as exc:
            out["gpu"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        out["gpu"]["install"] = "python3 tools/install_gpu.py --install"
        out["gpu"]["buys"] = ("speed on array-parallel kernels. cupy is "
                              "transparent and NVIDIA-only; wgpu is explicit "
                              "and vendor-neutral. Neither adds a capability.")
        out["core_requires"] = ["numpy", "python stdlib"]
        return out

    def semantic_to_scene(self, semantic, scene=None):
        """A SEMANTIC scene -> a RENDERABLE Scene document -- the bridge scene_from_image needed.
        scene_from_image returns a REPORT whose `scene` is a SemanticScene: objects as dicts of
        {label, shape, position, colour, material} that DESCRIBE a scene rather than carrying
        geometry. Every renderer wants objects with an SDF the tracer can .eval(), so
        render_scene_document(scene_from_image(img), camera) failed at three different depths --
        dict-vs-Scene, then list-vs-dict objects, then objects with no geometry at all.
        RULE 0 FOUND THE BRIDGE ALREADY BUILT: `realize_scene` turns parsed objects into
        renderables with an .eval sdf, and describe_to_scene has used it all along. THE CONVERTER
        WAS NEVER MISSING; THE DOOR FROM THE IMAGE SIDE TO IT WAS. This adds no geometry logic.
        ONE REAL WRINKLE: realize_scene's material names are SEMANTIC ("matte") while the library
        holds "matte_gray"/"matte_white", and its `material` dict is not what the shader reads.
        Unresolved names leave the material unset so the renderer's default applies -- a wrong
        material renders, a missing attribute does not.
        MEASURED: scene_from_image(img) -> semantic_to_scene -> render_scene_document produces a
        (18, 24, 3) frame with 1,296 lit pixels. See holographic_coerce.semantic_to_scene."""
        from holographic.io_and_interop.holographic_coerce import (
            semantic_to_scene)
        return semantic_to_scene(semantic, scene=scene)

    def read_image_section(self, section):
        """Read a `lecore.image` section back -- (image, meta), whoever wrote it.
        The WRITE half (container_kinds -> image_section) was wired and the READ half was not, which makes a
        canonical interchange kind half a format: an app could publish a texture and no app could consume it
        through a faculty. The orphan audit caught it as an unwired public function, which is exactly the
        signal that check exists to give. See holographic_container.read_image_section."""
        from holographic.io_and_interop.holographic_container import (
            read_image_section)
        return read_image_section(section)

    def boot_substrate_keys(self, weights, report=None):
        """Which tensors carry the boot record -- what an exporter must NOT narrow to bf16.
        A manifest larger than one embedding row SPILLS into the LOW BITS of surface weights and leaves a
        pointer in the row; bf16 keeps eight mantissa bits and the surface encoding lives below that. So a
        bf16 export erases the payload while the pointer survives, and boot() then finds a header promising
        bytes that are gone -- field-caught on a real Qwen3.5 where the install said ok and the audit said
        NO BOOT RECORD, both true about different bytes.
        Pass the write_boot report to include the SPILLED tensors (6 for a 146-byte spill); without it this
        returns the row's tensor only, which is correct for an unspilled record.
        See holographic_boot.boot_substrate_keys, export_portable(keep_f32=...)."""
        from holographic.io_and_interop.holographic_boot import (
            boot_substrate_keys)
        return boot_substrate_keys(weights, report=report)

    def learning_compact(self, dry_run=False):
        """Drop duplicate rows from the taught log -- repair for a partition that grew.

        WHY THIS EXISTS. register_doctrine was not idempotent for a long time: every
        boot, and every mount of a partition that already held doctrine, re-taught the
        same 14 facts. Both paths are fixed now, but THE FIX STOPS THE GROWTH AND DOES
        NOT RECLAIM WHAT WAS ALREADY WRITTEN. My own audit partition carried 381 rows
        of which 73 were distinct -- the same four doctrine facts twenty-three times
        each. Every partition that lived through the bug is permanently bloated and
        there was NO REPAIR PATH, which makes the bug's cost outlive the bug.

        Dedupe is by EXACT ROW CONTENT and keeps the FIRST occurrence, so the ordering
        the recall path may depend on is preserved. Returns
        {before, after, removed, distinct}. `dry_run=True` measures without changing
        anything -- run that first on a partition you care about.

        Call learning_save(root) afterwards to write the compacted store back; this
        touches live state only, exactly like teach().
        """
        lad = self.zoo["ladder"]
        log = list(getattr(lad, "taught_log", ()) or ())
        seen, keep = set(), []
        for row in log:
            key = repr(row)
            if key in seen:
                continue
            seen.add(key)
            keep.append(row)
        if not dry_run:
            lad.taught_log = keep
        return {"before": len(log), "after": len(keep),
                "removed": len(log) - len(keep), "distinct": len(seen),
                "dry_run": bool(dry_run)}

    def unicron_edit_health(self, base_weights, edited_weights, keys=(),
                            cond_budget=None):
        """Is a sequence of weight edits degrading the model? PRUNE's diagnostic, measured.

        THE PROBLEM THIS ANSWERS. An install is SEQUENTIAL KNOWLEDGE EDITING by
        another name: registers, then the HRNN ladder, then the router, then the
        memory index, then self-write, then state tracking -- each writing the same
        tensors. The literature on sequential editing (Ma et al. 2024, PRUNE; Gu et
        al. 2024, RECT; and the superimposed-noise-accumulation line) reports the
        same failure every time: each edit is fine ALONE and the model degrades as
        they compose, because the edited matrix's CONDITION NUMBER climbs and small
        input differences start producing large output differences.
        leCore's install already measures per-step perplexity and drift. Neither
        sees this: perplexity can hold while the matrix becomes numerically fragile,
        and drift measures HOW FAR the weights moved, not HOW BADLY CONDITIONED they
        became. THE INSTALL REPORTED "no step improved perplexity without making
        generation more repetitive" -- the exact symptom, with no instrument
        pointing at the cause.

        Returns per-tensor {cond_before, cond_after, ratio, rank_before, rank_after}
        plus a `worst` summary and, when `cond_budget` is given, `over_budget`: the
        tensors whose conditioning grew past it. PRUNE's own remedy is to CONSTRAIN
        the condition number during editing; this is the measurement that has to
        exist before any such constraint can be honest, and it is deliberately
        diagnostic-only -- nothing here changes a weight.

        NumPy only (np.linalg.svd), and it samples: full SVD of every tensor in a
        24-layer model is minutes, so 2-D tensors are capped by `_COND_MAX_DIM` and
        larger ones are measured on a deterministic slice.
        """
        import numpy as _np

        _COND_MAX_DIM = 512
        names = list(keys) or [k for k in base_weights
                               if k in edited_weights
                               and _np.asarray(base_weights[k]).ndim == 2]
        out, worst = {}, None
        for name in names:
            a = _np.asarray(base_weights[name], dtype=_np.float64)
            b = _np.asarray(edited_weights[name], dtype=_np.float64)
            if a.ndim != 2 or a.shape != b.shape:
                continue
            if max(a.shape) > _COND_MAX_DIM:
                # deterministic slice -- a seeded sample would make the diagnostic
                # itself a source of run-to-run variation, which is the last thing
                # a degradation check should have.
                a = a[:_COND_MAX_DIM, :_COND_MAX_DIM]
                b = b[:_COND_MAX_DIM, :_COND_MAX_DIM]
            try:
                sa = _np.linalg.svd(a, compute_uv=False)
                sb = _np.linalg.svd(b, compute_uv=False)
            except _np.linalg.LinAlgError:
                continue
            tol_a = max(a.shape) * (sa[0] if sa.size else 0.0) * 2.22e-16
            tol_b = max(b.shape) * (sb[0] if sb.size else 0.0) * 2.22e-16
            ca = float(sa[0] / sa[-1]) if sa.size and sa[-1] > 0 else float("inf")
            cb = float(sb[0] / sb[-1]) if sb.size and sb[-1] > 0 else float("inf")
            rec = {"cond_before": ca, "cond_after": cb,
                   "ratio": (cb / ca) if ca not in (0.0, float("inf")) else float("inf"),
                   "rank_before": int((sa > tol_a).sum()),
                   "rank_after": int((sb > tol_b).sum())}
            out[name] = rec
            if worst is None or rec["ratio"] > out[worst]["ratio"]:
                worst = name
        rep = {"tensors": out, "worst": worst,
               "worst_ratio": out[worst]["ratio"] if worst else 1.0}
        if cond_budget is not None:
            rep["over_budget"] = sorted(
                n for n, r in out.items() if r["cond_after"] > float(cond_budget))
        return rep

    def table_vacuum(self, db, qualified, dry_run=False):
        """Reclaim tombstoned rows in one table and rebuild its indexes.

        UPDATE in this engine is TOMBSTONE-AND-REINSERT -- which is what keeps the
        indexes correct without an update hook -- so a table that is written to
        only ever GROWS. Measured: 400 rows became 1,260 after five updates, with
        1,260 index entries, and every later SELECT scanned the dead ones.
        The Table method existed only on the object; the catalog reflects the
        MIND'S surface, so without this faculty "my table keeps growing" reached
        mesh_repair and learning_compact. Same lesson as autoboot being dark.

        Returns {before, after, removed}. `dry_run=True` measures without touching
        anything. See holographic_query.UserTable.vacuum."""
        return db.resolve(qualified).vacuum(dry_run=dry_run)

    def db_vacuum_idle(self, db, threshold=0.25):
        """Vacuum every table in a database whose tombstone share exceeds `threshold`.

        Call when the database is IDLE, like cool_idle -- vacuum renumbers rows and
        rebuilds indexes. NOT automatic on write, deliberately: a vacuum inside an
        INSERT would renumber rows under a caller holding indices and make one
        unlucky write pay for everyone else's churn.
        The threshold exists because ORDINARY CHURN IS CHEAP -- measured, 5% dead
        costs ~5% on a scan and nothing on an indexed lookup. What earns a vacuum
        is repeated updates to the SAME rows, where tombstone-and-reinsert
        compounds to 75% dead and a 3.5x scan slowdown.
        Returns {tables, rows_removed}. See holographic_query.Database.vacuum_idle."""
        return db.vacuum_idle(threshold=threshold)

    def teach_about(self, question, answer, paths, root="."):
        """Teach a fact AND record which files it describes, so it can go stale loudly.

        THE PROBLEM. A partition accumulates facts about a codebase -- "vacuum
        rebuilds every index", "the journal logs after the write succeeds" -- and
        the codebase moves. Nothing connects the two, so a fact stays at tier T0
        and confidently answers about code that changed months ago. STALE MEMORY
        IS WORSE THAN NO MEMORY: it is indistinguishable from current memory right
        up to the moment it is wrong.

        Both halves already existed and were never joined: `teach` stores the fact,
        and FileMap/ingest_files records size+mtime+hash per file with a
        `changed()` that re-stats the disk. This records the file fingerprints
        alongside the fact so `stale_facts()` can compare them later.

        Fingerprints are (size, mtime, sha256-prefix) -- hashlib, not hash(), so
        they are stable across processes. Returns the taught record plus the
        fingerprints stored."""
        import hashlib
        import os

        marks = {}
        for rel in ([paths] if isinstance(paths, str) else list(paths)):
            full = rel if os.path.isabs(rel) else os.path.join(root, rel)
            try:
                st = os.stat(full)
                with open(full, "rb") as fh:
                    dig = hashlib.sha256(fh.read()).hexdigest()[:16]
                marks[rel] = {"size": st.st_size, "mtime": int(st.st_mtime),
                              "sha": dig}
            except OSError:
                marks[rel] = None            # missing NOW is itself a finding
        rec = self.teach(question, answer)
        book = getattr(self, "_fact_files", None)
        if book is None:
            book = self._fact_files = {}
        book[str(question)] = marks
        return {"taught": rec, "files": marks}

    def stale_facts(self, root="."):
        """Which taught facts describe files that have CHANGED since they were taught.

        Returns {stale, missing, fresh, unknown} -- lists of questions. `missing`
        is separated from `stale` on purpose: a file that was DELETED is a
        different problem from one that was EDITED, and lumping them together
        makes the deletions invisible in a long list.
        `unknown` counts facts taught without file references at all, which is the
        honest report of how much of a partition this check cannot speak for."""
        import hashlib
        import os

        book = getattr(self, "_fact_files", None) or {}
        out = {"stale": [], "missing": [], "fresh": [], "unknown": 0}
        for q, marks in book.items():
            if not marks:
                out["unknown"] += 1
                continue
            verdict = "fresh"
            for rel, mark in marks.items():
                full = rel if os.path.isabs(rel) else os.path.join(root, rel)
                if mark is None or not os.path.exists(full):
                    verdict = "missing"
                    break
                try:
                    with open(full, "rb") as fh:
                        dig = hashlib.sha256(fh.read()).hexdigest()[:16]
                except OSError:
                    verdict = "missing"
                    break
                # HASH, NOT MTIME. A checkout or a touch moves mtime without
                # changing content, and reporting those as stale trains a reader
                # to ignore the report -- which is how a staleness check dies.
                if dig != mark.get("sha"):
                    verdict = "stale"
            out[verdict].append(q)
        return out

    def explain_pair(self, x1, x2):
        """WHY two things are similar -- the per-role verdict, not just a cosine.

        CONVERGENT FIX, RECONCILED (merge sweep 2): upstream and this line both
        cured the explain-name collision independently -- upstream added
        explain_pair (delegating to the shadowed p02 body), this line RENAMED
        that body to explain_similarity. One body stands (p02's
        explain_similarity); this is upstream's canonical name delegating to
        it, and explain_similarity remains callable. `explain` still resolves
        to the topic form -- no existing caller changes."""
        return self.explain_similarity(x1, x2)

    def check_math(self, text, tolerance=1e-9):
        """Verify every arithmetic claim in `text` by ACTUALLY COMPUTING it.

        A model produces the token most likely to follow "137 * 4 = ", which is
        not multiplication. It is right often enough to be trusted and wrong
        often enough to matter, and NOTHING IN THE OUTPUT TELLS THE TWO APART --
        a wrong sum is written with the confidence of a right one.
        This finds `expression = result` claims, evaluates them over an ast that
        permits arithmetic and refuses names, calls and attributes (so model text
        is never eval'd), and reports what disagrees.

        Returns {ok, checked, wrong, unverifiable, claims}. UNVERIFIABLE IS NOT
        WRONG and is counted separately: "I could not check this" and "this is
        false" are different results, and a checker that conflates them is
        untrustworthy in both directions.
        See holographic_mathcheck."""
        from holographic.agents_and_reasoning.holographic_mathcheck import check
        return check(text, tolerance=tolerance)

    def do_math(self, expr):
        """Evaluate ONE arithmetic expression here, rather than asking a model.

        The cheapest way to stop a model doing arithmetic badly is to not ask it.
        Use the model (or the substrate) to find WHICH expression to compute, and
        then compute it. Raises Unverifiable rather than guessing.
        See holographic_mathcheck.evaluate."""
        from holographic.agents_and_reasoning.holographic_mathcheck import evaluate
        return evaluate(expr)

    def remote_llm(self, url=None, model=None, api_key=None, **kw):
        """A `text -> text` callable for an LLM in ANOTHER PROCESS (OpenAI-compatible).

        Every rung seam in this engine -- attach_llm, agent_bridge, autoboot's
        llm= -- takes a LOCAL CALLABLE, which fits an in-process model and nothing
        else. The common deployment is Claude or ChatGPT behind OpenWebUI,
        openzoo, ollama or a vendor API, in a different process. This returns
        exactly the callable those seams already accept, so nothing else changed.
        Reads LECORE_LLM_URL / LECORE_LLM_MODEL / LECORE_LLM_KEY (OPENAI_* too).
        See holographic_remotellm.remote_llm."""
        from holographic.io_and_interop.holographic_remotellm import remote_llm
        return remote_llm(url=url, model=model, api_key=api_key, **kw)

    def logic_encode_atom(self, pred, args=()):
        """Encode a ground atom into THIS mind's hypervector space (predicate bound with
        role-tagged arguments, via the engine's own derived_atom/bind/bundle -- one algebra,
        another costume), so fact bases join the substrate and similarity search is a matmul.
        Returns the raw vector. See holographic_lean.encode_atom."""
        from holographic.agents_and_reasoning import holographic_lean as _L
        from holographic.agents_and_reasoning.holographic_ai import derived_atom, bind, bundle
        import numpy as _np
        dim, seed = self.encoder.dim, self.encoder.seed
        sym = lambda name: derived_atom(seed, "lean:" + name, dim)
        bnd = lambda vs: bundle(_np.stack(vs))
        return _L.encode_atom(_L.Atom(pred, tuple(args)), sym, bind, bnd)


def _selftest():
    """The shared part contract (check_part), plus the full faculty loop proved end-to-end
    through a real mind: prove -> check -> export, an honest None, and an encode."""
    n = check_part("holographic.unified.holographic_unified_p18_lean", "_UnifiedPart18")
    from lecore import UnifiedMind as _UM
    m = _UM(dim=64, seed=0)
    rules = [{"head": ["human", ["socrates"]], "name": "h_soc"},
             {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "mortality"}]
    p = m.logic_prove(["mortal", ["socrates"]], rules)
    assert p is not None and m.logic_check_proof(p, rules)
    out = m.lean_export(["mortal", ["socrates"]], rules, theorem_name="soc")
    assert out["ok"] and "theorem soc : mortal socrates :=" in out["lean"]
    assert m.logic_prove(["mortal", ["zeus"]], rules) is None  # the honest None, pinned
    v = m.logic_encode_atom("human", ["socrates"])
    assert v.shape == (64,)
    print("OK: unified p18 (lean/logic) part contract holds over %d facade defs; "
          "faculties proved end-to-end on the assembled mind" % n)


if __name__ == "__main__":
    _selftest()

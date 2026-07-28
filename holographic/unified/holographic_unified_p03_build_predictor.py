"""Part 03 of UnifiedMind's faculty surface -- 115 methods, build_predictor .. denoise.

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


class _UnifiedPart03:

    def build_predictor(self, order=2, reinforce_threshold=0.15, novelty_threshold=0.55):
        """Give the mind a PREDICTIVE LOOP over symbol sequences: it anticipates
        the next symbol from recent context, measures its surprise, and learns
        error-gated (see holographic_predictive). This is the active layer on top
        of storage -- the mind now expects, notices when it is wrong, and adapts.
        Returns self."""
        from holographic.agents_and_reasoning.holographic_predictive import PredictiveMemory
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
        from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor
        stream = [w for s in sentences for w in (s if isinstance(s, list) else s.split())]
        self._meaning_pred = (MeaningPredictor(dim=self.dim, order=order, seed=0)
                              .fit_space(sentences, window=window)
                              .fit_transitions(stream))
        # calibrate a structure verifier on the same corpus, so the mind can PROVE
        # whether a sequence carries structure and steer generation to stay in it
        from holographic.misc.holographic_structure import StructureVerifier
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
        from holographic.misc.holographic_structure import steered_generate
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
        from holographic.misc.holographic_partition import SharedMind
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
            from holographic.misc.holographic_codec import PredictiveCodec
            self._codec = PredictiveCodec(self._meaning_pred)
        toks = tokens.split() if isinstance(tokens, str) else list(tokens)
        return {"code": self._codec.compress(toks), "cost": self._codec.cost(toks),
                "lossless": self._codec.roundtrip_ok(toks)}

    def decompress_lossless(self, code):
        """Recover the exact original sequence from a code produced by
        compress_lossless, by replaying the shared predictor."""
        if not hasattr(self, "_codec"):
            from holographic.misc.holographic_codec import PredictiveCodec
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
        from holographic.io_and_interop.holographic_video import HolographicVideo
        return HolographicVideo(dim=dim or self.dim, key_keep=key_keep, res_keep=res_keep, bits=bits,
                                gop_len=gop_len, max_shift=max_shift, seed=seed)

    def attribute_sources(self, tokens, sources, topk=15, order=2):
        """Source attribution: trace which stored material a passage drew on. sources
        is {name: token_stream}; returns a provenance distribution over those sources
        from the predictor's resonance couplings (holographic_codec.SourceAttributor)."""
        from holographic.misc.holographic_codec import SourceAttributor
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
        from holographic.misc.holographic_traverse import gated_traverse
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
        from holographic.rendering.holographic_occlusion import occlusion_recall
        if cache and gram is None:
            if not hasattr(self, "_gram_cache"):
                from holographic.rendering.holographic_occlusion import GramCache
                self._gram_cache = GramCache()
            gram = self._gram_cache.gram(codebook)         # RAM-1: build once, reuse across cues (id-keyed, GC-safe)
        return occlusion_recall(cue, codebook, m=m, min_share=min_share, gram=gram)

    def build_occlusion_gram(self, codebook):
        """The cached Gram matrix for occlusion_recall's fast path (holographic_occlusion, SPEED-1) -- G = codebook @
        codebook.T, computed ONCE and reused across cues. Pass it as occlusion_recall(..., gram=G): the readout then
        updates correlations through a Gram column per pick instead of rescanning the dictionary (O(N) vs O(N*D) per
        step). Pays whenever the same codebook is queried more than once (the engine's normal case); costs O(N^2)
        memory -- the storage-for-speed trade (Rubinstein-Zibulevsky-Elad 2008, Batch-OMP)."""
        from holographic.rendering.holographic_occlusion import build_gram
        return build_gram(codebook)

    def build_occlusion_forest(self, codebook, n_trees=4, leaf_size=64, seed=0):
        """Build a HoloForest over `codebook` for forest-routed occlusion selection (holographic_occlusion, SPEED-2,
        the N-factor) -- built once, reused across cues like the SPEED-1 Gram. Pass it as occlusion_recall_forest(...,
        forest=F). See occlusion_recall_forest for the measured trade-off (it is a kept negative at current scale)."""
        from holographic.rendering.holographic_occlusion import build_occlusion_forest
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
        from holographic.rendering.holographic_occlusion import occlusion_recall_forest
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
        from holographic.sampling_and_signal.holographic_iht import iht_recall
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
        from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall
        return cosamp_recall(cue, codebook, K, iters=iters, tol=tol, stats=stats)

    def wods_solve(self, interface, boundary_value, lo=(0.0, 0.0), hi=(1.0, 1.0), capture_r=None,
                   walks=256, max_steps=64, eps=1e-3, seed=0, iters=500, tol=1e-12, stats=None):
        """WALK ON DECOMPOSED SUBDOMAINS (holographic_wods, WoDS-1) -- solve Laplace by having short random
        walks estimate LOCAL COUPLING OPERATORS between interface points, then solving the resulting sparse
        system DETERMINISTICALLY with the engine's shared conjugate gradient. The shipped pointwise solvers
        (wos / wost) send long walks all the way to the boundary for every query; here a walk only has to
        reach a neighbour, and the long-range structure is resolved by exact linear algebra instead of by
        sampling. Two of leCore's own levers stacked: partition, then tile under an orchestrator.
        MEASURED vs pure WoS at matched budget (unit square, u=x^2-y^2, 10 seeds): 0.043 vs 0.075 mean abs
        error at 32 walks, 0.034 vs 0.048 at 64 -- roughly HALF the error at a tight budget.
        KEPT NEGATIVES, both from measurement: (1) IT IS BIASED. The capture radius is a discretisation, and
        no number of walks removes it -- only a finer interface does; pure WoS is unbiased and OVERTAKES at
        high budgets (0.0238 vs 0.0260 at 256 walks). (2) THE PAPER'S LOW-VARIANCE HEADLINE IS NOT
        REPRODUCED HERE -- pure WoS often has the smaller across-seed spread (0.0021 vs 0.0042). What this
        earns is SAMPLE EFFICIENCY, not variance reduction. Scope: 2-D axis-aligned rectangle with Dirichlet
        data; use wost for general SDFs and Neumann."""
        from holographic.simulation_and_physics.holographic_wods import solve_decomposed
        return solve_decomposed(interface, boundary_value, lo=lo, hi=hi, capture_r=capture_r, walks=walks,
                                max_steps=max_steps, eps=eps, seed=seed, iters=iters, tol=tol, stats=stats)

    def wods_interface_grid(self, nx, ny, lo=(0.0, 0.0), hi=(1.0, 1.0)):
        """The interior lattice nodes a WoDS solve treats as unknowns (holographic_wods). Boundary nodes are
        excluded because their values are given data. Fixed row-major order, so the linear system's indexing
        is reproducible run to run."""
        from holographic.simulation_and_physics.holographic_wods import interface_grid
        return interface_grid(nx, ny, lo=lo, hi=hi)

    def wods_measure_vs_pure_wos(self, nx=6, ny=6, walks=128, seeds=5, seed0=0):
        """MEASURE WoDS against the SHIPPED pointwise WoS solver at matched walk budget on a problem with a
        known analytic answer (holographic_wods.measure_vs_pure_wos). Wired because two claims written for
        this module BEFORE measuring were refuted by it: the paper's low-variance headline does not
        reproduce on this simplified operator, and the accuracy advantage is not monotone -- unbiased WoS
        overtakes at high budgets. Returns wods_err/sd/ms and wos_err/sd/ms so both halves stay visible."""
        from holographic.simulation_and_physics.holographic_wods import measure_vs_pure_wos
        return measure_vs_pure_wos(nx=nx, ny=ny, walks=walks, seeds=seeds, seed0=seed0)

    def advise_restarts(self, codebooks, targets=(0.95,), budgets=(4, 16, 64, 256), iters=300,
                        trials=8, seed=0):
        """HOW MANY RESTARTS DOES THIS FACTORING PROBLEM NEED (holographic_resonator.advise_restarts) --
        measured on YOUR codebooks, not looked up. The F>=4 'capacity cliff' is a SEARCH BUDGET: the same
        network, dimension and codebooks solve at restarts=256 what they fail at restarts=4 (25% -> 100% at
        N=2048, V=16, F=4).
        WHY THE DEFAULT WAS NOT SIMPLY RAISED -- measured, and it is the cost profile, not correctness:
            N=2048 V=16 F=4       r=20      r=64      r=256
            solvable             1071 ms   1460 ms    1465 ms
            UNSOLVABLE           1488 ms   4743 ms   19439 ms
        A bigger cap is nearly free when an answer exists (factor returns at the restart that succeeds) and
        costs 13x when there is NONE, because a refusal must exhaust the budget to be a refusal. The price of
        a bigger default falls entirely on the problems that were never going to work. The sequence is also
        PREFIX-STABLE (restarts=64 returns the identical factors AND restart count as restarts=20 on every
        already-solved case), so raising it could not flip an existing answer -- the objection is cost alone.
        Returns the smallest budget reaching each target rate, with the full curve; restarts=None means no
        budget tried reached it, which is itself the answer."""
        from holographic.misc.holographic_resonator import advise_restarts
        return advise_restarts(codebooks, targets=targets, budgets=budgets, iters=iters,
                               trials=trials, seed=seed)

    def agent_benchmark(self, n_has=60, n_no=20, seed=0, z_min=0.8):
        """THE AGENT-SOCKET BENCHMARK (holographic_agentbench, BENCH-1). PRE-REGISTERED PRIMARY METRIC:
        FALSE-ACTION RATE ON A NO-TOOL SET -- the number the reference system (97.9% on capability records)
        does not publish, and the one tool-calling benchmarks are documented as missing.
        THE NO-TOOL SET IS BUILT BY REMOVAL, which is what makes it worth running: each task is a REAL
        capability's own author-written alias, asked against an index REBUILT WITHOUT that capability. So it
        is a coherent, idiomatic request with genuinely nothing behind it, and every near neighbour is still
        present to tempt a match -- strictly harder than word salad, which is merely incoherent.
        MEASURED, 60 has-tool / 20 no-tool, seeded: resolution 100.0%, FALSE-ACTION RATE 0.0%, run-to-run
        variance ZERO at max_rung=5, model calls 0.
        KEPT NEGATIVE, reported because the plan demands it: RUNGS 1-5 FIRED ON 0/60 BODIES. Rung 0 answered
        everything. That is not 'ceremony around an LLM call' -- no model was reached at all -- but it does
        mean rungs 1-3 are UNEXERCISED and their gates unproven on real traffic. The fixture is free-text-to-
        faculty requests, which is precisely rung 0's job, so the FIXTURE may be the limiting factor rather
        than the ladder; a fixture with typed or vector goals would be needed to exercise the rest.
        Returns resolution_rate, false_action_rate, rung_distribution, model_calls and the raw counts."""
        from holographic.agents_and_reasoning.holographic_agentbench import run_benchmark
        return run_benchmark(self, n_has=n_has, n_no=n_no, seed=seed, z_min=z_min)

    def catalog_without(self, names):
        """A Catalog holding every capability EXCEPT `names` (holographic_agentbench) -- the instrument
        behind the benchmark's no-tool arm. Removes the answer while leaving every distractor in place, so a
        task can be asked in a world where nothing serves it. REBUILT, never mutated: a benchmark that damages
        the system it measures is measuring something else by the second run."""
        from holographic.agents_and_reasoning.holographic_agentbench import catalog_without
        return catalog_without(names)

    def expand_query(self, query, llm=None, min_faithfulness=0.5, z_min=0.8, seed=0):
        """MODEL-PROPOSED QUERY EXPANSION, GATED ON FAITHFULNESS (holographic_queryexpand, EXPAND-1). Asks a
        model to rewrite a request into catalog vocabulary before retrieval -- the model proposes, the engine
        disposes -- and REFUSES the rewrite unless it retains the original's meaning.
        WHY FAITHFULNESS AND NOT JUST THE NULL, which was the stated gate: MEASURED FIRST, random padding
        cannot smuggle a no-tool query past the router (0/8) because the null is built at MATCHED TOKEN
        COUNT, so lengthening a query lengthens its null and dilution scores WORSE. But a TARGETED rewrite
        sails through (1/3 smuggled: 'purple monkey dishwasher' -> 'smooth a bumpy mesh surface' routes
        confidently), because the rewrite IS a perfectly valid query. A NULL CANNOT DETECT INFIDELITY, ONLY
        IRRELEVANCE. So the primary gate is overlap with the ORIGINAL: an expansion sharing no content word
        with what the user asked is a SUBSTITUTION, and is refused however well it scores.
        Both gates apply, not either -- the expansion must also still clear the null floor.
        Returns {query, expanded, faithfulness, proposal, why}; the ORIGINAL query is returned whenever the
        expansion is refused, so a caller can use the result unconditionally and a bad model degrades to
        today's behaviour rather than to a wrong answer."""
        from holographic.semantic_router.holographic_queryexpand import expand_query
        fn = llm if llm is not None else getattr(self, "_llm", None)
        if fn is None:
            raise RuntimeError("no LLM attached -- call mind.attach_llm(callable) first, or pass llm=")
        return expand_query(self, query, fn, min_faithfulness=min_faithfulness, z_min=z_min, seed=seed)

    def agent_loop(self, task, llm=None, max_steps=6, z_min=0.8, k_tools=6, seed=0):
        """THE IN-PROCESS TOOL-USE LOOP (holographic_agentloop, LOOP-1). Hands a model the RELEVANT slice of
        the manifest, parses a tool call from its reply, dispatches through invoke(), feeds the result back
        and iterates. Over HTTP this already worked (/tools + /invoke); in process there was no loop, so
        every embedder wrote their own -- routing around the choke point invoke() exists to be.
        THE DIFFERENTIATOR IS NOT THE LOOP, IT IS THE GATE BELOW IT. Before any step runs, route_or_abstain
        scores the task against a null built from the catalog's own vocabulary at matched token count; below
        the floor the loop REFUSES AND SAYS WHY, and THE MODEL IS NEVER CONSULTED. Measured with a scripted
        stub that always claims completion (worst case): has-tool 20/20 completed, no-tool 0/20 -- FALSE-
        ACTION RATE 0%, with 20/20 refused before the model was reached. The stub never abstains, so every
        refusal is the engine, not the model's restraint -- which matters because models measurably do NOT
        abstain reliably for themselves.
        Refuses non-finite args (json parses bare NaN, so a model can emit one), refuses any tool outside the
        offered manifest, and NEVER GUESSES at an unparsed reply. Args are recorded as a blake2b digest plus
        a short repr, never the live object: a live object in a job's args once crashed a worker after the
        job had already succeeded. Returns {done, refused, answer, why, steps, gate}.
        Pass llm= or attach one first with attach_llm()."""
        from holographic.agents_and_reasoning.holographic_agentloop import AgentLoop
        fn = llm if llm is not None else getattr(self, "_llm", None)
        if fn is None:
            raise RuntimeError("no LLM attached -- call mind.attach_llm(callable) first, or pass llm=")
        return AgentLoop(self, fn, max_steps=max_steps, z_min=z_min, k_tools=k_tools, seed=seed).run(task)

    def llm_tool(self, name="llm", description="", in_type="text", out_type="text", on_error="raise",
                 llm=None):
        """MAKE THE ATTACHED LLM PLANNER-VISIBLE (holographic_orchestrator.register_llm). attach_llm() sets
        this mind's _llm and wires a bus bridge, but it does NOT register the model as a tool -- so
        Planner.plan, optimize_toolchain, CircuitBreaker and SkeletonLibrary have all been blind to it. The
        one tool that can do fuzzy language work was the one the planner could not reach. This closes that.
        Registers on THIS mind's orchestrator and returns the Tool. Pass llm= to register a callable directly
        without attaching it; otherwise the attached model is used and a missing one raises rather than
        silently registering nothing.
        WHY IT MATTERS BEYOND PLUMBING: a registered model can be FAILED OVER AWAY FROM. Wrap a flaky model,
        watch its breaker open, watch the planner reroute onto a deterministic faculty -- which a system
        whose only mechanism IS the model structurally cannot do, because you cannot fail over away from the
        thing doing your planning.
        on_error='raise' (default) lets failures reach the breaker; 'empty' degrades instead, and HIDES the
        failure from the breaker, which is why it is not the default."""
        fn = llm if llm is not None else getattr(self, "_llm", None)
        if fn is None:
            raise RuntimeError("no LLM attached -- call mind.attach_llm(callable) first, or pass llm=")
        return self.orchestrator.register_llm(fn, name=name, description=description, in_type=in_type,
                                              out_type=out_type, on_error=on_error)

    def bundle_capacity(self, dim=None, method="cosamp", floor=0.95, seeds=range(4), codebook=None,
                        ratios=(0.02, 0.05, 0.10, 0.17, 0.25, 0.33, 0.40)):
        """HOW MANY THINGS FIT IN A BUNDLE -- answered as a MEASURED LOAD RATIO with its variables attached,
        never as a constant (holographic_capacity, CAP-1). The folklore answer ("20-32 instructions") was a
        LINEAR-READOUT ARTIFACT: measured here, naive cosine readout holds safe M/D = 0.02 while cosamp/amp
        hold 0.17 at floor 0.95 -- 44 items at D=256, 87 at D=512, 174 at D=1024, an 8.7x difference the
        constant hid. The safe ratio COLLAPSES across dims (0.17 at every D tested), which is why capacity
        is a RATIO m/D, not a count: per-item signal-to-crosstalk is governed by m/D.
        MEASURES AT CALL TIME rather than shipping a table, because the reference numbers hold for an
        INCOHERENT dictionary only -- coherence inverts the method ranking (AMP collapses to 0.052 where
        CoSaMP holds 1.000 at coherence 0.5, measured). Pass codebook= to answer on YOUR atoms.
        The gate is mean MINUS sd across seeds: a capacity only the lucky seed reaches is not a capacity.
        Returns {capacity, safe_ratio, method, dim, floor, curve} -- the curve travels with the number so it
        cannot be quoted without the configuration that produced it. dim defaults to this mind's."""
        from holographic.sampling_and_signal.holographic_capacity import bundle_capacity
        return bundle_capacity(int(dim or self.dim), method=method, floor=floor, seeds=seeds,
                               codebook=codebook, ratios=ratios)

    def cleanup_batch(self, codebook, queries, backend=None, workgroup=64):
        """CLEAN UP MANY CUES AT ONCE -> (indices, scores) (holographic_capacity). The missing `UP` direction
        of cleanup, and IT PAYS ON THE CPU ALONE: one (K,D)x(D,M) matmul instead of K separate matvecs is
        2.58x at K=32, 5.36x at K=64 and 5.92x at K=128 (measured, no device involved) -- BLAS getting one
        big matmul rather than K small ones. The argmax is microseconds either way.
        backend='wgsl' routes the same computation to any GPU. DEFAULT OFF, DELIBERATELY: the host<->device
        crossover has never been measured on real hardware, so enabling it by default would act on arithmetic
        rather than a measurement, and the one thing worse than not using a device is using it ON A GUESS.
        The seam exists so somebody WITH a device can measure it without editing the engine.
        Indices resolve by LOWEST INDEX on both paths, so the backend cannot change which atom wins a tie."""
        from holographic.sampling_and_signal.holographic_capacity import cleanup_batch
        return cleanup_batch(codebook, queries, backend=backend, workgroup=workgroup)

    def drop_budget(self, dim=None, n_items=1, safe_ratio=0.02, floor=0.95):
        """HOW MANY SLOTS CAN BE DROPPED under memory pressure and still recall (holographic_capacity, W6).
        Dropping slots to save memory reduces the EFFECTIVE DIMENSION, so the constraint is the load-ratio
        law already measured here: recall holds while n_items / (keep * dim) stays under the safe ratio. NO
        NEW THEORY WAS NEEDED -- verified across five configurations, every one at or below ~0.02 holding and
        every one above degrading.
        A CORRECTION KEPT LOUD: the README's 100%-recall-at-40%-slots-destroyed figure is about DAMAGE, where
        slots are ZEROED and NO MEMORY IS SAVED. It does NOT transfer to memory saving -- TRUNCATING to 40%
        of dimensions at the same load gives 85%, not 100%, because a zeroed slot still occupies its
        dimension in the readout while a dropped one does not. Corruption-robustness and a memory budget are
        DIFFERENT QUANTITIES.
        safe_ratio defaults to the LINEAR-readout figure; pass bundle_capacity's measured ratio for your
        decoder (0.17 for cosamp/amp) to drop far more aggressively."""
        from holographic.sampling_and_signal.holographic_capacity import drop_budget
        return drop_budget(int(dim or self.dim), n_items, safe_ratio=safe_ratio, floor=floor)

    def measure_recovery_curve(self, dim=None, method="cosamp", ratios=(0.05, 0.10, 0.17, 0.25, 0.33),
                               n_atoms=None, seeds=range(4), codebook=None):
        """Support-recovery F1 as a function of LOAD RATIO M/D, measured live for one readout method
        (holographic_capacity). The raw curve behind bundle_capacity, for when the shape matters more than
        the single safe number -- e.g. reading where a method's phase transition sits, or comparing readouts
        on your own codebook. Returns [{ratio, m, f1_mean, f1_sd}] across seeds."""
        from holographic.sampling_and_signal.holographic_capacity import measure_recovery_curve
        return measure_recovery_curve(int(dim or self.dim), method, ratios=ratios, n_atoms=n_atoms,
                                      seeds=seeds, codebook=codebook)

    def gap_gate_null(self, library, goal_sig, threshold=0.85, max_length=4, steps=200, n_null=64,
                      seed=0, alpha=0.05):
        """NULL-REFERENCE the capability-synthesis coherence gate (holographic_voidsynth): is `threshold` a
        meaningful bar ON YOUR LIBRARY? synthesize_for_goal accepts a chain when coherence clears a bare
        0.85 -- and that constant encodes an assumption about how coherent a RANDOM goal can get, which is
        a property of the library, not of the algorithm. Near-orthogonal atoms and near-duplicates give
        random goals very different ceilings; one constant cannot be right for both.
        Scores the real goal, re-runs the IDENTICAL synthesis on n_null random unit goals (which have no
        chain behind them by construction -- the procedure match), and reports where the real score sits.
        MEASURED on random unit libraries, 12 seeds: real goals 1.000 +/- 0.000, random goals 0.136-0.237
        depending on dim and library size, so 0.85 separates comfortably HERE -- which is exactly the number
        the bare constant hides and that a caller must confirm on their own data.
        Returns p / collapsed / null_mean / observed plus separation and threshold_is_meaningful."""
        from holographic.misc.holographic_voidsynth import gap_gate_null
        return gap_gate_null(library, goal_sig, threshold=threshold, max_length=max_length, steps=steps,
                             n_null=n_null, seed=seed, alpha=alpha)

    def declare(self, request, args=None, max_rung=5, z_min=0.8, seed=0, dry_run=False,
                null_check=False, n_null=64, cache=None):
        """DECLARED BODY, FILLED BY AN ESCALATING LADDER (holographic_declare, DECLARE-1). Describe what you
        want in plain English; the engine walks rungs cheapest-and-most-provable FIRST and stops at the
        first that clears its own gate:
            0  route_or_abstain -> invoke        INHERITS  a shipped faculty answered
            1  Planner.plan typed chain          INHERITS  a typed chain composed and ran
            2  synthesize_procedure -> run       EXACT     EXECUTION-VERIFIED, it proves its answer
            3  fill_capability_gap -> chain      TOL       cleared a coherence gate
        Returns a Resolution carrying {rung, mechanism, exactness, reversibility, confidence, why} plus a
        DESCENT LOG saying why every rung above the answering one declined -- that log IS the explanation.
        Two axes on purpose: exactness answers 'can I reproduce it', reversibility 'can I recover what went
        in', and cleanup is EXACT and LOSSY at once, so one field could not carry both.
        REFUSAL IS A RESULT: an unresolvable request returns ok=False with the full descent rather than a
        confident guess. That is the whole point -- a fluent filler always returns something.
        max_rung DEFAULTS TO 5 so 'stay deterministic' is a hard guarantee: rungs 6-7 are the model and are
        opt-in per call. Every gate is NaN-guarded, because argmax_tiebreak([0.1, nan, 0.9]) returns 1 --
        the NaN's index, not the maximum -- and json parses bare NaN, so one can arrive over /invoke and
        WIN a gate. The descent log is stored BESIDE the result, never bundled into it: this engine's own
        measurement says nesting stays cheap only while each level is uncluttered.
        PASS cache={} (any dict-like) TO MEMOISE BY CONTENT. MEASURED: a warm walk costs ~45.6 ms because
        find_scored runs over ~2,374 capabilities every call, so a repeat served from cache is ~2 us; a COLD
        walk at a new token count costs ~4.3 s while the router's null is built. Against the machine model's
        break_even_n=1.63 a request repeating even twice pays for it.
        THE HARD RULE IS ENFORCED IN CODE, NOT DOCUMENTED: a resolution whose exactness is NONDETERMINISTIC
        is NEVER stored -- caching a model's answer is a bug, not a slowdown, because it is a fact about one
        moment rather than a value you can key on. Rungs 0-3 never produce one, which is exactly why the
        guard is written before rungs 6-7 exist. Args are hashed to a blake2b digest, never held: a LIVE
        OBJECT in cached args once crashed a worker after its job had already succeeded."""
        from holographic.agents_and_reasoning.holographic_declare import Ladder
        return Ladder(self, max_rung=max_rung, z_min=z_min, seed=seed, null_check=null_check,
                      n_null=n_null, cache=cache).resolve(request, args=args, dry_run=dry_run)

    def declare_explain(self, request, args=None, max_rung=5, z_min=0.8, seed=0):
        """DRY RUN of declare(): report which rung WOULD answer and why the ones above would decline,
        without executing anything (holographic_declare). Same descent log, no side effects -- for asking
        'what will this do, and on what evidence' before letting it do it."""
        from holographic.agents_and_reasoning.holographic_declare import Ladder
        return Ladder(self, max_rung=max_rung, z_min=z_min, seed=seed).resolve(
            request, args=args, dry_run=True)

    def declares(self, fn):
        """DECORATOR form of declare(): attach the ladder to a function whose body is `...`, using its
        DOCSTRING as the request and its keyword arguments as the args (holographic_declare).

            @mind.declares
            def smooth_the_surface(mesh, iters: int = 8):
                \"\"\"Remove bumps from a 3D model without it shrinking.\"\"\"
                ...

        Calling the wrapped function returns a Resolution, not a bare value, so the caller cannot
        accidentally use an answer without its provenance -- which is the failure mode the whole ladder
        exists to prevent. The undecorated function is kept on `.declared` for inspection."""
        import functools
        request = (fn.__doc__ or fn.__name__.replace("_", " ")).strip()

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            return self.declare(request, args=kw)

        wrapper.declared = fn
        wrapper.request = request
        return wrapper

    def decision_flip_rate(self, vectors, queries, bits=8, mode="uniform", half_step=True, seed=0):
        """DECISION-SAFE rate-distortion (holographic_ratedistortion, B5b): what fraction of queries change
        their TOP-1 ANSWER when the index is quantized? Every other distortion measure here asks how close
        the reconstructed vector is; this asks whether the DECISION survives -- and a flipped argmax is a
        different answer, not a slightly worse one. Returns flips / n / flip_rate plus the margin
        distribution (median, p05, min) at full precision, because a rate without margins says what happened
        and not why.
        WHY IT EXISTS: no published paper measures top-1 flip rate under quantization at 500-2,000 items --
        the retrieval literature reports recall@k and nDCG@10 at 100K-1M scale, where a few flips vanish
        into a rank metric. MEASURED here on the shipped 509x128 index: exact-row and even +60%-noise
        queries flip 0.00% (margins ~0.53-0.58), while queries sitting MIDWAY BETWEEN TWO DOCUMENTS collapse
        to margin ~0.058 and flip ~1%. By bit width on those ambiguous queries: 8b 1.3%, 5b 5.7%, 3b 22.3%,
        2b 37.3% -- while normal queries stay 0.00% at EVERY width down to 2 bits.
        THE FINDING: FLIP RATE IS GOVERNED BY MARGIN, NOT BY CORPUS SIZE OR BIT WIDTH. So decision-safety is
        not answerable from N and bits; it needs the margin distribution of the queries you actually serve.
        KEPT NEGATIVE: a uint8-vs-float8 verdict CANNOT be taken on the shipped index, which is already
        uint8 -- uniform re-quantization there is a no-op (max|err| 0.000000) while float8 genuinely
        re-quantizes, so the comparison would measure the source grid, not the quantizers."""
        from holographic.misc.holographic_ratedistortion import decision_flip_rate
        return decision_flip_rate(vectors, queries, bits=bits, mode=mode, half_step=half_step, seed=seed)

    def crowded_subset(self, vectors, size, seed=0):
        """The `size` mutually MOST SIMILAR rows of `vectors` (holographic_ratedistortion) -- a synthetic
        stand-in for a catalog whose entries are variations on a theme, grown greedily from the closest pair
        because crowding is a cluster, not a random sample. Pair with decision_flip_rate to test whether a
        decision-safety proof taken on a well-separated corpus still holds on a crowded one; it is exactly
        the transfer that must NOT be assumed."""
        from holographic.misc.holographic_ratedistortion import crowded_subset
        return crowded_subset(vectors, size, seed=seed)

    def qfhrr_quantize(self, v, levels=16):
        """QUANTIZED INTEGER-PHASE FHRR (holographic_qfhrr, qFHRR-1) -- turn a complex FHRR phasor vector
        into integer phase indices in {0..levels-1}. At 16 levels that is 4 bits per dimension against
        complex128's 128, a 96.9% storage cut, and everything downstream (bind/unbind/similarity) becomes
        pure integer arithmetic. This is the ONLY lossy step; use qfhrr_dequantize to go back. Opt-in tier
        over the existing complex FHRR -- nothing in holographic_fhrr or the real-valued defaults changes."""
        from holographic.sampling_and_signal.holographic_qfhrr import quantize_phases
        return quantize_phases(v, levels)

    def qfhrr_dequantize(self, q, levels=16):
        """Integer phase indices back to complex unit phasors (holographic_qfhrr) -- the inverse of
        qfhrr_quantize up to the bin width already discarded. Use it to hand a quantized vector back to the
        complex FHRR path."""
        from holographic.sampling_and_signal.holographic_qfhrr import dequantize_phases
        return dequantize_phases(q, levels)

    def qfhrr_bind(self, qa, qb, levels=16):
        """Bind quantized phase vectors EXACTLY (holographic_qfhrr): phases add, so indices add mod levels.
        Pure integer, no floating point, identical on every machine."""
        from holographic.sampling_and_signal.holographic_qfhrr import qfhrr_bind
        return qfhrr_bind(qa, qb, levels)

    def qfhrr_unbind(self, qc, qa, levels=16):
        """Unbind quantized phase vectors EXACTLY (holographic_qfhrr) -- and this is a TRUE inverse, not the
        quasi-inverse the real-valued path gives. Phases subtract, so unbind(bind(a,b),a) returns b's
        indices BIT FOR BIT at every K, needing no cleanup at all; real HRR recovers b at cosine ~0.70 and
        does need it. The strongest exactness guarantee in the engine -- bought by giving up closed
        bundling (see qfhrr_bundle)."""
        from holographic.sampling_and_signal.holographic_qfhrr import qfhrr_unbind
        return qfhrr_unbind(qc, qa, levels)

    def qfhrr_bundle(self, qs, levels=16):
        """Superpose quantized phase vectors -- BY LEAVING THE REPRESENTATION, because no closed integer
        operation does this (holographic_qfhrr). Dequantize to Cartesian, sum, re-quantize via atan2 +
        round. KEPT NEGATIVE, AND IT IS THE ONE THIS RESEARCH LINE KEEPS ARRIVING AT: the round() at a bin
        boundary IS a tie-arbitration point and the atan2 is floating point, so this single call is where
        the tier's exactness and machine-independence stop. QUANTIZED VSA DOES NOT LET THE ENGINE DELETE ITS
        TIE-ARBITRATION MACHINERY -- exactness holds for bind/unbind only."""
        from holographic.sampling_and_signal.holographic_qfhrr import qfhrr_bundle
        return qfhrr_bundle(qs, levels)

    def qfhrr_measure_fidelity(self, dim=1024, levels_list=(4, 8, 16, 32, 64, 256), bundle_n=16, seeds=8):
        """RE-MEASURE the quantized-vs-complex fidelity table on this substrate (holographic_qfhrr).
        Wired because RESEARCH_CONSOLIDATED.md quotes the qFHRR preprint's numbers and explicitly flags them
        as UNVERIFIABLE from the abstract. Measured here: BIND fidelity matches the paper almost exactly
        (0.9498 / 0.9875 / 0.9999 at K=8/16/256 vs the paper's 0.9497 / 0.9872 / 0.9999). BUNDLE fidelity
        matches ONLY against a PHASE-ONLY reference (0.9156 / 0.9738 / 0.9998 vs 0.9147 / 0.9731 / 0.9997);
        against the actual complex bundle it SATURATES AT ~0.892 and more phase levels do not help, because
        the ceiling is discarding the MAGNITUDE. Both are returned -- bundle_fid and bundle_fid_phase -- so
        the optimistic figure cannot silently stand in for the one an engineer gets."""
        from holographic.sampling_and_signal.holographic_qfhrr import measure_fidelity
        return measure_fidelity(dim=dim, levels_list=levels_list, bundle_n=bundle_n, seeds=seeds)

    def amp_recall(self, cue, codebook, K=None, iters=30, alpha=None, tol=1e-12, stats=None):
        """AMP RECALL (holographic_amp, AMP-1) -- the FIFTH member of the bundle-recovery family. IHT with
        one extra term: the ONSAGER CORRECTION, a memory term proportional to the fraction of active
        coefficients, which cancels the estimate/design correlations that make a plain gradient residual
        non-Gaussian. With it the residual behaves like AWGN, so the threshold is read from the residual norm
        (STATE EVOLUTION) instead of tuned -- which is why K IS OPTIONAL here and required by every other
        member. Returns (index, weight) descending by |weight|, the same shape as cosamp_recall.
        MEASURED vs CoSaMP (D=512, N=2048, 8 seeds): TIE at 1.000 through M/D=0.17, then AMP WINS -- 0.896
        vs 0.709 at M/D=0.25, 0.558 vs 0.167 at M/D=0.33 -- at a flat ~21 ms while CoSaMP's per-round
        least-squares climbs to ~1 s (48x faster at M=200). The baseline is not a strawman: CoSaMP at 120
        iterations (6.5 s) still scores 0.154 there.
        KEPT NEGATIVES, and neither method dominates: (1) A COHERENT DICTIONARY DESTROYS AMP -- at coherence
        0.5 it scores 0.052 where CoSaMP holds 1.000, because state evolution assumes a roughly i.i.d.
        design. CoSaMP stays the method there. (2) AMP is SLOWER at light load (21 ms vs 3 ms at M=16), its
        cost being flat in M. (3) K-free operation only works at LIGHT load; at heavy load supply K."""
        from holographic.sampling_and_signal.holographic_amp import amp_recall
        return amp_recall(cue, codebook, K=K, iters=iters, alpha=alpha, tol=tol, stats=stats)

    def amp_measure_vs_cosamp(self, dim=512, n_atoms=2048, loads=(16, 32, 64, 86, 128, 171), seeds=6, seed0=0):
        """MEASURE AMP against the HONEST baseline -- CoSaMP, already shipped -- at matched load with variance
        (holographic_amp.measure_vs_cosamp). Wired because the research consolidation ranks AMP #1 by scoring
        it against Bottleneck 2's '20-32 instruction' ceiling, which was measured with the LINEAR readout and
        never existed. Benchmarking AMP against that figure manufactures a 5x win; against CoSaMP the honest
        result is a CROSSOVER -- a win in the M/D 0.25-0.39 band, a catastrophic loss on coherent
        dictionaries. Returns dicts with M, M_over_D, amp_f1/sd, cosamp_f1/sd, amp_ms, cosamp_ms."""
        from holographic.sampling_and_signal.holographic_amp import measure_vs_cosamp
        return measure_vs_cosamp(dim=dim, n_atoms=n_atoms, loads=loads, seeds=seeds, seed0=seed0)

    def ntt_bind(self, a, b):
        """EXACT INTEGER BINDING via the Number-Theoretic Transform (holographic_ntt, NTT-1). The same algebra
        as bind() -- circular convolution -- but computed in modular integer arithmetic over Z_q, so nothing
        rounds and nothing depends on floating-point summation order. WHY: numpy.fft (pocketfft) is not
        guaranteed bit-identical across CPUs (SIMD width changes the summation order; NumPy #11926 reports up
        to 0.1% divergence between two Xeons), and here a ULP flip is an argmax flip. Integer input only; the
        modulus bound 2*n*max|a|*max|b| < q is CHECKED and raises rather than wrapping silently.
        KEPT NEGATIVE, measured: 19-50x SLOWER than the float FFT bind across D=256..4096 (a Python loop over
        log2(n) modular stages against C-level pocketfft). This buys EXACTNESS AND REPRODUCIBILITY, not speed
        -- call ntt_measure_vs_fft() rather than trusting this sentence."""
        from holographic.sampling_and_signal.holographic_ntt import ntt_bind
        return ntt_bind(a, b)

    def ntt_unbind(self, c, a):
        """Unbind exactly-computed integer bindings by correlation with the involution (holographic_ntt).
        KEPT NEGATIVE, AND IT IS THE IMPORTANT ONE: this does NOT make unbinding exact. The involution is
        HRR's QUASI-inverse, so unbind(bind(a,b), a) recovers b in DIRECTION only and cleanup is still
        required -- exactly as with the float path. What the NTT removes is floating-point nondeterminism
        from the OPERATION, not the algebraic approximation inside HRR. True exact deconvolution would need
        a's spectrum to be invertible mod q, which no arbitrary atom guarantees. The standing claim that
        integer VSA lets the engine delete its tie-arbitration machinery remains REFUTED."""
        from holographic.sampling_and_signal.holographic_ntt import ntt_unbind
        return ntt_unbind(c, a)

    def ntt_convolve(self, a, b):
        """EXACT cyclic convolution of two integer vectors (holographic_ntt) -- the exact-arithmetic
        replacement for irfft(rfft(a)*rfft(b)), bit-identical on every machine. ntt_bind is this under its
        VSA name; use this spelling for signal work. Verified against a naive O(n^2) integer convolution with
        array_equal, never a tolerance. Raises if the modulus cannot hold the signed result."""
        from holographic.sampling_and_signal.holographic_ntt import ntt_convolve
        return ntt_convolve(a, b)

    def ntt_measure_vs_fft(self, sizes=(256, 512, 1024, 2048, 4096), repeats=40, seed=0):
        """MEASURE exact NTT convolution against the float FFT convolution bind() actually uses
        (holographic_ntt.measure_vs_fft), warmed, medians with spread. Wired because the research
        consolidation flagged that NO NumPy NTT-vs-FFT benchmark exists and the NTT might well lose: it does,
        by 19-50x. A cost that is not runnable becomes folklore. ratio > 1 means the NTT is slower."""
        from holographic.sampling_and_signal.holographic_ntt import measure_ntt_vs_fft
        return measure_ntt_vs_fft(sizes=sizes, repeats=repeats, seed=seed)

    def hadamard_codebook(self, dim, seed=0, signed=True):
        """STRUCTURED CODEBOOK whose cleanup is ONE TRANSFORM, not a K-scan (holographic_htcodebook, HT-1).
        Atoms are the sign-permuted rows of a Hadamard matrix, so correlating a cue against ALL of them is a
        single Walsh-Hadamard transform: cleanup costs O(D log D) instead of O(K*D), the atoms are generated
        rather than stored (a sign vector, not a K x D matrix), and because the rows are mutually ORTHOGONAL
        the crosstalk between distinct atoms is exactly zero. argmax of the correlations is then the exact
        maximum-likelihood nearest-codeword decode -- the classic Reed-Muller 'Green machine', reached from the
        VSA side. Returns a HadamardCodebook with .cleanup(cue) -> (index, score), .atom(i), .K.
        MEASURED at equal K and D, transform vs matmul scan: 2.3x at D=512, 6.9x at D=1024 (628 us -> 91 us),
        39x at D=2048, 219x at D=8192 (77 ms -> 0.35 ms) -- the win GROWS with D.
        KEPT NEGATIVES, both load-bearing: (1) it LOSES at D=256 (0.49x) -- the Python-level transform loop
        costs more than a small matmul, so the crossover is around D=512, below which use the scan. (2) K IS
        CAPPED AT 2*D by the construction and cannot be raised; the backlog item that asked for this proposed
        replacing a K=16384 scan at D=1024, which no Hadamard codebook can hold (that needs D=8192). (3) the
        SIGNED codebook is a DEGENERATE RECOVERY DICTIONARY -- signed=True adds each atom's exact negation, so
        coherence is exactly 1.000 and cosamp_recall/iht_recall return a WRONG support (measured 0/3 signed vs
        3/3 unsigned at D=256); pass signed=False for anything that unbundles. The atom set is fixed, so this
        is an opt-in codebook TYPE -- Vocabulary and every existing cleanup path are untouched."""
        from holographic.caching_and_storage.holographic_htcodebook import HadamardCodebook
        return HadamardCodebook(dim, seed=seed, signed=signed)

    def hadamard_codebook_measure(self, dims=(256, 512, 1024, 2048, 4096), repeats=60, seed=0):
        """MEASURE transform cleanup against a matmul scan at EQUAL K and EQUAL D (holographic_htcodebook).
        Wired so the crossover and the cap stay runnable rather than quotable: equal-K is the only honest
        comparison, because the structured codebook cannot hold more than 2*D atoms. Returns dicts with dim, K,
        wht_us, scan_us, speedup (>1 = the transform wins; it is BELOW 1 at D=256)."""
        from holographic.caching_and_storage.holographic_htcodebook import measure_vs_scan
        return measure_vs_scan(dims=dims, repeats=repeats, seed=seed)

    def wht(self, a):
        """FAST WALSH-HADAMARD TRANSFORM (holographic_wht, WHT-1) -- the O(D log D) matrix-free transform, D a
        power of two, unnormalised so wht(wht(x)) == D*x. Every butterfly is one add and one subtract: no twiddle
        factors, no stored matrix, nothing to round. Integer dtypes are PRESERVED (see wht_exact); float input is
        promoted to float64. Use it for structured matrix-free operators (this is what HolographicArchive's key
        operator has always run on) and wherever the transform must be reorder-independent. Kept negative, measured:
        it is NOT an FFT speedup -- 4-9x SLOWER than numpy.rfft across D=256..16384 on this codebase, because it is a
        Python loop over log2(D) vectorised passes against C-level pocketfft. Call wht_measure_vs_fft() to re-run
        that comparison rather than trusting this sentence."""
        from holographic.sampling_and_signal.holographic_wht import fwht
        return fwht(a)

    def wht_exact(self, a):
        """EXACT INTEGER Walsh-Hadamard transform (holographic_wht) -- identical to wht() but REFUSES float input,
        so the bit-exactness guarantee is enforced by a TypeError instead of trusted from a docstring. On integer
        input every butterfly is integer add/subtract, so the result is bit-exact and IDENTICAL ON EVERY MACHINE.
        That is the point: numpy.fft (pocketfft) can return bitwise-different results across CPUs because SIMD
        width changes the summation order (NumPy issue #11926 reports up to 0.1% divergence between two Xeons),
        and in this engine a ULP difference flips a cleanup argmax, which is a different creature. Input is widened
        to int64 first -- int8/int16 would silently wrap, since a bipolar transform reaches |tap| <= D."""
        from holographic.sampling_and_signal.holographic_wht import fwht_exact
        return fwht_exact(a)

    def wht_inverse(self, a):
        """INVERSE fast Walsh-Hadamard transform (holographic_wht): the WHT is its own inverse up to a 1/D scale,
        so this is wht(a)/D. Returns float -- the division is what breaks integrality. For an exact integer round
        trip call wht_exact twice and divide by D yourself, which stays exact whenever D divides every entry."""
        from holographic.sampling_and_signal.holographic_wht import ifwht
        return ifwht(a)

    def wht_measure_vs_fft(self, sizes=(256, 512, 1024, 2048, 4096), repeats=200, seed=0):
        """MEASURE the Walsh-Hadamard transform against numpy.rfft, per size, warmed, median plus spread
        (holographic_wht.measure_vs_fft). This is wired as a faculty ON PURPOSE: the research shortlist floats the
        WHT as a possible speed win over the FFT, this codebase measures it as a 4-9x LOSS, and a negative that is
        not runnable decays into folklore. Returns dicts with dim / wht_us / fft_us / ratio (>1 means SLOWER) plus
        the means, so the skew stays visible. The harness warms up and takes medians because the first version of
        it did neither and duly asserted the wrong direction off first-call noise."""
        from holographic.sampling_and_signal.holographic_wht import measure_wht_vs_fft
        return measure_wht_vs_fft(sizes=sizes, repeats=repeats, seed=seed)

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
            from holographic.misc.holographic_sbc import decompose_structure
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
        from holographic.misc.holographic_resonator import ResonatorNetwork
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
        from holographic.misc.holographic_sbc import decompose_structure as _decompose
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
        from holographic.misc.holographic_verify import CompositionTree
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
        from holographic.misc.holographic_tree import StructuredIndex
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
        from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
        return VectorFunctionEncoder(n_dims, dim=min(self.dim, 1024), bounds=bounds,
                                     kernel=kernel, bandwidth=bandwidth, seed=self.seed)

    def tile_field(self, enc, function, period, counts):
        """Tile an FPE field hypervector over an n-D lattice -- domain repetition as bind+bundle, so the result
        is itself a composable hypervector (works in 2-D and 3-D). See holographic_tiling.tile."""
        from holographic.mesh_and_geometry.holographic_tiling import tile
        return tile(enc, function, period, counts)

    def tile_field_recursive(self, enc, function, period, counts, levels):
        """Recursive tiling (inception): tile the tiling `levels` deep -- count^levels copies per axis from
        linear binds, in one fixed-size vector. See holographic_tiling.tile_recursive."""
        from holographic.mesh_and_geometry.holographic_tiling import tile_recursive
        return tile_recursive(enc, function, period, counts, levels)

    def fractal_field(self, enc, function, base_period, levels, count=3, decay=1.0):
        """A multi-scale (fBm-like) tiling: the motif tiled at base_period, /2, /4, ... summed -- richness at
        many scales from one motif and a few binds. See holographic_tiling.fractal_bands."""
        from holographic.mesh_and_geometry.holographic_tiling import fractal_bands
        return fractal_bands(enc, function, base_period, levels, count=count, decay=decay)

    def fractal_volume(self, enc, period, counts, levels, motif=None, beta=2.0, seed=0, motif_size=5,
                       motif_grid=None, motif_coords=None):
        """ONE call: inception over ANY VSA object -> one hypervector. The seed can be a precomputed motif
        hypervector (a smoke puff, an SDF surface, an archive image, or ANOTHER fractal_volume's output --
        inception over the engine itself), a NumPy grid (motif_grid, crossed into VSA once), or, by default, a
        synthesised localized fractal grain. tile_recursive replicates it count^levels deep in one fixed-size
        vector, composable as any VSA object. See holographic_tiling.fractal_volume."""
        from holographic.mesh_and_geometry.holographic_tiling import fractal_volume
        return fractal_volume(enc, period, counts, levels, motif=motif, beta=beta, seed=seed,
                              motif_size=motif_size, motif_grid=motif_grid, motif_coords=motif_coords)

    def inception(self, enc, period, counts, depth, motif=None, beta=2.0, seed=0, motif_size=5):
        """One-parameter recursion DEPTH over fractal_volume + an honest capacity-ceiling measurement. Returns
        (volume, profile): the volume is the recursive tiling carried `depth` levels (composable as any VSA
        object); the profile reports, at each depth, copies-per-axis, mean per-copy read, and round-trip
        recovery -- so the SNR fall as counts**depth instances share one fixed dim is a measured table. See
        holographic_tiling.inception."""
        from holographic.mesh_and_geometry.holographic_tiling import inception
        return inception(enc, period, counts, depth, motif=motif, beta=beta, seed=seed, motif_size=motif_size)

    def grid_to_hypervector(self, enc, grid, coords, threshold=1e-3):
        """Encode a NumPy field (a fluid density, an SDF slice) as an FPE hypervector so it can be tiled /
        bound / bundled / stored like any VSA object -- the one crossing from grid into VSA. See
        holographic_tiling.grid_to_function."""
        from holographic.mesh_and_geometry.holographic_tiling import grid_to_function
        return grid_to_function(enc, grid, coords, threshold=threshold)

    def hypervector_to_grid(self, enc, function, coords):
        """Read an FPE field hypervector back onto a grid (the inverse of grid_to_hypervector). See
        holographic_tiling.function_to_grid."""
        from holographic.mesh_and_geometry.holographic_tiling import function_to_grid
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
        from holographic.sampling_and_signal.holographic_harmonic import harmonic_atom
        return harmonic_atom(thetas, meanings, n_harmonics)

    def harmonic_decode(self, atom, theta):
        """Read a context-conditioned atom (holographic_harmonic, RT-VI) at context angle `theta` -- the harmonic
        sum giving the meaning in that context. The analog of unbinding with an FPE-encoded role(theta)."""
        from holographic.sampling_and_signal.holographic_harmonic import harmonic_decode
        return harmonic_decode(atom, theta)

    def harmonic_dc(self, atom):
        """The DC (degree-0), context-FREE meaning of a context-conditioned atom (holographic_harmonic, RT-VI) --
        exactly the plain fixed atom, the backward-compatible fallback."""
        from holographic.sampling_and_signal.holographic_harmonic import harmonic_dc
        return harmonic_dc(atom)

    def spectral_basis(self, points, k=10, n_basis=12):
        """The data-driven decomposition basis for a signal on a manifold (holographic_spectral, EXP-6): the
        lowest eigenvectors of the kNN-graph Laplacian of the sample points -- the smoothest functions the
        manifold admits. On a line this is the DCT / elementary basis and on a ring the harmonic basis (so it
        matches decompose_signal's hand-picked choice), and on a manifold the topology detector cannot name
        (a sphere, a torus, a curved surface) it is the right basis where the line/elementary fallback is not.
        Returns a SpectralBasis with decompose / reconstruct / denoise. Dense eigh -> moderate N (C1)."""
        from holographic.sampling_and_signal.holographic_spectral import SpectralBasis
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
        from holographic.sampling_and_signal.holographic_nystrom import nystrom_embedding
        return nystrom_embedding(points, n_basis=n_basis, m=m, sigma=sigma, landmarks=landmarks, seed=seed)

    def spectral_landmarks(self, points, m, seed=0):
        """Farthest-point-sampled landmark indices: a coverage set (every local manifold gets an anchor) for the
        Nystrom embedding -- the discrete cousin of the engine's blue-noise sampling. See holographic_nystrom."""
        from holographic.sampling_and_signal.holographic_nystrom import farthest_point_landmarks
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
        from holographic.mesh_and_geometry.holographic_octree import HoloOctree
        tree = HoloOctree(bounds, capacity=capacity, dim=dim, bandwidth=bandwidth, max_depth=max_depth, seed=seed)
        if points is not None:
            tree.insert(points)
        return tree

    def similarity_graph(self, vectors, k=6, weighted=True):
        """A geometry-weighted kNN graph over hypervectors (holographic_simgraph, ARCH-3): the cotangent-Laplacian
        idea turned inward. weighted=True makes each edge carry the COSINE SIMILARITY (the geometry of the vector
        space) -- the analogue of cotangent weights; weighted=False is the engine's existing BINARY kNN graph.
        Returns a symmetric adjacency matrix."""
        from holographic.misc.holographic_simgraph import similarity_adjacency
        return similarity_adjacency(vectors, k, weighted=weighted)

    def graph_spectral_embedding(self, vectors, k=6, dims=2, weighted=True):
        """Laplacian eigenmaps of a hypervector set (holographic_simgraph, ARCH-3): the `dims` lowest non-trivial
        eigenvectors of the (similarity-weighted) graph Laplacian -- the data-driven coordinates the manifold's
        points live on. Recovers a ring as a circle, a curve as a line. Returns (N, dims). Kept negative: under
        UNIFORM high-D sampling the weighted and binary graphs essentially tie (concentration of measure, unlike a
        mesh's sharp cotangent gap); weighting helps most under IRREGULAR sampling."""
        from holographic.misc.holographic_simgraph import spectral_embedding
        return spectral_embedding(vectors, k=k, dims=dims, weighted=weighted)

    def graph_ring_order(self, vectors, k=6, weighted=True):
        """For hypervectors sampled on a 1-D ring, the recovered cyclic coordinate (holographic_simgraph, ARCH-3):
        atan2 of the first two non-trivial Laplacian eigenvectors -- a ring's eigenmap is a circle, so this recovers
        the points' order around the ring from the high-D vectors alone. Returns (N,) angles."""
        from holographic.misc.holographic_simgraph import ring_order
        return ring_order(vectors, k=k, weighted=weighted)

    def subdivide_sequence(self, points, levels=1, closed=False):
        """Subdivide a SEQUENCE of hypervectors into a smooth limit curve (holographic_subdivcurve, ARCH-5):
        Chaikin corner-cutting, the 1-D inward mirror of FWD-8's mesh subdivision. Each level doubles the point
        count (refine) and low-pass smooths (corner-cutting). Reproduces a straight line of vectors exactly (affine,
        like FWD-8's 'flat stays flat'), converges to a limit curve, and shrinks a zig-zag's roughness. Returns the
        refined (M, dim) sequence. Kept negative: Chaikin is APPROXIMATING -- the original control points are cut,
        not interpolated (the same approximating nature as Loop; an interpolating 4-point scheme is deferred)."""
        from holographic.mesh_and_geometry.holographic_subdivcurve import subdivide_sequence
        return subdivide_sequence(points, levels=levels, closed=closed)

    def manifold_topology(self, points, lo=1.3, hi=2.6, steps=7, max_points=250):
        """Name a point cloud's topology by persistent homology (holographic_topology, EXP-7): the Betti
        signature (components, loops, voids) that persists across a scale band. The principled generalisation
        of detect_topology -- it reproduces "line" (1,0,0) and "ring" (1,1,0) on the cases that detector knows,
        and extends to ones it cannot name structurally: a torus (1,2,1) and a sphere (1,0,1). Returns
        (name, (B0,B1,B2), scale-histogram). It reads a WELL-SAMPLED manifold's topology -- finicky on uneven
        or noisy clouds and blind to non-topological geometry (pair it with spectral_basis for the geometry),
        and the cloud is subsampled to max_points for tractability (dense reduction -> moderate N, C1)."""
        from holographic.mesh_and_geometry.holographic_topology import persistent_topology
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
        from holographic.sampling_and_signal.holographic_spectral import hodge_decomposition as _hd
        return _hd(n_verts, edges, flow, triangles)

    def denoise_flow(self, n_verts, edges, flow, triangles=None, keep=("gradient", "harmonic")):
        """Denoise an edge flow by keeping only its structurally-valid Hodge components and dropping the rest
        (holographic_spectral, EXP-8) -- e.g. drop the curl of a transport flow that should not circulate
        around cells, removing the share of isotropic noise that lands there. Beats naive edge-smoothing on a
        flow. Returns the reconstruction from the kept parts."""
        from holographic.sampling_and_signal.holographic_spectral import denoise_flow as _df
        return _df(n_verts, edges, flow, triangles, keep=keep)

    # -- one scene faculty, on the same substrate ---------------------------
    def scene(self):
        """The mind's own scene coder (compose/decompose visual attribute scenes), built
        lazily on this mind's dim and seed so it shares the substrate rather than being a
        separate engine. All scene methods below go through it."""
        if self._scene is None:
            from holographic.scene_and_pipeline.holographic_scene import SceneCoder
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
            from holographic.agents_and_reasoning.holographic_ai import Vocabulary
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
        from holographic.sampling_and_signal.holographic_fhrr import PhasorMemory, PhasorVocabulary
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
        from holographic.simulation_and_physics.holographic_phasemorph import phase_morph
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
        from holographic.agents_and_reasoning.holographic_ai import bind
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
        from holographic.agents_and_reasoning.holographic_ai import unbind
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
        from holographic.misc.holographic_recipe import StructureRecipe
        return StructureRecipe(self.dim, self.seed)

    def realize(self, recipe):
        """Replay a StructureRecipe to its output vector(s) -- the single realize path for any structure."""
        outs = recipe.outputs()
        return outs[0] if len(outs) == 1 else outs

    def validate_recipe(self, recipe):
        """Check a StructureRecipe is WELL-FORMED (holographic_recipeops, ARCH-1) -- the recipe's is_manifold():
        every op references only EARLIER existing results (a DAG, no forward/dangling/out-of-range refs), raw
        indices and repeat templates in range, every DECLARED OUTPUT points to a produced result, and set-ops
        (bundle/superpose) have a non-empty member list. Returns (ok, problems). Catches recipes that pass the DAG
        check but crash at build (a dangling output -> IndexError; an empty bundle -> a degenerate zero vector).
        Pairs with the recipe EDIT operators below -- the recipe equivalent of the mesh Euler operators."""
        from holographic.misc.holographic_recipeops import validate
        return validate(recipe)

    def recipe_commute_bind(self, recipe, handle):
        """Edit a recipe (holographic_recipeops, ARCH-1): swap the two arguments of the bind at `handle`,
        bind(a,b)->bind(b,a). Because bind is circular convolution (commutative), the realized vector is unchanged
        (FFT precision). ITS OWN INVERSE -- the recipe analogue of mesh flip_edge. Returns a new recipe."""
        from holographic.misc.holographic_recipeops import commute_bind
        return commute_bind(recipe, handle)

    def recipe_reorder_members(self, recipe, handle, perm):
        """Edit a recipe (holographic_recipeops, ARCH-1): permute the members of the bundle/superpose at `handle`
        by `perm`. Because bundle/superpose are sums (commutative), the realized vector is unchanged; invertible by
        the inverse permutation. The parameterised cousin of recipe_commute_bind. Returns a new recipe."""
        from holographic.misc.holographic_recipeops import reorder_members
        return reorder_members(recipe, handle, perm)

    def recipe_substitute_atom(self, recipe, handle, new_name):
        """Edit a recipe (holographic_recipeops, ARCH-1): rename the atom leaf at `handle` -- the recipe analogue
        of moving a vertex (keeps the STRUCTURE valid while changing the realized vector predictably; reversed
        exactly by substituting the original name back). Returns a new recipe."""
        from holographic.misc.holographic_recipeops import substitute_atom
        return substitute_atom(recipe, handle, new_name)

    def template_names(self):
        """The names of the available parameterized recipe templates (ISA-6, the macro layer)."""
        from holographic.simulation_and_physics.holographic_template import STARTER_LIBRARY
        return sorted(STARTER_LIBRARY)

    def instantiate_template(self, name, **args):
        """Instantiate a named parameterized template (ISA-6) at this mind's dim/seed, filling its HOLES with
        `args` (a string is a named atom; an array is a literal vector). Returns the built structure vector --
        different arguments give distinct, BIT-EXACT structures, and the templates are hygienic (internal atoms
        are namespaced under a reserved prefix and cannot collide with the caller's atoms)."""
        from holographic.simulation_and_physics.holographic_template import STARTER_LIBRARY
        if name not in STARTER_LIBRARY:
            raise ValueError(f"unknown template {name!r}; available: {sorted(STARTER_LIBRARY)}")
        return STARTER_LIBRARY[name].build_vector(self.dim, self.seed, **args)

    def compile_structure(self, spec):
        """Compile a structure-description spec (ISA-7) to a StructureRecipe at this mind's dim/seed. The spec is
        S-expression text or a parsed AST -- a declarative surface (atoms, bind/bundle/permute, ISA-6 templates)
        that lowers to the recipe IR. Scoped to structure description, not a general language."""
        from holographic.misc.holographic_lang import compile_spec
        return compile_spec(spec, self.dim, self.seed)

    def realize_structure(self, spec):
        """Compile a structure-description spec (ISA-7) and materialize its vector -- bit-exact for a given
        spec/dim/seed. The high-level surface for the same structures realize/compose build by hand."""
        from holographic.misc.holographic_lang import realize_spec
        return realize_spec(spec, self.dim, self.seed)

    def reversibility_audit(self):
        """Classify each base instruction as reversible (bind/unbind/permute/involution) or information-
        destroying (bundle/superpose/cleanup) -- the ISA-8 reversibility model. cleanup is error correction;
        capacity is the coherence budget. (Framing; the practical payoff is run_with_auto_cleanup.)"""
        from holographic.misc.holographic_reversible import reversibility_audit
        return reversibility_audit()

    def run_with_auto_cleanup(self, initial, steps, codebook, floor=0.9, schedule="adaptive", k=3):
        """Run a vector 'program' (a list of vector->vector steps) under an error-correction policy (ISA-8),
        inserting a cleanup before the crosstalk cliff. 'adaptive' cleans only when the nearest-atom health drops
        below `floor`; it holds fidelity at far fewer cleanups than a fixed cadence under variable damage.
        Returns (final_vector, n_cleanups)."""
        from holographic.misc.holographic_reversible import auto_cleanup_run
        return auto_cleanup_run(initial, steps, codebook, floor=floor, schedule=schedule, k=k)

    def steering_regress(self, X, y, X_query, bounds, base=2.0, dim=None):
        """Anisotropic (steering) kernel regression (RT-IV1): fit a PER-AXIS bandwidth to the data's directional
        structure (a sharp axis gets a large bandwidth, a flat axis a small one), build an anisotropic FPE
        encoder, and predict X_query by FPE-kernel-weighted averaging. Returns (predictions, bandwidths). Beats
        the isotropic RBF on DENSE directional data (an edge/ridge); on sparse or isotropic data the advantage is
        marginal and the bandwidth estimate is unreliable -- isotropic is the honest baseline there."""
        from holographic.misc.holographic_steering import steer_bandwidths, kernel_regress
        from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
        bw = steer_bandwidths(X, y, base=base)
        enc = VectorFunctionEncoder(len(bounds), dim=(dim or self.dim), bounds=bounds, bandwidth=bw, seed=self.seed)
        return kernel_regress(enc, X, y, X_query), bw

    def propagator_jump(self, states, state, k):
        """Jump a learned dynamics operator k steps in ONE eval (RT-I1): the closed-form k-step iterate via the
        FREE Fourier spectrum (a bind is diagonal in the Fourier basis), matching the k-bind rollout to FFT
        tolerance. Diagonalise once, evaluate any level -- the same math Stam's subdivision eval uses."""
        from holographic.misc.holographic_iterate import step_k
        U = self.learn_dynamics(states).U
        return step_k(state, U, k)

    def propagator_spectrum(self, states):
        """Read a learned dynamics operator's convergence off its FREE FFT spectrum (RT-I1) WITHOUT running:
        the regime (contractive -> decays; marginal -> persists; divergent -> blows up), the spectral_gap
        (small -> slow/near-degenerate stall, the linear cousin of a resonator stall), and the dominant frequency.
        The eigendecomposition of the bind operator is just its rfft -- no dense O(n^3) work."""
        from holographic.misc.holographic_iterate import spectral_profile
        U = self.learn_dynamics(states).U
        return spectral_profile(U)

    def tree_structure(self, tree):
        """Encode an expression tree as a typed structure at this mind's dim/seed. A leaf is a str symbol;
        an internal node is (op, *children). The EML-tree's holographic encoding, generalised."""
        from holographic.misc.holographic_typed import tree_to_recipe
        return tree_to_recipe(self.dim, self.seed, tree)

    def nested_scene_structure(self, groups):
        """compose_nested AS a typed structure: the same super-scene vector, now a replayable build-graph
        (group-role atoms + bind + superpose, rng sub-scenes as raw leaves) that can be saved and inspected."""
        from holographic.misc.holographic_typed import nested_scene_to_recipe
        return nested_scene_to_recipe(self, groups)

    def scene_graph(self, transform=None, mesh=None, children=None, name=None):
        """Build a SCENE-GRAPH node (holographic_scenegraph): a 4x4 `transform`, an optional leaf `mesh`, optional
        child nodes -- the geometry capstone that joins the FWD mesh kernel to the ARCH-1 recipe algebra. The node
        can be read two ways (see scene_flatten / scene_to_recipe). Transform builders are on the mind:
        scene_translation / scene_scaling / scene_rotation / scene_compose_transforms. Returns a SceneNode."""
        from holographic.scene_and_pipeline.holographic_scenegraph import SceneNode
        return SceneNode(transform=transform, mesh=mesh, children=children, name=name)

    def scene_flatten(self, node):
        """The GEOMETRY view of a scene graph (holographic_scenegraph): instance every leaf mesh through its
        ACCUMULATED transform (parent transforms composed down the graph) and MERGE into one Mesh. Returns a Mesh.
        Kept negative: this INSTANCES and concatenates -- it does not weld or boolean-merge overlapping geometry
        (that is mesh_csg's job); two touching cubes flatten to two components, not one solid."""
        from holographic.scene_and_pipeline.holographic_scenegraph import flatten_scene
        return flatten_scene(node)

    def scene_to_recipe(self, node, dim=None, seed=None):
        """The STRUCTURE view of a scene graph (holographic_scenegraph): encode the graph as a StructureRecipe --
        transforms BOUND to content, siblings BUNDLED -- realising to one hypervector. A well-formed recipe that the
        ARCH-1 operators (validate_recipe / recipe_reorder_members) apply to. The consistency theorem: swapping
        siblings leaves BOTH the flattened geometry and this vector identical (merge and bundle are commutative) --
        the scene is one object in two costumes, and they agree. Returns a StructureRecipe."""
        from holographic.scene_and_pipeline.holographic_scenegraph import scene_to_recipe
        return scene_to_recipe(node, dim=self.dim if dim is None else dim, seed=self.seed if seed is None else seed)

    def scene_delta(self, base, variant):
        """The component DIFF between two scenes (holographic_scenedelta): {'added', 'removed'} content-hashed
        component ids, so a variant can be TRANSMITTED as its delta (send the base once, then small deltas). A
        one-subtree change is a couple of components; apply_scene_delta rebuilds the variant's component set exactly.
        Kept negative (honest): the component SHARING itself is AUTOMATIC from content-addressed atoms (shared
        subtrees share ids for free) -- this adds the explicit diff/transmission, not a new dedup mechanism."""
        from holographic.scene_and_pipeline.holographic_scenedelta import scene_delta
        return scene_delta(base, variant)

    def scene_dedup_saving(self, scenes):
        """Measure the content-addressed dedup saving across a set of scenes (holographic_scenedelta): {'naive',
        'unique', 'saving_x'} -- how much the automatic component sharing buys (measured ~4-6x across a base + its
        variants). The saving is automatic from content-hashing; this quantifies it. Returns a dict."""
        from holographic.scene_and_pipeline.holographic_scenedelta import scene_dedup_saving
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
        from holographic.caching_and_storage.holographic_history import VersionedStore
        return VersionedStore(self.dim, gop_len=gop_len)

    def scene_translation(self, t):
        """A 4x4 translation transform for scene_graph nodes (holographic_scenegraph)."""
        from holographic.scene_and_pipeline.holographic_scenegraph import translation
        return translation(t)

    def scene_scaling(self, s):
        """A 4x4 scale transform (uniform scalar or per-axis length-3) for scene_graph nodes."""
        from holographic.scene_and_pipeline.holographic_scenegraph import scaling
        return scaling(s)

    def scene_rotation(self, axis, angle):
        """A 4x4 rotation transform (Rodrigues, radians) for scene_graph nodes."""
        from holographic.scene_and_pipeline.holographic_scenegraph import rotation
        return rotation(axis, angle)

    def scene_compose_transforms(self, *matrices):
        """Compose 4x4 transforms (the product M0 @ M1 @ ..., parent then child) for scene_graph nodes."""
        from holographic.scene_and_pipeline.holographic_scenegraph import compose_transforms
        return compose_transforms(*matrices)

    def chain_structure(self, n):
        """Build an n-node linked-list CHAIN as a typed structure (B7), at this mind's dim/seed:
        M = superpose_i bind(node_i, node_{i+1}). Returns (recipe, nodes) -- realize(recipe) gives the
        chain-memory vector, and decode_structure(memory, nodes) traverses it back. The smallest honest
        forward object whose INVERSE (per-peel decode) is the interesting part (holographic_peel)."""
        from holographic.rendering.holographic_peel import chain_recipe
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
        from holographic.rendering.holographic_peel import traverse
        nodes = np.asarray(nodes)
        steps = (len(nodes) - 1) if steps is None else steps
        return traverse(np.asarray(memory), nodes, steps, cleanup=cleanup, beta=beta)

    def _plan_vocab(self):
        """The seed-deterministic atom source for shaped plans/records on this mind (regenerable from the
        mind's seed, like every other store). Cached."""
        if getattr(self, "_pvocab", None) is None:
            from holographic.mesh_and_geometry.holographic_planshape import ShapeVocab
            self._pvocab = ShapeVocab(min(self.dim, 1024), seed=self.seed + 31)
        return self._pvocab

    def plan_shape(self, actions, scopes, branch_skeleton=None):
        """Build the SHAPE (the decode key) for a contingency plan: the action and scope value codebooks and a
        nested-dict branch skeleton {name: {name: {...}}}. Hand this to decode_plan / descend so reading a plan
        vector is a deterministic unbind-and-clean walk, not the resonator's blind search."""
        from holographic.mesh_and_geometry.holographic_planshape import plan_shape
        return plan_shape(actions, scopes, branch_skeleton)

    def encode_plan(self, plan):
        """Encode a PlanNode contingency tree (a primary action, a scope, named branches each a PlanNode) as ONE
        hypervector -- the structured branching output the planner was missing, the HRR role-filler nested
        encoding. import PlanNode from holographic_planshape to build the tree."""
        from holographic.mesh_and_geometry.holographic_planshape import encode_plan
        return encode_plan(plan, self._plan_vocab())

    def decode_plan(self, vec, shape):
        """Decode a plan vector back to a PlanNode GIVEN ITS SHAPE (from plan_shape) -- schema-guided, so it
        stays exact far past the resonator's blind-parse cap. The returned node's confidence is the measured
        decode cosine of its action."""
        from holographic.mesh_and_geometry.holographic_planshape import decode_plan
        return decode_plan(vec, shape, self._plan_vocab())

    def descend(self, vec, situation, shape):
        """Walk a plan vector to the branch matching the current SITUATION (a branch-name str, or a state
        vector), returning the actions along that path -- the generalisation of IFMATCH from one gated
        instruction to a named branch tree WITH ABSTENTION (no branch clears the measured noise floor -> the
        node's primary action). Togelius's behavior-tree selector + Cranmer's measured floor."""
        from holographic.mesh_and_geometry.holographic_planshape import descend
        return descend(vec, situation, shape, self._plan_vocab())

    def encode_record(self, fields):
        """Encode a flat record {field_name: value_name} as one vector -- the GENERAL 'bring your own shape'
        path for any structured output (a scientific decision record, a classified state), not just plans. One
        bundle of role-bound value atoms."""
        from holographic.mesh_and_geometry.holographic_planshape import encode_record
        return encode_record(fields, self._plan_vocab())

    def decode_record(self, vec, schema):
        """Decode a flat record GIVEN ITS SHAPE: schema = {field_name: [possible_value_names]} (the per-field
        codebooks). Unbinds each role and cleans against that field's codebook -- a deterministic walk. Returns
        {field_name: value_name}."""
        from holographic.mesh_and_geometry.holographic_planshape import decode_record
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
        from holographic.misc.holographic_directed import build
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
        from holographic.misc.holographic_directed import successors
        return successors(ds, node_index, topk=topk, thresh=thresh)

    def directed_traverse(self, ds, start_index=0, floor=0.15, max_steps=64, min_steps=1):
        """Walk a directed chain FORWARD from `start_index`, gated by recovery confidence -- the directed
        substrate (RAY-3) under the throughput-gated traversal (RAY-1). Each hop recovers the successor and
        reports its cleanup cosine as throughput; the walk stops when that drops below `floor` (the chain
        exhausted, the ray dark). Returns the TraversalResult (payloads = the recovered node indices in
        order). Unambiguously forward, because the direction role suppressed the predecessor leak."""
        from holographic.misc.holographic_directed import make_step
        from holographic.misc.holographic_traverse import gated_traverse
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
        from holographic.scene_and_pipeline.holographic_plan import plan as _plan
        return _plan(start, field_step, max_steps=max_steps, floor=floor,
                     seed=self.seed if seed is None else seed,
                     action_of=action_of, is_branch=is_branch)

    def replan_needed(self, p, executed, tile_ok=None, floor=0.15):
        """The cheap per-tick guard for a baked Plan: should the courier abandon it and re-anchor (call
        plan() again)? True when the plan is exhausted, the next baked step's throughput has fallen below
        `floor`, or `tile_ok(next_tile)` reports the next tile is no longer clear/on-route. Otherwise False
        -- execute the next baked step. No value() calls, no decode work: a list index and a comparison."""
        from holographic.scene_and_pipeline.holographic_plan import replan_needed as _replan
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
        from holographic.scene_and_pipeline.holographic_plan import plan_route as _plan_route
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
        from holographic.scene_and_pipeline.holographic_plan import chunk_route as _chunk_route
        return _chunk_route(items, chunk=chunk, floor=floor,
                            seed=self.seed if seed is None else seed, action_of=action_of)

    def index_route(self, route):
        """Build a sub-linear RANDOM-ACCESS index over a chunked route (from plan_route / chunk_route): a BVH
        over the chunks. "Where am I on this route?" becomes a jump, not a replay from the start -- index each
        chunk by a summary vector and locate a query two-level (nearest chunk summary, then nearest tile within
        it), ~(#chunks + chunk_size) comparisons instead of #tiles. Build once, query many (the courier asking
        its position every tick). Returns a RouteIndex; call its .locate(query) -> (chunk, position, global_step)."""
        from holographic.scene_and_pipeline.holographic_plan import RouteIndex
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
        from holographic.scene_and_pipeline.holographic_plan import dedup_chunks
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
        from holographic.mesh_and_geometry.holographic_manifold import detect_topology, decompose_on_manifold
        from holographic.agents_and_reasoning.holographic_symbolic import compress_signal
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
        from holographic.misc.holographic_downscale import find_pattern_by_downscale
        return find_pattern_by_downscale(np.asarray(data, float), kind=kind, k=k, n_null=n_null, seed=seed)

    def multires_pyramid(self, signal, n_levels=5):
        """Build an anti-aliased mipmap of `signal` -- [full, half, quarter, ...], each level low-pass filtered
        before downsampling by two (SCALE-1). The decisive property is anti-aliasing on a COARSE read: a coarse
        pyramid level is a clean (alias-free), smaller view, where naively subsampling the full store folds
        high-frequency content into the low band. The levels are also a progressive code (coarsest is a usable
        approximation, finer levels add detail back, exact at the top). Returns the list of levels, coarsest last."""
        from holographic.misc.holographic_multires import build_pyramid
        return build_pyramid(signal, n_levels=n_levels)

    def pyramid_reconstruct(self, level, n):
        """Resample a pyramid level (from multires_pyramid) back to length `n` -- the LOD read, so a coarse,
        anti-aliased level can be used or compared at full length (SCALE-1)."""
        from holographic.misc.holographic_multires import upsample_to
        return upsample_to(level, n)

    def manifold_denoise(self, x, manifold, beta=18.0, steps=8):
        """Settle a (noisy) point ONTO a sample-defined manifold by looping a dense-Hopfield step (XDATA-2) --
        denoising as iterated projection. Generalises the codebook cleanup to ANY manifold given as a point cloud
        (a curved manifold, or a consolidation subspace from find_pattern_by_downscale). Idempotent: once on the
        manifold, further steps leave it fixed. Beats interpolation on a curved manifold (the chord midpoint
        leaves the manifold; this settles it back)."""
        from holographic.misc.holographic_diffuse import settle
        return settle(np.asarray(x, float), np.asarray(manifold, float), beta=beta, steps=steps)

    def manifold_generate(self, manifold, steps=30, beta_lo=2.0, beta_hi=25.0, noise_hi=0.5,
                          noise_lo=0.0, settle_steps=5, seed=0):
        """Generate a NOVEL-but-VALID sample on a sample-defined manifold by annealed diffusion (XDATA-2): from
        noise, loop the denoise step with beta rising and injected noise falling, then settle. Lands ON the
        manifold (valid) but BETWEEN the stored samples (novel) -- where bare-codebook generation just returns a
        stored sample. The B10 diffusion generalised off the discrete codebook to a learned/composed manifold."""
        from holographic.misc.holographic_diffuse import generate
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
        from holographic.rendering.holographic_denoisehome import Denoise                    # the Denoise home  consolidation R5
        return Denoise.sharpen(np.asarray(x, float), blur=blur, sigma=sigma, lam=lam, iters=iters, noise_level=noise_level)

    def smooth_sharp_split(self, x, k_smooth, k_sharp):
        """Split a signal into a SMOOTH layer (its k_smooth lowest-frequency coefficients) and a SHARP layer (the
        k_sharp largest residual samples -- sparse in the sample domain) (CACHE-2). At a budget covering both
        layers this beats any single basis, because no single basis is cheap across smooth-plus-sharp content (the
        spikes are broadband in frequency but sparse in samples). Returns a TwoLayerCode; reconstruct with
        smooth_sharp_reconstruct. The right sharp basis matches the sharp content (sample-sparse for spikes)."""
        from holographic.misc.holographic_twolayer import smooth_sharp_split
        return smooth_sharp_split(np.asarray(x, float), k_smooth, k_sharp)

    def smooth_sharp_reconstruct(self, code):
        """Reconstruct a signal from a two-layer code (CACHE-2): the smooth layer everywhere plus the exact sharp
        residual at the stored sharp positions."""
        from holographic.misc.holographic_twolayer import smooth_sharp_reconstruct
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
        from holographic.misc.holographic_graphsignal import graph_denoise
        forest = None
        if sublinear:
            from holographic.misc.holographic_tree import HoloForest
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
        from holographic.misc.holographic_chart import manifold_chart
        forest = None
        if sublinear:
            from holographic.misc.holographic_tree import HoloForest
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
        from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, fit_manifold_full, adaptive_manifold_denoise, codebook_denoise, nlm_denoise, pnp_restore, effective_rank, trajectory_denoise
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
            from holographic.sampling_and_signal.holographic_spectral import SpectralBasis
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


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p03_build_predictor", "_UnifiedPart03")
    print("holographic_unified_p03_build_predictor selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

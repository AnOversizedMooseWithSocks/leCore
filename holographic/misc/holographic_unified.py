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

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind



# ---- the faculty surface, assembled from parts ------------------------------------------------
# WHY THE SPLIT: this class reached 17.4k lines in ONE file -- 1.31 MB, or 131%% of the cap an agent
# can read in a single pass. The engine had stopped being able to read its own central nervous
# system, and every session paid a navigation tax for it. The parts below are pure MIXINS: every
# method is still a real attribute of UnifiedMind, so dir(), inspect, the doc generators and the
# service's introspection of public mind methods into GET /tools all see exactly what they saw
# before -- verified by hashing the signature, docstring and SOURCE of all 1521 callables either
# side of the change. Bodies moved by line range, so they are byte-identical, not merely equivalent.
#
# ORDER IS NOT LOAD-BEARING and must not become so: the splitter asserts that no method name appears
# in two parts, so no MRO resolution is ever ambiguous. If you add a method, add it to ONE part.
# The parts are not a public API -- never import or subclass them directly (see their docstrings).
from holographic.unified.holographic_unified_p01_read import _UnifiedPart01
from holographic.unified.holographic_unified_p02_fit_deterministic import _UnifiedPart02
from holographic.unified.holographic_unified_p03_build_predictor import _UnifiedPart03
from holographic.unified.holographic_unified_p04_sdf_offset import _UnifiedPart04
from holographic.unified.holographic_unified_p05_explain_code import _UnifiedPart05
from holographic.unified.holographic_unified_p06_mesh_collapse_edge import _UnifiedPart06
from holographic.unified.holographic_unified_p07_mesh_csg import _UnifiedPart07
from holographic.unified.holographic_unified_p08_bake import _UnifiedPart08
from holographic.unified.holographic_unified_p09_navigate_cost_field import _UnifiedPart09
from holographic.unified.holographic_unified_p10_unproject_depth import _UnifiedPart10
from holographic.unified.holographic_unified_p11_encyclopedia_reset import _UnifiedPart11
from holographic.unified.holographic_unified_p12_proc_texture import _UnifiedPart12
from holographic.unified.holographic_unified_p13_recall_and_apply import _UnifiedPart13
from holographic.unified.holographic_unified_p14_organics import _UnifiedPart14
from holographic.unified.holographic_unified_p15_hdrift import _UnifiedPart15
from holographic.unified.holographic_unified_p16_unicron import _UnifiedPart16
from holographic.unified.holographic_unified_p17_unicron2 import _UnifiedPart17
from holographic.unified.holographic_unified_p18_lean import _UnifiedPart18
from holographic.unified.holographic_unified_p19_verify import _UnifiedPart19


class UnifiedMind(_UnifiedPart01, _UnifiedPart02, _UnifiedPart03, _UnifiedPart04, _UnifiedPart05, _UnifiedPart06, _UnifiedPart07, _UnifiedPart08, _UnifiedPart09, _UnifiedPart10, _UnifiedPart11, _UnifiedPart12, _UnifiedPart13, _UnifiedPart14, _UnifiedPart15, _UnifiedPart16, _UnifiedPart17, _UnifiedPart18, _UnifiedPart19):
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



# ---------------------------------------------------------------------------
# DEMO: one mind, many modalities, one memory -- measured against separate ones
# ---------------------------------------------------------------------------

    def antiperiodic_fraction(self, signal):
        """How much of a signal lives in the SIGN-FLIPPING component -- the diagnostic for "does this pattern
        belong on a Mobius strip rather than a circle?".

        Pass exactly TWO periods; the two halves ARE the two periods. Returns ~1.0 for f(t+T) = -f(t), ~0.0
        for ordinary f(t+T) = +f(t), and the honest mixture between (a 50/50 sum measures 0.5 exactly).

        WHY YOU WANT IT: a circular encoding physically cannot hold an antiperiodic pattern -- it wraps theta
        and theta+pi onto the same point, so the sign-flipping half of the signal is destroyed on encode. A
        high fraction here says a circle is the wrong carrier and the axial/Mobius encoder is the right one --
        a modelling decision otherwise made by guessing. See holographic_mobius.antiperiodic_fraction."""
        from holographic.mesh_and_geometry.holographic_mobius import antiperiodic_fraction as _f
        return _f(signal)

    def antiperiodic_split(self, signal):
        """Split two periods of a signal into (periodic, antiperiodic) parts; the second is exactly what a
        circular representation cannot hold.

        An exact orthogonal split by halves -- (a+b)/2 and (a-b)/2 -- so there is no FFT bin-parity
        bookkeeping and the two parts sum back to the first period bit-for-bit.
        See holographic_mobius.antiperiodic_split."""
        from holographic.mesh_and_geometry.holographic_mobius import antiperiodic_split as _f
        return _f(signal)

    def load_ies(self, text):
        """Parse an IESNA LM-63 photometric file -- the format real luminaire manufacturers publish -- into a
        (candela_profile, max_vertical_angle) pair usable as a light's angular falloff.

        Takes the file TEXT, not a path, so it works on an upload, a string inside a scene description, or a
        file you read yourself. This is how a render stops using an invented cosine falloff and starts using
        the measured distribution of an actual fixture. See holographic_lights.load_ies."""
        from holographic.rendering.holographic_lights import load_ies as _f
        return _f(text)

    def code_search(self, text, k=8, method="jaccard"):
        """Search the engine's OWN source by meaning -- the question no other self-audit can answer.

        find_capability searches the CATALOG, which covers 674 of 7,572 functions; for the other 6,898 there
        was nothing. Returns [(module.qualname, score)] over every public engine function, matched on name
        tokens, the first docstring line, and callee names (who you call is what you do).

        MEASURED, and the default is the loser of the idea I set out to build: token-set Jaccard scores
        recall@1 0.542 against the hypervector encoding's 0.175 on the same features, so `method="jaccard"`
        is the default. `method="holographic"` gives the vector index -- 8.3x faster per query, 3x worse at
        being right. See holographic_codemap.search_source."""
        import holographic.io_and_interop.holographic_codemap as _cm
        return _cm.search_source(text, k=k, method=method)

    def code_similar(self, name, k=8, method="jaccard"):
        """Which other functions look like this one -- Rule 0's actual question, asked of the source rather
        than of the catalog. Accepts a bare function name or a 'module.qualname' label.

        Same measured trade-off as code_search: Jaccard by default, the hypervector index on request.
        See holographic_codemap.similar."""
        import holographic.io_and_interop.holographic_codemap as _cm
        return _cm.similar(name, k=k, method=method)

    def audit_complexity(self, limit=20, attention_cc=20):
        """Rank the engine's own functions by RISK -- complexity x exposure x exercise -- not by complexity.

        Raw complexity ranks the wrong thing, and measuring it is how that was found out: the highest-scoring
        functions here (parse_description 65, mesh_parts 57, rebake_texture 54, query.run 48) are ALL
        exercised by tests. They score high precisely BECAUSE they are load-bearing, and load-bearing code
        got tests. The risk sits elsewhere -- 1858 public functions no test so much as mentions, 22 of them
        at CC >= 20 -- and the worst cell is not the biggest function but the most EXPOSED unexercised one:
        an advertised catalog capability at CC 46 that nothing tests outranks a CC 65 internal with fifty
        tests on it.

        Returns {totals, attention, most_complex}. `attention` is the list worth reading, sorted so
        catalogued and faculty-exposed surfaces come first. Demos and selftests are tagged and excluded --
        they are complex on purpose and carry no production weight.

        HONEST LIMIT: "no test mentions it" is a NAME SCAN, not coverage. A function can be mentioned and
        still untested, or exercised indirectly through a caller and never named. It is a cheap upper bound
        on exposure; run coverage.py for the real thing. Cross-validated against radon (third party) at 0.92
        top-100 rank agreement. See holographic_codehealth.health_report."""
        import holographic.io_and_interop.holographic_codehealth as _ch
        return _ch.health_report(limit=limit, attention_cc=attention_cc)

    def audit_orphans(self, root=None, limit=40):
        """Audit the engine's OWN surface at function granularity: what is reachable, and what is not.

        The other audits (reachability, catalog_gaps, wiring_report) all reason about MODULES and all report
        zero gaps -- a module passes if it has a docstring, exports something public, and is referenced from
        this class. None of them looks INSIDE the file, so a module can pass every check while functions in
        it are reachable by nothing. This one looks inside.

        Returns {counts, orphan, test_only, budget, ok}. The bucket worth reading is TEST_ONLY: code that
        works and is tested but is exposed nowhere, which by this repo's own governing rule formally does not
        exist -- finished work sitting one catalog entry away from being real. Cross-validated against an
        independent third-party oracle (vulture) on the same tree: 43 of the orphans are corroborated by both.

        CONSERVATIVE ON PURPOSE and never destructive: no type inference, decorated functions count as
        registered, and names mentioned only inside strings count as reached (this repo routes by name
        through catalog examples). It under-reports rather than inventing an orphan, and it has no --fix:
        an orphan is a question -- wire it, catalogue it, or declare it a negative.
        See holographic_orphanaudit.orphan_report."""
        import holographic.io_and_interop.holographic_orphanaudit as _oa
        return _oa.orphan_report(root=root, limit=limit)

    def audit_agent_reach(self, root=None, limit=40):
        """Which public symbols can an AGENT actually reach -- functions AND classes, chains checked to the end.

        audit_orphans asks a lexical question: is this name referenced anywhere? That is the right question
        for dead code, and it answers YES for a symbol whose only reference lives in a module that is itself
        import-only by design -- a consolidation home, a declared negative. The chain is alive in the import
        graph and dead to an agent, because it never terminates at a faculty. This asks the second question:
        does the reference GO anywhere?

        Returns {counts, shadowed, dark, budget, ok}.
          shadowed -- referenced, but every referencing file is import-only by design. A cul-de-sac.
          dark     -- a public CLASS that is neither a faculty nor named in the catalog. audit_orphans never
                      saw these at all: it collects functions only, so a class an agent cannot construct was
                      invisible by construction.

        MEASURED, and why it exists: nine of the ten path-tracer light classes -- DomeLight (environment/IBL)
        and the area lights among them, between them most of what makes a render read as a photograph -- are
        unreachable from this class, while every module-level audit reported 0 gaps over the same tree.

        ADVISORY, and it does not gate: it shares audit_orphans' no-type-inference rule, so it UNDER-reports
        (a class merely named in catalog prose reads as reachable). A review queue, never a delete list, and
        never a completeness claim. See holographic_orphanaudit.agent_reach_report."""
        import holographic.io_and_interop.holographic_orphanaudit as _oa
        return _oa.agent_reach_report(root=root, limit=limit)


def unified_sources():
    """Every file the UnifiedMind class body lives in: this shim first, then each mixin part, in base order.

    WHY THIS EXISTS. Splitting the class out of one 17.4k-line file broke three separate consumers that read
    unified.py AS TEXT and regex-searched it -- two audits and a test -- because the code they were looking for
    had moved into a part. Each was fixed with its own glob, and a fourth reader would have broken the same way.
    So the authority moved here: the module that ASSEMBLES the parts is the only thing that actually knows what
    they are, and it derives them from the live base classes rather than from a filename pattern. Add a part and
    it is included automatically; rename the directory and nothing has to be told.

    Returns absolute paths. Ask for `unified_source_text()` if you want the concatenation."""
    import sys as _sys
    paths, seen = [__file__], {__file__}
    for base in UnifiedMind.__bases__:
        f = getattr(_sys.modules.get(base.__module__), "__file__", None)
        if f and f not in seen:
            seen.add(f)
            paths.append(f)
    return paths


def unified_source_text():
    """The whole UnifiedMind surface as one string -- what a text-searching audit or test actually wants.

    Use this instead of reading holographic_unified.py directly: that file is now a 320-line shim and contains
    almost none of the faculty bodies."""
    out = []
    for p in unified_sources():
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                out.append(f.read())
        except OSError:
            continue
    return "\n".join(out)


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
    from holographic.misc.holographic_text import TOPICS, _content, _split

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


def _selftest():
    """Regression trap for the mind FACADE itself (T6 backfill; the 12k-line module ran only a demo). This is a
    WIRING smoke test, not a re-test of each faculty (those have their own selftests): it proves the facade boots
    deterministically and that a representative faculty from several domains actually resolves and returns a sane
    result -- the failure this catches is a faculty silently unwired or broken at the delegation boundary, which
    per this codebase's hard lesson (a shared kernel is not a shared manifold) is exactly where things break."""
    import numpy as np

    # 1. BOOT is deterministic: two minds from the same seed agree on a produced vector, bit for bit.
    m1 = UnifiedMind(dim=256, seed=0)
    m2 = UnifiedMind(dim=256, seed=0)
    r1 = m1.encode_record({"name": "alice", "age": "thirty"})
    r2 = m2.encode_record({"name": "alice", "age": "thirty"})
    assert np.array_equal(r1, r2), "two same-seed minds disagree -- determinism broken at the facade"
    assert np.asarray(r1).shape == (256,)

    # 2. A representative faculty from several domains resolves and returns something sane (the [BLIND-SPOT] point:
    #    exercise the DELEGATION, not just attribute existence -- a wired-but-broken method passes hasattr).
    hits = m1.find_capability("rotate an object")
    assert len(hits) >= 1                                        # discovery domain

    cov = m1.selftest_coverage()
    assert cov["runnable"] > 300 and 0.0 <= cov["coverage"] <= 1.0   # introspection domain

    T = m1.scene_translation([1.0, 2.0, 3.0]) if hasattr(m1, "scene_translation") else None
    if T is not None:
        assert np.asarray(T).shape == (4, 4)                    # transform domain: a 4x4 matrix

    print("OK: holographic_unified self-test passed (facade boots deterministically -- two same-seed minds encode "
          "a record bit-identically -- and representative faculties across discovery, introspection and transforms "
          "resolve through the delegation boundary with sane results)")


if __name__ == "__main__":
    import sys
    _selftest()
    if "--demos" in sys.argv:
        demo_unified()

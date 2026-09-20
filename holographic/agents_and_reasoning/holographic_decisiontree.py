"""
holographic_decisiontree.py -- the suggestion node, grown into a TREE that remembers.

WHY THIS MODULE EXISTS
----------------------
The semantic router ALREADY generates a decision node on the fly. Ask it something it cannot
parse and `route` returns {'decision':'choose', 'prompt':'Ambiguous -- did you mean one of
these?', 'options':[...]} -- context in, available choices out, built fresh from the live
catalog. `route_or_abstain` adds the null gate and keeps the turned-down ranking in `refused`
so the decision stays appealable. That is one node of a decision tree, and it is good.

An audit of that path (2026-09-19, sweep 173) found it stops in exactly three places:

  1. DEPTH 1. A suggested option carries {name, does, call} and DROPS the produces/consumes
     the catalog already holds, so an option cannot be expanded into what it RESULTS in and
     which choices follow. The typed edge itself works -- Catalog.suggest_pipeline chains
     points -> mesh to fit_pose / sdf_to_mesh -- it simply is not reachable from a suggestion.
  2. AMNESIA. The same request twice returns a byte-identical suggestion. Nothing records
     which option was taken or what it produced, so the tree cannot improve.
  3. NO EQUIVALENCE. Nothing answers "which inputs reach the same result", and nothing warns
     when SIMILAR inputs reach DIFFERENT results.

This module closes 1 and 3 and gives 2 a place to live. It is a composition layer, not a new
router: the ranking is the catalog's, the per-node decision is decide_or_abstain, the tree
object is holographic_planshape.PlanNode, the encoding is encode_plan, the walk is descend.

MEASURED COVERAGE, STATED UP FRONT (the honest limit of typed expansion):
only 103/874 catalog capabilities declare `produces` and 79/874 declare `consumes`. So a
TYPED tree can expand roughly 12% of the surface; everywhere else a node is a leaf whose
branches are {done, abstain}. Pretending otherwise would make the tree look deeper than the
catalog's own type annotations can justify. Deeper typing is a catalog job, not a knob here.

KEPT NEGATIVES (on the record so they are not re-attempted as new ideas):
  * Typed children are ranked BY NAME with equal scores, not by a second semantic scorer.
    Equal scores mean decide_or_abstain abstains (gap 0 < margin) and the node asks -- which
    is the honest outcome: the type says which options are legal, it does not say which one
    the user meant. Fusing the intent score into typed children needs one scorer that can
    score an ARBITRARY named capability against a context; find_scored only returns top-k, so
    doing it today would mean a second, inconsistent ranking. Named, deferred, not faked.
  * OutcomeMemory keys on bind(input, tree), never on the tree alone. The resonator lesson
    (and plan_warm's docstring) is that the key is the goal context; a chain keyed on itself
    recalls whatever it just did.
  * A recall that cannot clear its margin returns result=None. An outcome store that always
    names its nearest row is a lookup table with a confidence sticker on it.
"""
import hashlib

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, cosine
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
from holographic.mesh_and_geometry.holographic_planshape import (
    PlanNode,
    ShapeVocab,
    decode_plan,
    derived_atom,
    descend,
    encode_plan,
    plan_shape,
)
from holographic.misc.holographic_relations import decide_or_abstain

# The two honest terminal branches every node carries. "done" = the choice produced its
# result and nothing further is typed; "abstain" = the node could not decide and must ask,
# which is exactly what route() already does at depth 1.
DONE = "done"
ABSTAIN = "abstain"
# The branch name used when a capability declares no `produces`: "that worked". ~88% of the
# catalog is untyped, so this is the ordinary case and the tree still gets depth there.
OK = "ok"


class CapabilityView:
    """The live capability surface as the grower needs to see it: rank the choices for a
    context, say what a choice PRODUCES, and say who CONSUMES a result kind.

    Three callables so the grower can be driven by the real catalog in production and by a
    tiny fixed stub in the selftest -- a grower that can only be tested against 874 live
    capabilities is a grower whose tests drift every time the catalog does.
    """

    def __init__(self, rank, produces, consumers):
        self._rank = rank
        self._produces = produces
        self._consumers = consumers

    @classmethod
    def from_catalog(cls, catalog=None):
        """Bind to the real catalog. DELEGATES ranking to Catalog.find_scored -- the same
        scoring and the same deterministic tie-break the router itself uses."""
        if catalog is None:
            from holographic.caching_and_storage.holographic_catalog import default_catalog
            catalog = default_catalog()

        def rank(context, k):
            return [(c.name, float(s)) for c, s in catalog.find_scored(context, k=k)]

        def produces(name):
            cap = catalog.get(name)
            p = getattr(cap, "produces", None) or ()
            return tuple(p) if not isinstance(p, str) else (p,)

        def consumers(kind, k):
            # Deterministic by name: the type relation is a SET, not a ranking. Equal scores
            # make the node abstain and ask, which is the honest read of "several legal next
            # steps". See the kept negative in the module docstring.
            out = []
            for cap in catalog.all():
                c = getattr(cap, "consumes", None) or ()
                c = (c,) if isinstance(c, str) else tuple(c)
                if kind in c:
                    out.append(cap.name)
            return [(n, 1.0) for n in sorted(out)[:k]]

        return cls(rank, produces, consumers)

    def rank(self, context, k=3):
        return list(self._rank(context, k))

    def produces(self, name):
        return tuple(self._produces(name))

    def consumers(self, kind, k=3):
        return list(self._consumers(kind, k))


class GrownTree:
    """A contingency tree grown on the fly, plus the evidence behind every node.

    `root` is a plain PlanNode (the shared decision-tree type), so everything that already
    reads a PlanNode -- encode_plan, decode_plan, descend, validate -- works on it unchanged.
    `options` keeps the ranked alternatives and the confidence at each node keyed by its path,
    which is the part a suggestion list is FOR: the node's action is what we would do, the
    options are what we would offer if asked.
    """

    def __init__(self, root, options, coverage):
        self.root = root
        self.options = options        # {path tuple: {"ranked": [...], "confident": bool}}
        self.coverage = coverage      # {"nodes", "typed", "untyped"}

    def __repr__(self):
        return "GrownTree(root=%r, nodes=%d, typed=%d)" % (
            self.root.action, self.coverage["nodes"], self.coverage["typed"])

    def offer(self, path=()):
        """The suggestion a caller should show at `path` -- the ranked options and whether the
        node was confident enough to just act. This is route()'s 'choose' payload, at depth."""
        return self.options.get(tuple(path))


def grow_tree(context, view, depth=2, fanout=3, margin=0.1):
    """GROW a contingency tree for `context` from the live capability surface.

    At each node: rank the available choices for the context (the catalog's own ranking), take
    the shared decision step (decide_or_abstain), and branch on what that choice RESULTS in --
    one branch per produced kind, whose child node is the decision among the capabilities that
    CONSUME that kind. Every node also carries an `abstain` branch, because "I could not decide,
    ask the user" is a real outcome and hiding it is how a tree pretends to know things.

    Returns a GrownTree. `depth` counts decision levels (depth=1 reproduces today's flat
    suggestion node); `fanout` caps both the options per node and the branches per node.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    options = {}
    counts = {"nodes": 0, "typed": 0, "untyped": 0}

    def build(ctx, scope, d, path, seen):
        counts["nodes"] += 1
        # Ask for extra rows so that dropping already-taken actions still leaves `fanout`
        # real options -- otherwise a node near the top of the tree silently narrows.
        ranked = [(n, s) for n, s in view.rank(ctx, fanout + len(seen)) if n not in seen][:fanout]
        if not ranked:
            # Nothing in the catalog answers this at all -- an honest dead end, not a guess.
            options[path] = {"ranked": [], "confident": False, "context": ctx}
            return PlanNode(ABSTAIN, scope=scope, confidence=0.0)

        winner, score, confident = decide_or_abstain(ranked, margin=margin)
        action = winner if winner is not None else ranked[0][0]
        # `context` is kept per node (sweep 176): a pick reported against this node's record id teaches the
        # reflex keyed on THIS context, so the next walk of the same tree can answer the node from experience.
        options[path] = {"ranked": list(ranked), "confident": bool(confident), "context": ctx}

        # Branch NAMES come from the declared result kinds where the catalog has them (specific
        # and honest), and fall back to a single OK branch where it does not -- which is ~88%
        # of the surface, so this fallback is the common path, not the exception.
        kinds = list(view.produces(action)[:fanout])
        if kinds:
            counts["typed"] += 1
        else:
            counts["untyped"] += 1
            kinds = [OK]

        branches = {}
        for kind in kinds:
            if d > 1:
                # THE EDGE, and it is the measured one: the child's context is the ORIGINAL
                # request plus what just happened -- never the bare type kind. Measured on
                # "turn a point cloud into a mesh": the context+result edge kept 4/5 children
                # on topic, the consumers-of-that-type edge kept 0/5 (its 7 candidates all
                # scored 0.0 against the request). Coarse types cannot carry a second level.
                nxt = ("%s next step after %s" % (ctx, action) if kind == OK
                       else "%s now I have a %s" % (ctx, kind))
                branches[kind] = build(nxt, kind, d - 1, path + (kind,), seen | {action})
            else:
                branches[kind] = PlanNode(DONE, scope=kind, confidence=1.0)
        branches[ABSTAIN] = PlanNode("ask", scope=scope, confidence=0.0)
        return PlanNode(action, scope=scope, confidence=float(score or 0.0), branches=branches)

    root = build(context, "global", depth, (), frozenset())
    return GrownTree(root, options, counts)


def _collect(node, actions, scopes, skel):
    """Walk a PlanNode tree collecting its codebooks and nested branch skeleton."""
    actions.add(node.action)
    scopes.add(node.scope)
    for name, child in node.branches.items():
        sub = {}
        skel[name] = sub
        _collect(child, actions, scopes, sub)


def encode_tree(tree, dim=1024, seed=0):
    """Encode a grown tree as ONE hypervector, with the shape needed to read it back.

    DELEGATES to plan_shape / encode_plan -- the HRR nested role-filler encoding that already
    exists for contingency plans. Returns (vec, shape, vocab) so two trees are comparable by
    cosine and a walk can be replayed with descend().
    """
    root = tree.root if isinstance(tree, GrownTree) else tree
    actions, scopes, skel = set(), set(), {}
    _collect(root, actions, scopes, skel)
    # Sorted codebooks: the atoms are seed-derived by NAME, but a stated order keeps the shape
    # itself reproducible across runs and machines.
    shape = plan_shape(sorted(actions), sorted(scopes), skel)
    vocab = ShapeVocab(dim, seed)
    return encode_plan(root, vocab), shape, vocab


def read_tree(vec, shape, vocab):
    """Read a tree vector back to a PlanNode -- the schema-guided unbind walk (decode_plan)."""
    return decode_plan(vec, shape, vocab)


def walk(vec, situation, shape, vocab, floor=None):
    """Walk the encoded tree to the branch matching `situation` -- DELEGATES to descend()."""
    return descend(vec, situation, shape, vocab, floor=floor)


def routing_fingerprint(catalog=None, dim=1024, seed=0, k=8):
    """Encode a request by HOW IT ROUTES -- a score/rank-weighted bundle of the top-k capability
    atoms -- instead of by its own words.

    WHAT IT IS FOR, after four rounds of measurement (sweep 173): PAIRWISE EQUIVALENCE -- "do
    these two inputs lead to the same place?" -- and NOTHING predictive. Measured on held-out
    capability aliases with both aliases ABLATED from the router's index, 3 seeds x 150:

        does input similarity separate SAME-result pairs from DIFFERENT-result pairs? (AUROC)
        this fingerprint                 0.874   (spread 0.032)
        hashed n-gram bag encoder        0.777
        raw word overlap (Jaccard)       0.695

    That is the only place it beat every baseline outside the spread, and it makes sense: the
    fingerprint captures "routes to the same NEIGHBOURHOOD", which pairwise equivalence
    rewards and exact top-1 does not. Two phrasings that route alike encode alike.

    WHAT IT IS NOT FOR -- every framing below was tried against a proper baseline and LOST,
    so the next session does not re-run them:
      * As a router (closed world, answer always in the store): 0.720 vs router 0.540 LOOKED
        like a win. A one-line filter -- restrict the router to capabilities the store has
        seen -- scored 0.718. The whole gain was candidate-set restriction, not learning.
      * Open world (half the queries target never-observed capabilities): the router wins
        outright (0.549); the filter is wrong BY CONSTRUCTION on every unseen query (0.000);
        the store answers worst of all (0.304). Its one virtue: it ABSTAINED on 83% of the
        unseen queries instead of naming something wrong.
      * As a gate with the router as fallback: 0.561 vs 0.549, +0.012 inside a 0.067 spread.
      * Predicting whether the router's own top-1 is right: 0.608, vs 0.731 for the router's
        own top1-top2 margin -- a signal that is already free and that decide_or_abstain
        already uses. leCore already owned the best predictor of its own correctness.
    Ceiling that explains all of it: the answer must be in the router's top-k for the
    fingerprint to carry it (top-3 recall 0.696, top-8 0.793 on this split); at k=3 the
    store hit the top-3 ceiling exactly (+0.002). k=8 with rank decay is the default;
    score/rank 0.720, score^2 0.718, score^3 0.707, plain 0.704 are NOT distinguishable
    (spread 0.073), and k=20 dilutes to 0.593.
    """
    if catalog is None:
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        catalog = default_catalog()
    dim = int(dim)

    def encode(text):
        hits = catalog.find_scored(text, k=k)
        if not hits:
            return np.zeros(dim)
        v = np.zeros(dim)
        for rank, (cap, score) in enumerate(hits):
            # Rank decay: later hits are real signal (they lift us past the top-3 ceiling) but
            # noisier, so they contribute less rather than being cut off at an arbitrary k.
            v += (float(score) / (rank + 1.0)) * derived_atom(seed, "cap:" + cap.name, dim)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    return encode


class OutcomeMemory:
    """What a given INPUT, run through a given TREE, actually RESULTED in -- an AUDIT LOG of
    outcomes plus the equivalence classes that fall out of it. Its measured strength is
    pairwise: `compare` (same place or not, alike or not) rests on routing_fingerprint's
    AUROC 0.874. `recall` exists so the log can be queried and it ABSTAINS honestly, but it
    carries NO predictive claim: measured as a router, a filter, a gate and a self-correctness
    predictor it lost to a proper baseline every time (see routing_fingerprint's docstring).

    The key is bind(input, tree): the same question asked of a different tree is a different
    row, which is the whole point of remembering the tree as well as the input. Recall groups
    rows by RESULT (so many phrasings that led to the same place reinforce each other) and
    then takes the shared decision step, so an unfamiliar query abstains instead of being
    handed its nearest neighbour.
    """

    def __init__(self, dim=1024, seed=0, encoder=None):
        self.dim = int(dim)
        self.seed = int(seed)
        self._encode = encoder or hashed_ngram_encode(dim=self.dim)
        self._keys = []        # unit vectors, one per recorded run
        self._results = []     # the result label of each run
        self._labels = []      # a human-readable tag for the INPUT of each run
        self._states = []      # the raw input when it was text -- so an audit can match it EXACTLY

    # -- encoding helpers -------------------------------------------------------------
    def encode_state(self, state):
        """A state may arrive as text (encoded here, deterministically) or as a vector."""
        if isinstance(state, str):
            return self._encode(state)
        v = np.asarray(state, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def key(self, state, tree_vec=None):
        """The row key: bind(input, tree), or the input alone when no tree is given."""
        s = self.encode_state(state)
        if tree_vec is None:
            return s
        t = np.asarray(tree_vec, dtype=float)
        tn = float(np.linalg.norm(t))
        k = bind(s, t / tn if tn > 0 else t)
        kn = float(np.linalg.norm(k))
        return k / kn if kn > 0 else k

    # -- the loop ---------------------------------------------------------------------
    def record(self, state, result, tree_vec=None, label=None):
        """Remember that this input, through this tree, produced this result."""
        self._keys.append(self.key(state, tree_vec))
        self._results.append(result)
        self._labels.append(label if label is not None else
                            (state if isinstance(state, str) else "vec%d" % len(self._keys)))
        self._states.append(state if isinstance(state, str) else None)
        return {"n": len(self._keys), "result": result}

    def recall(self, state, tree_vec=None, margin=0.1, min_score=None):
        """What did inputs like this, through a tree like this, result in?

        Scores every recorded row by cosine, keeps the BEST row per result, ranks results, and
        abstains when the top result is not clearly ahead. Returns result=None on an abstain,
        with the ranking kept so the decision is appealable -- the same contract route_or_abstain
        uses for `refused`.
        """
        if not self._keys:
            return {"result": None, "score": 0.0, "confident": False,
                    "why": "empty", "ranked": [], "n": 0}
        q = self.key(state, tree_vec)
        best = {}
        for k, r in zip(self._keys, self._results):
            c = float(cosine(q, k))
            if r not in best or c > best[r]:
                best[r] = c
        # One tie rule everywhere (ISA-1): sort by (-score, name).
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], str(kv[0])))
        winner, score, confident = decide_or_abstain(ranked, margin=margin, min_score=min_score)
        why = "ok"
        if not confident:
            why = "min_score" if (min_score is not None and ranked[0][1] < min_score) else "margin"
        # ONE RECORDED RESULT IS NOT EVIDENCE. decide_or_abstain judges a top-1 by its lead over
        # the runner-up, so a single-element ranking is confident BY CONSTRUCTION -- measured:
        # ([("only", 0.02)], margin=0.1) returns confident=True. A store holding one result class
        # would therefore answer ANY query, nonsense included, with that class. With nothing to
        # out-rank, confidence can only rest on the absolute score, so a floor must be STATED:
        # without min_score we refuse to be confident rather than inherit a vacuous True.
        if len(ranked) == 1:
            if min_score is None:
                confident, why = False, "single_class"
            else:
                confident = ranked[0][1] >= float(min_score)
                why = "ok" if confident else "min_score"
        return {"result": winner if confident else None,
                "score": float(ranked[0][1]), "confident": bool(confident),
                "why": why, "ranked": ranked, "n": len(self._keys)}

    # -- equivalence ------------------------------------------------------------------
    def same_result(self, result):
        """Every recorded input that reached this result -- the equivalence class."""
        return [lab for lab, r in zip(self._labels, self._results) if r == result]

    def classes(self):
        """All equivalence classes: {result: [input labels]}."""
        out = {}
        for lab, r in zip(self._labels, self._results):
            out.setdefault(r, []).append(lab)
        return out

    def compare(self, a_state, b_state, a_tree=None, b_tree=None, near=0.5):
        """Do two inputs reach the same place, and were they alike to begin with?

        The four honest quadrants, because 'same result' alone hides the interesting cases:
          consistent -- alike in, same result out (the boring, good case)
          equivalent -- DIFFERENT inputs, same result: two roads to one place
          brittle    -- ALIKE inputs, different results: the alarm worth having
          distinct   -- different in, different out
        `near` is the input-cosine line between 'alike' and 'different'; it is a stated
        parameter, not a discovered constant, and it is the caller's to set per encoder.
        """
        # AUDIT FIRST, PREDICT NEVER: if an input was RECORDED, its result is a fact on file and is
        # used as such. recall() is only consulted for an input the log has never seen, and even
        # then it abstains rather than guess -- the audit role rests on facts, not on the
        # predictive role this store measurably does not have.
        def known(state):
            if isinstance(state, str) and state in self._states:
                return {"result": self._results[self._states.index(state)]}
            return None
        ra = known(a_state) or self.recall(a_state, a_tree)
        rb = known(b_state) or self.recall(b_state, b_tree)
        ca = float(cosine(self.encode_state(a_state), self.encode_state(b_state)))
        same = (ra["result"] is not None and ra["result"] == rb["result"])
        alike = ca >= near
        if ra["result"] is None or rb["result"] is None:
            quadrant = "unknown"
        elif same and alike:
            quadrant = "consistent"
        elif same:
            quadrant = "equivalent"
        elif alike:
            quadrant = "brittle"
        else:
            quadrant = "distinct"
        return {"quadrant": quadrant, "input_cosine": ca, "same_result": same,
                "a": ra["result"], "b": rb["result"]}

    def fingerprint(self):
        """A content hash of the store -- hashlib, never hash(), so it is stable across runs."""
        h = hashlib.sha256()
        for lab, r in zip(self._labels, self._results):
            h.update(("%s\x00%s\x00" % (lab, r)).encode("utf-8"))
        return h.hexdigest()[:16]


def _selftest():
    # A fixed stub surface: small, deterministic, and it does not drift when the real catalog
    # gains a capability. The real catalog is exercised by tools/decisiontree_bench.py.
    # The child context is built as "<ctx> now I have a <kind>" / "<ctx> next step after
    # <action>", so the stub answers those too -- the stub encodes the EDGE, which is the part
    # the live measurement chose.
    ranked = {"make a mesh from points": [("points_to_mesh", 9.0), ("mesh_smooth", 2.0)],
              "make a mesh from points now I have a mesh": [("mesh_smooth", 8.0),
                                                            ("field_displace", 1.0)],
              "tie town": [("alpha", 5.0), ("beta", 5.0)],
              "tie town next step after alpha": [("beta", 7.0), ("gamma", 1.0)]}
    produces = {"points_to_mesh": ("mesh",), "mesh_smooth": (), "alpha": (), "beta": (),
                "gamma": (), "field_displace": ()}
    consumes = {"mesh": [("field_displace", 1.0), ("mesh_smooth", 1.0)]}
    view = CapabilityView(
        rank=lambda ctx, k: ranked.get(ctx, [])[:k],
        produces=lambda n: produces.get(n, ()),
        consumers=lambda kind, k: consumes.get(kind, [])[:k])

    # 1. The root acts when one choice clearly wins (gap 7.0 >> margin).
    t = grow_tree("make a mesh from points", view, depth=2)
    assert t.root.action == "points_to_mesh", t.root
    assert t.offer(())["confident"] is True

    # 2. Branches are the RESULT kinds plus the honest abstain -- nothing else.
    assert set(t.root.branches) == {"mesh", ABSTAIN}, t.root.branches

    # 3. The child is ranked by the SEMANTIC edge (context + what just happened), and the
    #    parent's own action is excluded so a tree cannot recommend itself in a loop.
    child = t.root.branches["mesh"]
    off = t.offer(("mesh",))
    assert [n for n, _ in off["ranked"]] == ["mesh_smooth", "field_displace"], off
    assert off["confident"] is True, off          # 8.0 vs 1.0 is a clear win
    assert child.action == "mesh_smooth", child
    assert "points_to_mesh" not in [n for n, _ in off["ranked"]], off

    # 4. An untyped choice still gets depth via the OK branch -- the ~88% of the catalog that
    #    declares no `produces` is not stranded at depth 1.
    t2 = grow_tree("tie town", view, depth=2)
    assert set(t2.root.branches) == {OK, ABSTAIN}, t2.root.branches
    assert t2.offer(())["confident"] is False          # 5.0 vs 5.0 is a tie -> ask
    assert t2.root.branches[OK].action == "beta", t2.root.branches[OK]
    assert t2.coverage["untyped"] == 2 and t2.coverage["typed"] == 0, t2.coverage

    # 5. Encode -> decode round-trips the tree EXACTLY (PlanNode equality), and the decode
    #    confidence is a measured cosine, not a sticker.
    vec, shape, vocab = encode_tree(t, dim=1024, seed=0)
    back = read_tree(vec, shape, vocab)
    assert back == t.root, (back, t.root)
    assert back.confidence >= 0.5, back.confidence

    # 6. descend() walks to the branch matching the situation.
    path = walk(vec, vocab.value("mesh"), shape, vocab)
    assert path and path[0] == "points_to_mesh", path

    # 7. Determinism: the same grow, encoded twice, is BIT-identical. Tie-sensitive path.
    v2, _, _ = encode_tree(grow_tree("make a mesh from points", view, depth=2), dim=1024, seed=0)
    assert np.array_equal(vec, v2), "encoding is not bit-deterministic"

    # 8. Outcome memory: what went in comes back, exactly.
    om = OutcomeMemory(dim=1024, seed=0)
    om.record("turn these points into a mesh", "points_to_mesh", vec, label="q1")
    # An exact self-recall is cosine 1.0, but with ONE class on file the floor must be stated
    # (pin 12) -- so this asserts the exactness AND that a stated floor is what licenses it.
    r = om.recall("turn these points into a mesh", vec, min_score=0.9)
    assert r["result"] == "points_to_mesh" and r["score"] > 1.0 - 1e-9, r

    # 9. The SAME input through a DIFFERENT tree is a different row -- the tree is part of the
    #    key, which is the whole reason to remember it. MEASURED: 1.000 with the matching tree
    #    vs 0.541 with the other one. It does NOT fall to ~0, and that is correct rather than
    #    disappointing: both trees carry the same abstain/ask skeleton, so their encodings are
    #    genuinely correlated. The tree is a SOFT key -- a near-identical tree still recalls,
    #    an unrelated one decays -- not a hard partition. Pinned as a drop, not as a zero.
    other, _, _ = encode_tree(t2, dim=1024, seed=0)
    r_other = om.recall("turn these points into a mesh", other)
    assert r_other["score"] < 0.75, r_other
    assert r["score"] - r_other["score"] > 0.35, (r["score"], r_other["score"])

    # 10. Abstention: an unrelated query does not get handed the nearest row.
    om.record("smooth this mesh", "mesh_smooth", vec, label="q2")
    r3 = om.recall("photosynthesis in coastal algae", vec, min_score=0.35)
    assert r3["result"] is None and r3["why"] in ("margin", "min_score"), r3

    # 11. Equivalence: different phrasings, same result -> "equivalent"; and the brittle alarm
    #     fires when near-identical inputs reached different results.
    om.record("build a surface out of this point cloud", "points_to_mesh", vec, label="q3")
    assert sorted(om.same_result("points_to_mesh")) == ["q1", "q3"], om.classes()
    cmp_eq = om.compare("turn these points into a mesh",
                        "build a surface out of this point cloud", vec, vec)
    assert cmp_eq["same_result"] is True, cmp_eq
    assert cmp_eq["quadrant"] in ("equivalent", "consistent"), cmp_eq
    om2 = OutcomeMemory(dim=1024, seed=0)
    om2.record("smooth this mesh", "A", vec, label="a")
    om2.record("smooth this mesh please", "B", vec, label="b")
    cmp_br = om2.compare("smooth this mesh", "smooth this mesh please", vec, vec, near=0.5)
    assert cmp_br["quadrant"] == "brittle", cmp_br

    # 12. A single recorded result class is NOT evidence: with no runner-up to beat, recall
    #     refuses to be confident unless an absolute floor is stated. Without this, a store
    #     holding one class answers every query, nonsense included. Regression trap.
    om3 = OutcomeMemory(dim=1024, seed=0)
    om3.record("smooth this mesh", "only_one", vec, label="a")
    lone = om3.recall("photosynthesis in coastal algae", vec)
    assert lone["confident"] is False and lone["why"] == "single_class", lone
    assert om3.recall("smooth this mesh", vec, min_score=0.9)["confident"] is True
    assert om3.recall("photosynthesis in coastal algae", vec, min_score=0.9)["result"] is None

    # 13. The routing fingerprint: two DIFFERENT phrasings that route to the same capabilities
    #     encode to (nearly) the same vector, and a phrasing that routes elsewhere does not.
    #     This is the property the whole re-ranker rests on, so it is pinned, not assumed.
    class _StubCat:
        def __init__(self, table):
            self.table = table

        def find_scored(self, text, k=8):
            class _C:
                def __init__(self, n):
                    self.name = n
            return [(_C(n), s) for n, s in self.table.get(text, [])][:k]

    fp = routing_fingerprint(_StubCat({
        "phrasing one":   [("cap_a", 9.0), ("cap_b", 2.0)],
        "phrasing two":   [("cap_a", 8.0), ("cap_b", 3.0)],   # routes alike
        "something else": [("cap_z", 9.0), ("cap_y", 2.0)],   # routes elsewhere
    }), dim=1024, seed=0)
    alike = float(cosine(fp("phrasing one"), fp("phrasing two")))
    apart = float(cosine(fp("phrasing one"), fp("something else")))
    assert alike > 0.95, alike
    assert abs(apart) < 0.25, apart
    assert fp("never seen").tolist() == [0.0] * 1024      # no hits -> no fingerprint, not a guess

    return {"ok": True, "pinned": 13}


if __name__ == "__main__":
    print(_selftest())

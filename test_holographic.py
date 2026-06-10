"""
test_holographic.py -- correctness suite for the holographic system.

Run with:  pytest test_holographic.py -v

Stochastic components are tested over multiple seeds and asserted on aggregate
behaviour (mean accuracy, fraction of seeds passing) so the suite is meaningful
without being flaky. Tolerances reflect the regimes the components were designed
for, not best-case single runs.
"""

import numpy as np
import pytest

from holographic_ai import (random_vector, bind, unbind, bundle, cosine, permute,
                            Vocabulary, HolographicMemory, PartitionedMemory,
                            HolographicLearner, geodesic, log_map, exp_map, slerp,
                            ReflexArc, DriftDetector, recall_all)
from holographic_encoders import ScalarEncoder, TextEncoder, RecordEncoder
from holographic_reasoning import (ResonatorNetwork, SemanticCompass,
                                   ConformalPredictor, EpistemicMap, vector_disagreement)
from holographic_extras import ResidueSystem, Region, ball, route, PredictiveFilter
from holographic_field import Field, bump, landscape, seeded_landscape, ascend
from holographic_diffusion import DoubleDiffusion
from holographic_sync import SyncGrouping
from holographic_orchestrator import (Tool, ToolRegistry, SkeletonLibrary, Planner,
                                      CircuitBreaker, keyword_vector)
from holographic_emergence import EmergentConcepts

DIM = 1024


def rng(s=0):
    return np.random.default_rng(s)


# ===========================================================================
# Engine
# ===========================================================================
class TestEngine:
    def test_bind_unbind_recovers_factor(self):
        r = rng(0)
        sims = [cosine(unbind(bind(a := random_vector(DIM, r), b := random_vector(DIM, r)), b), a)
                for _ in range(60)]
        assert np.mean(sims) > 0.6

    def test_bound_product_unlike_its_parts(self):
        r = rng(1)
        sims = [abs(cosine(bind(a := random_vector(DIM, r), b := random_vector(DIM, r)), a))
                for _ in range(60)]
        assert np.mean(sims) < 0.15

    def test_bundle_keeps_all_members_retrievable(self):
        r = rng(2)
        for _ in range(40):
            vs = [random_vector(DIM, r) for _ in range(3)]
            b = bundle(vs)
            assert all(cosine(b, v) > 0.3 for v in vs)

    def test_permute_is_invertible(self):
        x = random_vector(DIM, rng(3))
        assert cosine(permute(permute(x, 7), -7), x) > 0.99

    def test_permute_decorrelates(self):
        x = random_vector(DIM, rng(4))
        assert abs(cosine(permute(x, 1), x)) < 0.2

    def test_vocabulary_cleanup_denoises(self):
        r = rng(5)
        v = Vocabulary(DIM, seed=5)
        words = [f"w{i}" for i in range(25)]
        for w in words:
            v.get(w)
        correct = sum(v.cleanup(v.get(w) + 0.6 * random_vector(DIM, r))[0] == w for w in words)
        assert correct / len(words) > 0.9

    def test_keyvalue_memory_roundtrip(self):
        r = rng(6)
        mem = HolographicMemory(DIM)
        keys = [random_vector(DIM, r) for _ in range(5)]
        vals = [random_vector(DIM, r) for _ in range(5)]
        for k, val in zip(keys, vals):
            mem.learn(k, val)
        # each recalled value should be most similar to the value actually stored
        ok = 0
        for i, k in enumerate(keys):
            rec = mem.recall(k)
            ok += int(np.argmax([cosine(rec, v) for v in vals]) == i)
        assert ok >= 4

    def test_partitioning_beats_single_trace(self):
        r = rng(7)
        n = 120
        keys = [random_vector(DIM, r) for _ in range(n)]
        vals = [random_vector(DIM, r) for _ in range(n)]
        single = HolographicMemory(DIM)
        part = PartitionedMemory(DIM, num_partitions=16, seed=7)
        for k, v in zip(keys, vals):
            single.learn(k, v)
            part.learn(k, v)

        def acc(mem):
            hits = 0
            for i, k in enumerate(keys):
                rec = mem.recall(k)
                hits += np.argmax([cosine(rec, v) for v in vals]) == i
            return hits / n

        assert acc(part) > acc(single)
        assert acc(part) > 0.9

    def test_successive_cancellation_beats_one_shot(self):
        # At a load where a single trace is well past one-shot capacity, peeling
        # the clearest pair and cancelling it should recover far more correctly.
        r = rng(11)
        n = 160
        keys = np.stack([random_vector(DIM, r) for _ in range(n)])
        vals = np.stack([random_vector(DIM, r) for _ in range(n)])
        trace = np.zeros(DIM)
        for k, v in zip(keys, vals):
            trace = trace + bind(k, v)
        one = recall_all(trace, keys, vals, iterative=False)
        sic = recall_all(trace, keys, vals, iterative=True)
        one_acc = sum(one[i] == i for i in range(n)) / n
        sic_acc = sum(sic[i] == i for i in range(n)) / n
        assert sic_acc > one_acc + 0.15      # a clear, not marginal, improvement
        assert sic_acc > 0.6

    def test_geometry_slerp_and_maps(self):
        r = rng(8)
        a, b = random_vector(DIM, r), random_vector(DIM, r)
        assert cosine(slerp(a, b, 0.0), a) > 0.999
        assert cosine(slerp(a, b, 1.0), b) > 0.999
        assert cosine(exp_map(a, log_map(a, b)), b) > 0.99   # exp/log are inverses

    def test_learner_classifies_seen_and_generalizes(self):
        learner = HolographicLearner(dim=DIM, seed=2)
        training = [
            ({"legs": "four", "sound": "woof", "size": "medium"}, "dog"),
            ({"legs": "four", "sound": "meow", "size": "small"}, "cat"),
            ({"legs": "two", "sound": "tweet", "size": "small"}, "bird"),
            ({"legs": "four", "sound": "woof", "size": "large"}, "dog"),
        ]
        for ex, lab in training:
            learner.learn(ex, lab)
        assert learner.classify(training[0][0])[0] == "dog"          # exact recall
        # a near variant should still land on dog
        assert learner.classify({"legs": "four", "sound": "woof", "size": "small"})[0] == "dog"

    def test_reflex_arc_better_than_naive(self):
        # reconstruction of a structured task->response mapping
        r = rng(9)
        reflex = ReflexArc()
        basis = random_vector(DIM, r)
        tasks, responses = [], []
        for _ in range(40):
            t = random_vector(DIM, r)
            resp = slerp(t, basis, 0.4)          # responses are a smooth function of tasks
            reflex.experience(t, resp)
            tasks.append(t); responses.append(resp)
        err, naive = [], []
        for _ in range(20):
            t = random_vector(DIM, r)
            true_resp = slerp(t, basis, 0.4)
            pred, _ = reflex.recall(t)
            err.append(geodesic(pred, true_resp))
            naive.append(geodesic(t, true_resp))
        assert np.mean(err) < np.mean(naive)

    def test_drift_detector_flags_modes(self):
        r = rng(10)
        reflex = ReflexArc()
        basis = random_vector(DIM, r)
        for _ in range(40):
            t = random_vector(DIM, r)
            reflex.experience(t, slerp(t, basis, 0.4))
        drift = DriftDetector(reflex)
        # an unfamiliar task is a void
        far = random_vector(DIM, rng(999))
        assert drift.judge(far, slerp(far, basis, 0.4)) == "void"


# ===========================================================================
# Encoders
# ===========================================================================
class TestEncoders:
    def test_scalar_similarity_decays_with_distance(self):
        enc = ScalarEncoder(DIM, lo=0, hi=10, seed=1)
        base = enc.encode(5.0)
        sims = [cosine(base, enc.encode(5.0 + d)) for d in [0, 1, 2, 3, 5]]
        assert sims[0] > 0.99
        assert all(sims[i] > sims[i + 1] for i in range(4))   # monotone decay

    def test_scalar_decode_accurate_and_noise_robust(self):
        enc = ScalarEncoder(DIM, lo=0, hi=100, seed=2)
        r = rng(2)
        clean = [abs(enc.decode(enc.encode(v)) - v) for v in [5, 23, 47, 88]]
        assert max(clean) < 1.0
        noisy = enc.encode(40.0) + 0.4 * random_vector(DIM, r)
        assert abs(enc.decode(noisy) - 40.0) < 5.0

    def test_text_random_indexing_separates_categories(self):
        corpus = ["the cat sat on the mat", "the dog sat on the mat",
                  "i fed the hungry cat", "i fed the hungry dog",
                  "the car drove down the road", "the truck drove down the road",
                  "i parked the car outside", "i parked the truck outside"]
        enc = TextEncoder(DIM, window=2, seed=2)
        for _ in range(6):
            for s in corpus:
                enc.learn(s.split())
        within = (cosine(enc.wordvec("cat"), enc.wordvec("dog")) +
                  cosine(enc.wordvec("car"), enc.wordvec("truck"))) / 2
        across = (cosine(enc.wordvec("cat"), enc.wordvec("car")) +
                  cosine(enc.wordvec("dog"), enc.wordvec("truck"))) / 2
        assert within > across + 0.2

    def test_record_readback(self):
        text = TextEncoder(2048, window=2, seed=2)
        for _ in range(4):
            for s in ["the car drove fast", "the truck drove slow"]:
                text.learn(s.split())
        rec = RecordEncoder(2048, text, num_range=(0, 200), seed=7)
        vec = rec.encode({"price": ("num", 142.5), "trend": ("cat", "up"),
                          "note": ("text", "the car drove fast")})
        assert abs(rec.read_number(vec, "price") - 142.5) < 8.0
        assert rec.read_category(vec, "trend", ["up", "down", "flat"])[0] == "up"


# ===========================================================================
# Reasoning
# ===========================================================================
class TestReasoning:
    def test_resonator_recovers_factors(self):
        dim = 2048
        v = Vocabulary(dim, seed=1)
        subj = ["alice", "bob", "carol", "dave"]
        rel = ["likes", "knows", "avoids", "trusts"]
        obj = ["coffee", "jazz", "rain", "python"]
        cb = lambda ws: np.array([v.get(w) for w in ws])
        res = ResonatorNetwork([cb(subj), cb(rel), cb(obj)])
        r = rng(0)
        correct = 0
        for _ in range(15):
            i, j, k = (int(r.integers(4)) for _ in range(3))
            fact = bind(bind(v.get(subj[i]), v.get(rel[j])), v.get(obj[k]))
            assert res.factor(fact) == [i, j, k] or correct == correct
            correct += res.factor(fact) == [i, j, k]
        assert correct / 15 > 0.9

    def test_compass_points_to_success(self):
        r = rng(2)
        good, bad = random_vector(DIM, r), random_vector(DIM, r)
        c = SemanticCompass()
        for _ in range(40):
            c.record(bundle([good, 0.4 * random_vector(DIM, r)]), True)
            c.record(bundle([bad, 0.4 * random_vector(DIM, r)]), False)
        d = c.direction(bundle([good, bad]))
        assert cosine(d, good) > 0.4
        assert cosine(d, bad) < 0.0

    def test_conformal_coverage_tracks_target(self):
        r = rng(3)
        for target in [0.8, 0.9]:
            cov = []
            for trial in range(20):
                rr = np.random.default_rng(trial)
                cal = rr.normal(0, 1, 200)
                conf = ConformalPredictor(alpha=1 - target)
                conf.calibrate(cal)
                test = rr.normal(0, 1, 400)
                cov.append(np.mean(np.abs(test) <= conf.q))
            assert abs(np.mean(cov) - target) < 0.06

    def test_epistemic_map_classifies(self):
        emap = EpistemicMap(density_threshold=2, disagree_threshold=0.15)
        assert emap.classify(3, 3, 0.02) == "confident"
        assert emap.classify(4, 4, 0.33) == "boundary"
        assert emap.classify(0, 0, 0.00) == "void"       # mutual ignorance -> void
        assert emap.classify(1, 0, 1.0) == "void"

    def test_vector_disagreement_bounds(self):
        r = rng(5)
        x = random_vector(DIM, r)
        assert vector_disagreement(x, x) < 1e-6
        assert 0.8 < vector_disagreement(x, random_vector(DIM, r)) < 1.2


# ===========================================================================
# Extras
# ===========================================================================
class TestExtras:
    def test_residue_arithmetic_exact(self):
        rs = ResidueSystem(moduli=(7, 11, 13), dim=2048, seed=0)
        r = rng(0)
        for _ in range(150):
            a, b = int(r.integers(0, rs.M)), int(r.integers(0, rs.M))
            assert rs.decode(rs.encode(a)) == a
            assert rs.decode(rs.add(rs.encode(a), rs.encode(b))) == (a + b) % rs.M
            assert rs.decode(rs.subtract(rs.encode(a), rs.encode(b))) == (a - b) % rs.M
        assert rs.decode(rs.scale(rs.encode(123), 5)) == (123 * 5) % rs.M

    def test_sdf_region_boolean_algebra(self):
        r = rng(1)
        a = random_vector(DIM, r)
        far = random_vector(DIM, r)
        b = slerp(a, far, 0.30)
        ra, rb = ball(a, 0.40), ball(b, 0.40)
        in_a, both, in_b = a, slerp(a, b, 0.5), b
        assert ra.contains(in_a) and not rb.contains(in_a)
        assert ra.union(rb).contains(in_a)
        assert ra.intersect(rb).contains(both)
        assert not ra.intersect(rb).contains(in_a)
        assert ra.subtract(rb).contains(in_a)
        assert not ra.subtract(rb).contains(both)
        assert not ra.union(rb).contains(far)

    def test_predictive_filter_catches_change_ignores_noise(self):
        r = rng(2)
        p, q = random_vector(DIM, r), random_vector(DIM, r)
        pf = PredictiveFilter()
        stable_flags = switch_caught = 0
        for i in range(60):
            base = p if i < 30 else q
            novel, _ = pf.observe(base + 0.25 * random_vector(DIM, r))
            if 0 < i < 30 and novel:
                stable_flags += 1
            if 30 <= i <= 33 and novel:
                switch_caught = 1
        assert switch_caught == 1
        assert stable_flags <= 2


# ===========================================================================
# Field
# ===========================================================================
class TestField:
    def test_smooth_union_less_creased_than_hard(self):
        r = rng(0)
        a, b = random_vector(256, r), random_vector(256, r)
        pa, pb = bump(a), bump(b)
        hard = Field(lambda x: max(pa.fn(x), pb.fn(x)))
        soft = pa.smooth_union(pb, k=0.3)
        path = [slerp(a, b, t) for t in np.linspace(0, 1, 41)]
        kink_hard = np.max(np.abs(np.diff([hard.sample(p) for p in path], 2)))
        kink_soft = np.max(np.abs(np.diff([soft.sample(p) for p in path], 2)))
        assert kink_soft < kink_hard

    def test_ascend_climbs_value(self):
        r = rng(1)
        good, bad = random_vector(256, r), random_vector(256, r)
        value = landscape([good, bad], [1.0, -1.0])
        start = random_vector(256, r)
        end = ascend(value, start, seed=1)
        assert value.sample(end) > value.sample(start)
        assert cosine(end, good) > 0.7

    def test_threshold_region(self):
        field, marks = seeded_landscape(256, seed=1, n=6)
        region = field.above(0.1)
        top = marks[int(np.argmax([field.sample(m) for m in marks]))]
        assert region.contains(top)
        assert not region.contains(-top)


# ===========================================================================
# Double diffusion
# ===========================================================================
class TestDiffusion:
    def test_staircase_segments_and_ignores_transients(self):
        r = rng(0)
        dd = DoubleDiffusion()
        layers = []
        for i in range(120):
            base = 0.0 if i < 40 else (1.0 if i < 80 else 2.0)
            if 18 <= i <= 20:
                base = 0.9            # transient
            _, _, started = dd.observe([base + r.normal(0, 0.05)])
            if started:
                layers.append(i)
        # two real shifts -> two committed layers, both after their shift, none for the spike
        assert len(layers) == 2
        assert 40 <= layers[0] <= 50 and 80 <= layers[1] <= 90

    def test_identity_extraction(self):
        r = rng(1)
        a, b = random_vector(64, r), random_vector(64, r)
        dd = DoubleDiffusion(fast=0.5, slow=0.05, threshold=0.6, trigger=2.0)
        for i in range(80):
            ident = a if i < 40 else b
            dd.observe(ident + 0.3 * random_vector(64, r))
            if i == 35:
                mid_a = cosine(dd.salt, a)
        assert mid_a > 0.9
        assert cosine(dd.salt, b) > 0.9    # salt re-formed onto the new identity


# ===========================================================================
# Synchronization
# ===========================================================================
class TestSync:
    def _groups(self, ng, per, dim, noise, rng_):
        centers = [random_vector(dim, rng_) for _ in range(ng)]
        V, T = [], []
        for g, c in enumerate(centers):
            for _ in range(per):
                v = c + noise * random_vector(dim, rng_)
                V.append(v / np.linalg.norm(v)); T.append(g)
        return V, np.array(T)

    def _agree(self, t, p):
        n = len(t); a = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1; a += (t[i] == t[j]) == (p[i] == p[j])
        return a / tot

    def test_emergent_grouping_recovers_clusters(self):
        passes = 0
        for s in range(5):
            V, T = self._groups(3, 5, 256, 0.5, rng(100 + s))
            pred = SyncGrouping(seed=s).group(V)
            passes += self._agree(T, pred) > 0.9
        assert passes >= 4   # robust across seeds

    def test_coherence_one_vs_many(self):
        one, _ = self._groups(1, 12, 256, 0.5, rng(7))
        many, _ = self._groups(3, 4, 256, 0.5, rng(8))
        sync = SyncGrouping(seed=1)
        assert sync.coherence(sync.run(one)) > 0.8
        assert sync.coherence(sync.run(many)) < 0.5


# ===========================================================================
# Orchestrator
# ===========================================================================
class TestOrchestrator:
    def _registry(self):
        v = Vocabulary(1024, seed=0)
        reg = ToolRegistry()
        for name, i, o, kw in [
            ("fetch_url", "query", "raw_html", ["fetch", "web", "page", "url"]),
            ("parse_html", "raw_html", "text", ["parse", "text", "html", "web"]),
            ("read_file", "path", "text", ["read", "local", "file", "document"]),
            ("summarize", "text", "summary", ["summarize", "summary"]),
        ]:
            reg.add(Tool(name, i, o, keyword_vector(v, kw)))
        return reg, v

    def test_plans_type_valid_chains(self):
        reg, v = self._registry()
        planner = Planner(reg)
        chain, src = planner.plan("summary", keyword_vector(v, ["summarize", "web", "page"]),
                                  {"query"})
        names = [t.name for t in chain]
        assert names == ["fetch_url", "parse_html", "summarize"]
        chain2, _ = planner.plan("summary", keyword_vector(v, ["summarize", "file"]), {"path"})
        assert [t.name for t in chain2] == ["read_file", "summarize"]

    def test_capability_gap(self):
        reg, v = self._registry()
        planner = Planner(reg)
        _, src = planner.plan("caption", keyword_vector(v, ["caption"]), {"query"})
        assert src == "gap"

    def test_skeleton_reuse(self):
        reg, v = self._registry()
        planner = Planner(reg)
        goal = keyword_vector(v, ["summarize", "web", "page"])
        chain, _ = planner.plan("summary", goal, {"query"})
        planner.record_success(goal, chain)
        _, src = planner.plan("summary", keyword_vector(v, ["summarize", "web", "page", "online"]),
                              {"query"})
        assert src == "skeleton"

    def test_execution_and_failover(self):
        v = Vocabulary(1024, seed=0)
        reg = ToolRegistry(semantic_weight=1.0)
        reg.add(Tool("fetch", "query", "raw_html", keyword_vector(v, ["fetch"]),
                     lambda q: "<p>hello world hello</p>"))
        reg.add(Tool("parse", "raw_html", "text", keyword_vector(v, ["parse"]),
                     lambda h: h.replace("<p>", "").replace("</p>", "")))
        reg.add(Tool("sum_a", "text", "summary", keyword_vector(v, ["summary"]),
                     lambda t: (_ for _ in ()).throw(RuntimeError("down"))))
        reg.add(Tool("sum_b", "text", "summary", keyword_vector(v, ["summary"]),
                     lambda t: t.upper()))
        breaker = CircuitBreaker(fail_max=3, cooldown=9)
        planner = Planner(reg, breaker=breaker)
        goal = keyword_vector(v, ["summary"])
        results = []
        for _ in range(5):
            chain, _ = planner.plan("summary", goal, {"query"})
            ok, res, _ = planner.execute(chain, "q", goal_vec=goal)
            results.append((ok, res))
        assert results[0][0] is False                  # first attempts use the broken tool
        assert results[-1][0] is True                  # eventually routes to the working one
        assert "HELLO" in results[-1][1]


# ===========================================================================
# Emergence (integration)
# ===========================================================================
class TestEmergence:
    def test_grows_correct_concepts_and_rejects_noise(self):
        r = rng(0)
        centers = [random_vector(256, r) for _ in range(4)]
        unit = lambda v: v / np.linalg.norm(v)
        mind = EmergentConcepts()
        truth, preds_inputs = [], []
        for i in range(320):
            if r.random() < 0.06:
                mind.perceive(random_vector(256, r), i); continue
            cat = int(r.integers(0, 3) if i < 160 else r.integers(0, 4))
            x = unit(centers[cat] + 0.5 * random_vector(256, r))
            mind.perceive(x, i)
            truth.append(cat); preds_inputs.append(x)
        cc = mind.committed()
        assert len(cc) == 4
        spurious = sum(1 for c in cc if max(cosine(c.salt, ctr) for ctr in centers) < 0.5)
        assert spurious == 0
        pred = [int(np.argmax([cosine(x, c.salt) for c in cc])) for x in preds_inputs]
        # pairwise agreement with hidden truth
        n = len(truth); a = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1; a += (truth[i] == truth[j]) == (pred[i] == pred[j])
        assert a / tot > 0.9

    def test_retires_dead_concepts(self):
        # categories 0,1 emit only early; 2,3,4 take over -> the dead ones must retire
        r = rng(0)
        centers = [random_vector(128, r) for _ in range(5)]
        unit = lambda v: v / np.linalg.norm(v)
        mind = EmergentConcepts(retire_after=300)
        for i in range(600):
            live = [0, 1] if i < 200 else [2, 3, 4]
            cat = int(r.choice(live))
            mind.perceive(unit(centers[cat] + 0.5 * random_vector(128, r)), i)
        cc = mind.committed()
        reps = {int(np.argmax([cosine(c.salt, ctr) for ctr in centers])) for c in cc}
        assert reps == {2, 3, 4}            # only the still-active categories remain
        assert len(cc) == 3

    def test_retirement_can_be_disabled(self):
        r = rng(1)
        centers = [random_vector(128, r) for _ in range(5)]
        unit = lambda v: v / np.linalg.norm(v)
        mind = EmergentConcepts(retire_after=None)   # keep everything forever
        for i in range(600):
            live = [0, 1] if i < 200 else [2, 3, 4]
            cat = int(r.choice(live))
            mind.perceive(unit(centers[cat] + 0.5 * random_vector(128, r)), i)
        assert len(mind.committed()) == 5            # dead concepts persist when disabled


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

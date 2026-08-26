"""Deliberation: draft, judge, iterate, keep the best, with adaptive thinking time.
The loop beats a single pass; iteration count adapts to difficulty; the trace
exposes the inner drafts."""
import numpy as np

from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor, cooccurrence_space
from holographic.misc.holographic_structure import StructureVerifier
from holographic.agents_and_reasoning.holographic_deliberate import Deliberator
from holographic.agents_and_reasoning.holographic_respond import query_target


def _setup():
    a = "the president led the government and the senate passed a national law today".split()
    b = "the school taught young children to read good books in the bright classroom".split()
    c = "the team won the football game at the stadium before the loud happy crowd".split()
    sents = ([a, b, c] * 25)
    stream = [w for s in sents for w in s]
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=2, seed=0)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    v = StructureVerifier(vocab, M, idx).calibrate(stream, chunk=60, z_floor=2.0)
    return Deliberator(mp, v), mp, v, idx, M


def test_loop_beats_single_pass():
    # The deliberated best-of-N draft is at least as good as a single greedy pass.
    d, mp, v, idx, M = _setup()
    qs = ["president government law", "school children books", "football game team"]
    single, delib = [], []
    for q in qs:
        gist = query_target(q, idx, M)
        single.append(d.quality(d._realize(q, gist, temp=0.0), gist))
        delib.append(d.deliberate(q, max_iters=6, target_quality=0.9)["quality"])
    assert np.mean(delib) >= np.mean(single)


def test_iterations_are_adaptive():
    # An easy target (low bar) stops early; an unreachable bar uses the full budget.
    d, mp, v, idx, M = _setup()
    easy = d.deliberate("school children books", max_iters=8, target_quality=-10.0)
    hard = d.deliberate("school children books", max_iters=8, target_quality=10.0)
    assert easy["iterations"] == 1                  # first draft already clears a low bar
    assert hard["iterations"] == 8                  # never satisfied -> full budget


def test_trace_records_drafts():
    d, mp, v, idx, M = _setup()
    r = d.deliberate("president government law", max_iters=5, target_quality=10.0)
    assert len(r["trace"]) == 5
    assert all("draft" in t and "quality" in t for t in r["trace"])
    # the kept response is the best-scoring draft in the trace (trace rounds to 3dp)
    best_in_trace = max(t["quality"] for t in r["trace"])
    assert abs(r["quality"] - best_in_trace) < 1e-2


def test_first_draft_is_deterministic():
    # The first draft (greedy) is reproducible; the loop's diversity comes after.
    d, mp, v, idx, M = _setup()
    gist = query_target("school children books", idx, M)
    a = d._realize("school children books", gist, temp=0.0)
    b = d._realize("school children books", gist, temp=0.0)
    assert a == b


def test_brain_deliberate():
    from holographic.misc.holographic_unified import UnifiedMind
    a = "the president led the government and the senate passed a national law".split()
    b = "the school taught young children to read good books in the classroom".split()
    sents = [a, b] * 25
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    r = m.deliberate("school children books", max_iters=6, target_quality=0.45)
    assert len(r["response"]) > 3
    assert r["iterations"] >= 1
    assert "trace" in r and len(r["trace"]) >= 1


def test_negotiate_returns_per_judge_scores():
    # Multi-judge: each draft scored by coherence, relevance, novelty; the kept
    # draft's negotiated score is the minimum of its judge scores (the binding one).
    d, mp, v, idx, M = _setup()
    r = d.negotiate("president government law", max_iters=6, target_quality=10.0)
    assert set(r["scores"]) == {"coherence", "relevance", "novelty"}
    assert abs(r["negotiated"] - min(r["scores"].values())) < 1e-6
    # every trace entry carries the full judge breakdown
    assert all(set(t["scores"]) == {"coherence", "relevance", "novelty"} for t in r["trace"])


def test_negotiate_balances_rather_than_maximizes_one_axis():
    # The negotiated (min) score never exceeds any single judge -- it is bounded by
    # the weakest pressure, which is the whole point of balancing.
    d, mp, v, idx, M = _setup()
    r = d.negotiate("school children books", max_iters=6, target_quality=10.0)
    for name, val in r["scores"].items():
        assert r["negotiated"] <= val + 1e-9


def test_novelty_judge_penalises_repetition():
    # A judge panel includes novelty = type-token ratio; a repeated draft scores
    # low on it.
    d, mp, v, idx, M = _setup()
    judges = dict((n, f) for n, f in d.judges())
    repetitive = ["the", "the", "the", "the", "the", "the"]
    varied = ["the", "school", "taught", "young", "children", "books"]
    import numpy as np
    g = np.zeros(M.shape[1])
    assert judges["novelty"](varied, g) > judges["novelty"](repetitive, g)


def test_brain_negotiate():
    from holographic.misc.holographic_unified import UnifiedMind
    a = "the president led the government and the senate passed a national law".split()
    b = "the school taught young children to read good books in the classroom".split()
    sents = [a, b] * 25
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    r = m.negotiate("school children books", max_iters=6, target_quality=0.55)
    assert "scores" in r and len(r["response"]) > 3
    assert len(r["trace"]) >= 1

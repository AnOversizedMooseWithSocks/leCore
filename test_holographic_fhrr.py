"""FHRR (complex-phasor VSA) -- the high-capacity binding option. These tests pin the
MEASURED reason it exists: it holds far more key->value pairs in one trace than the
real-valued HRR core, AND the honest boundaries (no advantage at low load; the algebra
is exact). The numbers mirror the module docstring's table so the claim is reproducible.
"""
import numpy as np

from holographic_ai import random_vector, bind, unbind, cosine
from holographic_fhrr import (PhasorMemory, PhasorVocabulary, phasor_atom,
                              fhrr_bind, fhrr_unbind, fhrr_sim)

DIM = 256


def _real_capacity(n_pairs, trials=40):
    accs = []
    for t in range(trials):
        r = np.random.default_rng(1000 + t)
        keys = [random_vector(DIM, r) for _ in range(n_pairs)]
        vals = [random_vector(DIM, r) for _ in range(n_pairs)]
        trace = sum(bind(keys[i], vals[i]) for i in range(n_pairs))
        ok = sum(int(np.argmax([cosine(unbind(trace, keys[i]), v) for v in vals])) == i
                 for i in range(n_pairs))
        accs.append(ok / n_pairs)
    return float(np.mean(accs))


def _fhrr_capacity(n_pairs, trials=40):
    accs = []
    for t in range(trials):
        r = np.random.default_rng(1000 + t)
        keys = [phasor_atom(DIM, r) for _ in range(n_pairs)]
        vals = [phasor_atom(DIM, r) for _ in range(n_pairs)]
        trace = sum(fhrr_bind(keys[i], vals[i]) for i in range(n_pairs))
        ok = sum(int(np.argmax([fhrr_sim(fhrr_unbind(trace, keys[i]), v) for v in vals])) == i
                 for i in range(n_pairs))
        accs.append(ok / n_pairs)
    return float(np.mean(accs))


def test_fhrr_bind_unbind_is_exact():
    # The algebra: bind then unbind by the same key returns the value (similarity ~1).
    r = np.random.default_rng(0)
    a, b = phasor_atom(DIM, r), phasor_atom(DIM, r)
    assert fhrr_sim(fhrr_unbind(fhrr_bind(a, b), a), b) > 0.999


def test_fhrr_holds_more_pairs_than_real_hrr_at_high_load():
    # The reason FHRR exists: under capacity stress it keeps far more pairs per trace.
    for n in (30, 40, 60):
        assert _fhrr_capacity(n) > _real_capacity(n) + 0.1     # a wide, real margin
    # concrete anchor near the docstring's table (allowing seed/version drift)
    assert _fhrr_capacity(40) > 0.85
    assert _real_capacity(40) < 0.75


def test_fhrr_has_no_advantage_at_low_load():
    # Honest boundary: at low load both are perfect, so the real-valued default loses
    # nothing by staying the default.
    assert _real_capacity(8) > 0.999
    assert _fhrr_capacity(8) > 0.999


def test_phasor_memory_round_trips_a_handful_of_pairs():
    mem = PhasorMemory(DIM)
    voc = PhasorVocabulary(DIM, seed=1)
    pairs = {f"k{i}": f"v{i}" for i in range(6)}
    for k, v in pairs.items():
        mem.learn(voc.get(k), voc.get(v))
    vocab = [f"v{i}" for i in range(6)]
    for k, v in pairs.items():
        got, _ = voc.cleanup(mem.recall(voc.get(k)), candidates=vocab)
        assert got == v


def test_derived_phasor_vocab_regenerates_from_seed():
    # Same regenerate-from-seed principle as the real Vocabulary: derived atoms are a
    # pure function of (seed, name).
    a = PhasorVocabulary(DIM, seed=3, derived=True)
    b = PhasorVocabulary(DIM, seed=3, derived=True)
    for nm in ("alpha", "beta", "gamma"):
        assert np.allclose(a.get(nm), b.get(nm))


def test_unified_mind_high_capacity_faculty_is_owned_and_works():
    # The FHRR memory is reachable through the main brain (one brain), seed-deterministic,
    # and a singleton -- and it recovers a high-load trace the real-HRR core would lose.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    mem, voc = m.high_capacity_memory()
    assert m.high_capacity_memory()[0] is mem            # singleton (one faculty)
    pairs = {f"k{i}": f"v{i}" for i in range(30)}
    for k, v in pairs.items():
        mem.learn(voc.get(k), voc.get(v))
    vocab = [f"v{i}" for i in range(30)]
    ok = sum(voc.cleanup(mem.recall(voc.get(k)), candidates=vocab)[0] == v
             for k, v in pairs.items())
    assert ok >= 28                                      # ~30/30 measured; real-HRR would lose several


def test_phasor_cleanup_vectorized_matches_bruteforce():
    # the vectorized cleanup (real-matvec + lazy cache) must pick the SAME atom as a per-candidate loop, on
    # both the all-atoms (cached) path and the candidates-subset path -- the INV-5 vectorization, kept honest
    import numpy as np
    from holographic_fhrr import PhasorVocabulary, fhrr_sim, phasor_atom
    rng = np.random.default_rng(3)
    voc = PhasorVocabulary(256, seed=0)
    for i in range(120):
        voc.get(f"v{i}")
    names = list(voc.vectors)
    for _ in range(20):
        q = voc.vectors[names[int(rng.integers(120))]] + 0.4 * phasor_atom(256, rng)
        ref = max(names, key=lambda nm: fhrr_sim(q, voc.vectors[nm]))
        assert voc.cleanup(q)[0] == ref                       # all-atoms cached path
        assert voc.cleanup(q, candidates=names)[0] == ref     # subset path
    # the cache must invalidate when a new atom is minted (stale-cache guard)
    before = voc.cleanup(voc.vectors[names[0]])[0]
    voc.get("v_new")
    assert voc.cleanup(voc.vectors[names[0]])[0] in (before, "v_new") and len(voc._clean_cache[1]) == 121

"""Encyclopedia layer: one-hop facts, exact multi-hop taxonomy with decaying
throughput, structural relatedness beyond word overlap, and the curriculum
stacking -- hermetic synthetic taxonomy + a WordNet-gated scale check."""
import numpy as np

from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia, Curriculum


def _toy():
    e = Encyclopedia(dim=4096, seed=0)
    links = [("dog", "canine"), ("wolf", "canine"), ("fox", "canine"),
             ("cat", "feline"), ("lion", "feline"),
             ("canine", "carnivore"), ("feline", "carnivore"),
             ("carnivore", "mammal"), ("mammal", "animal"), ("animal", "organism"),
             ("rose", "flower"), ("tulip", "flower"), ("oak", "tree"),
             ("flower", "plant"), ("tree", "plant"), ("plant", "organism")]
    for c, p in links:
        e.add(c, is_a=p)
    return e, links


def test_one_hop_is_a_is_exact():
    # The store returns the parent it was given, reliably.
    e, links = _toy()
    for c, p in links:
        f, conf = e.is_a(c)
        assert f == p
        assert conf > 0.1


def test_multi_hop_taxonomy_is_exact_and_throughput_decays():
    # Walking the is_a chain is exact across several hops, and the relation-ray
    # throughput decays with depth (a calibrated 'how far has this deduction
    # traveled').
    e, _ = _toy()
    chain, tp = e.climb("dog")
    assert chain == ["dog", "canine", "carnivore", "mammal", "animal", "organism"]
    # transitive membership both ways
    assert e.is_a_transitive("dog", "animal")[0] is True
    assert e.is_a_transitive("dog", "plant")[0] is False
    # throughput falls monotonically as we climb further
    _, tp2 = e.climb("dog", hops=2)
    _, tp4 = e.climb("dog", hops=4)
    assert tp2 > tp4


def test_abstains_when_throughput_too_low():
    # A chain stops climbing once confidence falls below the floor, rather than
    # emitting an unreliable deeper answer. With exact (unitary-atom) unbinding each
    # hop is near-lossless, so depth-decay is now an explicit per-hop discount; a high
    # enough floor still truncates a deep chain.
    e, _ = _toy()
    full, _ = e.climb("dog", min_throughput=0.0)
    short, _ = e.climb("dog", min_throughput=0.75)
    assert len(short) < len(full)


def test_relatedness_is_structural_not_lexical():
    # The 'understanding beyond words' property: siblings are related through the
    # shared parent even when nothing about the symbols says so; unrelated
    # branches score zero.
    e, _ = _toy()
    # closer kin scores higher; distant cousins (sharing only a far ancestor like
    # 'organism') score low but nonzero -- relatedness is graph distance, and dog
    # and rose genuinely meet at 'organism'.
    assert e.relatedness("dog", "wolf") > e.relatedness("dog", "cat") > e.relatedness("dog", "rose")
    assert e.relatedness("dog", "rose") > 0.0                                    # they meet at 'organism'
    assert e.relatedness("dog", "wolf") == 1.0 / 3.0                            # siblings: parent at depth 1 in each chain
    assert set(e.siblings("dog")) == {"wolf", "fox"}


def test_curriculum_reports_layer_capabilities():
    # The Curriculum measures a capability the encyclopedia has that word meaning
    # alone does not: exact one-hop facts and structural relatedness.
    e, links = _toy()
    cur = Curriculum(encyclopedia=e)
    caps = cur.capabilities(taxonomy_probes=links,
                            sibling_pairs=[("dog", "wolf"), ("rose", "tulip")])
    assert caps["encyclopedia_onehop"] == 1.0
    assert caps["encyclopedia_relatedness"] > 0.0


def test_wordnet_scale_if_available():
    # Closed-world scale check on a REAL encyclopedia (WordNet is_a), keyed by
    # synset so senses don't collide: one-hop and multi-hop are exact, and the
    # naive-looking failure of the first attempt is avoided by consistent senses.
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except Exception:
        import pytest
        pytest.skip("WordNet not available")
    seeds = []
    for ss in wn.all_synsets('n'):
        nm = ss.lemmas()[0].name()
        if "_" not in nm and nm.isalpha() and len(nm) > 3:
            seeds.append(ss)
        if len(seeds) >= 200:
            break
    links = {}
    for ss in seeds:
        chain = [ss]
        cur = ss
        while cur.hypernyms():
            cur = cur.hypernyms()[0]
            chain.append(cur)
        for i in range(len(chain) - 1):
            links[chain[i].name()] = chain[i + 1].name()
    e = Encyclopedia(dim=8192, seed=0)
    for c, p in links.items():
        e.add(c, is_a=p)
    # one-hop exact
    ok = sum(1 for c, p in links.items() if e.is_a(c)[0] == p)
    assert ok == len(links)
    # multi-hop exact on concepts at least 3 deep
    def stored_chain(c, hops):
        out = [c]
        for _ in range(hops):
            if c not in links:
                break
            c = links[c]
            out.append(c)
        return out
    checked = 0
    for c in list(links)[:150]:
        truth = stored_chain(c, 3)
        if len(truth) < 4:
            continue
        got, _ = e.climb(c, hops=3)
        assert got == truth
        checked += 1
    assert checked > 20


def test_scalar_encoder_is_its_kernel_and_rbf_reads_density_better():
    # kernel_at lets you ASSERT the kernel rather than eyeball it: the measured cosine
    # between two encodings dx apart matches the analytic kernel. And RBF (non-negative,
    # tunable bandwidth) recovers a bimodal density that sinc -- one lobe over the range --
    # cannot resolve.
    import numpy as _np
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
    from holographic.agents_and_reasoning.holographic_ai import cosine, bundle

    enc = ScalarEncoder(2048, 0.0, 10.0, seed=1, kernel="rbf")
    for dx in (0.0, 0.5, 1.0, 2.0):
        assert abs(cosine(enc.encode(3.0), enc.encode(3.0 + dx)) - enc.kernel_at(dx)) < 0.03
    assert all(enc.kernel_at(dx) >= 0 for dx in _np.linspace(0, 20, 100))   # RBF never negative
    sinc = ScalarEncoder(2048, 0.0, 10.0, seed=1, kernel="sinc")
    assert min(sinc.kernel_at(dx) for dx in _np.linspace(0, 20, 100)) < -0.05  # sinc does dip

    # bimodal density recovery within [0,1]
    rng = _np.random.default_rng(0)
    samples = _np.concatenate([rng.normal(0.30, 0.05, 500), rng.normal(0.70, 0.05, 500)])
    grid = _np.linspace(0, 1, 150)
    from math import sqrt, pi
    true = _np.array([0.5 * _np.exp(-(g - 0.3) ** 2 / (2 * 0.05 ** 2)) / (0.05 * sqrt(2 * pi))
                      + 0.5 * _np.exp(-(g - 0.7) ** 2 / (2 * 0.05 ** 2)) / (0.05 * sqrt(2 * pi))
                      for g in grid])

    def corr(kernel, bw=8.0):
        e = ScalarEncoder(4096, 0.0, 1.0, seed=2, kernel=kernel, bandwidth=bw)
        b = bundle([e.encode(s) for s in samples])
        r = _np.array([cosine(b, e.encode(g)) for g in grid])
        return float(_np.corrcoef(r, true)[0, 1])

    assert corr("rbf") > corr("sinc") + 0.1        # RBF tracks the bimodal density far better

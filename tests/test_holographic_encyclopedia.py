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


# ======================================================================================================
# The merge bug, and the faculty. `add` used to CLOBBER a concept's record instead of merging.
# ======================================================================================================
def _taxonomy(obj, add):
    for concept, parent in (("dog.n.01", "canine.n.01"), ("wolf.n.01", "canine.n.01"),
                            ("canine.n.01", "carnivore.n.01"), ("carnivore.n.01", "mammal.n.01"),
                            ("cat.n.01", "feline.n.01"), ("feline.n.01", "carnivore.n.01")):
        add(concept, parent)
    return obj


def test_adding_parts_after_a_parent_does_not_clobber_the_is_a_role():
    """THE BUG: KnowledgeStore.add REPLACES a record (`self.attrs[name] = dict(attrs)`), which is the right contract
    for a primitive store. Encyclopedia called it as if it merged, so add(is_a=) then add(has=) silently dropped the
    is_a role from the bound vector -- while the Python-side `parent` dict still held it. is_a() reads the VECTOR,
    so it returned "tail", and climb() walked into ["dog.n.01", "tail"]. A taxonomy that breaks with nothing raised."""
    from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia
    enc = Encyclopedia(dim=2048, seed=0)
    enc.add("dog.n.01", is_a="canine.n.01")
    assert enc.is_a("dog.n.01")[0] == "canine.n.01"

    enc.add("dog.n.01", has=["tail", "fur"])                 # the second add used to clobber the first
    assert enc.is_a("dog.n.01")[0] == "canine.n.01"          # is_a survives...
    assert enc._read("dog.n.01", "has")[0] == "tail"         # ...and the new role is there
    assert enc.climb("dog.n.01")[0] == ["dog.n.01", "canine.n.01"]

    # and the single-call path is untouched
    e2 = Encyclopedia(dim=2048, seed=0)
    e2.add("x", is_a="y")
    assert e2.is_a("x")[0] == "y"


def test_relatedness_is_one_over_one_plus_the_summed_depths_not_one_for_siblings():
    """KEPT NEGATIVE against the module's own former docstring, which claimed 1.0 for siblings. Only a concept
    against ITSELF scores 1.0. The number ORDERS taxonomic distance; it is not a probability."""
    from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia
    enc = Encyclopedia(dim=2048, seed=0)
    _taxonomy(enc, lambda c, p: enc.add(c, is_a=p))
    r = enc.relatedness
    assert abs(r("dog.n.01", "dog.n.01") - 1.0) < 1e-12         # identical
    assert abs(r("dog.n.01", "canine.n.01") - 0.5) < 1e-12      # parent
    assert abs(r("dog.n.01", "wolf.n.01") - 1 / 3) < 1e-12      # SIBLINGS -- not 1.0
    assert abs(r("dog.n.01", "cat.n.01") - 0.2) < 1e-12         # cousins
    assert r("dog.n.01", "rock.n.01") == 0.0                    # unrelated
    assert r("dog.n.01", "wolf.n.01") > r("dog.n.01", "cat.n.01") > r("dog.n.01", "rock.n.01")


def test_climb_throughput_decays_with_depth_and_abstains():
    from holographic.agents_and_reasoning.holographic_encyclopedia import Encyclopedia
    enc = Encyclopedia(dim=2048, seed=0)
    _taxonomy(enc, lambda c, p: enc.add(c, is_a=p))
    chain, tp_deep = enc.climb("dog.n.01")
    assert chain == ["dog.n.01", "canine.n.01", "carnivore.n.01", "mammal.n.01"]
    _, tp_shallow = enc.climb("carnivore.n.01")
    assert tp_deep < tp_shallow                                  # a longer deduction is deliberately less certain
    assert enc.climb("dog.n.01", hops=1)[0] == ["dog.n.01", "canine.n.01"]
    short, _ = enc.climb("dog.n.01", min_throughput=0.95)        # abstain rather than emit low-confidence noise
    assert len(short) < len(chain)


def test_encyclopedia_faculty_through_the_mind_and_over_http():
    """The state lives on the mind, and every method takes and returns PLAIN DATA -- so a long-lived service
    accumulates knowledge across /invoke calls and needs no stateless twin."""
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    import holographic_service as svc_mod
    from holographic.misc.holographic_unified import UnifiedMind

    mind = UnifiedMind(dim=2048, seed=0)
    svc = svc_mod.Service(mind=mind)
    httpd = HTTPServer(("127.0.0.1", 0), svc_mod.make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]

    def invoke(name, args):
        body = json.dumps({"name": name, "args": args}).encode()
        req = urllib.request.Request(base + "/invoke", data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    try:
        # teach across SEPARATE calls -- the state persists on the mind between them
        for c, p in (("dog.n.01", "canine.n.01"), ("wolf.n.01", "canine.n.01"),
                     ("canine.n.01", "carnivore.n.01"), ("carnivore.n.01", "mammal.n.01")):
            assert invoke("encyclopedia_add", {"concept": c, "is_a": p})["ok"]
        assert invoke("encyclopedia_add", {"concept": "dog.n.01", "has": ["tail"]})["ok"]   # merge, not clobber

        r = invoke("encyclopedia_is_a", {"concept": "dog.n.01"})
        assert r["ok"] and r["result"]["parent"] == "canine.n.01"

        r = invoke("encyclopedia_is_a_transitive", {"concept": "dog.n.01", "ancestor": "mammal.n.01"})
        assert r["ok"] and r["result"]["reached"] and r["result"]["hops"] == 3

        r = invoke("encyclopedia_relatedness", {"a": "dog.n.01", "b": "wolf.n.01"})
        assert r["ok"] and abs(r["result"] - 1 / 3) < 1e-9      # siblings, plain JSON float
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert mind.encyclopedia_reset() == 4                        # concepts cleared
    assert mind.encyclopedia_siblings("dog.n.01") == []

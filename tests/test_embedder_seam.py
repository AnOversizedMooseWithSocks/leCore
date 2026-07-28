"""Regression traps for the bring-your-own-embedder seam (work plan item 1.3).

The seam itself is three lines of plumbing. What is worth testing is the SPACE-AGREEMENT PROBE, because
the failure it prevents is silent: a query embedded by a different model than the index still produces
five ranked names with confident-looking scores. Dimension is checkable; space is not. So the probe is
pinned in BOTH directions -- it must reject a wrong-space embedder AND accept an in-space one. A probe
that rejects everything would pass a one-sided test and be useless.
"""
import numpy as np
import pytest

import lecore
from holographic.semantic_router.holographic_embedseam import module_summaries, probe_embedder_space


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture(scope="module")
def oracle(mind):
    """An embedder that is in the index's space BY CONSTRUCTION: it returns the module's own dequantized
    index row. This is the positive control -- without it, 'the probe rejects a random embedder' would be
    satisfied by a probe that rejects everything."""
    router = mind._embedding_router()
    if router is None:
        pytest.skip("no routing index shipped")
    names = [str(n) for n in router.names]
    summaries = module_summaries(".", names=set(names))
    data = np.load("lecore_data/routing/index_128d.npz", allow_pickle=True)
    q = data["q"].astype(np.float64)
    lo, hi = data["lo"].astype(np.float64), data["hi"].astype(np.float64)
    rows = lo + (hi - lo) * (q / 255.0)
    by_text = {summaries[n]: i for i, n in enumerate(names) if n in summaries}

    def embed(text):
        i = by_text.get(" ".join(str(text).split()))
        return rows[i] if i is not None else np.zeros(rows.shape[1])

    return embed


def test_module_summaries_returns_docstrings_not_source():
    s = module_summaries(".")
    assert len(s) > 100
    assert all(nm.startswith("holographic_") for nm in s)          # the index's spelling, or nothing matches
    assert not any("import numpy" in v for v in s.values())


def test_probe_rejects_a_wrong_space_embedder(mind):
    # THE FAILURE THE SEAM EXISTS TO CATCH. Right dimension, random geometry -- passes any shape check and
    # produces meaningless cosines. Must land at chance.
    router = mind._embedding_router()
    if router is None:
        pytest.skip("no routing index shipped")
    rng = np.random.default_rng(0)
    report = probe_embedder_space(lambda t: rng.standard_normal(128), router, ".", sample=12)
    assert report["rate"] < 0.30
    assert report["chance"] < 0.05                                  # 5/509; the bar is ~30x this


def test_probe_accepts_an_in_space_embedder(mind, oracle):
    # THE OTHER DIRECTION. A probe that rejects everything is not an instrument.
    router = mind._embedding_router()
    report = probe_embedder_space(oracle, router, ".", sample=12)
    assert report["rate"] >= 0.9, "an in-space embedder only self-recalled at %.2f" % report["rate"]


def test_set_embedder_refuses_a_wrong_space_model(mind):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError) as exc:
        mind.set_embedder(lambda t: rng.standard_normal(128))
    assert "space-agreement" in str(exc.value)
    mind.set_embedder(None)


def test_set_embedder_accepts_the_oracle_and_unlocks_free_text(mind, oracle):
    # THE ITEM'S GATE: free text goes from an honest None to a ranked list.
    mind.set_embedder(None)
    assert mind.route_semantic("smooth a bumpy mesh") is None
    report = mind.set_embedder(oracle)
    assert report["ok"] is True
    ranked = mind.route_semantic("smooth a bumpy mesh")
    assert isinstance(ranked, list) and ranked
    mind.set_embedder(None)
    assert mind.route_semantic("smooth a bumpy mesh") is None       # clearing restores the old behaviour


def test_set_embedder_guards(mind):
    with pytest.raises(TypeError):
        mind.set_embedder(42)
    assert mind.set_embedder(None) is None


def test_a_broken_embedder_is_a_miss_not_a_raise(mind):
    # route_semantic's contract is an honest None, never an exception into the caller's request path.
    def boom(_):
        raise RuntimeError("model not loaded")

    mind.set_embedder(boom, verify=False)
    assert mind.route_semantic("smooth a bumpy mesh") is None
    mind.set_embedder(None)


def test_probe_is_deterministic():
    class _Stub:
        names = ["holographic_ai", "holographic_render", "holographic_mesh"]

        def route(self, vec, k=5, gamma=0.0):
            return [(self.names[0], 1.0)]

    a = probe_embedder_space(lambda t: np.zeros(8), _Stub(), ".", sample=3, seed=3)
    b = probe_embedder_space(lambda t: np.zeros(8), _Stub(), ".", sample=3, seed=3)
    assert a["misses"] == b["misses"] and a["rate"] == b["rate"]


def test_the_seam_is_discoverable(mind):
    for query in ("supply my own embedding model", "plug in an external embedder",
                  "bring your own vector encoder"):
        assert "query embedder" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the embedder seam" % query

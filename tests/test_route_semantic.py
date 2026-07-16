"""Regression trap for a measured production break: route_semantic called self._embedding_router(), which
had been LOST in a branch reconciliation -- so the semantic route raised AttributeError on EVERY call. The
audits did not catch it (they check wiring and catalog reachability, not execution), and no test exercised
the no-vector path. This test pins the contract: route_semantic NEVER raises -- it returns a ranking when an
index artifact ships, and an honest None when one does not (the caller then falls back to find_capability).
"""
import lecore


def test_route_semantic_never_raises():
    m = lecore.UnifiedMind(dim=64, seed=0)
    # no query vector, no cache, likely no shipped artifact in a source checkout: the contract is an honest
    # None (or a ranking if lecore_data/routing/index_*d.npz happens to be present) -- NEVER an exception.
    out = m.route_semantic("smooth a bumpy mesh surface")
    assert out is None or (isinstance(out, list) and all(isinstance(x, tuple) for x in out))
    # the gamma (workflow-bone fusion) path must obey the same contract
    out2 = m.route_semantic("smooth a bumpy mesh surface", gamma=0.5)
    assert out2 is None or isinstance(out2, list)


def test_route_semantic_with_shipped_artifact():
    """The coverage hole the first version of this test had: with NO artifact, route_semantic returned None
    BEFORE reaching the query-embed step -- so a second lost helper (_query_embedder) went undetected until
    the full loop ran against a real index. This test exercises the artifact-present path whenever the
    committed index ships (CI always has it; a sparse local checkout may not -- then it degrades to the
    no-artifact contract, which the first test already pins)."""
    import os
    import numpy as np
    m = lecore.UnifiedMind(dim=64, seed=0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    idx = os.path.join(root, "lecore_data", "routing", "index_128d.npz")
    # free text must NEVER raise, artifact or not (embedder may be absent -> honest None)
    assert m.route_semantic("smooth a mesh") is None or isinstance(m.route_semantic("smooth a mesh"), list)
    if os.path.isfile(idx):
        # vector path: a document's own (dequantized) vector must route to that module, fusion default on
        z = np.load(idx)
        names = [str(n) for n in z["names"]]
        i = names.index("holographic_meshsmooth")
        v = (z["q"][i].astype(np.float64) / 255.0
             * (z["hi"][i].astype(np.float64) - z["lo"][i].astype(np.float64)) + z["lo"][i].astype(np.float64))
        hits = m.route_semantic("", query_vec=v, k=3)
        assert hits and hits[0][0] == "holographic_meshsmooth", hits


def test_embedding_router_helper_exists_and_caches():
    m = lecore.UnifiedMind(dim=64, seed=0)
    assert hasattr(m, "_embedding_router"), "the helper route_semantic depends on must exist"
    assert hasattr(m, "_query_embedder"), "the SECOND lost helper must exist too (found via the full loop)"
    r1 = m._embedding_router()
    r2 = m._embedding_router()                                # second call must hit the cache, not re-scan
    assert r1 is r2 or (r1 is None and r2 is None)

"""The N31 chain (free-text -> vector -> router, NO model) was fully wired and completely dead. Three separate
small breaks, each invisible alone:

  1. `export_query_embed` sat BELOW `if __name__ == '__main__'` in distill_map.py and was never called -- a
     passing fit printed its verdict and wrote nothing.
  2. The artifact was therefore never produced (that part is real work: the fit needs the nomic token table,
     so it runs where the weights live -- see SEMANTIC_BACKLOG.md S2).
  3. The loader and the exporter disagreed on the FILENAME three ways (`queryembed_64d.npz` / `query_map_64d.npz`
     wanted vs `query_embed.npz` recommended) -- so even a fitted artifact would have landed in the right
     directory under a name nothing checks, and route_semantic would keep returning None with the cure on disk.

These tests pin the chain so no piece can silently detach again: the exporter's artifact loads in the runtime
embedder, embeds deterministically at the artifact's dim, routes through a real EmbeddingRouter, and the
exporter's recommended filename IS one the loader checks (first, in fact).
"""
import io
import os
import pathlib
import re
import sys
import tempfile

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools" / "semantic"))

from holographic.semantic_router.holographic_queryembed import QueryEmbedder


def _tiny_artifact(path, dim=16):
    """Build a small but REAL artifact through the exporter itself (not a hand-rolled npz), so the test
    exercises the exact write path distill_map --export uses."""
    from distill_map import export_query_embed

    rng = np.random.default_rng(0)
    vocab = {"[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "render": 3, "scene": 4, "noise": 5, "image": 6, "mesh": 7}
    T = rng.standard_normal((len(vocab), dim))
    freq = np.ones(len(vocab)); freq[3:] = (5, 4, 3, 3, 2)
    W = rng.standard_normal((dim, dim)) * 0.1 + np.eye(dim)      # near-identity: embeddings stay sane
    export_query_embed(str(path), T, vocab, freq, W, pc=None)


def test_exported_artifact_loads_and_embeds_deterministically(tmp_path):
    p = tmp_path / "query_embed_128d.npz"
    _tiny_artifact(p)
    qe = QueryEmbedder(str(p))
    v1, v2 = qe.embed("render the scene"), qe.embed("render the scene")
    assert v1 is not None and v1.shape == (16,)
    assert np.array_equal(v1, v2), "same text must embed identically (no RNG anywhere in the chain)"
    assert qe.embed("zzzz qqqq") is None, "all-unknown tokens must return None, not a fabricated vector"


def test_embedded_query_routes_through_a_real_router(tmp_path):
    """End to end at matching dim: exporter -> embedder -> EmbeddingRouter.route. This is the loop that was
    dead; if any interface drifts (shapes, dtypes, the correction contract), this is where it shows."""
    from holographic.semantic_router.holographic_router import EmbeddingRouter

    p = tmp_path / "query_embed_128d.npz"
    _tiny_artifact(p, dim=16)
    qe = QueryEmbedder(str(p))
    v = qe.embed("render the noise image")
    assert v is not None

    rng = np.random.default_rng(1)
    V = rng.standard_normal((6, 16)).astype(np.float32)
    mu = V.mean(0); Vc = V - mu
    pc = np.linalg.svd(Vc, full_matrices=False)[2][:1]
    Vr = Vc - (Vc @ pc.T) @ pc
    lo = Vr.min(1, keepdims=True); hi = Vr.max(1, keepdims=True)
    q = np.round((Vr - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
    ip = tmp_path / "index.npz"
    np.savez(ip, names=np.array([f"holographic_m{i}" for i in range(6)]), q=q,
             lo=lo.astype(np.float16), hi=hi.astype(np.float16),
             mu=mu.astype(np.float16), pc=pc.astype(np.float16),
             bone_src=np.array([], dtype=np.int32), bone_dst=np.array([], dtype=np.int32),
             bone_w=np.array([], dtype=np.float32))
    ranked = EmbeddingRouter(str(ip)).route(v, k=3)
    assert len(ranked) == 3 and all(isinstance(s, float) for _n, s in ranked)


def test_loader_and_exporter_agree_on_the_filename():
    """The three-way name mismatch, pinned. The exporter's recommended name must be in the loader's search
    list -- FIRST, so a fresh artifact wins over any legacy one."""
    unified = (_ROOT / "holographic" / "misc" / "holographic_unified.py").read_text(encoding="utf-8")
    m = re.search(r'for name in \(([^)]*)\):\s*\n\s*path = os\.path\.join\(root, "lecore_data", "routing"', unified)
    assert m, "could not find _query_embedder's search list"
    names = re.findall(r'"([^"]+\.npz)"', m.group(1))
    dm = (_ROOT / "tools" / "semantic" / "distill_map.py").read_text(encoding="utf-8")
    assert names and names[0] in dm, (
        "the loader's FIRST search name must be the one distill_map's export message recommends", names[:2])


def test_route_semantic_honest_none_without_an_artifact():
    """The contract that must survive all of this: no artifact, no vector, no fabrication -> None, and the
    token find_capability path untouched. (If an artifact ships to lecore_data/routing later, this test still
    holds for minds whose repo lacks one -- it asserts the FALLBACK, not the absence.)"""
    from holographic.misc.holographic_unified import UnifiedMind

    m = UnifiedMind(dim=64, seed=0)
    shipped = _ROOT / "lecore_data" / "routing"
    if any((shipped / n).is_file() for n in ("query_embed_128d.npz", "query_embed.npz",
                                             "queryembed_64d.npz", "query_map_64d.npz")):
        pytest.skip("a real artifact ships in this tree; the None-fallback case no longer applies")
    assert m.route_semantic("a phrase nobody cached, with no vector supplied") is None
    assert m.find_capability("damage a hypervector"), "the token path must be unaffected"


def test_the_export_gate_refuses_a_losing_map():
    """S2's MEASURED NEGATIVE, pinned as executable code so it cannot be re-proposed from memory.

    At 128d the ridge map scored top-1 1/12, median 19 -- WORSE than the [floor] SIF token-pool (3/12, median 13)
    that uses no learned map at all. The tool's --export used to write the artifact REGARDLESS of that, and the
    only thing that stopped an 8 MB routing-degrading file landing in lecore_data/ was a NameError (the function
    sat below the __main__ guard). A gate that lives in a docstring is not a gate.

    Reads the constants from the tool, so if someone loosens the bar this test tells them what they are doing.
    """
    import importlib.util
    import pathlib

    p = pathlib.Path(__file__).resolve().parent.parent / "tools" / "semantic" / "distill_map.py"
    spec = importlib.util.spec_from_file_location("_dm", p)
    dm = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dm)
    except Exception:                       # the fit needs encoder weights CI does not have; the CONSTANTS do not
        import re                           # -- so fall back to reading them out of the source text.
        src = p.read_text(encoding="utf-8")
        bar_t1 = int(re.search(r"EXPORT_BAR_TOP1\s*=\s*(\d+)", src).group(1))
        bar_med = int(re.search(r"EXPORT_BAR_MEDIAN\s*=\s*(\d+)", src).group(1))
    else:
        bar_t1, bar_med = dm.EXPORT_BAR_TOP1, dm.EXPORT_BAR_MEDIAN

    refused = lambda t1, med: t1 < bar_t1 or med > bar_med
    assert refused(1, 19), "the MEASURED [ours] result must be refused -- it is worse than using no map at all"
    assert refused(3, 13), "the [floor] must not clear either; the bar exists to beat the floor, not tie it"
    assert not refused(6, 2), "a genuine win must still be able to ship, or the gate is just a wall"
    assert bar_t1 <= 5, ("the bar must sit BELOW the measured 128d ceiling (5/12) -- a bar above the ceiling is "
                         "unfalsifiable, which is exactly the error the first bar (>=6/12) made")


def test_no_query_embed_artifact_was_committed():
    """The negative's other half: since the gate failed, NOTHING should have shipped. An 8 MB artifact appearing
    in lecore_data/routing/ means someone exported past a failed gate."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for stray in (root / "lecore_data" / "routing").glob("query_embed_*.npz"):
        raise AssertionError(
            "%s exists, but the S2 gate FAILED at 128d ([ours] 1/12 vs floor 3/12). If a later run genuinely "
            "cleared the bar, update docs/SEMANTIC_BACKLOG.md S2 with the numbers and delete this test -- do not "
            "just let the file sit here." % stray)
